// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <map>
#include "ctranMapper.h"
#include "ctranMapperImpl.h"
#include "comm.h"
#include "nccl_cvars.h"

static std::vector<double> registerDurs;
static std::vector<double> deregisterDurs;
static std::vector<double> lookupHitDurs;
static std::vector<double> lookupMissDurs;
static std::unordered_map<uint64_t, ctranMapper*> allCommHashCtranMapperMap;

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_CTRAN_PROFILING
   type        : enum
   default     : none
   choices     : none, stdout, kineto
   description : |-
     Kind of ctran profiling needed.
     none - No profiling
     stdout - Dump profiling data to stdout
     kineto - Dump profiling data to a kineto log
        (for kineto profiling, see also NCCL_CTRAN_KINETO_PROFILE_DIR)

 - name        : NCCL_CTRAN_KINETO_PROFILE_DIR
   type        : string
   default     : "/tmp"
   description : |-
     Directory to place Ctran kineto profiling logs.
     (see also NCCL_CTRAN_PROFILING)

 - name        : NCCL_CTRAN_REGISTER
   type        : enum
   default     : lazy
   choices     : none, lazy, eager
   description : |-
     Kind of registration to use for ctran user buffers
     none - No registration
     lazy - Lazy registration (keep track of user-provided registration
            buffers, but delay the actual registration till the buffer
            is used for a communication operation)
     eager - Eager registration (register buffers as soon as it is
             provided by the user)

 - name        : NCCL_CTRAN_BACKENDS
   type        : enumlist
   default     : ib
   choices     : ib, nvl
   description : |-
     Backends to enable for ctran
     ib - RoCE/IB backend
     nvl - NVLink backend

 - name        : NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT
   type        : int
   default     : 0
   description : |-
     In additional to registration snapshot reported at communicator destroy
     time, this variable allows a snapshot to be reported whenever once every
     N registrations.  Set to 0 to disable.  It helps understand the performance
     impact of registeration at different period of a long running job.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

ctranMapper::ctranMapper(ncclComm *comm) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  /* mapperRegElemList */
  this->pimpl->mapperRegElemList = new class ctranAvlTree();

  /* check user preference for backends */
  for (auto b : NCCL_CTRAN_BACKENDS) {
    if (b == NCCL_CTRAN_BACKENDS::ib) {
      this->pimpl->backends.push_back(ctranMapperBackend::IB);
    } else if (b == NCCL_CTRAN_BACKENDS::nvl) {
      this->pimpl->backends.push_back(ctranMapperBackend::NVL);
    }
  }

  /* enable backends that are possible */
  std::vector<enum ctranMapperBackend>::iterator it;

  this->pimpl->ctranIb = nullptr;
  it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
      ctranMapperBackend::IB);
  if (it != this->pimpl->backends.end()) {
    try {
      this->pimpl->ctranIb = std::unique_ptr<class ctranIb>(new class ctranIb(comm));
    } catch (const std::bad_alloc& e) {
      WARN("CTRAN: IB backend not enabled");
    }
  }

  this->pimpl->ctranNvl = nullptr;
  it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
      ctranMapperBackend::NVL);
  if (it != this->pimpl->backends.end()) {
    try {
      this->pimpl->ctranNvl = std::unique_ptr<class ctranNvl>(new class ctranNvl(comm));
    } catch (const std::bad_alloc& e) {
      WARN("CTRAN: Nvl backend not enabled");
    }
  }

  for (int i = 0; i < comm->nRanks; i++) {
    /* FIXME: we currently only support NVL for self communication */
    if (i == comm->rank && this->pimpl->ctranNvl != nullptr) {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::NVL);
    } else if (this->pimpl->ctranIb != nullptr) {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::IB);
    } else {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::UNSET);
    }
  }

  this->pimpl->numRegistrations = 0;
  this->pimpl->numCachedRegistrations = 0;
  this->pimpl->totalNumDynamicRegistrations = 0;
  this->pimpl->totalNumRegistrations = 0;
  this->pimpl->totalNumCachedRegistrations = 0;
  this->pimpl->totalNumRegLookupHit = 0;
  this->pimpl->totalNumRegLookupMiss = 0;

  CUDACHECKIGNORE(cudaStreamCreateWithFlags(&this->s, cudaStreamNonBlocking));

  this->rank = comm->rank;
  this->commHash = comm->commHash;

  /* Memory pool */
  this->pimpl->memPool = new class ctranMapperMemPool();
  this->pimpl->memPool->regMem(
      [&](const void* buf, std::size_t len, void** hdl) -> ncclResult_t {
          return this->regMem(buf, len, hdl);
      });
  allCommHashCtranMapperMap[this->commHash] = this;
}

static double sumDurations(std::vector<double>& durs) {
  double total = 0;
  for (auto& dur : durs) {
    total += dur;
  }
  return total;
}

void reportGlobalRegSnapshot(void) {
  // Counts per communicator
  for (auto& it : allCommHashCtranMapperMap) {
    auto& mapper = it.second;
    mapper->reportRegSnapshot();
  }

  // Timers accumulated from all communicators
  if (registerDurs.size()) {
    double totalRegisLat = sumDurations(registerDurs);
    INFO(
        NCCL_INIT,
        "CTRAN-MAPPER: [register snapshot] total registration latency across all comms %.2f ms, average %.2f ms across %lu registrations",
        totalRegisLat,
        totalRegisLat / registerDurs.size(),
        registerDurs.size());
  }
  if (deregisterDurs.size()) {
    double totalDeregistLat = sumDurations(deregisterDurs);
    INFO(
        NCCL_INIT,
        "CTRAN-MAPPER: [register snapshot] total deregistration latency across all comms %.2f ms, average %.2f ms across %lu registrations",
        totalDeregistLat,
        totalDeregistLat / deregisterDurs.size(),
        deregisterDurs.size());
  }
  if (lookupHitDurs.size()) {
    double totalLookupHitLat = sumDurations(lookupHitDurs);
    INFO(
        NCCL_INIT,
        "CTRAN-MAPPER: [register snapshot] total hit lookup latency across all comms %.2f ms, average %.2f ms across %lu hits",
        totalLookupHitLat,
        totalLookupHitLat / lookupHitDurs.size(),
        lookupHitDurs.size());
  }
  if (lookupMissDurs.size()) {
    double totalLookupMissLat = sumDurations(lookupMissDurs);
    INFO(
        NCCL_INIT,
        "CTRAN-MAPPER: [register snapshot] total missed lookup latency across all comms %.2f ms, average %.2f ms across %lu misses",
        totalLookupMissLat,
        totalLookupMissLat / lookupMissDurs.size(),
        lookupMissDurs.size());
  }
}

void ctranMapper::reportRegSnapshot(void) {
  INFO(
      NCCL_INIT,
      "CTRAN-MAPPER: [register snapshot] buffer registration with commHash %lu: "
      "total cached %u total registered %u total dynamically registered %u, total lookup hits %u misses %u",
      this->commHash,
      this->pimpl->totalNumCachedRegistrations,
      this->pimpl->totalNumRegistrations,
      this->pimpl->totalNumDynamicRegistrations,
      this->pimpl->totalNumRegLookupHit,
      this->pimpl->totalNumRegLookupMiss);
}

ctranMapper::~ctranMapper() {
  /* flush timestamps */
  if (!this->timestamps.empty()) {
    if (NCCL_CTRAN_PROFILING == NCCL_CTRAN_PROFILING::stdout) {
      std::cout << "[CTRAN-MAPPER] Communication Profiling:" << std::endl;
      for (auto& ts : this->timestamps) {
        std::cout << "    collective=" << ts.algo << std::endl;
        std::cout << "    startTime="
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                         ts.start.time_since_epoch())
                         .count()
                  << std::endl;
        for (auto& tsp : ts.recvCtrl) {
          std::cout << "        recvCtrl[" << tsp.peer << "]="
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           tsp.now.time_since_epoch())
                           .count()
                    << std::endl;
        }
        for (auto& tsp : ts.putIssued) {
          std::cout << "        putIssued[" << tsp.peer << "]="
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           tsp.now.time_since_epoch())
                           .count()
                    << std::endl;
        }
        for (auto& tsp : ts.putComplete) {
          std::cout << "        putComplete[" << tsp.peer << "]="
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(
                           tsp.now.time_since_epoch())
                           .count()
                    << std::endl;
        }
      }
      std::cout << std::flush;
    } else if (NCCL_CTRAN_PROFILING == NCCL_CTRAN_PROFILING::kineto) {
      auto pid = getpid();
      std::string filename(NCCL_CTRAN_KINETO_PROFILE_DIR +
          std::string("/nccl_ctran_log.") + std::to_string(pid) +
          std::string(".json"));
      INFO(NCCL_ALL, "Dumping ctran profile to %s\n", filename.c_str());
      std::ofstream f(filename);
      int id = 0;
      f << "[" << std::endl;
      for (auto& ts : this->timestamps) {
        int collId = id;
        f << "{\"name\": \"" << ts.algo << "\", "
          << "\"cat\": \"COL\", "
          << "\"id\": \"" << id++ << "\", "
          << "\"ph\": \"b\", "
          << "\"pid\": \"0\", "
          << "\"ts\": \""
          << std::chrono::duration_cast<std::chrono::milliseconds>(
              ts.start.time_since_epoch())
          .count()
          << "\"}," << std::endl;
        ctranMapperTimestampPoint last(0);
        for (auto& tsp : ts.recvCtrl) {
          f << "{\"name\": \"recvCtrl\", "
            << "\"cat\": \"NET\", "
            << "\"id\": \"" << id++ << "\", "
            << "\"ph\": \"X\", "
            << "\"pid\": \"" << tsp.peer << "\", "
            << "\"ts\": \""
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                tsp.now.time_since_epoch())
            .count()
            << "\", \"dur\": \"0\""
            << "\"}," << std::endl;
        }
        for (auto& tsp : ts.putIssued) {
          f << "{\"name\": \"put\", "
            << "\"cat\": \"NET\", "
            << "\"id\": \"" << id++ << "\", "
            << "\"ph\": \"b\", "
            << "\"pid\": \"" << tsp.peer << "\", "
            << "\"ts\": \""
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                tsp.now.time_since_epoch())
            .count()
            << "\"}," << std::endl;
        }
        id -= ts.putIssued.size();
        for (auto& tsp : ts.putComplete) {
          f << "{\"name\": \"put\", "
            << "\"cat\": \"NET\", "
            << "\"id\": \"" << id++ << "\", "
            << "\"ph\": \"e\", "
            << "\"pid\": \"" << tsp.peer << "\", "
            << "\"ts\": \""
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                tsp.now.time_since_epoch())
            .count()
            << "\"}," << std::endl;
          last = tsp;
        }
        f << "{\"name\": \"" << ts.algo << "\", "
          << "\"cat\": \"COL\", "
          << "\"id\": \"" << collId << "\", "
          << "\"ph\": \"e\", "
          << "\"pid\": \"0\", "
          << "\"ts\": \""
          << std::chrono::duration_cast<std::chrono::milliseconds>(
              last.now.time_since_epoch())
          .count()
          << "\"}," << std::endl;
      }
      f << "]" << std::endl;
      f.close();
    }
  }

  if (this->pimpl->memPool != nullptr) {
    this->pimpl->memPool->deregMem(
        [&](void* hdl) -> ncclResult_t { return this->deregMem(hdl); });
  }

  std::vector<void*> v = this->pimpl->mapperRegElemList->getAllElems();
  for (auto hdl : v) {
    NCCLCHECKIGNORE(this->deregMem(hdl));
  }

  // Report summary of this communicator before destroying it
  this->reportRegSnapshot();
  allCommHashCtranMapperMap.erase(this->commHash);

  // Report global counters only after all communicators have been destroyed
  if (allCommHashCtranMapperMap.empty()) {
    reportGlobalRegSnapshot();
  }

  delete this->pimpl->mapperRegElemList;

  delete this->pimpl->memPool;

  CUDACHECKIGNORE(cudaStreamDestroy(this->s));
}

ncclResult_t ctranMapper::impl::regMem(struct ctranMapperRegElem *mapperRegElem) {
  ncclResult_t res = ncclSuccess;
  auto dur = ctranMapperTimer();

  if (this->ctranIb != nullptr) {
    assert(mapperRegElem->ibRegElem == nullptr);
    NCCLCHECKGOTO(this->ctranIb->regMem(mapperRegElem->buf, mapperRegElem->len,
          &mapperRegElem->ibRegElem), res, exit);
  }

  if (this->ctranNvl != nullptr) {
    assert(mapperRegElem->nvlRegElem == nullptr);
    NCCLCHECKGOTO(this->ctranNvl->regMem(mapperRegElem->buf, mapperRegElem->len,
          &mapperRegElem->nvlRegElem), res, exit);
  }

  this->numRegistrations++;
  this->totalNumRegistrations++;

  mapperRegElem->state = ctranMapperRegElemState::REGISTERED;
  registerDurs.push_back(dur.durationMs());

  INFO(NCCL_COLL, "CTRAN-MAPPER: register buffer %p len %ld (cached %u registered %u total cached %u total registered %u total dynamically registered %u)",
      mapperRegElem->buf, mapperRegElem->len, this->numCachedRegistrations, this->numRegistrations,
      this->totalNumCachedRegistrations, this->totalNumRegistrations, this->totalNumDynamicRegistrations);

  // Allow snapshot report during long job running
  if (NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT > 0 &&
      registerDurs.size() % NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT == 0) {
    reportGlobalRegSnapshot();
  }

exit:
  return res;
}

ncclResult_t ctranMapper::impl::deregMem(struct ctranMapperRegElem *mapperRegElem) {
  ncclResult_t res = ncclSuccess;
  auto dur = ctranMapperTimer();

  if (this->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->ctranIb->deregMem(mapperRegElem->ibRegElem), res, exit);
  }

  if (this->ctranNvl != nullptr) {
    NCCLCHECKGOTO(this->ctranNvl->deregMem(mapperRegElem->nvlRegElem), res, exit);
  }

  this->numRegistrations--;
  deregisterDurs.push_back(dur.durationMs());

  INFO(NCCL_COLL, "CTRAN-MAPPER: deregiter buffer %p len %ld (cached %u registered %u total cached %u total registered %u total dynamically registered %u)",
      mapperRegElem->buf, mapperRegElem->len, this->numCachedRegistrations, this->numRegistrations,
      this->totalNumCachedRegistrations, this->totalNumRegistrations, this->totalNumDynamicRegistrations);

exit:
  return res;
}

ncclResult_t ctranMapper::regMem(const void *buf, std::size_t len, void **hdl, bool forceRegist) {
  ncclResult_t res = ncclSuccess;
  struct ctranMapperRegElem *mapperRegElem = nullptr;

  cudaPointerAttributes attr;
  CUDACHECKGOTO(cudaPointerGetAttributes(&attr, buf), res, exit);
  if (attr.type != cudaMemoryTypeDevice) {
    WARN("CTRAN-MAPPER: buf %p is not a device buffer\n", buf);
    res = ncclSystemError;
    goto exit;
  }

  mapperRegElem = new struct ctranMapperRegElem;
  mapperRegElem->buf = buf;
  mapperRegElem->len = len;
  mapperRegElem->ibRegElem = nullptr;
  mapperRegElem->nvlRegElem = nullptr;
  mapperRegElem->state = ctranMapperRegElemState::CACHED;

  this->pimpl->mapperRegElemList->insert(buf, len, reinterpret_cast<void *>(mapperRegElem), hdl);

  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::eager || forceRegist) {
    NCCLCHECKGOTO(this->pimpl->regMem(mapperRegElem), res, fail);
  } else {
    // In lazy registration
    this->pimpl->numCachedRegistrations++;
    this->pimpl->totalNumCachedRegistrations++;
  }

exit:
  return res;
fail:
  if (*hdl) {
    this->pimpl->mapperRegElemList->remove(*hdl);
  }
  delete mapperRegElem;
  goto exit;
}

ncclResult_t ctranMapper::deregMem(void *hdl) {
  ncclResult_t res = ncclSuccess;

  if (hdl == nullptr) {
    return ncclSuccess;
  }

  struct ctranMapperRegElem *mapperRegElem = nullptr;
  this->pimpl->mapperRegElemList->lookup(hdl, (void **) &mapperRegElem);

  if(mapperRegElem->state == ctranMapperRegElemState::REGISTERED) {
    NCCLCHECKGOTO(this->pimpl->deregMem(mapperRegElem), res, exit);
  } else {
    // Just remove cache if the buffer is never registered
    this->pimpl->numCachedRegistrations--;
  }

exit:
  this->pimpl->mapperRegElemList->remove(hdl);
  delete mapperRegElem;
  return res;
}

ncclResult_t ctranMapper::searchRegHandle(const void *buf, std::size_t len, void **hdl, bool *dynamicRegist) {
  ncclResult_t res = ncclSuccess;
  auto dur = ctranMapperTimer();
  // Determine whether the buffer has already registered
  bool lookupHit = true;

  this->pimpl->mapperRegElemList->search(buf, len, hdl);

  if (*hdl != nullptr) {
    struct ctranMapperRegElem *mapperRegElem;
    this->pimpl->mapperRegElemList->lookup(*hdl, (void **) &mapperRegElem);

    // User has registerd it but we delay it until now due to lazy registration
    if (mapperRegElem->state == ctranMapperRegElemState::CACHED) {
      this->pimpl->numCachedRegistrations--;
      NCCLCHECKGOTO(this->pimpl->regMem(mapperRegElem), res, exit);
      lookupHit = false;
    }
    *dynamicRegist = false;
  } else {
    // Oops, the buffer is not registered by user. Thus, we have to register it on demand
    this->pimpl->totalNumDynamicRegistrations++;
    NCCLCHECKGOTO(this->regMem(buf, len, hdl, true /* force register */), res, exit);
    // caller is responsible for deregisgration
    *dynamicRegist = true;
    lookupHit = false;
  }

  if (lookupHit) {
    lookupHitDurs.push_back(dur.durationMs());
    this->pimpl->totalNumRegLookupHit++;
  } else {
    lookupMissDurs.push_back(dur.durationMs());
    this->pimpl->totalNumRegLookupMiss++;
  }

exit:
  return res;
}

ncclResult_t ctranMapper::icopy(void *dbuf, const void *sbuf, std::size_t len, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  *req = new ctranMapperRequest(this);
  CUDACHECKGOTO(cudaMemcpyAsync(dbuf, sbuf, len, cudaMemcpyDefault, this->s), res, exit);

exit:
  return res;
}

ncclResult_t ctranMapper::progress(void) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->progress(), res, exit);
  }
  if (this->pimpl->ctranNvl != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranNvl->progress(), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranMapper::getTmpBuf(void** addr, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  this->tmpBufLock.lock();
  *hdl = nullptr;
  std::size_t bufLen;
  NCCLCHECKGOTO(this->pimpl->memPool->getBuf(len, addr, hdl, &bufLen), res, exit);
  if (*hdl == nullptr) {
    NCCLCHECKGOTO(this->regMem(*addr, bufLen, hdl), res, exit);
  }

exit:
  this->tmpBufLock.unlock();
  return res;
}

ncclResult_t ctranMapper::releaseTmpBuf(void* addr, void *hdl) {
  ncclResult_t res = ncclSuccess;

  this->tmpBufLock.lock();
  NCCLCHECKGOTO(this->pimpl->memPool->release(addr, hdl), res, exit);

exit:
  this->tmpBufLock.unlock();
  return res;
}

ncclResult_t ctranMapper::isendCtrl(void *buf, void *hdl, int rank, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    struct ctranMapperRegElem *mapperRegElem;
    this->pimpl->mapperRegElemList->lookup(hdl, (void **) &mapperRegElem);

    if (req == nullptr) {
      NCCLCHECKGOTO(this->pimpl->ctranIb->isendCtrl(buf, mapperRegElem->ibRegElem, rank, nullptr), res, exit);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->isendCtrl(buf, mapperRegElem->ibRegElem, rank, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return res;
}

ncclResult_t ctranMapper::irecvCtrl(void **buf, struct ctranMapperRemoteAccessKey *key, int rank,
    ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    if (req == nullptr) {
      NCCLCHECKGOTO(this->pimpl->ctranIb->irecvCtrl(buf, &key->ibKey, rank, nullptr), res, exit);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->irecvCtrl(buf, &key->ibKey, rank, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return res;
}

ncclResult_t ctranMapper::iput(const void *sbuf, void *dbuf, std::size_t len, int rank, void *shdl,
    struct ctranMapperRemoteAccessKey remoteAccessKey, bool notify, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    struct ctranMapperRegElem *mapperRegElem;
    this->pimpl->mapperRegElemList->lookup(shdl, (void **) &mapperRegElem);

    if (req == nullptr) {
      NCCLCHECKGOTO(this->pimpl->ctranIb->iput(sbuf, dbuf, len, rank, mapperRegElem->ibRegElem, remoteAccessKey.ibKey,
            notify, nullptr), res, exit);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->iput(sbuf, dbuf, len, rank, mapperRegElem->ibRegElem, remoteAccessKey.ibKey,
            notify, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return res;
}

ncclResult_t ctranMapper::checkNotify(int rank, bool *notify) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->checkNotify(rank, notify), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranMapper::waitNotify(int rank) {
  ncclResult_t res = ncclSuccess;

  bool notify = false;
  while (notify == false) {
    NCCLCHECKGOTO(this->checkNotify(rank, &notify), res, exit);
  }

exit:
  return res;
}
