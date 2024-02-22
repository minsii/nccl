// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CollTrace.h"
#include <string>
#include "FbInternal.h"
#include "bootstrap.h"
#include "comm.h"
#include "nccl.h"

#include <CtranUtils.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <sstream>

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_COLLTRACE
   type        : stringlist
   default     :
   description : |-
     Enable collective trace collection by leveraging cuda events
     and a background thread. Valid options are comma separated list
     of the following features. Leave empty to disable all features.
     verbose - print every completed event as NCCL INFO log. Mostly for debug.
     trace - Just enable collective trace.
     file    - dump traced events to file at communicator destory. Also see
            NCCL_COLLTRACE_DIR.
     online_tuning - enable online tuning
     (Other FB internal featuers are not listed)

 - name        : NCCL_COLLTRACE_DIR
   type        : string
   default     : ""
   description : |-
     Directory for CollTrace to dump.
     Can be either local or FB internal remote URL.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

CollTrace::CollTrace(ncclComm* comm) : comm_(comm) {
  std::vector<std::string> enabledFeatures;
  if (!NCCL_COLLTRACE.empty()) {
    for (auto& f : NCCL_COLLTRACE) {
      if (f == "verbose") {
        features |= CollTrace::Features::VERBOSE;
        enabledFeatures.push_back(f);
      } else if (f == "file") {
        features |= CollTrace::Features::FILE;
        enabledFeatures.push_back(f);
      } else if (f == "online_tuning") {
        features |= CollTrace::Features::ONLINE_TUNING;
        enabledFeatures.push_back(f);
      } else if (f == "fb") {
        features |= CollTrace::Features::FB_IO_DURING_RUN;
        enabledFeatures.push_back(f);
      } else if (f == "trace") {
        features |= CollTrace::Features::TRACE;
        enabledFeatures.push_back(f);
      }
    }
  }

  // create worker thread
  profilingWorkerThread_ = std::thread{collTraceThreadFn, this};

  std::string enabledFeaturesStr = vecToStr(enabledFeatures);
  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lu rank %d enabled features: %s - Init COMPLETE",
      comm,
      comm->commHash,
      comm->rank,
      enabledFeaturesStr.c_str());
}

CollTrace::~CollTrace() {
  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lu rank %d - Destroy START",
      comm_,
      comm_->commHash,
      comm_->rank);

  workerThreadExitSignal_ = true;
  profilingWorkerThread_.join();

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lu rank %d - Destroy COMPLETE",
      comm_,
      comm_->commHash,
      comm_->rank);
}

void CollTrace::outputResults() {
  // If NCCL_COLLTRACE_DIR is set, then write profiling data to file
  if (features & CollTrace::Features::FILE && !NCCL_COLLTRACE_DIR.empty()) {
    std::stringstream stream;
    stream << "[\n  {\n";
    for (auto it = results_.begin(); it != results_.end(); ++it) {
      if (it != results_.begin()) {
        stream << "  },\n  {\n";
      }
      stream << "    \"coll\": \"" << it->info.opName << "\",\n"
             << "    \"msg_size\": \""
             << (it->info.count * ncclTypeSize(it->info.datatype)) << "\",\n"
             << "    \"latency\": " << it->latency << "\n";
    }
    stream << "  }\n]";

    const std::string fileName =
        NCCL_COLLTRACE_DIR + "/" + std::to_string(comm_->rank) + "_online.json";
    INFO(
        NCCL_ALL,
        "COLLTRACE: rank %d writing %lu online profiler data to : %s",
        comm_->rank,
        results_.size(),
        fileName.c_str());

    if (ncclIsFbPath(fileName)) {
      ncclFbUpload(stream.str(), fileName);
    } else {
      std::ofstream f(fileName);
      f << stream.str();
      f.close();
    }
  }
}

CollTrace::CollTraceDump CollTrace::dumpTrace() {
  CollTraceDump dump {};
  if (curEvent_ != nullptr) {
    dump.currentEvent = curEvent_;
    dump.currentEventState = curEventState_;
  }

  dump.pendingEvents = eventQueue_.dumpQueue();
  // For now we will assume the collTraceThread is either finished or hang.
  // TODO: add a lock to protect the dump process.
  dump.pastResults = results_;
  return dump;
}

void* CollTrace::collTraceThreadFn(CollTrace* ct) {
  return ct->collTraceThreadFnImpl();
}

void* CollTrace::collTraceThreadFnImpl() {
  ncclResult_t res = ncclSuccess;
  CUDACHECKGOTO(cudaSetDevice(comm_->cudaDev), res, fail);

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lu rank %d - worker thread STARTED",
      comm_,
      comm_->commHash,
      comm_->rank);

  while (true) {
    curEvent_ = eventQueue_.tryPop();
    // Fixme: think of a better word to describe the state of the current event
    // here we are actually "uncertain" what is the current state of the event
    // rather than we know the event is waiting to be processed.
    curEventState_ = EventState::PENDING;
    if (curEvent_) {
      if (curEvent_->info.count != 0) {
        curEventState_ = EventState::RUNNING;
        cudaError_t res = cudaEventSynchronize(curEvent_->stop.get());
        curEventState_ = EventState::FINISHED;
        float latency = -1;
        res = res == cudaSuccess
            ? cudaEventElapsedTime(
                  &latency, curEvent_->start.get(), curEvent_->stop.get())
            : res;
        ResultInfo result;
        result.opCount = curEvent_->opCount;
        result.info = curEvent_->info;
        result.stream = curEvent_->stream;
        if (res == cudaSuccess) {
          result.latency = latency;
        }
        result.iteration = curEvent_->iteration;
        results_.push_back(result);

        // Free the event objects
        eventPool_.add(std::move(curEvent_->start));
        eventPool_.add(std::move(curEvent_->stop));

        // FIXME: cannot record protocol for sendrecvs since a grouped sendrecv
        // may contain multiple protocols
        if (features & CollTrace::Features::FB_IO_DURING_RUN) {
          COLLTRACE_IO_FB_DURING_RUN(result, comm_->rank);
        }

        if (features & CollTrace::Features::VERBOSE) {
          INFO(
              NCCL_COLL,
              "COLLTRACE: opCount %lx %s sendbuff %p recvbuff %p count %ld datatype %s op %s root %d algorithm %s protocol %s nchannels %d nthreads %d latency %.2f us",
              result.opCount,
              result.info.opName,
              result.info.sendbuff,
              result.info.recvbuff,
              result.info.count,
              getDatatypeStr(result.info.datatype).c_str(),
              getRedOpStr(result.info.op).c_str(),
              result.info.root,
              result.info.algorithm >= 0 ? ncclAlgoStr[result.info.algorithm]
                                         : "N/A",
              result.info.protocol >= 0 ? ncclProtoStr[result.info.protocol]
                                        : "N/A",
              result.info.nChannels,
              result.info.nThreads,
              result.latency * 1000);
        }

        // FIXME: we should revisit bootstrapAllGather() here since commAbort
        // may be called either on local rank or a remote rank causing socket
        // failure
        if (comm_->tuner != NULL &&
            features & CollTrace::Features::ONLINE_TUNING) {
          // Online tuning - average latencies across ranks & send to tuner
          float* latencies = NULL;
          NCCLCHECKIGNORE(ncclCalloc(&latencies, curEvent_->info.comm->nRanks));
          latencies[curEvent_->info.comm->rank] = latency;
          NCCLCHECKIGNORE(bootstrapAllGather(
              curEvent_->info.comm->bootstrap, latencies, sizeof(float)));
          float sum = 0.0;
          for (int i = 0; i < curEvent_->info.comm->nRanks; i++) {
            sum += latencies[i];
          }

          free(latencies);
          sum /= (float)curEvent_->info.comm->nRanks;

          curEvent_->info.comm->tuner->addOnlineResult(
              curEvent_->info.coll,
              curEvent_->info.count * ncclTypeSize(curEvent_->info.datatype),
              curEvent_->iteration,
              sum,
              curEvent_->info.algorithm,
              curEvent_->info.protocol,
              curEvent_->info.nChannels,
              curEvent_->info.nThreads);
        }
      }
    curEvent_ = nullptr;
    } else {
      if (workerThreadExitSignal_ && eventQueue_.isEmpty()) {
        outputResults();
        break;
      }
    }
  }

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lu rank %d - worker thread TERMINATE",
      comm_,
      comm_->commHash,
      comm_->rank);
  return NULL;

fail:
  WARN("COLLTRACE: error occured on worker thread, return error %d", res);
  return NULL;
}

std::unique_ptr<EventInfo> CollTrace::getEventFromPool() {
  std::unique_ptr<EventInfo> eventInfo(new EventInfo);
  eventInfo->start = eventPool_.takeOne();
  eventInfo->stop = eventPool_.takeOne();
  if (!eventInfo->start || !eventInfo->stop) {
    std::unique_ptr<EventInfo> nullEventInfo(nullptr);
    return nullEventInfo;
  }
  return eventInfo;
}

void CollTrace::enqueueEvent(std::unique_ptr<EventInfo> eventInfo) {
  eventQueue_.push(std::move(eventInfo));
}

ncclResult_t collTraceInit(ncclComm* comm) {
  try {
    if (!NCCL_COLLTRACE.empty()) {
      comm->collTrace = std::unique_ptr<CollTrace>(new CollTrace(comm));
    }
  } catch (const std::exception& e) {
    WARN("COLLTRACE initialization failed: %s\n", e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  try {
    if (comm->collTrace) {
      comm->collTrace.reset();
    }
  } catch (const std::exception& e) {
    WARN("COLLTRACE destruction failed: %s\n", e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}
