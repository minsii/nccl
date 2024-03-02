// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CollTrace.h"
#include "FbInternal.h"
#include "bootstrap.h"
#include "comm.h"
#include "nccl.h"
#include "ExtChecks.h"

#include <CtranUtils.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
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
  try {
    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lu rank %d - Destroy START",
        comm_,
        comm_->commHash,
        comm_->rank);

    eventQueue_.push(
      std::unique_ptr<EventInfo>(new EventInfo(EventInfo::EventType::TERMINATE)));
    if (profilingWorkerThread_.joinable()) {
      profilingWorkerThread_.join();
    }

    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lu rank %d - Destroy COMPLETE",
        comm_,
        comm_->commHash,
        comm_->rank);
  } catch (const std::exception& e) {
    WARN(
        "COLLTRACE: comm %p commHash %lu rank %d - Destroy FAILED: %s",
        comm_,
        comm_->commHash,
        comm_->rank,
        e.what());
  }
  catch(...) {
    WARN(
        "COLLTRACE: comm %p commHash %lu rank %d - Destroy FAILED: Unkown exception",
        comm_,
        comm_->commHash,
        comm_->rank);
  }
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
  std::lock_guard<std::mutex> lock(workerMutex_);
  CollTraceDump dump{};
  if (curEventState_ == EventState::IN_PROGRESS) {
    dump.currentColl = curEvent_;
    dump.currentCollState = curEventState_;
  }

  dump.pendingColls = eventQueue_.dumpQueue();
  dump.pastColls = results_;
  return dump;
}

void* CollTrace::collTraceThreadFn(CollTrace* ct) {
  return ct->collTraceThreadFnImpl();
}

void* CollTrace::collTraceThreadFnImpl() {
  CUDACHECKTHROW(cudaSetDevice(comm_->cudaDev));

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lu rank %d - worker thread STARTED",
      comm_,
      comm_->commHash,
      comm_->rank);

  while (true) {
    curEventState_ = EventState::PENDING;
    curEvent_ = nullptr;

    // For testing purpose only. During testing, we want to ensure the worker
    // thread reached a steady state before dumping so that the trace dump
    // result is predictable. Otherwise the test can be flaky.
    if (waitingForQueueEmpty_ && eventQueue_.isEmpty()) {
      {
        std::unique_lock<std::mutex> lock(waitQueueEmptyMutex_);
        waitingForQueueEmpty_ = false;
      }
      waitQueueEmptyCv_.notify_all();
    }

    // We intentionally didn't hold the event queue lock till curEvent is
    // updated. That will potentially create deadlock.
    // Downside of current approach is we might miss one pending event in the
    // dump in very rare occasion. But since the worker thread haven't started
    // to wait for the event, it should be fine.
    {
      auto tmp_event = eventQueue_.waitPop();
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_ = std::move(tmp_event);
    }

    if (curEvent_->eventType == EventInfo::EventType::TERMINATE) {
      break;
    } else if (curEvent_->eventType == EventInfo::EventType::WAKE_UP) {
      continue;
    }
    curEventState_ = EventState::IN_PROGRESS;
    cudaError_t res = cudaEventSynchronize(curEvent_->stop.get());
    curEventState_ = EventState::DONE;
    float latency = -1;

    if (res == cudaSuccess) {
      res = cudaEventElapsedTime(&latency, curEvent_->start.get(), curEvent_->stop.get());
    }

    ResultInfo result{
      .opCount= curEvent_->opCount,
      .info = curEvent_->info,
      .stream = curEvent_->stream,
      .iteration = curEvent_->iteration,
      .latency = res == cudaSuccess? latency: -1,
    };

    {
      std::lock_guard<std::mutex> lock(workerMutex_);
      results_.push_back(result);
    }

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

  outputResults();

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lu rank %d - worker thread TERMINATE",
      comm_,
      comm_->commHash,
      comm_->rank);
  return nullptr;
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

void CollTrace::waitForWorkerFinishQueue() {
  std::unique_lock<std::mutex> waitLock(waitQueueEmptyMutex_);
  waitingForQueueEmpty_ = true;
  eventQueue_.push(
  std::unique_ptr<EventInfo>(new EventInfo(EventInfo::EventType::WAKE_UP)));
  waitQueueEmptyCv_.wait(waitLock, [this] { return !waitingForQueueEmpty_; });
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
  if (comm->collTrace) {
    comm->collTrace.reset();
  }
  // Try catch clause here is not going to be useful as destructors are noexcept
  // by default. Instead of throwing an exception it will just crash the program.
  // We need to think about a better way to handle this.
  return ncclSuccess;
}
