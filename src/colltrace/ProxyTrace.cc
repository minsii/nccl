// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ProxyTrace.h"
#include <cstddef>
#include <map>
#include "ExtUtils.h"
#include "comm.h"
#include "debug.h"
#include "proxy.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_PROXYTRACE
   type        : stringlist
   default     :
   description : |-
     Enable proxy operation trace collection on the proxy thread. Valid options
     are comma separated list of the following features. Leave empty to disable
     all features.
     trace - enable trace only.
     verbose - print every proxy operation step as NCCL INFO log.

 - name        : NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK
   type        : stringlist
   default     :
   description : |-
     Backdoor to mock a hanging progress in proxy thread.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

static std::map<ProxyOpStepStatus, std::string> proxySendStepStatusStrMap = {
    {ProxyOpStepStatus::POSTED, "POSTED"},
    {ProxyOpStepStatus::TRANSMITTED, "TRANSMITTED"},
    {ProxyOpStepStatus::DONE, "DONE"},
};
static std::map<ProxyOpStepStatus, std::string> proxyRecvStepStatusStrMap = {
    {ProxyOpStepStatus::POSTED, "POSTED"},
    {ProxyOpStepStatus::RECEIVED, "RECEIVED"},
    {ProxyOpStepStatus::TRANSMITTED, "TRANSMITTED"},
    {ProxyOpStepStatus::DONE, "DONE"},
};

static std::unordered_map<ProxyCollTraceEntry::OpType, std::string>
    proxyOpTypetrMap = {
        {ProxyCollTraceEntry::OpType::SEND, "SEND"},
        {ProxyCollTraceEntry::OpType::RECV, "RECV"},
};

ProxyTrace::ProxyTrace() {
  std::vector<std::string> enabledFeatures;
  for (auto& f : NCCL_PROXYTRACE) {
    if (f == "verbose") {
      features_ |= ProxyTrace::Features::VERBOSE;
      enabledFeatures.push_back(f);
    } else if (f == "trace") {
      features_ |= ProxyTrace::Features::TRACE;
      enabledFeatures.push_back(f);
    }
  }

  std::string enabledFeaturesStr = vecToStr(enabledFeatures);
  INFO(
      NCCL_INIT,
      "PROXYTRACE: initialized with features: %s",
      enabledFeaturesStr.c_str());

  this->failureMockSetup();
}

std::string ProxyTrace::FailureMockConfig::serialize() {
  std::stringstream ss;
  ss << "{";
  ss << "opcount=" << opCount << ",";
  ss << "channelId=" << channelId << ",";
  ss << "rank=" << rank << ",";
  ss << "remoteRank=" << remoteRank << ",";
  ss << "step=" << step;
  ss << "}";
  return ss.str();
}

void ProxyTrace::failureMockSetup() {
  std::vector<std::string> sendFailureMockFmt = {
      "<opCount>", "<channelId>", "<rank>", "<remoteRank>", "<step>"};

  // Enable mock only when all required fields are set
  if (NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.size() ==
      sendFailureMockFmt.size()) {
    failureMockConfig_.opCount =
        std::stoi(NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK[0]);
    failureMockConfig_.channelId =
        std::stoi(NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK[1]);
    failureMockConfig_.rank =
        std::stoi(NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK[2]);
    failureMockConfig_.remoteRank =
        std::stoi(NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK[3]);
    failureMockConfig_.step =
        std::stoi(NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK[4]);

    failureMockConfig_.enabled = true;
    std::string sendFailureMockStr = failureMockConfig_.serialize();

    INFO(
        NCCL_INIT,
        "PROXYTRACE: setup network send failure mock: %s",
        sendFailureMockStr.c_str());
  } else if (!NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.empty()) {
    WARN(
        "PROXYTRACE: invalid value of NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK. Valid format: %s",
        vecToStr(sendFailureMockFmt, ",").c_str());
  }
}

static bool firstMock = true;
ncclResult_t ProxyTrace::runSendFailureMock(
    struct ncclProxyArgs* args,
    int sub,
    int step,
    bool& mocked) {
  if (failureMockConfig_.enabled &&
      args->commOpCount == failureMockConfig_.opCount &&
      args->subs[sub].channelId == failureMockConfig_.channelId &&
      args->rank == failureMockConfig_.rank &&
      args->remoteRank == failureMockConfig_.remoteRank &&
      step == failureMockConfig_.step) {
    std::string sendFailureMockStr = failureMockConfig_.serialize();
    // Only warn the first time, because proxy thread will hang here and repeat
    // the mock
    if (firstMock) {
      WARN(
          "PROXYTRACE: Mocked send failure, skiped SEND with %s",
          sendFailureMockStr.c_str());
    }
    firstMock = false;
    mocked = true;
  } else {
    mocked = false;
  }
  return ncclSuccess;
}

// Check if a given commHash:opCount:channelId exists in activeMap.
// Return true if it exists, false otherwise.
static inline bool checkActiveRecordExist(
    uint64_t commHash,
    uint64_t opCount,
    int channelId,
    std::unordered_map<
        uint64_t,
        std::unordered_map<
            uint64_t,
            std::unordered_map<int, std::unique_ptr<ProxyCollTraceEntry>>>>&
        activeMap) {
  return (
      activeMap.find(commHash) != activeMap.end() &&
      activeMap[commHash].find(opCount) != activeMap[commHash].end() &&
      activeMap[commHash][opCount].find(channelId) !=
          activeMap[commHash][opCount].end());
}

static inline bool checkActiveRecordExist(
    uint64_t commHash,
    uint64_t opCount,
    std::unordered_map<
        uint64_t,
        std::unordered_map<
            uint64_t,
            std::unordered_map<int, std::unique_ptr<ProxyCollTraceEntry>>>>&
        activeMap) {
  return (
      activeMap.find(commHash) != activeMap.end() &&
      activeMap[commHash].find(opCount) != activeMap[commHash].end());
}

inline ncclResult_t ProxyTrace::createActiveEntries(
    struct ncclProxyArgs* args,
    ProxyCollTraceEntry::OpType opType) {
  auto& activeMap =
      opType == ProxyCollTraceEntry::OpType::SEND ? activeSends_ : activeRecvs_;
  for (int subIdx = 0; subIdx < args->nsubs; subIdx++) {
    struct ncclProxySubArgs* sub = &args->subs[subIdx];
    if (checkActiveRecordExist(
            args->commHash, args->commOpCount, sub->channelId, activeMap)) {
      WARN(
          "PROXYTRACE: failed to create %s entry of commHash %lu opCount %lx channelId %d, because an active entry already exists",
          proxyOpTypetrMap[opType].c_str(),
          args->commHash,
          args->commOpCount,
          sub->channelId);
      return ncclInternalError;
    }

    auto entry =
        std::unique_ptr<ProxyCollTraceEntry>(new ProxyCollTraceEntry());
    entry->commHash = args->commHash;
    entry->opCount = args->commOpCount;
    entry->coll = args->coll;
    entry->nSteps = sub->nsteps;
    entry->channelId = sub->channelId;
    entry->rank = args->rank;
    entry->remoteRank = args->remoteRank;
    entry->startTs = std::chrono::high_resolution_clock::now();
    entry->opType = opType;

    if (features_ & ProxyTrace::Features::VERBOSE) {
      std::string entryStr = entry->serialize();
      INFO(NCCL_COLL, "PROXYTRACE: created entry %s", entryStr.c_str());
    }
    // Each sub corresponds to a channel, a collective may utilize multiple
    // channels; Create one entry for each channel
    activeMap[args->commHash][args->commOpCount][sub->channelId] =
        std::move(entry);
  }
  return ncclSuccess;
}

inline ncclResult_t ProxyTrace::completeTraceEntries(
    struct ncclProxyArgs* args,
    ProxyCollTraceEntry::OpType opType) {
  auto& activeMap =
      opType == ProxyCollTraceEntry::OpType::SEND ? activeSends_ : activeRecvs_;
  auto& completeDeque = opType == ProxyCollTraceEntry::OpType::SEND
      ? completeSends_
      : completeRecvs_;
  // For each completed channel, move to completed queue
  for (int subIdx = 0; subIdx < args->nsubs; subIdx++) {
    int channelId = args->subs[subIdx].channelId;
    if (!checkActiveRecordExist(
            args->commHash, args->commOpCount, channelId, activeMap)) {
      WARN(
          "PROXYTRACE: failed to complete %s entry of commHash %lu opCount %lx channelId %d, because no active entry exists",
          proxyOpTypetrMap[opType].c_str(),
          args->commHash,
          args->commOpCount,
          channelId);
      return ncclInternalError;
    }

    auto& entry = activeMap[args->commHash][args->commOpCount][channelId];
    entry->doneTs = std::chrono::high_resolution_clock::now();
    entry->done = true;

    if (features_ & ProxyTrace::Features::VERBOSE) {
      std::string entryStr = entry->serialize();
      INFO(NCCL_COLL, "PROXYTRACE: completed entry %s", entryStr.c_str());
    }

    completeDeque.push_back(std::move(entry));
    activeMap[args->commHash][args->commOpCount].erase(channelId);
  }

  // Erase collective if all channels have finished
  if (activeMap[args->commHash][args->commOpCount].empty()) {
    activeMap[args->commHash].erase(args->commOpCount);

    if (!checkActiveRecordExist(
            args->commHash, args->commOpCount, activeSends_) &&
        !checkActiveRecordExist(
            args->commHash, args->commOpCount, activeRecvs_)) {
      completedColls_[args->commHash].push_back(args->commOpCount);
      if (features_ & ProxyTrace::Features::VERBOSE) {
        INFO(
            NCCL_COLL,
            "PROXYTRACE: completed all entries of commHash %lu opCount %lx",
            args->commHash,
            args->commOpCount);
      }
    }
  }
  return ncclSuccess;
}

inline ncclResult_t ProxyTrace::updateTraceEntryStep(
    struct ncclProxyArgs* args,
    int sub,
    int step,
    ProxyOpStepStatus status,
    ProxyCollTraceEntry::OpType opType) {
  auto& activeMap =
      opType == ProxyCollTraceEntry::OpType::SEND ? activeSends_ : activeRecvs_;
  int channelId = args->subs[sub].channelId;
  if (!checkActiveRecordExist(
          args->commHash, args->commOpCount, channelId, activeMap)) {
    WARN(
        "PROXYTRACE: failed to update %s entry of commHash %lu opCount %lx channelId %d, because no active entry exists",
        proxyOpTypetrMap[opType].c_str(),
        args->commHash,
        args->commOpCount,
        channelId);

    mutex_.unlock();
    return ncclInternalError;
  }

  auto& entry = activeMap[args->commHash][args->commOpCount][channelId];
  entry->stepRecords[status].step = step;
  entry->stepRecords[status].ts = std::chrono::high_resolution_clock::now();

  mutex_.unlock();
  return ncclSuccess;
}

ncclResult_t ProxyTrace::startSend(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return createActiveEntries(args, ProxyCollTraceEntry::OpType::SEND);
};

ncclResult_t ProxyTrace::completeSend(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return completeTraceEntries(args, ProxyCollTraceEntry::OpType::SEND);
}

ncclResult_t ProxyTrace::recordSendProgress(
    struct ncclProxyArgs* args,
    int sub,
    int step,
    ProxyOpStepStatus status) {
  mutex_.lock(); // release lock inside updateTraceEntryStep
  return updateTraceEntryStep(
      args, sub, step, status, ProxyCollTraceEntry::OpType::SEND);
};

ncclResult_t ProxyTrace::startRecv(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return createActiveEntries(args, ProxyCollTraceEntry::OpType::RECV);
};

ncclResult_t ProxyTrace::completeRecv(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return completeTraceEntries(args, ProxyCollTraceEntry::OpType::RECV);
}

ncclResult_t ProxyTrace::recordRecvProgress(
    struct ncclProxyArgs* args,
    int sub,
    int step,
    ProxyOpStepStatus status) {
  mutex_.lock(); // release lock inside updateTraceEntryStep
  return updateTraceEntryStep(
      args, sub, step, status, ProxyCollTraceEntry::OpType::RECV);
};

void printActiveEntries(
    std::unordered_map<
        uint64_t,
        std::unordered_map<
            uint64_t,
            std::unordered_map<int, std::unique_ptr<ProxyCollTraceEntry>>>>&
        activeMap,
    const char* name) {
  INFO(NCCL_COLL, "PROXYTRACE: dump %s", name);
  for (auto& it : activeMap) {
    auto commHash = it.first;
    for (auto& it1 : it.second) {
      auto opCount = it1.first;
      for (auto& it2 : it1.second) {
        auto& entry = it2.second;
        std::string entryStr = entry->serialize();
        INFO(NCCL_COLL, "%s: %s", name, entryStr.c_str());
      }
    }
  }
}

void printCompleteEntries(
    std::deque<std::unique_ptr<ProxyCollTraceEntry>>& completeEntries,
    const char* name) {
  INFO(
      NCCL_COLL,
      "PROXYTRACE: dump %s, %ld completed entries",
      name,
      completeEntries.size());
  for (auto& entry : completeEntries) {
    std::string entryStr = entry->serialize();
    INFO(NCCL_COLL, "%s: %s", name, entryStr.c_str());
  }
}

void ProxyTrace::print() {
  std::lock_guard<std::mutex> lock(mutex_);
  printActiveEntries(this->activeSends_, "activeSends_");
  printActiveEntries(this->activeRecvs_, "activeRecvs_");
  printCompleteEntries(this->completeSends_, "completeSends_");
  printCompleteEntries(this->completeRecvs_, "completeRecvs_");
}

std::string ProxyCollTraceEntry::serialize() {
  std::stringstream ss;
  ss << "{commHash=" << commHash << ", opCount=" << std::hex << opCount
     << ", channelId=" << channelId << ", rank=" << rank
     << ", remoteRank=" << remoteRank << ", nsteps=" << nSteps << ", opType="
     << (opType == ProxyCollTraceEntry::OpType::SEND ? "SEND" : "RECV")
     << ", status=" << (done ? "DONE" : "IN_PROGRESS")
     << ", startTs=" << timePointToStr(startTs);
  if (done) {
    ss << ", doneTs=" << timePointToStr(doneTs);
  }
  ss << ", stepRecords=[";
  auto& statusStrMap = opType == ProxyCollTraceEntry::OpType::SEND
      ? proxySendStepStatusStrMap
      : proxyRecvStepStatusStrMap;
  bool first = true;
  for (auto it3 : statusStrMap) {
    auto status = it3.first;
    auto& stepRecord = stepRecords[status];
    if (!first) {
      ss << ", ";
    }
    ss << "{step=" << stepRecord.step << ", status=" << it3.second
       << ", ts=" << timePointToStr(stepRecord.ts) << "}";
    first = false;
  }
  ss << "]}";

  return ss.str();
}

static inline size_t queryNumActive(
    uint64_t commHash,
    std::unordered_map<
        uint64_t,
        std::unordered_map<
            uint64_t,
            std::unordered_map<int, std::unique_ptr<ProxyCollTraceEntry>>>>&
        activeMap) {
  size_t numActive = 0;
  if (activeMap.find(commHash) != activeMap.end()) {
    // iterate over all opCounts in a given commHash
    for (auto& it : activeMap[commHash]) {
      // for each opCount, count the number of active channels
      numActive += it.second.size();
    }
  }
  return numActive;
}

size_t ProxyTrace::queryNumActiveSends(uint64_t commHash) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryNumActive(commHash, activeSends_);
}

size_t ProxyTrace::queryNumActiveRecvs(uint64_t commHash) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryNumActive(commHash, activeRecvs_);
}

static inline size_t queryActives(
    uint64_t commHash,
    std::unordered_map<
        uint64_t,
        std::unordered_map<
            uint64_t,
            std::unordered_map<int, std::unique_ptr<ProxyCollTraceEntry>>>>&
        activeMap,
    std::vector<ProxyCollTraceEntry>& vec) {
  size_t numActive = 0;
  // iterate over all opCounts in a given commHash
  for (auto& it : activeMap[commHash]) {
    auto& entries = it.second;
    // iterate over all channels in a given opCount
    for (auto& it2 : entries) {
      auto& entry = it2.second;
      // copy entry
      vec.push_back(*entry);
      numActive++;
    }
  }
  return numActive;
}

size_t ProxyTrace::queryActiveSends(
    uint64_t commHash,
    std::vector<ProxyCollTraceEntry>& vec) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryActives(commHash, activeSends_, vec);
}

size_t ProxyTrace::queryActiveRecvs(
    uint64_t commHash,
    std::vector<ProxyCollTraceEntry>& vec) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryActives(commHash, activeRecvs_, vec);
}

static inline size_t queryNumCompleted(
    uint64_t commHash,
    std::deque<std::unique_ptr<ProxyCollTraceEntry>>& completeEntries) {
  size_t numCompleted = 0;
  for (auto& entry : completeEntries) {
    if (entry->commHash == commHash) {
      numCompleted++;
    }
  }
  return numCompleted;
}

size_t ProxyTrace::queryNumCompletedSends(uint64_t commHash) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryNumCompleted(commHash, completeSends_);
}

size_t ProxyTrace::queryNumCompletedRecvs(uint64_t commHash) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryNumCompleted(commHash, completeRecvs_);
}

static inline size_t queryLastNCompletes(
    uint64_t commHash,
    size_t numCompleted,
    std::deque<std::unique_ptr<ProxyCollTraceEntry>>& completeEntries,
    std::vector<ProxyCollTraceEntry>& vec) {
  size_t numFound = 0;
  for (auto rit = completeEntries.rbegin(); rit != completeEntries.rend();
       rit++) {
    auto& entry = *rit;
    if (entry->commHash == commHash) {
      // copy entry
      vec.push_back(*entry);
      numFound++;

      // stop search
      if (numFound == numCompleted) {
        return numFound;
      }
    }
  }
  return numFound;
}

size_t ProxyTrace::queryCompletedColls(
    uint64_t commHash,
    std::vector<uint64_t>& opCounts) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t numFound = 0;
  if (completedColls_.find(commHash) == completedColls_.end()) {
    return numFound;
  }

  for (auto rit = completedColls_[commHash].begin();
       rit != completedColls_[commHash].end();
       rit++) {
    auto opCount = *rit;
    opCounts.push_back(opCount);
    numFound++;
  }
  return numFound;
}

size_t ProxyTrace::queryLastNCompletedColls(
    uint64_t commHash,
    size_t numCompleted,
    std::vector<uint64_t>& opCounts) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t numFound = 0;
  if (completedColls_.find(commHash) == completedColls_.end()) {
    return numFound;
  }

  for (auto rit = completedColls_[commHash].rbegin();
       rit != completedColls_[commHash].rend();
       rit++) {
    auto opCount = *rit;
    opCounts.push_back(opCount);
    numFound++;

    // stop search; if numCompleted == -1 (condition below will be always
    // false), return all completed colls
    if (numFound == numCompleted) {
      return numFound;
    }
  }
  return numFound;
}

size_t ProxyTrace::queryLastNCompletedSends(
    uint64_t commHash,
    size_t numCompleted,
    std::vector<ProxyCollTraceEntry>& vec) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryLastNCompletes(commHash, numCompleted, completeSends_, vec);
}

size_t ProxyTrace::queryLastNCompletedRecvs(
    uint64_t commHash,
    size_t numCompleted,
    std::vector<ProxyCollTraceEntry>& vec) {
  std::lock_guard<std::mutex> lock(mutex_);
  return queryLastNCompletes(commHash, numCompleted, completeRecvs_, vec);
}

bool ProxyTrace::queryColl(
    uint64_t commHash,
    uint64_t opCount,
    std::vector<ProxyCollTraceEntry>& sends,
    std::vector<ProxyCollTraceEntry>& recvs) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool completed = true; // marked to false if any active entry is found

  if (activeSends_.find(commHash) != activeSends_.end() &&
      activeSends_[commHash].find(opCount) != activeSends_[commHash].end()) {
    completed = false;
    for (auto& it : activeSends_[commHash][opCount]) {
      // copy all active send entries
      sends.push_back(*it.second);
    }
  }

  if (activeRecvs_.find(commHash) != activeRecvs_.end() &&
      activeRecvs_[commHash].find(opCount) != activeRecvs_[commHash].end()) {
    completed = false;
    for (auto& it : activeRecvs_[commHash][opCount]) {
      // copy all active recv entries
      recvs.push_back(*it.second);
    }
  }

  for (auto& entry : completeSends_) {
    if (entry->commHash == commHash && entry->opCount == opCount) {
      // copy any completed send entries
      sends.push_back(*entry);
    }
  }

  for (auto& entry : completeRecvs_) {
    if (entry->commHash == commHash && entry->opCount == opCount) {
      // copy any completed receive entries
      recvs.push_back(*entry);
    }
  }
  return completed;
}

ncclResult_t proxyTraceInit(
    struct ncclProxyState* state,
    struct ncclComm* comm) {
  try {
    if (!NCCL_PROXYTRACE.empty()) {
      state->trace = std::unique_ptr<ProxyTrace>(new ProxyTrace());
    }
  } catch (const std::exception& e) {
    WARN(
        "PROXYTRACE initialization failed on comm %p commHash %lu rank %d: %s\n",
        comm,
        comm->commHash,
        comm->rank,
        e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}
