// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef PROXY_TRACE_H
#define PROXY_TRACE_H

#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "TraceUtils.h"
#include "debug.h"
#include "nccl_common.h"

// TODO: limite number of records in completeSends_
#define NCCL_PROXY_TRACE_NUM_RECORDS 100

enum ProxyOpStepStatus {
  POSTED,
  RECEIVED,
  TRANSMITTED,
  DONE,
  NUM_STATUS,
};

// record progress state per comm per collective per channel
struct ProxyCollTraceEntry {
  uint64_t commHash{0};
  uint64_t opCount{0};
  ncclFunc_t coll{ncclFuncBroadcast};
  int channelId{-1};
  int nSteps{0};
  // ranks in current communicator;
  // rank may be from another one on the node
  int rank{-1};
  int remoteRank{-1};

  enum OpType { SEND, RECV };
  OpType opType{SEND};

  struct StepRecord {
    int step{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> ts;
  };
  StepRecord stepRecords[NUM_STATUS];

  std::chrono::time_point<std::chrono::high_resolution_clock> startTs;
  std::chrono::time_point<std::chrono::high_resolution_clock> doneTs;
  bool done{false};

  // serialize the entry to a json format string
  std::string serialize();
};

struct ncclProxyArgs;
struct ncclComm;

class ProxyTrace {
 public:
  ProxyTrace();
  ~ProxyTrace(){};

  // Record when starts a send operation on proxy thread (see sendProxyProgress)
  ncclResult_t startSend(struct ncclProxyArgs* args);

  // Record when completes a send operation on proxy thread (see
  // sendProxyProgress)
  ncclResult_t completeSend(struct ncclProxyArgs* args);

  // Record internal step and timestamp for an ongoing send operation (see
  // sendProxyProgress)
  ncclResult_t recordSendProgress(
      struct ncclProxyArgs* args,
      int sub,
      int step,
      ProxyOpStepStatus status);

  // Record when starts a recv operation on proxy thread (see recvProxyProgress)
  ncclResult_t startRecv(struct ncclProxyArgs* args);

  // Record when completes a recv operation on proxy thread (see
  // recvProxyProgress)
  ncclResult_t completeRecv(struct ncclProxyArgs* args);

  // Record internal step and timestamp for an ongoing recv operation (see
  // recvProxyProgress)
  ncclResult_t recordRecvProgress(
      struct ncclProxyArgs* args,
      int sub,
      int step,
      ProxyOpStepStatus status);

  // Allow proxy thread to mock a send failure if the current send operation
  // matches user specified config (see NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK).
  // Return mocked flag to true if mocked, otherwise return false.
  ncclResult_t runSendFailureMock(
      struct ncclProxyArgs* args,
      int sub,
      int step,
      bool& mocked);

  // print details of internal structures for both active and completed
  // send/recvs. For debugging.
  void print();

  // Query number of active send or recv operations for a given communicator
  size_t queryNumActiveSends(uint64_t commHash);
  size_t queryNumActiveRecvs(uint64_t commHash);

  // Query all active send o receive operations for a given communicator. Return
  // number of actually found entries. The entries are copied to the entries
  // vector.
  size_t queryActiveSends(
      uint64_t commHash,
      std::vector<ProxyCollTraceEntry>& entries);
  size_t queryActiveRecvs(
      uint64_t commHash,
      std::vector<ProxyCollTraceEntry>& entries);

  // Query number of completed send or receive operations for a given
  // communicator
  size_t queryNumCompletedSends(uint64_t commHash);
  size_t queryNumCompletedRecvs(uint64_t commHash);

  // Query completed collectives for a given communicator. Return number of
  // completed collectives, list of opCounts are copied to the opCounts vector
  // starting from the first completed one.
  size_t queryCompletedColls(
      uint64_t commHash,
      std::vector<uint64_t>& opCounts);

  // Query completed collectives for a given communicator. Return number of
  // completed collectives, list of opCounts are copied to the opCounts vector
  // starting from the last completed one.
  size_t queryLastNCompletedColls(
      uint64_t commHash,
      size_t numCompleted,
      std::vector<uint64_t>& opCounts);

  // Query the last numCompleted number of completed of send or receive
  // operations for a given communicator. Return number of actually found
  // entries. The entries are copied to the entries vector starting from the
  // last completed one.
  size_t queryLastNCompletedSends(
      uint64_t commHash,
      size_t numCompleted,
      std::vector<ProxyCollTraceEntry>& entries);
  size_t queryLastNCompletedRecvs(
      uint64_t commHash,
      size_t numCompleted,
      std::vector<ProxyCollTraceEntry>& entries);

  bool queryColl(
      uint64_t commHash,
      uint64_t opCount,
      std::vector<ProxyCollTraceEntry>& sends,
      std::vector<ProxyCollTraceEntry>& recvs);

 private:
  inline ncclResult_t createActiveEntries(
      struct ncclProxyArgs* args,
      ProxyCollTraceEntry::OpType opType);
  inline ncclResult_t completeTraceEntries(
      struct ncclProxyArgs* args,
      ProxyCollTraceEntry::OpType opType);
  inline ncclResult_t updateTraceEntryStep(
      struct ncclProxyArgs* args,
      int sub,
      int step,
      ProxyOpStepStatus status,
      ProxyCollTraceEntry::OpType opType);

  enum Features {
    TRACE = 1,
    VERBOSE = 2,
  };
  int features_{0}; // bitwise OR of Features

  struct FailureMockConfig {
    bool enabled{false};
    uint64_t opCount;
    int channelId;
    int rank;
    int remoteRank;
    int step;
    std::string serialize();
  };
  FailureMockConfig failureMockConfig_;
  void failureMockSetup();

  std::mutex mutex_;

  // Current active send/recv operations.
  // Use map to quickly find the record during active progress
  std::unordered_map<
      uint64_t /* commHash*/,
      std::unordered_map<
          uint64_t /* opCount*/,
          std::unordered_map<
              int /* channelId*/,
              std::unique_ptr<ProxyCollTraceEntry>>>>
      activeSends_;
  std::unordered_map<
      uint64_t,
      std::unordered_map<
          uint64_t,
          std::unordered_map<int, std::unique_ptr<ProxyCollTraceEntry>>>>
      activeRecvs_;

  // Completed send/recv operations.
  // Use deque to mainain completed order. Quick search is not needed as it
  // occurs only when commDump is called
  std::deque<std::unique_ptr<ProxyCollTraceEntry>> completeSends_;
  std::deque<std::unique_ptr<ProxyCollTraceEntry>> completeRecvs_;

  // Completed collectives for each communicator. Updated when both send and
  // recv are completed
  std::unordered_map<uint64_t, std::vector<uint64_t>> completedColls_;

  friend class CollTrace;
};

ncclResult_t proxyTraceInit(
    struct ncclProxyState* state,
    struct ncclComm* comm);

#define PROXY_TRACE_CALL(state, cmd) \
  do {                               \
    if (state->trace) {              \
      NCCLCHECK((cmd));              \
    }                                \
  } while (0)
#endif
