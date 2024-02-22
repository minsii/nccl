// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <sstream>
#include "Ctran.h"
#include "ProxyTrace.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn on NCCL debug logging for verbose check
    setenv("NCCL_DEBUG", "INFO", 0);
    setenv("NCCL_DEBUG_SUBSYS", "INIT,COLL", 0);

    // Initialize CVAR so that we can overwrite global variable in each test
    initEnv();
  }

  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class ProxyTraceTest : public ::testing::Test {
 public:
  ProxyTraceTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  void runAllReduce(const int count, const int nColl, ncclComm_t comm) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));

    for (int i = 0; i < nColl; i++) {
      NCCLCHECK_TEST(ncclAllReduce(
          sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    }
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  cudaStream_t stream;
};

TEST_F(ProxyTraceTest, VerboseAllReduce) {
  // overwrite ProxyTrace features before creating comm
  NCCL_PROXYTRACE.push_back("verbose");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  EXPECT_NE(comm->proxyState->trace, nullptr);

  if (comm->nNodes < 2) {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    NCCL_PROXYTRACE.clear();

    printf("Skipping test since nNodes < 2\n");
    GTEST_SKIP();
  }

  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  runAllReduce(count, nColl, comm);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "PROXYTRACE: completed entry {commHash=" << comm->commHash
       << ", opCount=" << std::hex << i << ", channelId=";
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
  NCCL_PROXYTRACE.clear();
}

static void checkCompletedEntry(
    ProxyCollTraceEntry& entry,
    ProxyCollTraceEntry::OpType opType,
    uint64_t opCount,
    ncclFunc_t coll,
    ncclComm* comm,
    std::chrono::time_point<std::chrono::high_resolution_clock> begin,
    std::chrono::time_point<std::chrono::high_resolution_clock> end) {
  EXPECT_EQ(entry.opCount, opCount);
  EXPECT_EQ(entry.commHash, comm->commHash);
  EXPECT_GT(entry.channelId, -1);
  EXPECT_EQ(entry.rank, comm->rank);
  EXPECT_GE(entry.remoteRank, 0);
  EXPECT_LT(entry.remoteRank, comm->nRanks);
  EXPECT_GT(entry.nSteps, 0);
  EXPECT_EQ(entry.coll, coll);
  EXPECT_EQ(entry.opType, opType);

  EXPECT_GT(entry.startTs.time_since_epoch(), begin.time_since_epoch());
  EXPECT_GT(entry.doneTs.time_since_epoch(), entry.startTs.time_since_epoch());
  EXPECT_LT(entry.doneTs.time_since_epoch(), end.time_since_epoch());
}

TEST_F(ProxyTraceTest, QueryAllReduce) {
  // overwrite ProxyTrace features before creating comm
  NCCL_PROXYTRACE.push_back("trace");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  if (comm->nNodes < 2) {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    NCCL_PROXYTRACE.clear();

    printf("Skipping test since nNodes < 2\n");
    GTEST_SKIP();
  }

  EXPECT_NE(comm->proxyState->trace, nullptr);

  const int count = 1048576;
  const int nColl = 10;

  auto begin = std::chrono::high_resolution_clock::now();
  uint64_t opCountStart = comm->opCount;

  runAllReduce(count, nColl, comm);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  auto end = std::chrono::high_resolution_clock::now();

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  EXPECT_EQ(comm->proxyState->trace->queryNumActiveSends(comm->commHash), 0);
  EXPECT_EQ(comm->proxyState->trace->queryNumActiveRecvs(comm->commHash), 0);

  size_t completedSends = 0;
  size_t completedRecvs = 0;
  for (uint64_t opCount = opCountStart; opCount < comm->opCount; opCount++) {
    std::vector<ProxyCollTraceEntry> sends;
    std::vector<ProxyCollTraceEntry> recvs;
    bool completed = comm->proxyState->trace->queryColl(
        comm->commHash, opCount, sends, recvs);
    EXPECT_EQ(completed, true);

    for (auto& entry : sends) {
      checkCompletedEntry(
          entry,
          ProxyCollTraceEntry::OpType::SEND,
          opCount,
          ncclFuncAllReduce,
          comm,
          begin,
          end);
    }
    for (auto& entry : recvs) {
      checkCompletedEntry(
          entry,
          ProxyCollTraceEntry::OpType::RECV,
          opCount,
          ncclFuncAllReduce,
          comm,
          begin,
          end);
    }

    completedSends += sends.size();
    completedRecvs += recvs.size();
  }

  EXPECT_EQ(
      comm->proxyState->trace->queryNumCompletedSends(comm->commHash),
      completedSends);
  EXPECT_EQ(
      comm->proxyState->trace->queryNumCompletedRecvs(comm->commHash),
      completedRecvs);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
  NCCL_PROXYTRACE.clear();
}

TEST_F(ProxyTraceTest, QueryHangAllReduce) {
  constexpr int hangChannelId = 1;
  constexpr int hangRank = 1;
  constexpr int hangRemoteRank = 0;
  constexpr int hangOpCount = 5;
  constexpr int hangOpStep = 1;

  // overwrite ProxyTrace features before creating comm
  NCCL_PROXYTRACE.push_back("trace");
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(std::to_string(hangOpCount));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(
      std::to_string(hangChannelId));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(std::to_string(hangRank));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(
      std::to_string(hangRemoteRank));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(std::to_string(hangOpStep));

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  if (comm->nNodes < 2) {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    NCCL_PROXYTRACE.clear();

    printf("Skipping test since nNodes < 2\n");
    GTEST_SKIP();
  }

  EXPECT_NE(comm->proxyState->trace, nullptr);

  const int count = 1048576;
  const int nColl = 10;

  auto begin = std::chrono::high_resolution_clock::now();
  uint64_t opCountStart = comm->opCount;
  runAllReduce(count, nColl, comm);

  // sleep 5 seconds to reach the hanging point
  sleep(5);

  std::vector<ProxyCollTraceEntry> activeSends;
  std::vector<ProxyCollTraceEntry> activeRecvs;
  std::vector<uint64_t> completedColls;
  size_t nActiveSends =
      comm->proxyState->trace->queryActiveSends(comm->commHash, activeSends);
  size_t nActiveRecvs =
      comm->proxyState->trace->queryActiveRecvs(comm->commHash, activeRecvs);

  size_t nCompletedColls = comm->proxyState->trace->queryCompletedColls(
      comm->commHash, completedColls);

  EXPECT_GT(nActiveSends, 0);
  EXPECT_GT(nActiveRecvs, 0);
  EXPECT_EQ(nCompletedColls, hangOpCount);

  for (auto& entry : activeSends) {
    // the hanging collective and potentially the next collective can be active
    EXPECT_TRUE(
        entry.opCount == hangOpCount || entry.opCount == hangOpCount + 1);
  }
  for (auto& entry : activeRecvs) {
    EXPECT_TRUE(
        entry.opCount == hangOpCount || entry.opCount == hangOpCount + 1);
  }

  // DO NOT call ncclCommDestroy() here, as it will trigger the hang
  // Manually set abortFlag to abort kernel so that cudaStreamDestroy can finish
  // TODO: we should call ncclCommAbort() once it guarantees termination
  *comm->abortFlag = 1;

  NCCL_PROXYTRACE.clear();
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.clear();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
