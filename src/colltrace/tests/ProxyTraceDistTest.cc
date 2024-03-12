// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include "Ctran.h"
#include "ExtUtils.h"
#include "ProxyTrace.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

static bool VERBOSE = true;

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    setenv("NCCL_DEBUG", "WARN", 0);
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

  void runAllToAll(const int count, const int nColl, ncclComm_t comm) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * comm->nRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * comm->nRanks * sizeof(int)));
    for (int i = 0; i < nColl; i++) {
      NCCLCHECK_TEST(
          ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
    }
  }

  void runSendRecv(const int count, const int nColl, ncclComm_t comm) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
    for (int i = 0; i < nColl; i++) {
      // localRank on node0 sends to the same localRank on node1
      // TestCase ensures it runs with 2 nodes
      if (comm->node == 0) {
        int peer = comm->localRank + comm->localRanks;
        NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, peer, comm, stream));
      } else if (comm->node == 1) {
        int peer = comm->localRank;
        NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, peer, comm, stream));
      }
    }
  }

  bool skipSingleNodeRun(ncclComm_t comm) {
    if (comm->nNodes < 2) {
      NCCLCHECK_TEST(ncclCommDestroy(comm));
      NCCL_PROXYTRACE.clear();
      printf("Skipping test since nNodes < 2\n");
      return true;
    }
    return false;
  }

  void verbosePrintPastColl(ProxyTraceColl& past, ncclComm_t comm) {
    if (comm->rank == 0 && VERBOSE) {
      printf("Rank %d past coll: %s\n", comm->rank, past.serialize().c_str());
    }
  }

  // Common check for dumpping after finished collectives
  void checkCompletedDump(ProxyTrace::Dump& dump, int nCompletedColls) {
    EXPECT_EQ(dump.activeOps.size(), 0);
    EXPECT_EQ(dump.pastOps.size(), 0);
    EXPECT_EQ(dump.pastColls.size(), nCompletedColls);
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  cudaStream_t stream;
};

static void
checkPastColl(ProxyTraceColl& past, uint64_t opCount, ncclComm* comm) {
  EXPECT_EQ(past.collInfo.commHash, comm->commHash);
  EXPECT_EQ(past.collInfo.opCount, opCount);
  EXPECT_GT(past.collInfo.nChannels, 0);
}

TEST_F(ProxyTraceTest, QueryFinishedAllReduce) {
  // overwrite ProxyTrace features before creating comm
  NCCL_PROXYTRACE.push_back("trace");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  if (skipSingleNodeRun(comm)) {
    GTEST_SKIP();
  }

  EXPECT_NE(comm->proxyState->trace, nullptr);

  const int count = 1048500;
  const int nColl = 10;

  uint64_t opCountStart = comm->opCount;

  runAllReduce(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);

  // Check past collective details
  for (int i = 0; i < nColl; i++) {
    checkPastColl(dump.pastColls[i], opCountStart + i, comm);
    EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncAllReduce);
    // Skip check for nProxyOps as we don't know allreduce internal

    verbosePrintPastColl(dump.pastColls[i], comm);
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
  NCCL_PROXYTRACE.clear();
}

TEST_F(ProxyTraceTest, QueryFinishedAllToAll) {
  // overwrite ProxyTrace features before creating comm
  NCCL_PROXYTRACE.push_back("trace");
  // disable PXN so that each proxy thread can have deterministic behavior:
  // send and recv for the local rank with PPN remote ranks on the other node
  NCCL_PXN_DISABLE = 1;
  // ensure we use default proxy path
  NCCL_ALLTOALL_ALGO = NCCL_ALLTOALL_ALGO::orig;

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  if (skipSingleNodeRun(comm)) {
    GTEST_SKIP();
  }

  EXPECT_NE(comm->proxyState->trace, nullptr);

  // use size cannot be evenly divided by stepSize to test trace size
  // correctness
  const int count = 1048500;
  const int nColl = 10;
  uint64_t opCountStart = comm->opCount;

  runAllToAll(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);

  // Check past collective details
  for (int i = 0; i < nColl; i++) {
    checkPastColl(dump.pastColls[i], opCountStart + i, comm);
    EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncSendRecv);

    // Expect nChannels number of send and recv to each remote rank
    size_t nChannels = dump.pastColls[i].channelIds.size();
    int numRemoteRanks = comm->localRanks * (comm->nNodes - 1);
    EXPECT_EQ(dump.pastColls[i].nProxyOps, numRemoteRanks * 2 * nChannels);

    // Expect total send size to be count * sizeof(int) * numRemoteRanks
    EXPECT_EQ(
        dump.pastColls[i].totalSendSize, count * sizeof(int) * numRemoteRanks);
    // DO NOT check totalRecvSize which can be inaccurate (see ProxyTraceColl
    // description).

    verbosePrintPastColl(dump.pastColls[i], comm);
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
  NCCL_PROXYTRACE.clear();
}

TEST_F(ProxyTraceTest, QueryFinishedSendRecv) {
  // overwrite ProxyTrace features before creating comm
  NCCL_PROXYTRACE.push_back("trace");
  // disable PXN so that each proxy thread can have deterministic behavior:
  // send and recv for the local rank with PPN remote ranks on the other node
  NCCL_PXN_DISABLE = 1;
  // ensure we use default proxy path
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  if (skipSingleNodeRun(comm)) {
    GTEST_SKIP();
  }

  EXPECT_NE(comm->proxyState->trace, nullptr);

  // use size cannot be evenly divided by stepSize to test trace size
  // correctness
  const int count = 1048500;
  const int nColl = 2;
  uint64_t opCountStart = comm->opCount;

  runSendRecv(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);

  // Check past collective details
  EXPECT_EQ(dump.pastColls.size(), nColl);
  for (int i = 0; i < nColl; i++) {
    checkPastColl(dump.pastColls[i], opCountStart + i, comm);
    // localRank on node0 sends to the same localRank on node1 (see
    // runSendRecv). skipSingleNodeRun check ensures it runs with 2+nodes
    if (comm->node == 0) {
      EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncSend);
      EXPECT_EQ(dump.pastColls[i].totalSendSize, count * sizeof(int));
      EXPECT_EQ(dump.pastColls[i].totalRecvSize, 0);
    } else if (comm->node == 1) {
      EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncRecv);
      // DO NOT check totalRecvSize which can be inaccurate (see ProxyTraceColl
      // description).
      EXPECT_EQ(dump.pastColls[i].totalSendSize, 0);
    }
    // 1 send@sender and 1 recv@receiver are expected.
    // Each send/recv may be divided into nChannels.
    size_t nChannels = dump.pastColls[i].channelIds.size();
    EXPECT_EQ(dump.pastColls[i].nProxyOps, nChannels);

    verbosePrintPastColl(dump.pastColls[i], comm);
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
  NCCL_PROXYTRACE.clear();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
