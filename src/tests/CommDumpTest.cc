// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <unordered_map>
#include "comm.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    setenv("NCCL_DEBUG", "WARN", 0);
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_PROXYTRACE", "trace", 0);
  }
  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class CommDumpTest : public ::testing::Test {
 public:
  CommDumpTest() = default;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // Prepare data for sanity check after commSplit
    CUDACHECK_TEST(cudaMalloc(&this->dataBuf, sizeof(int) * this->dataCount));
  }

  void initData(int myRank) {
    std::vector<int> initVals(this->dataCount);
    for (int i = 0; i < this->dataCount; i++) {
      initVals[i] = i * myRank;
    }
    CUDACHECK_TEST(cudaMemcpy(
        this->dataBuf,
        initVals.data(),
        sizeof(int) * this->dataCount,
        cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(this->dataBuf));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* dataBuf{nullptr};
  const int dataCount{65536};
  cudaStream_t stream;
};

TEST_F(CommDumpTest, SingleComm) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  res = ncclCommDump(comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], std::to_string(comm->commHash));

  if (this->globalRank == 0) {
    printf("Dump on rank 0:\n");
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CommDumpTest, DumpAfterColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        comm,
        this->stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  // NOTE: the last 8 steps of the last collective may still ongoing on proxy
  // thread
  res = ncclCommDump(comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], std::to_string(comm->commHash));

  if (comm->nNodes > 1) {
    // PROXYTRACE is enabled only when nNodes > 1
    EXPECT_EQ(dump.count("PT_nCompletedColls"), 1);
    EXPECT_EQ(dump.count("PT_nActiveSends"), 1);
    EXPECT_EQ(dump.count("PT_nActiveRecvs"), 1);
    EXPECT_EQ(dump.count("PT_activeSends"), 1);
    EXPECT_EQ(dump.count("PT_activeRecvs"), 1);
    EXPECT_EQ(dump.count("PT_completedColls"), 1);
  }

  if (this->globalRank == 0) {
    printf("Dump on rank 0:\n");
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CommDumpTest, DumpDuringHangColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  // overwrite ProxyTrace mock before creating comm
  // FIXME: we cannot for sure which rank is sending network send/recv to the
  // other. It depends on ring and host topology. For now manually adjust it
  // (one may check the network send/recv details by enabling "verbose"). Need
  // find way to ensure mock works.
  constexpr int hangChannelId = 1;
  constexpr int hangRank = 1;
  constexpr int hangRemoteRank = 2;
  constexpr int hangOpCount = 5;
  constexpr int hangOpStep = 1;
  // NCCL_PROXYTRACE.push_back("verbose");
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(std::to_string(hangOpCount));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(
      std::to_string(hangChannelId));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(std::to_string(hangRank));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(
      std::to_string(hangRemoteRank));
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.push_back(std::to_string(hangOpStep));
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  if (comm->nRanks < 2) {
    GTEST_SKIP() << "This test requires at least 2 ranks";
  }

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        comm,
        this->stream));
  }

  // let's wait proxy thread to hang
  sleep(5);

  // Dump before waiting completion
  // TODO: add backdoor to control which collective to hang; without it, we may
  // dump arbitrary collective
  res = ncclCommDump(comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], std::to_string(comm->commHash));

  if (comm->nNodes > 1) {
    // PROXYTRACE is enabled only when nNodes > 1
    EXPECT_EQ(dump.count("PT_nCompletedColls"), 1);
    EXPECT_EQ(dump.count("PT_nActiveSends"), 1);
    EXPECT_EQ(dump.count("PT_nActiveRecvs"), 1);
    EXPECT_EQ(dump.count("PT_activeSends"), 1);
    EXPECT_EQ(dump.count("PT_activeRecvs"), 1);
    EXPECT_EQ(dump.count("PT_completedColls"), 1);
  }

  printf("Dump on rank %d:\n", this->globalRank);
  for (auto& it : dump) {
    printf(
        "rank %d %s: %s\n",
        this->globalRank,
        it.first.c_str(),
        it.second.c_str());
  }

  // DO NOT call ncclCommDestroy() here, as it will trigger the hang
  // Manually set abortFlag to abort kernel so that cudaStreamDestroy can finish
  // at TearDown.
  // TODO: we should call ncclCommAbort() once it guarantees termination
  *comm->abortFlag = 1;
  NCCL_PROXYTRACE_NET_SEND_FAILURE_MOCK.clear();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
