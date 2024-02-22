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
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
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
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* dataBuf{nullptr};
  const int dataCount{65536};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(CommDumpTest, SingleComm) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;

  res = ncclCommDump(this->comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], std::to_string(this->comm->commHash));
}

TEST_F(CommDumpTest, DumpAfterColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        this->comm,
        this->stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  // NOTE: the last 8 steps of the last collective may still ongoing on proxy thread
  res = ncclCommDump(this->comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], std::to_string(this->comm->commHash));
}

TEST_F(CommDumpTest, DumpDuringColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        this->comm,
        this->stream));
  }

  // Dump before waiting completion
  // TODO: add backdoor to control which collective to hang; without it, we may
  // dump arbitrary collective
  res = ncclCommDump(this->comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], std::to_string(this->comm->commHash));

  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
