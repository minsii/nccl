// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <sstream>
#include "Ctran.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn on NCCL debug logging for verbose testing
    // Allow user to change via command line
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

class CollTraceTest : public ::testing::Test {
 public:
  CollTraceTest() = default;
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

  void prepareAllreduce(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  cudaStream_t stream;
};

TEST_F(CollTraceTest, VerboseAllReduce) {
  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("verbose");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;

  // testing::internal::CaptureStdout();

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  // std::string output = testing::internal::GetCapturedStdout();
  // for (int i = 0; i < nColl; i++) {
  //   std::stringstream ss;
  //   ss << "COLLTRACE: opCount " << std::hex << i << " AllReduce";
  //   std::string traceLog = ss.str();
  //   EXPECT_THAT(output, testing::HasSubstr(traceLog));
  // }
  NCCL_COLLTRACE.clear();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
