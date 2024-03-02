// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <CollTrace.h>
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

  void prepareAllToAll(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
  }

  void prepareSendRecv(const int count) {
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

TEST_F(CollTraceTest, TraceFeatureEnableCollTrace) {
  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("trace");
  testing::internal::CaptureStdout();
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;


  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  std::string output = testing::internal::GetCapturedStdout();
  //
  EXPECT_THAT(output, testing::HasSubstr("enabled features: trace - Init COMPLETE"));
  EXPECT_THAT(output, testing::Not(testing::HasSubstr("COLLTRACE initialization failed")));
  NCCL_COLLTRACE.clear();
}

TEST_F(CollTraceTest, VerboseAllReduce) {
  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("verbose");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "COLLTRACE: opCount " << std::hex << i << " AllReduce";
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
  NCCL_COLLTRACE.clear();
}

TEST_F(CollTraceTest, VerboseAllToAll) {
  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("verbose");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareAllToAll(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(comm));

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "COLLTRACE: opCount " << std::hex << i << " SendRecv";
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
  NCCL_COLLTRACE.clear();
}

TEST_F(CollTraceTest, VerboseSendRecv) {
  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("verbose");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareSendRecv(count);
  int peer = (this->globalRank + 1) % this->numRanks;
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, peer, comm, stream));
    NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, peer, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(comm));

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "COLLTRACE: opCount " << std::hex << i << " SendRecv";
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
  NCCL_COLLTRACE.clear();
}

TEST_F(CollTraceTest, VerboseSendOrRecv) {
  if (this->numRanks % 2) {
    GTEST_SKIP() << "This test requires even number of ranks";
  }

  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("verbose");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;

  testing::internal::CaptureStdout();

  prepareSendRecv(count);
  for (int i = 0; i < nColl; i++) {
    // even rank sends to odd rank (e.g, 0->1, 2->3)
    if (this->globalRank % 2 == 0) {
      int peer = this->globalRank + 1;
      NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, peer, comm, stream));
    } else {
      int peer = this->globalRank - 1;
      NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, peer, comm, stream));
    }
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  NCCLCHECK_TEST(ncclCommDestroy(comm));

  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    if (this->globalRank % 2 == 0) {
      ss << "COLLTRACE: opCount " << std::hex << i << " Send";
    } else {
      ss << "COLLTRACE: opCount " << std::hex << i << " Recv";
    }
    std::string traceLog = ss.str();
    EXPECT_THAT(output, testing::HasSubstr(traceLog));
  }
  NCCL_COLLTRACE.clear();
}

TEST_F(CollTraceTest, DumpAllFinished) {
  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("trace");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->collTrace != nullptr);
  comm->collTrace->waitForWorkerFinishQueue();
  auto dump = comm->collTrace->dumpTrace();
  EXPECT_EQ(dump.pastColls.size(), nColl);
  EXPECT_EQ(dump.currentColl, nullptr);

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  NCCL_COLLTRACE.clear();
}

TEST_F(CollTraceTest, DumpWithUnfinished) {
  // overwrite CollTrace features before creating comm
  NCCL_COLLTRACE.push_back("trace");
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->collTrace != nullptr);
  comm->collTrace->waitForWorkerFinishQueue();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  auto dump = comm->collTrace->dumpTrace();

  EXPECT_TRUE(dump.pastColls.size() >= nColl);
  // +1 for the extra wakeup event that might be created by dumpTrace() function
  EXPECT_TRUE(dump.pendingColls.size() <= nColl);

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  NCCL_COLLTRACE.clear();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
