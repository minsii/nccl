// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "Ctran.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn off NCCL debug logging, allow user to turn on via command line
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class AllGatherTest : public ::testing::Test {
 public:
  AllGatherTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              this->globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

  void run(const int bufCount, const int count, bool registFlag = false) {
    // create and register buffers
    int *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHandle = nullptr, *recvHandle = nullptr;

    CUDACHECK_TEST(cudaMalloc(&sendBuf, bufCount * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, bufCount * this->numRanks * sizeof(int)));

    int expectedVal = this->globalRank * 100 + 1;
    assignChunkValue(sendBuf, count, expectedVal);
    assignChunkValue(recvBuf, count * this->numRanks, -1);

#ifdef NCCL_REGISTRATION_SUPPORTED
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBuf, bufCount * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBuf, bufCount * this->numRanks * sizeof(int), &recvHandle));
    }
#endif

    for (int i = 0; i < 5; i++) {
      auto res = ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream);
      ASSERT_EQ(res, ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    for (int r = 0; r < this->numRanks; r++) {
      int expectedVal = r * 100 + 1;
      int errs = checkChunkValue(recvBuf + r * count, count, expectedVal);
      EXPECT_EQ(errs, 0) << "rank " << this->globalRank << " checked chunk "
                         << r << " at " << recvBuf + r * count << " with "
                         << errs << " errors";
    }

#ifdef NCCL_REGISTRATION_SUPPORTED
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }
#endif

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
};

// TEST_F(AllGatherTest, OutOfPlace) {
//   run();
// }

TEST_F(AllGatherTest, CTRing) {
  setenv("NCCL_ALLGATHER_ALGO", "ctring", 1);
  ncclCvarInit();
  run(1048576, 1048576, true);
  unsetenv("NCCL_ALLGATHER_ALGO");
}

TEST_F(AllGatherTest, CtranP2P) {
  setenv("NCCL_SENDRECV_ALGO", "ctran", 1);
  ncclCvarInit();
  run(1048576, 1, true);
  unsetenv("NCCL_SENDRECV_ALGO");
}

// TEST_F(AllGatherTest, InvalidSendbuf) {
//   constexpr int count = 1048576;
//   int* buf = nullptr;
//   CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

//   // run alltoall
//   auto res = ncclAllGather(nullptr, buf, count, ncclInt, comm, stream);
//   ASSERT_EQ(res, ncclInvalidArgument);
//   CUDACHECK_TEST(cudaFree(buf));
// }

// TEST_F(AllGatherTest, InvalidRecvbuf) {
//   constexpr int count = 1048576;
//   int* buf = nullptr;
//   CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

//   // run alltoall
//   auto res = ncclAllGather(buf, nullptr, count, ncclInt, comm, stream);
//   ASSERT_EQ(res, ncclInvalidArgument);
//   CUDACHECK_TEST(cudaFree(buf));
// }

// TEST_F(AllGatherTest, InvalidInPlace) {
//   constexpr int count = 1048576;
//   int* buf = nullptr;
//   CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

//   // run alltoall
//   auto res = ncclAllGather(buf, buf, count, ncclInt, comm, stream);
//   ASSERT_EQ(res, ncclInvalidArgument);
//   CUDACHECK_TEST(cudaFree(buf));
// }

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
