// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
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

class SendRecvTest : public ::testing::Test {
 public:
  SendRecvTest() = default;
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

  void prepareBufs(const int count, bool registFlag = false) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));

    for (int r = 0; r < this->numRanks; r++) {
      int expectedVal = this->globalRank * 100 + r + 1;
      assignChunkValue(sendBuf + r * count, count, expectedVal);
      assignChunkValue(recvBuf + r * count, count, -1);
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBuf, count * this->numRanks * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBuf, count * this->numRanks * sizeof(int), &recvHandle));
    }

    // prepare sendrecv arguments
    sendCounts.resize(this->numRanks);
    sendDispls.resize(this->numRanks);
    recvCounts.resize(this->numRanks);
    recvDispls.resize(this->numRanks);
    for (int r = 0; r < this->numRanks; r++) {
      sendDispls[r] = r * count;
      recvDispls[r] = r * count;
      sendCounts[r] = count;
      recvCounts[r] = count;
    }
  }

  void checkResults(const int sendRank, const int recvRank, const int count) {
    int expectedVal = sendRank * 100 + recvRank + 1;
    int errs =
        checkChunkValue(recvBuf + recvDispls[sendRank], count, expectedVal);
    EXPECT_EQ(errs, 0) << "rank " << this->globalRank
                       << " checked result from rank " << sendRank
                       << " recvRank " << recvRank << " at "
                       << recvBuf + recvDispls[recvRank] << " with " << errs
                       << " errors";
  }

  void releaseBufs(bool registFlag = false) {
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;

  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};

  std::vector<size_t> sendCounts;
  std::vector<size_t> sendDispls;
  std::vector<size_t> recvCounts;
  std::vector<size_t> recvDispls;
};

TEST_F(SendRecvTest, DISABLED_Default) {
  // create and register buffers
  constexpr int count = 1048576, commCount = 1024;
  int sendRank, recvRank;
  prepareBufs(count, true);

  if (comm->rank % 2) {
    sendRank = comm->rank;
    recvRank = (comm->rank + 1) % comm->nRanks;
  } else {
    sendRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;
    recvRank = comm->rank;
  }

  printf("rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);

  if (comm->rank == sendRank) {
    NCCLCHECK_TEST(
        ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
  } else if (comm->rank == recvRank) {
    NCCLCHECK_TEST(
        ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  if (comm->rank == recvRank) {
    checkResults(sendRank, recvRank, commCount);
  }

  releaseBufs(true);
}

TEST_F(SendRecvTest, DISABLED_Ctran) {
  setenv("NCCL_SENDRECV_ALGO", "ctran", 1);
  ncclCvarInit();

  // create and register buffers
  constexpr int count = 1048576, commCount = 1024;
  int sendRank, recvRank;
  prepareBufs(count, true);

  if (comm->rank % 2) {
    sendRank = comm->rank;
    recvRank = (comm->rank + 1) % comm->nRanks;
  } else {
    sendRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;
    recvRank = comm->rank;
  }

  printf("rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);

  for (int x = 0; x < 5; x++) {
    if (comm->rank == sendRank) {
      NCCLCHECK_TEST(
          ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
    } else if (comm->rank == recvRank) {
      NCCLCHECK_TEST(
          ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
    }
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  if (comm->rank == recvRank) {
    checkResults(sendRank, recvRank, commCount);
  }

  releaseBufs(true);

  unsetenv("NCCL_SENDRECV_ALGO");
}

TEST_F(SendRecvTest, DISABLED_CtranGrouped) {
  setenv("NCCL_SENDRECV_ALGO", "ctran", 1);
  ncclCvarInit();

  // create and register buffers
  constexpr int count = 1048576, commCount = 1024;
  int sendRank, recvRank;
  prepareBufs(count, true);

  if (comm->rank % 2) {
    sendRank = comm->rank;
    recvRank = (comm->rank + 1) % comm->nRanks;
  } else {
    sendRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;
    recvRank = comm->rank;
  }

  printf("rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);
  const int sendCount = 1;

  for (int x = 0; x < 5; x++) {
    if (comm->rank == sendRank) {
      ncclGroupStart();
      NCCLCHECK_TEST(
          ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
      ncclGroupEnd();
    } else if (comm->rank == recvRank) {
      ncclGroupStart();
      NCCLCHECK_TEST(
          ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
      ncclGroupEnd();
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    if (comm->rank == recvRank) {
      checkResults(sendRank, recvRank, commCount);
    }
  }

  releaseBufs(true);

  unsetenv("NCCL_SENDRECV_ALGO");
}

TEST_F(SendRecvTest, CtranGroupedMixed) {
  setenv("NCCL_SENDRECV_ALGO", "ctran", 1);
  ncclCvarInit();

  // create and register buffers with aligned bytes but communicate
  // with random size
  constexpr int count = 1048576, commCount = 105;
  int sendRank = -1, recvRank = -1;

  if (comm->nNodes < 2) {
    GTEST_SKIP() << "This test requires at least 2 nodes";
  }

  // local rank x on node0 sends to local rank x on node1
  if (comm->node == 0) {
    sendRank = comm->rank;
    recvRank = comm->rank + comm->localRanks;
  } else if (comm->node == 1) {
    sendRank = comm->rank - comm->localRanks;
    recvRank = comm->rank;
  }

  // only send/recv ranks continue with the test
  if (comm->rank != sendRank && comm->rank != recvRank) {
    return;
  }

  prepareBufs(count, true);
  printf("rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);

  // send/recv to itself first, then send/recv to a remote rank, mimicing
  // xlformer pattern. Expect self send/recv handled by baseline and remote
  // send/recv handled by ctran.
  ncclGroupStart();

  if (comm->rank == sendRank) {
    NCCLCHECK_TEST(
        ncclSend(sendBuf, 1, ncclBfloat16, comm->rank, comm, stream));
    NCCLCHECK_TEST(
        ncclRecv(recvBuf, 1, ncclBfloat16, comm->rank, comm, stream));

    NCCLCHECK_TEST(
        ncclSend(sendBuf, 1, ncclBfloat16, recvRank, comm, stream));
    NCCLCHECK_TEST(
        ncclRecv((char *)recvBuf + 2, 1, ncclBfloat16, recvRank, comm, stream));
  } else if (comm->rank == recvRank) {
    NCCLCHECK_TEST(
        ncclSend(sendBuf, 1, ncclBfloat16, sendRank, comm, stream));
    NCCLCHECK_TEST(
        ncclRecv(recvBuf, 1, ncclBfloat16, sendRank, comm, stream));

    NCCLCHECK_TEST(
        ncclSend(sendBuf, 1, ncclBfloat16, comm->rank, comm, stream));
    NCCLCHECK_TEST(
        ncclRecv((char *)recvBuf + 2, 1, ncclBfloat16, comm->rank, comm, stream));
  }
  ncclGroupEnd();

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // checkResults(comm->rank, comm->rank, commCount);

  // if (comm->rank == recvRank) {
  //   checkResults(sendRank, recvRank, commCount);
  // }

  releaseBufs(true);

  unsetenv("NCCL_SENDRECV_ALGO");
}

// TEST_F(SendRecvTest, InvalidSendbuf) {
// #ifdef NCCL_ALLTOALLV_SUPPORTED

//   constexpr int count = 1048576;
//   int* buf = nullptr;
//   CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

//   // prepare alltoallv arguments
//   std::vector<size_t> sendCounts(this->numRanks, count);
//   std::vector<size_t> sendDispls(this->numRanks, 0);
//   std::vector<size_t> recvCounts(this->numRanks, count);
//   std::vector<size_t> recvDispls(this->numRanks, 0);

//   // run alltoallv
//   auto res = ncclAllToAllv(
//       nullptr,
//       sendCounts.data(),
//       sendDispls.data(),
//       buf,
//       recvCounts.data(),
//       recvDispls.data(),
//       ncclInt,
//       comm,
//       stream);
//   ASSERT_EQ(res, ncclInvalidArgument);
//   CUDACHECK_TEST(cudaFree(buf));
// #endif
// }

// TEST_F(SendRecvTest, InvalidRecvbuf) {
// #ifdef NCCL_ALLTOALLV_SUPPORTED
//   constexpr int count = 1048576;
//   int* buf = nullptr;
//   CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

//   // prepare alltoallv arguments
//   std::vector<size_t> sendCounts(this->numRanks, count);
//   std::vector<size_t> sendDispls(this->numRanks, 0);
//   std::vector<size_t> recvCounts(this->numRanks, count);
//   std::vector<size_t> recvDispls(this->numRanks, 0);

//   // run alltoallv
//   auto res = ncclAllToAllv(
//       buf,
//       sendCounts.data(),
//       sendDispls.data(),
//       nullptr,
//       recvCounts.data(),
//       recvDispls.data(),
//       ncclInt,
//       comm,
//       stream);
//   ASSERT_EQ(res, ncclInvalidArgument);
//   CUDACHECK_TEST(cudaFree(buf));
// #endif
// }

// TEST_F(SendRecvTest, InvalidInPlace) {
// #ifdef NCCL_ALLTOALLV_SUPPORTED
//   constexpr int count = 1048576;
//   int* buf = nullptr;
//   CUDACHECK_TEST(cudaMalloc(&buf, count * this->numRanks * sizeof(int)));

//   // prepare alltoallv arguments
//   std::vector<size_t> sendCounts(this->numRanks, count);
//   std::vector<size_t> sendDispls(this->numRanks, 0);
//   std::vector<size_t> recvCounts(this->numRanks, count);
//   std::vector<size_t> recvDispls(this->numRanks, 0);

//   // run alltoallv
//   auto res = ncclAllToAllv(
//       buf,
//       sendCounts.data(),
//       sendDispls.data(),
//       buf,
//       recvCounts.data(),
//       recvDispls.data(),
//       ncclInt,
//       comm,
//       stream);
//   ASSERT_EQ(res, ncclInvalidArgument);
//   CUDACHECK_TEST(cudaFree(buf));
// #endif
// }

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
