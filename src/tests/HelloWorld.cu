#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include "tests_common.cuh"

int main(int argc, char* argv[])
{
  int size = 32*1024;

  int localRank, globalRank, numRanks;

  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  std::tie(localRank, globalRank, numRanks, comm) = setupNccl(argc, argv);

  CUDACHECK_TEST(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK_TEST(cudaStreamCreate(&s));

  for (int x = 0; x < 32; x++) {
    //communicating using NCCL
    NCCLCHECK_TEST(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
          comm, s));
  }

  for (int x = 0; x < 32; x++) {
    NCCLCHECK_TEST(ncclAllToAll(sendbuff, recvbuff, size/2, ncclFloat, comm, s));
  }

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK_TEST(cudaStreamSynchronize(s));

  //free device buffers
  CUDACHECK_TEST(cudaFree(sendbuff));
  CUDACHECK_TEST(cudaFree(recvbuff));

  cleanupNccl(comm);

  printf("[MPI Rank %d] Success \n", globalRank);
  return 0;
}
