// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <assert.h>
#include <cmath>
#include "argcheck.h"
#include "enqueue.h"
#include "nccl.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_DDA_ALLREDUCE_MAX_BLOCKS
   type        : int
   default     : 1
   description : |-
     Number of CUDA blocks to use for DDA Allreduce.

 - name        : NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS
   type        : int
   default     : 262144
   description : |-
     Message size at which DDA Allreduce switches to the tree algorithm.
     Only applies for NVSwitch-based systems.

 - name        : NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM
   type        : int
   default     : 65536
   description : |-
     Message size at which DDA Allreduce switches to the tree algorithm.
     Only applies for HCM-based systems.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

#define ASSIGN_FUNC(func, templ, nranks)   \
  do {                                     \
    switch ((nranks)) {                    \
      case 2:                              \
        func = (const void*)&templ<T, 2>;  \
        break;                             \
                                           \
      case 4:                              \
        func = (const void*)&templ<T, 4>;  \
        break;                             \
                                           \
      case 8:                              \
        func = (const void*)&templ<T, 8>;  \
        break;                             \
                                           \
      case 16:                             \
        func = (const void*)&templ<T, 16>; \
        break;                             \
                                           \
      default:                             \
        return ncclInvalidUsage;           \
    }                                      \
  } while (0)

static inline int getMaxBlocks(ncclComm* comm) {
  int maxBlocks = NCCL_DDA_ALLREDUCE_MAX_BLOCKS;

  if (maxBlocks > comm->dda->devProp.multiProcessorCount) {
    WARN("NCCL_DDA_ALLREDUCE_MAX_BLOCKS cannot be larger than %d\n",
         comm->dda->devProp.multiProcessorCount);
    maxBlocks = comm->dda->devProp.multiProcessorCount;
  }

  return maxBlocks;
}

template <typename T>
static ncclResult_t launchKernel(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    cudaStream_t stream) {
  ncclDDAAllReduceAlgo_t algo = getAllReduceAlgo(sendbuff, recvbuff, count, datatype, ncclSum, comm);
  const void* func;

  if (comm->dda->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
    if (count * sizeof(T) < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS) {
      if (algo == NCCL_DDA_ALLREDUCE_ALGO_DDA_IPC) {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Flat_ipc, comm->nRanks);
      } else {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Flat, comm->nRanks);
      }
    } else {
      if (algo == NCCL_DDA_ALLREDUCE_ALGO_DDA_IPC) {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Tree_ipc, comm->nRanks);
      } else {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Tree, comm->nRanks);
      }
    }
  } else if (comm->dda->topoType == NCCL_DDA_TOPO_TYPE__HCM) {
    if (count * sizeof(T) < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM) {
      if (algo == NCCL_DDA_ALLREDUCE_ALGO_DDA_IPC) {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_HCM_Flat_ipc, comm->nRanks);
      } else {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_HCM_Flat, comm->nRanks);
      }
    } else {
      ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_HCM_Tree, comm->nRanks);
    }
  } else {
    return ncclInvalidUsage;
  }

  cudaFuncAttributes attr;
  CUDACHECK(cudaFuncGetAttributes(&attr, func));

  int maxBlocks = getMaxBlocks(comm);
  int numBlocks[2] = {1, maxBlocks};
  int numThreads[2] = {32, attr.maxThreadsPerBlock};
  int eltsPerThread = 16 / sizeof(T);
  dim3 grid;
  dim3 blocks;

  if (count <= numBlocks[0] * numThreads[0] * eltsPerThread) {
    /* for small counts, use the minimum number of blocks and
     * threads, while keeping eltsPerThread elements to be
     * computed by each thread. */
    grid.x = numBlocks[0];
    blocks.x = numThreads[0];
  } else if (count <= numBlocks[0] * numThreads[1] * eltsPerThread) {
    /* for slightly larger counts, increase the number of threads
     * per block to up to the maximum number of threads. */
    grid.x = numBlocks[0];
    blocks.x = (int)std::ceil(count / (numBlocks[0] * eltsPerThread));
  } else if (count <= numBlocks[1] * numThreads[1] * eltsPerThread) {
    /* for even larger counts, increase the number of blocks to up
     * to the maximum number of blocks. */
    grid.x = (int)std::ceil(count / (numThreads[1] * eltsPerThread));
    blocks.x = numThreads[1];
  } else {
    /* for even larger counts, use the maximum number of threads
     * and blocks, and let each thread compute more elements. */
    grid.x = numBlocks[1];
    blocks.x = numThreads[1];
  }
  grid.y = 1;
  grid.z = 1;
  blocks.y = 1;
  blocks.z = 1;

  /* mbox_id,barrierFlag: 1,0; 0,1; 1,1; 0,0 */
  /* We maintain two mboxes (0,1) and two flags (0,1) for each mbox.
   * We start with an mboxId of 1, and a barrierFlag of 0.  And each
   * mbox is set to all zeroes.
   *
   * At this point, the last bit in each element of the mbox is 0.
   *
   * The first time we call this collective, we switch the mboxId to
   * 0, and the barrierFlag to 1.  This way, the barrier operation
   * in our kernel has to wait for the barrierFlag to move to 1
   * before it can exit the barrier.
   *
   * The second time we call this collective, we switch the mboxId
   * to 1, and the barrierFlag to 1.  This way, the barrier
   * operation in our kernel has to wait for the barrierFlag to move
   * to 1 before it can exit the barrier.
   *
   * At this point, the last bit in each element of the mbox is 1.
   *
   * The third time we call this collective, we switch the mboxId to
   * 0, and the barrierFlag to 0.  This way, the barrier operation
   * in our kernel has to wait for the barrierFlag to move to 0
   * before it can exit the barrier.
   *
   * The fourth time we call this collective, we switch the mboxId
   * to 1, and the barrierFlag to 0.  This way, the barrier
   * operation in our kernel has to wait for the barrierFlag to move
   * to 0 before it can exit the barrier.
   *
   * At this point, the last bit in each element of the mbox is 0,
   * and we are back to our original state.
   *
   * We need two mboxes, so one rank can get started with the next
   * collective even if not all ranks have exited the previous
   * collective (and thus are still using the previous mbox).
   */
  if (comm->dda->barrierMboxId == 1 && comm->dda->barrierFlag == 0) {
    comm->dda->barrierMboxId = !comm->dda->barrierMboxId;
    comm->dda->barrierFlag = !comm->dda->barrierFlag;
  } else if (comm->dda->barrierMboxId == 0 && comm->dda->barrierFlag == 1) {
    comm->dda->barrierMboxId = !comm->dda->barrierMboxId;
  } else if (comm->dda->barrierMboxId == 1 && comm->dda->barrierFlag == 1) {
    comm->dda->barrierMboxId = !comm->dda->barrierMboxId;
    comm->dda->barrierFlag = !comm->dda->barrierFlag;
  } else if (comm->dda->barrierMboxId == 0 && comm->dda->barrierFlag == 0) {
    comm->dda->barrierMboxId = !comm->dda->barrierMboxId;
  }

  void* args[] = {
    &comm->dda->barrierFlag,
    &comm->dda->barrierMboxId,
    &comm->dda->commMdDev,
    &comm->rank,
    &sendbuff,
    &recvbuff,
    &count
  };

  if (algo == NCCL_DDA_ALLREDUCE_ALGO_DDA_IPC) {
    CUDACHECK(cudaMemcpyAsync(
        comm->dda->commMdHost[comm->rank].tmpbuff,
        sendbuff,
        count * sizeof(T),
        cudaMemcpyDefault,
        stream));
  }
  CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));

  return ncclSuccess;
}

ncclResult_t ncclAllReduceDDA(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  NCCLCHECK(ncclCommEnsureReady(comm));
  if (datatype < 0 || datatype >= ncclNumTypes) {
    WARN("AllReduce : invalid type %d", datatype);
    return ncclInvalidArgument;
  }
  if (op < 0 || ncclMaxRedOp < op) {
    WARN("AllReduce : invalid reduction operation %d", op);
    return ncclInvalidArgument;
  }
  NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "AllReduce"));
  NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "AllReduce"));

  switch (datatype) {
    case ncclInt8:
      NCCLCHECK(launchKernel<char>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclUint8:
      NCCLCHECK(launchKernel<uint8_t>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclInt32:
      NCCLCHECK(launchKernel<int32_t>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclUint32:
      NCCLCHECK(
          launchKernel<uint32_t>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclInt64:
      NCCLCHECK(launchKernel<int64_t>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclUint64:
      NCCLCHECK(
          launchKernel<uint64_t>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclFloat16:
      NCCLCHECK(launchKernel<half>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclFloat32:
      NCCLCHECK(launchKernel<float>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

    case ncclFloat64:
      NCCLCHECK(launchKernel<double>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;

#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      NCCLCHECK(launchKernel<__nv_bfloat16>(comm, sendbuff, recvbuff, count, datatype, stream));
      break;
#endif

    default:
      goto not_supported;
  }

  return ncclSuccess;

not_supported:
  return ncclInvalidUsage;
}
