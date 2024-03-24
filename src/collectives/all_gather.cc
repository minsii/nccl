/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "nccl_cvars.h"
#include "Ctran.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_ALLGATHER_DIRECT_CUTOFF
   type        : uint64_t
   default     : 524288
   description : |-
     Message size up to which we use the direct algorithm for Allgather.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  int nRanks = comm->nRanks;
  size_t rankOffset = sendcount * ncclTypeSize(datatype);
  bool directSend = (comm->localRanks == 1) && (rankOffset <= NCCL_ALLGATHER_DIRECT_CUTOFF);

  // CTRAN allgather: only support inter-node now
  if (ctranInitialized(comm) && comm->localRanks == 1 && nRanks > 1) {
    if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctdirect) {
      return ctranAllGatherDirect(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctring) {
      return ctranAllGatherRing(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctrd) {
      return ctranAllGatherRd(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }
  }

  // FIXME: do not run send/recv based direct if ctran sendrecv is enabled.
  // It can cause cuda error and need to be fixed.
  //  && NCCL_SENDRECV_ALGO != NCCL_SENDRECV_ALGO::ctran
  if (directSend) {
    if (sendcount == 0) return ncclSuccess;

    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; r++) {
      NCCLCHECK(ncclSend(
         ((char*)sendbuff), sendcount, datatype, r, comm, stream));
      NCCLCHECK(ncclRecv(
         ((char*)recvbuff) + r * rankOffset, sendcount, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
  }

  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
