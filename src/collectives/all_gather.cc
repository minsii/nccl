/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "nccl_cvars.h"
#include "ctranAlgos.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_ALLGATHER_DIRECT_CUTOFF
   type        : int
   default     : 0
   description : |-
     Message size up to which we use the direct algorithm for Allgather.

 - name        : NCCL_ALLGATHER_ALGO
   type        : enum
   default     : orig
   choices     : orig, ctdirect, ctring, ctrd
   description : |-
     The algorithm to use for Allgather communication
     orig - Copy-based ring algorithm
     ctdirect - Ctran-based direct point-to-point algorithm
     ctring - Ctran-based ring algorithm
     ctrd - Ctran-based recursive-doubling algorithm

 - name        : NCCL_CTRAN_ENABLE_LOCAL_IB
   type        : bool
   default     : false
   description : |-
     Disable using ctran/IB if there are multiple processes on the same node
     that need to communicate with each other.  This is a temporary variable
     that needs to be eventually deleted for two reasons.  First, we will
     eventually add an NVLink backend, which can function within the same node.
     Second, the disabling of loopback on our current RTSWs is a temporary
     workaround for an ongoing SEV and should be reenabled within a few weeks
     (few weeks from 11/6/2023).

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  int nRanks = comm->nRanks;
  size_t rankOffset = sendcount * ncclTypeSize(datatype);
  bool directSend = (comm->localRanks == 1) && (rankOffset <= NCCL_ALLGATHER_DIRECT_CUTOFF);
  bool enableCtran = (NCCL_CTRAN_ENABLE_LOCAL_IB || comm->localRanks == 1) ? true : false;

  // only use CTRAN for inter-node only allgather
  if (comm->ctranMapper != nullptr && enableCtran && nRanks > 1 && rankOffset > getpagesize()) {
    if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctdirect) {
      LOG_COLL_INFO("ctranAllGatherDirect", sendbuff, recvbuff, sendcount, datatype, comm, stream);
      return ctranAllGatherDirect(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctring) {
      LOG_COLL_INFO("ctranAllGatherRing", sendbuff, recvbuff, sendcount, datatype, comm, stream);
      return ctranAllGatherRing(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctrd) {
      LOG_COLL_INFO("ctranAllGatherRd", sendbuff, recvbuff, sendcount, datatype, comm, stream);
      return ctranAllGatherRd(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }
  }

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
