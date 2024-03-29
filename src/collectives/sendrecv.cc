/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "data_export.h"
#include "enqueue.h"
#include "collectives.h"
#include "argcheck.h" // Need some checks here since we access comm
#include "Ctran.h"
#include "nccl_cvars.h"

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  if (NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctran &&
      ctranSendRecvSupport(peer, comm)) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = ctranSend(sendbuff, count, datatype, peer, comm, stream);
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }

  ctranGroupTrackDefaultOp(comm);

  NCCLCHECK(ncclDataExport(
      sendbuff, count * ncclTypeSize(datatype), stream, comm, peer, datatype));

  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  if (NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctran &&
      ctranSendRecvSupport(peer, comm)) {
    // ctran send/recvs are enqueued within ctran wherease other non-ctran ones
    // are enqueued in the original queue. When reaching group end, these two
    // groups of ops will be issued separately.
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    ret = ctranRecv(recvbuff, count, datatype, peer, comm, stream);
    NCCLCHECK(ncclGroupEnd());
    return ret;
  }

  ctranGroupTrackDefaultOp(comm);

  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}
