// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "nccl.h"
#include "argcheck.h"
#include "comm.h"

uint64_t ncclParamCtranRegister();

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  ncclResult_t res = ncclSuccess;

  if (ncclParamCtranRegister()) {
    NCCLCHECKGOTO(PtrCheck(comm, "ncclCommRegister", "comm"), res, exit);
    NCCLCHECKGOTO(comm->ctranMapper->regMem(buff, size, handle), res, exit);
  } else {
    *handle = nullptr;
  }

exit:
  return res;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  ncclResult_t res = ncclSuccess;

  if (ncclParamCtranRegister()) {
    NCCLCHECKGOTO(PtrCheck(comm, "ncclCommRegister", "comm"), res, exit);
    NCCLCHECKGOTO(comm->ctranMapper->deregMem(handle), res, exit);
  }

exit:
  return res;
}
