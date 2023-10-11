// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_H_
#define CTRAN_IB_H_

#include "nccl.h"
#include <memory>

struct ctranIbRemoteAccessKey {
  uint32_t rkey;
};

class ctranIb;

class ctranIbRequest {
public:
  ctranIbRequest();
  ~ctranIbRequest();

  void complete();
  ncclResult_t test(bool *isComplete);

private:
  enum {
    INCOMPLETE,
    COMPLETE,
  } state;
};

class ctranIb {
public:
  ctranIb(ncclComm *comm);
  ~ctranIb();

  ncclResult_t regMem(const void *buf, std::size_t len, void **hdl);
  ncclResult_t deregMem(void *hdl);
  ncclResult_t progress(void);

  ncclResult_t isendCtrl(void *buf, void *hdl, int rank, ctranIbRequest **req);
  ncclResult_t irecvCtrl(void **buf, struct ctranIbRemoteAccessKey *key, int rank, ctranIbRequest **req);
  ncclResult_t iput(const void *sbuf, void *dbuf, std::size_t len, int rank, void *shdl,
      struct ctranIbRemoteAccessKey remoteAccessKey, bool notify, ctranIbRequest **req);
  ncclResult_t checkNotify(int rank, bool *notify);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
  friend class ctranIbRequest;
};

#endif
