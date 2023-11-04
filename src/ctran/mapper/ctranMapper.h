// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_H_
#define CTRAN_MAPPER_H_

#include <mutex>
#include <memory>
#include <functional>
#include <vector>
#include <chrono>
#include "nccl.h"
#include "checks.h"
#include "ctranIb.h"
#include "ctranNvl.h"

struct ctranMapperRemoteAccessKey {
  struct ctranIbRemoteAccessKey ibKey;
};

class ctranMapperMemPool {
 public:
  ctranMapperMemPool();
  ~ctranMapperMemPool();

  void printSnapshot();
  ncclResult_t init();
  ncclResult_t getBuf(std::size_t len, void** addr, void** hdl, std::size_t *bufLen);
  ncclResult_t getFreeBlk(std::size_t len, void** addr, void** hdl);
  ncclResult_t release(void* addr, void* hdl);
  ncclResult_t releaseAll();
  ncclResult_t alloc(void** addr, std::size_t len);
  ncclResult_t free(void* addr);
  ncclResult_t regMem(
      std::function<ncclResult_t(const void*, std::size_t, void**)> regMemFunc);
  ncclResult_t deregMem(std::function<ncclResult_t(void*)> deRegMemFunc);

 private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

class ctranMapper;
class ctranMapperRequest {
public:
  ctranMapperRequest(ctranMapper *mapper);
  ~ctranMapperRequest();
  ncclResult_t test(bool *isComplete);
  ncclResult_t wait();

  ctranIbRequest *ibReq;
  ctranNvlRequest *nvlReq;

private:
  ctranMapper *mapper;
  enum {
    INCOMPLETE,
    COMPLETE,
  } state;
};

struct ncclComm;

class ctranMapperTimestampPoint {
  public:
    ctranMapperTimestampPoint(int peer) {
      this->now = std::chrono::high_resolution_clock::now();
      this->peer = peer;
    }
    ~ctranMapperTimestampPoint() = default;

    std::chrono::time_point<std::chrono::high_resolution_clock> now;
    int peer;
};

class ctranMapperTimestamp {
  public:
    ctranMapperTimestamp(const std::string algo) {
      this->algo = algo;
      this->start = std::chrono::high_resolution_clock::now();
    }
    ~ctranMapperTimestamp() = default;

    std::vector<ctranMapperTimestampPoint> recvCtrl;
    std::vector<ctranMapperTimestampPoint> putIssued;
    std::vector<ctranMapperTimestampPoint> putComplete;
    std::string algo;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

class ctranMapperTimer {
  public:
    ctranMapperTimer() {
      this->start_ = std::chrono::high_resolution_clock::now();
    }
    ~ctranMapperTimer() = default;
    double durationUs(){
      auto end = std::chrono::high_resolution_clock::now();
      return std::chrono::duration_cast<std::chrono::microseconds>(end - this->start_).count();
    }
  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class ctranMapper {
public:
  ctranMapper(ncclComm *comm);
  ~ctranMapper();
  ncclResult_t regMem(const void *buf, std::size_t len, void **hdl, bool forceRegist = false);
  ncclResult_t deregMem(void *hdl);
  ncclResult_t searchRegHandle(const void *buf, std::size_t len, void **hdl, bool *dynamicRegist);
  ncclResult_t icopy(void *dbuf, const void *sbuf, std::size_t len, ctranMapperRequest **req);
  ncclResult_t getTmpBuf(void** addr, std::size_t len, void **hdl);
  ncclResult_t releaseTmpBuf(void* addr, void *hdl);

  ncclResult_t isendCtrl(void *buf, void *hdl, int rank, ctranMapperRequest **req);
  ncclResult_t irecvCtrl(void **buf, struct ctranMapperRemoteAccessKey *key, int rank,
      ctranMapperRequest **req);
  ncclResult_t iput(const void *sbuf, void *dbuf, std::size_t len, int rank, void *shdl,
      struct ctranMapperRemoteAccessKey remoteAccessKey, bool notify, ctranMapperRequest **req);
  ncclResult_t checkNotify(int rank, bool *notify);
  ncclResult_t waitNotify(int rank);
  void reportRegSnapshot();

  int rank;
  uint64_t commHash;
  std::vector<ctranMapperTimestamp> timestamps;

protected:
  ncclResult_t progress(void);
  cudaStream_t s;

private:
  class impl;
  std::unique_ptr<impl> pimpl;
  std::mutex tmpBufLock;
  friend class ctranMapperRequest;
};

#endif
