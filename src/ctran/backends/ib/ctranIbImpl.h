// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_IMPL_H_
#define CTRAN_IB_IMPL_H_

#include <vector>
#include <thread>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
#include "ibvwrap.h"
#include "bootstrap.h"

#define BOOTSTRAP_CMD_SETUP  (0)
#define BOOTSTRAP_CMD_TERMINATE  (1)

#define MAX_SEND_WR      (256)

struct pendingOp {
  enum pendingOpType {
    ISEND_CTRL,
    IRECV_CTRL,
  } type;
  struct {
    void *buf;
    void *ibRegElem;
    int peerRank;
    ctranIbRequest *req;
  } isendCtrl;
  struct {
    void **buf;
    struct ctranIbRemoteAccessKey *key;
    int peerRank;
    ctranIbRequest *req;
  } irecvCtrl;
};

class ctranIbSingleton {
  public:
    ctranIbSingleton(const ctranIbSingleton& obj) = delete;
    static ctranIbSingleton& getInstance();
    std::vector<int> ports;
    std::vector<struct ibv_context *> contexts;
    std::vector<struct ibv_pd *> pds;
    std::vector<std::string> devNames;
    void recordCtxTraffic(struct ibv_context *ctx, size_t nbytes);
    void recordQpTraffic(struct ibv_qp* qp, size_t nbytes);

  private:
    std::unordered_map<std::string, size_t> trafficPerDevice;
    std::unordered_map<uint32_t, size_t> trafficPerQP;
    std::mutex trafficRecordMutex;
    ctranIbSingleton();
    ~ctranIbSingleton();
};

class ctranIb::impl {
public:
  impl() = default;
  ~impl() = default;

  static void bootstrapAccept(ctranIb::impl *pimpl);
  ncclResult_t bootstrapConnect(int peerRank);
  ncclResult_t bootstrapTerminate();

  const char *ibv_wc_status_str(enum ibv_wc_status status);

  int rank;
  int nRanks;
  struct ibv_context *context;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  int port;

  struct ncclSocket listenSocket;
  ncclSocketAddress *allListenSocketAddrs;
  std::thread listenThread;

  /* individual VCs for each peer */
  class vc;
  std::vector<class vc *> vcList;
  std::vector<uint32_t> numUnsignaledPuts;
  ctranIbRequest fakeReq;
  std::unordered_map<uint32_t, int> qpToRank;
  std::mutex m;

  std::vector<struct pendingOp *> pendingOps;

private:
  ncclResult_t bootstrapConnect(int peerRank, int cmd);
};

#endif
