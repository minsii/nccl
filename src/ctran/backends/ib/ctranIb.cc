// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>
#include "nccl.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "ctranIbBase.h"
#include "ctranIb.h"
#include "ctranIbImpl.h"
#include "ctranIbVc.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_IB_HCA
   type        : stringlist
   default     :
   description : |-
     List of IB HCAs available for NCCL to use.
     (this needs to be renamed to NCCL_IB_HCA_LIST eventually)

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

class roceHca {
public:
  roceHca(std::string hcaStr) {
    std::string s = hcaStr;
    std::string delim = ":";

    auto pos = s.find(delim);
    if (pos == std::string::npos) {
      this->name = s;
      this->port = 0;
    } else {
      this->name = s.substr(0, pos);
      s.erase(0, pos + delim.length());
      this->port = std::stoi(s);
    }
  }

  std::string name;
  int port;
};

ctranIbSingleton &ctranIbSingleton::getInstance(void) {
  static ctranIbSingleton s;
  return s;
}

ctranIbSingleton::ctranIbSingleton(void) {
  std::vector<roceHca *> hcas;
  for (auto hca : NCCL_IB_HCA) {
    hcas.push_back(new roceHca(hca));
  }

  NCCLCHECKIGNORE(wrap_ibv_symbols());

  struct ibv_device **devs;
  std::vector<struct ibv_device *> devices;
  int nDevs;
  NCCLCHECKIGNORE(wrap_ibv_get_device_list(&devs, &nDevs));

  for (int i = 0; i < nDevs; i++) {
    bool found = false;
    int port;
    for (auto d : hcas) {
      if (!strcmp(d->name.c_str(), devs[i]->name)) {
        found = true;
        port = d->port;
        break;
      }
    }
    if (!found) {
      continue;
    }

    struct ibv_device *device = devs[i];
    devices.push_back(device);
    this->ports.push_back(port);
  }

  if (devices.empty()) {
    throw std::bad_alloc();
  }

  for (auto i = 0; i < devices.size(); i++) {
    struct ibv_context *context;
    struct ibv_pd *pd;
    NCCLCHECKIGNORE(wrap_ibv_open_device(&context, devices[i]));
    NCCLCHECKIGNORE(wrap_ibv_alloc_pd(&pd, context));

    this->contexts.push_back(context);
    this->pds.push_back(pd);
    this->devNames.push_back(devices[i]->name);
  }
}

ctranIbSingleton::~ctranIbSingleton() {
  for (auto pd : this->pds) {
    NCCLCHECKIGNORE(wrap_ibv_dealloc_pd(pd));
  }

  for (auto context : this->contexts) {
    NCCLCHECKIGNORE(wrap_ibv_close_device(context));
  }
}

ctranIb::ctranIb(ncclComm *comm) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  this->pimpl->rank = comm->rank;
  this->pimpl->nRanks = comm->nRanks;

  ctranIbSingleton& s = ctranIbSingleton::getInstance();

  this->pimpl->context = s.contexts[comm->cudaDev];
  this->pimpl->pd = s.pds[comm->cudaDev];
  this->pimpl->port = s.ports[comm->cudaDev];
  INFO(NCCL_INIT, "CTRAN-IB: using device %s, port %d", s.devNames[comm->cudaDev].c_str(), this->pimpl->port);

  struct ibv_device_attr devAttr;
  NCCLCHECKIGNORE(wrap_ibv_query_device(this->pimpl->context, &devAttr));

  /* The max CQEs would not be enough for us in the worst case, where
   * we have a lot of VCs, and there is a lot of posted messages on
   * each of the VCs.  Static partitioning would reduce the number of
   * CQEs available to each VC in the common case.  Instead, we are
   * making an assumption here that the progress thread will pull out
   * completion entries fast enough that we will never overflow the
   * CQ. */
  NCCLCHECKIGNORE(wrap_ibv_create_cq(&this->pimpl->cq, this->pimpl->context,
        devAttr.max_cqe, nullptr, nullptr, 0));

  char ifName[MAX_IF_NAME_SIZE+1];
  union ncclSocketAddress ifAddr;
  int nIfs = ncclFindInterfaces(ifName, &ifAddr, MAX_IF_NAME_SIZE, 1);
  if (nIfs <= 0) {
    WARN("CTRAN-IB: no socket interfaces found\n");
  } else {
    INFO(NCCL_INIT, "CTRAN-IB: socket interface set to %s", ifName);
  }

  NCCLCHECKIGNORE(ncclSocketInit(&this->pimpl->listenSocket, &ifAddr));
  NCCLCHECKIGNORE(ncclSocketListen(&this->pimpl->listenSocket));

  this->pimpl->allListenSocketAddrs =
    static_cast<ncclSocketAddress *>(malloc(this->pimpl->nRanks * sizeof(ncclSocketAddress)));
  NCCLCHECKIGNORE(ncclSocketGetAddr(&this->pimpl->listenSocket,
        &this->pimpl->allListenSocketAddrs[this->pimpl->rank]));

  this->pimpl->listenThread = std::thread{ctranIb::impl::bootstrapAccept, this->pimpl.get()};

  bootstrapAllGather(comm->bootstrap, this->pimpl->allListenSocketAddrs, sizeof(ncclSocketAddress));

  for (int r = 0; r < this->pimpl->nRanks; r++) {
    this->pimpl->vcList.push_back(new ctranIb::impl::vc(this->pimpl->context, this->pimpl->pd,
          this->pimpl->cq, this->pimpl->port, r));
    this->pimpl->numUnsignaledPuts.push_back(0);
  }
}

ctranIb::~ctranIb(void) {
  NCCLCHECKIGNORE(this->pimpl->bootstrapTerminate());
  this->pimpl->listenThread.join();

  for (int r = 0; r < this->pimpl->nRanks; r++) {
    delete this->pimpl->vcList[r];
  }

  free(this->pimpl->allListenSocketAddrs);
  NCCLCHECKIGNORE(ncclSocketClose(&this->pimpl->listenSocket));

  NCCLCHECKIGNORE(wrap_ibv_destroy_cq(this->pimpl->cq));
}

ncclResult_t ctranIb::regMem(const void *buf, std::size_t len, void **ibRegElem) {
  ncclResult_t res = ncclSuccess;

  int pageSize = getpagesize();
  if (len <= pageSize) {
    WARN("CTRAN-IB: cannot register buffer, size (%lu) <= page size (%d)", len, pageSize);
    res = ncclSystemError;
    goto exit;
  }

  struct ibv_mr *mr;
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, this->pimpl->pd, (void *) buf, len,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ), res, exit);
  *ibRegElem = reinterpret_cast<void *>(mr);

exit:
  return res;
}

ncclResult_t ctranIb::deregMem(void *ibRegElem) {
  ncclResult_t res = ncclSuccess;

  struct ibv_mr *mr = reinterpret_cast<struct ibv_mr *>(ibRegElem);
  NCCLCHECKGOTO(wrap_ibv_dereg_mr(mr), res, exit);

exit:
  return res;
}

ncclResult_t ctranIb::progress(void) {
  ncclResult_t res = ncclSuccess;

  /* complete as many requests as possible */
  while (1) {
    struct ibv_wc wc;
    int count;

    res = wrap_ibv_poll_cq(this->pimpl->cq, 1, &wc, &count);
    NCCLCHECKGOTO(res, res, exit);

    if (count == 0) {
      break;
    }

    /* wc.wr_id is valid even if the poll_cq returned an error; use it
     * to gather information about the error */
    this->pimpl->m.lock();
    auto vc = this->pimpl->vcList[this->pimpl->qpToRank[wc.qp_num]];
    this->pimpl->m.unlock();

    if (wc.status != IBV_WC_SUCCESS) {
      WARN("CTRAN-IB: wrap_ibv_poll_cq failed, peerRank=%d, with status=%d, '%s'",
          vc->peerRank, wc.status, this->pimpl->ibv_wc_status_str(wc.status));
      res = ncclSystemError;
      goto exit;
    }

    NCCLCHECKGOTO(vc->processCqe(wc.opcode, wc.qp_num, wc.imm_data), res, exit);
  }

  /* we should have pendingOps only if the connection was not
   * established yet.  The below algorithm is a bit inefficient, but
   * that is OK as it should not happen in the critical path. */
  if (!this->pimpl->pendingOps.empty()) {
    std::vector<int> peerRanks;
    std::vector<struct pendingOp *> tmp = this->pimpl->pendingOps;
    this->pimpl->pendingOps.clear();

    for (auto op : tmp) {
      int rank = (op->type == pendingOp::pendingOpType::ISEND_CTRL) ? op->isendCtrl.peerRank :
        op->irecvCtrl.peerRank;

      /* if we already encounted this peer, skip all operations to the
       * same peer; otherwise we might end up sending messages out of
       * order */
      if (std::find(peerRanks.begin(), peerRanks.end(), rank) != peerRanks.end()) {
        this->pimpl->pendingOps.push_back(op);
        continue;
      }

      auto vc = this->pimpl->vcList[rank];
      if (op->type == pendingOp::pendingOpType::ISEND_CTRL) {
        if (vc->isReady() == true) {
          NCCLCHECKGOTO(vc->isendCtrl(op->isendCtrl.buf, op->isendCtrl.ibRegElem, op->isendCtrl.req), res, exit);
          delete op;
        } else {
          this->pimpl->pendingOps.push_back(op);
          peerRanks.push_back(rank);
        }
      } else {
        if (vc->isReady() == true) {
          NCCLCHECKGOTO(vc->irecvCtrl(op->irecvCtrl.buf, op->irecvCtrl.key, op->irecvCtrl.req), res, exit);
          delete op;
        } else {
          this->pimpl->pendingOps.push_back(op);
          peerRanks.push_back(rank);
        }
      }
    }
  }

exit:
  return res;
}

ncclResult_t ctranIb::isendCtrl(void *buf, void *ibRegElem, int peerRank, ctranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl->vcList[peerRank];
  if (this->pimpl->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl->bootstrapConnect(peerRank), res, exit);
  }

  *req = new ctranIbRequest();
  if (vc->isReady() == true) {
    NCCLCHECKGOTO(vc->isendCtrl(buf, ibRegElem, *req), res, exit);
  } else {
    auto pendingOp = new struct pendingOp;
    pendingOp->type = pendingOp::pendingOpType::ISEND_CTRL;
    pendingOp->isendCtrl.buf = buf;
    pendingOp->isendCtrl.ibRegElem = ibRegElem;
    pendingOp->isendCtrl.peerRank = peerRank;
    pendingOp->isendCtrl.req = *req;
    this->pimpl->pendingOps.push_back(pendingOp);
  }

exit:
  return res;
}

ncclResult_t ctranIb::irecvCtrl(void **buf, struct ctranIbRemoteAccessKey *key, int peerRank,
    ctranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl->vcList[peerRank];
  if (this->pimpl->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl->bootstrapConnect(peerRank), res, exit);
  }

  *req = new ctranIbRequest();
  if (vc->isReady() == true) {
    NCCLCHECKGOTO(vc->irecvCtrl(buf, key, *req), res, exit);
  } else {
    auto pendingOp = new struct pendingOp;
    pendingOp->type = pendingOp::pendingOpType::IRECV_CTRL;
    pendingOp->irecvCtrl.buf = buf;
    pendingOp->irecvCtrl.key = key;
    pendingOp->irecvCtrl.peerRank = peerRank;
    pendingOp->irecvCtrl.req = *req;
    this->pimpl->pendingOps.push_back(pendingOp);
  }

exit:
  return res;
}

ncclResult_t ctranIb::iput(const void *sbuf, void *dbuf, std::size_t len, int peerRank, void *ibRegElem,
    struct ctranIbRemoteAccessKey remoteAccessKey, bool notify, ctranIbRequest **req) {
  ncclResult_t res = ncclSuccess;
  ctranIbRequest *r = nullptr;

  if (req != nullptr) {
    *req = new ctranIbRequest();
    r = *req;
    this->pimpl->numUnsignaledPuts[peerRank] = 0;
  } else {
    this->pimpl->numUnsignaledPuts[peerRank]++;
    if (this->pimpl->numUnsignaledPuts[peerRank] == MAX_SEND_WR) {
      r = &this->pimpl->fakeReq;
      this->pimpl->numUnsignaledPuts[peerRank] = 0;
    }
  }

  NCCLCHECKGOTO(this->pimpl->vcList[peerRank]->iput(sbuf, dbuf, len, ibRegElem, remoteAccessKey, notify, r),
      res, exit);

exit:
  return res;
}

ncclResult_t ctranIb::checkNotify(int peerRank, bool *notify) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->progress(), res, exit);
  NCCLCHECKGOTO(this->pimpl->vcList[peerRank]->checkNotify(notify), res, exit);

exit:
  return res;
}
