// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comm.h"
#include "ddaPrivateMd.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_DDA_FORCE_P2P_ACCESS
   type        : bool
   default     : false
   description : |-
     Force allow direct peer-to-peer access over NVLink, even for the
     cases where there is no direct NVLink connectivity.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

/* This function only returns two types of cliques: fully connected
 * (NVSwitch) or HCM, to support typically GPU topologies.  In all
 * other cases, we return an empty vector.
 *
 * We do not currently account for the number of NVLinks connecting
 * two GPUs because our algorithm does not take advantage of that
 * (yet!).
 *
 * This also enables p2p access between all GPUs that are connected
 * using NVLink.  Unfortunately, we do not have a good way to detect
 * which GPUs are being used for this particular job, so we enable p2p
 * access for all connected GPUs.
 */
static ncclResult_t topoDetect(
    ncclComm_t comm,
    std::vector<std::vector<int>>& cliques) {
  int nGPUs;
  CUDACHECK(cudaGetDeviceCount(&nGPUs));

  /* perf rank Matrix is like an adjacency matrix, but ranks links
   * based on performance.  Rank 0 means very fast connectivity. */
  std::vector<std::vector<uint8_t>> perfRankMatrix(nGPUs, std::vector<uint8_t>(nGPUs));
  std::vector<std::vector<uint8_t>> adjacencyMatrix(nGPUs, std::vector<uint8_t>(nGPUs));

  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (i == j) {
        perfRankMatrix[i][j] = 0;
      } else if (NCCL_DDA_FORCE_P2P_ACCESS) {
        perfRankMatrix[i][j] = 0;
      } else {
        int val;
        cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrPerformanceRank, i, j);
        perfRankMatrix[i][j] = (uint8_t) val;
      }
    }
  }

  /* set adjacency matrix */
  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (perfRankMatrix[i][j] < 2) {
        adjacencyMatrix[i][j] = 1;
      } else {
        adjacencyMatrix[i][j] = 0;
      }
    }
  }

  /* clear the cliques before we start */
  cliques.clear();


  /***** Detect fully connected (NVSwitch) topology *****/
  bool connected;
  connected = true;
  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (adjacencyMatrix[i][j] == 0) {
        connected = false;
        break;
      }
    }
    if (connected == false) {
      break;
    }
  }
  if (connected) {
    std::vector<int> v;
    for (int i = 0; i < nGPUs; i++) {
      v.push_back(i);
    }
    cliques.push_back(v);
    goto exit;
  }

  /***** Detect HCM topology *****/
  for (int i = 0; i < nGPUs; i++) {
    cliques.push_back(std::vector<int>{i});
  }

  /* find cliques of size nGPUs/2 */
  for (int k = 2; k <= nGPUs / 2; k++) {
    std::vector<std::vector<int>> tmp;
    for (auto v : cliques) {
      for (int i = v.back() + 1; i < nGPUs; i++) {
        bool connected = true;
        for (auto j : v) {
          if (adjacencyMatrix[i][j] == 0) {
            connected = false;
            break;
          }
        }
        if (connected) {
          std::vector<int> w = v;
          w.push_back(i);
          tmp.push_back(w);
        }
      }
    }

    cliques.clear();
    cliques = tmp;
    tmp.clear();
  }

  /* HCM has two cliques */
  if (cliques.size() != 2) {
    goto topo_not_found;
  }

  /* In HCM, each GPU is connected to (nGPUs/2 + 1) other GPUs */
  for (int i = 0; i < nGPUs; i++) {
    int count = 0;
    for (int j = 0; j < nGPUs; j++) {
      if (adjacencyMatrix[i][j]) {
        count++;
      }
    }
    if (count != (1 + nGPUs / 2)) {
      goto topo_not_found;
    }
  }

  /* layout the cliques, so the two HCM cliques are ordered so that
   * the i'th GPU in one clique is connected to the i'th GPU in the
   * other clique.  If this is not possible, it's not HCM. */
  {
    std::vector<int> front = cliques.front();
    std::vector<int> back = cliques.back();
    std::vector<int> tmp;
    for (auto f : front) {
      /* each element in front should be connected to exactly
       * one element in back */
      for (auto b : back) {
        if (adjacencyMatrix[f][b]) {
          tmp.push_back(b);
          break;
        }
      }
    }
    assert(tmp.size() == nGPUs / 2);
    cliques.pop_back();
    cliques.push_back(tmp);
  }

exit:
  for (int i = 0; i < nGPUs; i++) {
    int dev;
    CUDACHECK(cudaGetDevice(&dev));
    if (i == dev) {
      continue;
    } else if (adjacencyMatrix[comm->cudaDev][i] == 1) {
      cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
      if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
        CUDACHECK(e);
      }
    }
  }
  return ncclSuccess;

topo_not_found:
  cliques.clear();
  return ncclSuccess;
}

ddaPrivateMd::ddaPrivateMd(ddaThreadSharedMd *threadSharedMd, ncclComm *comm) {
  this->barrierFlag = 0;
  this->barrierMboxId = 1;
  this->comm = comm;
  CUDACHECKIGNORE(cudaGetDeviceProperties(&this->devProp, this->comm->cudaDev));

  this->threadSharedMd = threadSharedMd;
  CUDACHECKIGNORE(cudaHostAlloc(&this->commMdHost, this->comm->nRanks * sizeof(commMd), cudaHostAllocDefault));
  CUDACHECKIGNORE(cudaMalloc(&this->commMdHost[this->comm->rank].tmpbuff, NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE));
  CUDACHECKIGNORE(cudaMalloc(&this->commMdDev, sizeof(commMd) * this->comm->nRanks));

  int *cudaDevPtr;
  CUDACHECKIGNORE(cudaMalloc(&cudaDevPtr, this->comm->nRanks * sizeof(int)));
  CUDACHECKIGNORE(cudaMemcpy(&cudaDevPtr[this->comm->rank], &this->comm->cudaDev, sizeof(int), cudaMemcpyDefault));

  cudaStream_t s;
  CUDACHECKIGNORE(cudaStreamCreate(&s));
  NCCLCHECKIGNORE(ncclAllGather(&cudaDevPtr[this->comm->rank], cudaDevPtr, sizeof(int), ncclUint8, this->comm, s));
  CUDACHECKIGNORE(cudaStreamSynchronize(s));
  CUDACHECKIGNORE(cudaStreamDestroy(s));

  std::vector<int> cudaDevs(this->comm->nRanks);
  CUDACHECKIGNORE(cudaMemcpy(cudaDevs.data(), cudaDevPtr, this->comm->nRanks * sizeof(int), cudaMemcpyDefault));

  std::unordered_map<int, int> gpuToRank;
  for (int i = 0; i < this->comm->nRanks; i++) {
    this->rankToGpu[i] = cudaDevs[i];
    gpuToRank[cudaDevs[i]] = i;
  }

  std::vector<std::vector<int>> gpuCliques;
  NCCLCHECKIGNORE(topoDetect(this->comm, gpuCliques));

  this->topoType = NCCL_DDA_TOPO_TYPE__UNKNOWN;
  if (gpuCliques.size() == 1) {
    this->topoType = NCCL_DDA_TOPO_TYPE__NVS;
    this->u.nvs.gpus = gpuCliques.front();
  } else if (gpuCliques.size() == 2) {
    this->topoType = NCCL_DDA_TOPO_TYPE__HCM;
    this->u.hcm.clique[0].gpus = gpuCliques.front();
    this->u.hcm.clique[1].gpus = gpuCliques.back();
  }

  int numBarrierPtrs = 3 * this->comm->nRanks;
  for (int i = 0; i < 2; i++) {
    CUDACHECKIGNORE(cudaMalloc(&this->commMdHost[this->comm->rank].barrierMbox[i], numBarrierPtrs * sizeof(uintptr_t)));
    CUDACHECKIGNORE(cudaMemset(this->commMdHost[this->comm->rank].barrierMbox[i], 0, numBarrierPtrs * sizeof(uintptr_t)));
  }

  // insert local handles
  this->memHandles = new ddaMemHandles(this->threadSharedMd, this->comm);
  NCCLCHECKIGNORE(this->memHandles->insertMemHandle(this->commMdHost[this->comm->rank].barrierMbox[0], "barrierMbox0"));
  NCCLCHECKIGNORE(this->memHandles->insertMemHandle(this->commMdHost[this->comm->rank].barrierMbox[1], "barrierMbox1"));
  NCCLCHECKIGNORE(this->memHandles->insertMemHandle(this->commMdHost[this->comm->rank].tmpbuff, "tmpbuff"));

  // exchange handles
  NCCLCHECKIGNORE(this->memHandles->exchangeMemHandles());

  for (int i = 0; i < this->comm->nRanks; i++) {
    this->commMdHost[i].barrierMbox[0] = static_cast<uintptr_t *>(this->memHandles->getMemAddr(i, "barrierMbox0"));
    this->commMdHost[i].barrierMbox[1] = static_cast<uintptr_t *>(this->memHandles->getMemAddr(i, "barrierMbox1"));
    this->commMdHost[i].tmpbuff = this->memHandles->getMemAddr(i, "tmpbuff");
  }

  if (this->topoType == NCCL_DDA_TOPO_TYPE__HCM) {
    int idx = 0;
    std::vector<int> topoRanks(this->comm->nRanks);
    for (int i = 0; i < 2; i++) {
      for (auto g : this->u.hcm.clique[i].gpus) {
        topoRanks[idx] = gpuToRank[g];
        if (topoRanks[idx] == this->comm->rank) {
          this->commMdHost[this->comm->rank].topoRankIdx = idx;
        }
        idx++;
      }
    }

    CUDACHECKIGNORE(cudaMalloc(&this->commMdHost[this->comm->rank].topoRanks, this->comm->nRanks * sizeof(int)));
    CUDACHECKIGNORE(cudaMemcpy(this->commMdHost[this->comm->rank].topoRanks, topoRanks.data(),
          this->comm->nRanks * sizeof(int), cudaMemcpyDefault));
  } else {
    this->commMdHost[this->comm->rank].topoRanks = nullptr;
  }

  CUDACHECKIGNORE(cudaMemcpy(this->commMdDev, this->commMdHost, this->comm->nRanks * sizeof(commMd),
                             cudaMemcpyDefault));
}

ddaPrivateMd::~ddaPrivateMd(void) {
  delete this->memHandles;
  CUDACHECKIGNORE(cudaFree(this->commMdDev));
  CUDACHECKIGNORE(cudaFree(this->commMdHost[this->comm->rank].tmpbuff));
  for (int i = 0; i < 2; i++) {
    CUDACHECKIGNORE(cudaFree(this->commMdHost[this->comm->rank].barrierMbox[i]));
  }
  if (this->commMdHost[this->comm->rank].topoRanks) {
    CUDACHECKIGNORE(cudaFree(this->commMdHost[this->comm->rank].topoRanks));
  }
  CUDACHECKIGNORE(cudaFreeHost(this->commMdHost));
}
