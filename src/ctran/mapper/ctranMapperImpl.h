// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_IMPL_H_
#define CTRAN_MAPPER_IMPL_H_

#include "ctranIb.h"
#include "ctranNvl.h"
#include "ctranMapper.h"
#include "ctranAvlTree.h"

enum ctranMapperRegElemState {
  CACHED,
  REGISTERED,
};

struct ctranMapperRegElem {
  const void *buf;
  std::size_t len;
  void *ibRegElem;
  void *nvlRegElem;
  enum ctranMapperRegElemState state;
};

enum ctranMapperBackend {
  UNSET,
  NVL,
  IB,
};

class ctranMapper::impl {
public:
  impl() = default;
  ~impl() = default;

  ncclResult_t regMem(struct ctranMapperRegElem *mapperRegElem);
  ncclResult_t deregMem(struct ctranMapperRegElem *mapperRegElem);

  class ctranAvlTree *mapperRegElemList;
  class ctranMapperMemPool *memPool;

  std::vector<enum ctranMapperBackend> rankBackendMap;
  std::vector<enum ctranMapperBackend> backends;
  std::unique_ptr<class ctranIb> ctranIb;
  std::unique_ptr<class ctranNvl> ctranNvl;

  uint32_t numRegistrations; /* number of currently registered buffers */
  uint32_t numCachedRegistrations; /* number of currently cached but not yet registered buffers in lazy registration; buffer still pre-registered by user. */
  uint32_t totalNumDynamicRegistrations; /* total number of buffers at lifetime that were not pre-registered by user but temporarily in communication */
  uint32_t totalNumRegistrations; /* total number of registered buffers at lifetime */
  uint32_t totalNumCachedRegistrations; /* total number of cached buffers at lifetime */
  uint32_t totalNumRegLookupHit; /* total number of lookup calls to search buffer registration and found registered handle */
  uint32_t totalNumRegLookupMiss; /* total number of lookup calls to search buffer registration and could
                                   * not found registered handle (i.e., by either lazy registration or dynamic registration )*/
};

#endif
