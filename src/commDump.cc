// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <string>
#include <unordered_map>
#include "comm.h"
#include "nccl.h"

__attribute__((visibility("default"))) ncclResult_t ncclCommDump(
    ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  map["commHash"] = std::to_string(comm->commHash);
  return ncclSuccess;
}
