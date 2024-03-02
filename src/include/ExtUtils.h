// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef EXT_UTILS_H
#define EXT_UTILS_H

#include <cstdint>
#include <sstream>

std::string hashToHexStr(const uint64_t hash) {
  std::stringstream ss;
  ss << std::hex << hash;
  return ss.str();
}

#endif
