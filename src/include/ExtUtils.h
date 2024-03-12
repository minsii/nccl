// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef EXT_UTILS_H
#define EXT_UTILS_H

#include <cstdint>
#include <sstream>
#include <vector>
#include <unordered_set>

static inline std::string hashToHexStr(const uint64_t hash) {
  std::stringstream ss;
  ss << std::hex << hash;
  return ss.str();
}

template <typename T>
static inline std::string vecToStr(
    const std::vector<T>& vec,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (auto it : vec) {
    if (!first) {
      ss << delim;
    }
    ss << it;
    first = false;
  }
  return ss.str();
}


template <typename T>
static inline std::string unorderedSetToStr(
    const std::unordered_set<T>& vec,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (auto it : vec) {
    if (!first) {
      ss << delim;
    }
    ss << it;
    first = false;
  }
  return ss.str();
}

#endif
