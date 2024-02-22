// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef EXT_UTILS_H
#define EXT_UTILS_H

#include <sstream>
#include <vector>

template <typename T>
std::string vecToStr(const std::vector<T>& vec, std::string delim = ", ") {
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
