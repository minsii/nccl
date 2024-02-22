// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef TRACE_UTILES_H
#define TRACE_UTILES_H

#include <chrono>
#include <iomanip>
#include <sstream>

// TODO: should this be util function?
static inline std::string timePointToStr(
    std::chrono::time_point<std::chrono::high_resolution_clock> ts) {
  std::time_t ts_c = std::chrono::system_clock::to_time_t(ts);
  auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                   ts.time_since_epoch()) %
      1000000;
  std::stringstream ts_ss;
  ts_ss << std::put_time(std::localtime(&ts_c), "%T.") << std::setfill('0')
        << std::setw(6) << ts_us.count();
  return ts_ss.str();
}

#endif
