// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Automatically generated
//   by ./maint/extractcvars.py
// DO NOT EDIT!!!

#include <string>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <cstring>
#include "nccl_cvars.h"
#include "debug.h"

static std::set<std::string> tokenizer(const char *str_, const char *def_) {
  const char *def = def_ ? def_ : "";
  std::string str(getenv(str_) ? getenv(str_) : def);
  std::string delim = ",";
  std::set<std::string> tokens;

  while (auto pos = str.find(",")) {
    std::string newstr = str.substr(0, pos);
    if (tokens.find(newstr) != tokens.end()) {
      WARN("Duplicate token %s found in the value of %s", newstr.c_str(), str_);
    }
    tokens.insert(newstr);
    str.erase(0, pos + delim.length());
    if (pos == std::string::npos) {
      break;
    }
  }
  return tokens;
}

static bool env2bool(const char *str_, const char *def) {
  std::string str(getenv(str_) ? getenv(str_) : def);
  std::transform(str.cbegin(), str.cend(), str.begin(), [](unsigned char c) { return std::tolower(c); });
  if (str == "y") return true;
  else if (str == "n") return false;
  else if (str == "yes") return true;
  else if (str == "no") return false;
  else if (str == "t") return true;
  else if (str == "f") return false;
  else if (str == "true") return true;
  else if (str == "false") return false;
  else if (str == "1") return true;
  else if (str == "0") return false;
  else WARN("Unrecognized value for env %s\n", str_);
  return true;
}

static int env2int(const char *str, const char *def) {
  return getenv(str) ? atoi(getenv(str)) : atoi(def);
}

static std::string env2str(const char *str, const char *def_) {
  const char *def = def_ ? def_ : "";
  return getenv(str) ? std::string(getenv(str)) : std::string(def);
}

static std::set<std::string> env2strlist(const char *str, const char *def) {
  return tokenizer(str, def);
}

bool NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM;

int NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE;

int NCCL_DDA_MAX_RANKS;

enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO;

int NCCL_ALLGATHER_DIRECT_CUTOFF;

enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO;

bool NCCL_CTRAN_ENABLE_LOCAL_IB;

int NCCL_DDA_ALLREDUCE_MAX_BLOCKS;

int NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS;

int NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM;

int NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS;

int NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE;

bool NCCL_DDA_FORCE_P2P_ACCESS;

enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO;

std::set<std::string> NCCL_IB_HCA;

int NCCL_CTRAN_IB_MAX_QPS;

int NCCL_CTRAN_IB_QP_SCALING_THRESHOLD;

enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING;

std::string NCCL_CTRAN_KINETO_PROFILE_DIR;

enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER;

std::set<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS;

int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT;

extern char **environ;

void ncclCvarInit() {
  std::unordered_set<std::string> env;
  env.insert("NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM");
  env.insert("NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE");
  env.insert("NCCL_DDA_MAX_RANKS");
  env.insert("NCCL_ALLREDUCE_ALGO");
  env.insert("NCCL_ALLGATHER_DIRECT_CUTOFF");
  env.insert("NCCL_ALLGATHER_ALGO");
  env.insert("NCCL_CTRAN_ENABLE_LOCAL_IB");
  env.insert("NCCL_DDA_ALLREDUCE_MAX_BLOCKS");
  env.insert("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS");
  env.insert("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM");
  env.insert("NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS");
  env.insert("NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE");
  env.insert("NCCL_DDA_FORCE_P2P_ACCESS");
  env.insert("NCCL_SENDRECV_ALGO");
  env.insert("NCCL_IB_HCA");
  env.insert("NCCL_CTRAN_IB_MAX_QPS");
  env.insert("NCCL_CTRAN_IB_QP_SCALING_THRESHOLD");
  env.insert("NCCL_CTRAN_PROFILING");
  env.insert("NCCL_CTRAN_KINETO_PROFILE_DIR");
  env.insert("NCCL_CTRAN_REGISTER");
  env.insert("NCCL_CTRAN_BACKENDS");
  env.insert("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT");
  env.insert("NCCL_ALGO");
  env.insert("NCCL_COLLNET_ENABLE");
  env.insert("NCCL_COLLTRACE_LOCAL_SUBDIR");
  env.insert("NCCL_COMM_ID");
  env.insert("NCCL_CUDA_PATH");
  env.insert("NCCL_DEBUG");
  env.insert("NCCL_DEBUG_FILE");
  env.insert("NCCL_DEBUG_SUBSYS");
  env.insert("NCCL_GRAPH_DUMP_FILE");
  env.insert("NCCL_GRAPH_FILE");
  env.insert("NCCL_HOSTID");
  env.insert("NCCL_IB_GID_INDEX");
  env.insert("NCCL_IB_TC");
  env.insert("NCCL_LAUNCH_MODE");
  env.insert("NCCL_NET");
  env.insert("NCCL_NET_PLUGIN");
  env.insert("NCCL_NSOCKS_PERTHREAD");
  env.insert("NCCL_PROTO");
  env.insert("NCCL_PROXY_PROFILE");
  env.insert("NCCL_SHM_DISABLE");
  env.insert("NCCL_SOCKET_FAMILY");
  env.insert("NCCL_SOCKET_IFNAME");
  env.insert("NCCL_SOCKET_NTHREADS");
  env.insert("NCCL_THREAD_THRESHOLDS");
  env.insert("NCCL_TOPO_DUMP_FILE");
  env.insert("NCCL_TOPO_FILE");
  env.insert("NCCL_TUNER_PLUGIN");

  char **s = environ;
  for (; *s; s++) {
    if (!strncmp(*s, "NCCL_", strlen("NCCL_"))) {
      std::string str(*s);
      str = str.substr(0, str.find("="));
      if (env.find(str) == env.end()) {
        WARN("Unknown env %s in the NCCL namespace\n", str.c_str());
      }
    }
  }

  NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM = env2bool("NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM", "False");

  NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE = env2int("NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE", "33554432");

  NCCL_DDA_MAX_RANKS = env2int("NCCL_DDA_MAX_RANKS", "16");

  if (getenv("NCCL_ALLREDUCE_ALGO") == nullptr) {
    NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLREDUCE_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::orig;
    } else if (str == std::string("dda")) {
      NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::dda;
    } else {
      WARN("Unknown value %s for env NCCL_ALLREDUCE_ALGO", str.c_str());
    }
  }

  NCCL_ALLGATHER_DIRECT_CUTOFF = env2int("NCCL_ALLGATHER_DIRECT_CUTOFF", "0");

  if (getenv("NCCL_ALLGATHER_ALGO") == nullptr) {
    NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLGATHER_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::orig;
    } else if (str == std::string("ctdirect")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctdirect;
    } else if (str == std::string("ctring")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctring;
    } else if (str == std::string("ctrd")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctrd;
    } else {
      WARN("Unknown value %s for env NCCL_ALLGATHER_ALGO", str.c_str());
    }
  }

  NCCL_CTRAN_ENABLE_LOCAL_IB = env2bool("NCCL_CTRAN_ENABLE_LOCAL_IB", "False");

  NCCL_DDA_ALLREDUCE_MAX_BLOCKS = env2int("NCCL_DDA_ALLREDUCE_MAX_BLOCKS", "1");

  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS = env2int("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS", "262144");

  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM = env2int("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM", "65536");

  NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS = env2int("NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS", "-1");

  NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE = env2int("NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE", "-1");

  NCCL_DDA_FORCE_P2P_ACCESS = env2bool("NCCL_DDA_FORCE_P2P_ACCESS", "False");

  if (getenv("NCCL_SENDRECV_ALGO") == nullptr) {
    NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_SENDRECV_ALGO"));
    if (str == std::string("orig")) {
      NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;
    } else if (str == std::string("ctran")) {
      NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::ctran;
    } else {
      WARN("Unknown value %s for env NCCL_SENDRECV_ALGO", str.c_str());
    }
  }

  NCCL_IB_HCA = env2strlist("NCCL_IB_HCA", nullptr);

  NCCL_CTRAN_IB_MAX_QPS = env2int("NCCL_CTRAN_IB_MAX_QPS", "1");

  NCCL_CTRAN_IB_QP_SCALING_THRESHOLD = env2int("NCCL_CTRAN_IB_QP_SCALING_THRESHOLD", "1048576");

  if (getenv("NCCL_CTRAN_PROFILING") == nullptr) {
    NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::none;
  } else {
    std::string str(getenv("NCCL_CTRAN_PROFILING"));
    if (str == std::string("none")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::none;
    } else if (str == std::string("stdout")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::stdout;
    } else if (str == std::string("kineto")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::kineto;
    } else {
      WARN("Unknown value %s for env NCCL_CTRAN_PROFILING", str.c_str());
    }
  }

  NCCL_CTRAN_KINETO_PROFILE_DIR = env2str("NCCL_CTRAN_KINETO_PROFILE_DIR", "/tmp");

  if (getenv("NCCL_CTRAN_REGISTER") == nullptr) {
    NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::lazy;
  } else {
    std::string str(getenv("NCCL_CTRAN_REGISTER"));
    if (str == std::string("none")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::none;
    } else if (str == std::string("lazy")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::lazy;
    } else if (str == std::string("eager")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::eager;
    } else {
      WARN("Unknown value %s for env NCCL_CTRAN_REGISTER", str.c_str());
    }
  }

  {
    auto tokens = tokenizer("NCCL_CTRAN_BACKENDS", "ib");
    for (auto token : tokens) {
      if (token == std::string("ib")) {
        NCCL_CTRAN_BACKENDS.insert(NCCL_CTRAN_BACKENDS::ib);
      } else if (token == std::string("nvl")) {
        NCCL_CTRAN_BACKENDS.insert(NCCL_CTRAN_BACKENDS::nvl);
      } else {
        WARN("Unknown value %s for env NCCL_CTRAN_BACKENDS", token.c_str());
      }
    }
  }

  NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = env2int("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT", "0");

}
