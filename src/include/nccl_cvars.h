// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Automatically generated
//   by ./maint/extractcvars.py
// DO NOT EDIT!!!

#ifndef NCCL_CVARS_H_INCLUDED
#define NCCL_CVARS_H_INCLUDED

#include <string>
#include <set>

extern bool NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM;

extern int NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE;

extern int NCCL_DDA_MAX_RANKS;

enum class NCCL_ALLREDUCE_ALGO {
  orig,
  dda,
};
extern enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO;

extern int NCCL_ALLGATHER_DIRECT_CUTOFF;

enum class NCCL_ALLGATHER_ALGO {
  orig,
  ctdirect,
  ctring,
  ctrd,
};
extern enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO;

extern bool NCCL_CTRAN_ENABLE_LOCAL_IB;

extern int NCCL_DDA_ALLREDUCE_MAX_BLOCKS;

extern int NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS;

extern int NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM;

extern int NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS;

extern int NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE;

extern bool NCCL_DDA_FORCE_P2P_ACCESS;

enum class NCCL_SENDRECV_ALGO {
  orig,
  ctran,
};
extern enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO;

extern std::set<std::string> NCCL_IB_HCA;

extern int NCCL_CTRAN_IB_MAX_QPS;

extern int NCCL_CTRAN_IB_QP_SCALING_THRESHOLD;

enum class NCCL_CTRAN_PROFILING {
  none,
  stdout,
  kineto,
};
extern enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING;

extern std::string NCCL_CTRAN_KINETO_PROFILE_DIR;

enum class NCCL_CTRAN_REGISTER {
  none,
  lazy,
  eager,
};
extern enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER;

enum class NCCL_CTRAN_BACKENDS {
  ib,
  nvl,
};
extern std::set<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS;

extern int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT;

void ncclCvarInit();

#endif  /* NCCL_CVARS_H_INCLUDED */
