#pragma once

// [TAG_FN_L22_P10] round 22, package P10 (stream tuning + G3): declarations shared by llama-context.cpp,
// llama-prefill-stream.cpp and llama-fn-auto.cpp only. Every switch is off unless set (plan 3.4).

#include <cstddef>
#include <cstdint>

// [TAG_FN_L22_XOVER] [TAG_FN_L22_UBPLAN] the measured prompt cost of one target context, per depth bucket
// (llama_fn_l22_depth_bucket); the pure helpers are in llama-fn-auto.h
struct llama_fn_l22_cost {
    double   pass_ms[4]    = {};  // EWMA: wall of a stream pass bound by its copies
    double   gpu_ms_row[4] = {};  // EWMA: (pass wall - gate wait) / rows, passes of >= 2048 rows
    double   cpu_ms_row[4] = {};  // EWMA: CPU-path prompt ubatch ms per row
    double   fix_ms        = 0.0; // EWMA: lend + return of the banks, once per streamed prompt
    double   big_ms        = 0.0; // EWMA: borrow + unborrow of a 16K FULL period
    uint32_t n_obs[4]  = {};      // stream passes seen
    uint32_t n_pass[4] = {};      // samples of pass_ms
    uint32_t n_gpu[4]  = {};
    uint32_t n_cpu[4]  = {};
    uint32_t n_fix = 0;
    uint32_t n_big = 0;
    bool     pinned = false;      // the priors in use (pinned copies are faster)
};

// [TAG_FN_L22_STAGERS] [TAG_FN_L22_XOVER] one finished stream pass (llama-prefill-stream.cpp)
struct llama_pfs_pass {
    uint64_t seq      = 0;   // pass number, 1, 2, ...
    uint32_t rows     = 0;   // ubatch rows
    double   wall_ms  = 0.0; // gate 0 to the last layer's release
    double   wait_ms  = 0.0; // gates waited for copies
    uint64_t b_host   = 0;   // host bytes (ring + pinned + direct I/O)
    uint64_t b_pinned = 0;
    uint64_t b_ssd    = 0;
    uint64_t b_d2d    = 0;
    int      stagers  = 0;   // copy threads of the pass
};

// [TAG_FN_L22_STAGERS] mode: 0 off (the copy pool is never touched), -1 auto (n_max x staged / host, 4..n_max), n >= 2
// fixed. Applied at the next lend of the banks (streamer idle)
void    llama_prefill_stream_set_stagers(const void * owner, int mode);
// [TAG_FN_L22_XOVER] the stream's minimum ubatch rows (a stream turned off stays off); false: no stream of the owner
bool    llama_prefill_stream_set_min(const void * owner, int64_t n);
int64_t llama_prefill_stream_get_min(const void * owner); // 0: no stream of the owner, INT64_MAX: off
// the last finished pass; true once per new pass
bool    llama_prefill_stream_pass_info(const void * owner, llama_pfs_pass * out);
// host ms of the banks' lends and returns since the last call (taken and reset)
double  llama_prefill_stream_fix_take(const void * owner);
