#pragma once

// [TAG_FN_L22_P4] round 22, package P4 (prompt stream state machine): declarations shared by llama-context.cpp,
// llama-prefill-stream.cpp and llama-moecache.cpp only. Every switch below is off unless set (plan 3.4).

#include <cstddef>
#include <cstdint>

struct llama_model;

//
// prefill stream (llama-prefill-stream.cpp)
//

struct llama_pfs_l22_cfg {
    bool     keeplend   = false; // LLAMA_FN_L22_KEEPLEND: a narrow ubatch with prompt rows after it keeps the banks lent
    bool     endoverlap = false; // LLAMA_FN_L22_ENDOVERLAP: the prompt's narrow last part keeps them too
    bool     prestage   = false; // LLAMA_FN_L22_REFILL (c): layer 0 is copied at the lend, before the first gate
    bool     readahead  = false; // LLAMA_FN_L22_READAHEAD: one sample of every layer at gate 0, missing layers prefetched
    size_t   ra_call    = 0;     // LLAMA_FN_L22_READAHEAD_MIB: the most bytes one prefetch call asks for
    int      ssdtier    = 0;     // LLAMA_FN_L22_SSDTIER: 1 SSD-class experts, 2 also layers more than half out of RAM
    bool     pin        = false; // LLAMA_FN_L22_PIN: pinned segments (llama_fn_pin_query) go straight to the banks
    int      passlog    = 0;     // LLAMA_FN_L22_PASSLOG level
};

// once, after the stream exists (no-op without a stream of this owner)
void llama_prefill_stream_l22_config(const void * owner, const llama_model * model, const llama_pfs_l22_cfg & cfg);

// before each ubatch of the owner: rows of prompt still to come after it (any slot), and whether it is a prompt's
// narrow last part that keeps the FULL state (ENDOVERLAP)
void llama_prefill_stream_l22_set_after(const void * owner, int64_t rows_after, bool tail);

// the next return of the banks queues the hot set's refill on its upload worker (ENDOVERLAP: the answer's first step)
void llama_prefill_stream_l22_async_return(const void * owner, bool on);

// the banks are lent right now
bool llama_prefill_stream_l22_lent(const void * owner);

// [TAG_FN_L22_ENDOVERLAP] a prompt ended with the banks still lent (the answer's first step returns them): the lent layers
// are hidden from the hot chains until then (the MTP draft's graphs run in between). sched: the owner's scheduler
void llama_prefill_stream_l22_hold(const void * owner, struct ggml_backend_sched * sched);

// per-prompt sums for PASSLOG (taken and reset)
struct llama_pfs_l22_sums {
    uint64_t ubatches     = 0;
    uint64_t bytes_pinned = 0; // straight from pinned host pages
    uint64_t bytes_staged = 0; // through the ring (mapped read)
    uint64_t bytes_ssd    = 0; // through the ring by direct I/O
    uint64_t bytes_d2d    = 0; // from the hot set's slots
    double   wait_ms      = 0.0;
    double   miss_share   = 0.0; // the last gate-0 sample: share of sampled pages out of RAM
    uint32_t miss_layers  = 0;   // layers with > 2 % of the sample out
    uint64_t ra_bytes     = 0;   // asked for by the read-ahead
    uint64_t keeps        = 0;   // narrow ubatches that kept the lend
};
llama_pfs_l22_sums llama_prefill_stream_l22_take(const void * owner);
