#pragma once

// [TAG_UPSTREAM_29510] [TAG_FN_L22_KVROWS] round 22 P13: per-sequence K/V cell lists of a unified turbot cache (the port
// 9ca2f8145 of upstream #29510, moved out of llama-graph / llama-kv-cache behind P0's hook llama_fn_kvrows_attn,
// src/llama-fn-l22.h). The pieces below are exported for tests/test-fn-l22-kvrows.cpp.
//
// A ubatch of n_slices groups of equal size in token order (each the tokens of one sequence) gets one turbot FA per
// slice that reads only the cells of its sequence (a cell list, src[9] of the FA). kv_pos and the slice's q_pos stay the
// whole positional mask, so a listed cell of another sequence is still -INF: the list only removes work.
//
// Switches (the port's names, plain environment, read once): LLAMA_KV_ROWS 0 off (default), 1 ubatches of several
// sequences, 2 single-sequence ubatches too; LLAMA_KV_ROWS_MAX_TPS (8) / LLAMA_KV_ROWS_MIN_TPS (1) bound the tokens per
// slice; GGML_CUDA_KV_ROWS=0 (fattn.cu) keeps the per-slice graph but runs each FA dense.

#include "llama.h"
#include "ggml.h"

#include <cstdint>
#include <functional>
#include <vector>

struct llama_ubatch;
class llama_kv_cells;

struct llama_fn_kvrows_cfg {
    int      mode    = 0; // LLAMA_KV_ROWS (0, 1, 2)
    uint32_t max_tps = 8; // LLAMA_KV_ROWS_MAX_TPS
    uint32_t min_tps = 1; // LLAMA_KV_ROWS_MIN_TPS
};

struct llama_fn_kvrows_plan {
    uint32_t n_slices = 0; // groups of one sequence each (= ubatch.n_seqs_unq)
    uint32_t n_rows   = 0; // list length per slice, 0 = no lists
    uint32_t n_tps    = 0; // tokens per slice

    bool operator==(const llama_fn_kvrows_plan & o) const {
        return n_slices == o.n_slices && n_rows == o.n_rows && n_tps == o.n_tps;
    }
};

// the switches, read once from the environment
LLAMA_API llama_fn_kvrows_cfg llama_fn_kvrows_cfg_env();

// the number of slices: n_seqs_unq groups of equal size in token order, each of one sequence, every token in one
// sequence only; 0 if the ubatch is not like that (checked on the token data, not taken from the split). A reserve
// ubatch (no sequence ids) gives 0.
LLAMA_API uint32_t llama_fn_kvrows_slices(const llama_ubatch & ubatch);

// list length of each slice: the largest cell count of the slices' sequences in cells, padded to 256 (a graph is reused
// until the longest list crosses a step); 0 when the lists would not be shorter than n_kv, when a sequence has no cell,
// or when the cells of the slices add up to more than n_kv (shared cells: the one FA reads them once)
LLAMA_API uint32_t llama_fn_kvrows_n_rows(const llama_kv_cells & cells, const llama_ubatch & ubatch, uint32_t n_slices,
                                          uint32_t n_kv);

// the plan for a ubatch over the cells of the cache's one stream (cfg gates, slices, n_rows); the model / cache gates
// (flash attention, causal, no indexer, turbot, one stream) are the caller's
LLAMA_API llama_fn_kvrows_plan llama_fn_kvrows_plan_cells(const llama_fn_kvrows_cfg & cfg, const llama_ubatch & ubatch,
                                                          const llama_kv_cells & cells, uint32_t n_kv);

// the lists from the positional vectors the attention input just wrote: kv_pos [n_kv, kv_rows] (row 1 = sequence sets
// when kv_rows == 2) and q_pos [n_tokens, kv_rows] (row 1 = the bit of each token's sequence). List s = the cells
// kv_pos shows to the sequence of slice s (pos >= 0 and, for two rows, the slice's bit in the set), ascending, then -1
// up to n_rows. out: [n_slices * n_rows]. False (out unspecified) when a list would overflow, the slice's tokens carry
// different bits, or the shapes do not fit.
LLAMA_API bool llama_fn_kvrows_fill_lists(const int32_t * kv_pos, int64_t n_kv, int32_t kv_rows, const int32_t * q_pos,
                                          int64_t n_tokens, uint32_t n_slices, uint32_t n_rows, int32_t * out);

// q_pos of each slice, [n_tps, 2] each, the slice's columns of both rows of q_pos [n_tokens, 2] (the values the
// attention input wrote, keys included). out: [n_slices * 2 * n_tps]
LLAMA_API void llama_fn_kvrows_split_q_pos(const int32_t * q_pos, int64_t n_tokens, uint32_t n_slices, int32_t * out);

// one FA node as build_attn_mha builds it (llama_fn_kvrows_args::build_fa)
using llama_fn_kvrows_build_fa_t = std::function<ggml_tensor *(ggml_tensor * q_fa, ggml_tensor * k_fa, ggml_tensor * v_fa,
                                                               ggml_tensor * mask_fa, ggml_tensor * gtab_fa,
                                                               ggml_tensor * q_pos_fa)>;

// the per-slice FAs for q [D, n_tokens, n_head, 1] (permuted as the FA takes it), k, v: slice s = tokens
// [s*n_tps, (s + 1)*n_tps) with q_pos (one slice) or q_pos_rows[s] and the list rows[s] (null: no list); outputs
// concatenated in token order ([D, n_head, n_tps, n_slices], the memory layout of the one FA's output)
LLAMA_API ggml_tensor * llama_fn_kvrows_build_slices(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                                     ggml_tensor * gtab, ggml_tensor * q_pos,
                                                     const std::vector<ggml_tensor *> & q_pos_rows,
                                                     const std::vector<ggml_tensor *> & rows, uint32_t n_slices,
                                                     const llama_fn_kvrows_build_fa_t & build_fa);

// counters for tests and PASSLOG: graphs built with lists, layers that took the lists
struct llama_fn_kvrows_stats {
    uint64_t graphs_listed = 0;
    uint64_t graphs_seen   = 0;
    uint64_t layers_listed = 0;
    uint64_t fills         = 0;
};

LLAMA_API llama_fn_kvrows_stats llama_fn_kvrows_stats_get();
