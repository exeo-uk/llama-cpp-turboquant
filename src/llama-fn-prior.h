#pragma once

// [TAG_FN_L22_X_PRIOR] round 22 P12 (qwen4exp, opt-in, changes answers): the route prior LLAMA_FN_L22_X_PRIOR=<lambda> and
// the resident swap LLAMA_FN_L22_X_RESIDENT=<margin> (ranks LLAMA_FN_L22_X_RESIDENT_R0=6 .. k-1). The graph form is in
// ggml/include/ggml-fn-l22-prior.h. The bridge (llama-moe-bridge.cpp) owns the state: init at its creation (before ready),
// free at its end; one bridge per process, so one state per process.
//   LLAMA_FN_L22_X_PRIOR_STATS=raw|used (default raw): the hot-set observer gets the unbiased top-k (raw, from the fused
//     post kernel; a prior job without them is not observed) or the picked ids (used)
//   LLAMA_FN_L22_X_PRIOR_PROMPT=1: also prompt graphs whose hot experts run on the device (bridged or short hot-chain
//     prompts); a prompt routing change stays in the KV cache. Off by default inside the opt-in

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <utility>
#include <vector>

struct llama_model;

struct llama_fn_prior_cfg {
    float lambda    = 0.0f;  // LLAMA_FN_L22_X_PRIOR (> 0: the prior)
    float margin    = 0.0f;  // LLAMA_FN_L22_X_RESIDENT (> 0: the resident swap)
    int   r0        = 6;     // LLAMA_FN_L22_X_RESIDENT_R0
    bool  stats_raw = true;  // LLAMA_FN_L22_X_PRIOR_STATS
    bool  prompt    = false; // LLAMA_FN_L22_X_PRIOR_PROMPT

    bool on() const { return lambda > 0.0f || margin > 0.0f; }
};

// the switches (environment first, then the model's profile; a null model reads the environment)
LLAMA_API llama_fn_prior_cfg llama_fn_prior_read(const llama_model * model);

// bridge creation, before ready: layers = (il, ffn_up_exps) of the bridged layers. true: the table is on dev and the levers
// act; false: off (nothing allocated, the reason logged)
LLAMA_API bool llama_fn_prior_init(const llama_model & model, ggml_backend_dev_t dev,
        const std::vector<std::pair<int, const ggml_tensor *>> & layers);
// no graph that reads the table runs any more
LLAMA_API void llama_fn_prior_free(void);
LLAMA_API bool llama_fn_prior_on(void);
// the fused post writes the unbiased top-k and the observer reads it (LLAMA_FN_L22_X_PRIOR_STATS=raw)
LLAMA_API bool llama_fn_prior_raw_wanted(void);
LLAMA_API llama_fn_prior_cfg llama_fn_prior_get_cfg(void);
// a prior graph's set_input (owner thread): the rows of the layers whose hot table changed since the last write, in one
// copy; sched (may be null) is synchronized before the copy
LLAMA_API void llama_fn_prior_refresh(ggml_backend_sched_t sched);

//
// pure parts (CPU unit tests)
//

// the two table rows of one layer: scale[e] = exp(lambda) and mask[e] = 1 where tbl[e] != miss (hot), else 1 and 0;
// tbl null: every expert cold
LLAMA_API void llama_fn_prior_fill_rows(const int32_t * tbl, int32_t miss, int n_exp, float lambda, float * scale, float * mask);

// sel = probs * row, marked GGML_FN_L22_PRIOR with the op params (ggml_fn_l22_prior_set)
LLAMA_API ggml_tensor * llama_fn_prior_build_sel(ggml_context * ctx, ggml_tensor * probs, ggml_tensor * row, int32_t flags,
        float expnm, int32_t r0, int32_t il);

// the CPU reference of the fused CUDA routing of one row: p = the router's softmax [n_exp] (NaN read as -FLT_MAX), the top-k
// on p * scale (ties: the lower expert), the resident swap (flags & GGML_FN_L22_PRIOR_F_RESIDENT, mask), the weights
// p_k / max(sum p_k, clamp) * wscale (with_norm) or p_k * wscale, summed in the kernel's order. raw (may be null): the
// unbiased top-k. scale / mask null: 1 / 0
LLAMA_API void llama_fn_prior_ref_select(const float * p, int n_exp, const float * scale, const float * mask, int k,
        int32_t flags, float expnm, int r0, bool with_norm, float clamp, float wscale, int32_t * ids, float * w, int32_t * raw);

// one job's counts: the distinct cold experts (tbl null: all; else tbl[e] == miss) of the picked ids and of the raw ids
// (raw null: 0), and the picked (token, slot) entries not in that token's raw top-k (raw null: 0). scratch: n_exp zero
// bytes, zero again on return
LLAMA_API void llama_fn_prior_job_counts(const int32_t * ids, const int32_t * raw, int n_tokens, int k, const int32_t * tbl,
        int32_t miss, int n_exp, uint8_t * scratch, uint32_t * cold_used, uint32_t * cold_raw, uint32_t * changed);
