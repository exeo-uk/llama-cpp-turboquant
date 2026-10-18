#pragma once

// [TAG_FN_L22_X_PRIOR] round 22 P12 (qwen4exp, opt-in, changes answers): the route prior and the resident swap.
//
// In a decode graph (fn_gen) with LLAMA_FN_L22_X_PRIOR=<lambda> or LLAMA_FN_L22_X_RESIDENT=<margin>, the router's top-k of a
// bridged MoE layer runs on sel = probs * scale (a MUL marked here), scale = exp(lambda) for the experts in the VRAM hot set
// and 1 for the rest; the weights still come from the unscaled probs (the get_rows reads probs). The MUL's src[1] is row 0
// of layer il of an F32 table [n_expert, 2, n_layer]; row 1 of that layer is the hot mask (1 hot, 0 not). Every backend
// computes the plain MUL, so the prior is exact everywhere; the CUDA bridge code (moe-bridge.cu) fuses the pattern
// SOFT_MAX, RESHAPE, marked MUL, ARGSORT, VIEW, GET_ROWS [, norm] [, SCALE] into one launch (with the bridge post behind
// it when the post is marked GGML_FN_L4_POST) and only there applies the resident swap: for ranks r0..k-1, a cold pick is
// replaced by the most probable hot expert not yet picked when p_hot >= p_cold * exp(-margin).

#include "ggml-fn-l3.h"

#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// the mark (op_params slot GGML_FN_L3_SLOT of the MUL)
#define GGML_FN_L22_PRIOR 0x4C323201

// int32 op_params slots of the marked MUL (slots 13-15 are fork marks: L6, L4 QSA, L3)
#define GGML_FN_L22_PRIOR_SLOT_FLAGS 8  // GGML_FN_L22_PRIOR_F_*
#define GGML_FN_L22_PRIOR_SLOT_EXPNM 9  // f32 exp(-margin) of the resident swap
#define GGML_FN_L22_PRIOR_SLOT_R0    10 // first rank the resident swap may replace
#define GGML_FN_L22_PRIOR_SLOT_IL    11 // the layer

#define GGML_FN_L22_PRIOR_F_PRIOR    1  // the top-k runs on probs * scale
#define GGML_FN_L22_PRIOR_F_RESIDENT 2  // the resident swap after the top-k (fused path only)
#define GGML_FN_L22_PRIOR_F_RAW      4  // the fused post also writes the unbiased top-k to the channel (hot-set observer)

static inline void ggml_fn_l22_prior_set(struct ggml_tensor * t, int32_t flags, float expnm, int32_t r0, int32_t il) {
    t->op_params[GGML_FN_L22_PRIOR_SLOT_FLAGS] = flags;
    memcpy(&t->op_params[GGML_FN_L22_PRIOR_SLOT_EXPNM], &expnm, sizeof(float));
    t->op_params[GGML_FN_L22_PRIOR_SLOT_R0] = r0;
    t->op_params[GGML_FN_L22_PRIOR_SLOT_IL] = il;
    ggml_fn_l3_set(t, GGML_FN_L22_PRIOR);
}

// false (outputs untouched) unless t is a MUL with the mark
static inline bool ggml_fn_l22_prior_get(const struct ggml_tensor * t, int32_t * flags, float * expnm, int32_t * r0, int32_t * il) {
    if (t == NULL || t->op != GGML_OP_MUL || ggml_fn_l3_get(t) != GGML_FN_L22_PRIOR) {
        return false;
    }
    if (flags) {
        *flags = t->op_params[GGML_FN_L22_PRIOR_SLOT_FLAGS];
    }
    if (expnm) {
        memcpy(expnm, &t->op_params[GGML_FN_L22_PRIOR_SLOT_EXPNM], sizeof(float));
    }
    if (r0) {
        *r0 = t->op_params[GGML_FN_L22_PRIOR_SLOT_R0];
    }
    if (il) {
        *il = t->op_params[GGML_FN_L22_PRIOR_SLOT_IL];
    }
    return true;
}

#ifdef __cplusplus
}
#endif
