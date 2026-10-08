#pragma once

#include "common.cuh"

#include <cstdint>

// [TAG_UPSTREAM_29353] one-kernel chunked GDN prefill (upstream #29353): C = 16 token chunks, BF16 hi/lo split MMA
// operands, FP32 state. Strides in floats. The kernel reads tokens [0, n_tokens) and writes dst rows with a per-seq
// stride of n_tokens tokens, so a caller that runs only a token prefix must have n_seqs == 1.
struct ggml_cuda_gdn_mma_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;
};

// Returns false (nothing launched) when the shape, the device or the build does not take the MMA kernel.
bool ggml_cuda_gdn_mma_launch(int device, const ggml_cuda_gdn_mma_args & args, cudaStream_t stream);
