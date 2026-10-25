#include "common.cuh"

#define CUDA_POOL1D_BLOCK_SIZE 256

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// [TAG_FLINT_AMAX_FUSION] the MoE down-projection rescale of build_amax_src1_f16 (llama-graph.cpp) in one launch:
// ABS -> POOL_1D(MAX over the row) -> SCALE -> LOG -> SCALE -> CEIL -> CLAMP -> SCALE -> EXP, then DIV(x, e).
// Writes e (the EXP node, [1, ne1, ne2, ne3]) and the DIV node; the same float ops in the same order as the nodes.
struct ggml_cuda_amax_src1_params {
    float s0, b0;     // SCALE before LOG
    float s1, b1;     // SCALE after LOG
    float cmin, cmax; // CLAMP
    float s2, b2;     // SCALE before EXP
};

void ggml_cuda_op_amax_src1_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * e,
                                  ggml_tensor * dst, const ggml_cuda_amax_src1_params & p);
