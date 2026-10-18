// [TAG_FN_L22_CPUSHARE] round 22 P16: the CPU share of 768-3K-token streamed prompt ubatches, src/llama-fn-l22.h.
// P0 stub: no share (the stream computes every expert).

#include "llama-fn-l22.h"

bool llama_fn_cpushare_plan(llm_graph_context * g, const llama_ubatch & ubatch, int il, void * plan) {
    (void) g; (void) ubatch; (void) il; (void) plan;
    return false;
}

ggml_tensor * llama_fn_cpushare_finish(llm_graph_context * g, void * plan, ggml_tensor * moe_out) {
    (void) g; (void) plan;
    return moe_out;
}
