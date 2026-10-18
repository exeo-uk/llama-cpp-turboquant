// [TAG_FN_L22_DMASHARE] round 22 P14: the decode DMA share from pinned pages, src/llama-fn-l22.h.
// P0 stub: never wanted, no banks, no plan.

#include "llama-fn-l22.h"

bool llama_fn_dmashare_wanted(const llama_model * model, const llama_moe_bridge * br) {
    (void) model; (void) br;
    return false;
}

const llama_moe_dma_view * llama_fn_dmashare_view(const llama_moe_bridge * br, const ggml_tensor * up_exps, int64_t n_tokens) {
    (void) br; (void) up_exps; (void) n_tokens;
    return nullptr;
}

bool llama_fn_dmashare_plan(llama_fn_dmashare_job * jv, int32_t * skip_table) {
    (void) jv; (void) skip_table;
    return false;
}

void llama_fn_dmashare_job_done(const llama_moe_bridge * br, int32_t chan, uint32_t seq, uint64_t cpu_ns, uint32_t n_cpu) {
    (void) br; (void) chan; (void) seq; (void) cpu_ns; (void) n_cpu;
}

void llama_fn_dmashare_graph_end(const llama_moe_bridge * br, uint32_t err, double dev_wait_us_per_layer) {
    (void) br; (void) err; (void) dev_wait_us_per_layer;
}

void llama_fn_dmashare_free(const llama_moe_bridge * br) {
    (void) br;
}
