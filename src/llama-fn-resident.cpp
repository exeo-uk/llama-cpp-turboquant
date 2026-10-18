// [TAG_FN_L22_RESSET] round 22 P5: the per-expert residency table (VRAM / RAM / PIN / SSD), src/llama-fn-l22.h.
// P0 stub: every expert is RAM, the table never changes, no stream bytes are known.

#include "llama-fn-l22.h"

uint8_t llama_fn_res_class(const llama_model * model, int il, int e) {
    (void) model; (void) il; (void) e;
    return LLAMA_FN_RES_RAM;
}

uint64_t llama_fn_res_gen(const llama_model * model) {
    (void) model;
    return 0;
}

void llama_fn_res_mark_pin(const llama_model * model, int il, int e, bool on) {
    (void) model; (void) il; (void) e; (void) on;
}

uint64_t llama_fn_res_stream_bytes(const llama_model * model, uint8_t cls) {
    (void) model; (void) cls;
    return 0;
}
