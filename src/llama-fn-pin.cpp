// [TAG_FN_L22_PIN] round 22 P6: pinned (registered) host ranges of the streamed experts, the DXGI budget helper and the
// out-of-memory unpin hook, src/llama-fn-l22.h. P0 stub: no pins.

#include "llama-fn-l22.h"

void llama_fn_pin_on_ready(const llama_model * model) {
    (void) model;
}

bool llama_fn_pin_query(const void * p, size_t n, llama_fn_pin_seg * seg) {
    if (seg == nullptr) {
        return false;
    }
    seg->src    = p;
    seg->len    = n;
    seg->pinned = false;
    seg->ticket = -1;
    return true;
}

void llama_fn_pin_release(int32_t ticket) {
    (void) ticket;
}

bool llama_fn_pin_on_oom(size_t need_bytes) {
    (void) need_bytes;
    return false;
}

bool llama_fn_pin_active(void) {
    return false;
}

bool llama_fn_dxgi_query(uint64_t luid, int non_local, uint64_t * budget, uint64_t * usage) {
    (void) luid; (void) non_local;
    if (budget) {
        *budget = 0;
    }
    if (usage) {
        *usage = 0;
    }
    return false;
}

void llama_fn_pin_decode_mark(bool in_decode) {
    (void) in_decode;
}

void llama_fn_pin_note_need(const llama_model * model, int kind, size_t bytes) {
    (void) model; (void) kind; (void) bytes;
}

void llama_fn_pin_shutdown(const llama_model * model) {
    (void) model;
}
