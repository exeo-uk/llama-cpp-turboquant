// [TAG_FN_L22_SSDTIER] round 22 P5: direct (unbuffered) reads of expert bytes, src/llama-fn-l22.h.
// P0 stub: no direct I/O, the caller reads the mapped bytes.

#include "llama-fn-l22.h"

int llama_fn_dio_submit(const llama_model * model, const llama_fn_dio_req * req, int n) {
    (void) model; (void) req; (void) n;
    return -1;
}

bool llama_fn_dio_wait(int ticket) {
    (void) ticket;
    return false;
}
