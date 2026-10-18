// [TAG_FN_L22_DMASHARE] round 22 P14, see fn-l22-dmafetch.cuh. P0 stub: never takes the op.

#include "fn-l22-dmafetch.cuh"

bool ggml_cuda_fn_l22_dmafetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_fn_l22_dmafetch_supported(int device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    return false;
}

void * ggml_cuda_fn_l22_dmafetch_proc(const char * name) {
    GGML_UNUSED(name);
    return nullptr;
}
