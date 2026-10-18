#pragma once

// [TAG_FN_L22_DMASHARE] round 22 P14: the bridged decode graph's fetch (GGML_OP_MOE_HOST_FETCH) from pinned pages
// instead of the gen5 ring. The fetch op calls it first; true = it did the op (bank slots written), false = the ring fetch.

#include "common.cuh"

bool ggml_cuda_fn_l22_dmafetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// R-P14-2 (b): the fetch op is supported on this device without a gen5 ring (P14's state for the op's bridge exists)
bool ggml_cuda_fn_l22_dmafetch_supported(int device, const ggml_tensor * op);

// R-P14-4: P14's procs, looked up through ggml_backend_cuda_reg_get_proc_address (nullptr: not P14's name)
void * ggml_cuda_fn_l22_dmafetch_proc(const char * name);
