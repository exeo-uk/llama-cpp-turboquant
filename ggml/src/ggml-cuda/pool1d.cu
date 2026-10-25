#include "pool1d.cuh"

static __global__ void pool1d_nchw_kernel(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nc * iw;
    float * o_ptr = dst + nc * ow;

    const int start = cur_ow * sw - pw;
    const int b = max(0, start);
    const int e = min(iw, start + kw);

    float res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: return;
    }

    int count = 0;
    for (int i = b; i < e; i++) {
#if __CUDA_ARCH__ >= 350
        float cur = __ldg(i_ptr + i);
#else
        float cur = i_ptr[i];
#endif
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur;                break;
            case GGML_OP_POOL_MAX: res = max(res, cur);       break;
            default: break;
        }
        count++;
    }

    if (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : 0.0f;
    }

    o_ptr[cur_ow] = res;
}

static void pool1d_nchw_kernel_f32_f32_cuda(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {
    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool1d_nchw_kernel<<<block_nums, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, kw, sw, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const int parallel_elements = (int)(nr * OW);

    pool1d_nchw_kernel_f32_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}

// [TAG_FLINT_AMAX_FUSION] one block per row of x: the row's max |x| (POOL_1D MAX of ABS: fmaxf is exact, so the order of
// the reduction does not change the value), the factor chain of the unfused SCALE / LOG / CEIL / CLAMP / EXP nodes with
// the same float ops (logf / expf / ceilf as in unary.cu, fminf(fmaxf()) as in clamp.cu, s*x + b as in scale.cu), then
// x / e (binbcast DIV). Every value of the row is read before the first write, so dst may be x itself (in place).
template <int block_size>
static __global__ void amax_src1_f32_kernel(const float * x, float * e_out, float * dst, const int ncols,
        const ggml_cuda_amax_src1_params p) {
    const int64_t row = blockIdx.x;
    const float * xr = x   + row*ncols;
    float       * dr = dst + row*ncols;

    float amax = -FLT_MAX;
    for (int c = threadIdx.x; c < ncols; c += block_size) {
        amax = fmaxf(amax, fabsf(xr[c]));
    }
    amax = warp_reduce_max(amax);
    if (block_size > WARP_SIZE) {
        __shared__ float s_max[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_max[warp_id] = amax;
        }
        __syncthreads();
        amax = lane_id < block_size/WARP_SIZE ? s_max[lane_id] : -FLT_MAX;
        amax = warp_reduce_max(amax);
    }

    float v = p.s0*amax + p.b0;
    v = logf(v);
    v = p.s1*v + p.b1;
    v = ceilf(v);
    v = fminf(fmaxf(v, p.cmin), p.cmax);
    v = p.s2*v + p.b2;
    v = expf(v);

    if (threadIdx.x == 0) {
        e_out[row] = v;
    }
    __syncthreads(); // every thread has read its part of the row (the reduction) before any write below
    for (int c = threadIdx.x; c < ncols; c += block_size) {
        dr[c] = xr[c] / v;
    }
}

void ggml_cuda_op_amax_src1_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * e,
                                  ggml_tensor * dst, const ggml_cuda_amax_src1_params & p) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && e->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(e) && ggml_is_contiguous(dst));
    const int     ncols = (int) x->ne[0];
    const int64_t nrows = ggml_nrows(x);
    GGML_ASSERT(ggml_nelements(e) == nrows && ggml_nelements(dst) == ggml_nelements(x));
    cudaStream_t stream = ctx.stream();
    if (ncols <= 1024) {
        amax_src1_f32_kernel<128><<<(unsigned) nrows, 128, 0, stream>>>((const float *) x->data, (float *) e->data,
                (float *) dst->data, ncols, p);
    } else {
        amax_src1_f32_kernel<256><<<(unsigned) nrows, 256, 0, stream>>>((const float *) x->data, (float *) e->data,
                (float *) dst->data, ncols, p);
    }
}
