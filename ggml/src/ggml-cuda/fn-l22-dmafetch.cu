// [TAG_FN_L22_DMASHARE] round 22 P14, see fn-l22-dmafetch.cuh and ggml/include/ggml-fn-l22-dma.h.
//
// The bridged decode graph's GGML_OP_MOE_HOST_FETCH for a bridge with P14's device side: two launches on the compute
// stream. k_l22_plan waits (bounded, exactly as the ring fetch: plan word, done word, sticky error, release, timeout /
// job_max with taken) for the host's plan, copies the channel's side table and the plan's slot ids (16-byte words, all
// threads), checks them (seq, counts, slots inside the layer's range, pieces against the part sizes and the edge area)
// and writes the bank slot ids; a bad plan sets the sticky runner error and nothing is copied. k_l22_copy reads every
// planned piece (pinned pages through their device address, or the staged edges) with ld.global.cv into the bank slot;
// the last block writes the channel's done record on every path. No wait in the copy kernel. Every kernel argument is
// fixed at creation (captured CUDA graphs replay them); the plan is read at run time.

#include "fn-l22-dmafetch.cuh"
#include "moe-bridge.cuh"

#include "ggml-cuda.h"
#include "ggml-fn-l22-dma.h"
#include "ggml-impl.h"
#include "ggml-moe-bridge.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#define GGML_FN_L22_DMA_DISABLED // needs %globaltimer, __nanosleep and __ldcv
#endif

namespace {

constexpr int L22_MAX_BRIDGES = 8; // MB_MAX_BRIDGES of moe-bridge.cu
constexpr int L22_PLAN_THREADS = 128;
constexpr int L22_COPY_BLOCKS  = 48;
constexpr int L22_COPY_THREADS = 256;

// the plan of a channel's current job, in device memory (the copy blocks read it there)
struct l22_chan_dev {
    uint32_t           seq;
    int32_t            ok;
    int32_t            n_copy;
    uint32_t           done_blocks;
    unsigned long long t0;
    int32_t            slot[GGML_FN_L22_DMA_MAX_COPY];
    ggml_fn_l22_dma_part part[GGML_FN_L22_DMA_MAX_COPY][3];
};

struct l22_state {
    ggml_fn_l22_dma_desc   desc;
    uint8_t *              host       = nullptr; // the mapped block: side | done | edge
    uint8_t *              host_dev   = nullptr;
    size_t                 host_bytes = 0;
    ggml_fn_l22_dma_side * side_dev   = nullptr;
    ggml_fn_l22_dma_done * done_dev   = nullptr;
    uint8_t *              edge_dev   = nullptr;
    l22_chan_dev *         chan_dev   = nullptr; // device memory
};

std::mutex               g_l22_mtx;
std::atomic<l22_state *> g_l22[L22_MAX_BRIDGES];

l22_state * l22_get(int32_t bid) {
    if (bid < 0 || bid >= L22_MAX_BRIDGES) {
        return nullptr;
    }
    return g_l22[bid].load(std::memory_order_acquire);
}

size_t l22_pad(size_t n) {
    return (n + 63) & ~(size_t) 63;
}

void l22_destroy(l22_state * s) {
    if (s == nullptr) {
        return;
    }
    ggml_cuda_set_device(s->desc.device);
    if (s->chan_dev != nullptr) {
        (void) cudaFree(s->chan_dev);
    }
    if (s->host != nullptr) {
        (void) cudaFreeHost(s->host);
    }
    (void) cudaGetLastError();
    delete s;
}

void * l22_new(const ggml_fn_l22_dma_desc * d, ggml_fn_l22_dma_host * h) {
#ifdef GGML_FN_L22_DMA_DISABLED
    GGML_UNUSED(d);
    GGML_UNUSED(h);
    return nullptr;
#else
    if (d == nullptr || h == nullptr || d->bridge_id < 0 || d->bridge_id >= L22_MAX_BRIDGES || d->device < 0 ||
        d->device >= ggml_backend_cuda_get_device_count() || d->n_chan < 1 || d->n_chan > GGML_FN_L22_DMA_MAX_CHAN ||
        d->max_copy < 1 || d->max_copy > GGML_FN_L22_DMA_MAX_COPY || d->edge_bytes % 64 != 0 || d->edge_bytes > (1u << 20)) {
        GGML_LOG_ERROR("%s: invalid parameters\n", __func__);
        return nullptr;
    }
    for (int c = 0; c < d->n_chan; ++c) {
        if (d->slot_lo[c] < 0 || d->slot_n[c] < 0 || d->slot_n[c] > 64) {
            GGML_LOG_ERROR("%s: invalid slot range of channel %d\n", __func__, c);
            return nullptr;
        }
    }
    int can_map = 0;
    if (cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, d->device) != cudaSuccess || !can_map) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: device %d cannot map host memory\n", __func__, d->device);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_l22_mtx);
    if (g_l22[d->bridge_id].load() != nullptr) {
        GGML_LOG_WARN("%s: bridge %d already has a DMA share fetch\n", __func__, d->bridge_id);
        return nullptr;
    }
    ggml_cuda_set_device(d->device);

    auto * s = new l22_state();
    s->desc = *d;
    const size_t off_done = l22_pad((size_t) d->n_chan*sizeof(ggml_fn_l22_dma_side));
    const size_t off_edge = off_done + l22_pad((size_t) d->n_chan*sizeof(ggml_fn_l22_dma_done));
    s->host_bytes = off_edge + (size_t) d->n_chan*d->edge_bytes;

    auto fail = [&](const char * what, cudaError_t rc) -> void * {
        GGML_LOG_WARN("%s: %s failed: %s\n", __func__, what, cudaGetErrorString(rc));
        (void) cudaGetLastError();
        l22_destroy(s);
        return nullptr;
    };

    cudaError_t rc = cudaHostAlloc((void **) &s->host, s->host_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (rc != cudaSuccess) {
        s->host = nullptr;
        return fail("cudaHostAlloc", rc);
    }
    memset(s->host, 0, s->host_bytes);
    rc = cudaHostGetDevicePointer((void **) &s->host_dev, s->host, 0);
    if (rc != cudaSuccess) {
        return fail("cudaHostGetDevicePointer", rc);
    }
    rc = cudaMalloc((void **) &s->chan_dev, (size_t) d->n_chan*sizeof(l22_chan_dev));
    if (rc != cudaSuccess) {
        s->chan_dev = nullptr;
        return fail("cudaMalloc", rc);
    }
    {
        cudaStream_t st = nullptr;
        rc = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        if (rc == cudaSuccess) {
            rc = cudaMemsetAsync(s->chan_dev, 0, (size_t) d->n_chan*sizeof(l22_chan_dev), st);
            if (rc == cudaSuccess) {
                rc = cudaStreamSynchronize(st);
            }
            (void) cudaStreamDestroy(st);
        }
        if (rc != cudaSuccess) {
            return fail("clearing the device plans", rc);
        }
    }
    s->side_dev = (ggml_fn_l22_dma_side *) s->host_dev;
    s->done_dev = (ggml_fn_l22_dma_done *) (s->host_dev + off_done);
    s->edge_dev = s->host_dev + off_edge;

    h->side     = (ggml_fn_l22_dma_side *) s->host;
    h->done     = (ggml_fn_l22_dma_done *) (s->host + off_done);
    h->edge     = s->host + off_edge;
    h->edge_dev = (uint64_t) (uintptr_t) s->edge_dev;
    g_l22[d->bridge_id].store(s, std::memory_order_release);

    GGML_LOG_INFO("%s: bridge %d on device %d: DMA share fetch for %d channels, <= %d copies per job, %.2f MiB mapped\n", __func__,
            d->bridge_id, d->device, d->n_chan, d->max_copy, s->host_bytes/1048576.0);
    return s;
#endif
}

void l22_free(void * handle) {
    if (handle == nullptr) {
        return;
    }
    auto * s = (l22_state *) handle;
    {
        std::lock_guard<std::mutex> lk(g_l22_mtx);
        const int32_t bid = s->desc.bridge_id;
        if (bid >= 0 && bid < L22_MAX_BRIDGES && g_l22[bid].load() == s) {
            g_l22[bid].store(nullptr, std::memory_order_release);
        }
    }
    l22_destroy(s);
}

} // namespace

#ifndef GGML_FN_L22_DMA_DISABLED

static __device__ __forceinline__ unsigned long long l22_now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ void l22_sleep() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(256);
#endif
}

// the sticky bridge error and its first channel / job (as the ring fetch sets them)
static __device__ void l22_set_err(const ggml_cuda_mb_fetch_view & fv, const int chan, const uint32_t want, const uint32_t code) {
    if (atomicCAS(fv.err_state, 0u, code) == 0u) {
        *(volatile int32_t  *) fv.g_err_chan = chan;
        *(volatile uint32_t *) fv.g_err_seq  = want;
        __threadfence_system();
        *(volatile uint32_t *) fv.g_err = code;
        __threadfence_system();
    }
}

static __global__ void k_l22_plan(const ggml_cuda_mb_fetch_view fv, const int chan, const ggml_fn_l22_dma_side * __restrict__ side,
        l22_chan_dev * __restrict__ cd, const int32_t * __restrict__ ticket, int32_t * __restrict__ dst, const int n_ids,
        const int zero_slot, const int slot_lo, const int slot_n, const int max_copy,
        const unsigned long long pb0, const unsigned long long pb1, const unsigned long long pb2,
        const unsigned long long edge_lo, const unsigned long long edge_hi) {
    __shared__ int      s_ok;
    __shared__ uint32_t s_want;
    __shared__ int32_t  s_pn;
    __shared__ int32_t  s_pslot[GGML_FN_L22_DMA_MAX_COPY];
    __shared__ int32_t  s_ids[GGML_FN_L22_DMA_MAX_IDS];
    __shared__ uint4    s_side[sizeof(ggml_fn_l22_dma_side)/16];
    const int tid = threadIdx.x;
    if (tid == 0) {
        const uint32_t want = (uint32_t) ticket[0];
        int ok = 0;
        if (want != 0) {
            const volatile uint32_t * plan  = fv.plan_word;
            const volatile uint32_t * done  = fv.done_word;
            const volatile uint32_t * taken = fv.taken_word;
            const volatile uint32_t * err   = fv.err_state;
            const volatile uint32_t * rel   = fv.release;
            const unsigned long long t0 = l22_now_ns();
            for (;;) {
                if ((int32_t) (*plan - want) >= 0) {
                    ok = 1;
                    break;
                }
                if ((int32_t) (*done - want) >= 0) {
                    // answered: a plan published before done is visible after this fence; none = a failed job
                    __threadfence_system();
                    ok = (int32_t) (*plan - want) >= 0 ? 1 : 0;
                    break;
                }
                if (*err != 0 || *rel != 0) {
                    break;
                }
                const unsigned long long el = l22_now_ns() - t0;
                if (el > fv.timeout_ns && ((int32_t) (*taken - want) < 0 || el > fv.job_max_ns)) {
                    l22_set_err(fv, chan, want, GGML_MOE_BRIDGE_ERR_TIMEOUT);
                    break;
                }
                l22_sleep();
            }
            __threadfence_system(); // acquire: the host wrote the side table and the plan before the plan word
        }
        s_ok   = ok;
        s_want = want;
    }
    __syncthreads();
    if (s_ok) {
        const uint4 * src = (const uint4 *) side;
        for (int i = tid; i < (int) (sizeof(ggml_fn_l22_dma_side)/16); i += blockDim.x) {
            s_side[i] = __ldcv(src + i);
        }
        for (int i = tid; i < n_ids; i += blockDim.x) {
            s_ids[i] = __ldcv(&fv.plan->slot_ids[i]);
        }
        if (tid < GGML_FN_L22_DMA_MAX_COPY) {
            s_pslot[tid] = __ldcv(&fv.plan->slot[tid]);
        }
        if (tid == 0) {
            s_pn = __ldcv(&fv.plan->n_copy);
        }
    }
    __syncthreads();
    if (tid == 0) {
        int ok = s_ok;
        int n  = 0;
        if (ok) {
            const ggml_fn_l22_dma_side * sd = (const ggml_fn_l22_dma_side *) s_side;
            const unsigned long long pb[3] = { pb0, pb1, pb2 };
            n = sd->n_copy;
            bool good = sd->seq == s_want && n >= 0 && n <= max_copy && n == s_pn;
            for (int i = 0; good && i < n; ++i) {
                const int32_t sl = sd->slot[i];
                good = sl >= slot_lo && sl < slot_lo + slot_n && sl < zero_slot && s_pslot[i] == sl;
                for (int k = 0; good && k < i; ++k) {
                    good = sd->slot[k] != sl;
                }
                for (int p = 0; good && p < 3; ++p) {
                    good = ggml_fn_l22_dma_part_ok(&sd->part[i][p], pb[p], edge_lo, edge_hi) != 0;
                }
            }
            for (int i = 0; good && i < n_ids; ++i) {
                const int32_t id = s_ids[i];
                good = id == zero_slot || (id >= slot_lo && id < slot_lo + slot_n);
            }
            if (!good) {
                l22_set_err(fv, chan, s_want, GGML_MOE_BRIDGE_ERR_RUNNER);
                ok = 0;
                n  = 0;
            } else {
                for (int i = 0; i < n; ++i) {
                    cd->slot[i] = sd->slot[i];
                    for (int p = 0; p < 3; ++p) {
                        cd->part[i][p] = sd->part[i][p];
                    }
                }
            }
        }
        cd->n_copy      = n;
        cd->ok          = ok;
        cd->seq         = s_want;
        cd->done_blocks = 0;
        cd->t0          = l22_now_ns();
        s_ok = ok;
    }
    __syncthreads();
    for (int i = tid; i < n_ids; i += blockDim.x) {
        dst[i] = s_ok ? s_ids[i] : zero_slot;
    }
}

// blockIdx.z: the planned expert, blockIdx.y: its part (up, gate, down); the blocks of x stride over the part's pieces
static __global__ void k_l22_copy(l22_chan_dev * __restrict__ cd,
        char * __restrict__ d0, const long long nb0, char * __restrict__ d1, const long long nb1,
        char * __restrict__ d2, const long long nb2, ggml_fn_l22_dma_done * __restrict__ done) {
    const int e = blockIdx.z;
    const int n = cd->n_copy;
    if (e < n) {
        const int part = blockIdx.y;
        const ggml_fn_l22_dma_part * P = &cd->part[e][part];
        const int32_t sl  = cd->slot[e];
        char *        dst = part == 0 ? d0 + sl*nb0 : part == 1 ? d1 + sl*nb1 : d2 + sl*nb2;
        const long long stride = (long long) gridDim.x*blockDim.x;
        const long long i0     = (long long) blockIdx.x*blockDim.x + threadIdx.x;
        const uint32_t  n_seg  = P->n_seg;
        long long off = 0;
        for (uint32_t s = 0; s < n_seg && s < GGML_FN_L22_DMA_MAX_SEG; ++s) {
            const char *    src = (const char *) (uintptr_t) P->seg[s].src;
            const long long len = (long long) P->seg[s].len;
            char *          t   = dst + off;
            const uintptr_t a   = (uintptr_t) src | (uintptr_t) t | (uintptr_t) len;
            if ((a & 15) == 0) {
                const uint4 * s4 = (const uint4 *) src;
                uint4       * t4 = (uint4 *) t;
                for (long long i = i0; i < len/16; i += stride) {
                    t4[i] = __ldcv(s4 + i);
                }
            } else if ((a & 3) == 0) {
                const unsigned int * s1 = (const unsigned int *) src;
                unsigned int       * t1 = (unsigned int *) t;
                for (long long i = i0; i < len/4; i += stride) {
                    t1[i] = __ldcv(s1 + i);
                }
            } else {
                for (long long i = i0; i < len; i += stride) {
                    t[i] = __ldcv(src + i);
                }
            }
            off += len;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned int total = gridDim.x*gridDim.y*gridDim.z;
        if (atomicAdd(&cd->done_blocks, 1u) == total - 1u) {
            volatile ggml_fn_l22_dma_done * dn = done;
            dn->n        = n;
            dn->ok       = (uint32_t) cd->ok;
            dn->fetch_ns = n > 0 ? l22_now_ns() - cd->t0 : 0ull;
            __threadfence_system();
            dn->seq      = cd->seq;
            __threadfence_system();
        }
    }
}

#endif // GGML_FN_L22_DMA_DISABLED

bool ggml_cuda_fn_l22_dmafetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_FN_L22_DMA_DISABLED
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#else
    const int32_t bid  = ggml_get_op_params_i32(dst, 0);
    const int32_t chan = ggml_get_op_params_i32(dst, 1);
    l22_state * s = l22_get(bid);
    if (s == nullptr) {
        return false; // not P14's bridge: the ring fetch
    }
    const ggml_tensor * up   = dst->src[1];
    const ggml_tensor * gate = dst->src[2];
    const ggml_tensor * down = dst->src[3];
    const int64_t n_ids = ggml_nelements(dst);
    ggml_cuda_mb_fetch_view fv;
    if (s->desc.device != ctx.device || chan < 0 || chan >= s->desc.n_chan || s->desc.slot_n[chan] <= 0 || n_ids < 1 ||
        n_ids > GGML_FN_L22_DMA_MAX_IDS || !ggml_cuda_moe_bridge_fetch_view(bid, ctx.device, chan, &fv)) {
        GGML_ABORT("[TAG_FN_L22_DMASHARE] fetch of bridge %d channel %d on device %d without its DMA share state", bid, chan, ctx.device);
    }
    const int zero_slot = (int) up->ne[2] - 1;
    GGML_ASSERT(s->desc.slot_lo[chan] + s->desc.slot_n[chan] <= zero_slot);
    const unsigned long long edge_lo = (unsigned long long) (uintptr_t) (s->edge_dev + (size_t) chan*s->desc.edge_bytes);
    const unsigned long long edge_hi = edge_lo + s->desc.edge_bytes;

    cudaStream_t stream = ctx.stream();
    k_l22_plan<<<1, L22_PLAN_THREADS, 0, stream>>>(fv, chan, s->side_dev + chan, s->chan_dev + chan,
            (const int32_t *) dst->src[0]->data, (int32_t *) dst->data, (int) n_ids, zero_slot, s->desc.slot_lo[chan],
            s->desc.slot_n[chan], s->desc.max_copy, (unsigned long long) up->nb[2], (unsigned long long) gate->nb[2],
            (unsigned long long) down->nb[2], edge_lo, edge_hi);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(L22_COPY_BLOCKS, 3, (unsigned) s->desc.max_copy);
    k_l22_copy<<<grid, L22_COPY_THREADS, 0, stream>>>(s->chan_dev + chan,
            (char *) up->data, (long long) up->nb[2], (char *) gate->data, (long long) gate->nb[2],
            (char *) down->data, (long long) down->nb[2], s->done_dev + chan);
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}

bool ggml_cuda_fn_l22_dmafetch_supported(int device, const ggml_tensor * op) {
#ifdef GGML_FN_L22_DMA_DISABLED
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    return false;
#else
    if (op == nullptr || op->op != GGML_OP_MOE_HOST_FETCH) {
        return false;
    }
    const int32_t bid  = ggml_get_op_params_i32(op, 0);
    const int32_t chan = ggml_get_op_params_i32(op, 1);
    const l22_state * s = l22_get(bid);
    if (s == nullptr || s->desc.device != device || chan < 0 || chan >= s->desc.n_chan || s->desc.slot_n[chan] <= 0) {
        return false;
    }
    const ggml_tensor * up = op->src[1];
    if (op->src[0] == nullptr || up == nullptr || op->src[2] == nullptr || op->src[3] == nullptr || op->type != GGML_TYPE_I32 ||
        op->ne[1] < 1 || op->ne[1] > 8 || op->ne[2] != 1 || op->ne[3] != 1 || op->ne[0]*op->ne[1] > GGML_FN_L22_DMA_MAX_IDS ||
        s->desc.slot_lo[chan] + s->desc.slot_n[chan] > up->ne[2] - 1) {
        return false;
    }
    ggml_cuda_mb_fetch_view fv;
    return ggml_cuda_moe_bridge_fetch_view(bid, device, chan, &fv);
#endif
}

void * ggml_cuda_fn_l22_dmafetch_proc(const char * name) {
#ifdef GGML_FN_L22_DMA_DISABLED
    GGML_UNUSED(name);
    return nullptr;
#else
    if (name == nullptr) {
        return nullptr;
    }
    if (strcmp(name, "ggml_backend_fn_l22_dma_new") == 0) {
        return (void *) l22_new;
    }
    if (strcmp(name, "ggml_backend_fn_l22_dma_free") == 0) {
        return (void *) l22_free;
    }
    return nullptr;
#endif
}
