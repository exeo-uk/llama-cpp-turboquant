#pragma once

// [TAG_FN_L22_DMASHARE] round 22 P14: the decode DMA share from pinned pages (D11). The layout the host plan
// (src/llama-fn-dmashare.cpp, the bridge's executor thread) and the device fetch (ggml/src/ggml-cuda/fn-l22-dmafetch.cu)
// share. All of it is mapped host memory or plain values: the executor makes no CUDA call.
//
// Per bridge channel (one bridged MoE layer):
//   side  the planned copies of the channel's current job: bank slot and, per part (up, gate, down), the source pieces.
//         A piece is a pinned range read through its device address, or a short unpinned edge the host staged into
//         the channel's edge area. Written before the bridge publishes the job's plan (release store); the device
//         reads it after the plan word, checks seq against the job's ticket.
//   done  written by the device when the job's copies have ended (every path, also nothing copied): the host commits
//         the bank slots only on a matching record.
//   edge  the staging area of the channel's unpinned edges (16-byte aligned pieces).
// The procs "ggml_backend_fn_l22_dma_new" / "ggml_backend_fn_l22_dma_free" of the CUDA backend reg make and drop the
// device side of one bridge (outside a running graph).

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_FN_L22_DMA_MAX_COPY 8    // experts one job copies at most
#define GGML_FN_L22_DMA_MAX_SEG  4    // pieces of one part: head edge, up to 2 pinned ranges, tail edge
#define GGML_FN_L22_DMA_EDGE_MAX 8192 // bytes of one staged edge
#define GGML_FN_L22_DMA_MAX_CHAN 128  // channels of one bridge
#define GGML_FN_L22_DMA_MAX_IDS  512  // routed (slot, token) entries of one job (n_used x n_tokens)

struct ggml_fn_l22_dma_seg {
    uint64_t src;    // device address of the first byte
    uint32_t len;
    uint32_t staged; // 1: src lies in the channel's edge area
};

struct ggml_fn_l22_dma_part {
    uint32_t n_seg;
    uint32_t pad[3];
    struct ggml_fn_l22_dma_seg seg[GGML_FN_L22_DMA_MAX_SEG];
};

struct ggml_fn_l22_dma_side {
    uint32_t seq;    // the job (bridge ticket) these copies belong to
    int32_t  n_copy;
    int32_t  slot[GGML_FN_L22_DMA_MAX_COPY];
    uint32_t pad[6];
    struct ggml_fn_l22_dma_part part[GGML_FN_L22_DMA_MAX_COPY][3];
};

struct ggml_fn_l22_dma_done {
    uint32_t seq;      // the job whose copies ended
    int32_t  n;        // experts copied
    uint32_t ok;       // 1: the plan was valid and every copy ran
    uint32_t pad0;
    uint64_t fetch_ns; // device time from the plan to the last copied byte (0: nothing copied)
    uint64_t pad[5];
};

// what the device side of one bridge is made for
struct ggml_fn_l22_dma_desc {
    int32_t  bridge_id;
    int32_t  device;
    int32_t  n_chan;      // channels 0 .. n_chan - 1 (<= GGML_FN_L22_DMA_MAX_CHAN)
    int32_t  max_copy;    // 1 .. GGML_FN_L22_DMA_MAX_COPY
    uint64_t edge_bytes;  // edge area per channel (a multiple of 64)
    int32_t  slot_lo[GGML_FN_L22_DMA_MAX_CHAN]; // the bank slots of the channel's layer: [slot_lo, slot_lo + slot_n)
    int32_t  slot_n [GGML_FN_L22_DMA_MAX_CHAN]; // 0: the channel never fetches
};

// host views of the mapped areas the device side made
struct ggml_fn_l22_dma_host {
    struct ggml_fn_l22_dma_side * side; // [n_chan]
    struct ggml_fn_l22_dma_done * done; // [n_chan]
    uint8_t *                     edge; // [n_chan][edge_bytes]
    uint64_t                      edge_dev; // device address of edge[0]
};

typedef void * (*ggml_backend_fn_l22_dma_new_t)(const struct ggml_fn_l22_dma_desc * desc, struct ggml_fn_l22_dma_host * host);
typedef void   (*ggml_backend_fn_l22_dma_free_t)(void * handle);

#if defined(__CUDACC__)
#    define GGML_FN_L22_DMA_HD __host__ __device__
#else
#    define GGML_FN_L22_DMA_HD
#endif

// one part's pieces against the part size and the channel's edge area [edge_lo, edge_hi): 1 good, 0 bad. The pieces are
// copied in order to consecutive bytes of the bank slot
static inline GGML_FN_L22_DMA_HD int ggml_fn_l22_dma_part_ok(const struct ggml_fn_l22_dma_part * p, uint64_t part_bytes,
        uint64_t edge_lo, uint64_t edge_hi) {
    if (p->n_seg < 1 || p->n_seg > GGML_FN_L22_DMA_MAX_SEG) {
        return 0;
    }
    uint64_t sum = 0;
    for (uint32_t s = 0; s < p->n_seg; ++s) {
        const struct ggml_fn_l22_dma_seg * g = &p->seg[s];
        if (g->len == 0 || g->src == 0) {
            return 0;
        }
        if (g->staged && (g->len > GGML_FN_L22_DMA_EDGE_MAX || g->src < edge_lo || g->src >= edge_hi || edge_hi - g->src < g->len)) {
            return 0;
        }
        sum += g->len;
    }
    return sum == part_bytes ? 1 : 0;
}

#undef GGML_FN_L22_DMA_HD

#ifdef __cplusplus
}

static_assert(sizeof(ggml_fn_l22_dma_seg)  == 16,  "ggml_fn_l22_dma_seg size");
static_assert(sizeof(ggml_fn_l22_dma_part) == 80,  "ggml_fn_l22_dma_part size");
static_assert(offsetof(ggml_fn_l22_dma_side, part) == 64, "ggml_fn_l22_dma_side head");
static_assert(sizeof(ggml_fn_l22_dma_side) % 64 == 0, "ggml_fn_l22_dma_side size");
static_assert(sizeof(ggml_fn_l22_dma_done) == 64,  "ggml_fn_l22_dma_done size");
#endif
