#pragma once

// [TAG_FN_L22_TBO] round 22 D5: the pure parts of the two-chain decode graph ("two-batch overlap"). A decode ubatch of
// exactly two sequences (verify / generation rows, every row an output) is built as two row chains, one per sequence, in
// one graph. The node order lets the CPU pool compute one chain's cold experts while the GPU runs the other chain's
// half-layer. No model and no backend: the qwen4exp graph, the MoE bridge and tests/test-fn-l22-tbo.cpp use these (the
// row views at the end build ggml view nodes only).
//
// LLAMA_FN_L22_TBO (qwen4exp profile lookup): 0 / unset off, 1 interleaved chains, 2 the same chains one after the other
// (debug: the same bits as 1, which shows the interleave has no read-before-write hazard).

#include "ggml.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>

namespace fn_tbo {

constexpr int MAX_CHAINS = 2;    // the gate allows 2 chains; the walk and the channel layout take any count
constexpr int KEY_STRIDE = 1024; // post / hint key of (layer, chain) in the graph's maps: il + chain*KEY_STRIDE

// 0 off, 1 interleaved, 2 sequential
inline int parse_mode(const char * v) {
    if (v == nullptr) {
        return 0;
    }
    while (*v != 0 && isspace((unsigned char) *v)) {
        ++v;
    }
    if (*v == 0 || *v == '0') {
        return 0;
    }
    if (*v == '2') {
        return 2;
    }
    if (*v == '1') {
        return 1;
    }
    // words: on / true / yes = 1, anything else off
    const char c = (char) tolower((unsigned char) *v);
    return c == 'o' ? (tolower((unsigned char) v[1]) == 'n' ? 1 : 0) : (c == 't' || c == 'y') ? 1 : 0;
}

// chain 0 keeps today's keys, so a graph without chains is unchanged
inline int post_key(int il, int chain) {
    return il + chain*KEY_STRIDE;
}

// what the graph knows when it decides
struct gate_in {
    int      mode         = 0;     // parse_mode
    bool     mtp          = false; // an MTP draft graph
    bool     bridge       = false; // the graph has the bridge (active) or its sync op (paused)
    int      n_chains_br  = 0;     // llama_moe_bridge_n_chains: channel sets of the bridge
    bool     spin         = false; // the bridge waits by spinning (no host function nodes)
    bool     equal_seqs   = false;
    uint32_t n_seqs       = 0;
    uint32_t n_seqs_unq   = 0;
    uint32_t n_seq_tokens = 0;
    uint32_t n_tokens     = 0;
    uint32_t n_outputs    = 0;
    bool     token        = false; // token rows (no embedding rows, no image rows)
    bool     embd         = false; // embedding rows or an MTP embedding state
    bool     kv_unified   = false; // one KV stream
    bool     flash_attn   = false;
    bool     fits         = false; // the whole ubatch fits the bridge (llama_moe_bridge_fits)
    bool     layer_inp    = false; // a layer input is extracted (llama_set_embeddings_layer_inp)
    bool     trace        = false; // routing capture or a routing prediction (trace / DMA prefetch) is built
    bool     inputs_ok    = true;  // the per-row inputs have the layout the chain views take (one KV stream, no replay ring)
};

struct plan {
    bool         on    = false;
    int          n_ch  = 1;
    int64_t      T     = 0;     // rows per chain
    int          order = 0;     // 1 interleaved, 2 sequential
    const char * why   = "off"; // the first condition that failed (logs, tests)
};

inline plan decide(const gate_in & g) {
    plan p;
    auto no = [&p](const char * why) {
        p.why = why;
        return p;
    };
    if (g.mode <= 0)                                   return no("switch off");
    if (g.mtp)                                         return no("MTP graph");
    if (!g.bridge)                                     return no("no bridge");
    if (g.n_chains_br < 2)                             return no("bridge has one channel set");
    if (!g.spin)                                       return no("bridge waits are not spin waits");
    if (!g.equal_seqs)                                 return no("not an equal-sequence ubatch");
    if (g.n_seqs != 2 || g.n_seqs_unq != 2)            return no("not two sequences");
    if (g.n_seq_tokens < 1 || g.n_tokens != 2*g.n_seq_tokens) return no("row count");
    if (g.n_outputs != g.n_tokens)                     return no("not every row is an output");
    if (!g.token || g.embd)                            return no("not a token ubatch");
    if (!g.kv_unified)                                 return no("KV not unified");
    if (!g.flash_attn)                                 return no("no flash attention");
    if (!g.fits)                                       return no("ubatch does not fit the bridge");
    if (g.layer_inp)                                   return no("layer input extraction");
    if (g.trace)                                       return no("routing capture");
    if (!g.inputs_ok)                                  return no("input layout");
    p.on    = true;
    p.n_ch  = 2;
    p.T     = g.n_seq_tokens;
    p.order = g.mode == 2 ? 2 : 1;
    p.why   = "on";
    return p;
}

// the order of the chain steps: step(kind, chain, il) with kind 0 = begin (PLE, token mixer, FFN mixer, router, bridge
// post, hot chain, shared expert) and 1 = finish (bridge wait, FFN combine). Order 1: every chain's begin(0), then per
// layer each chain's finish(L) right before its own begin(L + 1), chains in turn; order 2: chain after chain.
// Every (chain, layer) gets one begin and one finish, finish after begin, and begin(L + 1) right after finish(L).
template <class F>
inline void walk(int n_layer, int n_ch, int order, F && step) {
    if (n_layer <= 0 || n_ch <= 0) {
        return;
    }
    if (order == 2) {
        for (int c = 0; c < n_ch; ++c) {
            for (int il = 0; il < n_layer; ++il) {
                step(0, c, il);
                step(1, c, il);
            }
        }
        return;
    }
    for (int c = 0; c < n_ch; ++c) {
        step(0, c, 0);
    }
    for (int il = 0; il + 1 < n_layer; ++il) {
        for (int c = 0; c < n_ch; ++c) {
            step(1, c, il);
            step(0, c, il + 1);
        }
    }
    for (int c = 0; c < n_ch; ++c) {
        step(1, c, n_layer - 1);
    }
}

// the bridge's channel numbers: the layer channels of chain 0 (n0, today's), the layer channels of the other chains right
// after them (chain c, layer channel i at c*n0 + i), then the PLE gate's channel, then P11's defer channels (chain 0 only:
// a deferred post comes from one-sequence windows, never from a chain). One chain = today's layout.
struct chan_layout {
    int  n0    = 0;
    int  n_ch  = 1;
    bool ple   = false;
    bool defer = false;

    int layer_chan(int chain, int i) const {
        return chain >= 0 && chain < n_ch && i >= 0 && i < n0 ? chain*n0 + i : -1;
    }
    int n_layer_chans() const { return n_ch*n0; }
    int ple_chan()      const { return ple ? n_layer_chans() : -1; }
    int defer_base()    const { return defer ? n_layer_chans() + (ple ? 1 : 0) : -1; }
    int defer_chan(int chan) const { return defer && chan >= 0 && chan < n0 ? defer_base() + chan : -1; }
    int n_chan()        const { return n_layer_chans() + (ple ? 1 : 0) + (defer ? n0 : 0); }
    // the chain of a layer channel, -1 for any other channel
    int chain_of(int chan) const { return n0 > 0 && chan >= 0 && chan < n_layer_chans() ? chan/n0 : -1; }
};

// the chain's part of a graph input or a gathered state. A chain holds rows [r0, r0 + T) of the ubatch and its sequence's
// column c of a per-sequence gather. The whole tensors stay the graph's inputs; set_input fills them as before.

// rows of a per-row input [n] (indices, turbot young rows)
inline ggml_tensor * rows1(ggml_context * ctx, ggml_tensor * t, int64_t r0, int64_t T) {
    return t ? ggml_view_1d(ctx, t, T, (size_t) r0*t->nb[0]) : nullptr;
}

// rows of a [k, n] input (k-pool mask, tail cells, PLE rows)
inline ggml_tensor * rows2(ggml_context * ctx, ggml_tensor * t, int64_t r0, int64_t T) {
    return t ? ggml_view_2d(ctx, t, t->ne[0], T, t->nb[1], (size_t) r0*t->nb[1]) : nullptr;
}

// rows of a residual [n_embd, hc, n]
inline ggml_tensor * rows3(ggml_context * ctx, ggml_tensor * t, int64_t r0, int64_t T) {
    return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], T, t->nb[1], t->nb[2], (size_t) r0*t->nb[2]);
}

// an explicit KQ mask [n_kv, n, 1, 1]: the chain's query rows (contiguous, as flash attention needs)
inline ggml_tensor * mask_rows(ggml_context * ctx, ggml_tensor * m, int64_t r0, int64_t T) {
    return m ? ggml_view_4d(ctx, m, m->ne[0], T, 1, 1, m->nb[1], m->nb[1]*T, m->nb[1]*T, (size_t) r0*m->nb[1]) : nullptr;
}

// positions [n_pos*n], section-major (M-RoPE): the chain's [n_pos*T], section s = entries [s*n + r0, s*n + r0 + T)
inline ggml_tensor * pos_rows(ggml_context * ctx, ggml_tensor * pos, int64_t n, int64_t r0, int64_t T) {
    const int64_t n_pos = pos->ne[0] / n;
    if (n_pos == 1) {
        return rows1(ctx, pos, r0, T);
    }
    ggml_tensor * v = ggml_view_2d(ctx, pos, T, n_pos, pos->nb[0]*n, (size_t) r0*pos->nb[0]);
    return ggml_reshape_1d(ctx, ggml_cont(ctx, v), n_pos*T);
}

// the query positions of the positional mask: [n] (one row) or [n, 2] (positions, sequence bits; contiguous for the
// attention kernels)
inline ggml_tensor * qpos_rows(ggml_context * ctx, ggml_tensor * q, int64_t r0, int64_t T) {
    if (q == nullptr) {
        return nullptr;
    }
    if (q->ne[1] == 1) {
        return rows1(ctx, q, r0, T);
    }
    return ggml_cont(ctx, ggml_view_2d(ctx, q, T, q->ne[1], q->nb[1], (size_t) r0*q->nb[0]));
}

// column c of a gather of every sequence's rows [n, n_seqs]
inline ggml_tensor * col(ggml_context * ctx, ggml_tensor * rows, int64_t c) {
    return ggml_view_2d(ctx, rows, rows->ne[0], 1, rows->nb[1], (size_t) c*rows->nb[1]);
}

} // namespace fn_tbo
