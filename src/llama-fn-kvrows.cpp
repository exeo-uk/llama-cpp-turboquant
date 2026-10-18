// [TAG_UPSTREAM_29510] [TAG_FN_L22_KVROWS] round 22 P13: per-sequence K/V cell lists of a unified turbot cache
// (src/llama-fn-kvrows.h). The port 9ca2f8145 put its graph code into llama-graph.cpp and its cell count into
// llama-kv-cache.cpp; here the same steps sit behind P0's hook in build_attn_mha (llama_fn_kvrows_attn) and the
// read-only cell accessor fn_cells():
//
//   build:     the first hook call of a graph creates one graph input (llm_fn_kvrows_input) for the attention input's
//              kv_pos, with the plan of the ubatch and, when the plan has lists, one I32 list per slice and one q_pos
//              [n_tps, 2] per slice (several slices); every attention layer that passes the port's layer test gets one
//              FA per slice with its list (src[9]) and its q_pos; the outputs are concatenated in token order and the
//              rest of build_attn_mha (inverse WHT, ...) runs once on them
//   set_input: after the attention input wrote kv_pos / q_pos (inputs run in the order they were added), the lists
//              come from those vectors and the slices' q_pos are copies of q_pos's columns (keys pass through)
//   can_reuse: the plan of the new ubatch must equal the built one (lists appear and go away with a rebuild)
//
// Unset (LLAMA_KV_ROWS 0): P0's hook never calls in (llm_fn_l22_kvrows_on), so no input, no node: byte-identical.

#include "llama-fn-kvrows.h"
#include "llama-fn-l22.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "llama-memory-hybrid.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

//
// pure pieces (exported for the tests)
//

llama_fn_kvrows_cfg llama_fn_kvrows_cfg_env() {
    static const llama_fn_kvrows_cfg cfg = [] {
        llama_fn_kvrows_cfg c;
        const char * e = getenv("LLAMA_KV_ROWS");
        const int    m = e ? atoi(e) : 0;
        c.mode = m == 1 || m == 2 ? m : 0;
        e = getenv("LLAMA_KV_ROWS_MAX_TPS");
        c.max_tps = (uint32_t) std::max(1, e ? atoi(e) : 8);
        e = getenv("LLAMA_KV_ROWS_MIN_TPS");
        c.min_tps = (uint32_t) std::max(1, e ? atoi(e) : 1);
        return c;
    }();
    return cfg;
}

uint32_t llama_fn_kvrows_slices(const llama_ubatch & ubatch) {
    const uint32_t n_slices = ubatch.n_seqs_unq;
    if (n_slices == 0 || ubatch.n_tokens == 0 || ubatch.n_tokens % n_slices != 0 || ubatch.n_seq_id == nullptr ||
            ubatch.seq_id == nullptr) {
        return 0;
    }
    const uint32_t n_tps = ubatch.n_tokens / n_slices;
    llama_seq_id first[32];
    if (n_slices > 32) {
        return 0;
    }
    for (uint32_t s = 0; s < n_slices; ++s) {
        const uint32_t i0 = s*n_tps;
        if (ubatch.n_seq_id[i0] != 1 || ubatch.seq_id[i0] == nullptr) {
            return 0;
        }
        const llama_seq_id seq_id = ubatch.seq_id[i0][0];
        for (uint32_t i = i0; i < i0 + n_tps; ++i) {
            if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != seq_id) {
                return 0;
            }
        }
        // each slice its own sequence (n_seqs_unq groups of n_seqs_unq distinct ids)
        for (uint32_t t = 0; t < s; ++t) {
            if (first[t] == seq_id) {
                return 0;
            }
        }
        first[s] = seq_id;
    }
    return n_slices;
}

uint32_t llama_fn_kvrows_n_rows(const llama_kv_cells & cells, const llama_ubatch & ubatch, uint32_t n_slices, uint32_t n_kv) {
    if (n_slices == 0 || ubatch.n_tokens % n_slices != 0 || ubatch.seq_id == nullptr) {
        return 0;
    }

    const uint32_t n_tps = ubatch.n_tokens / n_slices;

    uint32_t n_max = 0;
    uint64_t n_sum = 0;

    for (uint32_t s = 0; s < n_slices; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id[s*n_tps][0];
        if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
            return 0;
        }
        const uint32_t n_cells = cells.seq_n_cells(seq_id);
        if (n_cells == 0) {
            return 0;
        }

        n_max  = std::max(n_max, n_cells);
        n_sum += n_cells;
    }

    const uint32_t n_rows = GGML_PAD(n_max, 256u);

    // shared cells (a common prompt) are read once by the one FA over the whole cache
    if (n_max == 0 || n_rows >= n_kv || n_sum > n_kv) {
        return 0;
    }

    return n_rows;
}

llama_fn_kvrows_plan llama_fn_kvrows_plan_cells(const llama_fn_kvrows_cfg & cfg, const llama_ubatch & ubatch,
                                                const llama_kv_cells & cells, uint32_t n_kv) {
    llama_fn_kvrows_plan plan;

    const uint32_t n_seqs = ubatch.n_seqs_unq;
    if (cfg.mode == 0 || n_seqs == 0 || (n_seqs == 1 && cfg.mode < 2) || n_seqs > 32 ||
            ubatch.n_tokens > n_seqs*cfg.max_tps || ubatch.n_tokens < n_seqs*cfg.min_tps) {
        return plan;
    }

    const uint32_t n_slices = llama_fn_kvrows_slices(ubatch);
    if (n_slices == 0) {
        return plan;
    }

    const uint32_t n_rows = llama_fn_kvrows_n_rows(cells, ubatch, n_slices, n_kv);
    if (n_rows == 0) {
        return plan;
    }

    plan.n_slices = n_slices;
    plan.n_rows   = n_rows;
    plan.n_tps    = ubatch.n_tokens / n_slices;

    return plan;
}

bool llama_fn_kvrows_fill_lists(const int32_t * kv_pos, int64_t n_kv, int32_t kv_rows, const int32_t * q_pos,
                                int64_t n_tokens, uint32_t n_slices, uint32_t n_rows, int32_t * out) {
    if (kv_pos == nullptr || out == nullptr || n_kv <= 0 || n_slices == 0 || n_slices > 32 || n_rows == 0 ||
            n_tokens <= 0 || n_tokens % n_slices != 0 || (kv_rows != 1 && kv_rows != 2) ||
            (kv_rows == 1 && n_slices != 1) || (kv_rows == 2 && q_pos == nullptr)) {
        return false;
    }

    const int64_t n_tps = n_tokens / n_slices;

    // the sequence bit of each slice, as the kernel compares it: row 1 of q_pos (every token of the slice the same bit)
    uint32_t bits[32] = {};
    for (uint32_t s = 0; s < n_slices; ++s) {
        if (kv_rows == 1) {
            bits[s] = 1u;
            continue;
        }
        const int32_t * qb = q_pos + n_tokens + s*n_tps;
        const uint32_t  b  = (uint32_t) qb[0];
        if (b == 0 || (b & (b - 1)) != 0) {
            return false;
        }
        for (int64_t t = 1; t < n_tps; ++t) {
            if ((uint32_t) qb[t] != b) {
                return false;
            }
        }
        bits[s] = b;
    }

    const int32_t * sets = kv_rows == 2 ? kv_pos + n_kv : nullptr;

    std::fill(out, out + (int64_t) n_rows*n_slices, -1);
    uint32_t cnt[32] = {};

    for (int64_t j = 0; j < n_kv; ++j) {
        if (kv_pos[j] < 0) {
            continue;
        }
        const uint32_t set = sets ? (uint32_t) sets[j] : 1u;
        for (uint32_t s = 0; s < n_slices; ++s) {
            if ((set & bits[s]) != 0) {
                if (cnt[s] >= n_rows) {
                    return false;
                }
                out[(int64_t) s*n_rows + cnt[s]++] = (int32_t) j;
            }
        }
    }

    return true;
}

void llama_fn_kvrows_split_q_pos(const int32_t * q_pos, int64_t n_tokens, uint32_t n_slices, int32_t * out) {
    GGML_ASSERT(q_pos != nullptr && out != nullptr && n_slices > 0 && n_tokens % n_slices == 0);
    const int64_t n_tps = n_tokens / n_slices;
    for (uint32_t s = 0; s < n_slices; ++s) {
        int32_t * dst = out + (int64_t) s*2*n_tps;
        memcpy(dst,         q_pos +            s*n_tps, n_tps*sizeof(int32_t));
        memcpy(dst + n_tps, q_pos + n_tokens + s*n_tps, n_tps*sizeof(int32_t));
    }
}

ggml_tensor * llama_fn_kvrows_build_slices(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                           ggml_tensor * gtab, ggml_tensor * q_pos,
                                           const std::vector<ggml_tensor *> & q_pos_rows,
                                           const std::vector<ggml_tensor *> & rows, uint32_t n_slices,
                                           const llama_fn_kvrows_build_fa_t & build_fa) {
    GGML_ASSERT(n_slices > 0 && q->ne[1] % n_slices == 0 && q->ne[3] == 1);
    GGML_ASSERT(n_slices == 1 || q_pos_rows.size() == n_slices);
    GGML_ASSERT(rows.empty() || rows.size() == n_slices);

    const int64_t n_tps = q->ne[1] / n_slices;

    ggml_tensor * cur = nullptr;
    for (uint32_t s = 0; s < n_slices; ++s) {
        ggml_tensor * q_s    = ggml_view_4d(ctx, q, q->ne[0], n_tps, q->ne[2], 1, q->nb[1], q->nb[2], q->nb[3], s*n_tps*q->nb[1]);
        ggml_tensor * qpos_s = n_slices == 1 ? q_pos : q_pos_rows[s];

        ggml_tensor * fa = build_fa(q_s, k, v, nullptr, gtab, qpos_s);
        if (!rows.empty() && rows[s] != nullptr) {
            ggml_flash_attn_ext_set_kv_rows(fa, rows[s]);
        }
        cur = cur == nullptr ? fa : ggml_concat(ctx, cur, fa, 3);
    }

    return cur;
}

//
// graph side
//

namespace {

std::atomic<uint64_t> g_graphs_listed{0};
std::atomic<uint64_t> g_graphs_seen{0};
std::atomic<uint64_t> g_layers_listed{0};
std::atomic<uint64_t> g_fills{0};

// the graph's attention cache: a plain KV cache or the attention half of a hybrid memory; anything else (iSWA, the
// indexer memories, ...) keeps the one FA
const llama_kv_cache_context * kvrows_kctx(const llama_memory_context_i * m) {
    if (m == nullptr) {
        return nullptr;
    }
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(m)) {
        return h->get_attn();
    }
    return dynamic_cast<const llama_kv_cache_context *>(m);
}

// the plan with the model / cache gates of the port: flash attention, causal, no indexer (qwen4exp), turbot, and one
// KV stream for the whole cache (unified, or one sequence), so fn_cells(0) holds every sequence's cells
llama_fn_kvrows_plan kvrows_plan(const llama_ubatch & ubatch, const llama_hparams & hparams, const llama_cparams & cparams,
                                 const llama_kv_cache_context * kctx) {
    const llama_fn_kvrows_cfg & cfg = llama_fn_kvrows_cfg_env();
    if (cfg.mode == 0 || kctx == nullptr || !cparams.flash_attn || !cparams.causal_attn || hparams.indexer_n_head > 0 ||
            !kctx->is_turbot() || !(cparams.kv_unified || cparams.n_seq_max == 1)) {
        return {};
    }
    return llama_fn_kvrows_plan_cells(cfg, ubatch, kctx->fn_cells(0), kctx->get_n_kv());
}

// an I32 tensor's values: in place when it lives in host memory, else copied
const int32_t * kvrows_read_i32(const ggml_tensor * t, std::vector<int32_t> & tmp) {
    GGML_ASSERT(t != nullptr && t->buffer != nullptr && t->type == GGML_TYPE_I32 && ggml_is_contiguous(t));
    if (ggml_backend_buffer_is_host(t->buffer)) {
        return (const int32_t *) t->data;
    }
    tmp.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
    return tmp.data();
}

class llm_fn_kvrows_input : public llm_graph_input_i {
public:
    llm_fn_kvrows_input(const ggml_tensor * kv_pos, const ggml_tensor * q_pos, const llama_fn_kvrows_plan & plan) :
        kv_pos(kv_pos), q_pos(q_pos), plan(plan) {}
    ~llm_fn_kvrows_input() override = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (plan.n_rows == 0) {
            return;
        }
        bool used = false;
        for (const ggml_tensor * t : rows) {
            used = used || t->buffer != nullptr;
        }
        if (!used) {
            return;   // every layer kept the one FA
        }

        GGML_ASSERT(ubatch->n_tokens == plan.n_slices*plan.n_tps && "kv_rows: the ubatch does not fit the plan");
        GGML_ASSERT(kv_pos->ne[1] == (plan.n_slices > 1 ? 2 : 1) && q_pos->ne[0] == (int64_t) ubatch->n_tokens);

        std::vector<int32_t> tmp_kv;
        std::vector<int32_t> tmp_q;
        const int32_t * kvp = kvrows_read_i32(kv_pos, tmp_kv);
        const int32_t * qp  = plan.n_slices > 1 ? kvrows_read_i32(q_pos, tmp_q) : nullptr;

        lists.resize((size_t) plan.n_rows*plan.n_slices);
        const bool ok = llama_fn_kvrows_fill_lists(kvp, kv_pos->ne[0], (int32_t) kv_pos->ne[1], qp, ubatch->n_tokens,
                                                   plan.n_slices, plan.n_rows, lists.data());
        GGML_ASSERT(ok && "kv_rows: a sequence has more cells than its list, or a slice mixes sequences");

        for (uint32_t s = 0; s < plan.n_slices; ++s) {
            ggml_tensor * t = rows[s];
            if (t->buffer) {
                ggml_backend_tensor_set(t, lists.data() + (size_t) s*plan.n_rows, 0, (size_t) plan.n_rows*sizeof(int32_t));
            }
        }

        if (plan.n_slices > 1) {
            qsplit.resize((size_t) 2*ubatch->n_tokens);
            llama_fn_kvrows_split_q_pos(qp, ubatch->n_tokens, plan.n_slices, qsplit.data());
            for (uint32_t s = 0; s < plan.n_slices; ++s) {
                ggml_tensor * t = q_pos_rows[s];
                if (t->buffer) {
                    GGML_ASSERT(t->ne[0] == (int64_t) plan.n_tps && t->ne[1] == 2 && ggml_is_contiguous(t));
                    ggml_backend_tensor_set(t, qsplit.data() + (size_t) s*2*plan.n_tps, 0, ggml_nbytes(t));
                }
            }
        }

        g_fills.fetch_add(1, std::memory_order_relaxed);
    }

    bool can_reuse(const llm_graph_params & params) override {
        const llama_fn_kvrows_plan p = kvrows_plan(params.ubatch, params.hparams, params.cparams, kvrows_kctx(params.mctx));
        return p == plan;
    }

    const ggml_tensor * kv_pos;
    const ggml_tensor * q_pos;
    llama_fn_kvrows_plan plan;

    std::vector<ggml_tensor *> rows;       // n_slices x I32 [n_rows], -1 padded
    std::vector<ggml_tensor *> q_pos_rows; // n_slices x I32 [n_tps, 2], several slices only
    bool q_pos_kept = false;               // q_pos expanded into the graph (no FA of a list graph reads it)
    uint32_t n_layers = 0;                 // layers that took the lists

private:
    std::vector<int32_t> lists;
    std::vector<int32_t> qsplit;
};

void kvrows_log_switch(const llama_fn_kvrows_cfg & cfg) {
    static std::atomic<bool> done{false};
    if (!done.exchange(true)) {
        llama_fn_l22_log("switch", "[TAG_UPSTREAM_29510] kv_rows mode=%d max_tps=%u min_tps=%u", cfg.mode, cfg.max_tps,
                         cfg.min_tps);
    }
}

void kvrows_log_graph(const llama_fn_kvrows_plan & plan, int64_t n_kv) {
    static std::atomic<bool> first{false};
    const uint64_t n = g_graphs_listed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!first.exchange(true)) {
        llama_fn_l22_log("graph", "[TAG_UPSTREAM_29510] kv_rows first lists: slices=%u rows=%u tps=%u n_kv=%lld",
                         plan.n_slices, plan.n_rows, plan.n_tps, (long long) n_kv);
    } else if (llama_fn_l22_passlog_level() >= 2 && n % 4096 == 0) {
        llama_fn_l22_log("graph", "[TAG_UPSTREAM_29510] kv_rows list graphs=%llu of %llu, layers=%llu, fills=%llu",
                         (unsigned long long) n, (unsigned long long) g_graphs_seen.load(),
                         (unsigned long long) g_layers_listed.load(), (unsigned long long) g_fills.load());
    }
}

// this graph's input for kv_pos, created at the first hook call of the graph (even with an empty plan, so a reused
// graph without lists is refused once a ubatch has them)
llm_fn_kvrows_input * kvrows_input(llm_graph_context * g, const llama_fn_kvrows_args * a, const llama_kv_cache_context * kctx) {
    for (auto & in : g->res->inputs) {
        if (auto * r = dynamic_cast<llm_fn_kvrows_input *>(in.get())) {
            if (r->kv_pos == a->kv_pos) {
                return r;
            }
        }
    }

    const llama_fn_kvrows_plan plan = kvrows_plan(g->ubatch, g->hparams, g->cparams, kctx);

    auto inp = std::make_unique<llm_fn_kvrows_input>(a->kv_pos, a->q_pos, plan);
    for (uint32_t s = 0; s < plan.n_slices && plan.n_rows > 0; ++s) {
        ggml_tensor * rows = ggml_new_tensor_1d(g->ctx0, GGML_TYPE_I32, plan.n_rows);
        ggml_set_input(rows);
        ggml_format_name(rows, "attn_inp_kv_rows-%u", s);
        inp->rows.push_back(rows);
        if (plan.n_slices > 1) {
            ggml_tensor * qp = ggml_new_tensor_2d(g->ctx0, GGML_TYPE_I32, plan.n_tps, 2);
            ggml_set_input(qp);
            ggml_format_name(qp, "attn_inp_q_pos_rows-%u", s);
            inp->q_pos_rows.push_back(qp);
        }
    }

    g_graphs_seen.fetch_add(1, std::memory_order_relaxed);

    return static_cast<llm_fn_kvrows_input *>(g->res->add_input(std::move(inp)));
}

} // namespace

llama_fn_kvrows_stats llama_fn_kvrows_stats_get() {
    llama_fn_kvrows_stats st;
    st.graphs_listed = g_graphs_listed.load();
    st.graphs_seen   = g_graphs_seen.load();
    st.layers_listed = g_layers_listed.load();
    st.fills         = g_fills.load();
    return st;
}

ggml_tensor * llama_fn_kvrows_attn(llm_graph_context * g, const void * vargs) {
    const auto * a = (const llama_fn_kvrows_args *) vargs;

    const llama_fn_kvrows_cfg & cfg = llama_fn_kvrows_cfg_env();
    if (g == nullptr || a == nullptr || cfg.mode == 0 || !a->build_fa || a->kv_pos == nullptr || a->q_pos == nullptr ||
            a->turbot_params == nullptr) {
        return nullptr;
    }
    kvrows_log_switch(cfg);

    // only the graph's own attention cache (kv or hybrid); P0 passes the layer's cache context
    const auto * kctx = (const llama_kv_cache_context *) a->mctx;
    if (kctx == nullptr || kvrows_kctx(g->mctx) != kctx) {
        return nullptr;
    }

    llm_fn_kvrows_input * inp = kvrows_input(g, a, kctx);
    const llama_fn_kvrows_plan & plan = inp->plan;
    if (plan.n_rows == 0) {
        return nullptr;
    }

    // the port's layer test (permuted shapes: q [D, n_tokens, n_head, 1], k / v [D, n_kv, n_head_kv, 1]): the turbot
    // list kernel takes D 256, GQA > 4, no sinks, no softcap, no ALiBi, no explicit mask, no n_kv_max; any other layer
    // keeps the one FA over the whole cache, which gives the same result
    const ggml_tensor * q = a->q;
    const ggml_tensor * k = a->k;
    const ggml_tensor * v = a->v;
    const int64_t n_tok = (int64_t) plan.n_slices*plan.n_tps;
    const bool use = a->kq_mask == nullptr && a->sinks == nullptr && a->v_mla == nullptr && a->n_kv_max == 0 &&
        k->ne[3] == 1 && q->ne[3] == 1 && q->ne[0] == 256 && k->ne[0] == 256 && v->ne[0] == 256 &&
        k->ne[2] > 0 && q->ne[2] % k->ne[2] == 0 && q->ne[2]/k->ne[2] > 4 &&
        g->hparams.f_max_alibi_bias == 0.0f && !g->hparams.attn_soft_cap &&
        q->ne[1] == n_tok && a->q_pos->ne[0] == n_tok && a->kv_pos->ne[0] == k->ne[1] &&
        a->kv_pos->ne[1] == (plan.n_slices > 1 ? 2 : 1) && a->q_pos->ne[1] == a->kv_pos->ne[1];
    if (!use) {
        return nullptr;
    }

    // no FA of a list graph reads q_pos when there are several slices; the attention input still writes it and the
    // slices' copies come from it, so it stays in the graph
    if (plan.n_slices > 1 && !inp->q_pos_kept) {
        ggml_build_forward_expand(g->gf, a->q_pos);
        inp->q_pos_kept = true;
    }

    ggml_tensor * cur = llama_fn_kvrows_build_slices(g->ctx0, a->q, a->k, a->v, a->turbot_gtab, a->q_pos, inp->q_pos_rows,
                                                     inp->rows, plan.n_slices, a->build_fa);

    if (inp->n_layers++ == 0) {
        kvrows_log_graph(plan, a->kv_pos->ne[0]);
    }
    g_layers_listed.fetch_add(1, std::memory_order_relaxed);

    return cur;
}
