// [TAG_FN_L22_X_PRIOR] round 22 P12 (qwen4exp, opt-in, changes answers): the route prior and the resident swap on the
// selection probabilities, src/llama-fn-l22.h (hook) and src/llama-fn-prior.h (state, reference).
//
// State (one per process, the bridge's): an F32 table [n_expert, 2, n_layer] in its own device buffer, allocated at the
// bridge's creation (before ready, no compute-buffer growth): row 0 of layer il = the scale (exp(lambda) for an expert in
// the VRAM hot set, 1 else), row 1 = the hot mask (1 / 0). A prior graph's input object rewrites the rows of the layers
// whose host hot table changed (the table the bridge's CPU job skips with; it changes only between graphs, on the owner
// thread) before the graph runs. Hook: in a decode graph (fn_x has PRIOR or RESIDENT) of a bridged layer, sel = probs *
// row (marked MUL, ggml-fn-l22-prior.h) feeds the argsort; the weights still come from probs. Every backend computes the
// MUL (the prior is exact everywhere); the CUDA bridge code fuses it and adds the resident swap.

#include "llama-fn-l22.h"
#include "llama-fn-prior.h"

#include "llama-fn-auto.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-bridge.h"
#include "llama-moecache.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-fn-l22-prior.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct fn_prior_state {
    llama_fn_prior_cfg cfg;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         tab = nullptr; // [n_exp, 2, n_layer_tab]
    int n_exp       = 0;
    int n_layer_tab = 0;
    int n_layers    = 0;                 // bridged layers
    std::vector<const ggml_tensor *>  up;        // per il: the layer's ffn_up_exps, nullptr when not bridged
    std::vector<std::vector<int32_t>> last;      // per il: the hot table the rows were made from
    std::vector<int32_t>              last_miss; // per il: its miss value
    std::vector<uint8_t>              had;       // per il: 0 not written by a refresh yet, 1 from a table, 2 no table
    std::vector<float>                stage;     // the table's host copy [n_layer_tab][2][n_exp]
};

fn_prior_state g_pr;

std::atomic<bool> g_warn_probs{false};
std::atomic<int>  g_note_graph{0};
std::atomic<int>  g_note_prompt{0};

const char * prior_env(const llama_model * model, const char * name) {
    return model ? llama_fn_env(*model, name) : getenv(name);
}

// the refresh object of a graph with a prior MUL (no tensor of its own: the table is persistent)
class llm_graph_input_fn_prior : public llm_graph_input_i {
public:
    explicit llm_graph_input_fn_prior(ggml_backend_sched_t sched) : sched(sched) {}
    virtual ~llm_graph_input_fn_prior() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        llama_fn_prior_refresh(sched);
    }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return true; // the table is read through its persistent address; the refresh runs for every computation
    }

private:
    ggml_backend_sched_t sched;
};

// the lowest-index argmax (the kernel's tie rule: the larger value, then the lower expert)
int prior_argmax(const float * v, int n) {
    int be = 0;
    for (int e = 1; e < n; ++e) {
        if (v[e] > v[be]) {
            be = e;
        }
    }
    return be;
}

} // namespace

llama_fn_prior_cfg llama_fn_prior_read(const llama_model * model) {
    llama_fn_prior_cfg c;
    const char * vp = prior_env(model, "LLAMA_FN_L22_X_PRIOR");
    if (llama_fn_l22_switch_on(vp)) {
        const float l = (float) atof(vp);
        c.lambda = std::isfinite(l) ? std::min(16.0f, std::max(0.0f, l)) : 0.0f;
    }
    const char * vr = prior_env(model, "LLAMA_FN_L22_X_RESIDENT");
    if (llama_fn_l22_switch_on(vr)) {
        const float m = (float) atof(vr);
        c.margin = std::isfinite(m) ? std::min(16.0f, std::max(0.0f, m)) : 0.0f;
    }
    if (const char * v0 = prior_env(model, "LLAMA_FN_L22_X_RESIDENT_R0"); v0 && v0[0]) {
        c.r0 = std::min(32, std::max(0, atoi(v0)));
    }
    if (const char * vs = prior_env(model, "LLAMA_FN_L22_X_PRIOR_STATS"); vs && vs[0]) {
        c.stats_raw = strcmp(vs, "used") != 0 && strcmp(vs, "USED") != 0;
    }
    c.prompt = llama_fn_l22_switch_on(prior_env(model, "LLAMA_FN_L22_X_PRIOR_PROMPT"));
    return c;
}

bool llama_fn_prior_init(const llama_model & model, ggml_backend_dev_t dev,
        const std::vector<std::pair<int, const ggml_tensor *>> & layers) {
    llama_fn_prior_free(); // one bridge per process: a leftover state goes
    fn_prior_state & st = g_pr;
    const llama_fn_prior_cfg cfg = llama_fn_prior_read(&model);
    if (!cfg.on()) {
        return false;
    }
    if (model.arch != LLM_ARCH_QWEN4EXP) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] LLAMA_FN_L22_X_PRIOR / _X_RESIDENT act on qwen4exp only: off\n", __func__);
        return false;
    }
    if (dev == nullptr || layers.empty() || layers[0].second == nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] no bridged layer / device: off\n", __func__);
        return false;
    }
    const int64_t n_exp = layers[0].second->ne[2];
    int max_il = -1;
    for (const auto & l : layers) {
        if (l.first < 0 || l.second == nullptr || l.second->ne[2] != n_exp) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] the bridged layers differ in their expert count: off\n", __func__);
            return false;
        }
        max_il = std::max(max_il, l.first);
    }
    if (n_exp < 1 || n_exp > 4096 || max_il < 0 || max_il > 4096) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] %lld experts, layer %d: off\n", __func__, (long long) n_exp, max_il);
        return false;
    }

    ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
    st.ctx = ggml_init(ip);
    if (st.ctx == nullptr) {
        return false;
    }
    st.tab = ggml_new_tensor_3d(st.ctx, GGML_TYPE_F32, n_exp, 2, max_il + 1);
    ggml_set_name(st.tab, "fn_l22_prior_tab");
    const size_t bytes = ggml_nbytes(st.tab);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    st.buf = buft ? ggml_backend_alloc_ctx_tensors_from_buft(st.ctx, buft) : nullptr;
    if (st.buf == nullptr && buft && llama_fn_pin_active() && llama_fn_pin_on_oom(bytes + (1u << 20))) {
        st.buf = ggml_backend_alloc_ctx_tensors_from_buft(st.ctx, buft); // plan 3.6: the pins gave VRAM back once
    }
    if (st.buf == nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] the %.1f KiB table could not be allocated on %s: off\n", __func__,
                bytes/1024.0, ggml_backend_dev_name(dev));
        st.tab = nullptr;
        ggml_free(st.ctx);
        st.ctx = nullptr;
        return false;
    }

    st.cfg         = cfg;
    st.n_exp       = (int) n_exp;
    st.n_layer_tab = max_il + 1;
    st.n_layers    = (int) layers.size();
    st.up.assign(st.n_layer_tab, nullptr);
    st.last.assign(st.n_layer_tab, std::vector<int32_t>());
    st.last_miss.assign(st.n_layer_tab, 0);
    st.had.assign(st.n_layer_tab, 0);
    st.stage.assign((size_t) st.n_layer_tab*2*st.n_exp, 0.0f);
    for (int il = 0; il < st.n_layer_tab; ++il) {
        llama_fn_prior_fill_rows(nullptr, 0, st.n_exp, st.cfg.lambda, st.stage.data() + (size_t) il*2*st.n_exp,
                st.stage.data() + (size_t) il*2*st.n_exp + st.n_exp);
    }
    for (const auto & l : layers) {
        st.up[l.first] = l.second;
    }
    ggml_backend_tensor_set(st.tab, st.stage.data(), 0, st.stage.size()*sizeof(float)); // no graph runs yet

    LLAMA_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] decode windows of %d bridged layers pick their top-k with %s%s; the observer gets "
            "the %s ids%s; table %.1f KiB on %s. CHANGES ANSWERS\n", __func__, st.n_layers,
            st.cfg.lambda > 0.0f ? format("the route prior (lambda %.3f: hot experts' probability x %.3f)", st.cfg.lambda,
                    expf(st.cfg.lambda)).c_str() : "no route prior",
            st.cfg.margin > 0.0f ? format(" and the resident swap (ranks %d.., margin %.3f, fused path only)", st.cfg.r0,
                    st.cfg.margin).c_str() : "",
            st.cfg.stats_raw ? "unbiased (raw)" : "picked (used)", st.cfg.prompt ? "; also short prompt graphs" : "",
            bytes/1024.0, ggml_backend_dev_name(dev));
    llama_fn_l22_log("switch", "[TAG_FN_L22_X_PRIOR] lambda=%.3f margin=%.3f r0=%d stats=%s prompt=%d layers=%d bytes=%zu",
            st.cfg.lambda, st.cfg.margin, st.cfg.r0, st.cfg.stats_raw ? "raw" : "used", st.cfg.prompt ? 1 : 0, st.n_layers,
            bytes);
    return true;
}

void llama_fn_prior_free(void) {
    fn_prior_state & st = g_pr;
    st.tab = nullptr;
    if (st.buf) {
        ggml_backend_buffer_free(st.buf);
        st.buf = nullptr;
    }
    if (st.ctx) {
        ggml_free(st.ctx);
        st.ctx = nullptr;
    }
    st = fn_prior_state();
}

bool llama_fn_prior_on(void) {
    return g_pr.tab != nullptr;
}

bool llama_fn_prior_raw_wanted(void) {
    return g_pr.tab != nullptr && g_pr.cfg.stats_raw;
}

llama_fn_prior_cfg llama_fn_prior_get_cfg(void) {
    return g_pr.cfg;
}

void llama_fn_prior_refresh(ggml_backend_sched_t sched) {
    fn_prior_state & st = g_pr;
    if (st.tab == nullptr) {
        return;
    }
    const size_t n_exp = (size_t) st.n_exp;
    int lo = st.n_layer_tab;
    int hi = -1;
    for (int il = 0; il < st.n_layer_tab; ++il) {
        if (st.up[il] == nullptr) {
            continue;
        }
        // the host table only: valid also while the layer is out (lent, tail, KV), then "not hot" for every expert
        const llama_moe_cache_layer * mc = llama_moe_cache_lookup_table(st.up[il]);
        const int32_t * tbl = mc && mc->host_table && mc->host_table->data &&
                              (size_t) ggml_nelements(mc->host_table) == n_exp ? (const int32_t *) mc->host_table->data : nullptr;
        const int32_t miss = tbl ? mc->n_slots : 0;
        if (tbl) {
            if (st.had[il] == 1 && st.last_miss[il] == miss && memcmp(st.last[il].data(), tbl, n_exp*sizeof(int32_t)) == 0) {
                continue;
            }
            st.last[il].assign(tbl, tbl + n_exp);
            st.last_miss[il] = miss;
            st.had[il]       = 1;
        } else {
            if (st.had[il] == 2) {
                continue;
            }
            st.had[il] = 2;
        }
        float * rows = st.stage.data() + (size_t) il*2*n_exp;
        llama_fn_prior_fill_rows(tbl, miss, st.n_exp, st.cfg.lambda, rows, rows + n_exp);
        lo = std::min(lo, il);
        hi = std::max(hi, il);
    }
    if (hi < lo) {
        return;
    }
    // plan 3.6: the table is written only at a synchronized point (no earlier graph may still read it)
    if (sched) {
        ggml_backend_sched_synchronize(sched);
    }
    const size_t nb2 = st.tab->nb[2];
    ggml_backend_tensor_set(st.tab, st.stage.data() + (size_t) lo*2*n_exp, (size_t) lo*nb2, (size_t) (hi - lo + 1)*nb2);
}

ggml_tensor * llama_fn_route_prior(llm_graph_context * g, ggml_tensor * probs, int il) {
    fn_prior_state & st = g_pr;
    if (g == nullptr || probs == nullptr || st.tab == nullptr || il < 0 || il >= st.n_layer_tab || st.up[il] == nullptr) {
        return probs;
    }
    const llama_moe_bridge * br = g->moe_bridge ? g->moe_bridge : g->moe_bridge_sync;
    if (br == nullptr) {
        return probs;
    }
    const bool gen = (g->fn_x & (LLAMA_FN_L22_XB_PRIOR | LLAMA_FN_L22_XB_RESIDENT)) != 0;
    // plan 3.8: a lever acts only where fn_gen = 1, unless its own prompt rule (_X_PRIOR_PROMPT) says otherwise
    if (!gen && (!st.cfg.prompt || g->fn_gen)) {
        return probs;
    }
    // only where the hot experts run on the device (the bridge takes the graph, or a hot chain serves it): elsewhere the CPU
    // computes every expert and a prior would change the answer for nothing
    const int64_t T = g->n_tokens;
    const int64_t T_max = std::max<int64_t>(llama_moe_bridge_max_t(br), std::max(llama_moe_hot_graph_max_t(),
            llama_moe_hot_prompt_max_t()));
    if (T < 1 || T > T_max) {
        return probs;
    }
    int32_t bid = -1;
    int32_t bch = -1;
    if (!llama_moe_bridge_layer(br, st.up[il], &bid, &bch)) {
        return probs;
    }
    if (probs->type != GGML_TYPE_F32 || probs->ne[0] != st.n_exp || !ggml_is_contiguous(probs) ||
        ggml_nrows(probs) != probs->ne[1]) {
        if (!g_warn_probs.exchange(true)) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L22_X_PRIOR] layer %d: selection probabilities of an unexpected form (%s, %lld x %lld): no "
                    "prior there\n", __func__, il, ggml_type_name(probs->type), (long long) probs->ne[0], (long long) probs->ne[1]);
        }
        return probs;
    }
    int32_t flags = 0;
    flags |= st.cfg.lambda > 0.0f ? GGML_FN_L22_PRIOR_F_PRIOR    : 0;
    flags |= st.cfg.margin > 0.0f ? GGML_FN_L22_PRIOR_F_RESIDENT : 0;
    if (flags == 0) {
        return probs;
    }
    flags |= st.cfg.stats_raw ? GGML_FN_L22_PRIOR_F_RAW : 0;
    const float expnm = expf(-st.cfg.margin);

    // the row view goes into the graph first, so the router's SOFT_MAX, RESHAPE and the MUL stay next to each other (the
    // CUDA fusion's pattern)
    ggml_tensor * row = ggml_view_2d(g->ctx0, st.tab, st.n_exp, 1, st.tab->nb[1], (size_t) il*st.tab->nb[2]);
    ggml_format_name(row, "ffn_moe_prior_row-%d", il);
    ggml_build_forward_expand(g->gf, row);
    ggml_tensor * sel = llama_fn_prior_build_sel(g->ctx0, probs, row, flags, expnm, st.cfg.r0, il);
    ggml_format_name(sel, "ffn_moe_prior-%d", il);

    bool have = false;
    for (const auto & in : g->res->inputs) {
        if (dynamic_cast<const llm_graph_input_fn_prior *>(in.get()) != nullptr) {
            have = true;
            break;
        }
    }
    if (!have) {
        g->res->add_input(std::make_unique<llm_graph_input_fn_prior>(g->sched));
        std::atomic<int> & note = gen ? g_note_graph : g_note_prompt;
        if (note.fetch_add(1, std::memory_order_relaxed) < 2) {
            llama_fn_l22_log("graph", "[TAG_FN_L22_X_PRIOR] a %s graph of %lld tokens picks its experts with flags %d",
                    gen ? "decode" : "prompt", (long long) g->n_tokens, (int) flags);
        }
    }
    return sel;
}

//
// pure parts
//

void llama_fn_prior_fill_rows(const int32_t * tbl, int32_t miss, int n_exp, float lambda, float * scale, float * mask) {
    const float s_hot = expf(lambda);
    for (int e = 0; e < n_exp; ++e) {
        const bool hot = tbl != nullptr && tbl[e] != miss;
        scale[e] = hot ? s_hot : 1.0f;
        mask[e]  = hot ? 1.0f : 0.0f;
    }
}

ggml_tensor * llama_fn_prior_build_sel(ggml_context * ctx, ggml_tensor * probs, ggml_tensor * row, int32_t flags,
        float expnm, int32_t r0, int32_t il) {
    ggml_tensor * sel = ggml_mul(ctx, probs, row);
    ggml_fn_l22_prior_set(sel, flags, expnm, r0, il);
    return sel;
}

void llama_fn_prior_ref_select(const float * p_in, int n_exp, const float * scale, const float * mask, int k,
        int32_t flags, float expnm, int r0, bool with_norm, float clamp, float wscale, int32_t * ids, float * w, int32_t * raw) {
    if (n_exp < 1 || k < 1 || k > n_exp) {
        return;
    }
    std::vector<float> p((size_t) n_exp);
    std::vector<float> sel((size_t) n_exp);
    for (int e = 0; e < n_exp; ++e) {
        p[e] = std::isnan(p_in[e]) ? -FLT_MAX : p_in[e];
        const float s = p[e]*(scale ? scale[e] : 1.0f);
        sel[e] = s < -FLT_MAX ? -FLT_MAX : s;
    }
    if (raw) {
        std::vector<float> rv = p;
        for (int kk = 0; kk < k; ++kk) {
            const int e = prior_argmax(rv.data(), n_exp);
            raw[kk] = e;
            rv[e]   = -INFINITY;
        }
    }
    std::vector<int32_t> id((size_t) k);
    std::vector<float>   pk((size_t) k);
    std::vector<uint8_t> picked((size_t) n_exp, 0);
    for (int kk = 0; kk < k; ++kk) {
        const int e = prior_argmax(sel.data(), n_exp);
        id[kk]    = e;
        pk[kk]    = p[e];
        sel[e]    = -INFINITY;
        picked[e] = 1;
    }
    if ((flags & GGML_FN_L22_PRIOR_F_RESIDENT) && mask) {
        for (int r = std::max(0, r0); r < k; ++r) {
            if (mask[id[r]] != 0.0f) {
                continue;
            }
            int   be = -1;
            float bv = -INFINITY;
            for (int e = 0; e < n_exp; ++e) {
                if (mask[e] != 0.0f && !picked[e] && p[e] > bv) {
                    bv = p[e];
                    be = e;
                }
            }
            if (be >= 0 && bv >= pk[r]*expnm) {
                id[r]      = be;
                pk[r]      = bv;
                picked[be] = 1;
            }
        }
    }
    // the kernel's sum: lane (id & 31) adds its entries in rank order, then the butterfly of warp_reduce_sum
    float part[32] = {};
    for (int kk = 0; kk < k; ++kk) {
        part[id[kk] & 31] += pk[kk];
    }
    for (int off = 16; off > 0; off >>= 1) {
        float nw[32];
        for (int l = 0; l < 32; ++l) {
            nw[l] = part[l] + part[l ^ off];
        }
        memcpy(part, nw, sizeof(part));
    }
    const float inv = with_norm ? 1.0f/fmaxf(part[0], clamp) : 1.0f;
    for (int kk = 0; kk < k; ++kk) {
        ids[kk] = id[kk];
        const float o = with_norm ? pk[kk]*inv : pk[kk];
        w[kk] = o*wscale;
    }
}

void llama_fn_prior_job_counts(const int32_t * ids, const int32_t * raw, int n_tokens, int k, const int32_t * tbl,
        int32_t miss, int n_exp, uint8_t * scratch, uint32_t * cold_used, uint32_t * cold_raw, uint32_t * changed) {
    uint32_t cu = 0;
    uint32_t cr = 0;
    uint32_t ch = 0;
    const int n = n_tokens*k;
    auto cold = [&](int32_t e) { return tbl == nullptr || tbl[e] == miss; };
    for (int a = 0; a < n; ++a) {
        const int32_t e = ids[a];
        if (e >= 0 && e < n_exp && !(scratch[e] & 1)) {
            scratch[e] |= 1;
            cu += cold(e) ? 1 : 0;
        }
    }
    if (raw) {
        for (int a = 0; a < n; ++a) {
            const int32_t e = raw[a];
            if (e >= 0 && e < n_exp && !(scratch[e] & 2)) {
                scratch[e] |= 2;
                cr += cold(e) ? 1 : 0;
            }
        }
        for (int t = 0; t < n_tokens; ++t) {
            for (int i = 0; i < k; ++i) {
                const int32_t e = ids[t*k + i];
                bool found = false;
                for (int j = 0; j < k && !found; ++j) {
                    found = raw[t*k + j] == e;
                }
                ch += found ? 0 : 1;
            }
        }
    }
    for (int a = 0; a < n; ++a) {
        if (ids[a] >= 0 && ids[a] < n_exp) {
            scratch[ids[a]] = 0;
        }
        if (raw && raw[a] >= 0 && raw[a] < n_exp) {
            scratch[raw[a]] = 0;
        }
    }
    if (cold_used) {
        *cold_used = cu;
    }
    if (cold_raw) {
        *cold_raw = cr;
    }
    if (changed) {
        *changed = ch;
    }
}
