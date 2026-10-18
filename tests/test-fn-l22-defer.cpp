// [TAG_FN_L22_X_DEFER] round 22 P11: CPU unit test of the heavy / light split of a deferred MoE job (opt-in X2 DEFER).
// The DEFER part equals the same job with the light (and tail) experts marked hot in the table, the LIGHT part the job
// with everything but the light experts marked hot: bit for bit, in the pool (any split, thread count, dataflow) and in
// the sync op (GGML_OP_MOE_HOST_SUM with the flags). Unset flags give today's job. DEFER + LIGHT = the full sum. A graph
// with both sum nodes reports the routing once. No GPU, no model. Exit code 0 iff every check passes.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-moe-bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

namespace {

constexpr int64_t N_EMBD  = 256;
constexpr int64_t N_FF    = 128;
constexpr int64_t N_EXP   = 32;
constexpr int     N_USED  = 8;
constexpr int     N_SLOTS = 8; // table[e] in [0, N_SLOTS) is hot, N_SLOTS = miss

struct weights {
    ggml_context * ctx = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ~weights() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

void fill_q(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t n_per_row = t->ne[0];
    const int64_t nrows     = t->ne[1]*t->ne[2];
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> f((size_t) (n_per_row*nrows));
    for (float & v : f) {
        v = nd(rng);
    }
    ggml_quantize_chunk(t->type, f.data(), t->data, 0, nrows, n_per_row, nullptr);
}

void make_weights(weights & w, ggml_type t_gu, ggml_type t_d, uint32_t seed) {
    ggml_init_params ip = { (size_t) 32*1024*1024, nullptr, false };
    w.ctx  = ggml_init(ip);
    w.up   = ggml_new_tensor_3d(w.ctx, t_gu, N_EMBD, N_FF, N_EXP);
    w.gate = ggml_new_tensor_3d(w.ctx, t_gu, N_EMBD, N_FF, N_EXP);
    w.down = ggml_new_tensor_3d(w.ctx, t_d,  N_FF, N_EMBD, N_EXP);
    ggml_set_name(w.up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(w.gate, "blk.0.ffn_gate_exps.weight"); // the routing observer reports only this name
    ggml_set_name(w.down, "blk.0.ffn_down_exps.weight");
    std::mt19937 rng(seed);
    fill_q(w.up, rng);
    fill_q(w.gate, rng);
    fill_q(w.down, rng);
}

struct routing {
    int T = 1;
    std::vector<float>   x;   // [T][N_EMBD]
    std::vector<int32_t> ids; // [T][N_USED], distinct per token, by rank
    std::vector<float>   w;   // [T][N_USED], descending per token, sum 1
};

// experts recur over the tokens at other ranks, so an expert's best rank and largest weight mix tokens (light at one
// token, heavy at another)
routing make_routing(int T, uint32_t seed) {
    routing r;
    r.T = T;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    r.x.resize((size_t) T*N_EMBD);
    for (float & v : r.x) {
        v = nd(rng);
    }
    r.ids.resize((size_t) T*N_USED);
    r.w.resize((size_t) T*N_USED);
    for (int t = 0; t < T; ++t) {
        int32_t lst[N_USED];
        std::vector<char> used((size_t) N_EXP, 0);
        for (int i = 0; i < N_USED; ++i) {
            lst[i] = -1;
        }
        if (t > 0) {
            const int src[3] = { 0, 2, 6 };
            for (int k = 0; k < 3; ++k) {
                const int32_t e   = r.ids[src[k]];
                const int     pos = (src[k]*5 + t*3 + 2) % N_USED;
                if (lst[pos] < 0 && !used[e]) {
                    lst[pos] = e;
                    used[e]  = 1;
                }
            }
        }
        for (int i = 0; i < N_USED; ++i) {
            while (lst[i] < 0) {
                const int32_t e = (int32_t) (rng() % N_EXP);
                if (!used[e]) {
                    lst[i]  = e;
                    used[e] = 1;
                }
            }
        }
        std::vector<float> ws((size_t) N_USED);
        float sum = 0.0f;
        for (int i = 0; i < N_USED; ++i) {
            ws[i] = std::exp(-0.5f*i)*(0.8f + 0.4f*(float) (rng() % 1000)/1000.0f);
        }
        std::sort(ws.begin(), ws.end(), [](float a, float b) { return a > b; });
        for (float v : ws) {
            sum += v;
        }
        for (int i = 0; i < N_USED; ++i) {
            r.ids[(size_t) t*N_USED + i] = lst[i];
            r.w[(size_t) t*N_USED + i]   = ws[i]/sum;
        }
    }
    return r;
}

std::vector<int32_t> make_table(uint32_t seed) {
    std::vector<int32_t> tbl((size_t) N_EXP, N_SLOTS);
    std::mt19937 rng(seed);
    int s = 0;
    for (int e = 0; e < N_EXP && s < N_SLOTS; ++e) {
        if (rng() % 4 == 0) {
            tbl[e] = s++;
        }
    }
    return tbl;
}

struct rule {
    int   tail_r = 0;
    float tail_w = 0.0f;
    int   xd_r   = 0;
    float xd_w   = 0.0f;
};

// per routed cold expert: 0 not routed or hot, 1 tail, 2 light, 3 heavy (kept by DEFER)
std::vector<int> classify(const routing & r, const int32_t * table, const rule & ru) {
    std::vector<int> cls((size_t) N_EXP, 0);
    for (int e = 0; e < N_EXP; ++e) {
        if (table && table[e] != N_SLOTS) {
            continue;
        }
        int   minr = 1 << 20;
        float maxw = -1.0f;
        bool  seen = false;
        for (int t = 0; t < r.T; ++t) {
            for (int s = 0; s < N_USED; ++s) {
                if (r.ids[(size_t) t*N_USED + s] == e) {
                    seen = true;
                    minr = std::min(minr, s);
                    maxw = std::max(maxw, r.w[(size_t) t*N_USED + s]);
                }
            }
        }
        if (!seen) {
            continue;
        }
        const bool tail  = (ru.tail_r > 0 && minr >= ru.tail_r) || (ru.tail_w > 0.0f && maxw < ru.tail_w);
        const bool light = !tail && ((ru.xd_r > 0 && minr >= ru.xd_r) || (ru.xd_w > 0.0f && maxw < ru.xd_w));
        cls[e] = tail ? 1 : light ? 2 : 3;
    }
    return cls;
}

// the table that drops what the part drops (value 0 = hot): part 1 DEFER keeps class 3, part 2 LIGHT keeps class 2
std::vector<int32_t> part_table(const std::vector<int> & cls, const int32_t * table, int part) {
    std::vector<int32_t> t2((size_t) N_EXP, N_SLOTS);
    if (table) {
        t2.assign(table, table + N_EXP);
    }
    for (int e = 0; e < N_EXP; ++e) {
        if (cls[e] == 0) {
            continue;
        }
        const bool keep = part == 0 ? cls[e] != 1 : part == 1 ? cls[e] == 3 : cls[e] == 2;
        if (!keep) {
            t2[e] = 0;
        }
    }
    return t2;
}

int count_cls(const std::vector<int> & cls, int c) {
    int n = 0;
    for (int v : cls) {
        n += v == c ? 1 : 0;
    }
    return n;
}

struct pool_cfg {
    int  nt    = 4;
    int  split = 0;
    bool dflow = false;
    bool solo  = false;
};

// one weighted pool job (a fresh pool); part 0 plain (rule fields set, no defer flag), 1 DEFER, 2 LIGHT
std::vector<float> run_pool(const weights & W, const routing & r, const int32_t * table, const pool_cfg & cfg, const rule & ru,
        int part, int32_t * n_light = nullptr, const rule * pool_rule = nullptr) {
    ggml_cpu_moe_pool_params pp = ggml_cpu_moe_pool_params_default(cfg.nt);
    pp.split = cfg.split;
    pp.dflow = cfg.dflow;
    if (pool_rule) {
        pp.tail_rank  = pool_rule->tail_r;
        pp.tail_w     = pool_rule->tail_w;
        pp.defer_rank = pool_rule->xd_r;
        pp.defer_w    = pool_rule->xd_w;
    }
    ggml_cpu_moe_pool * pool = ggml_cpu_moe_pool_new(&pp);
    if (pool == nullptr) {
        TCHECK(false, "pool_new");
        return {};
    }
    if (cfg.solo) {
        ggml_cpu_moe_pool_set_solo(pool, true);
    }
    ggml_cpu_moe_layer L = { W.up, W.gate, W.down, table, N_SLOTS };
    std::vector<float> out((size_t) r.T*N_EMBD, -7.0f);
    ggml_cpu_moe_job job = {};
    job.layer      = &L;
    job.n_tokens   = r.T;
    job.n_used     = N_USED;
    job.x          = r.x.data();
    job.ids        = r.ids.data();
    job.w          = r.w.data();
    job.out        = out.data();
    job.tail_rank  = (int8_t) ru.tail_r;
    job.tail_w     = ru.tail_w;
    job.defer_rank = (int8_t) ru.xd_r;
    job.defer_w    = ru.xd_w;
    job.l22_flags  = (ru.tail_r > 0 || ru.tail_w > 0.0f ? GGML_MOE_BRIDGE_JOB_TAIL : 0) |
                     (part == 1 ? GGML_MOE_BRIDGE_JOB_DEFER : part == 2 ? GGML_MOE_BRIDGE_JOB_LIGHT : 0);
    job.n_light    = n_light;
    TCHECK(ggml_cpu_moe_run(pool, &job) == GGML_STATUS_SUCCESS, "run");
    ggml_cpu_moe_pool_free(pool);
    return out;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

bool all_pos_zero(const std::vector<float> & a) {
    for (float v : a) {
        uint32_t u;
        memcpy(&u, &v, sizeof(u));
        if (u != 0u) {
            return false;
        }
    }
    return !a.empty();
}

const char * part_name(int part) {
    return part == 1 ? "DEFER" : part == 2 ? "LIGHT" : "plain";
}

void test_partition(const weights & W, const char * name) {
    const std::vector<int32_t> tbl = make_table(17);
    const rule rules[] = {
        { 0, 0.0f, 0, 0.10f },    // W only (the build default)
        { 0, 0.0f, 7, 0.0f },     // R only (DEFER R7)
        { 0, 0.0f, 5, 0.06f },    // W + R
        { 7, 0.05f, 0, 0.10f },   // X1 tail on top: tail experts in neither part
        { 6, 0.0f, 4, 0.0f },
    };
    int n_light_total = 0;
    int n_heavy_total = 0;
    for (int T : { 1, 2, 3, 16 }) {
        const routing r = make_routing(T, 100 + T);
        for (int use_tbl = 0; use_tbl <= 1; ++use_tbl) {
            const int32_t * tp = use_tbl ? tbl.data() : nullptr;
            for (const rule & ru : rules) {
                const std::vector<int> cls = classify(r, tp, ru);
                n_light_total += count_cls(cls, 2);
                n_heavy_total += count_cls(cls, 3);
                for (int part = 1; part <= 2; ++part) {
                    const std::vector<int32_t> t2 = part_table(cls, tp, part);
                    pool_cfg ref;
                    ref.nt = 1;
                    const std::vector<float> o_ref = run_pool(W, r, t2.data(), ref, rule(), 0);
                    for (int split = 0; split <= 2; ++split) {
                        for (int nt : { 1, 4, 8 }) {
                            for (int dflow = 0; dflow <= 1; ++dflow) {
                                pool_cfg c;
                                c.nt    = nt;
                                c.split = split;
                                c.dflow = dflow;
                                c.solo  = nt == 4 && split == 1;
                                int32_t nl = -1;
                                const std::vector<float> o = run_pool(W, r, tp, c, ru, part, &nl);
                                TCHECK(same(o, o_ref), "%s T %d table %d rule (%d %.2f %d %.2f) %s split %d nt %d dflow %d: "
                                        "differs from the marked table", name, T, use_tbl, ru.tail_r, ru.tail_w, ru.xd_r, ru.xd_w,
                                        part_name(part), split, nt, dflow);
                                TCHECK(nl == count_cls(cls, 2), "%s T %d %s: n_light %d, expected %d", name, T, part_name(part),
                                        nl, count_cls(cls, 2));
                            }
                        }
                    }
                }
                // sum sanity: DEFER + LIGHT = the job without the defer split (tail rule only), up to the float order
                pool_cfg c;
                c.nt = 4;
                const std::vector<float> oa = run_pool(W, r, tp, c, ru, 1);
                const std::vector<float> ob = run_pool(W, r, tp, c, ru, 2);
                const std::vector<float> of = run_pool(W, r, part_table(cls, tp, 0).data(), c, rule(), 0);
                double max_abs = 0.0, max_err = 0.0;
                for (size_t i = 0; i < of.size(); ++i) {
                    max_abs = std::max(max_abs, (double) std::fabs(of[i]));
                    max_err = std::max(max_err, (double) std::fabs(oa[i] + ob[i] - of[i]));
                }
                TCHECK(max_err <= 1e-5*std::max(1.0, max_abs), "%s T %d table %d: DEFER + LIGHT differs from the full sum by %g (max %g)",
                        name, T, use_tbl, max_err, max_abs);
            }
        }
    }
    TCHECK(n_light_total > 0 && n_heavy_total > 0, "%s: test too weak (light %d heavy %d)", name, n_light_total, n_heavy_total);
    printf("%s: partition: %d light and %d heavy cold experts over the cases\n", name, n_light_total, n_heavy_total);
}

void test_edges(const weights & W, const char * name) {
    const std::vector<int32_t> tbl = make_table(29);
    pool_cfg c;
    c.nt    = 4;
    c.split = 2;
    c.dflow = true;
    for (int T : { 1, 3 }) {
        const routing r = make_routing(T, 700 + T);
        // strict < at w == Wd: a weight of the routing as the bound keeps that expert heavy
        {
            const float wd = r.w[3];
            rule ru;
            ru.xd_w = wd;
            const std::vector<int> cls = classify(r, nullptr, ru);
            TCHECK(cls[r.ids[3]] == 3, "%s T %d: the expert with weight == Wd must be heavy", name, T);
            const std::vector<float> o   = run_pool(W, r, nullptr, c, ru, 1);
            pool_cfg ref;
            ref.nt = 1;
            const std::vector<float> o_r = run_pool(W, r, part_table(cls, nullptr, 1).data(), ref, rule(), 0);
            TCHECK(same(o, o_r), "%s T %d: strict bound differs", name, T);
        }
        // min rank over the tokens: an expert at rank 2 at token 0 and rank >= R later is not rank-light
        {
            rule ru;
            ru.xd_r = 3;
            const std::vector<int> cls = classify(r, nullptr, ru);
            TCHECK(cls[r.ids[2]] == 3, "%s T %d: an expert at rank 2 of token 0 must be heavy with R 3", name, T);
        }
        // a hot expert is never light, never computed
        {
            rule ru;
            ru.xd_w = 2.0f; // every cold expert is light
            int32_t nl = -1;
            const std::vector<float> oa = run_pool(W, r, tbl.data(), c, ru, 1, &nl);
            TCHECK(all_pos_zero(oa), "%s T %d: DEFER with every cold expert light must be all +0.0", name, T);
            const std::vector<int> cls = classify(r, tbl.data(), ru);
            TCHECK(nl == count_cls(cls, 2) && count_cls(cls, 3) == 0, "%s T %d: all-light n_light %d", name, T, nl);
            for (int e = 0; e < N_EXP; ++e) {
                TCHECK(!(tbl[e] != N_SLOTS && cls[e] != 0), "%s: hot expert %d classified", name, e);
            }
            const std::vector<float> ob = run_pool(W, r, tbl.data(), c, ru, 2);
            const std::vector<float> of = run_pool(W, r, tbl.data(), c, rule(), 0);
            TCHECK(same(ob, of), "%s T %d: LIGHT with every cold expert light must equal the plain job", name, T);
        }
        // no light expert: LIGHT is all +0.0 and n_light 0; DEFER equals the plain job
        {
            rule ru;
            ru.xd_w = 1e-9f;
            int32_t nl = -1;
            const std::vector<float> ob = run_pool(W, r, tbl.data(), c, ru, 2, &nl);
            TCHECK(all_pos_zero(ob) && nl == 0, "%s T %d: LIGHT without light experts: n_light %d", name, T, nl);
            const std::vector<float> oa = run_pool(W, r, tbl.data(), c, ru, 1);
            const std::vector<float> of = run_pool(W, r, tbl.data(), c, rule(), 0);
            TCHECK(same(oa, of), "%s T %d: DEFER without light experts must equal the plain job", name, T);
        }
        // unset identity: rule fields set, no defer flag = today's job (and with the tail rule, today's tail job)
        {
            rule ru;
            ru.xd_r = 5;
            ru.xd_w = 0.1f;
            const std::vector<float> o  = run_pool(W, r, tbl.data(), c, ru, 0);
            const std::vector<float> of = run_pool(W, r, tbl.data(), c, rule(), 0);
            TCHECK(same(o, of), "%s T %d: defer fields without a flag changed the job", name, T);
            rule rt = ru;
            rt.tail_r = 6;
            rule rt0;
            rt0.tail_r = 6;
            TCHECK(same(run_pool(W, r, tbl.data(), c, rt, 0), run_pool(W, r, tbl.data(), c, rt0, 0)),
                    "%s T %d: defer fields changed a tail job", name, T);
        }
    }
}

// the sync op of a graph: one node, or the DEFER and LIGHT nodes (summed by the caller); flags as the graph sets them
std::vector<float> run_host_sum(const weights & W, const routing & r, const std::vector<int32_t> * table, int32_t flags, int nt) {
    ggml_init_params ip = { (size_t) 32*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, r.T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, r.T);
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_USED, r.T);
    memcpy(x->data, r.x.data(), ggml_nbytes(x));
    memcpy(ids->data, r.ids.data(), ggml_nbytes(ids));
    memcpy(w->data, r.w.data(), ggml_nbytes(w));
    ggml_tensor * tt = nullptr;
    if (table) {
        tt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_EXP);
        memcpy(tt->data, table->data(), ggml_nbytes(tt));
    }
    ggml_tensor * out = ggml_moe_host_sum(ctx, x, ids, w, W.up, W.gate, W.down, tt, N_SLOTS);
    if (flags) {
        ggml_moe_host_sum_set_flags(out, flags);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    TCHECK(ggml_graph_compute_with_ctx(ctx, gf, nt) == GGML_STATUS_SUCCESS, "host_sum compute");
    std::vector<float> o((size_t) r.T*N_EMBD);
    memcpy(o.data(), out->data, o.size()*sizeof(float));
    ggml_free(ctx);
    return o;
}

int g_obs_calls = 0;
void obs_count(const char * name, const ggml_tensor * ids, void * ud) {
    (void) name;
    (void) ids;
    (void) ud;
    g_obs_calls++;
}

void test_sync_op(const weights & W, const char * name) {
    const std::vector<int32_t> tbl = make_table(31);
    rule ru;
    ru.tail_r = 7;
    ru.tail_w = 0.0f;
    ru.xd_r   = 0;
    ru.xd_w   = 0.1f;
    for (int T : { 1, 3 }) {
        const routing r = make_routing(T, 500 + T);
        pool_cfg c;
        c.nt    = 4;
        c.split = 2;
        c.dflow = true;
        // the pool's creation sets the sync op's rules (the bridge passes the same values to both)
        const std::vector<float> oa = run_pool(W, r, tbl.data(), c, ru, 1, nullptr, &ru);
        const std::vector<float> ob = run_pool(W, r, tbl.data(), c, ru, 2, nullptr, &ru);
        for (int nt : { 1, 3, 8 }) {
            const std::vector<float> sa = run_host_sum(W, r, &tbl, GGML_MOE_BRIDGE_JOB_TAIL | GGML_MOE_BRIDGE_JOB_DEFER, nt);
            const std::vector<float> sb = run_host_sum(W, r, &tbl, GGML_MOE_BRIDGE_JOB_TAIL | GGML_MOE_BRIDGE_JOB_LIGHT, nt);
            TCHECK(same(sa, oa), "%s T %d nt %d: the sync op DEFER differs from the bridged job", name, T, nt);
            TCHECK(same(sb, ob), "%s T %d nt %d: the sync op LIGHT differs from the bridged job", name, T, nt);
        }
    }
    // a graph with both nodes (as build_moe_ffn makes it in a sync graph) reports the routing once
    {
        const routing r = make_routing(3, 900);
        ggml_init_params ip = { (size_t) 32*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, r.T);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, r.T);
        ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_USED, r.T);
        memcpy(x->data, r.x.data(), ggml_nbytes(x));
        memcpy(ids->data, r.ids.data(), ggml_nbytes(ids));
        memcpy(w->data, r.w.data(), ggml_nbytes(w));
        ggml_tensor * a = ggml_moe_host_sum(ctx, x, ids, w, W.up, W.gate, W.down, nullptr, N_SLOTS);
        ggml_moe_host_sum_set_flags(a, GGML_MOE_BRIDGE_JOB_DEFER);
        ggml_tensor * b = ggml_moe_host_sum(ctx, x, ids, w, W.up, W.gate, W.down, nullptr, N_SLOTS);
        ggml_moe_host_sum_set_flags(b, GGML_MOE_BRIDGE_JOB_LIGHT);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, a);
        ggml_build_forward_expand(gf, b);
        void * old_ud = nullptr;
        ggml_moe_obs_cb_t old = ggml_get_moe_obs_callback(&old_ud);
        g_obs_calls = 0;
        ggml_set_moe_obs_callback(obs_count, nullptr);
        TCHECK(ggml_graph_compute_with_ctx(ctx, gf, 4) == GGML_STATUS_SUCCESS, "compute A + B");
        ggml_set_moe_obs_callback(old, old_ud);
        TCHECK(g_obs_calls == 1, "%s: A + B reported the routing %d times", name, g_obs_calls);
        ggml_free(ctx);
    }
}

} // namespace

int main() {
    ggml_cpu_init();
    {
        weights W;
        make_weights(W, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1);
        test_partition(W, "q8_0");
        test_edges(W, "q8_0");
    }
    {
        weights W;
        make_weights(W, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, 2);
        test_partition(W, "q4_K/q5_1");
        test_edges(W, "q4_K/q5_1");
        test_sync_op(W, "q4_K/q5_1");
    }
    printf("test-fn-l22-defer: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
