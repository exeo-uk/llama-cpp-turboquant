// [TAG_FN_L22_POOL_STAGGER] [TAG_FN_L22_PRETOUCH] [TAG_FN_L22_JOBPF] [TAG_FN_L22_X_TAILSKIP] [TAG_FN_L22_SETROWS] round 22 P8:
// CPU unit test of the CPU MoE pool changes. Exactness: the staggered walk (every split, with and without the dataflow
// counters, solo, after a given prefetch), the pre-touch helper and the job read-ahead give the pool's answer bit for bit.
// The tail rule (rank and weight) equals the same job with the dropped experts marked hot in the table, in the pool and in
// the sync op (GGML_OP_MOE_HOST_SUM with GGML_MOE_BRIDGE_JOB_TAIL), for any thread count. SET_ROWS F16 -> F32 on many
// threads (#30454). No GPU, no model. Exit code 0 iff every check passes.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-moe-bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h> // [TAG_FN_L22_JOBPF] r22 T5: K32EmptyWorkingSet for the residency probe test
#endif

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

constexpr int64_t N_EMBD = 256;
constexpr int64_t N_FF   = 128;
constexpr int64_t N_EXP  = 64;
constexpr int     N_USED = 8;
constexpr int     N_SLOTS = 24; // hot slots: table[e] in [0, N_SLOTS) is hot, N_SLOTS = miss

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
    ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
    w.ctx  = ggml_init(ip);
    w.up   = ggml_new_tensor_3d(w.ctx, t_gu, N_EMBD, N_FF, N_EXP);
    w.gate = ggml_new_tensor_3d(w.ctx, t_gu, N_EMBD, N_FF, N_EXP);
    w.down = ggml_new_tensor_3d(w.ctx, t_d,  N_FF, N_EMBD, N_EXP);
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
        // distinct experts per token; from token 1 on, three of token 0's experts come back at other ranks, so an
        // expert's best rank over the window mixes ranks of several tokens
        int32_t lst[N_USED];
        std::vector<char> used((size_t) N_EXP, 0);
        for (int i = 0; i < N_USED; ++i) {
            lst[i] = -1;
        }
        if (t > 0) {
            const int src[3] = { 0, 1, 5 };
            for (int k = 0; k < 3; ++k) {
                const int32_t e   = r.ids[src[k]];
                const int     pos = (src[k]*3 + t*2 + 1) % N_USED;
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
        std::vector<int32_t> perm(lst, lst + N_USED);
        std::vector<float> ws((size_t) N_USED);
        float sum = 0.0f;
        for (int i = 0; i < N_USED; ++i) {
            ws[i] = std::exp(-0.45f*i)*(0.8f + 0.4f*(float) (rng() % 1000)/1000.0f);
        }
        std::sort(ws.begin(), ws.end(), [](float a, float b) { return a > b; });
        for (float v : ws) {
            sum += v;
        }
        for (int i = 0; i < N_USED; ++i) {
            r.ids[(size_t) t*N_USED + i] = perm[i];
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
        if (rng() % 3 == 0) {
            tbl[e] = s++;
        }
    }
    return tbl;
}

struct pool_cfg {
    int   nt       = 4;
    int   split    = 0;
    bool  dflow    = false;
    int   stagger  = 0;
    bool  pretouch = false;
    int   jobpf    = 0;
    bool  solo     = false;
    bool  given_pf = false; // a given prefetch list before the job
};

// one pool job (a fresh pool), weighted (w) or plain; tail fields on the job
std::vector<float> run_pool(const weights & W, const routing & r, const int32_t * table, bool weighted, const pool_cfg & cfg,
        int tail_rank, float tail_w, int pool_tail_rank = 0, float pool_tail_w = 0.0f) {
    ggml_cpu_moe_pool_params pp = ggml_cpu_moe_pool_params_default(cfg.nt);
    pp.split    = cfg.split;
    pp.dflow    = cfg.dflow;
    pp.stagger  = cfg.stagger;
    pp.pretouch = cfg.pretouch;
    pp.jobpf    = cfg.jobpf;
    pp.tail_rank = pool_tail_rank;
    pp.tail_w    = pool_tail_w;
    ggml_cpu_moe_pool * pool = ggml_cpu_moe_pool_new(&pp);
    if (pool == nullptr) {
        TCHECK(false, "pool_new");
        return {};
    }
    if (cfg.solo) {
        ggml_cpu_moe_pool_set_solo(pool, true);
    }
    ggml_cpu_moe_layer L = { W.up, W.gate, W.down, table, N_SLOTS };
    std::vector<float> out(weighted ? (size_t) r.T*N_EMBD : (size_t) r.T*N_USED*N_EMBD, -7.0f);
    if (cfg.given_pf && cfg.nt >= 2) {
        // the device's prediction for this layer: the job's own ids (by rank) plus a few others
        std::vector<int32_t> list = r.ids;
        for (int i = 0; i < r.T; ++i) {
            list[(size_t) i*N_USED + N_USED - 1] = (int32_t) ((i*13 + 5) % N_EXP);
        }
        ggml_cpu_moe_prefetch_job pj = {};
        pj.layer    = &L;
        pj.n_tokens = r.T;
        pj.k        = N_USED;
        pj.mode     = 0;
        pj.list     = list.data();
        TCHECK(ggml_cpu_moe_prefetch(pool, &pj) == GGML_STATUS_SUCCESS, "prefetch");
    }
    ggml_cpu_moe_job job = {};
    job.layer     = &L;
    job.n_tokens  = r.T;
    job.n_used    = N_USED;
    job.x         = r.x.data();
    job.ids       = r.ids.data();
    job.w         = weighted ? r.w.data() : nullptr;
    job.out       = out.data();
    job.tail_rank = (int8_t) tail_rank;
    job.tail_w    = tail_w;
    job.l22_flags = tail_rank > 0 || tail_w > 0.0f ? GGML_MOE_BRIDGE_JOB_TAIL : 0;
    TCHECK(ggml_cpu_moe_run(pool, &job) == GGML_STATUS_SUCCESS, "run");
    // a second job of the same layer: the read-ahead / pre-touch bookkeeping sees the experts as recent now
    std::vector<float> out2(out.size(), -9.0f);
    job.out = out2.data();
    TCHECK(ggml_cpu_moe_run(pool, &job) == GGML_STATUS_SUCCESS, "run 2");
    TCHECK(memcmp(out.data(), out2.data(), out.size()*sizeof(float)) == 0, "second job differs");
    if (cfg.pretouch || cfg.jobpf > 0) {
        // the helper works asynchronously: wait (bounded) until it handled the first job's entries
        ggml_cpu_moe_pool_stats st;
        for (int k = 0; k < 400; ++k) {
            ggml_cpu_moe_pool_get_stats(pool, &st, false);
            // [TAG_FN_L22_JOBPF] r22 T5: every candidate is either asked for (queued) or left out by the residency probe /
            // the lazy mode (skipped; heap weights here are in the working set)
            if ((cfg.jobpf == 0 || st.jpf_queued + st.jpf_skipped >= 1) && (!cfg.pretouch || st.pt_experts >= 1)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (cfg.jobpf > 0) { // the weights here are heap memory: the OS may refuse the read-ahead, the job must not care
            TCHECK(st.jpf_queued + st.jpf_skipped >= 1 && (st.jpf_calls == 0 || st.jpf_ranges >= 3*st.jpf_calls),
                    "jobpf queued %llu skipped %llu calls %llu ranges %llu", (unsigned long long) st.jpf_queued,
                    (unsigned long long) st.jpf_skipped, (unsigned long long) st.jpf_calls, (unsigned long long) st.jpf_ranges);
        }
        if (cfg.pretouch) {
            TCHECK(st.pt_queued >= 1 && st.pt_experts >= 1 && st.pt_pages >= 3*st.pt_experts, "pretouch queued %llu touched %llu pages %llu",
                    (unsigned long long) st.pt_queued, (unsigned long long) st.pt_experts, (unsigned long long) st.pt_pages);
        }
    }
    ggml_cpu_moe_pool_free(pool);
    return out;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// the experts the tail rule drops, as hot entries of a copy of the table (value 0 != miss)
std::vector<int32_t> tail_table(const routing & r, const int32_t * table, int R, float Wb, bool weighted) {
    std::vector<int32_t> t2((size_t) N_EXP, N_SLOTS);
    if (table) {
        t2.assign(table, table + N_EXP);
    }
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
        const bool drop = (R > 0 && minr >= R) || (weighted && Wb > 0.0f && maxw < Wb);
        if (drop) {
            t2[e] = 0;
        }
    }
    return t2;
}

int n_dropped(const std::vector<int32_t> & t2, const int32_t * table) {
    int n = 0;
    for (int e = 0; e < N_EXP; ++e) {
        n += t2[e] != N_SLOTS && !(table && table[e] != N_SLOTS) ? 1 : 0;
    }
    return n;
}

void test_exact(const weights & W, const char * name) {
    const std::vector<int32_t> tbl = make_table(11);
    for (int T : { 1, 3, 8 }) {
        const routing r = make_routing(T, 100 + T);
        for (int weighted = 0; weighted <= 1; ++weighted) {
            for (int use_tbl = 0; use_tbl <= 1; ++use_tbl) {
                const int32_t * tp = use_tbl ? tbl.data() : nullptr;
                pool_cfg ref;
                ref.nt = 1;
                const std::vector<float> o_ref = run_pool(W, r, tp, weighted, ref, 0, 0.0f);
                for (int split = 0; split <= 2; ++split) {
                    for (int dflow = 0; dflow <= 1; ++dflow) {
                        for (int stag : { 0, 2, 4 }) {
                            for (int variant = 0; variant < 4; ++variant) {
                                pool_cfg c;
                                c.nt       = variant == 3 ? 3 : 4;
                                c.split    = split;
                                c.dflow    = dflow;
                                c.stagger  = stag;
                                c.pretouch = variant == 1 || variant == 2;
                                c.jobpf    = variant == 1 ? 1 : variant == 2 ? 2 : 0;
                                c.solo     = variant == 3;
                                c.given_pf = variant >= 2;
                                const std::vector<float> o = run_pool(W, r, tp, weighted, c, 0, 0.0f);
                                TCHECK(same(o, o_ref), "%s T %d weighted %d table %d split %d dflow %d stagger %d variant %d: "
                                        "differs from the reference", name, T, weighted, use_tbl, split, dflow, stag, variant);
                            }
                        }
                    }
                }
            }
        }
    }
}

void test_tail(const weights & W, const char * name) {
    const std::vector<int32_t> tbl = make_table(23);
    int n_drop_total = 0;
    for (int T : { 1, 3, 8 }) {
        const routing r = make_routing(T, 300 + T);
        for (int weighted = 0; weighted <= 1; ++weighted) {
            for (int use_tbl = 0; use_tbl <= 1; ++use_tbl) {
                const int32_t * tp = use_tbl ? tbl.data() : nullptr;
                struct rule { int R; float Wb; };
                const rule rules[] = { { 3, 0.0f }, { 6, 0.0f }, { 99, 0.06f }, { 5, 0.03f }, { 1, 0.0f } };
                for (const rule & ru : rules) {
                    if (!weighted && ru.R >= N_USED) {
                        continue; // the weight rule needs w
                    }
                    const std::vector<int32_t> t2 = tail_table(r, tp, ru.R, ru.Wb, weighted);
                    n_drop_total += n_dropped(t2, tp);
                    pool_cfg ref;
                    ref.nt = 1;
                    const std::vector<float> o_ref = run_pool(W, r, t2.data(), weighted, ref, 0, 0.0f);
                    for (int split = 0; split <= 2; split += 2) {
                        for (int stag : { 0, 4 }) {
                            for (int nt : { 1, 4 }) {
                                pool_cfg c;
                                c.nt      = nt;
                                c.split   = split;
                                c.dflow   = split == 2;
                                c.stagger = stag;
                                const std::vector<float> o = run_pool(W, r, tp, weighted, c, ru.R, ru.Wb);
                                TCHECK(same(o, o_ref), "%s T %d weighted %d table %d rule R %d W %.2f split %d stagger %d nt %d: "
                                        "the tail job differs from the marked table", name, T, weighted, use_tbl, ru.R, ru.Wb,
                                        split, stag, nt);
                            }
                        }
                    }
                }
            }
        }
    }
    TCHECK(n_drop_total > 0, "%s: the rules dropped nothing (test too weak)", name);
}

// the sync op of a graph against the pool, with and without the TAIL flag (rule set process-wide by a pool's creation)
std::vector<float> run_host_sum(const weights & W, const routing & r, const std::vector<int32_t> * table, bool tail_flag, int nt) {
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
    if (tail_flag) {
        ggml_moe_host_sum_set_flags(out, GGML_MOE_BRIDGE_JOB_TAIL);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, nt);
    TCHECK(st == GGML_STATUS_SUCCESS, "host_sum compute");
    std::vector<float> o((size_t) r.T*N_EMBD);
    memcpy(o.data(), out->data, o.size()*sizeof(float));
    ggml_free(ctx);
    return o;
}

void test_sync_op(const weights & W, const char * name) {
    const std::vector<int32_t> tbl = make_table(31);
    const int   R  = 3;
    const float Wb = 0.05f;
    for (int T : { 1, 3 }) {
        const routing r = make_routing(T, 500 + T);
        pool_cfg c;
        c.nt    = 4;
        c.split = 2;
        c.dflow = true;
        // the pool's creation sets the sync op's rule (the bridge passes the same R / W to both)
        const std::vector<float> o_tail = run_pool(W, r, tbl.data(), true, c, R, Wb, R, Wb);
        const std::vector<float> o_none = run_pool(W, r, tbl.data(), true, c, 0, 0.0f, R, Wb);
        const int n_drop = n_dropped(tail_table(r, tbl.data(), R, Wb, true), tbl.data());
        TCHECK(n_drop == 0 || !same(o_tail, o_none), "%s T %d: the tail rule dropped %d experts and changed nothing", name, T, n_drop);
        printf("%s: sync op T %d: the tail rule drops %d cold experts\n", name, T, n_drop);
        for (int nt : { 1, 3, 8 }) {
            const std::vector<float> s_tail = run_host_sum(W, r, &tbl, true, nt);
            const std::vector<float> s_none = run_host_sum(W, r, &tbl, false, nt);
            TCHECK(same(s_tail, o_tail), "%s T %d nt %d: the sync op with TAIL differs from the bridged job", name, T, nt);
            TCHECK(same(s_none, o_none), "%s T %d nt %d: the sync op without TAIL differs from the plain job", name, T, nt);
        }
    }
}

// [TAG_FN_L22_JOBPF] r22 T5: the residency probe. Lazy while the weights are in the working set (one job in 8 with
// candidates probes, no read-ahead); after the working set is emptied the next probe finds pages out: read-ahead requests go
// out, every job probes for 2 s, and the outputs stay the reference's bit for bit (Windows; elsewhere: the lazy part only)
void test_jobpf_probe(const weights & W, const char * name) {
    const std::vector<int32_t> tbl = make_table(41);
    const routing r = make_routing(3, 4242);
    pool_cfg ref;
    ref.nt = 1;
    const std::vector<float> o_ref = run_pool(W, r, tbl.data(), true, ref, 0, 0.0f);
    ggml_cpu_moe_pool_params pp = ggml_cpu_moe_pool_params_default(4);
    pp.split = 2;
    pp.jobpf = 2; // every cold expert is a candidate in every job
    ggml_cpu_moe_pool * pool = ggml_cpu_moe_pool_new(&pp);
    TCHECK(pool != nullptr, "pool_new");
    if (pool == nullptr) {
        return;
    }
    ggml_cpu_moe_layer L = { W.up, W.gate, W.down, tbl.data(), N_SLOTS };
    std::vector<float> out((size_t) r.T*N_EMBD, -7.0f);
    ggml_cpu_moe_job job = {};
    job.layer    = &L;
    job.n_tokens = r.T;
    job.n_used   = N_USED;
    job.x        = r.x.data();
    job.ids      = r.ids.data();
    job.w        = r.w.data();
    job.out      = out.data();
    ggml_cpu_moe_pool_stats st;
    for (int j = 0; j < 7; ++j) { // jobs 1..7: lazy, no probe, no read-ahead
        TCHECK(ggml_cpu_moe_run(pool, &job) == GGML_STATUS_SUCCESS, "run %d", j);
        TCHECK(same(out, o_ref), "%s: lazy job %d differs from the reference", name, j);
    }
    ggml_cpu_moe_pool_get_stats(pool, &st, false);
    TCHECK(st.jpf_probes == 0 && st.jpf_calls == 0 && st.jpf_queued == 0 && st.jpf_skipped >= 7,
            "%s: lazy jobs: probes %llu calls %llu queued %llu skipped %llu", name, (unsigned long long) st.jpf_probes,
            (unsigned long long) st.jpf_calls, (unsigned long long) st.jpf_queued, (unsigned long long) st.jpf_skipped);
#if defined(_WIN32)
    K32EmptyWorkingSet(GetCurrentProcess()); // the weights' pages leave the working set (they stay in RAM, as standby)
    TCHECK(ggml_cpu_moe_run(pool, &job) == GGML_STATUS_SUCCESS, "run 8");
    TCHECK(same(out, o_ref), "%s: the probed job differs from the reference", name);
    ggml_cpu_moe_pool_get_stats(pool, &st, false);
    TCHECK(st.jpf_probes == 1 && st.jpf_out >= 1 && st.jpf_calls >= 1 && st.jpf_queued >= 1,
            "%s: after the working set was emptied: probes %llu out %llu calls %llu queued %llu", name,
            (unsigned long long) st.jpf_probes, (unsigned long long) st.jpf_out, (unsigned long long) st.jpf_calls,
            (unsigned long long) st.jpf_queued);
    // urgent now: the next job probes too (its pages are back in the working set: nothing out, nothing asked for)
    const uint64_t calls = st.jpf_calls;
    TCHECK(ggml_cpu_moe_run(pool, &job) == GGML_STATUS_SUCCESS, "run 9");
    TCHECK(same(out, o_ref), "%s: the urgent job differs from the reference", name);
    ggml_cpu_moe_pool_get_stats(pool, &st, false);
    TCHECK(st.jpf_probes == 2, "%s: the job after a probe that found pages out did not probe (probes %llu)", name,
            (unsigned long long) st.jpf_probes);
    TCHECK(st.jpf_calls == calls, "%s: a read-ahead for resident pages (calls %llu -> %llu)", name, (unsigned long long) calls,
            (unsigned long long) st.jpf_calls);
#else
    TCHECK(ggml_cpu_moe_run(pool, &job) == GGML_STATUS_SUCCESS, "run 8");
    TCHECK(same(out, o_ref), "%s: the probed job differs from the reference", name);
    ggml_cpu_moe_pool_get_stats(pool, &st, false);
    TCHECK(st.jpf_probes <= 1, "%s: probes %llu after 8 lazy jobs", name, (unsigned long long) st.jpf_probes);
#endif
    ggml_cpu_moe_pool_free(pool);
}

void test_set_rows() {
    const int64_t ne00 = 1000;
    const int64_t nr   = 37;
    const int64_t ndst = 64;
    ggml_init_params ip = { (size_t) 16*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne00, nr);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nr);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ndst);
    std::mt19937 rng(7);
    std::vector<float> ref((size_t) (ne00*nr));
    for (int64_t i = 0; i < ne00*nr; ++i) {
        const float v = (float) ((int) (rng() % 2001) - 1000)/256.0f;
        ((ggml_fp16_t *) src->data)[i] = ggml_fp32_to_fp16(v);
        ref[i] = ggml_fp16_to_fp32(((ggml_fp16_t *) src->data)[i]);
    }
    for (int64_t i = 0; i < nr; ++i) {
        ((int64_t *) idx->data)[i] = (i*5 + 3) % ndst;
    }
    memset(dst->data, 0, ggml_nbytes(dst));
    ggml_tensor * out = ggml_set_rows(ctx, dst, src, idx);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    TCHECK(ggml_graph_compute_with_ctx(ctx, gf, 16) == GGML_STATUS_SUCCESS, "set_rows compute");
    bool ok = true;
    for (int64_t i = 0; i < nr && ok; ++i) {
        const int64_t d = (i*5 + 3) % ndst;
        ok = memcmp((const float *) out->data + d*ne00, ref.data() + i*ne00, (size_t) ne00*sizeof(float)) == 0;
    }
    TCHECK(ok, "set_rows F16 -> F32 rows differ");
    ggml_free(ctx);
}

} // namespace

int main() {
    ggml_cpu_init();
    {
        weights W;
        make_weights(W, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1);
        test_exact(W, "q8_0");
        test_tail(W, "q8_0");
    }
    {
        weights W;
        make_weights(W, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, 2);
        test_exact(W, "q4_K/q5_1");
        test_tail(W, "q4_K/q5_1");
        test_sync_op(W, "q4_K/q5_1");
        test_jobpf_probe(W, "q4_K/q5_1");
    }
    test_set_rows();
    printf("test-fn-l22-pool: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
