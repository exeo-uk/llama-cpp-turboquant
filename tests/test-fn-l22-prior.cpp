// [TAG_FN_L22_X_PRIOR] round 22 P12: CPU unit test of the route prior and the resident swap (opt-in X3 PRIOR / X3b
// RESIDENT, src/llama-fn-prior.h, ggml/include/ggml-fn-l22-prior.h): the table rows, the CPU reference of the fused CUDA
// routing (no lever = the plain top-k; the prior; the resident swap; both), the graph form on the CPU backend (node order =
// the CUDA fusion's precondition, mark and op params, ids and weights against the reference), the op-param helpers and the
// bridge's job counts. No GPU, no model. Exit code 0 iff every check passes.
//
// --gpu (the test session only, never in the CPU runs): the same graphs on the first GPU device, through the fused route
// kernel, against the CPU graph (G1), bit for bit against the stock topk-moe graph where no expert is hot (G2), and the
// resident swap against the reference (G3). Prints SKIPPED without a GPU device. Run it a second time with
// GGML_CUDA_DISABLE_FUSION=1: G1 then checks the unfused path (G2 / G3 are skipped there).

#include "llama-fn-prior.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-fn-l22-prior.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

constexpr int   N_EXP  = 512;
constexpr int   K      = 10;
constexpr int   N_SLOT = 64; // table[e] in [0, N_SLOT) is hot, N_SLOT = miss
constexpr float CLAMP  = 6.103515625e-5f;

bool rel_close(float a, float b, float tol) {
    const float d = std::fabs(a - b);
    return d <= tol*std::max(std::fabs(a), std::fabs(b)) || d <= 1e-12f;
}

uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

// an independent naive top-k: indices by value descending, the lower index on ties; weights p / (sequential sum)
void naive_topk(const std::vector<float> & p, int k, std::vector<int32_t> & ids, std::vector<float> & w, float wscale) {
    std::vector<int32_t> idx((size_t) p.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return p[a] > p[b]; });
    ids.assign(idx.begin(), idx.begin() + k);
    double s = 0.0;
    for (int i = 0; i < k; ++i) {
        s += p[ids[i]];
    }
    w.resize((size_t) k);
    for (int i = 0; i < k; ++i) {
        w[i] = (float) (p[ids[i]]/std::max(s, (double) CLAMP))*wscale;
    }
}

std::vector<float> softmax_row(std::mt19937 & rng, float spread) {
    std::normal_distribution<float> nd(0.0f, spread);
    std::vector<float> lg((size_t) N_EXP);
    for (float & v : lg) {
        v = nd(rng);
    }
    const float mx = *std::max_element(lg.begin(), lg.end());
    double s = 0.0;
    for (float & v : lg) {
        v = std::exp(v - mx);
        s += v;
    }
    for (float & v : lg) {
        v = (float) (v/s);
    }
    return lg;
}

std::vector<int32_t> random_table(std::mt19937 & rng, int n_hot) {
    std::vector<int32_t> tbl((size_t) N_EXP, N_SLOT);
    for (int s = 0; s < n_hot; ++s) {
        int e;
        do {
            e = (int) (rng() % N_EXP);
        } while (tbl[e] != N_SLOT);
        tbl[e] = s;
    }
    return tbl;
}

// 1. the table rows
void test_fill_rows() {
    std::vector<int32_t> tbl((size_t) N_EXP, N_SLOT);
    tbl[3]   = 0;
    tbl[100] = 7;
    tbl[511] = N_SLOT - 1;
    std::vector<float> sc((size_t) N_EXP), mk((size_t) N_EXP);
    llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.5f, sc.data(), mk.data());
    int n_hot = 0;
    bool ok = true;
    for (int e = 0; e < N_EXP; ++e) {
        const bool hot = e == 3 || e == 100 || e == 511;
        n_hot += mk[e] != 0.0f ? 1 : 0;
        ok = ok && sc[e] == (hot ? expf(0.5f) : 1.0f) && mk[e] == (hot ? 1.0f : 0.0f);
    }
    TCHECK(ok && n_hot == 3, "lambda 0.5: exp(0.5) / 1, mask 1 / 0 (%d hot)", n_hot);
    llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.0f, sc.data(), mk.data());
    ok = true;
    for (int e = 0; e < N_EXP; ++e) {
        ok = ok && sc[e] == 1.0f;
    }
    TCHECK(ok && mk[3] == 1.0f && mk[4] == 0.0f, "lambda 0: scale 1 everywhere, the mask still marks the hot experts");
    llama_fn_prior_fill_rows(nullptr, 0, N_EXP, 2.0f, sc.data(), mk.data());
    ok = true;
    for (int e = 0; e < N_EXP; ++e) {
        ok = ok && sc[e] == 1.0f && mk[e] == 0.0f;
    }
    TCHECK(ok, "no table: every expert cold (1 / 0)");
}

// 2. no lever: the plain top-k
void test_ref_plain() {
    std::mt19937 rng(11);
    for (int it = 0; it < 64; ++it) {
        const std::vector<float> p = softmax_row(rng, it % 2 ? 1.0f : 3.0f);
        std::vector<float> ones((size_t) N_EXP, 1.0f), zeros((size_t) N_EXP, 0.0f);
        int32_t ids[K], ids2[K], raw[K];
        float   w[K], w2[K];
        llama_fn_prior_ref_select(p.data(), N_EXP, ones.data(), zeros.data(), K, 0, 1.0f, 6, true, CLAMP, 2.5f, ids, w, raw);
        llama_fn_prior_ref_select(p.data(), N_EXP, nullptr, nullptr, K, 0, 1.0f, 6, true, CLAMP, 2.5f, ids2, w2, nullptr);
        std::vector<int32_t> nid;
        std::vector<float>   nw;
        naive_topk(p, K, nid, nw, 2.5f);
        bool ids_ok = true, w_ok = true, same = true, raw_ok = true;
        for (int i = 0; i < K; ++i) {
            ids_ok = ids_ok && ids[i] == nid[i];
            raw_ok = raw_ok && raw[i] == nid[i];
            w_ok   = w_ok && rel_close(w[i], nw[i], 1e-6f);
            same   = same && ids[i] == ids2[i] && f2u(w[i]) == f2u(w2[i]);
        }
        TCHECK(ids_ok && raw_ok, "it %d: ids = the naive top-k (raw too)", it);
        TCHECK(w_ok, "it %d: weights = p / sum (1e-6)", it);
        TCHECK(same, "it %d: scale 1 / mask 0 = no rows, bit for bit", it);
        // without the norm: p * wscale
        llama_fn_prior_ref_select(p.data(), N_EXP, nullptr, nullptr, K, 0, 1.0f, 6, false, CLAMP, 1.5f, ids2, w2, nullptr);
        bool nn = true;
        for (int i = 0; i < K; ++i) {
            nn = nn && ids2[i] == nid[i] && f2u(w2[i]) == f2u(p[nid[i]]*1.5f);
        }
        TCHECK(nn, "it %d: no norm: p * scale", it);
    }
    // ties: the lower expert first
    std::vector<float> p((size_t) N_EXP, 0.001f);
    p[40] = 0.2f;
    p[7]  = 0.2f;
    p[300] = 0.1f;
    int32_t ids[K];
    float   w[K];
    llama_fn_prior_ref_select(p.data(), N_EXP, nullptr, nullptr, K, 0, 1.0f, 6, true, CLAMP, 1.0f, ids, w, nullptr);
    TCHECK(ids[0] == 7 && ids[1] == 40 && ids[2] == 300 && ids[3] == 0 && ids[4] == 1, "ties: the lower expert first (%d %d %d %d %d)",
            ids[0], ids[1], ids[2], ids[3], ids[4]);
    // NaN reads as -FLT_MAX: never picked before a finite one, ids stay distinct
    std::vector<float> pn((size_t) N_EXP, NAN);
    pn[5] = 0.5f;
    pn[9] = 0.25f;
    llama_fn_prior_ref_select(pn.data(), N_EXP, nullptr, nullptr, K, 0, 1.0f, 6, true, CLAMP, 1.0f, ids, w, nullptr);
    bool distinct = true;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < i; ++j) {
            distinct = distinct && ids[i] != ids[j];
        }
    }
    TCHECK(ids[0] == 5 && ids[1] == 9 && distinct, "NaN rows: the finite experts first, distinct ids");
}

// a row with nine big cold experts 0..8 and a contested tenth place
std::vector<float> contested_row() {
    std::vector<float> p((size_t) N_EXP, 1e-5f);
    for (int e = 0; e < 9; ++e) {
        p[e] = 0.1f - 0.005f*e;
    }
    p[100] = 0.030f; // cold, rank 9 (the tenth)
    p[150] = 0.027f; // cold, rank 10
    p[200] = 0.025f; // hot, rank 11
    p[300] = 0.005f; // hot, far down
    return p;
}

// 3. the prior
void test_ref_prior() {
    const std::vector<float> p = contested_row();
    std::vector<int32_t> tbl((size_t) N_EXP, N_SLOT);
    tbl[200] = 1;
    tbl[300] = 2;
    std::vector<float> sc((size_t) N_EXP), mk((size_t) N_EXP);
    llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.5f, sc.data(), mk.data());
    int32_t ids[K], raw[K];
    float   w[K];
    llama_fn_prior_ref_select(p.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_PRIOR, 1.0f, 6, true, CLAMP, 1.0f,
            ids, w, raw);
    bool has200 = false, has100 = false, has300 = false;
    double s = 0.0;
    for (int i = 0; i < K; ++i) {
        has200 = has200 || ids[i] == 200;
        has100 = has100 || ids[i] == 100;
        has300 = has300 || ids[i] == 300;
        s += p[ids[i]];
    }
    TCHECK(has200 && !has100 && !has300, "lambda 0.5: the hot rank-11 expert (0.025 x 1.65) enters over the cold rank-9 one "
            "(0.030), the hot 0.005 one does not");
    TCHECK(ids[9] == 200, "the hot expert takes rank 9 (%d)", ids[9]);
    bool w_ok = true;
    for (int i = 0; i < K; ++i) {
        w_ok = w_ok && rel_close(w[i], (float) (p[ids[i]]/s), 1e-6f);
    }
    TCHECK(w_ok, "weights = the unscaled p renormalized over the picked ten");
    bool raw_ok = true;
    for (int i = 0; i < 9; ++i) {
        raw_ok = raw_ok && raw[i] == i;
    }
    TCHECK(raw_ok && raw[9] == 100, "raw = the unbiased top-k");
    // a large lambda pulls the far one in too
    llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 3.0f, sc.data(), mk.data());
    llama_fn_prior_ref_select(p.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_PRIOR, 1.0f, 6, true, CLAMP, 1.0f,
            ids, w, raw);
    has300 = false;
    for (int i = 0; i < K; ++i) {
        has300 = has300 || ids[i] == 300;
    }
    TCHECK(has300 && raw[9] == 100, "lambda 3: the 0.005 hot expert (x 20) enters; raw unchanged");
    // raw ids never depend on lambda
    std::mt19937 rng(5);
    bool raw_same = true;
    for (int it = 0; it < 16; ++it) {
        const std::vector<float> q = softmax_row(rng, 2.0f);
        const std::vector<int32_t> t2 = random_table(rng, 100);
        int32_t r1[K], r2[K], i1[K];
        float   w1[K];
        llama_fn_prior_fill_rows(t2.data(), N_SLOT, N_EXP, 0.0f, sc.data(), mk.data());
        llama_fn_prior_ref_select(q.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_PRIOR, 1.0f, 6, true, CLAMP, 1.0f,
                i1, w1, r1);
        llama_fn_prior_fill_rows(t2.data(), N_SLOT, N_EXP, 1.0f, sc.data(), mk.data());
        llama_fn_prior_ref_select(q.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_PRIOR, 1.0f, 6, true, CLAMP, 1.0f,
                i1, w1, r2);
        for (int i = 0; i < K; ++i) {
            raw_same = raw_same && r1[i] == r2[i];
        }
    }
    TCHECK(raw_same, "raw ids are the same for any lambda");
}

// 4. the resident swap
void test_ref_resident() {
    std::vector<float> p((size_t) N_EXP, 1e-6f);
    const float big[6] = { 0.2f, 0.18f, 0.16f, 0.14f, 0.12f, 0.1f };
    for (int e = 0; e < 6; ++e) {
        p[e] = big[e]; // cold ranks 0..5
    }
    p[106] = 0.05f;   // cold ranks 6..9
    p[107] = 0.04f;
    p[108] = 0.03f;
    p[109] = 0.025f;
    p[400] = 0.02f;   // hot, not picked
    p[401] = 0.0125f; // hot, not picked
    std::vector<int32_t> tbl((size_t) N_EXP, N_SLOT);
    tbl[400] = 0;
    tbl[401] = 1;
    std::vector<float> sc((size_t) N_EXP), mk((size_t) N_EXP);
    llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.0f, sc.data(), mk.data());
    const float expnm = 0.5f; // margin ln 2
    int32_t ids[K];
    float   w[K];
    llama_fn_prior_ref_select(p.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_RESIDENT, expnm, 6, true, CLAMP,
            1.0f, ids, w, nullptr);
    const int32_t want[K] = { 0, 1, 2, 3, 4, 5, 106, 400, 108, 401 };
    bool ok = true;
    for (int i = 0; i < K; ++i) {
        ok = ok && ids[i] == want[i];
    }
    TCHECK(ok, "sequential swaps at ranks 7 and 9 (got %d %d %d %d)", ids[6], ids[7], ids[8], ids[9]);
    double s = 0.0;
    for (int i = 0; i < K; ++i) {
        s += p[ids[i]];
    }
    bool w_ok = true;
    for (int i = 0; i < K; ++i) {
        w_ok = w_ok && rel_close(w[i], (float) (p[ids[i]]/s), 1e-6f);
    }
    TCHECK(w_ok, "weights renormalized over the swapped set");
    // r0 = 8: rank 7 (which swaps with r0 6) stays; rank 8 takes 400 (0.02 >= 0.015), rank 9 then 401 (0.0125 >= 0.0125)
    llama_fn_prior_ref_select(p.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_RESIDENT, expnm, 8, true, CLAMP,
            1.0f, ids, w, nullptr);
    TCHECK(ids[6] == 106 && ids[7] == 107 && ids[8] == 400 && ids[9] == 401, "r0 8: ranks < 8 never swap (got %d %d %d %d)",
            ids[6], ids[7], ids[8], ids[9]);
    // a hot pick at a swap rank stays
    tbl[107] = 5;
    llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.0f, sc.data(), mk.data());
    llama_fn_prior_ref_select(p.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_RESIDENT, expnm, 6, true, CLAMP,
            1.0f, ids, w, nullptr);
    TCHECK(ids[6] == 106 && ids[7] == 107 && ids[8] == 400 && ids[9] == 401, "a hot pick stays; the cold ranks 8, 9 take the "
            "residents (%d %d %d %d)", ids[6], ids[7], ids[8], ids[9]);
    tbl[107] = N_SLOT;
    llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.0f, sc.data(), mk.data());
    // no swap within a tiny margin: the plain bits
    int32_t ids0[K];
    float   w0[K];
    llama_fn_prior_ref_select(p.data(), N_EXP, nullptr, nullptr, K, 0, 1.0f, 6, true, CLAMP, 1.0f, ids0, w0, nullptr);
    llama_fn_prior_ref_select(p.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_RESIDENT, expf(-0.01f), 6, true,
            CLAMP, 1.0f, ids, w, nullptr);
    bool same = true;
    for (int i = 0; i < K; ++i) {
        same = same && ids[i] == ids0[i] && f2u(w[i]) == f2u(w0[i]);
    }
    TCHECK(same, "a token without a swap keeps the plain ids and weights bit for bit");
    // no residents: nothing changes
    llama_fn_prior_fill_rows(nullptr, 0, N_EXP, 0.0f, sc.data(), mk.data());
    llama_fn_prior_ref_select(p.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_RESIDENT, 0.001f, 6, true, CLAMP,
            1.0f, ids, w, nullptr);
    same = true;
    for (int i = 0; i < K; ++i) {
        same = same && ids[i] == ids0[i] && f2u(w[i]) == f2u(w0[i]);
    }
    TCHECK(same, "no residents: the plain bits");
    // both: the prior first, then the swap
    const std::vector<float> q = contested_row();
    std::vector<int32_t> t2((size_t) N_EXP, N_SLOT);
    t2[200] = 0; // enters by the prior at rank 9
    t2[300] = 1; // a resident the swap may take
    llama_fn_prior_fill_rows(t2.data(), N_SLOT, N_EXP, 0.5f, sc.data(), mk.data());
    llama_fn_prior_ref_select(q.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_PRIOR | GGML_FN_L22_PRIOR_F_RESIDENT,
            expf(-2.0f), 6, true, CLAMP, 1.0f, ids, w, nullptr);
    // ranks 6..8 are the cold experts 6, 7, 8 (0.07, 0.065, 0.06): 300 (0.005) >= 0.07 x e^-2 = 0.0095? no; >= 0.06 x e^-2 =
    // 0.0081? no. Rank 9 is the hot 200: stays
    TCHECK(ids[9] == 200 && ids[6] == 6 && ids[7] == 7 && ids[8] == 8, "both: the prior's pick stays hot, no swap below margin");
    llama_fn_prior_ref_select(q.data(), N_EXP, sc.data(), mk.data(), K, GGML_FN_L22_PRIOR_F_PRIOR | GGML_FN_L22_PRIOR_F_RESIDENT,
            expf(-3.0f), 6, true, CLAMP, 1.0f, ids, w, nullptr);
    // e^-3 = 0.0498: 0.005 >= 0.07 x 0.0498 = 0.00349 -> rank 6 takes 300; then no resident left
    TCHECK(ids[6] == 300 && ids[7] == 7 && ids[8] == 8 && ids[9] == 200, "both, margin 3: rank 6 takes the resident 300");
}

struct pgraph {
    ggml_tensor * logits  = nullptr;
    ggml_tensor * probs   = nullptr;
    ggml_tensor * row     = nullptr;
    ggml_tensor * sel     = nullptr; // nullptr: the stock graph (no MUL)
    ggml_tensor * ids     = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_cgraph * gf      = nullptr;
};

// build_moe_ffn's routing for a softmax router with norm_w and a weight scale; prior = false: the stock graph
pgraph build_routing(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * tab, int il, int T, bool prior, int32_t flags,
        float expnm, int r0, float wscale) {
    pgraph g;
    g.gf     = ggml_new_graph(ctx);
    g.logits = logits;
    g.probs  = ggml_soft_max(ctx, logits);
    ggml_tensor * selp = g.probs;
    if (prior) {
        g.row = ggml_view_2d(ctx, tab, N_EXP, 1, tab->nb[1], (size_t) il*tab->nb[2]);
        ggml_build_forward_expand(g.gf, g.row); // as the hook does
        g.sel = llama_fn_prior_build_sel(ctx, g.probs, g.row, flags, expnm, r0, il);
        selp  = g.sel;
    }
    g.ids = ggml_argsort_top_k(ctx, selp, K);
    ggml_tensor * p3 = ggml_reshape_3d(ctx, g.probs, 1, N_EXP, T);
    ggml_tensor * w  = ggml_get_rows(ctx, p3, g.ids);
    w = ggml_reshape_2d(ctx, w, K, T);
    ggml_tensor * s = ggml_sum_rows(ctx, w);
    s = ggml_clamp(ctx, s, CLAMP, INFINITY);
    w = ggml_div(ctx, w, s);
    w = ggml_reshape_3d(ctx, w, 1, K, T);
    w = ggml_scale(ctx, w, wscale);
    g.weights = w;
    ggml_build_forward_expand(g.gf, g.weights);
    return g;
}

// 5. the graph form on the CPU backend
void test_graph_cpu() {
    std::mt19937 rng(77);
    const int L = 4;
    const int il = 2;
    for (int T : { 1, 3, 8 }) {
        ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * tab = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EXP, 2, L);
        for (int l = 0; l < L; ++l) {
            const std::vector<int32_t> tbl = random_table(rng, 60 + 20*l);
            float * r = (float *) tab->data + (size_t) l*2*N_EXP;
            llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.7f, r, r + N_EXP);
        }
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EXP, T);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (int64_t i = 0; i < ggml_nelements(logits); ++i) {
            ((float *) logits->data)[i] = nd(rng);
        }
        const int32_t flags = GGML_FN_L22_PRIOR_F_PRIOR | GGML_FN_L22_PRIOR_F_RAW;
        pgraph g = build_routing(ctx, logits, tab, il, T, true, flags, 0.25f, 6, 2.5f);

        // node order: the row VIEW first, then the fused pattern contiguous
        const int n = ggml_graph_n_nodes(g.gf);
        int i_sm = -1;
        for (int i = 0; i < n; ++i) {
            if (ggml_graph_node(g.gf, i)->op == GGML_OP_SOFT_MAX) {
                i_sm = i;
            }
        }
        const ggml_op want[] = { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_MUL, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
                                 GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE };
        bool order = i_sm >= 1 && i_sm + (int) (sizeof(want)/sizeof(want[0])) <= n;
        for (size_t k = 0; order && k < sizeof(want)/sizeof(want[0]); ++k) {
            order = ggml_graph_node(g.gf, i_sm + (int) k)->op == want[k];
        }
        TCHECK(order, "T %d: SOFT_MAX, RESHAPE, MUL, ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE, SCALE", T);
        bool view_first = false;
        for (int i = 0; i < i_sm; ++i) {
            view_first = view_first || ggml_graph_node(g.gf, i) == g.row;
        }
        TCHECK(view_first, "T %d: the row view is ahead of the router's SOFT_MAX", T);
        TCHECK(i_sm >= 0 && ggml_graph_node(g.gf, i_sm + 2) == g.sel && g.sel->src[0] == g.probs && g.sel->src[1] == g.row &&
               ggml_graph_node(g.gf, i_sm + 1)->src[0] == g.probs, "T %d: the MUL reads the softmax and the row", T);
        int32_t f = 0, r0 = 0, gil = 0;
        float   en = 0.0f;
        TCHECK(ggml_fn_l22_prior_get(g.sel, &f, &en, &r0, &gil) && f == flags && en == 0.25f && r0 == 6 && gil == il,
                "T %d: mark and op params on the MUL", T);
        TCHECK(g.row->view_src == tab && g.row->view_offs == (size_t) il*tab->nb[2] && ggml_is_contiguous(g.row),
                "T %d: the row is row 0 of layer %d", T, il);

        TCHECK(ggml_graph_compute_with_ctx(ctx, g.gf, 4) == GGML_STATUS_SUCCESS, "T %d: compute", T);
        const float * sc = (const float *) tab->data + (size_t) il*2*N_EXP;
        const float * mk = sc + N_EXP;
        bool ids_ok = true, w_ok = true;
        for (int t = 0; t < T; ++t) {
            const float * p = (const float *) g.probs->data + (size_t) t*N_EXP;
            int32_t rid[K];
            float   rw[K];
            llama_fn_prior_ref_select(p, N_EXP, sc, mk, K, GGML_FN_L22_PRIOR_F_PRIOR, 1.0f, 6, true, CLAMP, 2.5f, rid, rw, nullptr);
            const int32_t * gid = (const int32_t *) ((const char *) g.ids->data + (size_t) t*g.ids->nb[1]);
            const float *   gw  = (const float *) g.weights->data + (size_t) t*K;
            for (int i = 0; i < K; ++i) {
                ids_ok = ids_ok && gid[i] == rid[i];
                w_ok   = w_ok && rel_close(gw[i], rw[i], 1e-6f);
            }
        }
        TCHECK(ids_ok, "T %d: graph ids = the reference", T);
        TCHECK(w_ok, "T %d: graph weights = the reference (1e-6)", T);
        ggml_free(ctx);
    }
}

// 6. op-param helpers
void test_params() {
    ggml_init_params ip = { (size_t) 1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EXP, 2);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_EXP);
    ggml_tensor * m = ggml_mul(ctx, a, b);
    TCHECK(!ggml_fn_l22_prior_get(m, nullptr, nullptr, nullptr, nullptr), "an unmarked MUL reads back false");
    ggml_fn_l22_prior_set(m, 7, 0.125f, 9, 33);
    int32_t f = 0, r0 = 0, il = 0;
    float   en = 0.0f;
    TCHECK(ggml_fn_l22_prior_get(m, &f, &en, &r0, &il) && f == 7 && en == 0.125f && r0 == 9 && il == 33, "round trip");
    TCHECK(ggml_fn_l3_get(m) == GGML_FN_L22_PRIOR, "the mark in the L3 slot");
    ggml_tensor * s = ggml_scale(ctx, a, 2.0f);
    ggml_fn_l22_prior_set(s, 1, 1.0f, 0, 0);
    TCHECK(!ggml_fn_l22_prior_get(s, nullptr, nullptr, nullptr, nullptr), "a marked non-MUL reads back false");
    TCHECK(!ggml_fn_l22_prior_get(nullptr, nullptr, nullptr, nullptr, nullptr), "null reads back false");
    ggml_free(ctx);
}

// 7. the bridge's job counts
void test_job_counts() {
    std::vector<int32_t> tbl((size_t) N_EXP, N_SLOT);
    tbl[1] = 0;
    tbl[2] = 1;
    const int T = 3, k = 4;
    const int32_t ids[T*k] = { 1, 2, 10, 11,   1, 10, 12, 13,   2, 11, 14, 1 };
    const int32_t raw[T*k] = { 1, 10, 11, 20,  10, 12, 13, 21,  11, 14, 22, 23 };
    std::vector<uint8_t> scratch((size_t) N_EXP, 0);
    uint32_t cu = 0, cr = 0, ch = 0;
    llama_fn_prior_job_counts(ids, raw, T, k, tbl.data(), N_SLOT, N_EXP, scratch.data(), &cu, &cr, &ch);
    // picked distinct {1,2,10,11,12,13,14}: cold 5; raw distinct {1,10,11,20,12,13,21,14,22,23}: cold 9
    // changed: token 0: 2 not in raw -> 1; token 1: 1 -> 1; token 2: 2, 1 -> 2
    TCHECK(cu == 5 && cr == 9 && ch == 4, "cold used %u (5), cold raw %u (9), changed %u (4)", cu, cr, ch);
    bool zero = true;
    for (uint8_t v : scratch) {
        zero = zero && v == 0;
    }
    TCHECK(zero, "the scratch is zero again");
    llama_fn_prior_job_counts(ids, nullptr, T, k, nullptr, 0, N_EXP, scratch.data(), &cu, &cr, &ch);
    TCHECK(cu == 7 && cr == 0 && ch == 0, "no table: every picked expert cold (%u), no raw: 0 / 0", cu);
}

// switches (environment; a null model)
void test_cfg() {
#ifdef _WIN32
    auto set = [](const char * n, const char * v) { _putenv_s(n, v ? v : ""); };
#else
    auto set = [](const char * n, const char * v) { if (v) { setenv(n, v, 1); } else { unsetenv(n); } };
#endif
    set("LLAMA_FN_L22_X_PRIOR", nullptr);
    set("LLAMA_FN_L22_X_RESIDENT", nullptr);
    set("LLAMA_FN_L22_X_RESIDENT_R0", nullptr);
    set("LLAMA_FN_L22_X_PRIOR_STATS", nullptr);
    set("LLAMA_FN_L22_X_PRIOR_PROMPT", nullptr);
    llama_fn_prior_cfg c = llama_fn_prior_read(nullptr);
    TCHECK(!c.on() && c.lambda == 0.0f && c.margin == 0.0f && c.r0 == 6 && c.stats_raw && !c.prompt, "unset: off, defaults");
    set("LLAMA_FN_L22_X_PRIOR", "0.5");
    set("LLAMA_FN_L22_X_PRIOR_STATS", "used");
    c = llama_fn_prior_read(nullptr);
    TCHECK(c.on() && c.lambda == 0.5f && c.margin == 0.0f && !c.stats_raw, "X_PRIOR 0.5, stats used");
    set("LLAMA_FN_L22_X_PRIOR", "0");
    set("LLAMA_FN_L22_X_RESIDENT", "0.7");
    set("LLAMA_FN_L22_X_RESIDENT_R0", "8");
    set("LLAMA_FN_L22_X_PRIOR_PROMPT", "1");
    set("LLAMA_FN_L22_X_PRIOR_STATS", "raw");
    c = llama_fn_prior_read(nullptr);
    TCHECK(c.on() && c.lambda == 0.0f && c.margin == 0.7f && c.r0 == 8 && c.prompt && c.stats_raw, "X_RESIDENT 0.7, R0 8, prompt");
    set("LLAMA_FN_L22_X_RESIDENT", "off");
    c = llama_fn_prior_read(nullptr);
    TCHECK(!c.on(), "off words: off");
    set("LLAMA_FN_L22_X_PRIOR", nullptr);
    set("LLAMA_FN_L22_X_RESIDENT", nullptr);
    set("LLAMA_FN_L22_X_RESIDENT_R0", nullptr);
    set("LLAMA_FN_L22_X_PRIOR_STATS", nullptr);
    set("LLAMA_FN_L22_X_PRIOR_PROMPT", nullptr);
    TCHECK(!llama_fn_prior_on() && !llama_fn_prior_raw_wanted(), "no state without a bridge");
}

//
// --gpu: the test session only
//

struct gpu_run {
    std::vector<int32_t> ids; // [T][K]
    std::vector<float>   w;   // [T][K]
};

// the routing graph on a backend; tab_host [N_EXP, 2, L], logits_host [T][N_EXP]
bool run_on_backend(ggml_backend_t be, const std::vector<float> & tab_host, int L, int il, const std::vector<float> & logits_host,
        int T, bool prior, int32_t flags, float expnm, int r0, gpu_run & out) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tab    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EXP, 2, L);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EXP, T);
    pgraph g = build_routing(ctx, logits, tab, il, T, prior, flags, expnm, r0, 2.5f);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(tab, tab_host.data(), 0, ggml_nbytes(tab));
    ggml_backend_tensor_set(logits, logits_host.data(), 0, ggml_nbytes(logits));
    const bool ok = ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS;
    out.ids.assign((size_t) T*K, -1);
    out.w.assign((size_t) T*K, 0.0f);
    if (ok) {
        std::vector<int32_t> all((size_t) T*N_EXP);
        ggml_backend_tensor_get(g.ids->view_src, all.data(), 0, all.size()*sizeof(int32_t));
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < K; ++i) {
                out.ids[(size_t) t*K + i] = all[(size_t) t*N_EXP + i];
            }
        }
        ggml_backend_tensor_get(g.weights, out.w.data(), 0, out.w.size()*sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// no near-tie among the row's top k + 1 selection values (relative gap 1e-5): ids must agree exactly there
bool clear_row(const float * p, const float * sc) {
    std::vector<float> s((size_t) N_EXP);
    for (int e = 0; e < N_EXP; ++e) {
        s[e] = p[e]*sc[e];
    }
    std::sort(s.begin(), s.end(), [](float a, float b) { return a > b; });
    for (int i = 0; i < K; ++i) {
        if (s[i] - s[i + 1] <= 1e-5f*s[i]) {
            return false;
        }
    }
    return true;
}

int test_gpu() {
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && strncmp(ggml_backend_dev_name(d), "CUDA", 4) == 0) {
            dev = d; // the fused paths are CUDA's
        }
    }
    if (dev == nullptr) {
        printf("test-fn-l22-prior --gpu: SKIPPED (no GPU device)\n");
        return 0;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (gpu == nullptr || cpu == nullptr) {
        printf("test-fn-l22-prior --gpu: SKIPPED (backend init failed)\n");
        return 0;
    }
    const char * nf = getenv("GGML_CUDA_DISABLE_FUSION");
    const bool fused = !(nf && atoi(nf) != 0);
    std::mt19937 rng(2026);
    const int L = 3, il = 1;
    for (int T : { 1, 2, 4, 8 }) {
        std::vector<float> tab((size_t) N_EXP*2*L);
        for (int l = 0; l < L; ++l) {
            const std::vector<int32_t> tbl = random_table(rng, 120);
            llama_fn_prior_fill_rows(tbl.data(), N_SLOT, N_EXP, 0.6f, tab.data() + (size_t) l*2*N_EXP,
                    tab.data() + (size_t) l*2*N_EXP + N_EXP);
        }
        std::vector<float> lg((size_t) T*N_EXP);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (float & v : lg) {
            v = nd(rng);
        }
        const float * sc = tab.data() + (size_t) il*2*N_EXP;
        const float * mk = sc + N_EXP;

        // G1: the prior on the GPU (fused route kernel, or unfused with GGML_CUDA_DISABLE_FUSION=1) vs the CPU graph
        gpu_run rg, rc;
        const int32_t fp = GGML_FN_L22_PRIOR_F_PRIOR;
        TCHECK(run_on_backend(gpu, tab, L, il, lg, T, true, fp, 1.0f, 6, rg), "G1 T %d: GPU compute", T);
        TCHECK(run_on_backend(cpu, tab, L, il, lg, T, true, fp, 1.0f, 6, rc), "G1 T %d: CPU compute", T);
        for (int t = 0; t < T; ++t) {
            std::vector<float> p((size_t) N_EXP);
            {
                const float * l = lg.data() + (size_t) t*N_EXP;
                const float mx = *std::max_element(l, l + N_EXP);
                double s = 0.0;
                for (int e = 0; e < N_EXP; ++e) {
                    s += std::exp((double) l[e] - mx);
                }
                for (int e = 0; e < N_EXP; ++e) {
                    p[e] = (float) (std::exp((double) l[e] - mx)/s);
                }
            }
            if (!clear_row(p.data(), sc)) {
                continue;
            }
            bool ids_ok = true, w_ok = true;
            for (int i = 0; i < K; ++i) {
                ids_ok = ids_ok && rg.ids[(size_t) t*K + i] == rc.ids[(size_t) t*K + i];
                w_ok   = w_ok && rel_close(rg.w[(size_t) t*K + i], rc.w[(size_t) t*K + i], 1e-5f);
            }
            TCHECK(ids_ok, "G1 T %d row %d: GPU ids = CPU ids", T, t);
            TCHECK(w_ok, "G1 T %d row %d: GPU weights = CPU weights (1e-5)", T, t);
        }
        if (!fused) {
            continue;
        }

        // G2: no hot expert (scale 1, mask 0): the fused prior kernel = the stock topk-moe graph, bit for bit; RESIDENT
        // without residents too (G3a)
        std::vector<float> tab0((size_t) N_EXP*2*L);
        for (int l = 0; l < L; ++l) {
            llama_fn_prior_fill_rows(nullptr, 0, N_EXP, 0.6f, tab0.data() + (size_t) l*2*N_EXP, tab0.data() + (size_t) l*2*N_EXP + N_EXP);
        }
        gpu_run rs, rp, rr;
        TCHECK(run_on_backend(gpu, tab0, L, il, lg, T, false, 0, 1.0f, 6, rs), "G2 T %d: stock", T);
        TCHECK(run_on_backend(gpu, tab0, L, il, lg, T, true, fp, 1.0f, 6, rp), "G2 T %d: prior, no hot", T);
        TCHECK(run_on_backend(gpu, tab0, L, il, lg, T, true, GGML_FN_L22_PRIOR_F_RESIDENT, expf(-1.0f), 6, rr),
                "G3a T %d: resident, no residents", T);
        bool same = true, same_r = true;
        for (size_t a = 0; a < rs.ids.size(); ++a) {
            same   = same   && rs.ids[a] == rp.ids[a] && f2u(rs.w[a]) == f2u(rp.w[a]);
            same_r = same_r && rs.ids[a] == rr.ids[a] && f2u(rs.w[a]) == f2u(rr.w[a]);
        }
        TCHECK(same, "G2 T %d: no hot expert = the stock top-k bit for bit", T);
        TCHECK(same_r, "G3a T %d: no residents = the stock top-k bit for bit", T);

        // G3b: the resident swap against the reference (rows without near-ties and without near-threshold swaps)
        const float expnm = expf(-1.5f);
        gpu_run rw2;
        TCHECK(run_on_backend(gpu, tab, L, il, lg, T, true, fp | GGML_FN_L22_PRIOR_F_RESIDENT, expnm, 6, rw2), "G3b T %d", T);
        for (int t = 0; t < T; ++t) {
            std::vector<float> p((size_t) N_EXP);
            const float * l = lg.data() + (size_t) t*N_EXP;
            const float mx = *std::max_element(l, l + N_EXP);
            double s = 0.0;
            for (int e = 0; e < N_EXP; ++e) {
                s += std::exp((double) l[e] - mx);
            }
            for (int e = 0; e < N_EXP; ++e) {
                p[e] = (float) (std::exp((double) l[e] - mx)/s);
            }
            if (!clear_row(p.data(), sc)) {
                continue;
            }
            int32_t rid[K];
            float   rwt[K];
            llama_fn_prior_ref_select(p.data(), N_EXP, sc, mk, K, fp | GGML_FN_L22_PRIOR_F_RESIDENT, expnm, 6, true, CLAMP, 2.5f,
                    rid, rwt, nullptr);
            bool ok = true;
            for (int i = 0; i < K; ++i) {
                ok = ok && rw2.ids[(size_t) t*K + i] == rid[i] && rel_close(rw2.w[(size_t) t*K + i], rwt[i], 1e-5f);
            }
            TCHECK(ok, "G3b T %d row %d: GPU resident swap = the reference", T, t);
        }
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    bool gpu = false;
    for (int i = 1; i < argc; ++i) {
        gpu = gpu || strcmp(argv[i], "--gpu") == 0;
    }
    ggml_cpu_init();
    test_fill_rows();
    test_ref_plain();
    test_ref_prior();
    test_ref_resident();
    test_graph_cpu();
    test_params();
    test_job_counts();
    test_cfg();
    if (gpu) {
        test_gpu();
    }
    printf("test-fn-l22-prior: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
