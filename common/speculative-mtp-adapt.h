#pragma once

// [TAG_FN_L19_DRAFT_ADAPT] Adaptive MTP draft length for the MTP driver (common/speculative.cpp, draft-mtp), qwen4exp, one
// sequence. Switch: LLAMA_FN_L19_DRAFT_ADAPT=1 (default off: the fixed n_max / p_min rule, as before).
//
// Each round picks a cap K in [1, kmax] on the drafts; the p_min rule still cuts inside the cap. K maximizes the expected
// tokens per us of the round (the Infernix rule, with our measured costs instead of a fixed 1 + 0.38 K):
//   E[tok(K)] = 1 + sum_{k=1..K} prod_{i<=k} pass_i * acc_i
//   E[us(K)]  = sum_{j=0..K} P(j drafts | K) * tv[j + 1] + td0 + td1 * sum_{k=1..K-1} prod_{i<=k} pass_i
//   P(j | K)  = prod_{i<=j} pass_i * (1 - pass_{j+1}) for j < K, prod_{i<=K} pass_i for j = K
// pass_i: P(the drafter's top p passes p_min at position i | its decode ran); acc_i: P(the target accepts draft i | drafts
// before it were accepted). Both are EWMAs (weight w_acc) over the rounds that reached position i.
// tv[W]: EWMA of the time from the end of draft() to the next draft() when the verify had W rows (target decode, sampling,
// accept); a width with few samples is priced on the line through the two nearest measured widths (row_us per row when
// only one is measured). td0 / td1: EWMA of the first / a later draft decode, sampling included. Every `probe`-th round uses K + 1, so the deeper positions stay measured.
// Exact: only the draft length changes; the target verifies every draft (rejection sampling unchanged).
// Pure arithmetic, no llama calls: tests/test-mtp-adapt.cpp drives it with simulated rounds.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

struct common_mtp_adapt {
    static constexpr int K_MAX = 4;          // the longest cap ever chosen
    static constexpr int W_MAX = K_MAX + 1;  // verify widths tracked

    // configuration, LLAMA_FN_L19_DRAFT_ADAPT_* (init_env)
    int   kmax    = 3;       // _KMAX: the cap's upper bound (1..4)
    int   probe   = 8;       // _PROBE: every this-many rounds K + 1 (0 = never)
    float w_acc   = 0.10f;   // _WACC: EWMA weight of a new pass / accept observation
    float w_t     = 0.05f;   // _WT: EWMA weight of a new time sample
    float pass0   = 0.80f;   // _PASS0: prior pass_i
    float acc0    = 0.80f;   // _ACC0: prior acc_i
    float tv0     = 15000.f; // _TV0: prior us of a 1-row verify step (replaced by the first samples)
    float row_us  = 1500.f;  // _ROW_US: prior us of one more verify row (only for widths with fewer than nmin samples)
    float td0_us  = 600.f;   // _TD0: prior us of the first draft decode
    float td1_us  = 600.f;   // _TD1: prior us of a later draft decode
    int   nmin    = 8;       // _NMIN: samples before a width's own EWMA prices it
    float outlier = 3.0f;    // _OUTLIER: a time sample above this x its current estimate is dropped (stall, page-in)
    int   skip    = 4;       // _SKIP: rounds after a prompt that are no time samples (hot-set refill)
    int   log     = 0;       // _LOG: 1 = a summary every 512 rounds and at the end

    // state
    float   pass[K_MAX] = {};
    float   acc [K_MAX] = {};
    double  tv  [W_MAX] = {};
    int64_t tvn [W_MAX] = {};
    double  td0 = 0.0, td1 = 0.0;
    int64_t td0n = 0, td1n = 0;
    int64_t n_round = 0;
    int64_t n_drop  = 0;
    int64_t hist[K_MAX + 1] = {};  // chosen caps (probes included)
    int64_t n_probe = 0;

    static float env_f(const char * name, float def) {
        const char * e = getenv(name);
        return e && e[0] ? (float) atof(e) : def;
    }

    void init_env() {
        kmax    = std::min(K_MAX, std::max(1, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_KMAX", (float) kmax)));
        probe   = std::max(0, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_PROBE", (float) probe));
        w_acc   = std::min(1.0f, std::max(0.001f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_WACC", w_acc)));
        w_t     = std::min(1.0f, std::max(0.001f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_WT", w_t)));
        pass0   = std::min(1.0f, std::max(0.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_PASS0", pass0)));
        acc0    = std::min(1.0f, std::max(0.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_ACC0", acc0)));
        tv0     = std::max(1.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_TV0", tv0));
        row_us  = std::max(0.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_ROW_US", row_us));
        td0_us  = std::max(0.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_TD0", td0_us));
        td1_us  = std::max(0.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_TD1", td1_us));
        nmin    = std::max(1, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_NMIN", (float) nmin));
        outlier = std::max(1.5f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_OUTLIER", outlier));
        skip    = std::max(0, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_SKIP", (float) skip));
        log     = std::max(0, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_LOG", (float) log));
        reset();
    }

    void reset() {
        for (int i = 0; i < K_MAX; ++i) {
            pass[i] = pass0;
            acc [i] = acc0;
        }
        for (int w = 0; w < W_MAX; ++w) {
            tv [w] = 0.0;
            tvn[w] = 0;
        }
        td0 = td1 = 0.0;
        td0n = td1n = 0;
    }

    static void ewma(double & v, int64_t & n, double x, double w) {
        v = n == 0 ? x : v + w*(x - v);
        n++;
    }

    // the price of a verify step of W rows (1..W_MAX)
    double t_verify(int W) const {
        W = std::min(W_MAX, std::max(1, W));
        if (tvn[W - 1] >= nmin) {
            return tv[W - 1];
        }
        // the nearest width with enough samples (a), and the line through it and the measured width nearest to a (b,
        // slope >= 0); with one measured width row_us per row; with none the prior line
        int a = 0;
        for (int d = 1; d < W_MAX && a == 0; ++d) {
            if (W - d >= 1 && tvn[W - d - 1] >= nmin) {
                a = W - d;
            } else if (W + d <= W_MAX && tvn[W + d - 1] >= nmin) {
                a = W + d;
            }
        }
        if (a == 0) {
            return tv0 + row_us*(W - 1);
        }
        int b = 0;
        for (int d = 1; d < W_MAX && b == 0; ++d) {
            if (a - d >= 1 && tvn[a - d - 1] >= nmin) {
                b = a - d;
            } else if (a + d <= W_MAX && tvn[a + d - 1] >= nmin) {
                b = a + d;
            }
        }
        const double slope = b == 0 ? (double) row_us : std::max(0.0, (tv[b - 1] - tv[a - 1])/(b - a));
        return std::max(1.0, tv[a - 1] + slope*(W - a));
    }

    double t_d0() const { return td0n > 0 ? td0 : td0_us; }
    double t_d1() const { return td1n > 0 ? td1 : td1_us; }

    // expected tokens and us of a round with cap K
    void expect(int K, double & tok, double & us) const {
        tok = 1.0;
        us  = t_d0();
        double pp = 1.0; // P(drafts 1..k proposed)
        double pa = 1.0; // P(drafts 1..k proposed and accepted)
        for (int k = 1; k <= K; ++k) {
            const double p = pass[k - 1];
            us += (1.0 - p)*pp*t_verify(k);  // stops at position k: k - 1 drafts, k verify rows
            pp *= p;
            pa *= p*acc[k - 1];
            tok += pa;
            if (k < K) {
                us += pp*t_d1();              // draft k proposed: one more decode
            }
        }
        us += pp*t_verify(K + 1);             // all K drafts proposed
    }

    // the cap of the next round: the best K, or K + 1 on a probe round
    int choose(bool count = true) {
        int    best   = 1;
        double best_r = -1.0;
        for (int K = 1; K <= kmax; ++K) {
            double tok, us;
            expect(K, tok, us);
            const double r = us > 0.0 ? tok/us : 0.0;
            if (r > best_r*(1.0 + 1e-9)) {
                best_r = r;
                best   = K;
            }
        }
        int K = best;
        if (count) {
            n_round++;
            if (probe > 0 && n_round % probe == 0 && K < kmax) {
                K++;
                n_probe++;
            }
            hist[K]++;
        }
        return K;
    }

    // a finished round: dec draft decodes ran, prop drafts were proposed, n_acc of them accepted
    void observe(int dec, int prop, int n_acc) {
        dec   = std::min(K_MAX, std::max(0, dec));
        prop  = std::min(dec, std::max(0, prop));
        n_acc = std::min(prop, std::max(0, n_acc));
        for (int i = 1; i <= dec; ++i) {
            pass[i - 1] += w_acc*((i <= prop ? 1.0f : 0.0f) - pass[i - 1]);
        }
        for (int i = 1; i <= std::min(prop, n_acc + 1); ++i) {
            acc[i - 1] += w_acc*((i <= n_acc ? 1.0f : 0.0f) - acc[i - 1]);
        }
    }

    // the time of a verify step with W rows; false: dropped as an outlier
    bool add_verify(int W, double us) {
        if (W < 1 || W > W_MAX || !(us > 0.0)) {
            return false;
        }
        const double ref = t_verify(W);
        if (tvn[W - 1] + tvn[0] >= 8 && us > outlier*ref) {
            n_drop++;
            return false;
        }
        ewma(tv[W - 1], tvn[W - 1], us, w_t);
        return true;
    }

    void add_decode(bool first, double us) {
        if (!(us > 0.0)) {
            return;
        }
        double  & v = first ? td0  : td1;
        int64_t & n = first ? td0n : td1n;
        if (n >= 8 && us > outlier*v) {
            n_drop++;
            return;
        }
        ewma(v, n, us, w_t);
    }
};
