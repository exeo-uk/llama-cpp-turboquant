#pragma once

// [TAG_FN_L19_DRAFT_ADAPT] Adaptive MTP draft length for the MTP driver (common/speculative.cpp, draft-mtp), qwen4exp, one
// sequence. Switch: LLAMA_FN_L19_DRAFT_ADAPT=1 (default off: the fixed n_max / p_min rule, as before); =2 measures only
// ([TAG_FN_L20_DRAFTFIX_MEASURE]: the fixed n_max / p_min rule, the trace and the summaries, nothing else changes).
//
// Each round picks a cap K in [1, kmax] on the drafts; the p_min rule still cuts inside the cap. K maximizes the expected
// tokens per us of the round (the Infernix rule, with our measured costs instead of a fixed 1 + 0.38 K):
//   E[tok(K)] = 1 + sum_{k=1..K} prod_{i<=k} pass_i * acc_i
//   E[us(K)]  = sum_{j=0..K} P(j drafts | K) * tv[j + 1] + td0 + td1 * sum_{k=1..K-1} prod_{i<=k} pass_i
//   P(j | K)  = prod_{i<=j} pass_i * (1 - pass_{j+1}) for j < K, prod_{i<=K} pass_i for j = K
// pass_i: P(the drafter's top p passes p_min at position i | its decode ran); acc_i: P(the target accepts draft i | drafts
// before it were accepted). Both are EWMAs (weight w_acc) over the rounds that reached position i.
// tv[W]: the time from the end of draft() to the next draft() when the verify had W rows (target decode, sampling, accept),
// the median of the width's last N_RING samples taken in the last `age` rounds; a width with fewer such samples is priced
// on the line through the two nearest measured widths (row_us per row when only one is measured). td0 / td1: EWMA of the first / a later draft decode,
// sampling included. Every `probe`-th round uses K + 1, so the deeper positions stay measured.
// Cost cut (LLAMA_FN_L19_DRAFT_ADAPT_CUT=1, on top of the p_min rule, so it can only shorten a draft): a drafted token at
// position k is kept only if P * a_k(p) >= lambda * (tv[k + 1] - tv[k]): P the probability that the drafts before it are
// accepted, a_k(p) the acceptance calibrated per position and top-p bin (prior: p), lambda the best E[tok] / E[us].
// Exact: only the draft length changes; the target verifies every draft (rejection sampling unchanged).
// Pure arithmetic, no llama calls: tests/test-mtp-adapt.cpp drives it with simulated rounds.
//
// [TAG_FN_L20_DRAFTFIX] Round 19 measured ADAPT + CUT at 105 -> 75 t/s on the owner's command: after ~40 rounds the cut
// dropped every first draft for the rest of the session (20,587 of 20,616 rounds had no draft). Four defects, fixed here:
//  1. the verify-time EWMA started at its first sample with weight 0.05, so one 55 ms sample from the first answer after
//     the start (cold experts) kept ~66 % of the width-2 price after its 8 samples: 44.1 ms where its median was 15.8 ms.
//     Now the median of the last N_RING samples (and the draft-decode EWMAs average their first 1/w samples evenly);
//  2. the cut ran on probe rounds too, so once it dropped every first draft, width 2 was never measured again and its
//     stale price was permanent. Now a probe round runs without the cut (and with K + 1), so every width stays measured;
//  3. a draft the cut dropped counted as a p_min failure: pass_1 fell to 0.00 and the cap to 1. Now pass_i counts the
//     p_min rule only (observe() takes the drafts that passed it apart from those proposed);
//  4. width 3 was priced on the line through widths 1 and 2 (12.0 -> 42.8 ms: 30.7 ms per row, 73.5 ms). Now prices
//     never fall with the width (one more row costs 5 % to 100 % of a whole 1-row step).
// And a price is only as good as its samples: a sample older than `age` rounds leaves the median (a width the policy no
// longer visits is priced from its neighbours), and the cut (or the floor) acts on a draft only when both widths it
// compares are measured; otherwise the p_min rule decides, as with the switch off.
// New: LLAMA_FN_L19_DRAFT_ADAPT_FLOOR=<p> (with CUT, default -1 = off): a draft whose top p is below p_min but at least p
// is proposed when the cost rule says it pays (a 2-row verify costs ~2 ms over a 1-row step that makes one token in
// ~12 ms); a probe round proposes every such draft, so its acceptance stays calibrated.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

struct common_mtp_adapt {
    static constexpr int K_MAX  = 4;          // the longest cap ever chosen
    static constexpr int W_MAX  = K_MAX + 1;  // verify widths tracked
    static constexpr int N_RING = 32;         // [TAG_FN_L20_DRAFTFIX] time samples per width behind its median

    // configuration, LLAMA_FN_L19_DRAFT_ADAPT_* (init_env)
    int   kmax    = 3;       // _KMAX: the cap's upper bound (1..4)
    int   probe   = 16;      // _PROBE: every this-many rounds K + 1 and no cost cut (0 = never)
    float w_acc   = 0.03f;   // _WACC: EWMA weight of a new pass / accept observation (0.10 flipped caps on noise)
    float w_t     = 0.05f;   // _WT: EWMA weight of a new draft-decode time sample
    float pass0   = 0.80f;   // _PASS0: prior pass_i
    float acc0    = 0.80f;   // _ACC0: prior acc_i
    float tv0     = 15000.f; // _TV0: prior us of a 1-row verify step (replaced by the first samples)
    float row_us  = 1500.f;  // _ROW_US: prior us of one more verify row (only for widths with fewer than nmin samples)
    float td0_us  = 600.f;   // _TD0: prior us of the first draft decode
    float td1_us  = 600.f;   // _TD1: prior us of a later draft decode
    int   nmin    = 8;       // _NMIN: samples (in the last `age` rounds) before a width's own median prices it
    int   age     = 512;     // _AGE: [TAG_FN_L20_DRAFTFIX] rounds a time sample stays in its width's median
    float row_min = 0.05f;   // _ROW_MIN: [TAG_FN_L20_DRAFTFIX] one more verify row costs at least this x a 1-row step
    float outlier = 3.0f;    // _OUTLIER: a time sample above this x its current estimate is dropped (stall, page-in)
    int   skip    = 4;       // _SKIP: rounds after a prompt that are no time samples (hot-set refill)
    int   log     = 0;       // _LOG: 1 = a summary every 512 rounds and at the end
    int   cut     = 0;       // _CUT: 1 = the cost cut on top of p_min
    float cal_w0  = 4.0f;    // _CAL_W0: weight (samples) of the prior a(p) = p in a calibration bin
    float cal_dec = 0.995f;  // _CAL_DECAY: weight of a bin's past samples per new sample in it
    float floor_p = -1.0f;   // _FLOOR: [TAG_FN_L20_DRAFTFIX_FLOOR] with the cut: below p_min down to this top p the cost
                             // rule decides (-1 = off: p_min stops a draft, as before)

    static constexpr int N_BIN = 10; // top-p bins of the calibration

    // state
    float   pass[K_MAX] = {};
    float   acc [K_MAX] = {};
    double  tv  [W_MAX] = {};      // the median of the width's fresh samples (refresh())
    int64_t tvn [W_MAX] = {};      // its fresh samples (in the ring, not older than `age` rounds)
    int64_t tvt [W_MAX] = {};      // samples taken (the ring holds the last N_RING)
    float   tvr [W_MAX][N_RING] = {};
    int64_t tvs [W_MAX][N_RING] = {}; // the round of each sample
    double  td0 = 0.0, td1 = 0.0;
    int64_t td0n = 0, td1n = 0;
    int64_t n_round = 0;
    int64_t n_drop  = 0;
    int64_t hist[K_MAX + 1] = {};  // chosen caps (probes included)
    int64_t n_probe = 0;
    bool    probing = false;       // [TAG_FN_L20_DRAFTFIX] the last choose() made a probe round: no cost cut in it
    double  lam     = 0.0;         // the best E[tok] / E[us] of the last choose()
    float   cal_n[K_MAX][N_BIN] = {};
    float   cal_a[K_MAX][N_BIN] = {};
    int64_t n_cut   = 0;           // drafted tokens the cost cut dropped
    int64_t n_ext   = 0;           // drafted tokens below p_min the floor proposed

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
        nmin    = std::min(N_RING, std::max(1, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_NMIN", (float) nmin)));
        age     = std::max(16, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_AGE", (float) age));
        row_min = std::min(1.0f, std::max(0.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_ROW_MIN", row_min)));
        outlier = std::max(1.5f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_OUTLIER", outlier));
        skip    = std::max(0, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_SKIP", (float) skip));
        log     = std::max(0, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_LOG", (float) log));
        cut     = std::max(0, (int) env_f("LLAMA_FN_L19_DRAFT_ADAPT_CUT", (float) cut));
        cal_w0  = std::max(0.01f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_CAL_W0", cal_w0));
        cal_dec = std::min(1.0f, std::max(0.9f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_CAL_DECAY", cal_dec)));
        floor_p = std::min(1.0f, env_f("LLAMA_FN_L19_DRAFT_ADAPT_FLOOR", floor_p));
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
            tvt[w] = 0;
        }
        td0 = td1 = 0.0;
        td0n = td1n = 0;
        probing = false;
        for (int i = 0; i < K_MAX; ++i) {
            for (int b = 0; b < N_BIN; ++b) {
                cal_n[i][b] = cal_a[i][b] = 0.0f;
            }
        }
    }

    static int bin_of(float p) {
        return std::min(N_BIN - 1, std::max(0, (int) (p*N_BIN)));
    }

    // calibrated acceptance of a draft at position pos (0-based) whose top p is p
    double acc_p(int pos, float p) const {
        pos = std::min(K_MAX - 1, std::max(0, pos));
        const int b = bin_of(p);
        return (cal_a[pos][b] + cal_w0*p)/(cal_n[pos][b] + cal_w0);
    }

    void observe_p(int pos, float p, bool accepted) {
        pos = std::min(K_MAX - 1, std::max(0, pos));
        const int b = bin_of(p);
        cal_n[pos][b] = cal_n[pos][b]*cal_dec + 1.0f;
        cal_a[pos][b] = cal_a[pos][b]*cal_dec + (accepted ? 1.0f : 0.0f);
    }

    // the cost cut: keep the draft at position pos (0-based, top p) when the drafts before it are all accepted with
    // probability P_prev. A probe round keeps it; with no rate yet (lam = 0) or a compared width without fresh samples the
    // p_min rule decides (a draft below p_min, below = true, the floor, is then dropped)
    bool keep(int pos, double P_prev, float p, bool below = false) const {
        if (probing) {
            return true;                               // [TAG_FN_L20_DRAFTFIX] probes measure every width and bin
        }
        if (!(cut > 0) && !below) {
            return true;
        }
        // [TAG_FN_L20_DRAFTFIX] no rate yet, or a width without fresh samples: the p_min rule
        if (!(lam > 0.0) || !measured(pos + 1) || !measured(pos + 2)) {
            return !below;
        }
        const double dt = t_verify(pos + 2) - t_verify(pos + 1);
        return P_prev*acc_p(pos, p) >= lam*dt;
    }

    // [TAG_FN_L20_DRAFTFIX] the decode-time EWMAs average their first 1/w samples evenly (a cold first sample must not
    // carry the estimate for dozens of rounds)
    static void ewma(double & v, int64_t & n, double x, double w) {
        const double wn = std::max(w, 1.0/(double) (n + 1));
        v = n == 0 ? x : v + wn*(x - v);
        n++;
    }

    bool measured(int W) const {
        return W >= 1 && W <= W_MAX && tvn[W - 1] >= nmin;
    }

    // [TAG_FN_L20_DRAFTFIX] the median and the count of a width's samples not older than `age` rounds
    void refresh(int W) {
        const int     n   = (int) std::min<int64_t>(tvt[W - 1], N_RING);
        const int64_t old = n_round - age;
        float s[N_RING];
        int   m = 0;
        for (int i = 0; i < n; ++i) {
            if (tvs[W - 1][i] >= old) {
                s[m++] = tvr[W - 1][i];
            }
        }
        tvn[W - 1] = m;
        if (m == 0) {
            return;                                    // tv keeps the last median (only the outlier test reads it)
        }
        std::nth_element(s, s + m/2, s + m);
        double md = s[m/2];
        if (m % 2 == 0) {
            md = 0.5*(md + (double) *std::max_element(s, s + m/2));
        }
        tv[W - 1] = md;
    }

    void refresh() {
        for (int w = 1; w <= W_MAX; ++w) {
            refresh(w);
        }
    }

    // the raw price of a width: its own median, or the line through the nearest measured widths
    double t_raw(int W) const {
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

    // the price of a verify step of W rows (1..W_MAX): [TAG_FN_L20_DRAFTFIX] one more row costs at least row_min x and at
    // most 1 x a 1-row step (one batch of W rows is never slower than W 1-row steps; a row is never free)
    double t_verify(int W) const {
        W = std::min(W_MAX, std::max(1, W));
        const double p1 = t_raw(1);
        double p = p1;
        for (int w = 2; w <= W; ++w) {
            p = std::min(std::max(t_raw(w), p + row_min*p1), p + p1);
        }
        return p;
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

    // the cap of the next round: the best K, or K + 1 on a probe round (count = false: no round, no probe, no refresh)
    int choose(bool count = true) {
        if (count) {
            n_round++;
            refresh();
        }
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
        lam   = best_r;
        if (count) {
            // [TAG_FN_L20_DRAFTFIX] a probe round also runs without the cost cut (keep()), at the cap kmax too
            probing = probe > 0 && n_round % probe == 0;
            if (probing) {
                K = std::min(kmax, K + 1);
                n_probe++;
            }
            hist[K]++;
        }
        return K;
    }

    // a finished round: dec draft decodes ran, npass of them passed the p_min rule (or the floor), prop drafts were
    // proposed (npass minus a draft the cost cut dropped), n_acc of them accepted. [TAG_FN_L20_DRAFTFIX] pass_i learns the
    // p_min rule only: a draft the cut dropped passed it (counting it as a failure drove pass_1 to 0 and the cap to 1)
    void observe(int dec, int npass, int prop, int n_acc) {
        dec   = std::min(K_MAX, std::max(0, dec));
        npass = std::min(dec, std::max(0, npass));
        prop  = std::min(npass, std::max(0, prop));
        n_acc = std::min(prop, std::max(0, n_acc));
        for (int i = 1; i <= dec; ++i) {
            pass[i - 1] += w_acc*((i <= npass ? 1.0f : 0.0f) - pass[i - 1]);
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
        // only against the width's own estimate: a prior far from this machine's step time must not lock samples out
        if (tvn[W - 1] >= nmin && us > outlier*tv[W - 1]) {
            n_drop++;
            return false;
        }
        // [TAG_FN_L20_DRAFTFIX] the median of the last N_RING samples (not older than `age` rounds)
        tvr[W - 1][tvt[W - 1] % N_RING] = (float) us;
        tvs[W - 1][tvt[W - 1] % N_RING] = n_round;
        tvt[W - 1]++;
        refresh(W);
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
