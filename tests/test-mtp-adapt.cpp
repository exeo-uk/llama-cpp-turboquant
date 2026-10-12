// [TAG_FN_L19_DRAFT_ADAPT] CPU unit test of the adaptive MTP draft length (common/speculative-mtp-adapt.h).
//
// Part 1 checks the pieces: the expected tokens / us of a cap against a brute-force enumeration of every round outcome,
// the price of widths without samples, the argmax, the probe rounds, the EWMA updates and the outlier drop.
// Part 2 simulates rounds the way draft-mtp runs them (decode, p_min pass, verify width, sequential acceptance), on cost
// structures like the Flash-Next bridge (a verify row costs 0.3-3 ms, a draft decode 0.4-0.8 ms) and content that
// switches between prose (low acceptance) and code (high). The policy must reach 97 % of the best fixed cap chosen with
// hindsight per scenario, and never fall below the fixed cap 2 (the shipped rule) by more than 1 %.
// No GPU, no model. Exit code 0 iff every check passes.

#include "../common/speculative-mtp-adapt.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                                   \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_fail;                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
        }                                                                   \
    } while (0)

// every outcome of a round with cap K: drafts proposed j (0..K), then accepted a (0..j)
static void brute(const common_mtp_adapt & ad, int K, double & tok, double & us) {
    tok = 0.0;
    us  = 0.0;
    for (int j = 0; j <= K; ++j) {
        double pj = 1.0;
        for (int i = 1; i <= j; ++i) {
            pj *= ad.pass[i - 1];
        }
        if (j < K) {
            pj *= 1.0 - ad.pass[j];
        }
        const int    dec  = j < K ? j + 1 : K;
        const double t    = ad.t_d0() + (dec - 1)*ad.t_d1() + ad.t_verify(j + 1);
        double       et   = 1.0; // expected tokens given j proposed
        double       pacc = 1.0;
        for (int i = 1; i <= j; ++i) {
            pacc *= ad.acc[i - 1];
            et   += pacc;
        }
        tok += pj*et;
        us  += pj*t;
    }
}

static void test_pieces() {
    common_mtp_adapt ad;
    ad.init_env();
    TCHECK(ad.kmax == 3 && ad.probe == 16, "defaults kmax %d probe %d", ad.kmax, ad.probe);
    TCHECK(std::fabs(ad.t_verify(1) - ad.tv0) < 1e-6, "prior width 1");
    TCHECK(std::fabs(ad.t_verify(3) - (ad.tv0 + 2*ad.row_us)) < 1e-6, "prior width 3");

    // samples for width 2 only: the others are priced from it
    for (int k = 0; k < 20; ++k) {
        ad.add_verify(2, 16000.0);
    }
    TCHECK(std::fabs(ad.t_verify(2) - 16000.0) < 1e-6, "width 2 own %.1f", ad.t_verify(2));
    TCHECK(std::fabs(ad.t_verify(1) - (16000.0 - ad.row_us)) < 1e-6, "width 1 from 2: %.1f", ad.t_verify(1));
    TCHECK(std::fabs(ad.t_verify(4) - (16000.0 + 2*ad.row_us)) < 1e-6, "width 4 from 2: %.1f", ad.t_verify(4));

    // outlier drop
    const int64_t nd = ad.n_drop;
    TCHECK(!ad.add_verify(2, 16000.0*ad.outlier*1.5), "outlier kept");
    TCHECK(ad.n_drop == nd + 1, "outlier not counted");
    TCHECK(ad.add_verify(2, 17000.0), "normal sample dropped");

    // two measured widths: the others on their line
    {
        common_mtp_adapt e;
        e.init_env();
        for (int k = 0; k < 10; ++k) {
            e.add_verify(2, 16000.0);
            e.add_verify(3, 18000.0);
        }
        TCHECK(std::fabs(e.t_verify(4) - 20000.0) < 1e-6 && std::fabs(e.t_verify(5) - 22000.0) < 1e-6, "line up %.1f %.1f",
               e.t_verify(4), e.t_verify(5));
        TCHECK(std::fabs(e.t_verify(1) - 14000.0) < 1e-6, "line down %.1f", e.t_verify(1));
    }

    // expect() == enumeration, for several states
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(0.05f, 0.99f);
    for (int t = 0; t < 200; ++t) {
        for (int i = 0; i < common_mtp_adapt::K_MAX; ++i) {
            ad.pass[i] = U(rng);
            ad.acc[i]  = U(rng);
        }
        for (int w = 1; w <= common_mtp_adapt::W_MAX; ++w) {
            ad.tvn[w - 1] = (t + w) % 3 == 0 ? 0 : 100;
            ad.tv [w - 1] = 12000.0 + 1500.0*w + 500.0*U(rng);
        }
        ad.td0 = 500.0 + 300.0*U(rng); ad.td0n = 10;
        ad.td1 = 400.0 + 300.0*U(rng); ad.td1n = 10;
        for (int K = 1; K <= common_mtp_adapt::K_MAX; ++K) {
            double a_tok, a_us, b_tok, b_us;
            ad.expect(K, a_tok, a_us);
            brute(ad, K, b_tok, b_us);
            TCHECK(std::fabs(a_tok - b_tok) < 1e-9 && std::fabs(a_us - b_us) < 1e-6*b_us,
                   "expect K=%d: %.6f/%.3f vs %.6f/%.3f", K, a_tok, a_us, b_tok, b_us);
        }
        // choose() = the argmax over 1..kmax
        ad.kmax = 1 + t % common_mtp_adapt::K_MAX;
        int    best = 1;
        double br   = -1.0;
        for (int K = 1; K <= ad.kmax; ++K) {
            double tok, us;
            ad.expect(K, tok, us);
            if (tok/us > br*(1.0 + 1e-9)) {
                br   = tok/us;
                best = K;
            }
        }
        TCHECK(ad.choose(false) == best, "argmax");
    }

    // probe rounds: every probe-th round K + 1 (below kmax)
    {
        common_mtp_adapt b;
        b.init_env();
        b.kmax = 3;
        for (int i = 0; i < common_mtp_adapt::K_MAX; ++i) {
            b.pass[i] = 0.9f;
            b.acc[i]  = 0.3f; // drafts rarely pay: K = 1 is best
        }
        int n_two = 0;
        for (int r = 0; r < 64; ++r) {
            n_two += b.choose() == 2 ? 1 : 0;
        }
        TCHECK(n_two == 64/b.probe, "probes %d", n_two);
    }

    // observe(): a round with 3 decodes, 2 proposed (3rd failed p_min), 1 accepted
    {
        common_mtp_adapt c;
        c.init_env();
        c.w_acc = 1.0f; // the last observation
        c.observe(3, 2, 2, 1);
        TCHECK(c.pass[0] == 1.0f && c.pass[1] == 1.0f && c.pass[2] == 0.0f, "pass %.2f %.2f %.2f", c.pass[0], c.pass[1], c.pass[2]);
        TCHECK(c.acc[0] == 1.0f && c.acc[1] == 0.0f, "acc %.2f %.2f", c.acc[0], c.acc[1]);
        TCHECK(c.acc[2] == c.acc0, "acc 3 untouched %.2f", c.acc[2]); // never verified
        c.observe(1, 0, 0, 0); // first draft failed p_min
        TCHECK(c.pass[0] == 0.0f && c.acc[0] == 1.0f, "empty round");
    }
}

// one simulated scenario: true pass / acc per position (they switch between two content states), cost per width
struct scenario {
    const char * name;
    float pass[2][4];
    float acc [2][4];
    float tv1, row, td0, td1;
    int   period; // rounds per content state (0 = state 0 only)
};

static double run(const scenario & sc, int fixed_k, uint32_t seed, common_mtp_adapt * pol) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double>       N(0.0, 1.0);
    double tok = 0.0, us = 0.0;
    const int rounds = 6000;
    for (int r = 0; r < rounds; ++r) {
        const int st = sc.period > 0 ? (r / sc.period) % 2 : 0;
        const int K  = pol ? pol->choose() : fixed_k;
        int dec = 0, prop = 0;
        double t = 0.0;
        for (int i = 1; i <= K; ++i) {
            dec++;
            const double td = (i == 1 ? sc.td0 : sc.td1)*(1.0 + 0.1*N(rng));
            t += td;
            if (pol) {
                pol->add_decode(i == 1, td);
            }
            if (U(rng) >= sc.pass[st][i - 1]) {
                break;
            }
            prop++;
        }
        int acc = 0;
        while (acc < prop && U(rng) < sc.acc[st][acc]) {
            acc++;
        }
        const int W  = prop + 1;
        double    tv = (sc.tv1 + sc.row*(W - 1))*(1.0 + 0.08*N(rng));
        if (U(rng) < 0.01) {
            tv *= 4.0; // a stall
        }
        t += tv;
        if (pol) {
            pol->add_verify(W, tv);
            pol->observe(dec, prop, prop, acc);
        }
        tok += 1.0 + acc;
        us  += t;
    }
    return tok/us*1000.0; // tokens per ms
}

static void test_sim() {
    const scenario scs[] = {
        // name                         pass (state 0 / 1)                        acc (state 0 / 1)                          tv1    row   td0  td1  period
        { "prose, costly rows",      {{0.75f,0.65f,0.6f,0.6f},{0.75f,0.65f,0.6f,0.6f}}, {{0.70f,0.65f,0.6f,0.6f},{0.70f,0.65f,0.6f,0.6f}}, 15000, 2500, 650, 600, 0 },
        { "code, cheap rows",        {{0.95f,0.93f,0.9f,0.9f},{0.95f,0.93f,0.9f,0.9f}}, {{0.92f,0.90f,0.9f,0.88f},{0.92f,0.90f,0.9f,0.88f}}, 15000, 600, 650, 600, 0 },
        { "code, costly rows",       {{0.95f,0.93f,0.9f,0.9f},{0.95f,0.93f,0.9f,0.9f}}, {{0.92f,0.90f,0.9f,0.88f},{0.92f,0.90f,0.9f,0.88f}}, 15000, 2500, 650, 600, 0 },
        { "low acceptance",          {{0.45f,0.4f,0.4f,0.4f},{0.45f,0.4f,0.4f,0.4f}},   {{0.45f,0.4f,0.4f,0.4f},{0.45f,0.4f,0.4f,0.4f}},   15000, 2500, 650, 600, 0 },
        { "prose <-> code, 300",     {{0.75f,0.65f,0.6f,0.6f},{0.95f,0.93f,0.9f,0.9f}}, {{0.70f,0.65f,0.6f,0.6f},{0.92f,0.90f,0.9f,0.88f}}, 15000, 1200, 650, 600, 300 },
        { "prose <-> code, 40",      {{0.75f,0.65f,0.6f,0.6f},{0.95f,0.93f,0.9f,0.9f}}, {{0.70f,0.65f,0.6f,0.6f},{0.92f,0.90f,0.9f,0.88f}}, 15000, 1200, 650, 600, 40 },
    };
    for (const auto & sc : scs) {
        double best = 0.0, fix2 = 0.0, pol_r = 0.0;
        int    best_k = 0;
        const int n_seed = 4;
        for (int K = 1; K <= 3; ++K) {
            double r = 0.0;
            for (int s = 0; s < n_seed; ++s) {
                r += run(sc, K, 100 + s, nullptr);
            }
            r /= n_seed;
            if (r > best) {
                best   = r;
                best_k = K;
            }
            if (K == 2) {
                fix2 = r;
            }
        }
        for (int s = 0; s < n_seed; ++s) {
            common_mtp_adapt pol;
            pol.init_env();
            pol.kmax = 3;
            pol_r += run(sc, 0, 100 + s, &pol);
        }
        pol_r /= n_seed;
        printf("  %-24s best fixed K=%d %.4f tok/ms, fixed K=2 %.4f, adaptive %.4f (%.1f %% of best, %+.1f %% vs K=2)\n",
               sc.name, best_k, best, fix2, pol_r, 100.0*pol_r/best, 100.0*(pol_r/fix2 - 1.0));
        TCHECK(pol_r >= 0.97*best, "%s: adaptive %.4f < 97 %% of best fixed %.4f", sc.name, pol_r, best);
        TCHECK(pol_r >= 0.99*fix2, "%s: adaptive %.4f < 99 %% of fixed K=2 %.4f", sc.name, pol_r, fix2);
    }
}

// the cost cut: drafts carry a top p; acceptance grows with it; a draft is proposed only if p >= p_min (0.5), and with
// the cut only if it also pays for its verify row. The cut may only help: >= 99 % of the cap rule alone everywhere.
struct scenario_p {
    const char * name;
    float lo, hi;  // top p ~ U(lo, hi) per position
    float a_lo;    // acceptance at p = 0.5 (linear up to 0.97 at p = 1)
    float tv1, row2, row3, row4, td0, td1;
};

// one round the way the driver runs it (common/speculative.cpp keep_token, [TAG_FN_L20_DRAFTFIX]): p_min (or the floor)
// stops a draft, a draft that passes counts in npass, the cut (and below p_min the floor's cost rule) may drop it
// cold > 0: the first `cold` rounds cost 2.5x (the first answer after a start: cold experts) and the 11th one 55 ms (the
// round-19 trace); the rate is taken over the rounds after them
static double run_p(const scenario_p & sc, int fixed_k, uint32_t seed, common_mtp_adapt * pol, int cold = 0,
                    double * frac_draft = nullptr) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double>       N(0.0, 1.0);
    const float rows[common_mtp_adapt::W_MAX + 1] = { 0.0f, 0.0f, sc.row2, sc.row3, sc.row4, sc.row4 }; // W <= kmax + 1
    const float pmin = 0.5f;
    double tok = 0.0, us = 0.0;
    int64_t n_r = 0, n_d = 0;
    for (int r = 0; r < 6000 + cold; ++r) {
        const int K = pol ? pol->choose() : fixed_k;
        const bool fl = pol && pol->cut > 0 && pol->floor_p >= 0.0f && pol->floor_p < pmin;
        int    dec = 0, npass = 0, prop = 0;
        double t = 0.0, P = 1.0;
        float  ps[4] = {};
        for (int i = 1; i <= K; ++i) {
            dec++;
            const double td = (i == 1 ? sc.td0 : sc.td1)*(1.0 + 0.1*N(rng));
            t += td;
            if (pol) {
                pol->add_decode(i == 1, td);
            }
            const float p = (float) (sc.lo + (sc.hi - sc.lo)*U(rng));
            if (p < (fl ? pol->floor_p : pmin)) {
                break;
            }
            npass++;
            const bool below = p < pmin;
            if (pol && (pol->cut > 0 || below)) {
                if (!pol->keep(i - 1, P, p, below)) {
                    break;
                }
                P *= pol->acc_p(i - 1, p);
            }
            ps[prop++] = p;
        }
        int acc = 0;
        while (acc < prop) {
            const double a = std::max(0.02, sc.a_lo + (0.97 - sc.a_lo)*(ps[acc] - 0.5)/0.5);
            if (U(rng) >= a) {
                break;
            }
            acc++;
        }
        const int W  = prop + 1;
        double    tv = sc.tv1;
        for (int w = 2; w <= W; ++w) {
            tv += rows[w];
        }
        tv *= 1.0 + 0.08*N(rng);
        if (r < cold) {
            tv *= r == 10 ? 55000.0/sc.tv1 : 2.5;
        }
        t += tv;
        if (pol) {
            pol->add_verify(W, tv);
            pol->observe(dec, npass, prop, acc);
            for (int k = 0; k < std::min(prop, acc + 1); ++k) {
                pol->observe_p(k, ps[k], k < acc);
            }
        }
        if (r >= cold) {
            tok += 1.0 + acc;
            us  += t;
            n_r++;
            n_d += prop > 0 ? 1 : 0;
        }
    }
    if (frac_draft) {
        *frac_draft = n_r > 0 ? (double) n_d/(double) n_r : 0.0;
    }
    return tok/us*1000.0;
}

static void test_cut() {
    {
        common_mtp_adapt c;
        c.init_env();
        TCHECK(std::fabs(c.acc_p(0, 0.65f) - 0.65) < 1e-6, "prior a(p) = p");
        for (int k = 0; k < 200; ++k) {
            c.observe_p(1, 0.65f, k % 4 != 0); // 75 %
        }
        TCHECK(std::fabs(c.acc_p(1, 0.62f) - 0.75) < 0.03, "calibrated %.3f", c.acc_p(1, 0.62f));
        TCHECK(std::fabs(c.acc_p(0, 0.62f) - 0.62) < 1e-6, "other position untouched");
        c.cut = 1;
        for (int k = 0; k < 20; ++k) {
            c.add_verify(1, 12000.0);
            c.add_verify(2, 14000.0);
            c.add_verify(3, 18000.0);
        }
        c.lam = 0.11/1000.0; // tok/us
        TCHECK(c.keep(0, 1.0, 0.55f), "row 2 (2 ms = 0.22 tok) pays at a 0.55");
        TCHECK(!c.keep(1, 0.7, 0.55f), "row 3 (4 ms = 0.44 tok) does not pay at 0.7 x 0.55");
        TCHECK(c.keep(1, 0.95, 0.95f), "row 3 pays at 0.95 x 0.95");
        c.cut = 0;
        TCHECK(c.keep(1, 0.1, 0.5f), "cut off keeps everything");
    }
    const scenario_p scs[] = {
        // name                       lo     hi    a_lo    tv1    row2  row3  row4  td0  td1
        { "prose, row 3 costly",    0.20f, 1.0f, 0.45f, 12000, 2000, 4200, 4500, 720, 720 },
        { "code, row 3 costly",     0.45f, 1.0f, 0.70f, 12000, 2000, 4200, 4500, 720, 720 },
        { "copy, cheap rows",       0.70f, 1.0f, 0.85f, 12000, 1000, 1500, 2000, 720, 720 },
        { "flat costs",             0.30f, 1.0f, 0.60f, 12000, 1500, 1500, 1500, 720, 720 },
    };
    for (const auto & sc : scs) {
        // [TAG_FN_L20_DRAFTFIX] every arm also after a cold start (the first 40 rounds 2.5x, one 55 ms step): round 19's
        // policy locked in there (the cut dropped every first draft for the rest of the session)
        for (int cold : { 0, 40 }) {
            double fix2 = 0.0, ad = 0.0, adc = 0.0, adcf = 0.0, fd = 0.0, fdc = 0.0;
            const int n_seed = 4;
            for (int s = 0; s < n_seed; ++s) {
                fix2 += run_p(sc, 2, 300 + s, nullptr, cold);
                common_mtp_adapt a;
                a.init_env();
                ad += run_p(sc, 0, 300 + s, &a, cold);
                common_mtp_adapt b;
                b.init_env();
                b.cut = 1;
                double f = 0.0;
                adc += run_p(sc, 0, 300 + s, &b, cold, &f);
                fdc += f;
                common_mtp_adapt c;
                c.init_env();
                c.cut     = 1;
                c.floor_p = 0.2f;
                adcf += run_p(sc, 0, 300 + s, &c, cold, &f);
                fd += f;
            }
            fix2 /= n_seed; ad /= n_seed; adc /= n_seed; adcf /= n_seed; fd /= n_seed; fdc /= n_seed;
            printf("  %-22s cold %2d: fixed K=2 %.4f, adaptive %.4f (%+.1f %%), + cut %.4f (%+.1f %%, rounds with a draft "
                   "%.0f %%), + floor 0.2 %.4f (%+.1f %%, %.0f %%)\n", sc.name, cold, fix2, ad, 100.0*(ad/fix2 - 1.0), adc,
                   100.0*(adc/fix2 - 1.0), 100.0*fdc, adcf, 100.0*(adcf/fix2 - 1.0), 100.0*fd);
            TCHECK(adc >= 0.99*ad, "%s cold %d: the cut lost %.4f vs %.4f", sc.name, cold, adc, ad);
            // probes (1/16 rounds at K + 1) and estimate noise cost up to ~1.5 % where K = 2 beats K = 3 by only ~2 %
            TCHECK(ad >= 0.985*fix2, "%s cold %d: adaptive lost %.4f vs fixed 2 %.4f", sc.name, cold, ad, fix2);
            TCHECK(adc >= 0.98*fix2, "%s cold %d: adaptive + cut lost %.4f vs fixed 2 %.4f", sc.name, cold, adc, fix2);
            TCHECK(adcf >= 0.99*adc, "%s cold %d: the floor lost %.4f vs %.4f", sc.name, cold, adcf, adc);
            TCHECK(fdc >= 0.5, "%s cold %d: the cut left drafts in only %.1f %% of the rounds", sc.name, cold, 100.0*fdc);
        }
    }
}

// [TAG_FN_L20_DRAFTFIX] the four round-19 defects, one check each
static void test_draftfix() {
    // 1. the width-2 samples of the round-19 trace (the first one from the cold first answer): the old EWMA priced width
    //    2 at 44.1 ms after them; the median of the ring is ~16 ms
    {
        common_mtp_adapt a;
        a.init_env();
        const double w2[] = { 55380, 31162, 14883, 15782, 16357, 15222, 18677, 15017 };
        for (double x : w2) {
            a.add_verify(2, x);
        }
        TCHECK(a.tvn[1] == 8 && a.t_raw(2) > 15000.0 && a.t_raw(2) < 17000.0, "width 2 median %.0f", a.t_raw(2));
        // the ring forgets: 40 cold samples, then 32 warm ones
        common_mtp_adapt b;
        b.init_env();
        for (int k = 0; k < 40; ++k) {
            b.add_verify(1, 30000.0);
        }
        for (int k = 0; k < 32; ++k) {
            b.add_verify(1, 12000.0 + 10.0*k);
        }
        TCHECK(b.t_raw(1) < 12400.0, "the cold samples left the ring: %.0f", b.t_raw(1));
        // the decode-time EWMA averages its first samples evenly
        b.add_decode(true, 29572.0);
        for (int k = 0; k < 19; ++k) {
            b.add_decode(true, 640.0);
        }
        TCHECK(b.t_d0() < 0.1*29572.0, "first decode sample faded: %.0f", b.t_d0());
    }
    // 2. a probe round runs without the cut
    {
        common_mtp_adapt c;
        c.init_env();
        c.cut = 1;
        for (int k = 0; k < 20; ++k) {
            c.add_verify(1, 12000.0);
            c.add_verify(2, 24000.0); // priced high: the cut drops every first draft
        }
        int n_keep = 0, n_probe = 0;
        for (int r = 1; r <= 64; ++r) {
            c.choose();
            n_probe += c.probing ? 1 : 0;
            n_keep  += c.keep(0, 1.0, 0.6f) ? 1 : 0;
        }
        TCHECK(n_probe == 64/c.probe && n_keep == n_probe, "probes %d, kept %d", n_probe, n_keep);
    }
    // 3. a draft the cut dropped passed p_min: pass_1 rises, acc_1 untouched
    {
        common_mtp_adapt d;
        d.init_env();
        d.w_acc = 1.0f;
        d.observe(1, 1, 0, 0);
        TCHECK(d.pass[0] == 1.0f && d.acc[0] == d.acc0, "pass %.2f acc %.2f", d.pass[0], d.acc[0]);
        d.observe(2, 2, 1, 1); // the second passed, the cut dropped it
        TCHECK(d.pass[1] == 1.0f && d.acc[0] == 1.0f && d.acc[1] == d.acc0, "pass2 %.2f acc %.2f %.2f", d.pass[1], d.acc[0],
               d.acc[1]);
    }
    // 4. prices never fall with the width; one more row never above a 1-row step
    {
        common_mtp_adapt e;
        e.init_env();
        for (int k = 0; k < 20; ++k) {
            e.add_verify(1, 12040.0);
            e.add_verify(2, 42750.0);
            e.add_verify(4, 32220.0);
        }
        for (int w = 2; w <= common_mtp_adapt::W_MAX; ++w) {
            TCHECK(e.t_verify(w) >= e.t_verify(w - 1) - 1e-9, "monotone at %d: %.0f < %.0f", w, e.t_verify(w), e.t_verify(w - 1));
            TCHECK(e.t_verify(w) - e.t_verify(w - 1) <= e.t_verify(1) + 1e-9, "row bound at %d", w);
        }
        TCHECK(e.t_verify(2) <= 2*12040.0 + 1e-9, "width 2 bounded %.0f", e.t_verify(2));
    }
    // the floor: without a rate (no choose() yet) a draft below p_min is not proposed; on a probe round it is
    {
        common_mtp_adapt f;
        f.init_env();
        f.cut = 1;
        f.floor_p = 0.2f;
        TCHECK(!f.keep(0, 1.0, 0.3f, true) && f.keep(0, 1.0, 0.6f, false), "floor without a rate");
        f.probing = true;
        TCHECK(f.keep(0, 1.0, 0.3f, true), "floor on a probe round");
    }
}

int main() {
    test_pieces();
    printf("pieces: %d checks, %d failed\n", g_checks, g_fail);
    test_sim();
    test_cut();
    test_draftfix();
    printf("total: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
