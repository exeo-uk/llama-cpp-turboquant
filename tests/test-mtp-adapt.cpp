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
    TCHECK(ad.kmax == 3 && ad.probe == 8, "defaults kmax %d probe %d", ad.kmax, ad.probe);
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
        c.observe(3, 2, 1);
        TCHECK(c.pass[0] == 1.0f && c.pass[1] == 1.0f && c.pass[2] == 0.0f, "pass %.2f %.2f %.2f", c.pass[0], c.pass[1], c.pass[2]);
        TCHECK(c.acc[0] == 1.0f && c.acc[1] == 0.0f, "acc %.2f %.2f", c.acc[0], c.acc[1]);
        TCHECK(c.acc[2] == c.acc0, "acc 3 untouched %.2f", c.acc[2]); // never verified
        c.observe(1, 0, 0); // first draft failed p_min
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
            pol->observe(dec, prop, acc);
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

int main() {
    test_pieces();
    printf("pieces: %d checks, %d failed\n", g_checks, g_fail);
    test_sim();
    printf("total: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
