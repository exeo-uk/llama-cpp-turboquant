// [TAG_FN_L22_PHASE] round 22 review fix R-REV-6: CPU unit test of the plan 3.8 gate of the answer-changing levers.
// The levers (TAILSKIP, DEFER, PRIOR, RESIDENT) act only in graphs with fn_gen = 1. build_moe_ffn takes its bridge job
// flags from llama_fn_l22_job_flags(fn_x), fn_x_defer_mode from llama_fn_l22_defer_acts and llama_fn_route_prior from
// llama_fn_l22_prior_acts; fn_gen / fn_x come from llama_fn_l22_gen_rule / llama_fn_l22_x_bits (graph_params). This test
// runs the whole chain from the switches (environment, read as a context reads them) to each gate, for prompt, mixed,
// checkpoint-tail and decode batches. No GPU, no model. Exit code 0 iff every check passes.

#include "llama.h"
#include "ggml-moe-bridge.h"

#include "../src/llama-ext.h"
#include "../src/llama-fn-l22.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// the pure gates on their own
static void test_gates() {
    for (int x = 0; x < 256; ++x) {
        const uint8_t fx = (uint8_t) x;
        const int32_t jf = llama_fn_l22_job_flags(fx);
        TCHECK((jf & ~GGML_MOE_BRIDGE_JOB_TAIL) == 0, "job flags only TAIL (x %d: %d)", x, jf);
        TCHECK(((jf & GGML_MOE_BRIDGE_JOB_TAIL) != 0) == ((fx & LLAMA_FN_L22_XB_TAIL) != 0), "TAIL iff the bit (x %d)", x);
        for (int seqs = 1; seqs <= 3; ++seqs) {
            for (int br = 0; br < 2; ++br) {
                for (int tbo = 0; tbo < 2; ++tbo) {
                    const bool want = (fx & LLAMA_FN_L22_XB_DEFER) && seqs == 1 && br && !tbo;
                    TCHECK(llama_fn_l22_defer_acts(fx, (uint32_t) seqs, br != 0, tbo != 0) == want,
                            "defer x %d seqs %d bridge %d tbo %d", x, seqs, br, tbo);
                }
            }
        }
        for (int gen = 0; gen < 2; ++gen) {
            for (int pr = 0; pr < 2; ++pr) {
                const bool want = (fx & (LLAMA_FN_L22_XB_PRIOR | LLAMA_FN_L22_XB_RESIDENT)) != 0 || (pr && !gen);
                TCHECK(llama_fn_l22_prior_acts(fx, (uint8_t) gen, pr != 0) == want, "prior x %d gen %d prompt %d", x, gen, pr);
            }
        }
    }
}

struct batch_kind {
    const char * name;
    int32_t      phase;
    uint32_t     n_tokens;
    uint32_t     n_outputs;
    uint32_t     n_seqs_unq;
};

// the switches as a context reads them, then every batch kind through gen_rule -> x_bits -> the gates
static void test_chain() {
    static const char * names[] = { "LLAMA_FN_L22_X_TAILSKIP", "LLAMA_FN_L22_X_TAILSKIP_MULTI", "LLAMA_FN_L22_X_PRIOR",
                                    "LLAMA_FN_L22_X_DEFER", "LLAMA_FN_L22_X_RESIDENT" };
    struct sw { const char * v[5]; };
    static const sw cfgs[] = {
        { { nullptr, nullptr, nullptr, nullptr, nullptr } },
        { { "7",     nullptr, nullptr, nullptr, nullptr } },
        { { "7",     "1",     nullptr, nullptr, nullptr } },
        { { nullptr, nullptr, "0.5",   nullptr, nullptr } },
        { { nullptr, nullptr, nullptr, "1",     nullptr } },
        { { nullptr, nullptr, nullptr, nullptr, "0.2"   } },
        { { "7",     "1",     "0.5",   "1",     "0.2"   } },
    };
    static const batch_kind kinds[] = {
        { "prompt",            LLAMA_FN_PHASE_PROMPT, 512, 1, 1 },
        { "prompt all out",    LLAMA_FN_PHASE_PROMPT,   4, 4, 1 },
        { "checkpoint tail",   LLAMA_FN_PHASE_PROMPT,   4, 1, 1 },
        { "short follow-up",   LLAMA_FN_PHASE_PROMPT,   3, 3, 1 },
        { "mixed (GEN flag)",  LLAMA_FN_PHASE_GEN,     40, 2, 2 },
        { "GEN, one row out",  LLAMA_FN_PHASE_GEN,      4, 1, 1 },
        { "decode",            LLAMA_FN_PHASE_GEN,      1, 1, 1 },
        { "verify",            LLAMA_FN_PHASE_GEN,      4, 4, 1 },
        { "decode 2 slots",    LLAMA_FN_PHASE_GEN,      2, 2, 2 },
        { "empty",             LLAMA_FN_PHASE_GEN,      0, 0, 0 },
    };
    for (const sw & c : cfgs) {
        for (int i = 0; i < 5; ++i) {
            set_env(names[i], c.v[i]);
        }
        const llama_fn_l22_x_cfg cfg = llama_fn_l22_x_read(nullptr);
        const bool any = cfg.any();
        TCHECK(any == (c.v[0] || c.v[2] || c.v[3] || c.v[4]), "x_read: some lever set");
        for (int pr = 0; pr < 2; ++pr) {
            for (const batch_kind & k : kinds) {
                const uint8_t gen = llama_fn_l22_gen_rule(k.phase, k.n_tokens, k.n_outputs, any);
                const uint8_t fx  = llama_fn_l22_x_bits(cfg, gen, k.n_seqs_unq);
                const bool gen_want = k.phase == LLAMA_FN_PHASE_GEN && any && k.n_tokens > 0 && k.n_outputs == k.n_tokens;
                TCHECK(gen == (gen_want ? 1 : 0), "[%s] fn_gen %d", k.name, gen);
                if (!gen) {
                    // plan 3.8: no lever acts where fn_gen = 0, whatever the switches say (only the prior's prompt rule)
                    TCHECK(fx == 0, "[%s] fn_x %d without fn_gen", k.name, fx);
                    TCHECK(llama_fn_l22_job_flags(fx) == 0, "[%s] TAIL without fn_gen", k.name);
                    for (int br = 0; br < 2; ++br) {
                        TCHECK(!llama_fn_l22_defer_acts(fx, k.n_seqs_unq, br != 0, false), "[%s] DEFER without fn_gen", k.name);
                    }
                    TCHECK(llama_fn_l22_prior_acts(fx, gen, pr != 0) == (pr != 0), "[%s] prior without fn_gen (prompt rule %d)",
                            k.name, pr);
                    continue;
                }
                const bool tail = cfg.tailskip > 0 && (k.n_seqs_unq == 1 || cfg.tail_multi);
                TCHECK(((llama_fn_l22_job_flags(fx) & GGML_MOE_BRIDGE_JOB_TAIL) != 0) == tail, "[%s] TAIL", k.name);
                TCHECK(llama_fn_l22_defer_acts(fx, k.n_seqs_unq, true, false) == (cfg.defer && k.n_seqs_unq == 1),
                        "[%s] DEFER", k.name);
                TCHECK(!llama_fn_l22_defer_acts(fx, k.n_seqs_unq, true, true), "[%s] DEFER in a two-chain graph", k.name);
                TCHECK(!llama_fn_l22_defer_acts(fx, k.n_seqs_unq, false, false), "[%s] DEFER without a bridge", k.name);
                TCHECK(llama_fn_l22_prior_acts(fx, gen, pr != 0) == (cfg.prior || cfg.resident), "[%s] prior in a GEN graph",
                        k.name);
            }
        }
    }
    for (int i = 0; i < 5; ++i) {
        set_env(names[i], nullptr);
    }
}

int main() {
    test_gates();
    test_chain();
    printf("test-fn-l22-gate: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
