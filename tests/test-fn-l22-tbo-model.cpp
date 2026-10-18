// [TAG_FN_L22_TBO] round 22 D5: whole-decode CPU test of the two-chain decode graph on a tiny qwen4exp model (the
// test-llama-archs fixture: a GDN layer with the PLE block, a QSA attention layer, shared + routed experts, hyper-connections).
// Usage: test-fn-l22-tbo-model <qwen4exp-moe.gguf> [threads]   (make the file with: test-llama-archs -a qwen4exp -o <dir>)
// Without a file it prints SKIP and passes.
//
// Two sequences are prompted one by one, then decoded together (2 x T rows, every row an output: the two-chain graphs),
// with a speculative rollback of both sequences in between, then one more one-sequence step each (which reads the states
// the chains wrote). LLAMA_FN_L22_TBO_CPUTEST=1 lets the gate pass without the MoE bridge (CPU). Checks, for each KV
// layout (positional QSA masks and the explicit mask) and rollback depth (n_rs_seq 0 and 3):
//   - TBO=2 (the chains one after the other) gives the same bits as TBO=1 (interleaved);
//   - TBO=1 matches TBO=0 (one chain): the same values per row up to the float order of the batched kernels (reported);
//   - the two-chain graph was built (log line) with TBO 1 / 2 and never with TBO 0.
// No GPU (run with CUDA_VISIBLE_DEVICES=-1).

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail   = 0;
int g_checks = 0;

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

void set_env(const char * name, const char * value) {
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

std::atomic<int> g_tbo_lines{0};
int              g_verbose = 0;

void log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    if (strstr(text, "[TAG_FN_L22_TBO] two-chain decode graph") != nullptr) {
        g_tbo_lines++;
    }
    if (g_verbose > 0 || level == GGML_LOG_LEVEL_ERROR || strstr(text, "[TAG_FN_L22_TBO]") != nullptr) {
        fputs(text, stderr);
    }
}

struct run_cfg {
    int  tbo;      // LLAMA_FN_L22_TBO
    bool pos_mask; // LLAMA_QSA_POS_MASK
    int  n_rs;     // n_rs_seq (rollback snapshots)
};

struct run_out {
    std::vector<float> logits;
    bool ok = false;
};

uint32_t g_n_vocab = 0;

// one decode of the batch entries (token, pos, seq, output); the logits of every output row appended in batch order
bool decode(llama_context * ctx, const std::vector<llama_token> & tok, const std::vector<llama_pos> & pos,
        const std::vector<llama_seq_id> & seq, const std::vector<int8_t> & out, std::vector<float> & logits) {
    const int n = (int) tok.size();
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]     = tok[(size_t) i];
        b.pos[i]       = pos[(size_t) i];
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq[(size_t) i];
        b.logits[i]    = out[(size_t) i];
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    if (rc == 0) {
        for (int i = 0; i < n; ++i) {
            if (!out[(size_t) i]) {
                continue;
            }
            const float * l = llama_get_logits_ith(ctx, i);
            logits.insert(logits.end(), l, l + g_n_vocab);
        }
    }
    llama_batch_free(b);
    return rc == 0;
}

run_out run(const char * path, const run_cfg & cfg, int n_threads) {
    run_out r;
    set_env("LLAMA_FN_L22_TBO", cfg.tbo == 0 ? "0" : cfg.tbo == 1 ? "1" : "2");
    set_env("LLAMA_FN_L22_TBO_CPUTEST", "1");
    set_env("LLAMA_QSA_POS_MASK", cfg.pos_mask ? "1" : "0");

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_dev[1] = { nullptr };
    mp.devices = no_dev; // CPU only
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path);
        return r;
    }
    g_n_vocab = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 2;
    cp.kv_unified      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.n_rs_seq        = (uint32_t) cfg.n_rs;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "cannot create a context\n");
        llama_model_free(model);
        return r;
    }

    const llama_token V = (llama_token) g_n_vocab;
    auto tk = [V](int s, int i) { return (llama_token) (1 + (s*37 + i*11 + (i*i) % 7) % (V - 1)); };

    bool ok = true;
    llama_pos p[2] = { 0, 0 };
    // prompts, one sequence per decode
    const int n_prompt[2] = { 9, 6 };
    for (int s = 0; s < 2 && ok; ++s) {
        std::vector<llama_token> t;
        std::vector<llama_pos> ps;
        std::vector<llama_seq_id> sq;
        std::vector<int8_t> o;
        for (int i = 0; i < n_prompt[s]; ++i) {
            t.push_back(tk(s, i));
            ps.push_back(p[s]++);
            sq.push_back(s);
            o.push_back(i == n_prompt[s] - 1);
        }
        ok = decode(ctx, t, ps, sq, o, r.logits);
    }
    // two-sequence steps, every row an output; a rollback of the last two rows of both sequences after step 2
    const int steps_T[] = { 3, 1, 3, 3, 1, 2, 3 };
    int step = 0;
    for (int T : steps_T) {
        if (!ok) {
            break;
        }
        std::vector<llama_token> t;
        std::vector<llama_pos> ps;
        std::vector<llama_seq_id> sq;
        std::vector<int8_t> o;
        for (int s = 0; s < 2; ++s) {
            for (int i = 0; i < T; ++i) {
                t.push_back(tk(s, 100 + step*5 + i));
                ps.push_back(p[s]++);
                sq.push_back(s);
                o.push_back(1);
            }
        }
        ok = decode(ctx, t, ps, sq, o, r.logits);
        if (ok && step == 2 && cfg.n_rs >= 2 && T >= 3) {
            // a draft rejected after its first token: drop the last two rows of each sequence (the snapshot groups the chains
            // wrote serve the recurrent state)
            llama_memory_t mem = llama_get_memory(ctx);
            for (int s = 0; s < 2 && ok; ++s) {
                ok = llama_memory_seq_rm(mem, s, p[s] - 2, -1);
                p[s] -= 2;
            }
            TCHECK(ok, "rollback refused (n_rs_seq %d)", cfg.n_rs);
        }
        ++step;
    }
    // one-sequence steps after the chains (they read what the chains wrote), then a mixed batch (not every row an output)
    for (int s = 0; s < 2 && ok; ++s) {
        std::vector<llama_token> t = { tk(s, 200), tk(s, 201) };
        std::vector<llama_pos> ps  = { p[s], p[s] + 1 };
        std::vector<llama_seq_id> sq = { s, s };
        std::vector<int8_t> o = { 1, 1 };
        p[s] += 2;
        ok = decode(ctx, t, ps, sq, o, r.logits);
    }
    if (ok) {
        std::vector<llama_token> t = { tk(0, 300), tk(0, 301), tk(1, 300), tk(1, 301) };
        std::vector<llama_pos> ps  = { p[0], p[0] + 1, p[1], p[1] + 1 };
        std::vector<llama_seq_id> sq = { 0, 0, 1, 1 };
        std::vector<int8_t> o = { 0, 1, 0, 1 };
        ok = decode(ctx, t, ps, sq, o, r.logits);
    }

    llama_free(ctx);
    llama_model_free(model);
    r.ok = ok;
    return r;
}

void compare(const run_cfg & a, const run_out & ra, const run_cfg & b, const run_out & rb, bool bitwise) {
    TCHECK(ra.ok && rb.ok, "a run failed (TBO %d ok %d, TBO %d ok %d)", a.tbo, (int) ra.ok, b.tbo, (int) rb.ok);
    if (!ra.ok || !rb.ok) {
        return;
    }
    TCHECK(ra.logits.size() == rb.logits.size() && !ra.logits.empty(), "logit counts %zu vs %zu", ra.logits.size(), rb.logits.size());
    if (ra.logits.size() != rb.logits.size()) {
        return;
    }
    double max_abs = 0.0;
    double max_ref = 0.0;
    size_t n_diff  = 0;
    bool finite    = true;
    for (size_t i = 0; i < ra.logits.size(); ++i) {
        const double x = ra.logits[i];
        const double y = rb.logits[i];
        finite = finite && std::isfinite(x) && std::isfinite(y);
        max_abs = std::max(max_abs, std::fabs(x - y));
        max_ref = std::max(max_ref, std::fabs(x));
        n_diff += ra.logits[i] != rb.logits[i] ? 1 : 0;
    }
    printf("  TBO %d vs TBO %d (pos mask %d, n_rs_seq %d): %zu logits, %zu differ, max |d| %.3g (max |logit| %.3g)\n",
            a.tbo, b.tbo, (int) a.pos_mask, a.n_rs, ra.logits.size(), n_diff, max_abs, max_ref);
    TCHECK(finite, "non-finite logits");
    if (bitwise) {
        TCHECK(n_diff == 0, "TBO %d vs TBO %d must give the same bits (%zu differ)", a.tbo, b.tbo, n_diff);
    } else {
        // the float order of the batched kernels moves a logit by ~1e-7 of the largest one; a state written to the wrong
        // cell moves it by ~1e-4 (checked by hand with a wrong chain offset)
        TCHECK(max_abs <= 1e-5*max_ref, "TBO %d vs TBO %d: max |d| %.3g is more than float order (1e-5 x %.3g)", a.tbo, b.tbo,
                max_abs, max_ref);
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("test-fn-l22-tbo-model: SKIP (no model file; make one with test-llama-archs -a qwen4exp -o <dir>)\n");
        return 0;
    }
    const char * path = argv[1];
    const int n_threads = argc > 2 ? std::max(1, atoi(argv[2])) : 4;
    g_verbose = getenv("TBO_TEST_VERBOSE") ? atoi(getenv("TBO_TEST_VERBOSE")) : 0;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    for (bool pos_mask : { true, false }) {
        for (int n_rs : { 0, 3 }) {
            run_out outs[3];
            int lines[3] = { 0, 0, 0 };
            for (int tbo = 0; tbo <= 2; ++tbo) {
                const int before = g_tbo_lines.load();
                outs[tbo] = run(path, { tbo, pos_mask, n_rs }, n_threads);
                lines[tbo] = g_tbo_lines.load() - before;
            }
            TCHECK(lines[0] == 0 && lines[1] > 0 && lines[2] > 0, "two-chain graphs built: TBO 0 %d, 1 %d, 2 %d (pos mask %d, n_rs %d)",
                    lines[0], lines[1], lines[2], (int) pos_mask, n_rs);
            compare({ 2, pos_mask, n_rs }, outs[2], { 1, pos_mask, n_rs }, outs[1], true);
            compare({ 1, pos_mask, n_rs }, outs[1], { 0, pos_mask, n_rs }, outs[0], false);
        }
    }

    llama_backend_free();
    printf("test-fn-l22-tbo-model: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
