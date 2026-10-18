// [TAG_FN_L22] round 22 P0: CPU unit test of the scaffold (src/llama-fn-l22.h, the round 22 entries of src/llama-ext.h):
// the tok_aux side channel from common_batch through get_sub_batch into llama_batch_ext, the decode-phase rule (fn_gen)
// and the X lever bits, the switch parser, the ggml_moe_host_sum flags op param, and the stubs' "today" answers.
// No GPU, no model. Exit code 0 iff every check passes.

#include "llama.h"
#include "llama-cpp.h"
#include "common.h"
#include "ggml.h"
#include "ggml-moe-bridge.h"

#include "../src/llama-ext.h"
#include "../src/llama-fn-l22.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const char * k_x_names[] = {
    "LLAMA_FN_L22_X_TAILSKIP", "LLAMA_FN_L22_X_TAILSKIP_MULTI", "LLAMA_FN_L22_X_PRIOR", "LLAMA_FN_L22_X_DEFER",
    "LLAMA_FN_L22_X_RESIDENT", "LLAMA_FN_L22_CPUSHARE", "LLAMA_FN_L22_SAMPSTATIC",
};

static void clear_x_env() {
    for (const char * n : k_x_names) {
        set_env(n, nullptr);
    }
}

// tok_aux set on a common_batch reaches llama_batch_ext through get_sub_batch at the right indices
static void test_tok_aux() {
    const int n_embd = 4;
    const int n      = 8;

    common_batch cb;
    cb.batch.reset(llama_batch_ext_init_test(32, n_embd, 2, 1000, 1));
    TCHECK(cb.batch != nullptr, "test batch");
    if (!cb.batch) {
        return;
    }

    std::vector<float> rows((size_t) n*n_embd);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = 0.25f*(float) i;
    }

    // a merged image turn: every entry is an embedding; some carry the token they stand for
    const llama_token aux[n] = { LLAMA_TOKEN_NULL, 101, LLAMA_TOKEN_NULL, 103, 248056, LLAMA_TOKEN_NULL, 106, 248056 };
    for (int i = 0; i < n; ++i) {
        const llama_pos  pos  = i;
        const llama_embd embd = { rows.data() + (size_t) i*n_embd, 1, (size_t) n_embd };
        const int32_t idx = cb.add_embd(embd, &pos, 0, i == n - 1);
        TCHECK(idx == i, "add_embd index %d", idx);
        TCHECK(cb.tokens[idx].tok_aux == LLAMA_TOKEN_NULL, "a new entry has no tok_aux");
        if (aux[i] != LLAMA_TOKEN_NULL) {
            TCHECK(cb.set_token_aux(idx, aux[i]), "set_token_aux %d", i);
        }
    }
    TCHECK(!cb.set_token_aux(-1, 5) && !cb.set_token_aux(n, 5), "set_token_aux out of range");

    llama_batch_ext * be = cb.get_sub_batch(0, n);
    for (int i = 0; i < n; ++i) {
        TCHECK(llama_batch_ext_get_token_aux(be, i) == aux[i], "whole batch: entry %d tok_aux %d want %d", i,
               llama_batch_ext_get_token_aux(be, i), aux[i]);
    }
    TCHECK(llama_batch_ext_get_token_aux(be, n) == LLAMA_TOKEN_NULL && llama_batch_ext_get_token_aux(be, -1) == LLAMA_TOKEN_NULL,
           "get out of range");

    // a sub-batch: entry off + i lands at index i
    const int off = 3;
    const int m   = 4;
    be = cb.get_sub_batch(off, m);
    for (int i = 0; i < m; ++i) {
        TCHECK(llama_batch_ext_get_token_aux(be, i) == aux[off + i], "sub-batch: index %d tok_aux %d want %d", i,
               llama_batch_ext_get_token_aux(be, i), aux[off + i]);
    }

    // clearing an entry's value (LLAMA_TOKEN_NULL) and re-rendering: no stale id survives the batch clear
    TCHECK(cb.set_token_aux(off, LLAMA_TOKEN_NULL), "clear one");
    be = cb.get_sub_batch(off, m);
    TCHECK(llama_batch_ext_get_token_aux(be, 0) == LLAMA_TOKEN_NULL, "cleared entry renders NULL");

    // text tokens: no tok_aux unless set
    common_batch ct;
    ct.batch.reset(llama_batch_ext_init_test(32, n_embd, 2, 1000, 1));
    for (int i = 0; i < 5; ++i) {
        ct.add(10 + i, i, 0, i == 4);
    }
    be = ct.get_sub_batch(0, 5);
    bool none = true;
    for (int i = 0; i < 5; ++i) {
        none = none && llama_batch_ext_get_token_aux(be, i) == LLAMA_TOKEN_NULL;
    }
    TCHECK(none, "text batch without tok_aux");

    // direct API on a batch: set / get, bounds
    llama_batch_ext_set_token_aux(be, 2, 77);
    TCHECK(llama_batch_ext_get_token_aux(be, 2) == 77, "direct set");
    llama_batch_ext_set_token_aux(be, 99, 77); // ignored
    llama_batch_ext_set_token_aux(nullptr, 0, 77);
    TCHECK(llama_batch_ext_get_token_aux(nullptr, 0) == LLAMA_TOKEN_NULL, "null batch");
    llama_batch_ext_clear(be);
    TCHECK(llama_batch_ext_get_token_aux(be, 2) == LLAMA_TOKEN_NULL, "clear drops it");
}

static void test_switch_parse() {
    TCHECK(!llama_fn_l22_switch_on(nullptr), "null");
    TCHECK(!llama_fn_l22_switch_on(""), "empty");
    TCHECK(!llama_fn_l22_switch_on("0"), "0");
    TCHECK(!llama_fn_l22_switch_on(" 0.0 "), "0.0");
    TCHECK(!llama_fn_l22_switch_on("off") && !llama_fn_l22_switch_on("FALSE") && !llama_fn_l22_switch_on("no"), "off words");
    TCHECK(llama_fn_l22_switch_on("1") && llama_fn_l22_switch_on("7") && llama_fn_l22_switch_on("0.1"), "numbers");
    TCHECK(llama_fn_l22_switch_on("check") && llama_fn_l22_switch_on("on") && llama_fn_l22_switch_on("raw"), "words");
}

// plan 3.8 and P0 spec 4: fn_gen 0 with no X switch, 0 for fewer output rows than rows, 1 for an all-output
// single-sequence GEN ubatch with X_TAILSKIP=7
static void test_fn_gen() {
    clear_x_env();

    llama_fn_l22_x_cfg cfg = llama_fn_l22_x_read(nullptr);
    TCHECK(!cfg.any() && cfg.tailskip == 0 && !cfg.tail_multi && !cfg.prior && !cfg.defer && !cfg.resident, "unset = no lever");
    TCHECK(llama_fn_l22_gen_rule(LLAMA_FN_PHASE_GEN, 3, 3, cfg.any()) == 0, "no X switch: fn_gen 0");
    TCHECK(llama_fn_l22_x_bits(cfg, 1, 1) == 0, "no X switch: no bits even with fn_gen");
    TCHECK(llama_fn_l22_hooks_read(nullptr) == 0, "unset = no graph hook");

    set_env("LLAMA_FN_L22_X_TAILSKIP", "7");
    cfg = llama_fn_l22_x_read(nullptr);
    TCHECK(cfg.any() && cfg.tailskip == 7, "TAILSKIP=7 read");

    TCHECK(llama_fn_l22_gen_rule(LLAMA_FN_PHASE_PROMPT, 3, 3, cfg.any()) == 0, "PROMPT phase: 0");
    TCHECK(llama_fn_l22_gen_rule(LLAMA_FN_PHASE_GEN, 512, 1, cfg.any()) == 0, "prompt ubatch (1 output of 512): 0");
    TCHECK(llama_fn_l22_gen_rule(LLAMA_FN_PHASE_GEN, 4, 0, cfg.any()) == 0, "checkpoint tail without outputs: 0");
    TCHECK(llama_fn_l22_gen_rule(LLAMA_FN_PHASE_GEN, 4, 3, cfg.any()) == 0, "3 of 4 rows output: 0");
    TCHECK(llama_fn_l22_gen_rule(LLAMA_FN_PHASE_GEN, 0, 0, cfg.any()) == 0, "empty ubatch: 0");
    TCHECK(llama_fn_l22_gen_rule(7, 3, 3, cfg.any()) == 0, "unknown phase: 0");

    const uint8_t g = llama_fn_l22_gen_rule(LLAMA_FN_PHASE_GEN, 3, 3, cfg.any());
    TCHECK(g == 1, "all-output single-sequence GEN ubatch with X_TAILSKIP=7: fn_gen 1");
    TCHECK(llama_fn_l22_x_bits(cfg, g, 1) == LLAMA_FN_L22_XB_TAIL, "one sequence: TAIL");
    TCHECK(llama_fn_l22_x_bits(cfg, g, 2) == 0, "two sequences without MULTI: no TAIL");
    TCHECK(llama_fn_l22_x_bits(cfg, 0, 1) == 0, "fn_gen 0: no bits");

    set_env("LLAMA_FN_L22_X_TAILSKIP_MULTI", "1");
    cfg = llama_fn_l22_x_read(nullptr);
    TCHECK(llama_fn_l22_x_bits(cfg, 1, 2) == LLAMA_FN_L22_XB_TAIL, "two sequences with MULTI: TAIL");

    set_env("LLAMA_FN_L22_X_TAILSKIP", "0");
    set_env("LLAMA_FN_L22_X_TAILSKIP_MULTI", nullptr);
    set_env("LLAMA_FN_L22_X_DEFER", "check");
    set_env("LLAMA_FN_L22_X_PRIOR", "0.0");
    set_env("LLAMA_FN_L22_X_RESIDENT", "3");
    cfg = llama_fn_l22_x_read(nullptr);
    TCHECK(cfg.tailskip == 0 && cfg.defer && !cfg.prior && cfg.resident && cfg.any(), "DEFER=check, PRIOR=0.0, RESIDENT=3");
    TCHECK(llama_fn_l22_x_bits(cfg, 1, 1) == (LLAMA_FN_L22_XB_DEFER | LLAMA_FN_L22_XB_RESIDENT), "bits DEFER | RESIDENT");

    set_env("LLAMA_FN_L22_X_PRIOR", "0.5");
    cfg = llama_fn_l22_x_read(nullptr);
    TCHECK(llama_fn_l22_x_bits(cfg, 1, 4) == (LLAMA_FN_L22_XB_PRIOR | LLAMA_FN_L22_XB_DEFER | LLAMA_FN_L22_XB_RESIDENT),
           "PRIOR acts for any sequence count");

    set_env("LLAMA_FN_L22_CPUSHARE", "1");
    TCHECK(llama_fn_l22_hooks_read(nullptr) == (LLAMA_FN_L22_HOOK_CPUSHARE | LLAMA_FN_L22_HOOK_PRIOR),
           "CPUSHARE hook bit, PRIOR bit from X_PRIOR / X_RESIDENT");
    set_env("LLAMA_FN_L22_X_PRIOR", nullptr);
    TCHECK(llama_fn_l22_hooks_read(nullptr) == (LLAMA_FN_L22_HOOK_CPUSHARE | LLAMA_FN_L22_HOOK_PRIOR), "PRIOR bit from X_RESIDENT");
    set_env("LLAMA_FN_L22_X_RESIDENT", "0");
    set_env("LLAMA_FN_L22_CPUSHARE", "off");
    set_env("LLAMA_FN_L22_SAMPSTATIC", "1");
    TCHECK(llama_fn_l22_hooks_read(nullptr) == LLAMA_FN_L22_HOOK_SAMPSTATIC, "SAMPSTATIC hook bit only (R-P1-1)");

    clear_x_env();

    // the phase API on no context: PROMPT, no crash
    llama_set_decode_phase_ext(nullptr, LLAMA_FN_PHASE_GEN);
    TCHECK(llama_get_decode_phase_ext(nullptr) == LLAMA_FN_PHASE_PROMPT, "null ctx phase");
}

// the sync op's flags (op param 1): 0 until set, so an unset TAILSKIP builds the op exactly as before
static void test_host_sum_flags() {
    ggml_init_params ip = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    TCHECK(ctx != nullptr, "ggml_init");
    if (!ctx) {
        return;
    }
    const int64_t n_embd = 8, n_ff = 16, n_exp = 4, n_used = 2, T = 3;
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_used, T);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_exp);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_exp);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_exp);
    ggml_tensor * s    = ggml_moe_host_sum(ctx, x, ids, w, up, gate, down, nullptr, 0);
    TCHECK(ggml_moe_host_sum_get_flags(s) == 0, "a new sync op carries no flags");
    TCHECK(((const int32_t *) s->op_params)[1] == 0, "op param 1 zero");
    ggml_moe_host_sum_set_flags(s, GGML_MOE_BRIDGE_JOB_TAIL);
    TCHECK(ggml_moe_host_sum_get_flags(s) == GGML_MOE_BRIDGE_JOB_TAIL, "TAIL flag kept");
    TCHECK(((const int32_t *) s->op_params)[0] == 0, "table_miss untouched");
    TCHECK(ggml_moe_host_sum_get_flags(x) == 0 && ggml_moe_host_sum_get_flags(nullptr) == 0, "other ops: 0");
    TCHECK(GGML_MOE_BRIDGE_JOB_TAIL == 8 && GGML_MOE_BRIDGE_JOB_DEFER == 16, "flag values");
    TCHECK((GGML_MOE_BRIDGE_JOB_TAIL & (GGML_MOE_BRIDGE_JOB_TABLE | GGML_MOE_BRIDGE_JOB_DMA | GGML_MOE_BRIDGE_JOB_HINT)) == 0 &&
           (GGML_MOE_BRIDGE_JOB_DEFER & (GGML_MOE_BRIDGE_JOB_TABLE | GGML_MOE_BRIDGE_JOB_DMA | GGML_MOE_BRIDGE_JOB_HINT |
                                         GGML_MOE_BRIDGE_JOB_TAIL)) == 0, "flag bits distinct");
    ggml_free(ctx);
}

// the owners' stubs answer as today's code behaves
static void test_stubs() {
    llama_fn_pin_seg seg = {};
    int buf[16];
    TCHECK(llama_fn_pin_query(buf, sizeof(buf), &seg) && seg.src == buf && seg.len == sizeof(buf) && !seg.pinned && seg.ticket < 0,
           "pin_query: one unpinned segment");
    TCHECK(!llama_fn_pin_active() && !llama_fn_pin_on_oom(1 << 20) && !llama_fn_pin_on_oom_ext(nullptr, 1 << 20), "no pins");
    uint64_t b = 1, u = 1;
    TCHECK(!llama_fn_dxgi_query(0, 1, &b, &u) && b == 0 && u == 0, "dxgi stub");
    TCHECK(llama_fn_res_class(nullptr, 0, 0) == LLAMA_FN_RES_RAM && llama_fn_res_gen(nullptr) == 0 &&
           llama_fn_res_stream_bytes(nullptr, LLAMA_FN_RES_SSD) == 0, "residency stub");
    llama_fn_dio_req rq = { buf, sizeof(buf), buf };
    TCHECK(llama_fn_dio_submit(nullptr, &rq, 1) == -1, "dio stub");
    TCHECK(!llama_fn_dmashare_wanted(nullptr, nullptr) && llama_fn_dmashare_view(nullptr, nullptr, 1) == nullptr &&
           !llama_fn_dmashare_plan(nullptr, nullptr), "dmashare stub");
    TCHECK(llama_fn_kvrows_attn(nullptr, nullptr) == nullptr, "kvrows stub");
    TCHECK(llama_fn_route_prior(nullptr, nullptr, 0) == nullptr, "prior stub returns its input");
    TCHECK(llama_fn_cvec_project(nullptr, nullptr, nullptr, 0) == nullptr, "cvec stub returns its input");
    TCHECK(llama_model_media_token(nullptr) == LLAMA_TOKEN_NULL && llama_fn_imgpass_mask_mode(nullptr) == 1 &&
           !llama_model_token_embd_rows(nullptr, nullptr, 0, nullptr), "P1 stubs");
    const uint8_t * p = (const uint8_t *) buf;
    size_t n = 7;
    TCHECK(!llama_fn_expert_host_range(nullptr, 0, 0, 0, &p, &n) && p == nullptr && n == 0, "expert range without a model");
}

static void test_passlog() {
    // read once per process: unset here (the test never sets it) and no model loaded -> 0
    TCHECK(llama_fn_l22_passlog_level() == 0, "PASSLOG unset, no profile: 0");
    llama_fn_l22_log("pass", "this line must not appear (level 0)");
    const llama_fn_l22_counters c = llama_fn_l22_counters_now();
#ifdef _WIN32
    TCHECK(c.faults > 0, "page faults read");
    TCHECK(c.avail_ram > 0, "available RAM read");
#endif
    fprintf(stderr, "counters: faults %llu, sys page reads %llu, avail %.1f GiB, standby %.1f GiB\n",
            (unsigned long long) c.faults, (unsigned long long) c.sys_page_reads, c.avail_ram/1073741824.0,
            c.standby/1073741824.0);
}

int main() {
    set_env("LLAMA_FN_L22_PASSLOG", nullptr);
    test_passlog();
    test_tok_aux();
    test_switch_parse();
    test_fn_gen();
    test_host_sum_flags();
    test_stubs();
    printf("test-fn-l22-batch: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
