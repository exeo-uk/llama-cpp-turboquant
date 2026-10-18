// [TAG_FN_L22_P4] CPU unit test of round 22's package P4 (prompt stream state machine): the pure pieces in
// src/llama-fn-auto.h - the FULL reserve classes and their tail need, the measured graph node budget, the static sampling
// count, the state writer's merged reads (#30694) and the next prompt ubatch width. No GPU, no model.
// Exit code 0 iff every check passes.

#include "llama.h"

#include "../src/llama-fn-auto.h"

#include <cstdint>
#include <cstdio>
#include <utility>
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

static constexpr size_t MiB = 1024 * 1024;

// [TAG_FN_L22_LENDCLASS] the smallest class below the command's width that holds the whole prompt
static void test_lend_class() {
    TCHECK(llama_fn_l22_lend_class(8192, 900, 0) == 1024, "a 900-token prompt: 1K");
    TCHECK(llama_fn_l22_lend_class(8192, 1000, 24) == 1024, "1000 + 24 = 1024: 1K");
    TCHECK(llama_fn_l22_lend_class(8192, 1000, 25) == 2048, "1025: 2K");
    TCHECK(llama_fn_l22_lend_class(8192, 3000, 1096) == 4096, "4096: 4K");
    TCHECK(llama_fn_l22_lend_class(8192, 3000, 1097) == 0, "4097: the command's width");
    TCHECK(llama_fn_l22_lend_class(8192, 900, -1) == 0, "no hint: no class (a later batch could be wider)");
    TCHECK(llama_fn_l22_lend_class(0, 900, 0) == 0, "off");
    TCHECK(llama_fn_l22_lend_class(2048, 900, 0) == 1024, "base 2048: 1K");
    TCHECK(llama_fn_l22_lend_class(2048, 1500, 0) == 0, "base 2048, 1500 tokens: no class below the base");
    TCHECK(llama_fn_l22_lend_class(1024, 500, 0) == 0, "base 1024: none");
    TCHECK(llama_fn_l22_lend_class(8192, 4, 0) == 1024, "a 4-token tail of a short prompt: 1K");

    // the class's tail need: full - small + pool + max(128 MiB, full/16), capped by the command width's need
    TCHECK(llama_fn_l22_class_need(1000*MiB, 200*MiB, 0, 0) == 928*MiB, "800 + 128 MiB slack");
    TCHECK(llama_fn_l22_class_need(4096*MiB, 176*MiB, 0, 0) == 4176*MiB, "3920 + 256 MiB slack");
    TCHECK(llama_fn_l22_class_need(4096*MiB, 176*MiB, 64*MiB, 0) == 4240*MiB, "+ the pool extra");
    TCHECK(llama_fn_l22_class_need(4096*MiB, 176*MiB, 0, 1000*MiB) == 1000*MiB, "capped");
    TCHECK(llama_fn_l22_class_need(100*MiB, 200*MiB, 0, 0) == 0, "no larger than SMALL: 0");
}

// [TAG_FN_L22_SCHEDBUF] [TAG_FN_L22_SAMPSTATIC]
static void test_nodes() {
    TCHECK(llama_fn_l22_sched_nodes(327680, 10000) == 23192, "an 8K reserve of 10000 nodes: x 3/2 + 8192");
    TCHECK(llama_fn_l22_sched_nodes(20000, 10000) == 20000, "never above the formula");
    TCHECK(llama_fn_l22_sched_nodes(327680, 0) == 327680, "not measured: the formula");

    TCHECK(llama_fn_l22_sampling_nodes(12, 10, 3) == 30, "3 chains of 10 nodes");
    TCHECK(llama_fn_l22_sampling_nodes(50, 10, 3) == 50, "never below the old count");
    TCHECK(llama_fn_l22_sampling_nodes(10, 10, 1) == 10, "one chain: unchanged");
    TCHECK(llama_fn_l22_sampling_nodes(10, 10, 0) == 10, "0 outputs per sequence counts as 1");
}

// [TAG_FN_L22_E6] (#30694): small consecutive ranges of one tensor merge into one read
static void test_io_merge() {
    int t0 = 0;
    int t1 = 0;
    std::vector<llama_fn_l22_io_range> r = {
        { &t0, 0, 4096 }, { &t0, 8192, 4096 }, { &t0, 16384, 4096 }, // one group (gaps within 8x payload + 1 MiB)
        { &t1, 0, 4096 },                                           // another tensor
        { &t0, 32768, 100000 },                                     // not small
        { &t0, 200000, 1000 }, { &t0, 201000, 1000 },               // adjacent
        { &t0, 0, 1000 },                                           // goes back: a new group
        { &t0, (size_t) 10 << 20, 1000 },                           // too far from the previous one
    };
    std::vector<std::pair<size_t, size_t>> g;
    llama_fn_l22_io_merge(r, g);
    const std::vector<std::pair<size_t, size_t>> want = { { 0, 3 }, { 3, 4 }, { 4, 5 }, { 5, 7 }, { 7, 8 }, { 8, 9 } };
    TCHECK(g == want, "io merge groups (%zu groups)", g.size());

    // every byte of every range is covered by its group's span, and the spans are in bounds of the ranges
    for (const auto & gr : g) {
        const size_t lo = r[gr.first].offset;
        const size_t hi = r[gr.second - 1].offset + r[gr.second - 1].size;
        for (size_t k = gr.first; k < gr.second; ++k) {
            TCHECK(r[k].tensor == r[gr.first].tensor && r[k].offset >= lo && r[k].offset + r[k].size <= hi,
                    "range %zu inside its group's span", k);
        }
    }

    std::vector<llama_fn_l22_io_range> none;
    llama_fn_l22_io_merge(none, g);
    TCHECK(g.empty(), "no ranges, no groups");
}

// [TAG_FN_L22_GENSHARE2] the D2 getter's width for a SMALL context: the next switch's rule, then the class
static void test_next_width() {
    TCHECK(llama_fn_l22_next_width(8192, 16384, true, false, false, 20000) == 16384, "l21: a 20K prompt -> 16K");
    TCHECK(llama_fn_l22_next_width(8192, 16384, true, false, false, 5000) == 8192, "l21: 5K -> the command's width");
    TCHECK(llama_fn_l22_next_width(8192, 16384, true, false, true, 900) == 1024, "class 1K");
    TCHECK(llama_fn_l22_next_width(8192, 16384, true, false, true, -1) == 8192, "no hint: the command's width");
    TCHECK(llama_fn_l22_next_width(8192, 16384, true, true, false, 20000) == 8192, "no borrow: the command's width");
    TCHECK(llama_fn_l22_next_width(0, 8192, false, false, true, 900) == 8192, "no lend: n_ubatch");
    TCHECK(llama_fn_l22_next_width(8192, 16384, false, false, false, 9000) == 16384, "l20 rule: above the base -> big");
}

int main() {
    test_lend_class();
    test_nodes();
    test_io_merge();
    test_next_width();
    printf("test-fn-l22-p4: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
