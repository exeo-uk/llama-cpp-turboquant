// [TAG_UPSTREAM_29510] [TAG_FN_L22_KVROWS] [TAG_FN_L22_QSA_SEQPOOLS] [TAG_FN_L22_QSA_SHAREPFX] round 22 P13, CPU only.
//
// Part A (src/llama-fn-kvrows.h, the per-sequence K/V cell lists of a unified turbot cache): the slice test on the token
// data, the list length, the plan gates, the lists from the positional vectors, the per-slice q_pos, and the per-slice FA
// graph on the CPU backend against the one FA over the whole cache with the multi-sequence positional mask (the CPU FA
// ignores the list, so this proves the slicing, the views, the q_pos rows and the concat order; the CUDA rows mode is
// test-backend-ops -o FLASH_ATTN_EXT -p kv_rows).
// Part B (src/llama-memory-hybrid-idx.h): the QSA list width rule, the check of listed against full lists, the shared
// prefix shape and its safety rule, and the rep -> members check.

#include "ggml.h"
#include "ggml-cpu.h"

#include "llama-batch.h"
#include "llama-kv-cells.h"
#include "llama-fn-kvrows.h"
#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
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

// a llama_ubatch over tokens with one (or two) sequence ids each, and the storage it points into
struct ub_t {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   ids;     // two slots per token
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id>   unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         out;
    llama_ubatch                ub = {};
};

// seqs[i]: the first sequence of token i, seq2[i] >= 0: a second one; positions given
static void make_ub(ub_t & u, const std::vector<llama_seq_id> & seqs, const std::vector<llama_pos> & pos,
                    const std::vector<llama_seq_id> & seq2 = {}) {
    const size_t n = seqs.size();
    u.pos = pos;
    u.n_seq_id.assign(n, 1);
    u.ids.assign(2*n, 0);
    u.seq_id.assign(n, nullptr);
    u.out.assign(n, 1);
    u.seq_idx.assign(LLAMA_MAX_SEQ, -1);
    u.unq.clear();
    for (size_t i = 0; i < n; ++i) {
        u.ids[2*i] = seqs[i];
        if (!seq2.empty() && seq2[i] >= 0) {
            u.ids[2*i + 1] = seq2[i];
            u.n_seq_id[i]  = 2;
        }
        u.seq_id[i] = &u.ids[2*i];
        for (int k = 0; k < u.n_seq_id[i]; ++k) {
            const llama_seq_id s = u.ids[2*i + k];
            if (u.seq_idx[s] < 0) {
                u.seq_idx[s] = (int32_t) u.unq.size();
                u.unq.push_back(s);
            }
        }
    }
    u.ub.b_equal_seqs = 1;
    u.ub.n_tokens     = (uint32_t) n;
    u.ub.n_seq_tokens = (uint32_t) n;
    u.ub.n_seqs       = 1;
    u.ub.n_seqs_unq   = (uint32_t) u.unq.size();
    u.ub.n_pos        = 1;
    u.ub.token        = nullptr;
    u.ub.embd         = nullptr;
    u.ub.pos          = u.pos.data();
    u.ub.n_seq_id     = u.n_seq_id.data();
    u.ub.seq_id       = u.seq_id.data();
    u.ub.seq_id_unq   = u.unq.data();
    u.ub.seq_idx      = u.seq_idx.data();
    u.ub.output       = u.out.data();
    u.ub.decision_order = nullptr;
}

// n_slices groups of n_tps tokens, group s of sequence seqs[s], positions from start[s]
static void make_grouped(ub_t & u, const std::vector<llama_seq_id> & seqs, uint32_t n_tps, const std::vector<llama_pos> & start) {
    std::vector<llama_seq_id> s_of;
    std::vector<llama_pos>    p_of;
    for (size_t s = 0; s < seqs.size(); ++s) {
        for (uint32_t t = 0; t < n_tps; ++t) {
            s_of.push_back(seqs[s]);
            p_of.push_back(start[s] + (llama_pos) t);
        }
    }
    make_ub(u, s_of, p_of);
}

// the positional vectors as llama_kv_cache::set_input_kv_pos / set_input_q_pos write them (one stream): one row for a
// single-sequence ubatch, two rows (positions, sequence sets / bits) otherwise
static void make_pos_vectors(const llama_kv_cells & cells, const llama_ubatch & ub, int64_t n_kv,
                             std::vector<int32_t> & kv_pos, std::vector<int32_t> & q_pos) {
    const uint32_t n_seqs = ub.n_seqs_unq;
    const int rows = n_seqs > 1 ? 2 : 1;
    kv_pos.assign((size_t) n_kv*rows, 0);
    q_pos.assign((size_t) ub.n_tokens*rows, 0);
    for (int64_t j = 0; j < n_kv; ++j) {
        uint32_t set = 0;
        if (j < (int64_t) cells.size() && !cells.is_empty((uint32_t) j)) {
            for (uint32_t b = 0; b < n_seqs; ++b) {
                set |= cells.seq_has((uint32_t) j, ub.seq_id_unq[b]) ? (1u << b) : 0u;
            }
        }
        kv_pos[j] = set != 0 ? cells.pos_get((uint32_t) j) : -1;
        if (rows == 2) {
            kv_pos[n_kv + j] = (int32_t) set;
        }
    }
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        q_pos[i] = ub.pos[i];
        if (rows == 2) {
            q_pos[ub.n_tokens + i] = (int32_t) (1u << ub.seq_idx[ub.seq_id[i][0]]);
        }
    }
}

// a cache of n_cells cells: n_seq sequences take cells round-robin in runs, every share-th used cell is also given to the
// next sequence, every gap-th cell stays empty; positions count per sequence. own[s] = the cells of sequence s
static void make_cells(llama_kv_cells & cells, uint32_t n_cells, uint32_t n_used, int n_seq, int share, int gap,
                       std::vector<llama_pos> & next_pos, uint32_t seed) {
    std::mt19937 rng(seed);
    cells.resize(n_cells);
    next_pos.assign(LLAMA_MAX_SEQ, 0);
    uint32_t used = 0;
    for (uint32_t j = 0; j < n_cells && used < n_used; ++j) {
        if (gap > 0 && (j % (uint32_t) gap) == (uint32_t) gap - 1) {
            continue;
        }
        const llama_seq_id s = (llama_seq_id) ((j / 7 + rng() % 2) % (uint32_t) n_seq);
        cells.pos_set(j, next_pos[s]++);
        cells.seq_add(j, s);
        if (share > 0 && (used % (uint32_t) share) == 0 && n_seq > 1) {
            const llama_seq_id s2 = (s + 1) % n_seq;
            cells.seq_add(j, s2);
            next_pos[s2] = std::max(next_pos[s2], cells.pos_get(j) + 1);
        }
        ++used;
    }
}

//
// part A
//

static void test_cfg_default() {
    const llama_fn_kvrows_cfg cfg = llama_fn_kvrows_cfg_env();
    const char * e = getenv("LLAMA_KV_ROWS");
    if (e == nullptr) {
        TCHECK(cfg.mode == 0, "LLAMA_KV_ROWS unset: mode %d, want 0", cfg.mode);
    }
    if (getenv("LLAMA_KV_ROWS_MAX_TPS") == nullptr) {
        TCHECK(cfg.max_tps == 8, "max_tps %u, want 8", cfg.max_tps);
    }
    if (getenv("LLAMA_KV_ROWS_MIN_TPS") == nullptr) {
        TCHECK(cfg.min_tps == 1, "min_tps %u, want 1", cfg.min_tps);
    }
}

static void test_slices() {
    ub_t u;
    make_grouped(u, { 0, 1, 2, 3 }, 3, { 10, 20, 30, 40 });
    TCHECK(llama_fn_kvrows_slices(u.ub) == 4, "4 x 3 grouped: %u slices", llama_fn_kvrows_slices(u.ub));

    make_grouped(u, { 5 }, 3, { 7 });
    TCHECK(llama_fn_kvrows_slices(u.ub) == 1, "one sequence: %u slices", llama_fn_kvrows_slices(u.ub));

    // interleaved: 0 1 0 1 0 1
    make_ub(u, { 0, 1, 0, 1, 0, 1 }, { 0, 0, 1, 1, 2, 2 });
    TCHECK(llama_fn_kvrows_slices(u.ub) == 0, "interleaved sequences must give no slices");

    // a token in two sequences
    make_ub(u, { 0, 0, 1, 1 }, { 0, 1, 0, 1 }, { -1, 1, -1, -1 });
    TCHECK(llama_fn_kvrows_slices(u.ub) == 0, "a token in two sequences must give no slices");

    // 5 tokens of 2 sequences
    make_ub(u, { 0, 0, 0, 1, 1 }, { 0, 1, 2, 0, 1 });
    TCHECK(llama_fn_kvrows_slices(u.ub) == 0, "uneven groups must give no slices");

    // reserve-like ubatch: no sequence ids
    make_grouped(u, { 0, 1 }, 2, { 0, 0 });
    llama_ubatch r = u.ub;
    r.seq_id = nullptr;
    TCHECK(llama_fn_kvrows_slices(r) == 0, "no sequence ids must give no slices");
    r = u.ub;
    r.n_seq_id = nullptr;
    TCHECK(llama_fn_kvrows_slices(r) == 0, "no n_seq_id must give no slices");

    // groups of equal size whose tokens mix inside a group: 0 0 1 | 1 1 0 (n_seqs_unq 2, n_tps 3)
    make_ub(u, { 0, 0, 1, 1, 1, 0 }, { 0, 1, 0, 1, 2, 2 });
    TCHECK(llama_fn_kvrows_slices(u.ub) == 0, "a group with two sequences must give no slices");
}

static void test_n_rows_and_plan() {
    llama_kv_cells cells;
    cells.resize(4096);
    // seq 0: 300 cells, seq 1: 700 cells, seq 2: 10 cells (cells interleaved in blocks)
    std::vector<uint32_t> n_of = { 300, 700, 10 };
    std::vector<llama_pos> p(3, 0);
    uint32_t j = 0;
    while (p[0] < (llama_pos) n_of[0] || p[1] < (llama_pos) n_of[1] || p[2] < (llama_pos) n_of[2]) {
        for (int s = 0; s < 3; ++s) {
            for (int k = 0; k < 5 && p[s] < (llama_pos) n_of[s]; ++k) {
                cells.pos_set(j, p[s]++);
                cells.seq_add(j, s);
                ++j;
            }
        }
    }
    ub_t u;
    make_grouped(u, { 0, 1, 2 }, 2, { 300, 700, 10 });
    TCHECK(llama_fn_kvrows_n_rows(cells, u.ub, 3, 2048) == 768, "n_rows %u, want 768 (700 padded to 256)",
           llama_fn_kvrows_n_rows(cells, u.ub, 3, 2048));
    TCHECK(llama_fn_kvrows_n_rows(cells, u.ub, 3, 768) == 0, "lists not shorter than n_kv must be refused");
    TCHECK(llama_fn_kvrows_n_rows(cells, u.ub, 3, 1000) == 0, "the cells of the slices add up to more than n_kv (1010)");

    make_grouped(u, { 1, 2 }, 2, { 700, 10 });
    TCHECK(llama_fn_kvrows_n_rows(cells, u.ub, 2, 769) == 768, "n_rows just below n_kv (710 cells)");

    // an empty sequence in the ubatch
    make_grouped(u, { 0, 7 }, 2, { 300, 0 });
    TCHECK(llama_fn_kvrows_n_rows(cells, u.ub, 2, 2048) == 0, "a sequence without cells must give no lists");

    // shared cells: give 600 of seq 1's cells to seq 0 as well -> seq 0 has 900 cells, sum 1600
    int given = 0;
    for (uint32_t c = 0; c < cells.size() && given < 600; ++c) {
        if (!cells.is_empty(c) && cells.seq_has(c, 1) && !cells.seq_has(c, 0)) {
            cells.seq_add(c, 0);
            ++given;
        }
    }
    make_grouped(u, { 0, 1 }, 2, { 1000, 700 });
    TCHECK(llama_fn_kvrows_n_rows(cells, u.ub, 2, 1500) == 0, "shared cells with n_sum > n_kv must give no lists");
    TCHECK(llama_fn_kvrows_n_rows(cells, u.ub, 2, 4096) == 1024, "shared cells, n_kv 4096: n_rows %u, want 1024",
           llama_fn_kvrows_n_rows(cells, u.ub, 2, 4096));

    // the plan gates
    llama_fn_kvrows_cfg cfg;
    cfg.mode = 0;
    make_grouped(u, { 1, 2 }, 3, { 700, 10 });
    TCHECK(llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096).n_rows == 0, "mode 0: no lists");
    cfg.mode = 1;
    llama_fn_kvrows_plan pl = llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096);
    TCHECK(pl.n_slices == 2 && pl.n_rows == 768 && pl.n_tps == 3, "mode 1, 2 x 3: plan %u / %u / %u", pl.n_slices, pl.n_rows, pl.n_tps);
    make_grouped(u, { 1 }, 3, { 700 });
    TCHECK(llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096).n_rows == 0, "mode 1: one sequence gets no lists");
    cfg.mode = 2;
    pl = llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096);
    TCHECK(pl.n_slices == 1 && pl.n_rows == 768 && pl.n_tps == 3, "mode 2, one sequence: plan %u / %u / %u", pl.n_slices, pl.n_rows, pl.n_tps);
    cfg.mode = 1;
    make_grouped(u, { 1, 2 }, 9, { 700, 10 });
    TCHECK(llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096).n_rows == 0, "9 tokens per slice > max_tps 8: no lists");
    cfg.max_tps = 16;
    TCHECK(llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096).n_rows == 768, "max_tps 16: lists");
    cfg.min_tps = 10;
    TCHECK(llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096).n_rows == 0, "min_tps 10 > 9: no lists");
    cfg.min_tps = 1;
    make_ub(u, { 1, 2, 1, 2 }, { 0, 0, 1, 1 });
    TCHECK(llama_fn_kvrows_plan_cells(cfg, u.ub, cells, 4096).n_rows == 0, "interleaved: no lists");

    TCHECK((llama_fn_kvrows_plan{ 2, 768, 3 } == llama_fn_kvrows_plan{ 2, 768, 3 }), "plan equality");
    TCHECK(!(llama_fn_kvrows_plan{ 2, 768, 3 } == llama_fn_kvrows_plan{ 2, 1024, 3 }), "plan inequality");
}

static void test_fill_lists() {
    for (int n_seq = 2; n_seq <= 4; ++n_seq) {
        for (int share : { 0, 5 }) {
            llama_kv_cells cells;
            std::vector<llama_pos> next;
            make_cells(cells, 1200, 1000, n_seq, share, 13, next, 100u + n_seq*10 + share);

            std::vector<llama_seq_id> seqs;
            std::vector<llama_pos>    starts;
            for (int s = 0; s < n_seq; ++s) {
                seqs.push_back(s);
                starts.push_back(next[s]);
            }
            const uint32_t n_tps = 3;
            ub_t u;
            make_grouped(u, seqs, n_tps, starts);
            // the ubatch's own cells: append n_tps cells per sequence at the end
            uint32_t j = 1100;
            for (int s = 0; s < n_seq; ++s) {
                for (uint32_t t = 0; t < n_tps; ++t, ++j) {
                    cells.pos_set(j, starts[s] + (llama_pos) t);
                    cells.seq_add(j, s);
                }
            }
            const int64_t n_kv = 1200;
            std::vector<int32_t> kv_pos, q_pos;
            make_pos_vectors(cells, u.ub, n_kv, kv_pos, q_pos);

            const uint32_t n_rows = llama_fn_kvrows_n_rows(cells, u.ub, (uint32_t) n_seq, 4096);
            TCHECK(n_rows > 0, "n_seq %d share %d: no list length", n_seq, share);
            std::vector<int32_t> lists((size_t) n_rows*n_seq, 7);
            const bool ok = llama_fn_kvrows_fill_lists(kv_pos.data(), n_kv, 2, q_pos.data(), u.ub.n_tokens, (uint32_t) n_seq, n_rows,
                                                       lists.data());
            TCHECK(ok, "n_seq %d share %d: fill failed", n_seq, share);
            int n_bad = 0;
            int n_shared_listed = 0;
            for (int s = 0; s < n_seq; ++s) {
                // the reference: every non-empty cell of sequence s below n_kv, ascending, then -1
                std::vector<int32_t> ref;
                for (int64_t c = 0; c < n_kv; ++c) {
                    if (!cells.is_empty((uint32_t) c) && cells.seq_has((uint32_t) c, s)) {
                        ref.push_back((int32_t) c);
                        n_shared_listed += cells.seq_count((uint32_t) c) > 1 ? 1 : 0;
                    }
                }
                ref.resize(n_rows, -1);
                n_bad += memcmp(ref.data(), lists.data() + (size_t) s*n_rows, n_rows*sizeof(int32_t)) != 0 ? 1 : 0;
            }
            TCHECK(n_bad == 0, "n_seq %d share %d: %d lists differ from the cells", n_seq, share, n_bad);
            if (share > 0) {
                TCHECK(n_shared_listed > 0, "n_seq %d share %d: the case must list shared cells in two lists", n_seq, share);
            }

            // a list too short: overflow refused
            const bool ok2 = llama_fn_kvrows_fill_lists(kv_pos.data(), n_kv, 2, q_pos.data(), u.ub.n_tokens, (uint32_t) n_seq, 16,
                                                        lists.data());
            TCHECK(!ok2, "n_seq %d share %d: an overflowing list must be refused", n_seq, share);

            // a slice whose tokens carry different bits
            std::vector<int32_t> qbad = q_pos;
            qbad[u.ub.n_tokens + 1] = qbad[u.ub.n_tokens + 1] << 1;
            TCHECK(!llama_fn_kvrows_fill_lists(kv_pos.data(), n_kv, 2, qbad.data(), u.ub.n_tokens, (uint32_t) n_seq, n_rows, lists.data()),
                   "a slice with mixed bits must be refused");
            qbad = q_pos;
            qbad[u.ub.n_tokens] = 3;
            TCHECK(!llama_fn_kvrows_fill_lists(kv_pos.data(), n_kv, 2, qbad.data(), u.ub.n_tokens, (uint32_t) n_seq, n_rows, lists.data()),
                   "a token with two bits must be refused");
        }
    }

    // the one-row form (one sequence, mode 2)
    {
        llama_kv_cells cells;
        std::vector<llama_pos> next;
        make_cells(cells, 600, 500, 3, 0, 11, next, 7);
        ub_t u;
        make_grouped(u, { 1 }, 4, { next[1] });
        std::vector<int32_t> kv_pos, q_pos;
        make_pos_vectors(cells, u.ub, 600, kv_pos, q_pos);
        const uint32_t n_rows = 256;
        std::vector<int32_t> lists(n_rows);
        TCHECK(llama_fn_kvrows_fill_lists(kv_pos.data(), 600, 1, q_pos.data(), 4, 1, n_rows, lists.data()), "one row: fill failed");
        std::vector<int32_t> ref;
        for (uint32_t c = 0; c < 600; ++c) {
            if (!cells.is_empty(c) && cells.seq_has(c, 1)) {
                ref.push_back((int32_t) c);
            }
        }
        ref.resize(n_rows, -1);
        TCHECK(ref == lists, "one row: the list differs from the sequence's cells");
        TCHECK(!llama_fn_kvrows_fill_lists(kv_pos.data(), 600, 1, q_pos.data(), 4, 2, n_rows, lists.data()),
               "one row with two slices must be refused");
    }
}

static void test_split_q_pos() {
    const int64_t n_tok = 12;
    const uint32_t n_slices = 4;
    std::vector<int32_t> q(2*n_tok);
    for (int64_t i = 0; i < n_tok; ++i) {
        q[i]         = (int32_t) (1000 + i)*4096 + (int32_t) (i % 5);   // keyed values pass through
        q[n_tok + i] = 1 << (i / 3);
    }
    std::vector<int32_t> out(2*n_tok, -7);
    llama_fn_kvrows_split_q_pos(q.data(), n_tok, n_slices, out.data());
    int n_bad = 0;
    for (uint32_t s = 0; s < n_slices; ++s) {
        for (int64_t t = 0; t < 3; ++t) {
            n_bad += out[s*6 + t]     != q[s*3 + t]         ? 1 : 0;
            n_bad += out[s*6 + 3 + t] != q[n_tok + s*3 + t] ? 1 : 0;
        }
    }
    TCHECK(n_bad == 0, "split q_pos: %d values differ", n_bad);
}

static ggml_context * new_ctx(size_t mib) {
    ggml_init_params ip = { mib*1024*1024, nullptr, false };
    return ggml_init(ip);
}

// the per-slice FA graph (llama_fn_kvrows_build_slices) against the one FA with the multi-sequence positional mask
static void test_fa_slices(int64_t n_kv, int n_seq, uint32_t n_tps, bool decoys, uint32_t seed) {
    const int64_t D = 256, n_head = 24, n_head_kv = 4;
    const float   scale = 1.0f/16.0f;

    llama_kv_cells cells;
    std::vector<llama_pos> next;
    make_cells(cells, (uint32_t) n_kv, (uint32_t) (n_kv*3/4), n_seq, 6, 17, next, seed);
    std::vector<llama_seq_id> seqs;
    std::vector<llama_pos>    starts;
    for (int s = 0; s < n_seq; ++s) {
        seqs.push_back(s);
        starts.push_back(next[s]);
    }
    ub_t u;
    make_grouped(u, seqs, n_tps, starts);
    // the ubatch's cells: the last n_seq*n_tps cells of the cache
    uint32_t j = (uint32_t) n_kv - (uint32_t) (n_seq*n_tps);
    for (int s = 0; s < n_seq; ++s) {
        for (uint32_t t = 0; t < n_tps; ++t, ++j) {
            if (!cells.is_empty(j)) {
                cells.rm(j);
            }
            cells.pos_set(j, starts[s] + (llama_pos) t);
            cells.seq_add(j, s);
        }
    }
    std::vector<int32_t> kv_pos_h, q_pos_h;
    make_pos_vectors(cells, u.ub, n_kv, kv_pos_h, q_pos_h);
    const int64_t n_tok = (int64_t) u.ub.n_tokens;
    const int rows = n_seq > 1 ? 2 : 1;

    const uint32_t n_rows = llama_fn_kvrows_n_rows(cells, u.ub, (uint32_t) n_seq, (uint32_t) n_kv + 4096);
    TCHECK(n_rows > 0, "fa n_kv %lld n_seq %d: no list length", (long long) n_kv, n_seq);
    std::vector<int32_t> lists((size_t) n_rows*n_seq);
    TCHECK(llama_fn_kvrows_fill_lists(kv_pos_h.data(), n_kv, rows, q_pos_h.data(), n_tok, (uint32_t) n_seq, n_rows, lists.data()),
           "fa: fill failed");
    if (decoys) {
        // a foreign cell in each list's padding: the mask must still hide it
        for (int s = 0; s < n_seq; ++s) {
            int32_t * l = lists.data() + (size_t) s*n_rows;
            for (uint32_t r = 0; r < n_rows; ++r) {
                if (l[r] < 0) {
                    for (uint32_t c = 0; c < (uint32_t) n_kv; ++c) {
                        if (!cells.is_empty(c) && !cells.seq_has(c, s)) {
                            l[r] = (int32_t) c;
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
    std::vector<int32_t> qsplit((size_t) 2*n_tok);
    if (n_seq > 1) {
        llama_fn_kvrows_split_q_pos(q_pos_h.data(), n_tok, (uint32_t) n_seq, qsplit.data());
    }

    ggml_context * ctx = new_ctx(256);
    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ggml_tensor * q0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_head, n_tok);
    ggml_tensor * k0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_head_kv, n_kv);
    ggml_tensor * v0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_head_kv, n_kv);
    for (int64_t i = 0; i < ggml_nelements(q0); ++i) {
        ((float *) q0->data)[i] = dist(rng);
    }
    for (int64_t i = 0; i < ggml_nelements(k0); ++i) {
        ((ggml_fp16_t *) k0->data)[i] = ggml_fp32_to_fp16(dist(rng));
        ((ggml_fp16_t *) v0->data)[i] = ggml_fp32_to_fp16(dist(rng));
    }
    ggml_tensor * kvp = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, rows);
    ggml_tensor * qp  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_tok, rows);
    memcpy(kvp->data, kv_pos_h.data(), ggml_nbytes(kvp));
    memcpy(qp->data,  q_pos_h.data(),  ggml_nbytes(qp));

    std::vector<ggml_tensor *> l_t;
    std::vector<ggml_tensor *> qp_t;
    for (int s = 0; s < n_seq; ++s) {
        ggml_tensor * l = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
        memcpy(l->data, lists.data() + (size_t) s*n_rows, n_rows*sizeof(int32_t));
        l_t.push_back(l);
        if (n_seq > 1) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_tps, 2);
            memcpy(t->data, qsplit.data() + (size_t) s*2*n_tps, ggml_nbytes(t));
            qp_t.push_back(t);
        }
    }

    // as build_attn_mha: permuted views
    ggml_tensor * q = ggml_permute(ctx, q0, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx, k0, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx, v0, 0, 2, 1, 3);

    ggml_tensor * ref = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_pos(ref, kvp, qp);

    const llama_fn_kvrows_build_fa_t build_fa = [&](ggml_tensor * qf, ggml_tensor * kf, ggml_tensor * vf, ggml_tensor * mf,
                                                    ggml_tensor * gtab, ggml_tensor * qposf) {
        (void) gtab;
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, mf, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_pos(fa, kvp, qposf);
        return fa;
    };
    ggml_tensor * cand = llama_fn_kvrows_build_slices(ctx, q, k, v, nullptr, qp, qp_t, l_t, (uint32_t) n_seq, build_fa);

    TCHECK(cand->ne[0] == D && cand->ne[1] == n_head && cand->ne[2] == (int64_t) n_tps && cand->ne[3] == n_seq,
           "fa: slices output shape %lld %lld %lld %lld", (long long) cand->ne[0], (long long) cand->ne[1],
           (long long) cand->ne[2], (long long) cand->ne[3]);
    TCHECK(ggml_nbytes(cand) == ggml_nbytes(ref), "fa: output sizes differ");
    // every slice FA carries its list in src[9]
    int n_list_src = 0;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, cand);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_FLASH_ATTN_EXT && t->src[9] != nullptr) {
            ++n_list_src;
        }
    }
    TCHECK(n_list_src == n_seq, "fa: %d FAs carry a list, want %d", n_list_src, n_seq);

    TCHECK(ggml_graph_compute_with_ctx(ctx, gf, 4) == GGML_STATUS_SUCCESS, "fa: compute failed");
    // the CPU FA splits the KV walk of a one-query FA over 512+ cells across threads (F32 partials instead of the row
    // path's F16 V accumulator: ~1e-3 apart), so a one-token slice over 512+ cells is compared with a tolerance (a wrong
    // query row, q_pos row or concat order is off by ~0.1); every other case, n_kv 500 included, bit for bit
    const bool exact = !(n_tps == 1 && n_seq > 1 && n_kv >= 512);
    bool same = ggml_nbytes(cand) == ggml_nbytes(ref) && ggml_is_contiguous(cand) && ggml_is_contiguous(ref);
    double max_diff = 0.0;
    if (same) {
        for (int64_t i = 0; i < ggml_nelements(ref); ++i) {
            max_diff = std::max(max_diff, (double) std::fabs(((const float *) cand->data)[i] - ((const float *) ref->data)[i]));
        }
    }
    if (same && exact) {
        same = memcmp(cand->data, ref->data, ggml_nbytes(ref)) == 0;
    } else if (same) {
        same = max_diff <= 2e-3;
    }
    bool finite = true;
    for (int64_t i = 0; i < ggml_nelements(ref); ++i) {
        finite = finite && std::isfinite(((const float *) ref->data)[i]);
    }
    TCHECK(finite, "fa: the reference has non-finite values");
    TCHECK(same, "fa n_kv %lld n_seq %d n_tps %u decoys %d: the slices differ from the one FA (max diff %g)", (long long) n_kv,
           n_seq, n_tps, (int) decoys, max_diff);

    ggml_free(ctx);
}

//
// part B
//

static void test_width() {
    struct row { uint32_t n_all, n_seq, n_top, g, want; };
    const row rows[] = {
        { 8300,  300, 512,  512,  512 },   // select-all width
        { 8300, 1000, 512,  512, 2048 },   // 1024 is below the stable 2048
        { 8300, 3000, 512,  512, 3072 },
        { 8300, 8000, 512,  512, 8192 },
        { 8300, 8299, 512,  512,    0 },   // not narrower
        { 1500,  300, 512,  512,    0 },   // full < 2048
        { 2047,  100, 512,  512,  512 },
        { 8300,  300, 500,  512, 2048 },   // 512 != n_top 500
        { 8300,  300, 512,   64,  512 },
        { 8300,  600, 512,   64, 2048 },
        { 8300,  300, 512, 4096, 4096 },
        { 8300, 8300, 512,  512,    0 },   // one sequence
        { 8300,  300,   0,  512,    0 },
        { 8300,    0, 512,  512,  512 },   // no pool yet
        { 40000, 2500, 512, 512, 2560 },
    };
    for (const row & r : rows) {
        const uint32_t w = llama_fn_qsa_seqpools_width(r.n_all, r.n_seq, r.n_top, r.g);
        TCHECK(w == r.want, "width(%u, %u, top %u, g %u) = %u, want %u", r.n_all, r.n_seq, r.n_top, r.g, w, r.want);
    }
    TCHECK(llama_fn_qsa_kpool_pad(511, 512) == 512 && llama_fn_qsa_kpool_pad(512, 512) == 1024 &&
           llama_fn_qsa_kpool_pad(0, 64) == 64 && llama_fn_qsa_kpool_pad(63, 64) == 64 && llama_fn_qsa_kpool_pad(64, 64) == 128,
           "kpool_pad");

    // invariants over a grid
    std::mt19937 rng(5);
    int n_bad = 0;
    for (int it = 0; it < 20000; ++it) {
        const uint32_t g     = 64u << (rng() % 7);
        const uint32_t n_all = 1 + rng() % 70000;
        const uint32_t n_seq = rng() % (n_all + 1);
        const uint32_t n_top = 1 + rng() % 4096;
        const uint32_t w     = llama_fn_qsa_seqpools_width(n_all, n_seq, n_top, g);
        if (w == 0) {
            continue;
        }
        const uint32_t full = llama_fn_qsa_kpool_pad(n_all, g);
        const bool ok = full >= 2048 && w < full && w >= n_top && w >= n_seq + 1 && (w >= 2048 || w == n_top) &&
            std::min(w, n_top) == std::min(full, n_top) && w % g == 0;
        n_bad += ok ? 0 : 1;
    }
    TCHECK(n_bad == 0, "width invariants: %d of 20000 cases fail", n_bad);
}

static void test_seqpools_check() {
    const uint32_t kpool = 4;
    const uint32_t n_full = 64, n_lst = 16;
    const std::vector<uint32_t> N = { 5, 7, 3 };
    const std::vector<uint32_t> gs = { 0, 5, 12 };
    const int32_t dummy = 900, sentinel = 1000;
    const uint32_t n_tok = 3;
    const float NINF = -INFINITY;

    std::vector<int32_t> f_pcell(n_full, dummy), f_pidx(n_full*kpool, sentinel);
    for (uint32_t s = 0; s < 3; ++s) {
        for (uint32_t i = 0; i < N[s]; ++i) {
            const uint32_t ip = gs[s] + i;
            f_pcell[ip] = (int32_t) (100*s + 4*i + 3);
            for (uint32_t k = 0; k < kpool; ++k) {
                f_pidx[ip*kpool + k] = (int32_t) (100*s + 4*i + k);
            }
        }
    }
    // the tokens are of sequence 1 and see 4, 5, 7 of its pools
    const uint32_t nv[3] = { 4, 5, 7 };
    std::vector<float> f_mask(n_full*n_tok, NINF), l_mask(n_lst*n_tok, NINF);
    std::vector<int32_t> l_pcell(n_lst, dummy), l_pidx(n_lst*kpool, sentinel);
    for (uint32_t i = 0; i < N[1]; ++i) {
        l_pcell[i] = f_pcell[gs[1] + i];
        for (uint32_t k = 0; k < kpool; ++k) {
            l_pidx[i*kpool + k] = f_pidx[(gs[1] + i)*kpool + k];
        }
    }
    for (uint32_t t = 0; t < n_tok; ++t) {
        for (uint32_t i = 0; i < nv[t]; ++i) {
            f_mask[t*n_full + gs[1] + i] = 0.0f;
            l_mask[t*n_lst + i]          = 0.0f;
        }
    }
    auto run = [&]() {
        llama_fn_qsa_lists lst;
        lst.pcell = l_pcell.data();
        lst.pidx  = l_pidx.data();
        lst.mask  = l_mask.data();
        lst.n_pool = n_lst;
        llama_fn_qsa_lists full;
        full.pcell = f_pcell.data();
        full.pidx  = f_pidx.data();
        full.mask  = f_mask.data();
        full.n_pool = n_full;
        return llama_fn_qsa_seqpools_check(lst, N[1], full, gs[1], kpool, n_tok, dummy, sentinel);
    };
    const std::string ok = run();
    TCHECK(ok.empty(), "a correct pair must pass: %s", ok.c_str());

    std::swap(l_pcell[1], l_pcell[2]);
    TCHECK(!run().empty(), "shifted order must fail");
    std::swap(l_pcell[1], l_pcell[2]);

    std::swap(l_pidx[0], l_pidx[1]);
    TCHECK(!run().empty(), "swapped members must fail");
    std::swap(l_pidx[0], l_pidx[1]);

    f_mask[1*n_full + 0] = 0.0f;   // a foreign finite entry (sequence 0's first pool) for token 1
    TCHECK(!run().empty(), "a foreign finite mask entry must fail");
    f_mask[1*n_full + 0] = NINF;

    l_pcell[N[1]] = 5;
    TCHECK(!run().empty(), "a wrong pad cell must fail");
    l_pcell[N[1]] = dummy;

    l_pidx[N[1]*kpool + 2] = 3;
    TCHECK(!run().empty(), "a wrong pad member must fail");
    l_pidx[N[1]*kpool + 2] = sentinel;

    l_mask[2*n_lst + 10] = 0.0f;
    TCHECK(!run().empty(), "a finite mask entry past the list must fail");
    l_mask[2*n_lst + 10] = NINF;

    l_mask[0*n_lst + 4] = 0.0f;   // token 0 sees one pool too many
    TCHECK(!run().empty(), "a visibility difference must fail");
    l_mask[0*n_lst + 4] = NINF;

    TCHECK(run().empty(), "restored pair must pass again");
}

static void test_sharepfx() {
    auto shape = [](const std::vector<int32_t> & cnt, size_t split = 0) {
        llama_fn_qsa_pfx st;
        size_t j = 0;
        // in two chunks when split > 0 (the layout's append path)
        for (; j < (split ? split : cnt.size()); ++j) {
            llama_fn_qsa_pfx_step(st, j, cnt[j]);
        }
        for (; j < cnt.size(); ++j) {
            llama_fn_qsa_pfx_step(st, j, cnt[j]);
        }
        return st;
    };
    {
        const llama_fn_qsa_pfx a = shape({ 2, 2, 2, 1, 1 });
        TCHECK(a.shared && !a.irr && a.pfx_n == 3 && a.cmin == 2 && a.cmax == 2, "aligned prefix shape");
        const llama_fn_qsa_pfx a2 = shape({ 2, 2, 2, 1, 1 }, 2);
        TCHECK(a2.pfx_n == a.pfx_n && a2.irr == a.irr && a2.cmin == a.cmin && a2.cmax == a.cmax, "chunked scan equals one pass");
        const llama_fn_qsa_pfx b = shape({ 2, 2, 2, 1 });
        const llama_fn_qsa_pfx c = shape({ 1, 1 });
        const llama_fn_qsa_pfx s3[3] = { a, b, c };
        const int32_t pm[3] = { 0, 0, 5 };
        TCHECK(llama_fn_qsa_sharepfx_safe(s3, pm, 3), "A, B share an aligned prefix, C own cells: safe");
        const int32_t pm2[3] = { 0, 1, 5 };
        TCHECK(!llama_fn_qsa_sharepfx_safe(s3, pm2, 3), "different first positions: unsafe");
    }
    {
        const llama_fn_qsa_pfx a = shape({ 2, 2, 2, 1 });
        const llama_fn_qsa_pfx b = shape({ 2, 2, 1 });
        const llama_fn_qsa_pfx s2[2] = { a, b };
        const int32_t pm[2] = { 0, 0 };
        TCHECK(!llama_fn_qsa_sharepfx_safe(s2, pm, 2), "prefixes of different length: unsafe");
    }
    {
        const llama_fn_qsa_pfx a = shape({ 2, 1, 2 });
        TCHECK(a.irr, "a shared cell after an own cell: irregular");
        const llama_fn_qsa_pfx b = shape({ 2, 1, 1 });
        const llama_fn_qsa_pfx s2[2] = { a, b };
        const int32_t pm[2] = { 0, 0 };
        TCHECK(!llama_fn_qsa_sharepfx_safe(s2, pm, 2), "a shared cell after the prefix: unsafe");
        const llama_fn_qsa_pfx c = shape({ 1, 2 });
        TCHECK(c.irr, "own first cell then a shared one: irregular");
    }
    {
        // three sequences with a 2-way share: counts 2 but |S| = 3
        const llama_fn_qsa_pfx a = shape({ 2, 2, 1 });
        const llama_fn_qsa_pfx b = shape({ 2, 2, 1 });
        const llama_fn_qsa_pfx c = shape({ 2, 2 });
        const llama_fn_qsa_pfx s3[3] = { a, b, c };
        const int32_t pm[3] = { 0, 0, 0 };
        TCHECK(!llama_fn_qsa_sharepfx_safe(s3, pm, 3), "three sharing sequences with 2-way cells: unsafe");
        // a 3-way share
        const llama_fn_qsa_pfx d = shape({ 3, 3, 1 });
        const llama_fn_qsa_pfx e = shape({ 3, 3 });
        const llama_fn_qsa_pfx f = shape({ 3, 3, 1, 1 });
        const llama_fn_qsa_pfx t3[3] = { d, e, f };
        TCHECK(llama_fn_qsa_sharepfx_safe(t3, pm, 3), "a 3-way aligned prefix: safe");
        // mixed counts in one run
        const llama_fn_qsa_pfx g = shape({ 3, 2, 1 });
        TCHECK(g.cmin == 2 && g.cmax == 3, "mixed counts");
        const llama_fn_qsa_pfx u3[3] = { g, e, f };
        TCHECK(!llama_fn_qsa_sharepfx_safe(u3, pm, 3), "mixed counts: unsafe");
    }
    {
        const llama_fn_qsa_pfx a = shape({ 1, 1 });
        const llama_fn_qsa_pfx s1[1] = { a };
        const int32_t pm[1] = { 0 };
        TCHECK(!llama_fn_qsa_sharepfx_safe(s1, pm, 1), "no sharing at all: not a shared prefix");
    }

    // rep -> members
    const uint32_t kpool = 4;
    std::vector<int32_t> pcell = { 10, 11, 10, 12 };
    std::vector<int32_t> pidx  = { 7, 8, 9, 10,  1, 2, 3, 11,  7, 8, 9, 10,  4, 5, 6, 12 };
    TCHECK(llama_fn_qsa_rep_members_check(pcell.data(), pidx.data(), 4, kpool).empty(), "same rep, same members: ok");
    pidx[8] = 6;
    TCHECK(!llama_fn_qsa_rep_members_check(pcell.data(), pidx.data(), 4, kpool).empty(), "same rep, other members: fail");
}

int main() {
    test_cfg_default();
    test_slices();
    test_n_rows_and_plan();
    test_fill_lists();
    test_split_q_pos();
    for (int64_t n_kv : { 500, 1000, 2048 }) {
        for (int n_seq : { 1, 2, 3, 4 }) {
            for (uint32_t n_tps : { 1u, 3u, 8u }) {
                test_fa_slices(n_kv, n_seq, n_tps, false, (uint32_t) (n_kv + 31*n_seq + n_tps));
            }
        }
    }
    test_fa_slices(2048, 4, 3, true, 77);
    test_width();
    test_seqpools_check();
    test_sharepfx();

    printf("test-fn-l22-kvrows: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
