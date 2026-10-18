// [TAG_FN_L22_IMGPASS] [TAG_FN_L22_IMGPASS_MASK] round 22 P1: CPU unit test of src/llama-fn-l22-img.h, the code that
// llama_kv_cache runs for a merged image turn (plan P1 items 3 and 5):
//  - the PLE predecessors of a merged batch (text and image rows in one embedding batch, rows named by tok_aux) cut into
//    ubatches at any width, ubatch edges inside images included, equal the per-chunk ubatches of today (text as token
//    ubatches, each image as one embedding ubatch); without the edge rule an edge inside an image differs
//  - keyed positional vectors (pos * 4096 + raster rank): (1) the keyed visibility equals the explicit 2-D causal mask
//    (llama_kv_cache::set_input_kq_mask's cell walk rule); (2) ggml_qsa_mask with keys equals the explicit selection;
//    (3) raw positions do not (keys matter); (4) flash attention with the explicit mask equals flash attention with the
//    keyed vectors; (5) the fallback cases (cell order not raster order, an image > 4096 cells, positions >= 2^19, several
//    sequences) refuse; (6) a text-only embedding ubatch keys as pos * 4096
// CPU backend only, no model. Exit code 0 iff every check passes.

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-fn-l22-img.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

static const llama_token MEDIA = 248056; // the image token (ple_image_token_id in the owner's file)

// one row of a sequence: a text token or an image row, with its 4 M-RoPE positions (t, y, x, z)
struct row_t {
    bool        img;
    llama_token tok;
    llama_pos   p[4];
};

// a sequence as the server builds it: text rows at consecutive positions, an image of ny x nx rows at t = next with
// y = t + row and x = t + col, after which the sequence advances by max(ny, nx) (mtmd's n_pos)
struct stream_t {
    std::vector<row_t> rows;
    llama_pos          next = 0;

    void text(std::initializer_list<llama_token> toks) {
        for (const llama_token t : toks) {
            rows.push_back({ false, t, { next, next, next, 0 } });
            ++next;
        }
    }
    void image(int ny, int nx) {
        const llama_pos t = next;
        for (int r = 0; r < ny; ++r) {
            for (int c = 0; c < nx; ++c) {
                rows.push_back({ true, MEDIA, { t, t + r, t + c, 0 } });
            }
        }
        next += std::max(ny, nx);
    }
};

// a llama_ubatch over rows [i0, i0 + n) of one sequence and the storage it points into
struct ub_t {
    std::vector<llama_token>    token;
    std::vector<llama_token>    tok_aux;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   sid;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id>   unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         out;
    llama_ubatch                ub = {};
};

// as_tokens: a token ubatch (the allocator broadcasts the position to every section); else an embedding ubatch with the
// rows' 4 positions, and with_aux names each row (text: its token, image: MEDIA)
static void make_ub(ub_t & u, const std::vector<row_t> & rows, size_t i0, size_t n, bool as_tokens, bool with_aux,
        llama_seq_id s = 0) {
    u.token.assign(as_tokens ? n : 0, 0);
    u.tok_aux.assign(with_aux ? n : 0, LLAMA_TOKEN_NULL);
    u.pos.assign(4*n, 0);
    u.n_seq_id.assign(n, 1);
    u.sid.assign(n, s);
    u.seq_id.resize(n);
    u.unq.assign(1, s);
    u.seq_idx.assign(LLAMA_MAX_SEQ, -1);
    u.seq_idx[s] = 0;
    u.out.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const row_t & r = rows[i0 + i];
        for (int j = 0; j < 4; ++j) {
            u.pos[j*n + i] = as_tokens ? r.p[0] : r.p[j];
        }
        if (as_tokens) {
            u.token[i] = r.tok;
        }
        if (with_aux) {
            u.tok_aux[i] = r.img ? MEDIA : r.tok;
        }
        u.seq_id[i] = &u.sid[i];
    }
    u.ub = {};
    u.ub.b_equal_seqs = 1;
    u.ub.n_tokens     = (uint32_t) n;
    u.ub.n_seq_tokens = (uint32_t) n;
    u.ub.n_seqs       = 1;
    u.ub.n_seqs_unq   = 1;
    u.ub.n_pos        = 4;
    u.ub.token        = as_tokens ? u.token.data() : nullptr;
    u.ub.embd         = nullptr;
    u.ub.pos          = u.pos.data();
    u.ub.n_seq_id     = u.n_seq_id.data();
    u.ub.seq_id       = u.seq_id.data();
    u.ub.seq_id_unq   = u.unq.data();
    u.ub.seq_idx      = u.seq_idx.data();
    u.ub.output       = u.out.data();
    u.ub.decision_order = nullptr;
    u.ub.embd_state   = nullptr;
    u.ub.tok_aux      = with_aux ? u.tok_aux.data() : nullptr;
}

// what llama_kv_cache::apply_ubatch stores for these rows: the position, the 2-D ext and ext.tok (token, else tok_aux,
// else the image token), cells taken from `cell_of` (one per row)
static void apply(llama_kv_cells & cells, const llama_ubatch & ub, const std::vector<uint32_t> & cell_of, llama_seq_id s = 0) {
    const uint32_t n = ub.n_tokens;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t idx = cell_of[i];
        cells.pos_set(idx, ub.pos[i]);
        llama_kv_cell_ext ext;
        ext.y   = ub.pos[i + n];
        ext.x   = ub.pos[i + 2*n];
        ext.tok = ub.token ? ub.token[i] : (ub.tok_aux && ub.tok_aux[i] != LLAMA_TOKEN_NULL ? ub.tok_aux[i] : MEDIA);
        cells.ext_set(idx, ext);
        cells.seq_add(idx, s);
    }
}

static std::vector<uint32_t> next_cells(uint32_t & next, uint32_t n) {
    std::vector<uint32_t> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] = next++;
    }
    return v;
}

//
// predecessors
//

static void test_prev() {
    stream_t st;
    st.text({ 100, 101, 102, 103, 104 });
    st.image(3, 4);
    st.text({ 200, 201, 202, 203 });
    st.image(3, 3);
    st.text({ 300, 301, 302 });
    st.image(4, 3);
    st.text({ 400 });
    const auto & rows = st.rows;
    const size_t n_rows = rows.size();

    // n_prev 3: every image has max(ny, nx) >= 3, so a text row right after an image reads image cells through the
    // position gap in per-chunk mode, as a merged batch reads the image rows before it
    const uint32_t n_prev = 3;

    auto run_prev = [&](const llama_kv_cells & cells, const llama_ubatch & ub, llama_token media, std::vector<llama_token> & res) {
        llama_fn_l22_prev_tokens(ub, n_prev, media, [&](llama_seq_id) -> const llama_kv_cells & { return cells; }, res);
    };

    // reference: today's per-chunk ubatches
    std::vector<llama_token> ref((size_t) n_rows*n_prev, LLAMA_TOKEN_NULL);
    std::vector<llama_token> ref_self(n_rows);
    {
        llama_kv_cells cells;
        cells.resize(512);
        uint32_t next = 0;
        size_t i0 = 0;
        while (i0 < n_rows) {
            size_t i1 = i0;
            while (i1 < n_rows && rows[i1].img == rows[i0].img) {
                ++i1;
            }
            ub_t u;
            make_ub(u, rows, i0, i1 - i0, !rows[i0].img, false);
            apply(cells, u.ub, next_cells(next, (uint32_t) (i1 - i0)));
            std::vector<llama_token> res;
            run_prev(cells, u.ub, MEDIA, res);
            std::copy(res.begin(), res.end(), ref.begin() + i0*n_prev);
            for (size_t i = i0; i < i1; ++i) {
                ref_self[i] = rows[i].img ? MEDIA : rows[i].tok;
            }
            i0 = i1;
        }
    }

    // merged: one embedding batch with tok_aux, cut into ubatches of width w (edges anywhere, inside images too)
    auto run_merged = [&](size_t w, llama_token media, int & n_diff, int & n_diff_self) {
        llama_kv_cells cells;
        cells.resize(512);
        uint32_t next = 0;
        n_diff = 0;
        n_diff_self = 0;
        for (size_t i0 = 0; i0 < n_rows; i0 += w) {
            const size_t n = std::min(w, n_rows - i0);
            ub_t u;
            make_ub(u, rows, i0, n, false, true);
            apply(cells, u.ub, next_cells(next, (uint32_t) n));
            std::vector<llama_token> res;
            run_prev(cells, u.ub, media, res);
            for (size_t i = 0; i < n; ++i) {
                for (uint32_t j = 0; j < n_prev; ++j) {
                    n_diff += res[i*n_prev + j] != ref[(i0 + i)*n_prev + j] ? 1 : 0;
                }
                // the row's own id in the PLE hash (qwen4exp tok_of): tok_aux
                n_diff_self += u.ub.tok_aux[i] != ref_self[i0 + i] ? 1 : 0;
            }
        }
    };

    for (const size_t w : { (size_t) 1, (size_t) 2, (size_t) 3, (size_t) 4, (size_t) 5, (size_t) 6, (size_t) 7, (size_t) 8,
                            (size_t) 9, (size_t) 11, (size_t) 13, (size_t) 17, (size_t) 23, (size_t) 64 }) {
        int n_diff = 0;
        int n_self = 0;
        run_merged(w, MEDIA, n_diff, n_self);
        TCHECK(n_diff == 0, "merged ubatches of %zu rows: %d predecessor ids differ from the per-chunk ubatches", w, n_diff);
        TCHECK(n_self == 0, "merged ubatches of %zu rows: %d row ids differ", w, n_self);
    }

    // the rule matters: width 7 puts an edge after the 2nd row of the first image
    {
        int n_diff = 0;
        int n_self = 0;
        run_merged(7, LLAMA_TOKEN_NULL, n_diff, n_self);
        TCHECK(n_diff > 0, "without the edge rule an edge inside an image should change some predecessor ids");
    }

    // a token ubatch or an embedding ubatch without tok_aux ignores the media argument (today's code path)
    {
        llama_kv_cells cells;
        cells.resize(64);
        uint32_t next = 0;
        ub_t a;
        make_ub(a, rows, 0, 7, false, false); // text A + 2 image rows as plain embeddings (no tok_aux)
        apply(cells, a.ub, next_cells(next, 7));
        ub_t b;
        make_ub(b, rows, 7, 4, false, false);
        apply(cells, b.ub, next_cells(next, 4));
        std::vector<llama_token> r1;
        std::vector<llama_token> r2;
        run_prev(cells, b.ub, MEDIA, r1);
        run_prev(cells, b.ub, LLAMA_TOKEN_NULL, r2);
        TCHECK(r1 == r2, "no tok_aux: the edge rule must not act");
    }
}

//
// keyed masks
//

// the explicit mask's rule (llama_kv_cache::set_input_kq_mask, causal, one stream): a cell is visible from query i iff it
// holds the sequence, its position is <= the query's, and at an equal position its (y, x) is not after the query's
static bool explicit_visible(const llama_kv_cells & cells, uint32_t j, llama_seq_id s, llama_pos p1, llama_pos y1, llama_pos x1) {
    if (cells.is_empty(j) || !cells.seq_has(j, s)) {
        return false;
    }
    const llama_pos p0 = cells.pos_get(j);
    if (p0 > p1) {
        return false;
    }
    if (p0 == p1 && cells.ext_get(j).is_2d_gt(x1, y1)) {
        return false;
    }
    return true;
}

struct key_case {
    llama_kv_cells       cells;
    ub_t                 cur;    // the query ubatch
    int64_t              n_kv = 0;
};

// seq 0: text (3), image 3x3, text (2), image 2x3, text (1); a previous ubatch stored the text and the first 4 cells of
// the first image; the query ubatch holds the rest. Seq 0 takes the even cells, seq 1 some odd cells, the rest are empty.
// reverse_img2: the second image's cells are taken in reverse raster order (a cell order the keys refuse)
static void build_key_case(key_case & kc, bool reverse_img2) {
    stream_t st;
    st.text({ 10, 11, 12 });
    st.image(3, 3);
    st.text({ 20, 21 });
    st.image(2, 3);
    st.text({ 30 });
    const auto & rows = st.rows;
    const size_t n_rows = rows.size();
    const size_t n_prev = 3 + 4;

    kc.n_kv = 64;
    kc.cells.resize((uint32_t) kc.n_kv);

    std::vector<uint32_t> cell_of(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        cell_of[i] = (uint32_t) (2*i);
    }
    if (reverse_img2) {
        const size_t a = 3 + 9 + 2; // the second image's first row
        std::reverse(cell_of.begin() + a, cell_of.begin() + a + 6);
    }

    // seq 1: cells 1, 5, 9, 13 at positions 0..3 (they must stay invisible)
    for (uint32_t k = 0; k < 4; ++k) {
        const uint32_t j = 1 + 4*k;
        kc.cells.pos_set(j, (llama_pos) k);
        llama_kv_cell_ext ext;
        ext.y = (llama_pos) k;
        ext.x = (llama_pos) k;
        ext.tok = 7;
        kc.cells.ext_set(j, ext);
        kc.cells.seq_add(j, 1);
    }

    ub_t prev;
    make_ub(prev, rows, 0, n_prev, false, true);
    apply(kc.cells, prev.ub, std::vector<uint32_t>(cell_of.begin(), cell_of.begin() + n_prev));

    make_ub(kc.cur, rows, n_prev, n_rows - n_prev, false, true);
    apply(kc.cells, kc.cur.ub, std::vector<uint32_t>(cell_of.begin() + n_prev, cell_of.end()));
}

static ggml_context * new_ctx(size_t mib) {
    ggml_init_params ip = { mib*1024*1024, nullptr, false };
    return ggml_init(ip);
}

static void test_keys() {
    key_case kc;
    build_key_case(kc, false);
    const llama_ubatch & ub = kc.cur.ub;
    const uint32_t n_q  = ub.n_tokens;
    const int64_t  n_kv = kc.n_kv;

    std::vector<llama_fn_l22_imgkey_group> groups;
    const int r = llama_fn_l22_imgkeys_plan(kc.cells, ub, groups);
    TCHECK(r == LLAMA_FN_L22_IMGKEY_OK, "plan: %s", llama_fn_l22_imgkey_res_str(r));
    TCHECK(groups.size() == 2, "plan: %zu groups, want 2 (the two images)", groups.size());

    std::vector<int32_t> kv_key(n_kv);
    std::vector<int32_t> q_key(n_q);
    llama_fn_l22_imgkeys_kv(kc.cells, 0, groups, kv_key.data(), n_kv);
    llama_fn_l22_imgkeys_q(ub, groups, q_key.data());

    std::vector<int32_t> kv_raw(n_kv);
    for (int64_t j = 0; j < n_kv; ++j) {
        kv_raw[j] = !kc.cells.is_empty((uint32_t) j) && kc.cells.seq_has((uint32_t) j, 0) ? kc.cells.pos_get((uint32_t) j) : -1;
    }
    std::vector<int32_t> q_raw(ub.pos, ub.pos + n_q);

    // (1) keyed visibility == explicit rule
    std::vector<uint8_t> vis((size_t) n_kv*n_q);
    int n_bad = 0;
    int n_tie = 0; // cells the 2-D rule hides at the query's own position
    for (uint32_t i = 0; i < n_q; ++i) {
        const llama_pos p1 = ub.pos[i];
        const llama_pos y1 = ub.pos[i + n_q];
        const llama_pos x1 = ub.pos[i + 2*n_q];
        for (int64_t j = 0; j < n_kv; ++j) {
            const bool e = explicit_visible(kc.cells, (uint32_t) j, 0, p1, y1, x1);
            vis[(size_t) i*n_kv + j] = e ? 1 : 0;
            const bool k = kv_key[j] >= 0 && kv_key[j] <= q_key[i];
            n_bad += e != k ? 1 : 0;
            n_tie += (!e && kv_raw[j] >= 0 && kv_raw[j] == p1) ? 1 : 0;
        }
    }
    TCHECK(n_bad == 0, "keys: %d of %lld visibilities differ from the explicit rule", n_bad, (long long) n_kv*n_q);
    TCHECK(n_tie > 0, "the case must hold same-position cells the 2-D rule hides");

    // (2) / (3) ggml_qsa_mask with every cell selected
    auto qsa = [&](const std::vector<int32_t> & kvp, const std::vector<int32_t> & qp, std::vector<ggml_fp16_t> & out) {
        ggml_context * ctx = new_ctx(16);
        ggml_tensor * t_kv  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
        ggml_tensor * t_q   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_q);
        ggml_tensor * t_sel = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_q);
        memcpy(t_kv->data, kvp.data(), n_kv*sizeof(int32_t));
        memcpy(t_q->data,  qp.data(),  n_q*sizeof(int32_t));
        for (uint32_t i = 0; i < n_q; ++i) {
            for (int64_t s = 0; s < n_kv; ++s) {
                ((int32_t *) t_sel->data)[i*n_kv + s] = (int32_t) s;
            }
        }
        ggml_tensor * m = ggml_qsa_mask(ctx, t_kv, t_q, t_sel, nullptr, 1);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, m);
        const bool ok = ggml_graph_compute_with_ctx(ctx, gf, 2) == GGML_STATUS_SUCCESS;
        out.assign((const ggml_fp16_t *) m->data, (const ggml_fp16_t *) m->data + n_kv*n_q);
        ggml_free(ctx);
        return ok;
    };
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> ref((size_t) n_kv*n_q);
    for (size_t k = 0; k < ref.size(); ++k) {
        ref[k] = vis[k] ? f16_zero : f16_ninf;
    }
    {
        std::vector<ggml_fp16_t> m;
        TCHECK(qsa(kv_key, q_key, m), "qsa_mask (keys) compute failed");
        TCHECK(m.size() == ref.size() && memcmp(m.data(), ref.data(), ref.size()*sizeof(ggml_fp16_t)) == 0,
                "qsa_mask with keys differs from the explicit selection");
    }
    {
        std::vector<ggml_fp16_t> m;
        TCHECK(qsa(kv_raw, q_raw, m), "qsa_mask (raw) compute failed");
        int n_diff = 0;
        for (size_t k = 0; k < ref.size() && k < m.size(); ++k) {
            n_diff += memcmp(&m[k], &ref[k], sizeof(ggml_fp16_t)) != 0 ? 1 : 0;
        }
        TCHECK(n_diff > 0, "qsa_mask with raw positions should differ from the explicit selection (keys must matter)");
    }

    // (4) flash attention: explicit mask vs keyed positional vectors
    {
        const int64_t D = 64;
        const int64_t H = 2;
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> U(-1.0f, 1.0f);
        ggml_context * ctx = new_ctx(64);
        ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, H, 1);
        ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H, 1);
        ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H, 1);
        ggml_tensor * mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_q);
        ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
        ggml_tensor * qp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_q);
        for (int64_t e = 0; e < ggml_nelements(q); ++e) {
            ((float *) q->data)[e] = U(rng);
        }
        for (int64_t e = 0; e < ggml_nelements(k); ++e) {
            ((ggml_fp16_t *) k->data)[e] = ggml_fp32_to_fp16(U(rng));
            ((ggml_fp16_t *) v->data)[e] = ggml_fp32_to_fp16(U(rng));
        }
        memcpy(mk->data, ref.data(), ref.size()*sizeof(ggml_fp16_t));
        memcpy(kp->data, kv_key.data(), n_kv*sizeof(int32_t));
        memcpy(qp->data, q_key.data(), n_q*sizeof(int32_t));
        const float scale = 1.0f/sqrtf((float) D);
        ggml_tensor * fa1 = ggml_flash_attn_ext(ctx, q, k, v, mk, scale, 0.0f, 0.0f);
        ggml_tensor * fa2 = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_pos(fa2, kp, qp);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, fa1);
        ggml_build_forward_expand(gf, fa2);
        TCHECK(ggml_graph_compute_with_ctx(ctx, gf, 2) == GGML_STATUS_SUCCESS, "flash attention compute failed");
        double max_diff = 0.0;
        bool finite = true;
        for (int64_t e = 0; e < ggml_nelements(fa1); ++e) {
            const float a = ((const float *) fa1->data)[e];
            const float b = ((const float *) fa2->data)[e];
            finite = finite && std::isfinite(a) && std::isfinite(b);
            max_diff = std::max(max_diff, (double) fabsf(a - b));
        }
        TCHECK(finite, "flash attention produced a non-finite value");
        TCHECK(max_diff <= 1e-6, "flash attention: explicit mask vs keyed vectors differ by %g", max_diff);
        printf("test-fn-l22-mask: flash attention explicit vs keys: max |diff| = %g\n", max_diff);
        ggml_free(ctx);
    }

    // (6) a text-only embedding ubatch: no group, keys pos * 4096
    {
        stream_t st;
        st.text({ 1, 2, 3, 4 });
        llama_kv_cells cells;
        cells.resize(16);
        uint32_t next = 0;
        ub_t u;
        make_ub(u, st.rows, 0, st.rows.size(), false, true);
        apply(cells, u.ub, next_cells(next, (uint32_t) st.rows.size()));
        std::vector<llama_fn_l22_imgkey_group> g;
        TCHECK(llama_fn_l22_imgkeys_plan(cells, u.ub, g) == LLAMA_FN_L22_IMGKEY_OK && g.empty(), "text-only plan");
        std::vector<int32_t> kk(16);
        std::vector<int32_t> qk(u.ub.n_tokens);
        llama_fn_l22_imgkeys_kv(cells, 0, g, kk.data(), 16);
        llama_fn_l22_imgkeys_q(u.ub, g, qk.data());
        bool ok = true;
        for (uint32_t j = 0; j < 16; ++j) {
            ok = ok && kk[j] == (j < 4 ? (int32_t) j*LLAMA_FN_L22_KEY_MUL : -1);
        }
        for (uint32_t i = 0; i < u.ub.n_tokens; ++i) {
            ok = ok && qk[i] == u.ub.pos[i]*LLAMA_FN_L22_KEY_MUL;
        }
        TCHECK(ok, "text-only keys must be pos * 4096 (and -1 for empty cells)");
    }
}

static void test_fallbacks() {
    // cell order not raster order
    {
        key_case kc;
        build_key_case(kc, true);
        std::vector<llama_fn_l22_imgkey_group> g;
        const int r = llama_fn_l22_imgkeys_plan(kc.cells, kc.cur.ub, g);
        TCHECK(r == LLAMA_FN_L22_IMGKEY_CELL_ORDER, "reversed image cells: %s", llama_fn_l22_imgkey_res_str(r));
    }
    // an image of more than 4096 cells
    {
        stream_t st;
        st.image(64, 65); // 4160 rows
        llama_kv_cells cells;
        cells.resize(4200);
        uint32_t next = 0;
        ub_t u;
        make_ub(u, st.rows, 0, st.rows.size(), false, true);
        apply(cells, u.ub, next_cells(next, (uint32_t) st.rows.size()));
        std::vector<llama_fn_l22_imgkey_group> g;
        const int r = llama_fn_l22_imgkeys_plan(cells, u.ub, g);
        TCHECK(r == LLAMA_FN_L22_IMGKEY_GROUP_SIZE, "4160-cell image: %s", llama_fn_l22_imgkey_res_str(r));
    }
    // positions at 2^19 and above
    {
        stream_t st;
        st.next = LLAMA_FN_L22_KEY_POS_END - 2;
        st.image(2, 2);
        llama_kv_cells cells;
        cells.resize(8);
        uint32_t next = 0;
        ub_t u;
        make_ub(u, st.rows, 0, st.rows.size(), false, true);
        apply(cells, u.ub, next_cells(next, (uint32_t) st.rows.size()));
        std::vector<llama_fn_l22_imgkey_group> g;
        const int r = llama_fn_l22_imgkeys_plan(cells, u.ub, g);
        TCHECK(r == LLAMA_FN_L22_IMGKEY_OK, "positions just below 2^19: %s", llama_fn_l22_imgkey_res_str(r));
        llama_kv_cells cells2;
        cells2.resize(8);
        next = 0;
        stream_t st2;
        st2.next = LLAMA_FN_L22_KEY_POS_END;
        st2.image(2, 2);
        ub_t u2;
        make_ub(u2, st2.rows, 0, st2.rows.size(), false, true);
        apply(cells2, u2.ub, next_cells(next, (uint32_t) st2.rows.size()));
        const int r2 = llama_fn_l22_imgkeys_plan(cells2, u2.ub, g);
        TCHECK(r2 == LLAMA_FN_L22_IMGKEY_POS_RANGE, "positions at 2^19: %s", llama_fn_l22_imgkey_res_str(r2));
    }
    // several sequences
    {
        key_case kc;
        build_key_case(kc, false);
        llama_ubatch ub = kc.cur.ub;
        std::vector<llama_seq_id> unq = { 0, 1 };
        ub.n_seqs_unq = 2;
        ub.seq_id_unq = unq.data();
        std::vector<llama_fn_l22_imgkey_group> g;
        const int r = llama_fn_l22_imgkeys_plan(kc.cells, ub, g);
        TCHECK(r == LLAMA_FN_L22_IMGKEY_MULTI_SEQ, "two sequences: %s", llama_fn_l22_imgkey_res_str(r));
    }
}

int main() {
    test_prev();
    test_keys();
    test_fallbacks();

    printf("test-fn-l22-mask: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
