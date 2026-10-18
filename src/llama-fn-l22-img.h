#pragma once

// [TAG_FN_L22_IMGPASS] [TAG_FN_L22_IMGPASS_MASK] round 22 P1, header only: src/llama-kv-cache.cpp and
// tests/test-fn-l22-mask.cpp run the same code.
//  - llama_fn_l22_prev_tokens: the PLE n-gram predecessors of a ubatch (llama_kv_cache::get_prev_tokens), with the
//    edge rule for a merged image turn whose ubatch starts inside an image
//  - llama_fn_l22_imgkeys_*: keyed positional mask vectors for an image ubatch (key = pos * 4096 + raster rank), so the
//    positional mask and ggml_qsa_mask apply the 2-D causal rule of the explicit mask

#include "llama-batch.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

// image cells of sequence s at temporal position t whose (y, x) comes before (y0, x0) in raster order
static inline int64_t llama_fn_l22_img_r0(const llama_kv_cells & cells, llama_seq_id s, llama_pos t, llama_pos y0, llama_pos x0,
        llama_token media) {
    int64_t r0 = 0;
    const auto & sp = cells.seq_pos_get(s);
    for (auto it = sp.lower_bound({ t, 0u }); it != sp.end() && it->first == t; ++it) {
        const llama_kv_cell_ext & e = cells.ext_get(it->second);
        if (e.tok == media && (e.y < y0 || (e.y == y0 && e.x < x0))) {
            ++r0;
        }
    }
    return r0;
}

// res[i*n + j]: the token n - j positions before row i (oldest first), LLAMA_TOKEN_NULL when missing.
// cells_of(seq_id) returns the llama_kv_cells holding the sequence. apply_ubatch() has stored this ubatch already.
// media: the token of an image row (LLAMA_TOKEN_NULL = no edge rule). The edge rule acts only on embedding ubatches that
// carry tok_aux, for a sequence whose first row in the ubatch is an image row: the r0 image cells raster-before it
// (stored by an earlier ubatch) come first, then the text before the image.
template <typename F>
static inline void llama_fn_l22_prev_tokens(const llama_ubatch & ubatch, uint32_t n, llama_token media, F && cells_of,
        std::vector<llama_token> & res) {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize((size_t) n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens share the same
    // temporal pos. An embd (multimodal) ubatch can repeat one position for a whole image, so positions do not encode
    // the token order; resolve its predecessors by ubatch order instead
    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;
    std::unordered_map<llama_seq_id, int64_t> seq_r0; // [TAG_FN_L22_IMGPASS]

    if (!ubatch.token) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
        if (ubatch.tok_aux && media != LLAMA_TOKEN_NULL && ubatch.n_pos >= 3) {
            for (const auto & sv : seq_idx) {
                const uint32_t i0 = sv.second[0];
                if (ubatch.tok_aux[i0] != media) {
                    continue;
                }
                const int64_t r0 = llama_fn_l22_img_r0(cells_of(sv.first), sv.first, ubatch.pos[i0], ubatch.pos[i0 + n_tokens],
                        ubatch.pos[i0 + 2*(size_t) n_tokens], media);
                if (r0 > 0) {
                    seq_r0[sv.first] = r0;
                }
            }
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // a token that belongs to more than one sequence has an ambiguous history; the n-gram architectures reject such
        // batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (!ubatch.token) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                if (k >= 0) {
                    p = ubatch.pos[v[k]];
                } else {
                    int64_t r0 = 0;
                    if (!seq_r0.empty()) {
                        const auto it = seq_r0.find(seq_id);
                        r0 = it != seq_r0.end() ? it->second : 0;
                    }
                    if (-k <= r0) {
                        res[(size_t) i*n + j] = media; // an image cell stored by an earlier ubatch
                        continue;
                    }
                    p = ubatch.pos[v[0]] + (llama_pos) (k + r0);
                }
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[(size_t) i*n + j] = cells_of(seq_id).seq_pos_tok_le(seq_id, p);
        }
    }
}

//
// keyed positional masks
//

static constexpr int32_t   LLAMA_FN_L22_KEY_MUL     = 4096;      // ranks per position
static constexpr llama_pos LLAMA_FN_L22_KEY_POS_END = 1 << 19;   // pos * 4096 + 4095 fits in I32 below this

// the cells of one image (one temporal position of the sequence) that has query rows in the ubatch, in raster order
struct llama_fn_l22_imgkey_group {
    llama_pos t = 0;
    std::vector<std::pair<std::pair<llama_pos, llama_pos>, uint32_t>> cells; // ((y, x), cell)
};

enum llama_fn_l22_imgkey_res {
    LLAMA_FN_L22_IMGKEY_OK = 0,
    LLAMA_FN_L22_IMGKEY_MULTI_SEQ,  // not one sequence, or no 2-D positions
    LLAMA_FN_L22_IMGKEY_POS_RANGE,  // a position >= 2^19
    LLAMA_FN_L22_IMGKEY_GROUP_SIZE, // an image of more than 4096 cells
    LLAMA_FN_L22_IMGKEY_DUP_YX,     // two cells of an image at the same (y, x)
    LLAMA_FN_L22_IMGKEY_CELL_ORDER, // raster order is not cell order (the in-order k-pool ranks by cell)
    LLAMA_FN_L22_IMGKEY_NO_CELL,    // a query row without its cell
};

static inline const char * llama_fn_l22_imgkey_res_str(int r) {
    switch (r) {
        case LLAMA_FN_L22_IMGKEY_OK:         return "ok";
        case LLAMA_FN_L22_IMGKEY_MULTI_SEQ:  return "not one sequence";
        case LLAMA_FN_L22_IMGKEY_POS_RANGE:  return "position >= 2^19";
        case LLAMA_FN_L22_IMGKEY_GROUP_SIZE: return "image > 4096 cells";
        case LLAMA_FN_L22_IMGKEY_DUP_YX:     return "duplicate (y, x)";
        case LLAMA_FN_L22_IMGKEY_CELL_ORDER: return "cell order is not raster order";
        case LLAMA_FN_L22_IMGKEY_NO_CELL:    return "query row without a cell";
    }
    return "?";
}

// the groups of the ubatch's query positions that hold more than one cell of the sequence; LLAMA_FN_L22_IMGKEY_OK when
// the keys reproduce the explicit mask, else the reason the ubatch must keep it
static inline int llama_fn_l22_imgkeys_plan(const llama_kv_cells & cells, const llama_ubatch & ub,
        std::vector<llama_fn_l22_imgkey_group> & groups) {
    groups.clear();
    if (ub.n_seqs_unq != 1 || ub.n_pos < 3 || ub.n_tokens == 0) {
        return LLAMA_FN_L22_IMGKEY_MULTI_SEQ;
    }
    const llama_seq_id s = ub.seq_id[0][0];
    if (s < 0 || s >= LLAMA_MAX_SEQ) {
        return LLAMA_FN_L22_IMGKEY_MULTI_SEQ;
    }
    const uint32_t n = ub.n_tokens;
    for (uint32_t i = 0; i < n; ++i) {
        if (ub.n_seq_id[i] != 1 || ub.seq_id[i][0] != s) {
            return LLAMA_FN_L22_IMGKEY_MULTI_SEQ;
        }
    }
    if (cells.seq_pos_max(s) >= LLAMA_FN_L22_KEY_POS_END) {
        return LLAMA_FN_L22_IMGKEY_POS_RANGE;
    }
    std::vector<llama_pos> ts(ub.pos, ub.pos + n);
    std::sort(ts.begin(), ts.end());
    ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
    if (ts.front() < 0 || ts.back() >= LLAMA_FN_L22_KEY_POS_END) {
        return LLAMA_FN_L22_IMGKEY_POS_RANGE;
    }
    const auto & sp = cells.seq_pos_get(s);
    for (const llama_pos t : ts) {
        auto it = sp.lower_bound({ t, 0u });
        auto it2 = it;
        size_t cnt = 0;
        for (; it2 != sp.end() && it2->first == t; ++it2) {
            ++cnt;
        }
        if (cnt <= 1) {
            continue; // a text position: rank 0
        }
        if (cnt > (size_t) LLAMA_FN_L22_KEY_MUL) {
            groups.clear();
            return LLAMA_FN_L22_IMGKEY_GROUP_SIZE;
        }
        llama_fn_l22_imgkey_group g;
        g.t = t;
        g.cells.reserve(cnt);
        for (; it != it2; ++it) {
            const llama_kv_cell_ext & e = cells.ext_get(it->second);
            g.cells.push_back({ { e.y, e.x }, it->second });
        }
        std::sort(g.cells.begin(), g.cells.end());
        for (size_t r = 1; r < g.cells.size(); ++r) {
            if (g.cells[r].first == g.cells[r - 1].first) {
                groups.clear();
                return LLAMA_FN_L22_IMGKEY_DUP_YX;
            }
            if (g.cells[r].second <= g.cells[r - 1].second) {
                groups.clear();
                return LLAMA_FN_L22_IMGKEY_CELL_ORDER;
            }
        }
        groups.push_back(std::move(g));
    }
    // every query row of a group finds its own cell
    for (uint32_t i = 0; i < n && !groups.empty(); ++i) {
        const auto g = std::lower_bound(groups.begin(), groups.end(), ub.pos[i],
                [](const llama_fn_l22_imgkey_group & a, llama_pos t) { return a.t < t; });
        if (g == groups.end() || g->t != ub.pos[i]) {
            continue;
        }
        const std::pair<llama_pos, llama_pos> yx = { ub.pos[i + n], ub.pos[i + 2*(size_t) n] };
        const auto c = std::lower_bound(g->cells.begin(), g->cells.end(), std::make_pair(yx, 0u));
        if (c == g->cells.end() || c->first != yx) {
            groups.clear();
            return LLAMA_FN_L22_IMGKEY_NO_CELL;
        }
    }
    return LLAMA_FN_L22_IMGKEY_OK;
}

// the KV side: -1 for an empty cell or one of another sequence, else pos * 4096 + the cell's raster rank in its group
static inline void llama_fn_l22_imgkeys_kv(const llama_kv_cells & cells, llama_seq_id s,
        const std::vector<llama_fn_l22_imgkey_group> & groups, int32_t * data, int64_t n_kv) {
    const int64_t n_cells = cells.size();
    for (int64_t j = 0; j < n_kv; ++j) {
        data[j] = (j < n_cells && !cells.is_empty(j) && cells.seq_has(j, s)) ? cells.pos_get(j)*LLAMA_FN_L22_KEY_MUL : -1;
    }
    for (const auto & g : groups) {
        for (size_t r = 0; r < g.cells.size(); ++r) {
            const uint32_t j = g.cells[r].second;
            if ((int64_t) j < n_kv) {
                data[j] = g.t*LLAMA_FN_L22_KEY_MUL + (int32_t) r;
            }
        }
    }
}

// the query side: pos * 4096 + the rank of the row's own cell in its group (0 outside the groups)
static inline void llama_fn_l22_imgkeys_q(const llama_ubatch & ub, const std::vector<llama_fn_l22_imgkey_group> & groups,
        int32_t * out) {
    const uint32_t n = ub.n_tokens;
    for (uint32_t i = 0; i < n; ++i) {
        const llama_pos t = ub.pos[i];
        int32_t key = t*LLAMA_FN_L22_KEY_MUL;
        const auto gi = std::lower_bound(groups.begin(), groups.end(), t,
                [](const llama_fn_l22_imgkey_group & a, llama_pos v) { return a.t < v; });
        if (gi != groups.end() && gi->t == t) {
            const std::pair<llama_pos, llama_pos> yx = { ub.pos[i + n], ub.pos[i + 2*(size_t) n] };
            const auto c = std::lower_bound(gi->cells.begin(), gi->cells.end(), std::make_pair(yx, 0u));
            key += (int32_t) (c - gi->cells.begin());
        }
        out[i] = key;
    }
}
