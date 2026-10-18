#pragma once

// [TAG_FN_L22_TOKCACHE] round 22 E11: tokenize a text that shares a long prefix with a recent one by re-using the ids up
// to the last special-token boundary of that prefix and encoding only the rest. The ids equal a full llama_tokenize.
//
// Why it is exact: with parse_special, special tokens cut the text into fragments before BPE, and BPE never merges across
// a fragment edge. A boundary b is the byte offset right after a special token of the full encode of the old text. For a
// new text that shares the first L bytes, every special-token match that ends at or before L - max_special_len is the
// same in both texts (a match that starts later cannot reach back across b), so the new text also has a fragment edge at
// b with the same fragments before it, and the fragments after b are what encoding text[b:] alone gives.
// Guards: BPE vocabularies only (no space prefix per fragment), not with add_special when the vocabulary adds EOS / SEP,
// and boundaries only where the ids' pieces reproduce the text byte for byte (a stripped space around a special token,
// or any non-lossless piece, leaves the entry without boundaries: exact re-use only).
//
// Header only and llama.h only, so libmtmd can use it without linking common.

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct common_tokcache_stats {
    int    path         = 0; // 0 full encode, 1 exact text, 2 prefix re-use, 3 cache off
    size_t reused_ids   = 0;
    size_t reused_bytes = 0;
    bool   verified     = false; // compared with a full encode
    bool   mismatch     = false; // the comparison failed: the full encode was returned and the cache is off
};

struct common_tokcache {
    struct entry {
        std::string              text;
        std::vector<llama_token> ids;
        std::vector<std::pair<uint32_t, uint32_t>> bounds; // (byte offset, token index) after each special token
        bool add_special   = false;
        bool parse_special = false;
        size_t bytes() const { return text.size() + ids.size()*sizeof(llama_token) + bounds.size()*8 + 64; }
    };

    explicit common_tokcache(size_t n_max = 4, size_t bytes_max = (size_t) 64 << 20, size_t min_bytes = 256)
        : n_max(n_max), bytes_max(bytes_max), min_bytes(min_bytes) {}

    size_t n_max;
    size_t bytes_max;
    size_t min_bytes;         // shorter texts are encoded directly and not stored
    std::atomic<int>  n_verify{0};   // compare this many cached results with a full encode
    std::atomic<bool> off{false};    // set after a verify mismatch

    std::mutex mu;
    std::vector<std::shared_ptr<const entry>> lru; // most recent first
    size_t bytes = 0;

    void clear() {
        std::lock_guard<std::mutex> lk(mu);
        lru.clear();
        bytes = 0;
    }
};

namespace common_tokcache_impl {

// per vocabulary: the pieces (special = true) of every token, which tokens end a fragment, the longest special text
struct vocab_info {
    const llama_vocab *   vocab = nullptr;
    bool                  ok    = false;
    std::string           pieces;  // all pieces back to back
    std::vector<uint32_t> off;     // n_tokens + 1 offsets into pieces
    std::vector<uint8_t>  bound;   // 1: a CONTROL or USER_DEFINED token (a fragment of its own when parse_special)
    size_t                max_special = 0;
    llama_token           bos = LLAMA_TOKEN_NULL;
    bool                  add_bos = false, add_eos = false, add_sep = false;
};

inline std::vector<llama_token> tokenize_plain(const llama_vocab * vocab, const char * text, size_t len, bool add_special,
        bool parse_special) {
    if (len > (size_t) INT32_MAX - 4) {
        throw std::runtime_error("tokcache: input text too large");
    }
    std::vector<llama_token> res(len + 2 * (add_special ? 1 : 0));
    int32_t n = llama_tokenize(vocab, text, (int32_t) len, res.data(), (int32_t) res.size(), add_special, parse_special);
    if (n == INT32_MIN) {
        throw std::runtime_error("tokcache: tokenization result exceeds int32_t limit");
    }
    if (n < 0) {
        res.resize(-n);
        const int32_t check = llama_tokenize(vocab, text, (int32_t) len, res.data(), (int32_t) res.size(), add_special, parse_special);
        if (check != -n) {
            throw std::runtime_error("tokcache: tokenize size changed");
        }
    } else {
        res.resize(n);
    }
    return res;
}

inline const vocab_info & get_info(const llama_vocab * vocab) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<vocab_info>> all; // a process holds one or two vocabularies
    std::lock_guard<std::mutex> lk(mu);
    for (const auto & v : all) {
        if (v->vocab == vocab) {
            return *v;
        }
    }
    auto vi = std::make_unique<vocab_info>();
    vi->vocab   = vocab;
    vi->add_bos = llama_vocab_get_add_bos(vocab);
    vi->add_eos = llama_vocab_get_add_eos(vocab);
    vi->add_sep = llama_vocab_get_add_sep(vocab);
    vi->bos     = llama_vocab_bos(vocab);
    const int32_t n = llama_vocab_n_tokens(vocab);
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_BPE && n > 0) {
        vi->off.resize((size_t) n + 1);
        vi->bound.resize((size_t) n);
        std::vector<char> buf(256);
        for (int32_t id = 0; id < n; ++id) {
            vi->off[id] = (uint32_t) vi->pieces.size();
            int32_t k = llama_token_to_piece(vocab, id, buf.data(), (int32_t) buf.size(), 0, true);
            if (k < 0) {
                buf.resize((size_t) -k);
                k = llama_token_to_piece(vocab, id, buf.data(), (int32_t) buf.size(), 0, true);
            }
            if (k > 0) {
                vi->pieces.append(buf.data(), (size_t) k);
            }
            const auto attr = llama_vocab_get_attr(vocab, id);
            if (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                vi->max_special = std::max(vi->max_special, (size_t) std::max(k, 0));
                // a token that strips spaces around it changes the bytes next to the edge: never an edge here
                if ((attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) &&
                    !(attr & (LLAMA_TOKEN_ATTR_LSTRIP | LLAMA_TOKEN_ATTR_RSTRIP)) && k > 0) {
                    vi->bound[id] = 1;
                }
            }
        }
        vi->off[n] = (uint32_t) vi->pieces.size();
        vi->ok = true;
    }
    all.push_back(std::move(vi));
    return *all.back();
}

// boundaries of ids[i0..] that encode text[b0..]: false (out cleared) unless the pieces reproduce those bytes exactly
inline bool find_bounds(const vocab_info & vi, const std::string & text, size_t b0, const std::vector<llama_token> & ids,
        size_t i0, bool skip_bos, std::vector<std::pair<uint32_t, uint32_t>> & out) {
    const size_t n0 = out.size();
    size_t b = b0;
    const int32_t n_vocab = (int32_t) vi.bound.size();
    for (size_t i = i0; i < ids.size(); ++i) {
        const llama_token id = ids[i];
        if (i == 0 && skip_bos && id == vi.bos) {
            continue; // the BOS that add_special put in front: not in the text
        }
        if (id < 0 || id >= n_vocab) {
            out.resize(n0);
            return false;
        }
        const size_t len = vi.off[id + 1] - vi.off[id];
        if (b + len > text.size() || (len > 0 && std::memcmp(text.data() + b, vi.pieces.data() + vi.off[id], len) != 0)) {
            out.resize(n0);
            return false;
        }
        b += len;
        if (vi.bound[id] && b <= UINT32_MAX && i + 1 <= UINT32_MAX) {
            out.emplace_back((uint32_t) b, (uint32_t) (i + 1));
        }
    }
    if (b != text.size()) {
        out.resize(n0);
        return false;
    }
    return true;
}

inline size_t common_prefix(const std::string & a, const std::string & b) {
    const size_t n = std::min(a.size(), b.size());
    const auto mm = std::mismatch(a.data(), a.data() + n, b.data());
    return (size_t) (mm.first - a.data());
}

} // namespace common_tokcache_impl

// the ids of llama_tokenize(vocab, text, add_special, parse_special), through the cache when it can help
inline std::vector<llama_token> common_tokcache_tokenize(common_tokcache & c, const llama_vocab * vocab, const std::string & text,
        bool add_special, bool parse_special, common_tokcache_stats * st = nullptr) {
    using namespace common_tokcache_impl;
    common_tokcache_stats local;
    common_tokcache_stats & s = st ? *st : local;
    s = common_tokcache_stats();

    if (c.off.load(std::memory_order_relaxed) || !parse_special || text.size() < c.min_bytes || c.n_max == 0) {
        s.path = 3;
        return tokenize_plain(vocab, text.data(), text.size(), add_special, parse_special);
    }
    const vocab_info & vi = get_info(vocab);
    if (!vi.ok || (add_special && (vi.add_eos || vi.add_sep))) {
        s.path = 3;
        return tokenize_plain(vocab, text.data(), text.size(), add_special, parse_special);
    }

    std::vector<std::shared_ptr<const common_tokcache::entry>> snap;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        snap = c.lru;
    }

    std::shared_ptr<const common_tokcache::entry> best;
    size_t best_l = 0;
    for (const auto & e : snap) {
        if (e->add_special != add_special || e->parse_special != parse_special) {
            continue;
        }
        const size_t l = common_prefix(e->text, text);
        if (l == text.size() && l == e->text.size()) {
            best   = e;
            best_l = l;
            break;
        }
        if (l > best_l) {
            best   = e;
            best_l = l;
        }
    }

    auto ne = std::make_shared<common_tokcache::entry>();
    ne->add_special   = add_special;
    ne->parse_special = parse_special;
    const bool skip_bos = add_special && vi.add_bos;

    if (best && best_l == text.size() && best_l == best->text.size()) {
        s.path         = 1;
        s.reused_ids   = best->ids.size();
        s.reused_bytes = text.size();
        // touch: move to the front
        std::lock_guard<std::mutex> lk(c.mu);
        auto it = std::find(c.lru.begin(), c.lru.end(), best);
        if (it != c.lru.end()) {
            std::rotate(c.lru.begin(), it, it + 1);
        }
        return best->ids;
    }

    // the last boundary of the best prefix that no special token of the new text can reach across
    const std::pair<uint32_t, uint32_t> * cut = nullptr;
    size_t n_keep_bounds = 0;
    if (best && best_l > vi.max_special && !best->bounds.empty()) {
        const size_t lim = best_l - vi.max_special;
        auto it = std::upper_bound(best->bounds.begin(), best->bounds.end(), lim,
                [](size_t v, const std::pair<uint32_t, uint32_t> & p) { return v < (size_t) p.first; });
        if (it != best->bounds.begin()) {
            cut = &*(it - 1);
            n_keep_bounds = (size_t) (it - best->bounds.begin());
        }
    }

    if (cut != nullptr) {
        const size_t b   = cut->first;
        const size_t idx = cut->second;
        std::vector<llama_token> rest = tokenize_plain(vocab, text.data() + b, text.size() - b, false, parse_special);
        ne->ids.reserve(idx + rest.size());
        ne->ids.assign(best->ids.begin(), best->ids.begin() + idx);
        ne->ids.insert(ne->ids.end(), rest.begin(), rest.end());
        ne->bounds.assign(best->bounds.begin(), best->bounds.begin() + n_keep_bounds);
        ne->text = text;
        if (!find_bounds(vi, ne->text, b, ne->ids, idx, false, ne->bounds)) {
            ne->bounds.resize(n_keep_bounds); // the rest is not lossless: re-use up to b only
        }
        s.path         = 2;
        s.reused_ids   = idx;
        s.reused_bytes = b;
        if (c.n_verify.load(std::memory_order_relaxed) > 0) {
            c.n_verify.fetch_sub(1);
            s.verified = true;
            std::vector<llama_token> full = tokenize_plain(vocab, text.data(), text.size(), add_special, parse_special);
            if (full != ne->ids) {
                s.mismatch = true;
                c.off.store(true);
                c.clear();
                return full;
            }
        }
    } else {
        ne->text = text;
        ne->ids  = tokenize_plain(vocab, text.data(), text.size(), add_special, parse_special);
        find_bounds(vi, ne->text, 0, ne->ids, 0, skip_bos, ne->bounds);
        s.path = 0;
    }

    std::vector<llama_token> res = ne->ids;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        c.bytes += ne->bytes();
        c.lru.insert(c.lru.begin(), std::move(ne));
        while (!c.lru.empty() && (c.lru.size() > c.n_max || (c.bytes > c.bytes_max && c.lru.size() > 1))) {
            c.bytes -= std::min(c.bytes, c.lru.back()->bytes());
            c.lru.pop_back();
        }
    }
    return res;
}
