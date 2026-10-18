#pragma once

// [TAG_FN_L22_LOGITDUMP] round 22 P15: the logit dump file format L22DUMP1 and its pure parts (std only: the server's
// dump and the CPU test tests/test-fn-l22-dump.cpp use it; r22/test/kld_dump.py reads and writes the same format).
//
// Little endian. Header (64 bytes + the forced ids):
//   char magic[8] "L22DUMP1"; int32 version (1), n_vocab, k, window, n_rec, n_forced, n_prompt, flags;
//   uint64 prompt_hash, forced_hash, ref_hash; int32 forced[n_forced]
// n_rec records follow, record n = the distribution after the prompt (n = 0) or after forced[n - 1]:
//   int32 n, next (forced[n], -1 after the last forced token); float lp_next (log-prob of next, 0 without one),
//   log_tail (log of the probability outside ids, LOG_ZERO when it is 0); int32 ids[k] (the k most likely, higher logit
//   first, ties by lower id: the order of perplexity's sparse base rows); float lps[k] (their log-probs);
//   with FLAG_REF also float ref_lps[k] (this run's log-probs at the ref dump's ids of record n) and float ref_log_tail
//   (log of this run's probability outside those ids); both NaN when the ref has no record n (FLAG_REF_PARTIAL).
// With the REF part a comparer computes the exact sparse KLD of perplexity's kld_sparse_token (base top-k plus tail).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace l22dump {

static constexpr char     MAGIC[8]     = { 'L', '2', '2', 'D', 'U', 'M', 'P', '1' };
static constexpr int32_t  VERSION      = 1;
static constexpr int32_t  K_MAX        = 256;
static constexpr int32_t  WINDOW_MAX   = 64;
static constexpr int32_t  REC_MAX      = 4096; // records incl. record 0, so at most REC_MAX - 1 forced tokens
static constexpr float    LOG_ZERO     = -1e30f;
static constexpr uint64_t FNV_OFFSET   = 0xcbf29ce484222325ULL;
static constexpr uint64_t FNV_PRIME    = 0x100000001b3ULL;
static constexpr size_t   HEADER_BYTES = 64;
static constexpr long     OFF_N_REC    = 24; // 8 + 4*4
static constexpr long     OFF_FLAGS    = 36; // 8 + 7*4

enum : int32_t {
    FLAG_REF         = 1, // records carry ref_lps / ref_log_tail
    FLAG_ABORTED     = 2, // the forced decode stopped early (the records present are valid)
    FLAG_FORCED_TEXT = 4, // the forced ids came from "forced_text"
    FLAG_REF_PARTIAL = 8, // the ref had fewer records: the REF part of the rest is NaN
};

struct header {
    int32_t  version     = VERSION;
    int32_t  n_vocab     = 0;
    int32_t  k           = 0;
    int32_t  window      = 0;
    int32_t  n_rec       = 0;
    int32_t  n_forced    = 0;
    int32_t  n_prompt    = 0;
    int32_t  flags       = 0;
    uint64_t prompt_hash = 0;
    uint64_t forced_hash = 0;
    uint64_t ref_hash    = 0;
};

struct record {
    int32_t              n        = 0;
    int32_t              next     = -1;
    float                lp_next  = 0.0f;
    float                log_tail = LOG_ZERO;
    std::vector<int32_t> ids;
    std::vector<float>   lps;
    std::vector<float>   ref_lps;            // empty unless FLAG_REF
    float                ref_log_tail = NAN; // FLAG_REF only
};

// work buffers of make_record, reused across records
struct scratch {
    std::vector<std::pair<float, int32_t>> heap;
    std::vector<float>                     e;    // exp(logit - max)
    std::vector<uint8_t>                   mark;
};

inline uint64_t fnv1a(const void * data, size_t n, uint64_t h = FNV_OFFSET) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

// FNV-1a over the int32 ids (little endian bytes), then over the int64 extra
inline uint64_t hash_ids(const int32_t * ids, size_t n, int64_t extra) {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < n; ++i) {
        const int32_t v = ids[i];
        h = fnv1a(&v, 4, h);
    }
    return fnv1a(&extra, 8, h);
}

// the hash a REF dump names its ref by: header fields and every record's n and ids
inline uint64_t ref_hash_of(const header & h, const std::vector<record> & recs) {
    const int32_t v[4] = { h.n_vocab, h.k, h.n_rec, h.n_forced };
    uint64_t x = fnv1a(v, sizeof(v));
    x = fnv1a(&h.prompt_hash, 8, x);
    x = fnv1a(&h.forced_hash, 8, x);
    for (const record & r : recs) {
        x = fnv1a(&r.n, 4, x);
        x = fnv1a(r.ids.data(), r.ids.size()*4, x);
    }
    return x;
}

// a dump name: 1-96 chars of [A-Za-z0-9._-], starting with a letter or digit, no Windows device name
inline bool name_ok(const std::string & s) {
    if (s.empty() || s.size() > 96) {
        return false;
    }
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        const bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (!alnum && (i == 0 || (c != '.' && c != '_' && c != '-'))) {
            return false;
        }
    }
    std::string stem = s.substr(0, s.find('.'));
    for (char & c : stem) {
        c = (char) ((c >= 'a' && c <= 'z') ? c - 32 : c);
    }
    static const char * dev[] = { "CON", "PRN", "AUX", "NUL" };
    for (const char * d : dev) {
        if (stem == d) {
            return false;
        }
    }
    if (stem.size() == 4 && (stem.compare(0, 3, "COM") == 0 || stem.compare(0, 3, "LPT") == 0) && stem[3] >= '0' && stem[3] <= '9') {
        return false;
    }
    return true;
}

// forced tokens used for a request: at most REC_MAX - 1 and what the slot has room for (room = n_ctx_slot - n_prompt - 1)
inline int32_t plan_forced(int64_t n_req, int64_t room) {
    int64_t n = std::min<int64_t>(n_req, REC_MAX - 1);
    n = std::min<int64_t>(n, room);
    return (int32_t) std::max<int64_t>(n, 0);
}

inline int32_t clamp_i(int64_t v, int32_t lo, int32_t hi) {
    return (int32_t) std::min<int64_t>(std::max<int64_t>(v, lo), hi);
}

inline size_t record_bytes(int32_t k, bool ref) {
    return 16 + size_t(k)*8 + (ref ? size_t(k)*4 + 4 : 0);
}

// one record from a logits row. ref_ids: the ref's ids of this record (k of them) or null. false: bad arguments or a
// non-finite logit (nothing is written to r then)
inline bool make_record(const float * logits, int32_t n_vocab, int32_t k, int32_t n, int32_t next,
                        const int32_t * ref_ids, scratch & s, record & r) {
    if (logits == nullptr || n_vocab <= 0 || k < 1 || k > n_vocab || next < -1 || next >= n_vocab) {
        return false;
    }
    float mx = logits[0];
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float l = logits[i];
        if (!std::isfinite(l)) {
            return false;
        }
        mx = l > mx ? l : mx;
    }
    if (ref_ids != nullptr) {
        for (int32_t i = 0; i < k; ++i) {
            if (ref_ids[i] < 0 || ref_ids[i] >= n_vocab) {
                return false;
            }
        }
    }
    s.e.resize(n_vocab);
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        s.e[i] = expf(logits[i] - mx);
        sum += s.e[i];
    }
    const double lse = (double) mx + log(sum);

    // better(a, b): higher logit, then lower id; as the heap order the front is the worst of the k kept
    auto better = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    s.heap.clear();
    for (int32_t i = 0; i < n_vocab; ++i) {
        const std::pair<float, int32_t> e(logits[i], i);
        if ((int32_t) s.heap.size() < k) {
            s.heap.push_back(e);
            std::push_heap(s.heap.begin(), s.heap.end(), better);
        } else if (better(e, s.heap.front())) {
            std::pop_heap(s.heap.begin(), s.heap.end(), better);
            s.heap.back() = e;
            std::push_heap(s.heap.begin(), s.heap.end(), better);
        }
    }
    std::sort_heap(s.heap.begin(), s.heap.end(), better);

    // the mass outside a set of ids, summed directly over the rest (no 1 - sum cancellation)
    s.mark.assign(n_vocab, 0);
    auto log_outside = [&](const int32_t * ids) -> float {
        for (int32_t i = 0; i < k; ++i) {
            s.mark[ids[i]] = 1;
        }
        double t = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            t += s.mark[i] ? 0.0 : (double) s.e[i];
        }
        for (int32_t i = 0; i < k; ++i) {
            s.mark[ids[i]] = 0;
        }
        return t > 0.0 ? (float) (log(t) - log(sum)) : LOG_ZERO;
    };

    r.n    = n;
    r.next = next;
    r.ids.resize(k);
    r.lps.resize(k);
    for (int32_t i = 0; i < k; ++i) {
        r.ids[i] = s.heap[i].second;
        r.lps[i] = (float) ((double) s.heap[i].first - lse);
    }
    r.lp_next  = next >= 0 ? (float) ((double) logits[next] - lse) : 0.0f;
    r.log_tail = log_outside(r.ids.data());
    if (ref_ids != nullptr) {
        r.ref_lps.resize(k);
        for (int32_t i = 0; i < k; ++i) {
            r.ref_lps[i] = (float) ((double) logits[ref_ids[i]] - lse);
        }
        r.ref_log_tail = log_outside(ref_ids);
    } else {
        r.ref_lps.clear();
        r.ref_log_tail = NAN;
    }
    return true;
}

// a record whose ref has no record n: the REF part is NaN
inline void set_ref_missing(record & r, int32_t k) {
    r.ref_lps.assign(k, NAN);
    r.ref_log_tail = NAN;
}

inline bool write_header(FILE * f, const header & h, const int32_t * forced) {
    const int32_t v[8] = { h.version, h.n_vocab, h.k, h.window, h.n_rec, h.n_forced, h.n_prompt, h.flags };
    const uint64_t u[3] = { h.prompt_hash, h.forced_hash, h.ref_hash };
    bool ok = fwrite(MAGIC, 1, 8, f) == 8;
    ok = ok && fwrite(v, 4, 8, f) == 8;
    ok = ok && fwrite(u, 8, 3, f) == 3;
    ok = ok && (h.n_forced == 0 || fwrite(forced, 4, (size_t) h.n_forced, f) == (size_t) h.n_forced);
    return ok;
}

inline bool write_record(FILE * f, const record & r, int32_t k, bool ref) {
    if ((int32_t) r.ids.size() != k || (int32_t) r.lps.size() != k || (ref && (int32_t) r.ref_lps.size() != k)) {
        return false;
    }
    const int32_t a[2] = { r.n, r.next };
    const float   b[2] = { r.lp_next, r.log_tail };
    bool ok = fwrite(a, 4, 2, f) == 2;
    ok = ok && fwrite(b, 4, 2, f) == 2;
    ok = ok && fwrite(r.ids.data(), 4, (size_t) k, f) == (size_t) k;
    ok = ok && fwrite(r.lps.data(), 4, (size_t) k, f) == (size_t) k;
    if (ref) {
        ok = ok && fwrite(r.ref_lps.data(), 4, (size_t) k, f) == (size_t) k;
        ok = ok && fwrite(&r.ref_log_tail, 4, 1, f) == 1;
    }
    return ok;
}

// rewrites n_rec and flags in place and returns to the end of the file
inline bool patch_header(FILE * f, int32_t n_rec, int32_t flags) {
    bool ok = fseek(f, OFF_N_REC, SEEK_SET) == 0 && fwrite(&n_rec, 4, 1, f) == 1;
    ok = ok && fseek(f, OFF_FLAGS, SEEK_SET) == 0 && fwrite(&flags, 4, 1, f) == 1;
    ok = ok && fseek(f, 0, SEEK_END) == 0;
    return ok;
}

// reads a whole dump; false with err set on a bad or short file
inline bool read_dump(const std::string & path, header & h, std::vector<int32_t> & forced, std::vector<record> & recs,
                      std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    auto fail = [&](const std::string & why) {
        fclose(f);
        err = path + ": " + why;
        return false;
    };
    char    magic[8];
    int32_t v[8];
    uint64_t u[3];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MAGIC, 8) != 0) {
        return fail("not an L22DUMP1 file");
    }
    if (fread(v, 4, 8, f) != 8 || fread(u, 8, 3, f) != 3) {
        return fail("short header");
    }
    h.version = v[0]; h.n_vocab = v[1]; h.k = v[2]; h.window = v[3]; h.n_rec = v[4]; h.n_forced = v[5]; h.n_prompt = v[6];
    h.flags = v[7]; h.prompt_hash = u[0]; h.forced_hash = u[1]; h.ref_hash = u[2];
    if (h.version != VERSION) {
        return fail("version " + std::to_string(h.version));
    }
    if (h.n_vocab < 1 || h.k < 1 || h.k > K_MAX || h.k > h.n_vocab || h.n_forced < 0 || h.n_forced > REC_MAX - 1 ||
            h.n_rec < 0 || h.n_rec > h.n_forced + 1) {
        return fail("bad header values");
    }
    forced.resize(h.n_forced);
    if (h.n_forced > 0 && fread(forced.data(), 4, (size_t) h.n_forced, f) != (size_t) h.n_forced) {
        return fail("short forced ids");
    }
    const bool ref = (h.flags & FLAG_REF) != 0;
    recs.assign(h.n_rec, record());
    for (int32_t i = 0; i < h.n_rec; ++i) {
        record & r = recs[i];
        int32_t a[2];
        float   b[2];
        r.ids.resize(h.k);
        r.lps.resize(h.k);
        bool ok = fread(a, 4, 2, f) == 2 && fread(b, 4, 2, f) == 2;
        ok = ok && fread(r.ids.data(), 4, (size_t) h.k, f) == (size_t) h.k;
        ok = ok && fread(r.lps.data(), 4, (size_t) h.k, f) == (size_t) h.k;
        if (ok && ref) {
            r.ref_lps.resize(h.k);
            ok = fread(r.ref_lps.data(), 4, (size_t) h.k, f) == (size_t) h.k && fread(&r.ref_log_tail, 4, 1, f) == 1;
        }
        if (!ok) {
            return fail("short record " + std::to_string(i));
        }
        r.n = a[0]; r.next = a[1]; r.lp_next = b[0]; r.log_tail = b[1];
        if (r.n != i || r.next < -1 || r.next >= h.n_vocab) {
            return fail("bad record " + std::to_string(i));
        }
        for (int32_t j = 0; j < h.k; ++j) {
            if (r.ids[j] < 0 || r.ids[j] >= h.n_vocab) {
                return fail("record " + std::to_string(i) + " holds an id outside the vocabulary");
            }
        }
    }
    char extra;
    if (fread(&extra, 1, 1, f) != 0) {
        return fail("trailing bytes");
    }
    fclose(f);
    return true;
}

} // namespace l22dump
