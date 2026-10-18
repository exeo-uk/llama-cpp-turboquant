#pragma once

// [TAG_FN_L22_DMASHARE] round 22 P14: the pure parts of the decode DMA share (D11), the code src/llama-fn-dmashare.cpp runs
// on the bridge's executor thread and its owner thread. No ggml, no device, no model: the pin registry and the clock come in
// as functions, so tests/test-fn-l22-dmashare.cpp runs it on the CPU.
//
// One bridged decode job (one MoE layer, T <= 8 tokens): its cold experts (not in the hot set; the opt-in levers' dropped
// experts left out) are served by the layer's VRAM bank slots when a slot still holds them (bank hit, no copy), else up to
// k of them are copied from pinned pages into free or least recently used slots (k from the CPU and DMA rates); the CPU
// pool computes the rest. A copied slot becomes a hit only after the device wrote a matching done record.

#include "ggml-fn-l22-dma.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace fn_dma {

constexpr int MAX_COPY = GGML_FN_L22_DMA_MAX_COPY;
constexpr int MAX_SEG  = GGML_FN_L22_DMA_MAX_SEG;
constexpr int EDGE_MAX = GGML_FN_L22_DMA_EDGE_MAX;
constexpr int MAX_T    = 8;  // LLAMA_MOE_DMA_MAX_T: one zero slot per bank, MMVQ widths only

// ---------------------------------------------------------------------------------------------------------------------
// switches

struct cfg {
    bool on    = false; // LLAMA_FN_L22_DMASHARE on and LLAMA_FN_L22_PIN >= 1
    int  slots = 4;     // LLAMA_FN_L22_DMASHARE_SLOTS: bank slots per layer = copies per job at most
};

inline cfg make_cfg(bool dmashare, int pin, int slots) {
    cfg c;
    c.on    = dmashare && pin >= 1;
    c.slots = std::min(MAX_COPY, std::max(1, slots));
    return c;
}

// ---------------------------------------------------------------------------------------------------------------------
// the opt-in levers' drop rule (P8 tail skip, P11 light rows): a cold expert whose smallest rank over the job's tokens is
// >= r, or whose largest weight is < w (weights only when the job has them), is not the GPU's: the bridged job and its sync
// retry then leave it to the same CPU rule

struct xrule {
    int   r = 0;    // 0: no rank rule
    float w = 0.0f; // 0: no weight rule
    bool on() const { return r > 0 || w > 0.0f; }
};

struct cold_e {
    int32_t e;
    int32_t n_tok;    // tokens that route to it
    int32_t min_rank; // smallest slot index
    float   max_w;    // largest weight (-inf without weights)
};

inline bool dropped(const xrule & x, const cold_e & c, bool have_w) {
    return (x.r > 0 && c.min_rank >= x.r) || (have_w && x.w > 0.0f && !(c.max_w >= x.w));
}

struct job_in {
    const int32_t * ids      = nullptr; // [n_used, T]
    const float   * w        = nullptr; // [n_used, T] or null
    int             n_used   = 0;
    int             T        = 0;
    int             n_exp    = 0;
    const int32_t * hot      = nullptr; // expert -> hot slot, hot_miss when not hot (null: nothing hot)
    int32_t         hot_miss = 0;
    xrule           tail;               // set when the job carries GGML_MOE_BRIDGE_JOB_TAIL
    xrule           defer;              // set when the job carries GGML_MOE_BRIDGE_JOB_DEFER
};

// per-layer scratch, n_exp entries each, -1 between jobs
struct scratch {
    std::vector<int32_t> idx;    // expert -> index in raw
    std::vector<int32_t> served; // expert -> global bank slot that serves it
    std::vector<cold_e>  raw;    // every cold expert of the job
    std::vector<cold_e>  cold;   // the ones the levers keep
    std::vector<cold_e>  miss;   // the kept ones without a bank hit

    void init(int n_exp) {
        idx.assign((size_t) n_exp, -1);
        served.assign((size_t) n_exp, -1);
        raw.reserve(64);
        cold.reserve(64);
        miss.reserve(64);
    }
};

// the job's cold experts in first-seen order (out), and how many the levers dropped
inline void cold_set(const job_in & j, scratch & s, std::vector<cold_e> & out, int * n_excl) {
    out.clear();
    *n_excl = 0;
    std::vector<cold_e> & all = s.raw;
    all.clear();
    for (int t = 0; t < j.T; ++t) {
        for (int i = 0; i < j.n_used; ++i) {
            const int32_t e = j.ids[(size_t) t*j.n_used + i];
            if (e < 0 || e >= j.n_exp || (j.hot != nullptr && j.hot[e] != j.hot_miss)) {
                continue;
            }
            int32_t k = s.idx[e];
            if (k < 0) {
                k = (int32_t) all.size();
                s.idx[e] = k;
                all.push_back({ e, 0, INT_MAX, -INFINITY });
            }
            cold_e & c = all[k];
            c.n_tok++;
            c.min_rank = std::min(c.min_rank, i);
            if (j.w != nullptr) {
                c.max_w = std::max(c.max_w, j.w[(size_t) t*j.n_used + i]);
            }
        }
    }
    const bool have_w = j.w != nullptr;
    for (const cold_e & c : all) {
        s.idx[c.e] = -1;
        if ((j.tail.on() && dropped(j.tail, c, have_w)) || (j.defer.on() && dropped(j.defer, c, have_w))) {
            (*n_excl)++;
            continue;
        }
        out.push_back(c);
    }
    all.clear();
}

// ---------------------------------------------------------------------------------------------------------------------
// how many to copy: the CPU computes m - k experts, the GPU copies and computes k and computes the hits; minimise the
// larger of the two (ties to the smaller k), then scale by the guard g in [0, 1]

struct rates {
    double cpu_us = 45.0; // CPU us per expert (EWMA of the jobs)
    double dma_us = 95.0; // device copy us per expert (EWMA of the done records)
    double gpu_us = 2.0;  // GPU compute us per bank expert
    double g      = 1.0;  // guard
};

inline int best_k(int m, int hits, const rates & r, int cap) {
    const int kmax = std::min(m, cap);
    if (kmax <= 0 || !(r.g > 0.0)) {
        return 0;
    }
    int    best = 0;
    double bt   = std::max(r.cpu_us*m, r.gpu_us*hits);
    for (int k = 1; k <= kmax; ++k) {
        const double t = std::max(r.cpu_us*(m - k), (r.dma_us + r.gpu_us)*k + r.gpu_us*hits);
        if (t < bt) {
            bt   = t;
            best = k;
        }
    }
    const double g = std::min(1.0, r.g);
    return std::min(kmax, (int) std::floor(best*g + 1e-9));
}

inline double ewma(double cur, double x, double a = 0.05) {
    return cur + a*(x - cur);
}

// device wait per layer (us; < 0 unknown): the GPU waits long for the CPU -> copy more, it hardly waits -> copy less
inline double guard_step(double g, double wait_us) {
    if (wait_us < 0.0) {
        return 1.0;
    }
    if (wait_us > 20.0) {
        g += 0.05;
    } else if (wait_us < 3.0) {
        g -= 0.05;
    }
    return std::min(1.0, std::max(0.0, g));
}

// ---------------------------------------------------------------------------------------------------------------------
// pieces of one expert slice: pinned ranges read by device address (their tickets held), short unpinned edges staged
// into the channel's edge area

enum reject {
    REJ_UNPINNED = 0, // an unpinned piece longer than an edge
    REJ_EDGE,         // the edge area is full
    REJ_SEGS,         // more than MAX_SEG pieces
    REJ_DEVADDR,      // a pinned piece without a device address
    REJ_RANGE,        // no host range for the slice, or a bad query answer
    REJ_SLOTS,        // no free bank slot
    REJ_N,
};

inline const char * reject_name(int r) {
    static const char * k[REJ_N] = { "unpinned", "edge", "segs", "devaddr", "range", "slots" };
    return r >= 0 && r < REJ_N ? k[r] : "?";
}

struct seg_q {
    const void * src    = nullptr; // the bytes the device reads (an arena copy may differ from the asked address)
    size_t       len    = 0;
    bool         pinned = false;
    int32_t      ticket = -1;
    uint64_t     dev    = 0;       // device address of src when pinned (0: unknown)
};

using query_fn   = std::function<bool(const void * p, size_t n, seg_q * out)>;
using release_fn = std::function<void(int32_t ticket)>;

struct edge_area {
    uint8_t * host = nullptr;
    uint64_t  dev  = 0;
    size_t    cap  = 0;
    size_t    used = 0;
};

// -1: ok (part filled, tickets appended), else a reject code (tickets of this slice already appended stay: the caller
// releases the expert's tickets)
inline int walk_slice(const uint8_t * p, size_t n, const query_fn & q, edge_area & ea, ggml_fn_l22_dma_part * part,
        std::vector<int32_t> & tickets) {
    memset(part, 0, sizeof(*part));
    if (p == nullptr || n == 0) {
        return REJ_RANGE;
    }
    size_t off = 0;
    while (off < n) {
        seg_q s;
        if (!q(p + off, n - off, &s) || s.len == 0 || s.len > n - off || s.src == nullptr) {
            if (s.pinned && s.ticket >= 0) {
                tickets.push_back(s.ticket);
            }
            return REJ_RANGE;
        }
        if (s.pinned && s.ticket >= 0) {
            tickets.push_back(s.ticket);
        }
        if (part->n_seg >= (uint32_t) MAX_SEG) {
            return REJ_SEGS;
        }
        ggml_fn_l22_dma_seg & g = part->seg[part->n_seg];
        if (s.pinned) {
            if (s.dev == 0 || s.len > UINT32_MAX) {
                return REJ_DEVADDR;
            }
            g.src    = s.dev;
            g.len    = (uint32_t) s.len;
            g.staged = 0;
        } else {
            if (s.len > (size_t) EDGE_MAX) {
                return REJ_UNPINNED;
            }
            const size_t at = (ea.used + 15) & ~(size_t) 15;
            if (ea.host == nullptr || at > ea.cap || ea.cap - at < s.len) {
                return REJ_EDGE;
            }
            memcpy(ea.host + at, s.src, s.len);
            ea.used  = at + s.len;
            g.src    = ea.dev + at;
            g.len    = (uint32_t) s.len;
            g.staged = 1;
        }
        part->n_seg++;
        off += s.len;
    }
    return -1;
}

// the three slices of one expert; on a reject the expert's tickets are released and the edge area is as before
inline int resolve_expert(const uint8_t * const p[3], const size_t n[3], const query_fn & q, const release_fn & rel,
        edge_area & ea, ggml_fn_l22_dma_part parts[3], std::vector<int32_t> & tickets) {
    const size_t t0 = tickets.size();
    const size_t u0 = ea.used;
    for (int k = 0; k < 3; ++k) {
        const int rc = walk_slice(p[k], n[k], q, ea, &parts[k], tickets);
        if (rc >= 0) {
            for (size_t i = t0; i < tickets.size(); ++i) {
                rel(tickets[i]);
            }
            tickets.resize(t0);
            ea.used = u0;
            return rc;
        }
    }
    return -1;
}

// the kernel's copy rule on the CPU (tests): the pieces in order at consecutive bytes of dst. map: device -> host address
inline size_t emulate_copy(const ggml_fn_l22_dma_part & part, uint8_t * dst, const std::function<const uint8_t *(uint64_t)> & map) {
    size_t off = 0;
    for (uint32_t s = 0; s < part.n_seg; ++s) {
        const uint8_t * src = map(part.seg[s].src);
        if (src != nullptr) {
            memcpy(dst + off, src, part.seg[s].len);
        }
        off += part.seg[s].len;
    }
    return off;
}

// ---------------------------------------------------------------------------------------------------------------------
// the bank slots of one layer

struct layer_state {
    int                   S         = 0;
    int32_t               slot_lo   = 0; // global bank slot of local slot 0
    int32_t               zero_slot = 0; // global zero slot
    std::vector<int32_t>  slot_exp;      // [S] expert of the slot (valid or pending), -1
    std::vector<uint8_t>  valid;         // [S] holds slot_exp (a committed copy)
    std::vector<uint8_t>  pending;       // [S] written by a copy not committed yet
    std::vector<uint64_t> last_use;      // [S]
    std::vector<uint8_t>  used;          // [S] scratch: served by the job being planned
    std::vector<int8_t>   mirror;        // [n_exp] local slot of a valid expert, -1

    void init(int s, int32_t lo, int32_t zero, int n_exp) {
        S         = s;
        slot_lo   = lo;
        zero_slot = zero;
        slot_exp.assign((size_t) s, -1);
        valid.assign((size_t) s, 0);
        pending.assign((size_t) s, 0);
        last_use.assign((size_t) s, 0);
        used.assign((size_t) s, 0);
        mirror.assign((size_t) n_exp, -1);
    }

    // a slot neither served by this job nor pending: an empty one first, else the least recently used; -1 none
    int victim() const {
        int best = -1;
        for (int s = 0; s < S; ++s) {
            if (used[s] || pending[s]) {
                continue;
            }
            if (!valid[s]) {
                return s;
            }
            if (best < 0 || last_use[s] < last_use[best]) {
                best = s;
            }
        }
        return best;
    }

    void drop(int s) {
        if (valid[s] && slot_exp[s] >= 0 && slot_exp[s] < (int32_t) mirror.size() && mirror[slot_exp[s]] == s) {
            mirror[slot_exp[s]] = -1;
        }
        valid[s]    = 0;
        slot_exp[s] = -1;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// one job's plan

// fills the expert's three parts (tickets appended); -1 ok, else a reject code with nothing held
using resolve_fn = std::function<int(int32_t e, ggml_fn_l22_dma_part parts[3], std::vector<int32_t> & tickets)>;

struct plan_res {
    int     n_cold = 0; // cold experts (dropped ones included)
    int     n_excl = 0; // dropped by the levers' rule
    int     n_hit  = 0;
    int     n_copy = 0;
    int     k      = 0; // copies wanted
    int     n_try  = 0; // candidates resolved
    int     rej[REJ_N] = {};
    int32_t copy_slot[MAX_COPY] = {}; // local slots
    int32_t copy_e   [MAX_COPY] = {};
};

// writes side (seq, n_copy, slots, parts), slot_ids [n_used, T] (served slot or the zero slot) and skip[e] = 1 for the
// served experts; marks the copies' slots pending. pins = false: hits only. tick: the slots' use clock
inline plan_res plan_job(layer_state & L, const job_in & j, scratch & sc, const rates & r, int max_copy, bool pins, uint64_t tick,
        const resolve_fn & resolve, ggml_fn_l22_dma_side * side, uint32_t seq, int32_t * slot_ids, int32_t * skip,
        std::vector<int32_t> & tickets) {
    plan_res res;
    side->seq    = seq;
    side->n_copy = 0;
    for (int i = 0; i < j.n_used*j.T; ++i) {
        slot_ids[i] = L.zero_slot;
    }
    cold_set(j, sc, sc.cold, &res.n_excl);
    res.n_cold = (int) sc.cold.size() + res.n_excl;
    std::fill(L.used.begin(), L.used.end(), 0);

    // bank hits
    sc.miss.clear();
    for (const cold_e & c : sc.cold) {
        const int s = L.mirror[c.e];
        if (s >= 0 && s < L.S && L.valid[s] && !L.pending[s] && L.slot_exp[s] == c.e) {
            L.used[s]     = 1;
            L.last_use[s] = tick;
            sc.served[c.e] = L.slot_lo + s;
            res.n_hit++;
        } else {
            sc.miss.push_back(c);
        }
    }

    // copies: the most tokens first, then the heaviest
    int n_free = 0;
    for (int s = 0; s < L.S; ++s) {
        n_free += (!L.used[s] && !L.pending[s]) ? 1 : 0;
    }
    const int cap = std::min({ max_copy, n_free, MAX_COPY });
    res.k = pins ? best_k((int) sc.miss.size(), res.n_hit, r, cap) : 0;
    if (res.k > 0) {
        std::sort(sc.miss.begin(), sc.miss.end(), [](const cold_e & a, const cold_e & b) {
            if (a.n_tok != b.n_tok) {
                return a.n_tok > b.n_tok;
            }
            if (a.max_w != b.max_w) {
                return a.max_w > b.max_w;
            }
            return a.e < b.e;
        });
        for (const cold_e & c : sc.miss) {
            if (res.n_copy >= res.k || res.n_try >= 2*res.k + 4) {
                break;
            }
            const int s = L.victim();
            if (s < 0) {
                res.rej[REJ_SLOTS]++;
                break;
            }
            res.n_try++;
            ggml_fn_l22_dma_part parts[3];
            const int rc = resolve(c.e, parts, tickets);
            if (rc >= 0) {
                res.rej[rc < REJ_N ? rc : REJ_RANGE]++;
                continue;
            }
            L.drop(s);
            L.pending[s]  = 1;
            L.slot_exp[s] = c.e;
            L.used[s]     = 1;
            L.last_use[s] = tick;
            const int i = res.n_copy++;
            side->slot[i] = L.slot_lo + s;
            for (int k = 0; k < 3; ++k) {
                side->part[i][k] = parts[k];
            }
            res.copy_slot[i] = s;
            res.copy_e[i]    = c.e;
            sc.served[c.e]   = L.slot_lo + s;
        }
    }
    side->n_copy = res.n_copy;

    // slot ids and skip marks of the served experts
    for (int t = 0; t < j.T; ++t) {
        for (int i = 0; i < j.n_used; ++i) {
            const int32_t e = j.ids[(size_t) t*j.n_used + i];
            if (e >= 0 && e < j.n_exp && sc.served[e] >= 0) {
                slot_ids[(size_t) t*j.n_used + i] = sc.served[e];
            }
        }
    }
    for (const cold_e & c : sc.cold) {
        if (sc.served[c.e] >= 0) {
            skip[c.e]      = 1;
            sc.served[c.e] = -1;
        }
    }
    return res;
}

// ---------------------------------------------------------------------------------------------------------------------
// the commit after the graph synced

struct pending_job {
    bool                 on     = false;
    uint32_t             seq    = 0;
    int                  n      = 0;
    int32_t              slot[MAX_COPY] = {}; // local slots
    int32_t              e   [MAX_COPY] = {};
    std::vector<int32_t> tickets;
};

enum commit_rc {
    CM_NONE = 0, // nothing pending
    CM_OK,       // the copies are in their slots: hits from now on
    CM_BAD,      // no matching done record: the slots are empty again
};

// done: the device's record of the channel. Every ticket goes back either way; a second call does nothing
inline int commit(layer_state & L, pending_job & P, const ggml_fn_l22_dma_done & done, const release_fn & rel) {
    if (!P.on) {
        return CM_NONE;
    }
    const bool good = done.seq == P.seq && done.ok == 1 && done.n == P.n;
    for (int i = 0; i < P.n; ++i) {
        const int s = P.slot[i];
        if (s < 0 || s >= L.S) {
            continue;
        }
        L.pending[s] = 0;
        if (good && L.slot_exp[s] == P.e[i]) {
            const int32_t e = P.e[i];
            if (e >= 0 && e < (int32_t) L.mirror.size()) {
                const int old = L.mirror[e];
                if (old >= 0 && old != s && old < L.S) {
                    L.valid[old]    = 0; // a second copy of the same expert: keep the new one
                    L.slot_exp[old] = -1;
                }
                L.mirror[e] = (int8_t) s;
            }
            L.valid[s] = 1;
        } else {
            L.valid[s]    = 0;
            L.slot_exp[s] = -1;
        }
    }
    for (int32_t t : P.tickets) {
        rel(t);
    }
    P.tickets.clear();
    P.on = false;
    P.n  = 0;
    return good ? CM_OK : CM_BAD;
}

} // namespace fn_dma
