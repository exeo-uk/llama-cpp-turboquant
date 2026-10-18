// [TAG_FN_L22_RESSET] [TAG_FN_L22_REWARM] [TAG_FN_L22_SSDTIER] round 22, P5: the residency table (one class per routed
// expert: RAM / VRAM / PIN / SSD), its service thread (standby drop, eviction prefetch, idle re-warm, SSD tier, the
// residency log) and the warm list. See llama-fn-l22-p5.h. Residency and read paths only: nothing changes a value the
// model computes.

#include "llama-fn-l22.h"
#include "llama-fn-l22-p5.h"
#include "llama-fn-res-core.h"

#include "llama-fn-auto.h"
#include "llama-fn-stall.h"
#include "llama-fn-tier.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moecache.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <psapi.h>
#endif

namespace {

using namespace fn_res;

constexpr size_t GIB   = (size_t) 1 << 30;
constexpr size_t BLOCK = (size_t) 2 << 20; // the re-warm's sample and read unit

struct res_layer {
    int                                      il      = -1;
    const ggml_tensor *                      t[3]    = {}; // up, gate, down
    const char *                             base[3] = {};
    size_t                                   nb[3]   = {};
    int                                      n_exp   = 0;
    size_t                                   eb      = 0;  // bytes of one expert (the three slices)
    bool                                     lend_est = false; // in the stream's lend range (estimate at ready)
    std::unique_ptr<std::atomic<uint8_t>[]>  fl;
    std::vector<int32_t>                     slot;
    std::vector<float>                       cnt;
    std::vector<uint8_t>                     vol;
};

struct res_table {
    const llama_model *     model = nullptr;
    std::vector<res_layer>  L;
    std::vector<int>        by_il;
    int                     n_layer = 0;
    size_t                  page    = 4096;
    bool                    resset  = false;
    bool                    rewarm  = false;
    int                     ssd     = 0;
    size_t                  floor_b = (size_t) 4096 << 20; // LLAMA_FN_L15_WARM_FREE_MIB
    size_t                  rewarm_floor = (size_t) 6 << 30;
    std::atomic<uint64_t>   gen{1};
    std::mutex              mtx;  // SSD planning and the eviction queue (taken before pmtx)
    std::mutex              pmtx; // the hot set's view: slot / cnt / vol of every layer
    std::vector<std::pair<int, int>> evq;  // left VRAM, to touch when idle
    std::vector<std::pair<int, int>> dropq; // to leave the working set at the next quiet moment (no decode for 50 ms)
    std::thread             svc;
    std::atomic<bool>       stop{false};
    std::atomic<bool>       warm_done{false};
    bool                    lend_known = false; // the lend estimate ran (or nothing lends): the service thread retries until then
    // log counters since the last line
    std::atomic<uint64_t>   n_drop{0};
    std::atomic<uint64_t>   b_drop{0};
    std::atomic<uint64_t>   b_evpf{0};
    std::atomic<uint64_t>   b_rewarm{0};
    std::atomic<uint64_t>   n_ssd_mark{0};
    std::atomic<uint64_t>   n_ssd_unmark{0};
    std::atomic<double>     share{-1.0}; // resident share of the streamed set at the last sample
    std::mutex              lmtx;        // the log line's interval state
    uint64_t                dio_b0 = 0;
    uint64_t                dio_r0 = 0;
};

std::atomic<res_table *> g_res{nullptr};
std::atomic<int>         g_res_readers{0};

// a reader of the table: the deleter waits until no reader holds it
struct res_ref {
    res_table * t       = nullptr;
    bool        counted = false;
    explicit res_ref(const llama_model * m) {
        if (g_res.load(std::memory_order_relaxed) == nullptr) {
            return; // no table: no shared counter traffic for the callers' threads
        }
        g_res_readers.fetch_add(1);
        counted = true;
        t = g_res.load();
        if (t != nullptr && t->model != m) {
            t = nullptr;
        }
    }
    ~res_ref() {
        if (counted) {
            g_res_readers.fetch_sub(1);
        }
    }
    res_ref(const res_ref &) = delete;
    res_ref & operator=(const res_ref &) = delete;
};

const res_layer * res_find(const res_table * t, int il) {
    if (il < 0 || il >= (int) t->by_il.size() || t->by_il[il] < 0) {
        return nullptr;
    }
    return &t->L[t->by_il[il]];
}

// the flag byte changed by fn (a CAS loop: the planner's PIN marks and the poll's VRAM bits never overwrite each other)
template <typename F>
uint8_t res_update(std::atomic<uint8_t> & a, F fn) {
    uint8_t f = a.load();
    uint8_t nf;
    do {
        nf = fn(f);
    } while (nf != f && !a.compare_exchange_weak(f, nf));
    return f;
}

size_t res_slice_bytes(const res_layer & L) {
    return L.nb[0] + L.nb[1] + L.nb[2];
}

#ifdef _WIN32

size_t fn_avail() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return (size_t) ms.ullAvailPhys;
}

// one Valid bit per address (QueryWorkingSetEx in chunks)
void fn_sample(const std::vector<uintptr_t> & a, std::vector<uint8_t> & valid) {
    valid.assign(a.size(), 0);
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q;
    const size_t chunk = 65536;
    for (size_t i0 = 0; i0 < a.size(); i0 += chunk) {
        const size_t i1 = std::min(a.size(), i0 + chunk);
        q.assign(i1 - i0, PSAPI_WORKING_SET_EX_INFORMATION{});
        for (size_t i = i0; i < i1; ++i) {
            q[i - i0].VirtualAddress = (PVOID) a[i];
        }
        if (!K32QueryWorkingSetEx(GetCurrentProcess(), q.data(), (DWORD) (q.size()*sizeof(q[0])))) {
            continue; // unknown: counted as missing
        }
        for (size_t i = i0; i < i1; ++i) {
            valid[i] = q[i - i0].VirtualAttributes.Valid ? 1 : 0;
        }
    }
}

using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
prefetch_t fn_prefetch_proc() {
    static const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    return pf;
}

void fn_prefetch(const std::vector<range> & rs) {
    const prefetch_t pf = fn_prefetch_proc();
    if (pf == nullptr || rs.empty()) {
        return;
    }
    std::vector<WIN32_MEMORY_RANGE_ENTRY> e;
    for (const auto & r : rs) {
        if (r.size() > 0) {
            e.push_back({ (PVOID) r.lo, (SIZE_T) r.size() });
        }
        if (e.size() == 4096) {
            pf(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
            e.clear();
        }
    }
    if (!e.empty()) {
        pf(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
    }
}

#endif

// per-expert resident share (0..1): the first page of each slice and one page per 2 MiB inside it
void res_expert_share(const res_table * t, const std::vector<std::pair<int, int>> & ex, std::vector<float> & share) {
    share.assign(ex.size(), 0.0f);
#ifdef _WIN32
    std::vector<uintptr_t> a;
    std::vector<uint32_t>  cnt(ex.size(), 0);
    for (size_t i = 0; i < ex.size(); ++i) {
        const res_layer & L = t->L[ex[i].first];
        for (int k = 0; k < 3; ++k) {
            const range r = align_out({ (uintptr_t) L.base[k] + (uintptr_t) ex[i].second*L.nb[k],
                                        (uintptr_t) L.base[k] + (uintptr_t) (ex[i].second + 1)*L.nb[k] }, t->page);
            for (uintptr_t p = r.lo; p < r.hi; p += BLOCK) {
                a.push_back(p);
                cnt[i]++;
            }
        }
    }
    std::vector<uint8_t> v;
    fn_sample(a, v);
    size_t j = 0;
    for (size_t i = 0; i < ex.size(); ++i) {
        uint32_t ok = 0;
        for (uint32_t k = 0; k < cnt[i]; ++k) {
            ok += v[j++];
        }
        share[i] = cnt[i] ? (float) ok/cnt[i] : 0.0f;
    }
#else
    GGML_UNUSED(t);
#endif
}

// the pages of these experts leave the working set for the standby list (VirtualUnlock of unlocked pages); whole pages
// inside runs of consecutive listed experts only, so a neighbour's shared edge page stays. Returns the bytes
size_t res_drop(res_table * t, const std::vector<std::pair<int, int>> & ex) {
#ifdef _WIN32
    if (ex.empty()) {
        return 0;
    }
    std::vector<std::vector<uint8_t>> sel(t->L.size());
    for (const auto & x : ex) {
        auto & s = sel[x.first];
        if (s.empty()) {
            s.assign((size_t) t->L[x.first].n_exp, 0);
        }
        s[x.second] = 1;
    }
    size_t b = 0;
    std::vector<range> rs;
    for (size_t li = 0; li < t->L.size(); ++li) {
        if (sel[li].empty()) {
            continue;
        }
        const res_layer & L = t->L[li];
        for (int k = 0; k < 3; ++k) {
            rs.clear();
            runs_of((uintptr_t) L.base[k], L.nb[k], sel[li].data(), L.n_exp, rs);
            for (const auto & r : rs) {
                const range in = align_in(r, t->page);
                if (in.size() > 0) {
                    VirtualUnlock((LPVOID) in.lo, in.size()); // ERROR_NOT_LOCKED: the pages left the working set
                    b += in.size();
                }
            }
        }
    }
    t->n_drop.fetch_add(ex.size());
    t->b_drop.fetch_add(b);
    return b;
#else
    GGML_UNUSED(t);
    GGML_UNUSED(ex);
    return 0;
#endif
}

// the VRAM-held experts that may leave RAM (not pinned), and the SSD-marked ones out of every slot
std::vector<std::pair<int, int>> res_droppable(const res_table * t, bool vram, bool ssd) {
    std::vector<std::pair<int, int>> out;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        const res_layer & L = t->L[li];
        for (int e = 0; e < L.n_exp; ++e) {
            const uint8_t f = L.fl[e].load();
            if (f & F_PIN) {
                continue;
            }
            if ((vram && (f & F_VRAM)) || (ssd && (f & F_SSD) && !(f & F_SLOT))) {
                out.push_back({ li, e });
            }
        }
    }
    return out;
}

// a drop removes pages from the working set, and every processor that runs a thread of this process flushes its TLB:
// the service thread does it only while no decode or prompt runs (queued here)
void res_queue_drop(res_table * t, const std::vector<std::pair<int, int>> & ex) {
    if (ex.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lk(t->mtx);
    t->dropq.insert(t->dropq.end(), ex.begin(), ex.end());
    if (t->dropq.size() > 65536) { // a long answer: each expert once
        std::sort(t->dropq.begin(), t->dropq.end());
        t->dropq.erase(std::unique(t->dropq.begin(), t->dropq.end()), t->dropq.end());
    }
}

// no decode or prompt scope runs now nor ran in the last 50 ms
bool res_quiet() {
    int64_t act = 0;
    return !llama_fn_p5_busy(&act) && llama_fn_p5_now_us() - act >= 50000;
}

// [TAG_FN_L22_RESSET] the layers in the prefill stream's lend range, estimated as llama_moe_hot_lend places it
// (llama-moecache.cpp) for the banks pfs_take asks for (llama-prefill-stream.cpp): before the first lend the hot set's
// view does not mark them yet, and their residents would look VRAM-held at ready
bool res_lend_estimate(res_table * t) {
    const char * le = getenv("LLAMA_PREFILL_STREAM_LEND");
    if (le == nullptr || atoi(le) == 0) {
        return true;
    }
    const char * nbs = getenv("LLAMA_PREFILL_STREAM_BUFS");
    const int n_bufs = std::max(1, std::min(3, nbs && nbs[0] ? atoi(nbs) : 2));
    ggml_backend_buffer_t dbuf = nullptr;
    std::vector<const llama_moe_cache_layer *> cl(t->L.size(), nullptr);
    for (size_t li = 0; li < t->L.size(); ++li) {
        const llama_moe_cache_layer * c = llama_moe_cache_lookup_table(t->L[li].t[0]);
        if (c == nullptr || !c->up_c || !c->gate_c || !c->down_c) {
            continue;
        }
        if (dbuf == nullptr) {
            dbuf = c->up_c->buffer;
        }
        if (c->up_c->buffer != dbuf || c->gate_c->buffer != dbuf || c->down_c->buffer != dbuf) {
            return true; // not one device buffer: the stream does not lend
        }
        cl[li] = c;
    }
    if (dbuf == nullptr) {
        return false; // no hot set yet
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(dbuf);
    // the stream sizes its banks with the buffer type of the first host layer's router (llama-moe-gen5.cpp: d.buft)
    const ggml_tensor *        router = t->model->layers[t->L[0].il].ffn_gate_inp;
    ggml_backend_buffer_type_t sbuft  = router && router->buffer ? ggml_backend_buffer_get_type(router->buffer) : buft;
    const size_t balign = std::max<size_t>(256, ggml_backend_buft_get_alignment(sbuft));
    size_t mx[3] = { 0, 0, 0 };
    for (const auto & L : t->L) {
        for (int k = 0; k < 3; ++k) {
            mx[k] = std::max(mx[k], ggml_backend_buft_get_alloc_size(sbuft, L.t[k]));
        }
    }
    const size_t bank  = align_up(mx[0], balign) + align_up(mx[1], balign) + align_up(mx[2], balign);
    const size_t size  = ggml_backend_buffer_get_size(dbuf);
    const size_t align = std::max<size_t>(256, ggml_backend_buffer_get_alignment(dbuf));
    size_t lo = 0;
    size_t hi = 0;
    if (!lend_range(size, llama_moe_hot_cbuf_tail_bytes(), (size_t) n_bufs*bank, align, &lo, &hi)) {
        return true;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(dbuf);
    int n = 0;
    for (size_t li = 0; li < t->L.size(); ++li) {
        if (cl[li] == nullptr) {
            continue;
        }
        for (const ggml_tensor * x : { cl[li]->up_c, cl[li]->gate_c, cl[li]->down_c }) {
            const uintptr_t a = (uintptr_t) x->data;
            if (overlaps(a, a + ggml_backend_buft_get_alloc_size(buft, x), base + lo, base + hi)) {
                t->L[li].lend_est = true;
            }
        }
        n += t->L[li].lend_est ? 1 : 0;
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] %d layers lie in the prefill stream's lend range (%.0f MiB, estimate): their "
            "residents count as streamed\n", __func__, n, (hi - lo)/1048576.0);
    return true;
}

// one layer's view from the hot set; entered / left: experts that came into / left a VRAM-held slot. false: unchanged
bool res_poll_layer(res_table * t, int li, std::vector<std::pair<int, int>> * entered, std::vector<std::pair<int, int>> * left) {
    res_layer & L = t->L[li];
    std::vector<int32_t> s((size_t) L.n_exp, -1);
    std::vector<float>   c((size_t) L.n_exp, 0.0f);
    std::vector<uint8_t> v((size_t) L.n_exp, 0);
    const int r = llama_moe_hot_tier_view(L.t[0], s.data(), c.data(), v.data(), L.n_exp);
    if (r == 0) {
        return false; // busy right now: the last view stays
    }
    if (r < 0) {
        std::fill(s.begin(), s.end(), -1); // no hot set (yet): every expert is the host's
        std::fill(v.begin(), v.end(), 0);
    }
    bool changed = false;
    for (int e = 0; e < L.n_exp; ++e) {
        const bool in_slot = s[e] >= 0;
        const bool vram    = in_slot && !v[e] && !L.lend_est;
        const uint8_t f = res_update(L.fl[e], [&](uint8_t x) {
            uint8_t n = (uint8_t) (x & ~(F_VRAM | F_SLOT));
            n |= in_slot ? F_SLOT : 0;
            n |= vram ? F_VRAM : 0;
            if (in_slot) {
                n &= (uint8_t) ~F_SSD; // a hot expert is never the least routed
            }
            return n;
        });
        const bool was = (f & F_VRAM) != 0;
        if (vram && !was && entered) {
            entered->push_back({ li, e });
        }
        if (!vram && was && left) {
            left->push_back({ li, e });
        }
        changed = changed || was != vram || ((f & F_SLOT) != 0) != in_slot || (in_slot && (f & F_SSD));
    }
    L.slot.swap(s);
    L.cnt.swap(c);
    L.vol.swap(v);
    return changed;
}

bool res_poll(res_table * t, std::vector<std::pair<int, int>> * entered, std::vector<std::pair<int, int>> * left) {
    std::lock_guard<std::mutex> lk(t->pmtx);
    bool changed = false;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        changed = res_poll_layer(t, li, entered, left) || changed;
    }
    if (changed) {
        t->gen.fetch_add(1);
    }
    return changed;
}

// experts of the streamed set (not VRAM-held, not pinned); with_ssd: the SSD-marked ones too
std::vector<std::pair<int, int>> res_streamed(const res_table * t, bool with_ssd) {
    std::vector<std::pair<int, int>> out;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        const res_layer & L = t->L[li];
        for (int e = 0; e < L.n_exp; ++e) {
            const uint8_t f = L.fl[e].load();
            if ((f & (F_VRAM | F_PIN)) || (!with_ssd && (f & F_SSD))) {
                continue;
            }
            out.push_back({ li, e });
        }
    }
    return out;
}

// [TAG_FN_L22_SSDTIER] SSD marks for the least-routed cold experts of the streamed set until it fits in RAM.
// stage 0 (ready, before the warm pass): the whole set against resident + room, resident picks go to standby;
// stage 1 (after the warm pass, and every 60 s when idle): the non-resident part against the room; stage 1 also gives
// marks back (most routed first) when more than 1 GiB of room is left. Under t->mtx
void res_ssd_plan(res_table * t, int stage) {
#ifdef _WIN32
    if (t->ssd < 1) {
        return;
    }
    std::lock_guard<std::mutex> lk(t->mtx);
    std::lock_guard<std::mutex> plk(t->pmtx); // the counts
    const auto S = res_streamed(t, stage == 0);
    std::vector<float> sh;
    res_expert_share(t, S, sh);
    size_t need = 0;
    size_t res  = 0;
    for (size_t i = 0; i < S.size(); ++i) {
        const size_t b = t->L[S[i].first].eb;
        need += b;
        res  += (size_t) (b*sh[i]);
    }
    // pinned experts not in RAM yet (the driver brings them in as it registers them)
    std::vector<std::pair<int, int>> P;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        for (int e = 0; e < t->L[li].n_exp; ++e) {
            if (t->L[li].fl[e].load() & F_PIN) {
                P.push_back({ li, e });
            }
        }
    }
    std::vector<float> psh;
    res_expert_share(t, P, psh);
    size_t pin_pending = 0;
    for (size_t i = 0; i < P.size(); ++i) {
        pin_pending += (size_t) (t->L[P[i].first].eb*(1.0f - psh[i]));
    }
    const size_t avail = fn_avail();
    std::vector<ssd_cand> c;
    std::vector<size_t>   ci; // index into S
    size_t deficit = 0;
    if (stage == 0) {
        deficit = ssd_deficit(need, res, avail, t->floor_b, pin_pending);
    } else {
        const size_t nonres = need - std::min(need, res);
        const size_t room   = avail > t->floor_b + pin_pending ? avail - t->floor_b - pin_pending : 0;
        deficit = nonres > room ? nonres - room : 0;
        if (deficit == 0 && room > nonres + GIB) {
            // marks back while the room allows (the idle re-warm reads them)
            std::vector<ssd_cand> m;
            std::vector<std::pair<int, int>> mx;
            for (int li = 0; li < (int) t->L.size(); ++li) {
                const res_layer & L = t->L[li];
                for (int e = 0; e < L.n_exp; ++e) {
                    const uint8_t f = L.fl[e].load();
                    if ((f & F_SSD) && !(f & (F_PIN | F_SLOT))) {
                        m.push_back({ li, e, e < (int) L.cnt.size() ? L.cnt[e] : 0.0f, L.eb, false });
                        mx.push_back({ li, e });
                    }
                }
            }
            const auto un = ssd_unpick(m, room - nonres - GIB);
            for (size_t i : un) {
                res_update(t->L[mx[i].first].fl[mx[i].second], [](uint8_t x) { return (uint8_t) (x & ~F_SSD); });
            }
            if (!un.empty()) {
                t->n_ssd_unmark.fetch_add(un.size());
                t->gen.fetch_add(1);
            }
            return;
        }
    }
    if (deficit == 0) {
        return;
    }
    for (size_t i = 0; i < S.size(); ++i) {
        const res_layer & L = t->L[S[i].first];
        const int e = S[i].second;
        const uint8_t f = L.fl[e].load();
        if ((f & (F_SLOT | F_SSD | F_PIN)) || (stage == 1 && sh[i] >= 0.5f)) {
            continue; // only cold experts; after the warm pass only the ones not in RAM
        }
        c.push_back({ S[i].first, e, e < (int) L.cnt.size() ? L.cnt[e] : 0.0f, L.eb, sh[i] >= 0.5f });
        ci.push_back(i);
    }
    const auto pick = ssd_pick(c, deficit, true);
    std::vector<std::pair<int, int>> drop;
    size_t marked = 0;
    for (size_t k : pick) {
        const auto & x = S[ci[k]];
        bool set = false;
        res_update(t->L[x.first].fl[x.second], [&](uint8_t f) {
            set = !(f & (F_SLOT | F_PIN));
            return set ? (uint8_t) (f | F_SSD) : f;
        });
        if (set) {
            marked += t->L[x.first].eb;
            if (c[k].resident) {
                drop.push_back(x);
            }
        }
    }
    if (marked > 0) {
        t->n_ssd_mark.fetch_add(pick.size());
        t->gen.fetch_add(1);
    }
    size_t dropped = 0;
    for (const auto & x : drop) {
        dropped += t->L[x.first].eb;
    }
    if (res_quiet()) {
        res_drop(t, drop); // now: the room it frees counts for the warm pass and the next plan
    } else {
        t->dropq.insert(t->dropq.end(), drop.begin(), drop.end()); // t->mtx is held
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_SSDTIER] %s: the streamed set needs %.2f GiB, %.2f GiB in RAM, %.2f GiB available (floor %.1f, "
            "pins pending %.2f): %.2f GiB of %zu least-routed cold experts are read by direct I/O in prompts (%.2f GiB of them "
            "leave the working set)\n", __func__, stage == 0 ? "before the warm pass" : "what does not fit", need/1073741824.0,
            res/1073741824.0, avail/1073741824.0, t->floor_b/1073741824.0, pin_pending/1073741824.0, marked/1073741824.0,
            pick.size(), dropped/1073741824.0);
#else
    GGML_UNUSED(t);
    GGML_UNUSED(stage);
#endif
}

// [TAG_FN_L22_REWARM] the streamed set's missing 2 MiB blocks, in stream order (layer order); *share: resident share
void res_missing_blocks(res_table * t, std::vector<range> & miss, double * share) {
    miss.clear();
#ifdef _WIN32
    std::vector<range> runs;
    std::vector<uint8_t> sel;
    for (const auto & L : t->L) {
        sel.assign((size_t) L.n_exp, 0);
        for (int e = 0; e < L.n_exp; ++e) {
            const uint8_t f = L.fl[e].load();
            sel[e] = (f & (F_VRAM | F_PIN | F_SSD)) ? 0 : 1;
        }
        for (int k = 0; k < 3; ++k) {
            const size_t r0 = runs.size();
            runs_of((uintptr_t) L.base[k], L.nb[k], sel.data(), L.n_exp, runs);
            for (size_t i = r0; i < runs.size(); ++i) {
                runs[i] = align_out(runs[i], t->page);
            }
        }
    }
    std::vector<range>     blocks;
    std::vector<uintptr_t> a;
    for (const auto & r : runs) {
        for (uintptr_t p = r.lo; p < r.hi; p += BLOCK) {
            blocks.push_back({ p, std::min<uintptr_t>(r.hi, p + BLOCK) });
            a.push_back(p);
        }
    }
    std::vector<uint8_t> v;
    fn_sample(a, v);
    size_t tot = 0;
    size_t ok  = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        tot += blocks[i].size();
        if (v[i]) {
            ok += blocks[i].size();
        } else {
            miss.push_back(blocks[i]);
        }
    }
    *share = tot ? (double) ok/tot : 1.0;
#else
    GGML_UNUSED(t);
    *share = 1.0;
#endif
}

// touch these ranges at low priority until activity: false when it stopped early (*done: bytes touched)
bool res_touch(const std::vector<range> & rs, size_t page, int64_t act0, size_t * done) {
#ifdef _WIN32
    *done = 0;
    for (const auto & r : rs) {
        for (uintptr_t p = r.lo; p < r.hi; p += BLOCK) {
            int64_t act = 0;
            if (llama_fn_p5_busy(&act) || act != act0) {
                return false;
            }
            const uintptr_t q = std::min<uintptr_t>(r.hi, p + BLOCK);
            for (uintptr_t x = p; x < q; x += page) {
                (void) *(volatile const char *) x;
            }
            *done += (size_t) (q - p);
        }
    }
    return true;
#else
    GGML_UNUSED(rs);
    GGML_UNUSED(page);
    GGML_UNUSED(act0);
    *done = 0;
    return true;
#endif
}

std::vector<range> res_expert_ranges(const res_table * t, const std::vector<std::pair<int, int>> & ex) {
    std::vector<range> rs;
    for (const auto & x : ex) {
        const res_layer & L = t->L[x.first];
        for (int k = 0; k < 3; ++k) {
            rs.push_back(align_out({ (uintptr_t) L.base[k] + (uintptr_t) x.second*L.nb[k],
                                     (uintptr_t) L.base[k] + (uintptr_t) (x.second + 1)*L.nb[k] }, t->page));
        }
    }
    return rs;
}

void res_log(res_table * t, const char * when) {
    if (llama_fn_l22_passlog_level() < 1) {
        return;
    }
    std::lock_guard<std::mutex> lk(t->lmtx);
    size_t cls[4] = { 0, 0, 0, 0 };
    for (const auto & L : t->L) {
        for (int e = 0; e < L.n_exp; ++e) {
            cls[class_of(L.fl[e].load())] += L.eb;
        }
    }
    uint64_t db = 0;
    uint64_t dr = 0;
    uint64_t df = 0;
    llama_fn_p5_dio_stats(&db, &dr, &df);
#ifdef _WIN32
    const size_t avail = fn_avail();
#else
    const size_t avail = 0;
#endif
    llama_fn_l22_log("res", "when=%s avail_gib=%.2f streamed_gib=%.2f resident_share=%.3f vram_gib=%.2f pin_gib=%.2f ssd_gib=%.2f "
            "drops=%" PRIu64 " drop_gib=%.2f evict_pf_gib=%.2f rewarm_gib=%.2f ssd_marks=%" PRIu64 " ssd_unmarks=%" PRIu64
            " dio_gib=%.2f dio_reads=%" PRIu64 " dio_fails=%" PRIu64 " gen=%" PRIu64, when, avail/1073741824.0,
            (cls[0] + cls[3])/1073741824.0, t->share.load(), cls[1]/1073741824.0, cls[2]/1073741824.0, cls[3]/1073741824.0,
            t->n_drop.exchange(0), t->b_drop.exchange(0)/1073741824.0, t->b_evpf.exchange(0)/1073741824.0,
            t->b_rewarm.exchange(0)/1073741824.0, t->n_ssd_mark.exchange(0), t->n_ssd_unmark.exchange(0),
            (db - t->dio_b0)/1073741824.0, dr - t->dio_r0, df, t->gen.load());
    t->dio_b0 = db;
    t->dio_r0 = dr;
}

// the service thread: the hot set's view every second (drops, eviction prefetch), the idle work (eviction touch,
// re-warm, the 60 s re-drop and SSD plan), the residency log every 60 s
void res_service(res_table * t) {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    llama_fn_thread_name("res");
    int64_t t_poll = 0;
    int64_t t_log  = llama_fn_p5_now_us();
    int64_t t_per  = t_log;
    int64_t t_samp = 0;
    int64_t act_samp = -1;
    uint64_t gen_samp = 0;
    std::vector<range> miss;
    size_t miss_at = 0;
    std::vector<std::pair<int, int>> entered;
    std::vector<std::pair<int, int>> left;
    while (!t->stop.load()) {
        Sleep(100);
        const int64_t now = llama_fn_p5_now_us();
        if (now - t_poll >= 1000000) {
            t_poll = now;
            entered.clear();
            left.clear();
            if (!t->lend_known) {
                t->lend_known = res_lend_estimate(t); // the hot set came after the table
            }
            res_poll(t, &entered, &left);
            if (t->resset) {
                // drop the host copies of experts that came into VRAM (queued); prefetch the ones that left it
                res_queue_drop(t, entered);
                if (!left.empty()) {
                    const auto rs = res_expert_ranges(t, left);
                    fn_prefetch(rs);
                    std::lock_guard<std::mutex> lk(t->mtx);
                    t->evq.insert(t->evq.end(), left.begin(), left.end());
                    if (t->evq.size() > 65536) { // never idle for long: the oldest go (they were prefetched)
                        t->evq.erase(t->evq.begin(), t->evq.begin() + (t->evq.size() - 32768));
                    }
                }
            }
        }
        if (now - t_log >= 60000000) {
            t_log = now;
            res_log(t, "60s");
        }
        int64_t act = 0;
        const bool busy = llama_fn_p5_busy(&act);
        if (!busy && now - act >= 50000) {
            std::vector<std::pair<int, int>> q;
            {
                std::lock_guard<std::mutex> lk(t->mtx);
                q.swap(t->dropq);
            }
            // pinned meanwhile, or no longer VRAM-held / SSD (the eviction prefetch wants it back): skip
            q.erase(std::remove_if(q.begin(), q.end(), [&](const std::pair<int, int> & x) {
                const uint8_t f = t->L[x.first].fl[x.second].load();
                return (f & F_PIN) || !(f & (F_VRAM | F_SSD));
            }), q.end());
            res_drop(t, q);
        }
        const bool idle = !busy && now - act >= 1000000 && !llama_fn_warm_running() && t->warm_done.load();
        if (!idle) {
            continue;
        }
        // eviction touches first (small)
        if (t->resset) {
            std::vector<std::pair<int, int>> q;
            {
                std::lock_guard<std::mutex> lk(t->mtx);
                q.swap(t->evq);
            }
            if (!q.empty()) {
                size_t done = 0;
                const auto rs = res_expert_ranges(t, q);
                if (!res_touch(rs, t->page, act, &done)) {
                    std::lock_guard<std::mutex> lk(t->mtx); // stopped by activity: the rest later
                    t->evq.insert(t->evq.begin(), q.begin(), q.end());
                }
                t->b_evpf.fetch_add(done);
            }
        }
        if (t->rewarm) {
            // a new sample at each idle period, after a table change, and every 10 s of a long one
            if (act != act_samp || gen_samp != t->gen.load() || now - t_samp >= 10000000) {
                double sh = 1.0;
                res_missing_blocks(t, miss, &sh);
                t->share.store(sh);
                miss_at  = 0;
                t_samp   = now;
                act_samp = act;
                gen_samp = t->gen.load();
            }
            if (miss_at < miss.size()) {
                const size_t avail = fn_avail();
                const size_t room = avail > t->rewarm_floor ? avail - t->rewarm_floor : 0;
                // a window of up to 256 MiB per tick: one prefetch for all of it (the reads overlap), then the touches in
                // order, for at most 500 ms, stopped by any activity
                std::vector<range> win;
                size_t b = 0;
                while (miss_at + win.size() < miss.size() && b < ((size_t) 256 << 20) &&
                       b + miss[miss_at + win.size()].size() <= room) {
                    b += miss[miss_at + win.size()].size();
                    win.push_back(miss[miss_at + win.size()]);
                }
                if (win.empty()) {
                    miss_at = miss.size(); // no room: nothing more this period
                } else {
                    fn_prefetch(win);
                    const int64_t t_end = now + 500000;
                    size_t i = 0;
                    for (; i < win.size() && llama_fn_p5_now_us() < t_end && !t->stop.load(); ++i) {
                        size_t done = 0;
                        const bool all = res_touch({ win[i] }, t->page, act, &done);
                        t->b_rewarm.fetch_add(done);
                        if (!all) {
                            break; // activity: this block again at the next idle sample
                        }
                    }
                    miss_at += i;
                }
            }
        }
        if (now - t_per >= 60000000) {
            t_per = now;
            if (t->resset) {
                res_queue_drop(t, res_droppable(t, true, false)); // pages read back since (a refill, a CPU path)
            }
            res_ssd_plan(t, 1);
        }
    }
#else
    GGML_UNUSED(t);
#endif
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// the residency interface (llama-fn-l22.h)

uint8_t llama_fn_res_class(const llama_model * model, int il, int e) {
    res_ref r(model);
    if (r.t == nullptr) {
        return LLAMA_FN_RES_RAM;
    }
    const res_layer * L = res_find(r.t, il);
    if (L == nullptr || e < 0 || e >= L->n_exp) {
        return LLAMA_FN_RES_RAM;
    }
    return class_of(L->fl[e].load(std::memory_order_relaxed));
}

uint64_t llama_fn_res_gen(const llama_model * model) {
    res_ref r(model);
    return r.t ? r.t->gen.load() : 0;
}

void llama_fn_res_mark_pin(const llama_model * model, int il, int e, bool on) {
    res_ref r(model);
    if (r.t == nullptr) {
        return;
    }
    const res_layer * L = res_find(r.t, il);
    if (L == nullptr || e < 0 || e >= L->n_exp) {
        return;
    }
    const uint8_t f = res_update(L->fl[e], [on](uint8_t x) {
        return on ? (uint8_t) ((x | F_PIN) & ~F_SSD) : (uint8_t) (x & ~F_PIN);
    });
    if (((f & F_PIN) != 0) != on) {
        r.t->gen.fetch_add(1);
    }
}

uint64_t llama_fn_res_stream_bytes(const llama_model * model, uint8_t cls) {
    res_ref r(model);
    if (r.t == nullptr) {
        return 0;
    }
    uint64_t b = 0;
    for (const auto & L : r.t->L) {
        for (int e = 0; e < L.n_exp; ++e) {
            b += class_of(L.fl[e].load(std::memory_order_relaxed)) == cls ? L.eb : 0;
        }
    }
    return b;
}

// ---------------------------------------------------------------------------------------------------------------------
// P5 internal (llama-fn-l22-p5.h)

bool llama_fn_p5_ready(const llama_model & model) {
#ifdef _WIN32
    if (model.arch != LLM_ARCH_QWEN4EXP) {
        return false;
    }
    const bool resset = llama_fn_l3_int(model, "LLAMA_FN_L22_RESSET", 0) != 0;
    const bool rewarm = llama_fn_l3_int(model, "LLAMA_FN_L22_REWARM", 0) != 0;
    const int  ssd    = std::max(0, std::min(2, llama_fn_l3_int(model, "LLAMA_FN_L22_SSDTIER", 0)));
    const bool pin    = llama_fn_l3_int(model, "LLAMA_FN_L22_PIN", 0) != 0;
    if (!resset && !rewarm && ssd == 0 && !pin) {
        return false;
    }
    if (g_res.load() != nullptr) {
        return g_res.load()->model == &model;
    }
    auto * t = new res_table();
    t->model  = &model;
    t->resset = resset;
    t->rewarm = rewarm;
    t->ssd    = ssd;
    t->floor_b = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L15_WARM_FREE_MIB", 4096)) << 20;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    t->page = si.dwPageSize ? si.dwPageSize : 4096;
    t->n_layer = (int) std::min<size_t>(model.hparams.n_layer(), model.layers.size());
    t->by_il.assign((size_t) t->n_layer, -1);
    for (int il = 0; il < t->n_layer; ++il) {
        const auto & ML = model.layers[il];
        const ggml_tensor * ts[3] = { ML.ffn_up_exps, ML.ffn_gate_exps, ML.ffn_down_exps };
        res_layer L;
        bool ok = true;
        for (int k = 0; k < 3 && ok; ++k) {
            const ggml_tensor * x = ts[k];
            ok = x && x->buffer && ggml_backend_buffer_is_host(x->buffer) && x->data && x->ne[2] >= 1 &&
                 (k == 0 || x->ne[2] == L.n_exp) && x->nb[2]*x->ne[2] == ggml_nbytes(x);
            if (ok) {
                L.t[k]    = x;
                L.base[k] = (const char *) x->data;
                L.nb[k]   = x->nb[2];
                L.n_exp   = (int) x->ne[2];
            }
        }
        if (!ok) {
            continue;
        }
        L.il = il;
        L.eb = res_slice_bytes(L);
        L.fl.reset(new std::atomic<uint8_t>[(size_t) L.n_exp]());
        L.slot.assign((size_t) L.n_exp, -1);
        L.cnt.assign((size_t) L.n_exp, 0.0f);
        L.vol.assign((size_t) L.n_exp, 0);
        t->by_il[il] = (int) t->L.size();
        t->L.push_back(std::move(L));
    }
    if (t->L.empty()) {
        delete t;
        return false;
    }
    t->lend_known = res_lend_estimate(t);
    // the first view (the hot set's bookkeeping may be busy for a moment)
    for (int li = 0; li < (int) t->L.size(); ++li) {
        for (int k = 0; k < 20; ++k) {
            uint8_t probe = 0;
            int32_t s = -1;
            float   c = 0.0f;
            if (llama_moe_hot_tier_view(t->L[li].t[0], &s, &c, &probe, 0) != 0) {
                break;
            }
            Sleep(10);
        }
        res_poll_layer(t, li, nullptr, nullptr);
    }
    g_res.store(t);
    llama_fn_p5_track(true);
    size_t cls[4] = { 0, 0, 0, 0 };
    for (const auto & L : t->L) {
        for (int e = 0; e < L.n_exp; ++e) {
            cls[class_of(L.fl[e].load())] += L.eb;
        }
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] residency table: %zu layers, %.2f GiB streamed from the host, %.2f GiB held in VRAM "
            "(resset %d, rewarm %d, ssdtier %d, pin %d)\n", __func__, t->L.size(), cls[0]/1073741824.0, cls[1]/1073741824.0,
            resset ? 1 : 0, rewarm ? 1 : 0, ssd, pin ? 1 : 0);
    // the pin planner starts (its own thread): it marks its experts PIN, the warm pass skips them
    llama_fn_pin_on_ready(&model);
    // R-P6-3: with pins planned (PIN-marked experts, or pins already active) the warm, SSD and re-warm floors stay at least
    // LLAMA_FN_L22_PIN_FLOOR_MIB + 1 GiB of available RAM: the pin valve unpins below its floor and never registers again
    if (pin) {
        bool marked = llama_fn_pin_active();
        for (size_t li = 0; li < t->L.size() && !marked; ++li) {
            for (int e = 0; e < t->L[li].n_exp && !marked; ++e) {
                marked = (t->L[li].fl[e].load() & F_PIN) != 0;
            }
        }
        if (marked) {
            const size_t pf = ((size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L22_PIN_FLOOR_MIB", 6144)) << 20) + GIB;
            t->floor_b      = std::max(t->floor_b, pf);
            t->rewarm_floor = std::max(t->rewarm_floor, pf);
            LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] pins planned: warm / SSD floor %.1f GiB, re-warm floor %.1f GiB\n", __func__,
                    t->floor_b/1073741824.0, t->rewarm_floor/1073741824.0);
        }
    }
    if (resset) {
        res_ssd_plan(t, 0); // the table-driven warm pass then skips the SSD experts (the old pass reads everything)
    }
    t->svc = std::thread(res_service, t);
    return true;
#else
    GGML_UNUSED(model);
    return false;
#endif
}

bool llama_fn_p5_resset(const llama_model & model) {
    res_ref r(&model);
    return r.t != nullptr && r.t->resset;
}

size_t llama_fn_p5_floor(const llama_model & model) {
    res_ref r(&model);
    return r.t != nullptr ? r.t->floor_b : 0;
}

std::vector<llama_fn_p5_piece> llama_fn_p5_warm_list(const llama_model & model, size_t max_bytes, std::vector<size_t> & layer_end,
        size_t * pin_pending, size_t * skipped) {
    std::vector<llama_fn_p5_piece> out;
    *pin_pending = 0;
    *skipped     = 0;
    res_ref r(&model);
    if (r.t == nullptr) {
        return out;
    }
    res_table * t = r.t;
    layer_end.assign((size_t) t->n_layer, 0);
    std::vector<uint8_t> sel;
    std::vector<std::pair<int, int>> runs;
    std::vector<std::pair<int, int>> P;
    for (int il = 0; il < t->n_layer; ++il) {
        const int li = t->by_il[il];
        if (li >= 0) {
            const res_layer & L = t->L[li];
            sel.assign((size_t) L.n_exp, 0);
            for (int e = 0; e < L.n_exp; ++e) {
                const uint8_t f = L.fl[e].load();
                sel[e] = (f & (F_VRAM | F_PIN | F_SSD)) ? 0 : 1;
                if (!sel[e]) {
                    *skipped += L.eb;
                }
                if (f & F_PIN) {
                    P.push_back({ li, e });
                }
            }
            for (int k = 0; k < 3; ++k) {
                runs.clear();
                piece_runs(sel.data(), L.n_exp, L.nb[k], max_bytes, runs);
                for (const auto & r : runs) {
                    out.push_back({ il, k, r.first, r.second, L.base[k] + (size_t) r.first*L.nb[k], (size_t) (r.second - r.first)*L.nb[k] });
                }
            }
        }
        layer_end[il] = out.size();
    }
    std::vector<float> sh;
    res_expert_share(t, P, sh);
    for (size_t i = 0; i < P.size(); ++i) {
        *pin_pending += (size_t) (t->L[P[i].first].eb*(1.0f - sh[i]));
    }
    return out;
}

size_t llama_fn_p5_piece_ranges(const llama_model & model, const llama_fn_p5_piece & pc, std::vector<std::pair<const char *, size_t>> & out) {
    res_ref r(&model);
    if (r.t == nullptr) {
        out.push_back({ pc.p, pc.n });
        return pc.n;
    }
    const res_layer * L = res_find(r.t, pc.il);
    if (L == nullptr) {
        out.push_back({ pc.p, pc.n });
        return pc.n;
    }
    size_t b = 0;
    for (int e = pc.e0; e < pc.e1; ) {
        if (L->fl[e].load() & (F_VRAM | F_PIN | F_SSD)) {
            ++e;
            continue;
        }
        int e1 = e + 1;
        while (e1 < pc.e1 && !(L->fl[e1].load() & (F_VRAM | F_PIN | F_SSD))) {
            ++e1;
        }
        const size_t n = (size_t) (e1 - e)*L->nb[pc.part];
        out.push_back({ L->base[pc.part] + (size_t) e*L->nb[pc.part], n });
        b += n;
        e = e1;
    }
    return b;
}

void llama_fn_p5_after_warm(const llama_model & model, size_t warmed, double secs) {
    res_ref r(&model);
    if (r.t == nullptr) {
        return;
    }
    res_table * t = r.t;
    size_t dropped = 0;
    if (t->resset) {
        // [TAG_FN_L22_RESSET] the host copies of VRAM-held experts (and of resident SSD-marked ones) to the standby list
        const auto d = res_droppable(t, true, t->ssd > 0);
        for (const auto & x : d) {
            dropped += t->L[x.first].eb;
        }
        if (res_quiet()) {
            res_drop(t, d); // now: the stage-1 plan below sees the room it frees
        } else {
            res_queue_drop(t, d);
        }
    }
    res_ssd_plan(t, 1);
    double share = 1.0;
    std::vector<range> miss;
    res_missing_blocks(t, miss, &share);
    t->share.store(share);
    t->warm_done.store(true);
    if (t->resset) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] ready: %.2f GiB warmed in %.1f s, %.2f GiB of VRAM-held host copies go to standby, "
                "streamed set %.1f %% in RAM\n", __func__, warmed/1073741824.0, secs, dropped/1073741824.0, 100.0*share);
    } else {
        LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] ready after the full warm pass (resset off): streamed set %.1f %% in RAM\n",
                __func__, 100.0*share);
    }
    res_log(t, "ready");
}

void llama_fn_p5_free(const llama_model & model) {
    res_table * t = g_res.load();
    if (t != nullptr && t->model == &model) {
        t->stop.store(true);
        if (t->svc.joinable()) {
            t->svc.join();
        }
        llama_fn_p5_track(false);
        g_res.store(nullptr);
        while (g_res_readers.load() != 0) {
            std::this_thread::yield();
        }
        delete t;
    }
    llama_fn_p5_dio_free(model);
}
