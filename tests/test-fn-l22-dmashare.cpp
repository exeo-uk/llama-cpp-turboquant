// [TAG_FN_L22_DMASHARE] round 22 P14: the pure parts of the decode DMA share (src/llama-fn-dmashare-core.h) and the shared
// layout (ggml/include/ggml-fn-l22-dma.h). CPU only, no model, no device.
//
// Checks: the plan on random jobs (served experts never hot or dropped, copies within the caps, victims never this job's
// hits, slot ids inside the layer's range or the zero slot, skip marks = hot + served, hits only from committed slots whose
// bank bytes are the expert's), how many to copy (table cases, monotonic in the rates, the guard), the piece walker on a fake
// pin registry (edges staged byte-exact, rejects give every ticket back, arena copies, adjacent ranges), the kernel's copy
// rule, the commit machine, the levers' drop rule against P8's mark pass and P11's light rule, the layout and the switches.

#include "../src/llama-fn-dmashare-core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
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

static constexpr size_t   PAGE    = 4096;
static constexpr uint64_t DEV_OFF = 1ull << 44; // fake device address = host address + DEV_OFF

// ---------------------------------------------------------------------------------------------------------------------
// a fake pin registry with P6's query semantics

struct fake_range {
    const uint8_t * p;
    size_t          n;
    const uint8_t * copy = nullptr; // arena copy (the device reads it), else p
    bool            has_dev = true;
};

struct fake_reg {
    std::vector<fake_range> r;     // sorted, disjoint
    std::vector<int>        users;
    int                     queries = 0;

    void add(const uint8_t * p, size_t n, const uint8_t * copy = nullptr, bool has_dev = true) {
        r.push_back({ p, n, copy, has_dev });
        users.push_back(0);
    }
    int outstanding() const {
        int s = 0;
        for (int u : users) {
            s += u;
        }
        return s;
    }
    bool query(const void * p0, size_t n, fn_dma::seg_q * out) {
        queries++;
        const uint8_t * p = (const uint8_t *) p0;
        *out = fn_dma::seg_q();
        out->src = p0;
        out->len = n;
        for (size_t i = 0; i < r.size(); ++i) {
            if (p >= r[i].p && p < r[i].p + r[i].n) {
                users[i]++;
                const uint8_t * src = r[i].copy ? r[i].copy + (p - r[i].p) : p;
                out->src    = src;
                out->len    = std::min(n, (size_t) (r[i].p + r[i].n - p));
                out->pinned = true;
                out->ticket = (int32_t) i;
                out->dev    = r[i].has_dev ? (uint64_t) (uintptr_t) src + DEV_OFF : 0;
                return true;
            }
        }
        const uint8_t * end = p + n;
        for (const auto & x : r) {
            if (x.p > p && x.p < end) {
                end = x.p;
            }
        }
        out->len = (size_t) (end - p);
        return true;
    }
    void release(int32_t t) {
        if (t >= 0 && t < (int32_t) users.size()) {
            users[t]--;
        }
    }
};

static const uint8_t * map_dev(uint64_t d) {
    return d >= DEV_OFF ? (const uint8_t *) (uintptr_t) (d - DEV_OFF) : nullptr;
}

// ---------------------------------------------------------------------------------------------------------------------
// 7. layout and switches

static void test_layout_cfg() {
    TCHECK(sizeof(ggml_fn_l22_dma_side) % 64 == 0 && offsetof(ggml_fn_l22_dma_side, part) == 64, "side layout");
    TCHECK(sizeof(ggml_fn_l22_dma_done) == 64, "done layout");
    TCHECK(sizeof(ggml_fn_l22_dma_part) == 80 && sizeof(ggml_fn_l22_dma_seg) == 16, "part / seg layout");
    TCHECK(GGML_FN_L22_DMA_MAX_COPY <= 32, "copies fit the bridge plan (GGML_MOE_BRIDGE_MAX_FETCH)");

    TCHECK(!fn_dma::make_cfg(false, 1, 4).on, "DMASHARE off -> off");
    TCHECK(!fn_dma::make_cfg(true, 0, 4).on, "PIN 0 -> off");
    TCHECK(fn_dma::make_cfg(true, 1, 4).on && fn_dma::make_cfg(true, 2, 4).on, "DMASHARE and PIN -> on");
    TCHECK(fn_dma::make_cfg(true, 1, 0).slots == 1 && fn_dma::make_cfg(true, 1, -3).slots == 1, "slots clamp low");
    TCHECK(fn_dma::make_cfg(true, 1, 9).slots == 8 && fn_dma::make_cfg(true, 1, 100).slots == 8, "slots clamp high");
    TCHECK(fn_dma::make_cfg(true, 1, 4).slots == 4, "slots default");

    // part check
    ggml_fn_l22_dma_part p;
    memset(&p, 0, sizeof(p));
    const uint64_t lo = 1000000, hi = lo + 65536;
    p.n_seg = 3;
    p.seg[0] = { lo + 16, 100, 1 };
    p.seg[1] = { 77777777, 4096, 0 };
    p.seg[2] = { lo + 128, 300, 1 };
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 4496, lo, hi) == 1, "good part");
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 4497, lo, hi) == 0, "size mismatch");
    p.seg[2].src = hi - 100; // runs past the area
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 4496, lo, hi) == 0, "edge past the area");
    p.seg[2].src = lo - 16;
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 4496, lo, hi) == 0, "edge before the area");
    p.seg[2].src = lo + 128;
    p.seg[1].len = 0;
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 4396, lo, hi) == 0, "empty piece");
    p.seg[1].len = 4096;
    p.seg[1].src = 0;
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 4496, lo, hi) == 0, "null source");
    p.seg[1].src = 77777777;
    p.n_seg = 0;
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 0, lo, hi) == 0, "no piece");
    p.n_seg = 5;
    TCHECK(ggml_fn_l22_dma_part_ok(&p, 4496, lo, hi) == 0, "too many pieces");
    p.n_seg = 1;
    p.seg[0] = { lo, GGML_FN_L22_DMA_EDGE_MAX + 16, 1 };
    TCHECK(ggml_fn_l22_dma_part_ok(&p, GGML_FN_L22_DMA_EDGE_MAX + 16, lo, hi) == 0, "staged piece longer than an edge");
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. how many to copy

static void test_best_k() {
    fn_dma::rates r;
    r.cpu_us = 45;
    r.dma_us = 95;
    r.gpu_us = 2;
    r.g      = 1;
    TCHECK(fn_dma::best_k(4, 0, r, 8) == 1, "4 cold, CPU 45 / DMA 95 -> 1 (%d)", fn_dma::best_k(4, 0, r, 8));
    TCHECK(fn_dma::best_k(0, 3, r, 8) == 0, "nothing to copy");
    TCHECK(fn_dma::best_k(10, 0, r, 0) == 0, "no slot");
    r.dma_us = 1e6;
    TCHECK(fn_dma::best_k(10, 0, r, 8) == 0, "DMA very slow -> 0");
    r.dma_us = 1;
    r.cpu_us = 1000;
    TCHECK(fn_dma::best_k(6, 0, r, 8) == 6, "CPU very slow -> all (%d)", fn_dma::best_k(6, 0, r, 8));
    TCHECK(fn_dma::best_k(6, 0, r, 3) == 3, "capped by the slots");
    r.g = 0.5;
    TCHECK(fn_dma::best_k(6, 0, r, 8) == 3, "guard halves (%d)", fn_dma::best_k(6, 0, r, 8));
    r.g = 0.0;
    TCHECK(fn_dma::best_k(6, 0, r, 8) == 0, "guard 0 -> 0");

    // monotonic: more CPU time -> not fewer copies, more DMA time -> not more
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(1.0, 200.0);
    int bad_cpu = 0, bad_dma = 0, bad_cap = 0;
    for (int it = 0; it < 20000; ++it) {
        fn_dma::rates a;
        a.cpu_us = U(rng);
        a.dma_us = U(rng);
        a.gpu_us = U(rng)/50.0;
        a.g      = 1.0;
        const int m    = (int) (rng() % 40);
        const int hits = (int) (rng() % 8);
        const int cap  = (int) (rng() % 9);
        const int k0   = fn_dma::best_k(m, hits, a, cap);
        fn_dma::rates b = a;
        b.cpu_us *= 1.0 + U(rng)/100.0;
        bad_cpu += fn_dma::best_k(m, hits, b, cap) < k0 ? 1 : 0;
        b = a;
        b.dma_us *= 1.0 + U(rng)/100.0;
        bad_dma += fn_dma::best_k(m, hits, b, cap) > k0 ? 1 : 0;
        bad_cap += (k0 < 0 || k0 > std::min(m, cap)) ? 1 : 0;
    }
    TCHECK(bad_cpu == 0, "monotonic in cpu_us (%d)", bad_cpu);
    TCHECK(bad_dma == 0, "monotonic in dma_us (%d)", bad_dma);
    TCHECK(bad_cap == 0, "within [0, min(m, cap)] (%d)", bad_cap);

    TCHECK(fn_dma::guard_step(0.5, -1.0) == 1.0, "no wait info -> 1");
    TCHECK(fn_dma::guard_step(0.5, 30.0) > 0.5 && fn_dma::guard_step(0.5, 1.0) < 0.5 && fn_dma::guard_step(0.5, 10.0) == 0.5, "guard steps");
    TCHECK(fn_dma::guard_step(1.0, 30.0) == 1.0 && fn_dma::guard_step(0.0, 1.0) == 0.0, "guard clamps");
    TCHECK(std::fabs(fn_dma::ewma(10.0, 20.0) - 10.5) < 1e-12, "ewma");
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. + 4. the piece walker and the copy rule

static void fill_random(std::vector<uint8_t> & v, uint32_t seed) {
    std::mt19937 rng(seed);
    for (auto & b : v) {
        b = (uint8_t) rng();
    }
}

static uint8_t * page_align(std::vector<uint8_t> & v) {
    return (uint8_t *) (((uintptr_t) v.data() + PAGE - 1) & ~(uintptr_t) (PAGE - 1));
}

static void test_walker() {
    std::vector<uint8_t> mem(64*PAGE);
    fill_random(mem, 11);
    uint8_t * base = page_align(mem);
    std::vector<uint8_t> edge_mem(64*1024 + 64);
    uint8_t * edge = (uint8_t *) (((uintptr_t) edge_mem.data() + 15) & ~(uintptr_t) 15);
    auto make_ea = [&](size_t cap) {
        fn_dma::edge_area ea;
        ea.host = edge;
        ea.dev  = (uint64_t) (uintptr_t) edge + DEV_OFF;
        ea.cap  = cap;
        ea.used = 0;
        return ea;
    };
    auto copy_ok = [&](const ggml_fn_l22_dma_part & part, const uint8_t * src, size_t n) {
        std::vector<uint8_t> dst(n + 64, 0xCD);
        const size_t got = fn_dma::emulate_copy(part, dst.data(), map_dev);
        return got == n && memcmp(dst.data(), src, n) == 0 && dst[n] == 0xCD;
    };

    // A: the slice inside one pinned range
    {
        fake_reg R;
        R.add(base, 8*PAGE);
        fn_dma::query_fn q = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::edge_area ea = make_ea(64*1024);
        ggml_fn_l22_dma_part part;
        std::vector<int32_t> tk;
        const int rc = fn_dma::walk_slice(base + 100, 3*PAGE, q, ea, &part, tk);
        TCHECK(rc == -1 && part.n_seg == 1 && part.seg[0].staged == 0 && tk.size() == 1 && ea.used == 0, "one pinned piece");
        TCHECK(copy_ok(part, base + 100, 3*PAGE), "A bytes");
        TCHECK(R.outstanding() == 1, "A ticket held");
    }
    // B: head edge, pinned middle, tail edge (P6 aligns ranges inward to pages)
    {
        fake_reg R;
        R.add(base + PAGE, 3*PAGE);
        fn_dma::query_fn q = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::edge_area ea = make_ea(64*1024);
        ggml_fn_l22_dma_part part;
        std::vector<int32_t> tk;
        const uint8_t * s = base + PAGE - 96;
        const size_t    n = 96 + 3*PAGE + 208;
        const int rc = fn_dma::walk_slice(s, n, q, ea, &part, tk);
        TCHECK(rc == -1 && part.n_seg == 3, "edges + middle (rc %d, %u pieces)", rc, part.n_seg);
        TCHECK(part.seg[0].staged == 1 && part.seg[0].len == 96 && part.seg[1].staged == 0 && part.seg[2].staged == 1 &&
               part.seg[2].len == 208, "piece kinds");
        TCHECK(part.seg[2].src % 16 == 0 && part.seg[0].src % 16 == 0, "staged pieces 16-byte aligned");
        TCHECK(copy_ok(part, s, n), "B bytes (staged edges byte-exact)");
        TCHECK(ggml_fn_l22_dma_part_ok(&part, n, ea.dev, ea.dev + ea.cap) == 1, "B passes the device check");
        TCHECK(tk.size() == 1 && R.outstanding() == 1, "B one ticket");
    }
    // C: an unpinned piece longer than an edge -> reject, tickets back
    {
        fake_reg R;
        R.add(base, PAGE);
        fn_dma::query_fn   q   = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::release_fn rel = [&](int32_t t) { R.release(t); };
        fn_dma::edge_area ea = make_ea(64*1024);
        const uint8_t * ps[3] = { base, base + 10*PAGE, base + 20*PAGE };
        const size_t    ns[3] = { PAGE + 3*PAGE, PAGE, PAGE }; // part 0: pinned page, then 3 unpinned pages
        ggml_fn_l22_dma_part parts[3];
        std::vector<int32_t> tk;
        const int rc = fn_dma::resolve_expert(ps, ns, q, rel, ea, parts, tk);
        TCHECK(rc == fn_dma::REJ_UNPINNED, "long unpinned piece (rc %d)", rc);
        TCHECK(tk.empty() && R.outstanding() == 0 && ea.used == 0, "C tickets back, edge area as before");
    }
    // D: an unpinned middle between two pinned ranges
    {
        fake_reg R;
        R.add(base, PAGE);
        R.add(base + 4*PAGE, PAGE);
        fn_dma::query_fn   q   = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::release_fn rel = [&](int32_t t) { R.release(t); };
        fn_dma::edge_area ea = make_ea(64*1024);
        const uint8_t * ps[3] = { base + 30*PAGE, base, base + 40*PAGE };
        const size_t    ns[3] = { 64, 5*PAGE, 64 };
        ggml_fn_l22_dma_part parts[3];
        std::vector<int32_t> tk;
        const int rc = fn_dma::resolve_expert(ps, ns, q, rel, ea, parts, tk);
        TCHECK(rc == fn_dma::REJ_UNPINNED, "unpinned middle (rc %d)", rc);
        TCHECK(tk.empty() && R.outstanding() == 0 && ea.used == 0, "D tickets back and part 0's edge undone");
    }
    // E: five pieces -> reject
    {
        fake_reg R;
        R.add(base + PAGE, PAGE);
        R.add(base + 3*PAGE, PAGE);
        fn_dma::query_fn   q   = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::release_fn rel = [&](int32_t t) { R.release(t); };
        fn_dma::edge_area ea = make_ea(64*1024);
        const uint8_t * ps[3] = { base + PAGE - 100, base + 30*PAGE, base + 40*PAGE };
        const size_t    ns[3] = { 100 + PAGE + PAGE + PAGE + 100, 64, 64 }; // edge, pinned, gap, pinned, edge
        ggml_fn_l22_dma_part parts[3];
        std::vector<int32_t> tk;
        const int rc = fn_dma::resolve_expert(ps, ns, q, rel, ea, parts, tk);
        TCHECK(rc == fn_dma::REJ_SEGS, "five pieces (rc %d)", rc);
        TCHECK(tk.empty() && R.outstanding() == 0, "E tickets back");
    }
    // F: a pinned range without a device address
    {
        fake_reg R;
        R.add(base, 4*PAGE, nullptr, false);
        fn_dma::query_fn   q   = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::release_fn rel = [&](int32_t t) { R.release(t); };
        fn_dma::edge_area ea = make_ea(64*1024);
        const uint8_t * ps[3] = { base, base + PAGE, base + 2*PAGE };
        const size_t    ns[3] = { PAGE, PAGE, PAGE };
        ggml_fn_l22_dma_part parts[3];
        std::vector<int32_t> tk;
        const int rc = fn_dma::resolve_expert(ps, ns, q, rel, ea, parts, tk);
        TCHECK(rc == fn_dma::REJ_DEVADDR && R.outstanding() == 0, "no device address (rc %d)", rc);
    }
    // G: an arena copy (the device reads the copy, not the asked bytes)
    {
        std::vector<uint8_t> arena(8*PAGE + PAGE);
        uint8_t * ab = page_align(arena);
        memcpy(ab, base + 8*PAGE, 4*PAGE);
        fake_reg R;
        R.add(base + 8*PAGE, 4*PAGE, ab);
        fn_dma::query_fn q = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::edge_area ea = make_ea(64*1024);
        ggml_fn_l22_dma_part part;
        std::vector<int32_t> tk;
        const int rc = fn_dma::walk_slice(base + 8*PAGE + 200, 2*PAGE, q, ea, &part, tk);
        TCHECK(rc == -1 && part.n_seg == 1 && map_dev(part.seg[0].src) == ab + 200, "arena copy address");
        TCHECK(copy_ok(part, base + 8*PAGE + 200, 2*PAGE), "G bytes");
    }
    // H: two adjacent ranges -> two pinned pieces
    {
        fake_reg R;
        R.add(base + 16*PAGE, 2*PAGE);
        R.add(base + 18*PAGE, 2*PAGE);
        fn_dma::query_fn q = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::edge_area ea = make_ea(64*1024);
        ggml_fn_l22_dma_part part;
        std::vector<int32_t> tk;
        const int rc = fn_dma::walk_slice(base + 16*PAGE, 4*PAGE, q, ea, &part, tk);
        TCHECK(rc == -1 && part.n_seg == 2 && part.seg[0].staged == 0 && part.seg[1].staged == 0 && tk.size() == 2, "adjacent ranges");
        TCHECK(copy_ok(part, base + 16*PAGE, 4*PAGE), "H bytes");
    }
    // I: the edge area is full
    {
        fake_reg R;
        R.add(base + PAGE, 2*PAGE);
        fn_dma::query_fn   q   = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
        fn_dma::release_fn rel = [&](int32_t t) { R.release(t); };
        fn_dma::edge_area ea = make_ea(512);
        const uint8_t * ps[3] = { base + PAGE - 400, base + PAGE, base + PAGE + 64 };
        const size_t    ns[3] = { 400 + PAGE, 64, PAGE + PAGE - 64 + 300 };
        ggml_fn_l22_dma_part parts[3];
        std::vector<int32_t> tk;
        const int rc = fn_dma::resolve_expert(ps, ns, q, rel, ea, parts, tk);
        TCHECK(rc == fn_dma::REJ_EDGE && R.outstanding() == 0 && ea.used == 0, "edge area full (rc %d)", rc);
    }
    // J: random slices over random ranges: accepted ones copy byte-exact and pass the device check; rejects hold nothing
    {
        std::mt19937 rng(99);
        int n_ok = 0, n_rej = 0;
        for (int it = 0; it < 3000; ++it) {
            fake_reg R;
            size_t at = 0;
            while (at < 60*PAGE) {
                const size_t gap = (rng() % 3)*PAGE;
                const size_t len = (1 + rng() % 6)*PAGE;
                at += gap;
                if (at + len > 60*PAGE) {
                    break;
                }
                R.add(base + at, len, nullptr, rng() % 50 != 0);
                at += len;
            }
            fn_dma::query_fn   q   = [&](const void * p, size_t n, fn_dma::seg_q * o) { return R.query(p, n, o); };
            fn_dma::release_fn rel = [&](int32_t t) { R.release(t); };
            fn_dma::edge_area ea = make_ea(48*1024);
            const uint8_t * ps[3];
            size_t          ns[3];
            for (int k = 0; k < 3; ++k) {
                ns[k] = 32 + (rng() % (5*PAGE/32))*32;
                ps[k] = base + (rng() % ((60*PAGE - ns[k])/32))*32;
            }
            ggml_fn_l22_dma_part parts[3];
            std::vector<int32_t> tk;
            const int rc = fn_dma::resolve_expert(ps, ns, q, rel, ea, parts, tk);
            if (rc == -1) {
                n_ok++;
                bool good = true;
                for (int k = 0; k < 3; ++k) {
                    good = good && copy_ok(parts[k], ps[k], ns[k]) && ggml_fn_l22_dma_part_ok(&parts[k], ns[k], ea.dev, ea.dev + ea.cap) == 1;
                }
                TCHECK(good, "random accepted expert copies byte-exact (it %d)", it);
                TCHECK(R.outstanding() == (int) tk.size(), "tickets held = pinned pieces (it %d)", it);
                for (int32_t t : tk) {
                    rel(t);
                }
            } else {
                n_rej++;
                TCHECK(tk.empty() && ea.used == 0, "reject holds nothing (it %d)", it);
            }
            TCHECK(R.outstanding() == 0, "all tickets back (it %d)", it);
        }
        TCHECK(n_ok > 100 && n_rej > 100, "both outcomes seen (%d ok, %d rejected)", n_ok, n_rej);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 6. the levers' drop rule against P8's mark pass (ggml-cpu.c ggml_fn_moe_compute) and P11's light rule

static std::set<int32_t> ref_dropped(const fn_dma::job_in & j) {
    // P8: bit 4 seen, bit 1 a rank below R, bit 2 a weight at or above W (weights only when the job has them)
    std::set<int32_t> out;
    auto pass = [&](const fn_dma::xrule & x) {
        if (!x.on()) {
            return;
        }
        const int   tr = x.r;
        const float tw = j.w != nullptr ? x.w : 0.0f;
        if (tr <= 0 && tw <= 0.0f) {
            return;
        }
        std::map<int32_t, uint8_t> f;
        for (int t = 0; t < j.T; ++t) {
            for (int s = 0; s < j.n_used; ++s) {
                const int32_t e = j.ids[t*j.n_used + s];
                if (j.hot != nullptr && j.hot[e] != j.hot_miss) {
                    continue;
                }
                uint8_t b = 4;
                if (tr > 0 && s < tr) {
                    b |= 1;
                }
                if (tw > 0.0f && j.w[s + j.n_used*t] >= tw) {
                    b |= 2;
                }
                f[e] |= b;
            }
        }
        for (const auto & kv : f) {
            if ((tr > 0 && !(kv.second & 1)) || (tw > 0.0f && !(kv.second & 2))) {
                out.insert(kv.first);
            }
        }
    };
    pass(j.tail);
    pass(j.defer); // P11: light(e) = !tail(e) && (W rule || R rule): the union with tail has the same form
    return out;
}

static void random_job(std::mt19937 & rng, int n_exp, int n_used, int T, std::vector<int32_t> & ids, std::vector<float> & w) {
    ids.resize((size_t) n_used*T);
    w.resize((size_t) n_used*T);
    for (int t = 0; t < T; ++t) {
        std::set<int32_t> seen;
        float wsum = 0.0f;
        for (int i = 0; i < n_used; ++i) {
            int32_t e;
            do {
                e = (int32_t) (rng() % (uint32_t) n_exp);
            } while (!seen.insert(e).second);
            ids[(size_t) t*n_used + i] = e;
            w[(size_t) t*n_used + i]   = 1.0f/(1.0f + i + (float) (rng() % 100)/50.0f);
            wsum += w[(size_t) t*n_used + i];
        }
        for (int i = 0; i < n_used; ++i) {
            w[(size_t) t*n_used + i] /= wsum;
        }
    }
}

static void test_drop_rule() {
    std::mt19937 rng(5);
    fn_dma::scratch sc;
    const int n_exp = 96;
    sc.init(n_exp);
    std::vector<int32_t> hot(n_exp);
    int bad = 0;
    for (int it = 0; it < 20000; ++it) {
        const int n_used = 1 + (int) (rng() % 10);
        const int T      = 1 + (int) (rng() % 8);
        std::vector<int32_t> ids;
        std::vector<float>   w;
        random_job(rng, n_exp, n_used, T, ids, w);
        for (int e = 0; e < n_exp; ++e) {
            hot[e] = rng() % 3 == 0 ? e % 7 : 7;
        }
        fn_dma::job_in j;
        j.ids      = ids.data();
        j.w        = rng() % 5 == 0 ? nullptr : w.data();
        j.n_used   = n_used;
        j.T        = T;
        j.n_exp    = n_exp;
        j.hot      = rng() % 4 == 0 ? nullptr : hot.data();
        j.hot_miss = 7;
        if (rng() % 2) {
            j.tail.r = (int) (rng() % 12);
            j.tail.w = rng() % 2 ? (float) (rng() % 20)/100.0f : 0.0f;
        }
        if (rng() % 2) {
            j.defer.r = (int) (rng() % 10);
            j.defer.w = rng() % 2 ? (float) (rng() % 20)/100.0f : 0.0f;
        }
        std::vector<fn_dma::cold_e> kept;
        int n_excl = 0;
        fn_dma::cold_set(j, sc, kept, &n_excl);
        const std::set<int32_t> drop = ref_dropped(j);
        std::set<int32_t> cold_ref;
        for (int a = 0; a < n_used*T; ++a) {
            const int32_t e = ids[a];
            if (j.hot == nullptr || j.hot[e] == j.hot_miss) {
                cold_ref.insert(e);
            }
        }
        std::set<int32_t> kept_set;
        for (const auto & c : kept) {
            kept_set.insert(c.e);
        }
        std::set<int32_t> kept_ref;
        for (int32_t e : cold_ref) {
            if (!drop.count(e)) {
                kept_ref.insert(e);
            }
        }
        if (kept_set != kept_ref || (int) drop.size() != n_excl || kept.size() != kept_set.size()) {
            bad++;
        }
        for (int e = 0; e < n_exp; ++e) {
            if (sc.idx[e] != -1) {
                bad++;
                break;
            }
        }
    }
    TCHECK(bad == 0, "drop rule = P8 mark pass + P11 light rule (%d mismatches)", bad);
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. the commit machine

static void test_commit() {
    fn_dma::layer_state L;
    L.init(4, 8, 40, 64);
    int outstanding = 0;
    fn_dma::release_fn rel = [&](int32_t) { outstanding--; };
    auto make = [&](uint32_t seq, int n) {
        fn_dma::pending_job P;
        P.on  = true;
        P.seq = seq;
        P.n   = n;
        for (int i = 0; i < n; ++i) {
            P.slot[i] = i;
            P.e[i]    = 10 + i;
            L.pending[i]  = 1;
            L.slot_exp[i] = 10 + i;
            L.valid[i]    = 0;
            P.tickets.push_back(i);
            P.tickets.push_back(100 + i);
            outstanding += 2;
        }
        return P;
    };
    auto rec = [](uint32_t seq, int n, uint32_t ok) {
        ggml_fn_l22_dma_done d;
        memset(&d, 0, sizeof(d));
        d.seq = seq;
        d.n   = n;
        d.ok  = ok;
        return d;
    };
    {
        fn_dma::pending_job P = make(5, 2);
        TCHECK(fn_dma::commit(L, P, rec(5, 2, 1), rel) == fn_dma::CM_OK, "match -> commit");
        TCHECK(L.valid[0] && L.valid[1] && L.mirror[10] == 0 && L.mirror[11] == 1 && !L.pending[0] && !L.pending[1], "slots valid");
        TCHECK(outstanding == 0 && P.tickets.empty() && !P.on, "tickets back");
        TCHECK(fn_dma::commit(L, P, rec(5, 2, 1), rel) == fn_dma::CM_NONE && outstanding == 0, "second commit does nothing");
    }
    const ggml_fn_l22_dma_done bad[4] = { rec(6, 2, 1), rec(7, 2, 0), rec(7, 1, 1), rec(0, 0, 0) };
    const char * why[4] = { "seq mismatch", "ok 0", "n mismatch", "never written" };
    for (int k = 0; k < 4; ++k) {
        L.init(4, 8, 40, 64);
        fn_dma::pending_job P = make(7, 2);
        TCHECK(fn_dma::commit(L, P, bad[k], rel) == fn_dma::CM_BAD, "%s -> invalidate", why[k]);
        TCHECK(!L.valid[0] && !L.valid[1] && L.slot_exp[0] == -1 && L.mirror[10] == -1 && !L.pending[0], "%s: slots empty", why[k]);
        TCHECK(outstanding == 0, "%s: tickets back", why[k]);
    }
    // a second copy of an expert already in a slot keeps the new slot only
    {
        L.init(4, 8, 40, 64);
        L.valid[3] = 1;
        L.slot_exp[3] = 10;
        L.mirror[10] = 3;
        fn_dma::pending_job P = make(9, 1);
        TCHECK(fn_dma::commit(L, P, rec(9, 1, 1), rel) == fn_dma::CM_OK && L.mirror[10] == 0 && !L.valid[3] && L.slot_exp[3] == -1,
                "duplicate copy");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. the plan on random jobs, with a bank model (what each slot really holds after the device copies)

static void test_plan_fuzz() {
    std::mt19937 rng(1234);
    const int n_exp    = 128;
    const int S        = 4;
    const int slot_lo  = 2*S;
    const int zero     = 5*S;
    fn_dma::layer_state L;
    L.init(S, slot_lo, zero, n_exp);
    fn_dma::scratch sc;
    sc.init(n_exp);
    std::vector<int32_t> bank(S, -1);   // device truth: the expert a slot's bytes are (-1 empty, -2 torn)
    std::vector<int32_t> hot(n_exp);
    std::vector<int32_t> skip(n_exp);
    std::vector<int32_t> slot_ids(10*8);
    int outstanding = 0;
    uint64_t tick = 0;
    long n_hit = 0, n_copy = 0, n_rej = 0, n_jobs_copy = 0;
    int bad = 0;
    for (int it = 0; it < 20000; ++it) {
        const int n_used = 10;
        const int T      = 1 + (int) (rng() % 8);
        std::vector<int32_t> ids;
        std::vector<float>   w;
        random_job(rng, n_exp, n_used, T, ids, w);
        if (it % 50 == 0) {
            for (int e = 0; e < n_exp; ++e) {
                hot[e] = rng() % 3 == 0 ? (int32_t) (e % 30) : 30;
            }
        }
        fn_dma::job_in j;
        j.ids      = ids.data();
        j.w        = w.data();
        j.n_used   = n_used;
        j.T        = T;
        j.n_exp    = n_exp;
        j.hot      = hot.data();
        j.hot_miss = 30;
        if (rng() % 4 == 0) {
            j.tail.r = 4 + (int) (rng() % 6);
            j.tail.w = rng() % 2 ? 0.05f : 0.0f;
        }
        if (rng() % 4 == 0) {
            j.defer.r = 5 + (int) (rng() % 4);
            j.defer.w = rng() % 2 ? 0.08f : 0.0f;
        }
        for (int e = 0; e < n_exp; ++e) {
            skip[e] = hot[e] != 30 ? 1 : 0;
        }
        const std::set<int32_t> drop = ref_dropped(j);
        fn_dma::rates r;
        r.cpu_us = 20.0 + (double) (rng() % 200);
        r.dma_us = 10.0 + (double) (rng() % 150);
        const bool pins     = rng() % 10 != 0;
        const int  max_copy = 1 + (int) (rng() % S);
        fn_dma::resolve_fn resolve = [&](int32_t e, ggml_fn_l22_dma_part parts[3], std::vector<int32_t> & tk) {
            if ((e*7 + it) % 5 == 0) {
                return (int) fn_dma::REJ_UNPINNED;
            }
            for (int k = 0; k < 3; ++k) {
                memset(&parts[k], 0, sizeof(parts[k]));
                parts[k].n_seg  = 1;
                parts[k].seg[0] = { (uint64_t) (1000000 + e*3 + k), 1000, 0 };
            }
            tk.push_back(e);
            outstanding++;
            return -1;
        };
        ggml_fn_l22_dma_side side;
        memset(&side, 0, sizeof(side));
        fn_dma::pending_job P;
        const uint32_t seq = (uint32_t) it + 1;
        const fn_dma::plan_res res = fn_dma::plan_job(L, j, sc, r, max_copy, pins, ++tick, resolve, &side, seq, slot_ids.data(),
                skip.data(), P.tickets);

        // invariants
        std::map<int32_t, int32_t> served; // expert -> slot
        std::set<int32_t> hit_slots;
        for (int a = 0; a < n_used*T; ++a) {
            const int32_t e  = ids[a];
            const int32_t sl = slot_ids[a];
            if (sl == zero) {
                continue;
            }
            if (sl < slot_lo || sl >= slot_lo + S) {
                bad++;
                fprintf(stderr, "it %d: slot id %d outside the layer\n", it, sl);
                continue;
            }
            auto f = served.find(e);
            if (f != served.end() && f->second != sl) {
                bad++;
                fprintf(stderr, "it %d: expert %d in two slots\n", it, e);
            }
            served[e] = sl;
        }
        std::set<int32_t> copy_slots;
        for (int i = 0; i < res.n_copy; ++i) {
            copy_slots.insert(side.slot[i]);
            if (side.slot[i] != slot_lo + res.copy_slot[i] || served[res.copy_e[i]] != side.slot[i]) {
                bad++;
                fprintf(stderr, "it %d: copy %d bookkeeping\n", it, i);
            }
        }
        for (const auto & kv : served) {
            const int32_t e = kv.first;
            if (hot[e] != 30 || drop.count(e)) {
                bad++;
                fprintf(stderr, "it %d: served expert %d is hot or dropped\n", it, e);
            }
            if (!copy_slots.count(kv.second)) {
                hit_slots.insert(kv.second);
                if (bank[kv.second - slot_lo] != e) {
                    bad++;
                    fprintf(stderr, "it %d: hit on slot %d holds %d, not %d\n", it, kv.second, bank[kv.second - slot_lo], e);
                }
            }
        }
        if ((int) hit_slots.size() != res.n_hit || (int) copy_slots.size() != res.n_copy) {
            bad++;
            fprintf(stderr, "it %d: counts hit %d/%zu copy %d/%zu\n", it, res.n_hit, hit_slots.size(), res.n_copy, copy_slots.size());
        }
        for (int32_t sl : copy_slots) {
            if (hit_slots.count(sl)) {
                bad++;
                fprintf(stderr, "it %d: victim %d was a hit\n", it, sl);
            }
        }
        if (res.n_copy > std::min(max_copy, S - res.n_hit) || res.n_copy > res.k || side.n_copy != res.n_copy || side.seq != seq ||
            (!pins && res.n_copy > 0)) {
            bad++;
            fprintf(stderr, "it %d: caps n_copy %d k %d hits %d max %d pins %d\n", it, res.n_copy, res.k, res.n_hit, max_copy, (int) pins);
        }
        for (int e = 0; e < n_exp; ++e) {
            const int want = (hot[e] != 30 || served.count(e)) ? 1 : 0;
            if (skip[e] != want) {
                bad++;
                fprintf(stderr, "it %d: skip[%d] = %d, want %d\n", it, e, skip[e], want);
                break;
            }
            if (sc.served[e] != -1 || sc.idx[e] != -1) {
                bad++;
                fprintf(stderr, "it %d: scratch not clean\n", it);
                break;
            }
        }
        if ((int) P.tickets.size() != res.n_copy || outstanding != res.n_copy) {
            bad++;
            fprintf(stderr, "it %d: tickets %zu / outstanding %d vs copies %d\n", it, P.tickets.size(), outstanding, res.n_copy);
        }
        n_hit  += res.n_hit;
        n_copy += res.n_copy;
        n_rej  += res.rej[fn_dma::REJ_UNPINNED];

        // the device runs the copies; most records match, some do not (torn slot)
        P.on  = res.n_copy > 0;
        P.seq = seq;
        P.n   = res.n_copy;
        for (int i = 0; i < res.n_copy; ++i) {
            P.slot[i] = res.copy_slot[i];
            P.e[i]    = res.copy_e[i];
        }
        ggml_fn_l22_dma_done d;
        memset(&d, 0, sizeof(d));
        const bool good = rng() % 8 != 0;
        d.seq = good ? seq : seq - 1;
        d.n   = res.n_copy;
        d.ok  = 1;
        for (int i = 0; i < res.n_copy; ++i) {
            bank[res.copy_slot[i]] = good ? res.copy_e[i] : -2;
        }
        if (res.n_copy > 0) {
            n_jobs_copy++;
            fn_dma::release_fn rel = [&](int32_t) { outstanding--; };
            const int rc = fn_dma::commit(L, P, d, rel);
            if (rc != (good ? fn_dma::CM_OK : fn_dma::CM_BAD)) {
                bad++;
            }
        } else {
            P.tickets.clear();
        }
        if (outstanding != 0) {
            bad++;
            fprintf(stderr, "it %d: %d tickets open after the commit\n", it, outstanding);
            outstanding = 0;
        }
        // the slot table agrees with the device truth
        for (int s = 0; s < S; ++s) {
            if (L.pending[s] || (L.valid[s] && (bank[s] != L.slot_exp[s] || L.mirror[L.slot_exp[s]] != s))) {
                bad++;
                fprintf(stderr, "it %d: slot %d table (valid %d exp %d bank %d)\n", it, s, L.valid[s], L.slot_exp[s], bank[s]);
            }
        }
        for (int e = 0; e < n_exp; ++e) {
            const int s = L.mirror[e];
            if (s >= 0 && (!L.valid[s] || L.slot_exp[s] != e)) {
                bad++;
                fprintf(stderr, "it %d: mirror of %d\n", it, e);
            }
        }
        if (bad > 20) {
            break;
        }
    }
    TCHECK(bad == 0, "plan invariants (%d violations)", bad);
    TCHECK(n_hit > 1000 && n_copy > 1000 && n_rej > 100 && n_jobs_copy > 1000, "coverage: %ld hits, %ld copies, %ld rejects, %ld jobs with copies",
            n_hit, n_copy, n_rej, n_jobs_copy);
}

// the plan with no pins: hits only, nothing pending, nothing held
static void test_plan_nopin() {
    fn_dma::layer_state L;
    L.init(4, 0, 4, 32);
    fn_dma::scratch sc;
    sc.init(32);
    L.valid[2] = 1;
    L.slot_exp[2] = 9;
    L.mirror[9] = 2;
    std::vector<int32_t> ids = { 9, 3, 4, 5 };
    std::vector<float>   w   = { 0.4f, 0.3f, 0.2f, 0.1f };
    std::vector<int32_t> skip(32, 0), slot_ids(4), tk;
    fn_dma::job_in j;
    j.ids    = ids.data();
    j.w      = w.data();
    j.n_used = 4;
    j.T      = 1;
    j.n_exp  = 32;
    int calls = 0;
    fn_dma::resolve_fn resolve = [&](int32_t, ggml_fn_l22_dma_part *, std::vector<int32_t> &) { calls++; return -1; };
    ggml_fn_l22_dma_side side;
    memset(&side, 0, sizeof(side));
    fn_dma::rates r;
    r.cpu_us = 1000;
    r.dma_us = 1;
    const fn_dma::plan_res res = fn_dma::plan_job(L, j, sc, r, 4, false, 1, resolve, &side, 77, slot_ids.data(), skip.data(), tk);
    TCHECK(res.n_hit == 1 && res.n_copy == 0 && calls == 0 && tk.empty(), "no pins: hits only");
    TCHECK(slot_ids[0] == 2 && slot_ids[1] == 4 && slot_ids[2] == 4 && slot_ids[3] == 4, "slot ids (hit, zero slot)");
    TCHECK(skip[9] == 1 && skip[3] == 0 && skip[4] == 0 && skip[5] == 0, "skip marks");
    TCHECK(side.seq == 77 && side.n_copy == 0, "side written for an empty plan");
    for (int s = 0; s < 4; ++s) {
        TCHECK(!L.pending[s], "nothing pending");
    }
    // with pins the CPU-slow rates copy the three misses into the three free slots
    std::fill(skip.begin(), skip.end(), 0);
    const fn_dma::plan_res r2 = fn_dma::plan_job(L, j, sc, r, 4, true, 2, resolve, &side, 78, slot_ids.data(), skip.data(), tk);
    TCHECK(r2.n_hit == 1 && r2.n_copy == 3 && calls == 3 && side.n_copy == 3, "with pins: 3 copies (%d)", r2.n_copy);
    TCHECK(slot_ids[0] == 2 && slot_ids[1] != 4 && slot_ids[2] != 4 && slot_ids[3] != 4 && slot_ids[1] != 2, "copies in free slots");
}

int main() {
    test_layout_cfg();
    test_best_k();
    test_walker();
    test_drop_rule();
    test_commit();
    test_plan_fuzz();
    test_plan_nopin();
    printf("test-fn-l22-dmashare: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
