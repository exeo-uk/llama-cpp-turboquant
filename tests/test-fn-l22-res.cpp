// [TAG_FN_L22_RESSET] [TAG_FN_L22_SSDTIER] round 22, P5: the residency core (src/llama-fn-res-core.h, header only).
// CPU only, no model: class precedence, expert runs and page alignment, the SSD pick, the lend estimate, and the direct-I/O
// engine against a temporary file (Windows).

#include "llama-fn-res-core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#endif

static int g_fail = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            g_fail++;                                                              \
        }                                                                          \
    } while (0)

using namespace fn_res;

static void test_class() {
    CHECK(class_of(0) == 0);
    CHECK(class_of(F_SLOT) == 0);
    CHECK(class_of(F_VRAM | F_SLOT) == 1);
    CHECK(class_of(F_PIN) == 2);
    CHECK(class_of(F_PIN | F_VRAM | F_SSD) == 2);
    CHECK(class_of(F_VRAM | F_SSD) == 1);
    CHECK(class_of(F_SSD) == 3);
}

static void test_runs() {
    const uint8_t sel[10] = { 1, 1, 0, 1, 0, 0, 1, 1, 1, 0 };
    std::vector<range> rs;
    runs_of(1000, 100, sel, 10, rs);
    CHECK(rs.size() == 3);
    CHECK(rs[0].lo == 1000 && rs[0].hi == 1200);
    CHECK(rs[1].lo == 1300 && rs[1].hi == 1400);
    CHECK(rs[2].lo == 1600 && rs[2].hi == 1900);
    const uint8_t none[3] = { 0, 0, 0 };
    rs.clear();
    runs_of(0, 8, none, 3, rs);
    CHECK(rs.empty());
    const uint8_t all[4] = { 1, 1, 1, 1 };
    runs_of(0, 8, all, 4, rs);
    CHECK(rs.size() == 1 && rs[0].lo == 0 && rs[0].hi == 32);

    // warm pieces: runs cut at max_bytes, at least one expert each
    std::vector<std::pair<int, int>> pr;
    piece_runs(sel, 10, 100, 250, pr); // two experts per piece at most
    CHECK(pr.size() == 4);
    CHECK(pr[0] == std::make_pair(0, 2) && pr[1] == std::make_pair(3, 4) && pr[2] == std::make_pair(6, 8) &&
          pr[3] == std::make_pair(8, 9));
    pr.clear();
    piece_runs(sel, 10, 100, 10, pr); // smaller than one expert: one expert per piece
    CHECK(pr.size() == 6);
    pr.clear();
    piece_runs(sel, 10, 100, 1 << 20, pr);
    CHECK(pr.size() == 3 && pr[2] == std::make_pair(6, 9));

    // inward: whole pages inside only; outward: every page touched
    range a = align_in({ 4097, 3*4096 + 5 }, 4096);
    CHECK(a.lo == 8192 && a.hi == 12288 && a.size() == 4096);
    range b = align_in({ 4097, 8191 }, 4096);
    CHECK(b.size() == 0);
    range c = align_out({ 4097, 8193 }, 4096);
    CHECK(c.lo == 4096 && c.hi == 12288);
    range d = align_in({ 8192, 16384 }, 4096);
    CHECK(d.lo == 8192 && d.hi == 16384);
}

static void test_ssd() {
    const size_t G = (size_t) 1 << 30;
    // fits: no deficit
    CHECK(ssd_deficit(60*G, 50*G, 20*G, 4*G, 0) == 0);
    // need 64, resident 50, avail 14, floor 4 -> room 10 -> cap 60 -> deficit 4
    CHECK(ssd_deficit(64*G, 50*G, 14*G, 4*G, 0) == 4*G);
    // pins pending eat the room
    CHECK(ssd_deficit(64*G, 50*G, 14*G, 4*G, 2*G) == 6*G);
    // avail below the floor: no room at all
    CHECK(ssd_deficit(64*G, 50*G, 3*G, 4*G, 0) == 14*G);

    std::vector<ssd_cand> c = {
        { 0, 0, 5.0f, 10, false },
        { 0, 1, 1.0f, 10, true  },
        { 1, 0, 1.0f, 10, false },
        { 1, 1, 0.0f, 10, true  },
        { 2, 0, 9.0f, 10, false },
    };
    auto p = ssd_pick(c, 25, true);
    CHECK(p.size() == 3);
    CHECK(p[0] == 3);             // count 0 first
    CHECK(p[1] == 2 && p[2] == 1); // count 1: the non-resident one first
    auto q = ssd_pick(c, 0, true);
    CHECK(q.empty());
    auto all = ssd_pick(c, 1000, false);
    CHECK(all.size() == c.size());
    // equal counts are spread over layers deterministically (same input, same order)
    std::vector<ssd_cand> z;
    for (int l = 0; l < 4; ++l) {
        for (int e = 0; e < 8; ++e) {
            z.push_back({ l, e, 0.0f, 1, false });
        }
    }
    auto z1 = ssd_pick(z, 8, false);
    auto z2 = ssd_pick(z, 8, false);
    CHECK(z1 == z2 && z1.size() == 8);
    int per_layer[4] = { 0, 0, 0, 0 };
    for (size_t i : z1) {
        per_layer[z[i].layer]++;
    }
    int max_l = 0;
    for (int l = 0; l < 4; ++l) {
        max_l = std::max(max_l, per_layer[l]);
    }
    CHECK(max_l < 8); // not all from one layer

    // marks back: most routed first while they fit
    auto u = ssd_unpick(c, 20);
    CHECK(u.size() == 2 && u[0] == 4 && u[1] == 0);
    CHECK(ssd_unpick(c, 5).empty());
}

static void test_lend() {
    size_t lo = 0;
    size_t hi = 0;
    // 10000-byte buffer, 2000-byte tail, 3000 needed, align 256: end 8000, start (5000/256)*256 = 4864
    CHECK(lend_range(10000, 2000, 3000, 256, &lo, &hi));
    CHECK(hi == 8000 && lo == 4864);
    CHECK(lend_range(10000, 0, 3000, 256, &lo, &hi));
    CHECK(hi == 10000 && lo == 6912);
    CHECK(!lend_range(10000, 2000, 7900, 256, &lo, &hi));
    CHECK(!lend_range(10000, 0, 0, 256, &lo, &hi));
    CHECK(overlaps(100, 200, 199, 300));
    CHECK(!overlaps(100, 200, 200, 300));
    CHECK(!overlaps(300, 400, 200, 300));
    CHECK(overlaps(0, 1000, 200, 300));
}

#ifdef _WIN32

static uint8_t pat(uint64_t i) {
    uint64_t h = i*0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    return (uint8_t) (h >> 7);
}

static void test_dio(int n_threads) {
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    std::string path = std::string(tmp) + "test-fn-l22-res-" + std::to_string(GetCurrentProcessId()) + "-" +
                       std::to_string(n_threads) + ".bin";
    // 13 MiB + 1234 bytes: the last sector is partial (the end-of-file path)
    const size_t size = ((size_t) 13 << 20) + 1234;
    std::vector<uint8_t> mem(size);
    for (size_t i = 0; i < size; ++i) {
        mem[i] = pat(i);
    }
    FILE * f = fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) {
        return;
    }
    fwrite(mem.data(), 1, size, f);
    fclose(f);

    {
        dio_engine eng;
        // two regions of one file: [0, 5 MiB) and [5 MiB, size) mapped at their offsets in mem
        const size_t split = (size_t) 5 << 20;
        eng.add_region(mem.data(), split, 0, 0);
        eng.add_region(mem.data() + split, size - split, 0, split);
        std::string err;
        CHECK(eng.open({ path }, n_threads, err));
        if (!eng.is_open()) {
            fprintf(stderr, "open: %s\n", err.c_str());
            DeleteFileA(path.c_str());
            return;
        }
        // destination arena: page aligned, with guard bytes around every destination
        const size_t cap = (size_t) 16 << 20;
        uint8_t * arena = (uint8_t *) VirtualAlloc(nullptr, cap, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        CHECK(arena != nullptr);
        std::mt19937_64 rng(1234 + n_threads);
        struct job {
            size_t src;
            size_t len;
            size_t dst;
        };
        for (int round = 0; round < 40; ++round) {
            memset(arena, 0xA5, cap);
            std::vector<job> jobs;
            std::vector<dio_req> reqs;
            size_t at = 4096;
            const int n = 1 + (int) (rng() % 6);
            for (int k = 0; k < n; ++k) {
                job j;
                const int kind = (int) (rng() % 5);
                if (kind == 0) {
                    j.len = 1 + rng() % 100; // tiny, inside one sector or across two
                } else if (kind == 1) {
                    j.len = ((size_t) 4 << 20) + 1 + rng() % (3 << 20); // more than one piece
                } else {
                    j.len = 1 + rng() % (512 << 10);
                }
                const size_t region_end = (rng() % 2) ? split : size;
                const size_t region_lo  = region_end == split ? 0 : split;
                if (j.len > region_end - region_lo) {
                    j.len = region_end - region_lo;
                }
                j.src = region_lo + rng() % (region_end - region_lo - j.len + 1);
                if (round == 0 && k == 0) {
                    j.src = size - 777; // the file's last bytes
                    j.len = 777;
                }
                // half the destinations share the source's sector phase (the direct path)
                const bool phase = (rng() % 2) == 0;
                size_t dst = (at + 4095)/4096*4096 + (phase ? j.src % 4096 : (rng() % 4096));
                if (dst + j.len + 64 > cap) {
                    break;
                }
                j.dst = dst;
                at = dst + j.len + 64;
                jobs.push_back(j);
                reqs.push_back({ mem.data() + j.src, j.len, arena + j.dst });
            }
            if (reqs.empty()) {
                continue;
            }
            const int t = eng.submit(reqs.data(), (int) reqs.size());
            CHECK(t > 0);
            CHECK(eng.wait(t, 20000));
            for (const auto & j : jobs) {
                CHECK(memcmp(arena + j.dst, mem.data() + j.src, j.len) == 0);
                CHECK(arena[j.dst - 1] == 0xA5);         // nothing written before
                CHECK(arena[j.dst + j.len] == 0xA5);     // nor after
            }
        }
        // several tickets in flight at once
        {
            memset(arena, 0, cap);
            std::vector<int> ts;
            // all inside the second region ([5 MiB, 13 MiB + 1234)); the destinations 1.5 MiB apart in the arena
            for (int k = 0; k < 8; ++k) {
                dio_req r = { mem.data() + split + (size_t) k*(1 << 20) + 17, (size_t) 1 << 20, arena + (size_t) k*(3 << 19) + 5 };
                ts.push_back(eng.submit(&r, 1));
                CHECK(ts.back() > 0);
            }
            for (int k = 7; k >= 0; --k) {
                CHECK(eng.wait(ts[k], 20000));
            }
            for (int k = 0; k < 8; ++k) {
                CHECK(memcmp(arena + (size_t) k*(3 << 19) + 5, mem.data() + split + (size_t) k*(1 << 20) + 17, (size_t) 1 << 20) == 0);
            }
        }
        // a source outside every region, or across two regions: refused, nothing queued
        {
            uint8_t other[64];
            dio_req bad = { other, sizeof(other), arena };
            CHECK(eng.submit(&bad, 1) == -1);
            dio_req across = { mem.data() + split - 10, 20, arena };
            CHECK(eng.submit(&across, 1) == -1);
            CHECK(!eng.wait(123456, 10)); // unknown ticket
        }
        const dio_stats s = eng.stats();
        CHECK(s.fails == 0);
        CHECK(s.bytes > 0 && s.reads > 0);
        CHECK(s.direct > 0); // the phase-matched destinations used the direct path
        printf("dio %d threads: %.1f MiB delivered, %.1f MiB direct, %llu reads\n", n_threads, s.bytes/1048576.0,
                s.direct/1048576.0, (unsigned long long) s.reads);
        VirtualFree(arena, 0, MEM_RELEASE);
        eng.close();
        CHECK(eng.submit(nullptr, 0) == -1);
    }
    DeleteFileA(path.c_str());
}

#endif

int main() {
    test_class();
    test_runs();
    test_ssd();
    test_lend();
#ifdef _WIN32
    test_dio(1);
    test_dio(8);
#endif
    if (g_fail) {
        fprintf(stderr, "test-fn-l22-res: %d failure(s)\n", g_fail);
        return 1;
    }
    printf("test-fn-l22-res: all passed\n");
    return 0;
}
