#pragma once

// [TAG_FN_L22] round 22 P2: small pure helpers of libmtmd (header only, unit-tested in tests/test-mtmd-impl.cpp)
//   - mtmd_l22_parallel_for       [TAG_FN_L22_IMGDEC]     run fn(i) for i < n on up to n_threads threads
//   - mtmd_l22_hash128            [TAG_FN_L22_IMGCACHE]   two independent 64-bit word hashes over a byte range
//   - mtmd_l22_embd_cache         [TAG_FN_L22_IMGCACHE]   LRU of encoder outputs by content key, entry and byte caps
//   - mtmd_l22_lend_size          [TAG_FN_L22_VLEND_SIZE] the encoder compute bytes a picture of n patches needs

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <list>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

// fn(i) for every i in [0, n): the caller's thread works too. The first exception (by index) is rethrown after all ran.
static inline void mtmd_l22_parallel_for(int n, int n_threads, const std::function<void(int)> & fn) {
    if (n <= 0) {
        return;
    }
    const int nt = std::max(1, std::min(n_threads, n));
    if (nt == 1) {
        for (int i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::exception_ptr> errs((size_t) n);
    auto work = [&]() {
        for (int i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
            try {
                fn(i);
            } catch (...) {
                errs[(size_t) i] = std::current_exception();
            }
        }
    };
    std::vector<std::thread> th;
    th.reserve((size_t) nt - 1);
    for (int t = 1; t < nt; ++t) {
        try {
            th.emplace_back(work);
        } catch (...) {
            break; // no thread: the rest runs on the caller
        }
    }
    work();
    for (auto & t : th) {
        t.join();
    }
    for (auto & e : errs) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

struct mtmd_l22_hash128 {
    uint64_t a = 0x243F6A8885A308D3ull;
    uint64_t b = 0x13198A2E03707344ull;

    static uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

    void add_u64(uint64_t w) {
        a = (a ^ w) * 0x9E3779B97F4A7C15ull;
        a ^= a >> 29;
        b = rotl(b + w * 0xC2B2AE3D27D4EB4Full, 31) * 0x165667B19E3779F9ull;
    }

    void add(const void * p, size_t n) {
        const uint8_t * s = (const uint8_t *) p;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint64_t w;
            std::memcpy(&w, s + i, 8);
            add_u64(w);
        }
        uint64_t tail = 0;
        std::memcpy(&tail, s + i, n - i);
        add_u64(tail ^ ((uint64_t) n << 56));
    }

    bool operator==(const mtmd_l22_hash128 & o) const { return a == o.a && b == o.b; }
};

// LRU of encoder outputs: key = content hash (the caller hashes every input that changes the output)
struct mtmd_l22_embd_cache {
    struct entry {
        mtmd_l22_hash128   key;
        size_t             n_floats = 0;
        std::vector<float> embd;
    };

    size_t n_max     = 0;                 // 0: off
    size_t bytes_max = (size_t) 256 << 20;
    size_t bytes     = 0;
    uint64_t n_hit = 0, n_miss = 0;
    std::list<entry> lru;                 // most recent first
    std::mutex mu;

    // copies the cached output into out (resized) and returns true on a hit
    bool get(const mtmd_l22_hash128 & key, size_t n_floats, std::vector<float> & out) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lru.begin(); it != lru.end(); ++it) {
            if (it->key == key && it->n_floats == n_floats) {
                out.assign(it->embd.begin(), it->embd.end());
                lru.splice(lru.begin(), lru, it);
                n_hit++;
                return true;
            }
        }
        n_miss++;
        return false;
    }

    void put(const mtmd_l22_hash128 & key, const float * data, size_t n_floats) {
        if (n_max == 0 || n_floats * sizeof(float) > bytes_max) {
            return;
        }
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lru.begin(); it != lru.end(); ++it) {
            if (it->key == key && it->n_floats == n_floats) {
                lru.splice(lru.begin(), lru, it);
                return;
            }
        }
        entry e;
        e.key      = key;
        e.n_floats = n_floats;
        try {
            e.embd.assign(data, data + n_floats);
        } catch (...) {
            return; // no memory: not cached
        }
        lru.push_front(std::move(e));
        bytes += n_floats * sizeof(float);
        while (!lru.empty() && (lru.size() > n_max || bytes > bytes_max)) {
            bytes -= std::min(bytes, lru.back().n_floats * sizeof(float));
            lru.pop_back();
        }
    }
};

// encoder compute bytes per picture size: measured at the warm-up size and at the largest picture, exact once a size
// was seen; in between a chord (above a convex size curve) plus a margin
struct mtmd_l22_lend_size {
    bool    on    = false;
    int64_t n_w   = 0; // patches of the warm-up picture
    size_t  c_w   = 0; // its compute bytes (today's lend)
    int64_t n_max = 0; // patches of the largest picture
    size_t  c_max = 0;
    std::vector<std::tuple<int, int, int, size_t>> seen; // (entries, nx, ny) -> bytes of a fresh compute buffer

    const size_t * find(int ne, int nx, int ny) const {
        for (const auto & s : seen) {
            if (std::get<0>(s) == ne && std::get<1>(s) == nx && std::get<2>(s) == ny) {
                return &std::get<3>(s);
            }
        }
        return nullptr;
    }

    void record(int ne, int nx, int ny, size_t bytes) {
        for (auto & s : seen) {
            if (std::get<0>(s) == ne && std::get<1>(s) == nx && std::get<2>(s) == ny) {
                std::get<3>(s) = std::max(std::get<3>(s), bytes);
                return;
            }
        }
        if (seen.size() >= 64) {
            seen.erase(seen.begin());
        }
        seen.emplace_back(ne, nx, ny, bytes);
    }

    // compute bytes to take for a picture of n patches (never below the warm-up bytes)
    size_t estimate(int64_t n, const size_t * exact) const {
        if (!on) {
            return c_w;
        }
        if (exact != nullptr) {
            return std::max(c_w, *exact);
        }
        if (n <= n_w || n_max <= n_w || c_max <= c_w) {
            return c_w;
        }
        const double slope = (double) (c_max - c_w) / (double) (n_max - n_w);
        const double est   = (double) c_w + slope * (double) (n - n_w);
        const double marg  = est * 0.03 + 16.0 * 1048576.0;
        return std::max(c_w, (size_t) (est + marg));
    }
};
