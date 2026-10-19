#pragma once

// [TAG_FN_L22_RESSET] [TAG_FN_L22_SSDTIER] round 22, P5 core: the pure residency logic (expert runs, page alignment,
// the SSD pick, the stream lend estimate) and the direct-I/O engine. Depends on the C++ library and the OS only, so the
// unit test includes it directly (tests/test-fn-l22-res.cpp). Nothing here touches a value the model computes.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

namespace fn_res {

// per-expert flag bits of the residency table
enum : uint8_t {
    F_VRAM = 1, // resident in a hot slot the prompt stream reads device to device (not lent, not tail, not KV)
    F_SLOT = 2, // resident in any hot slot
    F_PIN  = 4, // marked by the pin planner (P6)
    F_SSD  = 8, // marked for direct I/O by the SSD tier
};

// llama_fn_res values: RAM 0, VRAM 1, PIN 2, SSD 3. Precedence PIN > VRAM > SSD > RAM
inline uint8_t class_of(uint8_t f) {
    return (f & F_PIN) ? 2 : (f & F_VRAM) ? 1 : (f & F_SSD) ? 3 : 0;
}

struct range {
    uintptr_t lo = 0;
    uintptr_t hi = 0;
    size_t size() const { return hi > lo ? (size_t) (hi - lo) : 0; }
};

// runs of consecutive selected experts of one tensor (expert e = [base + e*nb, base + (e+1)*nb)), appended to out
inline void runs_of(uintptr_t base, size_t nb, const uint8_t * sel, int n, std::vector<range> & out) {
    for (int e = 0; e < n; ) {
        if (!sel[e]) {
            ++e;
            continue;
        }
        int e1 = e + 1;
        while (e1 < n && sel[e1]) {
            ++e1;
        }
        out.push_back({ base + (uintptr_t) e*nb, base + (uintptr_t) e1*nb });
        e = e1;
    }
}

// expert runs [e0, e1) of consecutive selected experts, each at most max_bytes (at least one expert), appended to out
inline void piece_runs(const uint8_t * sel, int n, size_t nb, size_t max_bytes, std::vector<std::pair<int, int>> & out) {
    const size_t per = std::max<size_t>(1, max_bytes/std::max<size_t>(1, nb));
    for (int e = 0; e < n; ) {
        if (!sel[e]) {
            ++e;
            continue;
        }
        int e1 = e + 1;
        while (e1 < n && sel[e1] && (size_t) (e1 - e) < per) {
            ++e1;
        }
        out.push_back({ e, e1 });
        e = e1;
    }
}

// the whole pages inside r (empty when none): pages shared with a neighbour stay untouched
inline range align_in(range r, size_t page) {
    range o;
    o.lo = (r.lo + page - 1)/page*page;
    o.hi = r.hi/page*page;
    if (o.hi < o.lo) {
        o.hi = o.lo;
    }
    return o;
}

// the pages that hold any byte of r
inline range align_out(range r, size_t page) {
    range o;
    o.lo = r.lo/page*page;
    o.hi = (r.hi + page - 1)/page*page;
    return o;
}

// the bytes the streamed set lacks in RAM: need - (resident + room), room = avail - floor - pin_pending (>= 0)
inline size_t ssd_deficit(size_t need, size_t resident, size_t avail, size_t floor_b, size_t pin_pending) {
    const size_t room = avail > floor_b + pin_pending ? avail - floor_b - pin_pending : 0;
    const size_t cap  = resident + room;
    return need > cap ? need - cap : 0;
}

struct ssd_cand {
    int    layer    = 0;     // table index
    int    e        = 0;
    float  cnt      = 0.0f;  // decayed routing count (0 when unknown)
    size_t bytes    = 0;
    bool   resident = false; // sampled in the working set
};

// a deterministic spread over layers and experts for equal counts (an untrained set has every count 0)
inline uint32_t ssd_spread(int layer, int e) {
    uint32_t h = (uint32_t) layer*0x9E3779B1u ^ (uint32_t) e*0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

// indices into c, least routed first, until their bytes reach `deficit`. prefer_nonres: the non-resident candidates
// of a count go first (marking them needs no page drop)
inline std::vector<size_t> ssd_pick(const std::vector<ssd_cand> & c, size_t deficit, bool prefer_nonres) {
    std::vector<size_t> idx(c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        idx[i] = i;
    }
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (c[a].cnt != c[b].cnt) {
            return c[a].cnt < c[b].cnt;
        }
        if (prefer_nonres && c[a].resident != c[b].resident) {
            return !c[a].resident;
        }
        return ssd_spread(c[a].layer, c[a].e) < ssd_spread(c[b].layer, c[b].e);
    });
    std::vector<size_t> out;
    size_t got = 0;
    for (size_t i = 0; i < idx.size() && got < deficit; ++i) {
        out.push_back(idx[i]);
        got += c[idx[i]].bytes;
    }
    return out;
}

// indices into c (marked SSD now), most routed first, while their bytes fit in `room`
inline std::vector<size_t> ssd_unpick(const std::vector<ssd_cand> & c, size_t room) {
    std::vector<size_t> idx(c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        idx[i] = i;
    }
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (c[a].cnt != c[b].cnt) {
            return c[a].cnt > c[b].cnt;
        }
        return ssd_spread(c[a].layer, c[a].e) > ssd_spread(c[b].layer, c[b].e);
    });
    std::vector<size_t> out;
    size_t used = 0;
    for (size_t i : idx) {
        if (used + c[i].bytes > room) {
            break;
        }
        out.push_back(i);
        used += c[i].bytes;
    }
    return out;
}

// [TAG_FN_L22_SSDFIX] the SSD tier's cache probe. The SSD set is sized against the warm floor (4 GiB of available RAM
// kept free), and direct reads never bring its pages in, so a set marked once stayed on the disk for good: every prompt
// pass read it again (2 passes per 10K prompt). The probe reads it through the mapping when the RAM has room for it above
// the probe's own floor, then watches whether the streamed set stays in RAM; if pages leave, the SSD plan marks again
// (the least-routed missing ones) and the next probe waits twice as long.
//   probe_fits: `want` more bytes in RAM leave more than `min_free` available
inline bool probe_fits(size_t want, size_t avail, size_t min_free) {
    return avail > min_free && want <= avail - min_free;
}
//   probe_verdict: from the streamed set's missing bytes at the watch's start (miss0) and now (miss1): -1 lost (more than
//   max(256 MiB, a quarter of the probed bytes) left RAM), 1 kept (no loss after a prompt or answer ran and >= min_us
//   passed), 0 not yet. Missing bytes, not resident ones: an expert that moves into VRAM leaves the set without a loss
inline int probe_verdict(size_t miss0, size_t miss1, size_t probed, bool activity, int64_t since_us, int64_t min_us) {
    const size_t lost  = miss1 > miss0 ? miss1 - miss0 : 0;
    const size_t limit = std::max<size_t>((size_t) 256 << 20, probed/4);
    if (lost > limit) {
        return -1;
    }
    return activity && since_us >= min_us ? 1 : 0;
}
//   probe_backoff: the wait before the next probe after a verdict (kept: the base; lost: twice the last, at most max_us)
inline int64_t probe_backoff(int64_t last_us, bool kept, int64_t base_us, int64_t max_us) {
    if (kept) {
        return base_us;
    }
    return std::min(max_us, std::max(base_us, last_us*2));
}

// [TAG_FN_L22_RESSET] the stream's lend range inside the hot set's device buffer, as llama_moe_hot_lend places it:
// it ends where the tail starts (or at the buffer's end) and starts `need` bytes below, rounded down to align.
// false: it does not fit
inline bool lend_range(size_t size, size_t tail, size_t need, size_t align, size_t * lo, size_t * hi) {
    const size_t end = tail > 0 && tail < size ? size - tail : size;
    align = std::max<size_t>(align, 1);
    if (need == 0 || need + align > end) {
        return false;
    }
    *lo = (end - need)/align*align;
    *hi = end;
    return true;
}

inline bool overlaps(uintptr_t a_lo, uintptr_t a_hi, uintptr_t lo, uintptr_t hi) {
    return a_hi > lo && a_lo < hi;
}

inline size_t align_up(size_t x, size_t a) {
    return (x + a - 1)/a*a;
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_L22_SSDTIER] direct I/O: mapped ranges read again from their file with unbuffered overlapped reads.
// n threads, each with its own read in flight (queue depth = threads, no completion port). A piece whose file offset
// and destination share the sector phase reads its aligned middle straight into the destination; everything else goes
// through the thread's aligned bounce buffer. Nothing is written outside [dst, dst + len).

struct dio_req {
    const void * src;
    size_t       len;
    void *       dst;
};

struct dio_stats {
    uint64_t bytes   = 0; // bytes delivered to callers
    uint64_t reads   = 0; // read calls
    uint64_t direct  = 0; // bytes read straight into a destination
    uint64_t fails   = 0; // pieces that failed or were cancelled
    uint64_t tickets = 0;
    uint64_t timeouts = 0;
};

class dio_engine {
public:
    static constexpr size_t SECTOR = 4096;
    static constexpr size_t PIECE  = (size_t) 4 << 20;

    dio_engine() = default;
    dio_engine(const dio_engine &) = delete;
    dio_engine & operator=(const dio_engine &) = delete;
    ~dio_engine() { close(); }

    // a mapped range [p, p + n) holds bytes [off, off + n) of file `file` (index into the paths given to open)
    void add_region(const void * p, size_t n, int file, uint64_t off) {
        regions.push_back({ (uintptr_t) p, (uintptr_t) p + n, file, off });
    }

    // the files (UTF-8 paths) and the reader threads; false + err when a file does not open (nothing runs then)
    bool open(const std::vector<std::string> & paths, int n_threads, std::string & err) {
#ifdef _WIN32
        close();
        std::sort(regions.begin(), regions.end(), [](const region & a, const region & b) { return a.lo < b.lo; });
        for (const auto & path : paths) {
            HANDLE h = INVALID_HANDLE_VALUE;
            if (!path.empty()) {
                const int nw = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
                std::wstring w(nw > 0 ? nw - 1 : 0, L'\0');
                if (nw > 1) {
                    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), nw);
                }
                h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                        OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
                if (h == INVALID_HANDLE_VALUE) {
                    err = "CreateFileW(" + path + ") failed: error " + std::to_string((unsigned long) GetLastError());
                    close();
                    return false;
                }
            }
            files.push_back(h);
        }
        stop.store(false);
        n_threads = std::max(1, std::min(32, n_threads));
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([this]() { run(); });
        }
        running.store(true);
        return true;
#else
        (void) paths;
        (void) n_threads;
        err = "direct I/O is implemented for Windows only";
        return false;
#endif
    }

    bool is_open() const { return running.load(); }

    // every src range must lie in one region; -1: not all of them do (or not open), nothing is queued then
    int submit(const dio_req * r, int n) {
        if (!running.load() || r == nullptr || n <= 0) {
            return -1;
        }
        std::vector<piece> ps;
        auto t = std::make_shared<ticket>();
        for (int i = 0; i < n; ++i) {
            if (r[i].len == 0) {
                continue;
            }
            const region * g = find(r[i].src, r[i].len);
            if (g == nullptr || r[i].dst == nullptr) {
                return -1;
            }
            const uint64_t off = g->off + ((uintptr_t) r[i].src - g->lo);
            for (size_t o = 0; o < r[i].len; o += PIECE) {
                ps.push_back({ g->file, off + o, std::min(PIECE, r[i].len - o), (char *) r[i].dst + o, t });
            }
        }
        int id;
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (stop.load() || workers.empty()) {
                return -1; // closing: no reader would take the pieces
            }
            id = ++next_id;
            if (id <= 0) {
                id = next_id = 1;
            }
            t->left.store((int) ps.size());
            tickets[id] = t;
            for (auto & p : ps) {
                queue.push_back(std::move(p));
            }
        }
        cv.notify_all();
        {
            std::lock_guard<std::mutex> g(smtx);
            st.tickets++;
        }
        return id;
    }

    // true when every byte of the ticket arrived. Bounded: after timeout_ms the rest is cancelled, the reads in flight
    // drain, false comes back (the caller then reads the mapping). An unknown ticket: false
    bool wait(int id, uint32_t timeout_ms) {
        std::shared_ptr<ticket> t;
        {
            std::lock_guard<std::mutex> lk(mtx);
            auto it = tickets.find(id);
            if (it == tickets.end()) {
                return false;
            }
            t = it->second;
            tickets.erase(it);
        }
        std::unique_lock<std::mutex> lk(t->m);
        const bool in_time = t->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&]() { return t->left.load() == 0; });
        if (!in_time) {
            t->cancel.store(true);
            t->cv.wait(lk, [&]() { return t->left.load() == 0; }); // cancelled reads complete promptly
            std::lock_guard<std::mutex> g(smtx);
            st.timeouts++;
        }
        return in_time && !t->failed.load();
    }

    dio_stats stats() const {
        std::lock_guard<std::mutex> g(smtx);
        return st;
    }

    void close() {
        running.store(false);
        if (!workers.empty()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                stop.store(true);
            }
            cv.notify_all();
            for (auto & w : workers) {
                w.join();
            }
            workers.clear();
        }
        // pieces nobody will read: their tickets fail
        std::deque<piece> q;
        {
            std::lock_guard<std::mutex> lk(mtx);
            q.swap(queue);
        }
        for (auto & p : q) {
            finish(p, false, 0, 0);
        }
#ifdef _WIN32
        for (HANDLE h : files) {
            if (h != INVALID_HANDLE_VALUE) {
                CloseHandle(h);
            }
        }
#endif
        files.clear();
    }

private:
    struct region {
        uintptr_t lo;
        uintptr_t hi;
        int       file;
        uint64_t  off;
    };
    struct ticket {
        std::atomic<int>        left{0};
        std::atomic<bool>       failed{false};
        std::atomic<bool>       cancel{false};
        std::mutex              m;
        std::condition_variable cv;
    };
    struct piece {
        int                     file;
        uint64_t                off;
        size_t                  len;
        char *                  dst;
        std::shared_ptr<ticket> t;
    };

    const region * find(const void * p, size_t n) const {
        const uintptr_t a = (uintptr_t) p;
        auto it = std::upper_bound(regions.begin(), regions.end(), a, [](uintptr_t x, const region & g) { return x < g.lo; });
        if (it == regions.begin()) {
            return nullptr;
        }
        --it;
        return a >= it->lo && a + n <= it->hi && a + n >= a ? &*it : nullptr;
    }

    void finish(piece & p, bool ok, uint64_t bytes, uint64_t direct) {
        {
            std::lock_guard<std::mutex> g(smtx);
            st.bytes  += ok ? bytes : 0;
            st.direct += ok ? direct : 0;
            st.fails  += ok ? 0 : 1;
        }
        if (!ok) {
            p.t->failed.store(true);
        }
        std::lock_guard<std::mutex> lk(p.t->m);
        if (p.t->left.fetch_sub(1) == 1) {
            p.t->cv.notify_all();
        }
    }

#ifdef _WIN32
    // one overlapped read of [pos, pos + len) (sector multiples, aligned buffer) with the cancel flag checked every
    // 50 ms; got = bytes read (fewer at the end of the file)
    bool read_at(HANDLE h, HANDLE ev, uint64_t pos, void * buf, size_t len, size_t * got, const std::atomic<bool> & cancel) {
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (pos & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (pos >> 32);
        ov.hEvent     = ev;
        ResetEvent(ev);
        {
            std::lock_guard<std::mutex> g(smtx);
            st.reads++;
        }
        if (!ReadFile(h, buf, (DWORD) len, nullptr, &ov) && GetLastError() != ERROR_IO_PENDING) {
            return GetLastError() == ERROR_HANDLE_EOF ? (*got = 0, true) : false;
        }
        while (WaitForSingleObject(ev, 50) == WAIT_TIMEOUT) {
            if (cancel.load() || stop.load()) {
                CancelIoEx(h, &ov);
                break;
            }
        }
        DWORD n = 0;
        if (!GetOverlappedResult(h, &ov, &n, TRUE)) {
            if (GetLastError() == ERROR_HANDLE_EOF) {
                *got = 0;
                return true;
            }
            return false;
        }
        *got = n;
        return true;
    }

    // bytes [off, off + len) of the file into dst, through the bounce buffer bb (PIECE + 2 sectors)
    bool read_piece(HANDLE h, HANDLE ev, piece & p, char * bb, uint64_t * direct) {
        const uint64_t a0 = p.off/SECTOR*SECTOR;
        const uint64_t e  = p.off + p.len;
        const uint64_t i0 = align_up(p.off, SECTOR);
        const uint64_t i1 = e/SECTOR*SECTOR;
        const bool same_phase = ((uintptr_t) p.dst - (uintptr_t) (p.off - a0)) % SECTOR == 0;
        size_t got = 0;
        if (same_phase && i1 > i0) {
            // the aligned middle straight into dst, the partial head and tail sectors through the bounce buffer
            if (!read_at(h, ev, i0, p.dst + (i0 - p.off), (size_t) (i1 - i0), &got, p.t->cancel) || got < i1 - i0) {
                return false;
            }
            *direct = i1 - i0;
            if (i0 > p.off) {
                if (!read_at(h, ev, a0, bb, SECTOR, &got, p.t->cancel) || got < i0 - a0) {
                    return false;
                }
                memcpy(p.dst, bb + (p.off - a0), (size_t) (i0 - p.off));
            }
            if (e > i1) {
                if (!read_at(h, ev, i1, bb, SECTOR, &got, p.t->cancel) || got < e - i1) {
                    return false;
                }
                memcpy(p.dst + (i1 - p.off), bb, (size_t) (e - i1));
            }
            return true;
        }
        const size_t span = (size_t) (align_up((size_t) e, SECTOR) - a0);
        if (!read_at(h, ev, a0, bb, span, &got, p.t->cancel) || got < e - a0) {
            return false;
        }
        memcpy(p.dst, bb + (p.off - a0), p.len);
        return true;
    }
#endif

    void run() {
#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); // mostly waiting: completions handled at once
        char * bb = (char *) VirtualAlloc(nullptr, PIECE + 2*SECTOR, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        for (;;) {
            piece p;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]() { return stop.load() || !queue.empty(); });
                if (queue.empty()) {
                    break; // stop with nothing queued
                }
                p = std::move(queue.front());
                queue.pop_front();
            }
            uint64_t direct = 0;
            const bool ok = bb != nullptr && ev != nullptr && !p.t->cancel.load() && !stop.load() &&
                            p.file >= 0 && p.file < (int) files.size() && files[p.file] != INVALID_HANDLE_VALUE &&
                            read_piece(files[p.file], ev, p, bb, &direct);
            finish(p, ok, p.len, direct);
        }
        if (ev) {
            CloseHandle(ev);
        }
        if (bb) {
            VirtualFree(bb, 0, MEM_RELEASE);
        }
#endif
    }

    std::vector<region>                               regions;
#ifdef _WIN32
    std::vector<HANDLE>                               files;
#else
    std::vector<int>                                  files;
#endif
    std::vector<std::thread>                          workers;
    std::mutex                                        mtx;
    std::condition_variable                           cv;
    std::deque<piece>                                 queue;
    std::unordered_map<int, std::shared_ptr<ticket>>  tickets;
    int                                               next_id = 0;
    std::atomic<bool>                                 stop{false};
    std::atomic<bool>                                 running{false};
    mutable std::mutex                                smtx;
    dio_stats                                         st;
};

} // namespace fn_res
