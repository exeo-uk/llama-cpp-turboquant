// [TAG_FN_L18_FIRSTANS_STALL] see llama-fn-stall.h

#include "llama-fn-stall.h"

#include "llama-fn-tier.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
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

namespace {

int64_t st_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct st_mark {
    std::atomic<uint64_t> seq{0}; // written last: 2*k+1 while being written, 2*k+2 when complete
    int64_t               t   = 0;
    uint32_t              tid = 0;
    const char *          what = nullptr;
    int64_t               a = 0;
    int64_t               b = 0;
};

constexpr size_t N_MARKS = 8192;

#ifdef _WIN32

struct st_thr {
    uint32_t tid;
    uint8_t  state;
    uint8_t  wait;
    int8_t   prio;
    uint8_t  pad;
    uint64_t cpu;   // kernel + user, 100 ns
    uint32_t csw;
};

struct st_sample {
    int64_t  t = 0;
    uint64_t cpu = 0;       // process kernel + user, 100 ns
    uint64_t sys_cpu = 0;   // the System process (pid 4)
    uint64_t other_cpu = 0; // every other process but the idle one
    uint64_t faults = 0;
    uint64_t hard = 0;
    uint64_t read_b = 0;
    uint64_t ws = 0;
    uint64_t avail = 0;
    int      warm = 0;      // 0 not running, 1 reading, 2 paused for an answer step
    size_t   warm_mark = 0;
    std::vector<st_thr> thr;
    std::vector<st_thr> sys; // the System process's threads that run or are ready now
};

// NtQuerySystemInformation(SystemProcessInformation) records (x64 layout)
struct nt_unicode {
    uint16_t len;
    uint16_t max;
    uint32_t pad;
    wchar_t * buf;
};
struct nt_thread_info {
    int64_t  kernel;
    int64_t  user;
    int64_t  create;
    uint32_t wait_time;
    uint32_t pad0;
    void *   start;
    void *   pid;
    void *   tid;
    int32_t  prio;
    int32_t  base_prio;
    uint32_t csw;
    uint32_t state;
    uint32_t wait_reason;
    uint32_t pad1;
};
struct nt_proc_info {
    uint32_t   next;
    uint32_t   n_threads;
    int64_t    ws_private;
    uint32_t   hard_faults;
    uint32_t   n_threads_hw;
    uint64_t   cycles;
    int64_t    create;
    int64_t    user;
    int64_t    kernel;
    nt_unicode name;
    int32_t    base_prio;
    uint32_t   pad0;
    void *     pid;
    void *     ppid;
    uint32_t   handles;
    uint32_t   session;
    uintptr_t  key;
    size_t     peak_vsize;
    size_t     vsize;
    uint32_t   faults;
    uint32_t   pad1;
    size_t     peak_ws;
    size_t     ws;
    size_t     q_peak_paged;
    size_t     q_paged;
    size_t     q_peak_np;
    size_t     q_np;
    size_t     pagefile;
    size_t     peak_pagefile;
    size_t     private_pages;
    int64_t    read_ops;
    int64_t    write_ops;
    int64_t    other_ops;
    int64_t    read_bytes;
    int64_t    write_bytes;
    int64_t    other_bytes;
    // nt_thread_info threads[n_threads]
};
static_assert(sizeof(nt_thread_info) == 80, "SYSTEM_THREAD_INFORMATION layout");
static_assert(sizeof(nt_proc_info) == 256, "SYSTEM_PROCESS_INFORMATION layout");

using nt_query_t = LONG (WINAPI *)(ULONG, PVOID, ULONG, PULONG);
using set_desc_t = HRESULT (WINAPI *)(HANDLE, PCWSTR);
using get_desc_t = HRESULT (WINAPI *)(HANDLE, PWSTR *);

struct st_state {
    bool       on        = false;
    int        period_ms = 10;
    int        win_ms    = 1500;
    int        slow_ms   = 250;
    int        prompt_ms = 0;   // LLAMA_FN_L18_STALL_PROMPT: also dump after a wider decode slower than this (0: never)
    nt_query_t query     = nullptr;
    set_desc_t set_desc  = nullptr;
    get_desc_t get_desc  = nullptr;

    std::vector<st_mark> marks = std::vector<st_mark>(N_MARKS);
    std::atomic<uint64_t> n_marks{0};

    std::mutex             mtx; // the sample ring
    std::vector<st_sample> ring;
    size_t                 n_samples = 0;
    std::vector<uint8_t>   buf;

    std::mutex                                 dump_mtx;
    std::unordered_map<uint32_t, std::string>  names;
    int64_t                                    last_dump_us = 0;
    uint64_t                                   n_dumps = 0;

    std::thread       th;
    std::atomic<bool> stop{false};

    // the decode that runs now (0: none) and when the watchdog dumped it last
    std::atomic<int64_t> dec_t0{0};
    std::atomic<int>     dec_n{0};
    int64_t              dec_dump_us = 0;
};

st_state * g_st = nullptr;

} // namespace
void llama_fn_stall_dump(const char * why);
namespace {

st_state * st_get() {
    static std::once_flag once;
    std::call_once(once, []() {
        const char * e = getenv("LLAMA_FN_L18_STALL");
        if (e == nullptr || atoi(e) <= 0) {
            return;
        }
        auto * s = new st_state();
        s->period_ms = std::max(2, std::min(1000, atoi(e)));
        if (const char * v = getenv("LLAMA_FN_L18_STALL_WIN"))  { s->win_ms  = std::max(100, std::min(10000, atoi(v))); }
        if (const char * v = getenv("LLAMA_FN_L18_STALL_SLOW")) { s->slow_ms = std::max(1, atoi(v)); }
        if (const char * v = getenv("LLAMA_FN_L18_STALL_PROMPT")) { s->prompt_ms = std::max(0, atoi(v)); }
        HMODULE nt = GetModuleHandleW(L"ntdll.dll");
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        s->query    = nt  ? (nt_query_t) (void *) GetProcAddress(nt, "NtQuerySystemInformation") : nullptr;
        s->set_desc = k32 ? (set_desc_t) (void *) GetProcAddress(k32, "SetThreadDescription") : nullptr;
        s->get_desc = k32 ? (get_desc_t) (void *) GetProcAddress(k32, "GetThreadDescription") : nullptr;
        if (s->query == nullptr) {
            LLAMA_LOG_WARN("fn_stall: [TAG_FN_L18_FIRSTANS_STALL] NtQuerySystemInformation not found - off\n");
            delete s;
            return;
        }
        s->ring.resize((size_t) std::max(64, 2*s->win_ms/s->period_ms + 16));
        s->buf.resize((size_t) 8 << 20);
        s->on = true;
        g_st  = s;
        s->th = std::thread([s]() {
            if (s->set_desc) {
                s->set_desc(GetCurrentThread(), L"fn-stall");
            }
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
            const uintptr_t me = (uintptr_t) GetCurrentProcessId();
            while (!s->stop.load(std::memory_order_relaxed)) {
                const int64_t t = st_now_us();
                ULONG need = 0;
                LONG rc = s->query(5, s->buf.data(), (ULONG) s->buf.size(), &need);
                if (rc == (LONG) 0xC0000004L) { // STATUS_INFO_LENGTH_MISMATCH
                    s->buf.resize((size_t) need + ((size_t) 1 << 20));
                    continue;
                }
                if (rc >= 0) {
                    st_sample smp;
                    smp.t = t;
                    size_t off = 0;
                    for (;;) {
                        const auto * p = (const nt_proc_info *) (s->buf.data() + off);
                        const uintptr_t pid = (uintptr_t) p->pid;
                        const uint64_t  cpu = (uint64_t) p->kernel + (uint64_t) p->user;
                        if (pid == me) {
                            smp.cpu    = cpu;
                            smp.faults = p->faults;
                            smp.hard   = p->hard_faults;
                            smp.read_b = (uint64_t) p->read_bytes;
                            smp.ws     = p->ws;
                            const auto * th = (const nt_thread_info *) (p + 1);
                            smp.thr.reserve(p->n_threads);
                            for (uint32_t i = 0; i < p->n_threads; ++i) {
                                st_thr x;
                                x.tid   = (uint32_t) (uintptr_t) th[i].tid;
                                x.state = (uint8_t) std::min<uint32_t>(255, th[i].state);
                                x.wait  = (uint8_t) std::min<uint32_t>(255, th[i].wait_reason);
                                x.prio  = (int8_t) th[i].prio;
                                x.pad   = 0;
                                x.cpu   = (uint64_t) th[i].kernel + (uint64_t) th[i].user;
                                x.csw   = th[i].csw;
                                smp.thr.push_back(x);
                            }
                        } else if (pid == 4) {
                            smp.sys_cpu = cpu;
                            const auto * th = (const nt_thread_info *) (p + 1);
                            for (uint32_t i = 0; i < p->n_threads; ++i) {
                                if (th[i].state == 2 || th[i].state == 1 || th[i].state == 7) {
                                    st_thr x;
                                    x.tid   = (uint32_t) (uintptr_t) th[i].tid;
                                    x.state = (uint8_t) th[i].state;
                                    x.wait  = 0;
                                    x.prio  = (int8_t) th[i].prio;
                                    x.pad   = 0;
                                    x.cpu   = (uint64_t) th[i].kernel + (uint64_t) th[i].user;
                                    x.csw   = th[i].csw;
                                    smp.sys.push_back(x);
                                }
                            }
                        } else if (pid != 0) {
                            smp.other_cpu += cpu;
                        }
                        if (p->next == 0) {
                            break;
                        }
                        off += p->next;
                    }
                    MEMORYSTATUSEX ms;
                    ms.dwLength = sizeof(ms);
                    if (GlobalMemoryStatusEx(&ms)) {
                        smp.avail = ms.ullAvailPhys;
                    }
                    smp.warm = llama_fn_warm_state(&smp.warm_mark);
                    std::lock_guard<std::mutex> lk(s->mtx);
                    s->ring[s->n_samples % s->ring.size()] = std::move(smp);
                    s->n_samples++;
                }
                // a decode that has run longer than the window: a dump now and then while it runs
                const int64_t d0 = s->dec_t0.load(std::memory_order_acquire);
                if (d0 > 0 && t - d0 > (int64_t) s->win_ms*1000 && t - s->dec_dump_us > (int64_t) s->win_ms*1000) {
                    s->dec_dump_us = t;
                    char why[96];
                    snprintf(why, sizeof(why), "a %d-token decode has run for %.0f ms", s->dec_n.load(), (t - d0)/1000.0);
                    llama_fn_stall_dump(why);
                }
                const int64_t el = st_now_us() - t;
                const int64_t sl = (int64_t) s->period_ms*1000 - el;
                if (sl > 0) {
                    Sleep((DWORD) std::max<int64_t>(1, sl/1000));
                }
            }
        });
        s->th.detach(); // lives as long as the process
        LLAMA_LOG_WARN("fn_stall: [TAG_FN_L18_FIRSTANS_STALL] thread states sampled every %d ms; dumps of the last %d ms at a "
                "bridge wait timeout and after a decode of <= 64 tokens slower than %d ms\n", s->period_ms, s->win_ms, s->slow_ms);
    });
    return g_st;
}

char st_letter(uint8_t state, uint8_t wait) {
    switch (state) {
        case 1: case 7: return 'r'; // ready, deferred ready
        case 2: return 'R';
        case 3: return 's';         // standby
        case 6: return 'T';         // transition (kernel stack out)
        case 5: break;              // waiting: the reason below
        default: return '.';
    }
    static const char * letters = "EFPADSUefpadsuvqlLVozktwcCKxHmQiny"; // KWAIT_REASON 0..33
    if (wait < 34) {
        return letters[wait];
    }
    switch (wait) {
        case 34: return 'M'; // WrFastMutex
        case 35: return 'G'; // WrGuardedMutex
        case 36: return 'O'; // WrRundown
        case 37: return 'b'; // WrAlertByThreadId (SRW locks, condition variables, WaitOnAddress)
        case 38: return 'N'; // WrDeferredPreempt
        case 39: return 'Y'; // WrPhysicalFault
        default: return '?';
    }
}

// "aff <mask> ideal <cpu>" of a thread of this process ("" when it cannot be read)
std::string st_affinity(uint32_t tid) {
    using nt_qit_t = LONG (WINAPI *)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    static const nt_qit_t qit = (nt_qit_t) (void *) GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread");
    struct tbi {
        LONG      exit_status;
        PVOID     teb;
        PVOID     pid;
        PVOID     tid;
        ULONG_PTR affinity;
        LONG      prio;
        LONG      base_prio;
    } b = {};
    std::string out;
    HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, tid);
    if (!h) {
        return out;
    }
    char tmp[96];
    if (qit && qit(h, 0, &b, sizeof(b), nullptr) >= 0) {
        snprintf(tmp, sizeof(tmp), "aff %llx", (unsigned long long) b.affinity);
        out = tmp;
    }
    PROCESSOR_NUMBER pn = {};
    if (GetThreadIdealProcessorEx(h, &pn)) {
        snprintf(tmp, sizeof(tmp), " ideal %u", (unsigned) pn.Number);
        out += tmp;
    }
    CloseHandle(h);
    return out;
}

std::string st_name(st_state * s, uint32_t tid) {
    auto it = s->names.find(tid);
    if (it != s->names.end()) {
        return it->second;
    }
    std::string name;
    if (s->get_desc) {
        HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid);
        if (h) {
            PWSTR w = nullptr;
            if (SUCCEEDED(s->get_desc(h, &w)) && w) {
                for (const wchar_t * c = w; *c; ++c) {
                    name.push_back(*c < 128 ? (char) *c : '?');
                }
                LocalFree(w);
            }
            CloseHandle(h);
        }
    }
    if (!name.empty()) {
        s->names[tid] = name; // unnamed threads may get a name later
    }
    return name;
}

#endif // _WIN32

} // namespace

#ifdef _WIN32

// [TAG_FN_L18_FIRSTANS_STALL] see llama-fn-tier.h: the warm pass's state for the sampler
bool llama_fn_stall_on() {
    st_state * s = st_get();
    return s && s->on;
}

void llama_fn_thread_name(const char * name, int k) {
    st_state * s = st_get();
    if (!s || !s->set_desc || name == nullptr) {
        return;
    }
    wchar_t w[64];
    char    c[64];
    if (k >= 0) {
        snprintf(c, sizeof(c), "%s %d", name, k);
    } else {
        snprintf(c, sizeof(c), "%s", name);
    }
    size_t i = 0;
    for (; c[i] && i + 1 < sizeof(w)/sizeof(w[0]); ++i) {
        w[i] = (wchar_t) (unsigned char) c[i];
    }
    w[i] = 0;
    s->set_desc(GetCurrentThread(), w);
}

void llama_fn_stall_mark(const char * what, int64_t a, int64_t b) {
    st_state * s = st_get();
    if (!s) {
        return;
    }
    const uint64_t k = s->n_marks.fetch_add(1, std::memory_order_relaxed);
    st_mark & m = s->marks[k % N_MARKS];
    m.seq.store(2*k + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    m.t    = st_now_us();
    m.tid  = (uint32_t) GetCurrentThreadId();
    m.what = what;
    m.a    = a;
    m.b    = b;
    m.seq.store(2*k + 2, std::memory_order_release);
}

void llama_fn_stall_dump(const char * why) {
    st_state * s = st_get();
    if (!s) {
        return;
    }
    std::lock_guard<std::mutex> dlk(s->dump_mtx);
    const int64_t now = st_now_us();
    const int64_t t0  = now - (int64_t) s->win_ms*1000;
    std::vector<st_sample> smp;
    {
        std::lock_guard<std::mutex> lk(s->mtx);
        const size_t n = std::min(s->n_samples, s->ring.size());
        for (size_t i = s->n_samples - n; i < s->n_samples; ++i) {
            const st_sample & x = s->ring[i % s->ring.size()];
            if (x.t >= t0 - (int64_t) s->period_ms*1000) {
                smp.push_back(x);
            }
        }
    }
    s->n_dumps++;
    s->last_dump_us = now;
    LLAMA_LOG_WARN("fn_stall: ===== dump %llu: %s (last %d ms, %zu samples) =====\n", (unsigned long long) s->n_dumps, why,
            s->win_ms, smp.size());
    if (smp.size() < 2) {
        return;
    }
    // the marks of the window, oldest first
    const uint64_t nm = s->n_marks.load(std::memory_order_acquire);
    std::vector<std::tuple<int64_t, uint32_t, const char *, int64_t, int64_t>> mk;
    for (uint64_t k = nm > N_MARKS ? nm - N_MARKS : 0; k < nm; ++k) {
        const st_mark & m = s->marks[k % N_MARKS];
        if (m.seq.load(std::memory_order_acquire) != 2*k + 2) {
            continue;
        }
        const int64_t t = m.t;
        const uint32_t tid = m.tid;
        const char * w = m.what;
        const int64_t a = m.a, b = m.b;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (m.seq.load(std::memory_order_relaxed) != 2*k + 2 || t < smp.front().t) {
            continue;
        }
        mk.emplace_back(t, tid, w, a, b);
    }
    // 50 ms buckets from the first sample
    const int64_t tb0 = smp.front().t;
    const int64_t bw  = 50000;
    const int     nb  = (int) ((smp.back().t - tb0)/bw) + 1;
    std::vector<int> first_of(nb, -1), last_of(nb, -1);
    for (int i = 0; i < (int) smp.size(); ++i) {
        const int b = (int) ((smp[i].t - tb0)/bw);
        if (first_of[b] < 0) {
            first_of[b] = i;
        }
        last_of[b] = i;
    }
    LLAMA_LOG_WARN("fn_stall: t(ms) | server / System / other cores | faults hard | read MB | avail GB | ws GB | warm (0 off, 1 "
            "reads, 2 paused) pieces\n");
    int prev = 0;
    for (int b = 0; b < nb; ++b) {
        if (last_of[b] < 0) {
            continue;
        }
        const st_sample & a = smp[prev];
        const st_sample & z = smp[last_of[b]];
        const double dt = std::max<int64_t>(1, z.t - a.t)*10.0; // in 100 ns units
        if (last_of[b] != prev) {
            LLAMA_LOG_WARN("fn_stall: %+6.0f | %5.2f %5.2f %5.2f | %6llu %5llu | %7.1f | %6.2f | %5.2f | %d %zu\n",
                    (z.t - now)/1000.0, (z.cpu - a.cpu)/dt, (z.sys_cpu - a.sys_cpu)/dt,
                    z.other_cpu >= a.other_cpu ? (z.other_cpu - a.other_cpu)/dt : 0.0,
                    (unsigned long long) (z.faults - a.faults), (unsigned long long) (z.hard - a.hard),
                    (z.read_b - a.read_b)/1048576.0, z.avail/1073741824.0, z.ws/1073741824.0, z.warm, z.warm_mark);
        }
        prev = last_of[b];
    }
    // per thread: one letter per bucket ('#' >= 50 % of the bucket on a CPU, '+' >= 10 %, else the state of its last sample)
    std::unordered_map<uint32_t, size_t> idx;
    struct row { uint32_t tid; std::string tl; uint64_t cpu0 = 0, cpu1 = 0; uint32_t csw0 = 0, csw1 = 0; int prio = 0; bool busy = false; };
    std::vector<row> rows;
    for (int b = 0; b < nb; ++b) {
        if (last_of[b] < 0) {
            for (auto & r : rows) {
                r.tl.push_back(' ');
            }
            continue;
        }
        const st_sample & z = smp[last_of[b]];
        const st_sample & a = smp[first_of[b] > 0 ? first_of[b] - 1 : 0];
        std::unordered_map<uint32_t, const st_thr *> pa;
        for (const auto & x : a.thr) {
            pa[x.tid] = &x;
        }
        for (const auto & x : z.thr) {
            auto it = idx.find(x.tid);
            if (it == idx.end()) {
                it = idx.emplace(x.tid, rows.size()).first;
                row r;
                r.tid  = x.tid;
                r.tl   = std::string((size_t) b, ' ');
                auto p = pa.find(x.tid);
                r.cpu0 = p != pa.end() ? p->second->cpu : x.cpu;
                r.csw0 = p != pa.end() ? p->second->csw : x.csw;
                rows.push_back(r);
            }
            row & r = rows[it->second];
            auto p = pa.find(x.tid);
            const uint64_t c0 = p != pa.end() ? p->second->cpu : r.cpu1 ? r.cpu1 : x.cpu;
            const double share = (double) (x.cpu - std::min(x.cpu, c0))/std::max<double>(1.0, (z.t - a.t)*10.0);
            char ch = share >= 0.5 ? '#' : share >= 0.1 ? '+' : st_letter(x.state, x.wait);
            r.tl.resize((size_t) b, ' ');
            r.tl.push_back(ch);
            r.cpu1 = x.cpu;
            r.csw1 = x.csw;
            r.prio = x.prio;
            if (share >= 0.1 || x.state != 5 || (x.wait != 6 && x.wait != 13 && x.wait != 15 && x.wait != 37 && x.wait != 4 &&
                    x.wait != 11)) {
                r.busy = true; // busy, or waiting on something other than a plain wait / sleep / queue
            }
        }
    }
    int n_quiet = 0;
    LLAMA_LOG_WARN("fn_stall: threads, one letter per 50 ms: # >= 50%% CPU, + >= 10%%, R running, r ready, U/u user wait, b "
            "SRW/CV/WaitOnAddress, d/D sleep, q queue, E/e executive, P/p page-in, H push lock, F/f free page, V virtual "
            "memory, x resource, G/M mutex\n");
    for (auto & r : rows) {
        const std::string nm = st_name(s, r.tid);
        if (nm.empty() && !r.busy) {
            n_quiet++;
            continue;
        }
        LLAMA_LOG_WARN("fn_stall: %-14s %6u p%-2d |%s| %7.1f ms cpu, %u switches, %s\n", nm.empty() ? "?" : nm.c_str(), r.tid,
                r.prio, r.tl.c_str(), (r.cpu1 - std::min(r.cpu1, r.cpu0))/1e4, r.csw1 - r.csw0, st_affinity(r.tid).c_str());
    }
    LLAMA_LOG_WARN("fn_stall: %d unnamed threads idle in plain waits all window\n", n_quiet);
    // the System process's threads that ran (R) or waited for a CPU (r) in the most samples of the window
    {
        struct sysrow { int run = 0, ready = 0, prio = 0; uint64_t c0 = 0, c1 = 0; };
        std::unordered_map<uint32_t, sysrow> sr;
        for (const auto & x : smp) {
            for (const auto & t : x.sys) {
                sysrow & r = sr[t.tid];
                (t.state == 2 ? r.run : r.ready)++;
                r.prio = t.prio;
                if (r.c0 == 0) {
                    r.c0 = t.cpu;
                }
                r.c1 = t.cpu;
            }
        }
        std::vector<std::pair<uint32_t, sysrow>> v(sr.begin(), sr.end());
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.run + a.second.ready > b.second.run + b.second.ready; });
        std::string line;
        for (size_t i = 0; i < v.size() && i < 10; ++i) {
            char tmp[96];
            snprintf(tmp, sizeof(tmp), "%s%u p%d R%d r%d %.0fms", line.empty() ? "" : ", ", v[i].first, v[i].second.prio,
                    v[i].second.run, v[i].second.ready, (v[i].second.c1 - v[i].second.c0)/1e4);
            line += tmp;
        }
        LLAMA_LOG_WARN("fn_stall: System threads running (R) / ready (r) in the most of %zu samples: %s\n", smp.size(), line.c_str());
    }
    for (const auto & m : mk) {
        const std::string nm = st_name(s, std::get<1>(m));
        LLAMA_LOG_WARN("fn_stall: mark %+8.1f ms [%s %u] %s %lld %lld\n", (std::get<0>(m) - now)/1000.0,
                nm.empty() ? "?" : nm.c_str(), std::get<1>(m), std::get<2>(m), (long long) std::get<3>(m),
                (long long) std::get<4>(m));
    }
    LLAMA_LOG_WARN("fn_stall: ===== end of dump %llu =====\n", (unsigned long long) s->n_dumps);
}

int64_t llama_fn_stall_decode_begin(int n_tokens, int n_outputs) {
    st_state * s = st_get();
    if (!s) {
        return 0;
    }
    static thread_local bool named = false;
    if (!named) {
        named = true;
        llama_fn_thread_name("decode");
    }
    llama_fn_stall_mark("decode begin (tokens, mtp ctx)", n_tokens, n_outputs);
    const int64_t t0 = st_now_us();
    s->dec_n.store(n_tokens);
    s->dec_t0.store(t0, std::memory_order_release);
    return t0;
}

void llama_fn_stall_decode_end(int64_t t0_us, int n_tokens) {
    st_state * s = st_get();
    if (!s || t0_us == 0) {
        return;
    }
    const int64_t now = st_now_us();
    s->dec_t0.store(0, std::memory_order_release);
    llama_fn_stall_mark("decode end (tokens, us)", n_tokens, now - t0_us);
    if (n_tokens > 64 && s->prompt_ms > 0 && now - t0_us > (int64_t) s->prompt_ms*1000) {
        char why[96];
        snprintf(why, sizeof(why), "a %d-token decode took %.0f ms (prompt)", n_tokens, (now - t0_us)/1000.0);
        llama_fn_stall_dump(why);
        return;
    }
    if (n_tokens <= 64 && now - t0_us > (int64_t) s->slow_ms*1000 && now - s->last_dump_us > 2000000) {
        char why[96];
        snprintf(why, sizeof(why), "a %d-token decode took %.0f ms", n_tokens, (now - t0_us)/1000.0);
        llama_fn_stall_dump(why);
    }
}

#else

bool    llama_fn_stall_on() { return false; }
void    llama_fn_thread_name(const char *, int) {}
void    llama_fn_stall_mark(const char *, int64_t, int64_t) {}
void    llama_fn_stall_dump(const char *) {}
int64_t llama_fn_stall_decode_begin(int, int) { return 0; }
void    llama_fn_stall_decode_end(int64_t, int) {}

#endif
