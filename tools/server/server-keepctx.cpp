#include "server-keepctx.h"
#include "server-common.h"
#include "server-convstore.h" // [TAG_FN_L19_CONVSTORE]

#include "llama.h"
#include "log.h"
#include "speculative.h"
#include "../../src/llama-ext.h" // [TAG_FN_L18_KEEPCTX] llama_state_seq_write_stream_ext

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// unbuffered I/O moves whole sectors: every block is a multiple of this, the last one is padded with zeros
static constexpr size_t KC_ALIGN = 4096;
static constexpr size_t KC_BLOCK = 16u*1024*1024;
static constexpr int    KC_NBUF  = 3;

static size_t kc_round_up(size_t n) {
    return (n + KC_ALIGN - 1) / KC_ALIGN * KC_ALIGN;
}

static double kc_ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// FNV-1a, for params.verify only
static uint64_t kc_hash(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

static constexpr uint64_t KC_HASH_SEED = 0xcbf29ce484222325ull;

static unsigned long kc_pid() {
#if defined(_WIN32)
    return (unsigned long) GetCurrentProcessId();
#else
    return (unsigned long) getpid();
#endif
}

//
// one store file: no OS file cache, deleted when closed
//

class server_keepctx_file {
public:
    static std::shared_ptr<server_keepctx_file> create(const std::filesystem::path & path, std::string & err) {
        auto f = std::shared_ptr<server_keepctx_file>(new server_keepctx_file());
#if defined(_WIN32)
        f->h = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_NO_BUFFERING, nullptr);
        if (f->h == INVALID_HANDLE_VALUE) {
            err = "CreateFileW failed, error " + std::to_string((unsigned long) GetLastError());
            return nullptr;
        }
#else
        f->fd = open(path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (f->fd < 0) {
            err = std::string("open failed: ") + strerror(errno);
            return nullptr;
        }
        unlink(path.string().c_str()); // the file lives until the descriptor closes
#endif
        return f;
    }

    ~server_keepctx_file() {
#if defined(_WIN32)
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
        }
#else
        if (fd >= 0) {
            close(fd);
        }
#endif
    }

    // n and off are multiples of KC_ALIGN, buf is KC_ALIGN aligned
    bool pwrite(const void * buf, size_t n, uint64_t off) {
#if defined(_WIN32)
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD done = 0;
        return WriteFile(h, buf, (DWORD) n, &done, &ov) && done == n;
#else
        const uint8_t * p = (const uint8_t *) buf;
        while (n > 0) {
            const ssize_t r = ::pwrite(fd, p, n, (off_t) off);
            if (r <= 0) {
                return false;
            }
            p += r; n -= (size_t) r; off += (uint64_t) r;
        }
        return true;
#endif
    }

    bool pread(void * buf, size_t n, uint64_t off) {
#if defined(_WIN32)
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD done = 0;
        return ReadFile(h, buf, (DWORD) n, &done, &ov) && done == n;
#else
        uint8_t * p = (uint8_t *) buf;
        while (n > 0) {
            const ssize_t r = ::pread(fd, p, n, (off_t) off);
            if (r <= 0) {
                return false;
            }
            p += r; n -= (size_t) r; off += (uint64_t) r;
        }
        return true;
#endif
    }

private:
    server_keepctx_file() = default;
#if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

//
// a thread moves blocks between a few aligned buffers and the file, so the device copies of the caller and the disk
// overlap. Write: put() fills blocks, the thread writes them in order. Read: the thread reads ahead, get() drains.
//

class kc_pipe {
public:
    // [TAG_FN_L19_CONVSTORE_SPILL] base: where in the file the pipe starts (a multiple of KC_ALIGN)
    kc_pipe(server_keepctx_file & file, bool writing, uint64_t n_read, uint64_t base = 0) :
            file(file), writing(writing), n_read(kc_round_up(n_read)), base(base) {
#if defined(_WIN32)
        mem = (uint8_t *) VirtualAlloc(nullptr, KC_NBUF*KC_BLOCK, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        mem = (uint8_t *) std::aligned_alloc(KC_ALIGN, KC_NBUF*KC_BLOCK);
#endif
        if (mem == nullptr) {
            err = true;
            return;
        }
        for (int b = 0; b < KC_NBUF; ++b) {
            q_free.push_back(b);
        }
        th = std::thread([this]() { this->writing ? run_write() : run_read(); });
    }

    ~kc_pipe() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (th.joinable()) {
            th.join();
        }
        if (mem) {
#if defined(_WIN32)
            VirtualFree(mem, 0, MEM_RELEASE);
#else
            std::free(mem);
#endif
        }
    }

    bool     hashing = false;
    uint64_t hash    = KC_HASH_SEED;

    bool put(const void * src, size_t n) {
        if (hashing) {
            hash = kc_hash(hash, src, n);
        }
        const uint8_t * p = (const uint8_t *) src;
        while (n > 0) {
            if (cur < 0 && !take(q_free)) {
                return false;
            }
            const size_t k = std::min(n, KC_BLOCK - cur_off);
            std::memcpy(mem + (size_t) cur*KC_BLOCK + cur_off, p, k);
            cur_off += k; p += k; n -= k; n_logical += k;
            if (cur_off == KC_BLOCK) {
                give(q_full, KC_BLOCK);
            }
        }
        return !failed();
    }

    // write the last block (padded) and wait for the thread; the logical bytes written
    bool finish(uint64_t & n_out) {
        if (cur >= 0 && cur_off > 0) {
            const size_t n = kc_round_up(cur_off);
            std::memset(mem + (size_t) cur*KC_BLOCK + cur_off, 0, n - cur_off);
            give(q_full, n);
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            done = true;
        }
        cv.notify_all();
        if (th.joinable()) {
            th.join();
        }
        n_out = n_logical;
        return !err;
    }

    bool get(void * dst, size_t n) {
        uint8_t * p = (uint8_t *) dst;
        while (n > 0) {
            if (cur < 0 && !take(q_full)) {
                return false;
            }
            const size_t k = std::min(n, len[cur] - cur_off);
            std::memcpy(p, mem + (size_t) cur*KC_BLOCK + cur_off, k);
            cur_off += k; p += k; n -= k; n_logical += k;
            if (cur_off == len[cur]) {
                give(q_free, 0);
            }
        }
        return true;
    }

private:
    bool failed() {
        std::lock_guard<std::mutex> lk(mu);
        return err;
    }

    // the caller waits for a block of q
    bool take(std::deque<int> & q) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]() { return !q.empty() || err; });
        if (q.empty()) {
            return false;
        }
        cur     = q.front();
        cur_off = 0;
        q.pop_front();
        return true;
    }

    void give(std::deque<int> & q, size_t n) {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (&q == &q_full) {
                len[cur] = n;
            }
            q.push_back(cur);
        }
        cv.notify_all();
        cur     = -1;
        cur_off = 0;
    }

    void run_write() {
        uint64_t off = base;
        while (true) {
            int b = -1;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return !q_full.empty() || done || stop; });
                if (q_full.empty() || stop) {
                    return;
                }
                b = q_full.front();
                q_full.pop_front();
            }
            const bool ok = file.pwrite(mem + (size_t) b*KC_BLOCK, len[b], off);
            off += len[b];
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!ok) {
                    err = true;
                }
                q_free.push_back(b);
            }
            cv.notify_all();
            if (!ok) {
                return;
            }
        }
    }

    void run_read() {
        uint64_t off = 0;
        while (off < n_read) {
            int b = -1;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return !q_free.empty() || stop; });
                if (stop) {
                    return;
                }
                b = q_free.front();
                q_free.pop_front();
            }
            const size_t n  = (size_t) std::min<uint64_t>(KC_BLOCK, n_read - off);
            const bool   ok = file.pread(mem + (size_t) b*KC_BLOCK, n, base + off);
            off += n;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!ok) {
                    err = true;
                } else {
                    len[b] = n;
                    q_full.push_back(b);
                }
            }
            cv.notify_all();
            if (!ok) {
                return;
            }
        }
        // past the end: get() asking for more bytes than were written fails instead of waiting
        {
            std::lock_guard<std::mutex> lk(mu);
            err = true;
        }
        cv.notify_all();
    }

    server_keepctx_file & file;
    const bool            writing;
    const uint64_t        n_read;
    const uint64_t        base;

    uint8_t * mem = nullptr;
    size_t    len[KC_NBUF] = {};

    std::mutex              mu;
    std::condition_variable cv;
    std::deque<int>         q_free;
    std::deque<int>         q_full;
    bool                    err  = false;
    bool                    done = false;
    bool                    stop = false;
    std::thread             th;

    // caller side
    int      cur       = -1;
    size_t   cur_off   = 0;
    uint64_t n_logical = 0;
};

static bool kc_write_cb(const void * src, size_t size, void * ud) {
    return ((kc_pipe *) ud)->put(src, size);
}

static bool kc_read_cb(void * dst, size_t size, void * ud) {
    return ((kc_pipe *) ud)->get(dst, size);
}

//
// server_keepctx
//

server_keepctx::server_keepctx(const server_keepctx_params & params) : params_(params) {
    std::error_code ec;
    std::filesystem::create_directories(params_.dir, ec);
    if (!std::filesystem::is_directory(params_.dir, ec)) {
        SRV_ERR("[TAG_FN_L18_KEEPCTX] the store folder '%s' cannot be created\n", params_.dir.c_str());
        return;
    }
    // files of a process that ended without closing them (power loss): a file still open elsewhere cannot be removed
    for (const auto & de : std::filesystem::directory_iterator(params_.dir, ec)) {
        const std::string name = de.path().filename().string();
        if (name.rfind("keepctx-", 0) == 0 && de.path().extension() == ".kc") {
            std::error_code ec_rm;
            std::filesystem::remove(de.path(), ec_rm);
        }
    }
    ok_ = true;

    if (params_.verify) {
        std::string err;
        const bool pass = selftest(err);
        SRV_INF("[TAG_FN_L18_KEEPCTX] store I/O self-test %s%s\n", pass ? "PASS" : "FAIL: ", err.c_str());
        ok_ = pass;
    }
}

server_keepctx::~server_keepctx() = default;

// write a pattern in pieces of awkward sizes through the pipe, read it back in other pieces, compare
bool server_keepctx::selftest(std::string & err) {
    const auto path = std::filesystem::path(params_.dir) / string_format("keepctx-%lu-selftest.kc", kc_pid());
    auto file = server_keepctx_file::create(path, err);
    if (!file) {
        return false;
    }
    const size_t wsz[] = { 1, 4095, 4097, KC_BLOCK + 3, 7, 2*KC_BLOCK - 5, 33, 12345 };
    const size_t rsz[] = { 5, 4096, KC_BLOCK - 1, 17, 3*KC_BLOCK, 1 };
    size_t total = 0;
    for (size_t n : wsz) {
        total += n;
    }
    std::vector<uint8_t> ref(total);
    for (size_t i = 0; i < total; ++i) {
        ref[i] = (uint8_t) ((i * 2654435761u) >> 13);
    }
    uint64_t n_written = 0;
    {
        kc_pipe pipe(*file, true, 0);
        size_t off = 0;
        for (size_t n : wsz) {
            if (!pipe.put(ref.data() + off, n)) {
                err = "write";
                return false;
            }
            off += n;
        }
        if (!pipe.finish(n_written) || n_written != total) {
            err = "finish";
            return false;
        }
    }
    std::vector<uint8_t> got(total);
    {
        kc_pipe pipe(*file, false, total);
        size_t off = 0;
        for (size_t k = 0; off < total; ++k) {
            const size_t n = std::min(rsz[k % (sizeof(rsz)/sizeof(rsz[0]))], total - off);
            if (!pipe.get(got.data() + off, n)) {
                err = "read";
                return false;
            }
            off += n;
        }
    }
    if (got != ref) {
        err = "data differs";
        return false;
    }
    return true;
}

uint64_t server_keepctx::n_bytes() const {
    uint64_t n = 0;
    for (const auto & e : entries) {
        n += e.n_bytes_all();
    }
    return n;
}

uint64_t server_keepctx_entry::n_bytes_all() const {
    uint64_t n = n_bytes;
    for (const auto & s : spilled) {
        n += s.bytes();
    }
    return n;
}

// [TAG_FN_L19_CONVSTORE_SPILL] a running hash of the tokens (media chunks by their id and size). at: positions in
// ascending order; out[k] = the hash of tokens [0, at[k]), 0 when it cannot be computed (past the end, a position inside
// a media chunk)
static void kc_prefix_hashes(const server_tokens & tokens, const std::vector<int64_t> & at, std::vector<uint64_t> & out) {
    const llama_tokens & t = tokens.get_tokens();
    out.assign(at.size(), 0);
    uint64_t h = KC_HASH_SEED;
    size_t   i = 0;
    size_t   k = 0;
    try {
        while (k < at.size()) {
            if (at[k] < 0 || (size_t) at[k] > t.size()) {
                k++; // stays 0
                continue;
            }
            if ((size_t) at[k] == i) {
                out[k++] = h == 0 ? 1 : h;
                continue;
            }
            if ((size_t) at[k] < i) {
                k++; // inside a media chunk
                continue;
            }
            if (t[i] != LLAMA_TOKEN_NULL) {
                h = (h ^ (uint32_t) t[i]) * 0x100000001b3ull;
                ++i;
                continue;
            }
            const auto & chunk = tokens.find_chunk(i);
            if (!chunk) {
                return;
            }
            const char * id = mtmd_input_chunk_get_id(chunk.get());
            const size_t nt = mtmd_input_chunk_get_n_tokens(chunk.get());
            if (nt == 0) {
                return;
            }
            h = kc_hash(h, id ? id : "", id ? strlen(id) : 0);
            h = (h ^ (uint64_t) nt) * 0x100000001b3ull;
            i += nt;
        }
    } catch (const std::exception &) {
        for (; k < at.size(); ++k) {
            out[k] = 0;
        }
    }
}

uint64_t keepctx_prefix_hash(const server_tokens & tokens, size_t n) {
    std::vector<uint64_t> out;
    kc_prefix_hashes(tokens, { (int64_t) n }, out);
    return out[0];
}

int keepctx_spill_validate(const server_tokens & tokens, std::vector<server_keepctx_spilled> & list) {
    if (list.empty()) {
        return 0;
    }
    std::vector<int64_t> at;
    for (const auto & s : list) {
        at.push_back(s.n_tokens);
    }
    std::vector<uint64_t> h;
    kc_prefix_hashes(tokens, at, h);
    int n_drop = 0;
    std::vector<server_keepctx_spilled> keep;
    for (size_t k = 0; k < list.size(); ++k) {
        if (h[k] != 0 && h[k] == list[k].hash) {
            keep.push_back(std::move(list[k]));
        } else {
            n_drop++;
        }
    }
    list = std::move(keep);
    return n_drop;
}

void server_keepctx_spill::keep_first(int64_t n_keep) {
    while (!list.empty() && list.back().n_tokens > n_keep) {
        list.pop_back();
    }
}

bool server_keepctx::spill_write(server_keepctx_spill & spill, const common_prompt_checkpoint & c, uint64_t prefix_hash,
                                 std::string & info) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!spill.file) {
        static std::atomic<uint64_t> n_spill_files{0};
        const auto path = std::filesystem::path(params_.dir) /
                string_format("keepctx-%lu-spill%llu.kc", kc_pid(), (unsigned long long) ++n_spill_files);
        std::string err;
        spill.file = server_keepctx_file::create(path, err);
        spill.end  = 0;
        if (!spill.file) {
            info = "cannot create '" + path.string() + "': " + err;
            return false;
        }
    }
    server_keepctx_spilled s;
    s.n_tokens  = c.n_tokens;
    s.id_task   = c.id_task;
    s.pos_min   = c.pos_min;
    s.pos_max   = c.pos_max;
    s.size_tgt  = c.data_tgt.size();
    s.size_dft  = c.data_dft.size();
    s.size_spec = c.data_spec.size();
    s.off       = spill.end;
    s.hash      = prefix_hash;
    s.file      = spill.file;
    uint64_t n_written = 0;
    bool ok = true;
    {
        kc_pipe pipe(*spill.file, true, 0, s.off);
        pipe.hashing = params_.verify;
        ok = pipe.put(c.data_tgt.data(), c.data_tgt.size()) && pipe.put(c.data_dft.data(), c.data_dft.size()) &&
             pipe.put(c.data_spec.data(), c.data_spec.size());
        ok = pipe.finish(n_written) && ok;
        s.hash_data = pipe.hash;
    }
    if (!ok || n_written != s.bytes()) {
        info = string_format("writing a checkpoint at n_tokens = %lld failed", (long long) c.n_tokens);
        spill.file.reset(); // the next one starts a new file
        spill.end = 0;
        return false;
    }
    spill.end += kc_round_up(s.bytes());
    spill.n_written++;
    spill.bytes_written += s.bytes();
    auto pos = std::upper_bound(spill.list.begin(), spill.list.end(), s.n_tokens,
                                [](int64_t n, const server_keepctx_spilled & x) { return n < x.n_tokens; });
    spill.list.insert(pos, std::move(s));
    const double ms = kc_ms_since(t0);
    info = string_format("%.1f MiB in %.0f ms", c.size()/1048576.0, ms);
    return true;
}

bool server_keepctx::spill_read(const server_keepctx_spilled & s, common_prompt_checkpoint & c, std::string & info) {
    if (!s.file) {
        info = "no file";
        return false;
    }
    c.n_tokens = s.n_tokens;
    c.id_task  = s.id_task;
    c.pos_min  = s.pos_min;
    c.pos_max  = s.pos_max;
    c.data_tgt .resize(s.size_tgt);
    c.data_dft .resize(s.size_dft);
    c.data_spec.resize(s.size_spec);
    kc_pipe pipe(*s.file, false, s.bytes(), s.off);
    if (!(pipe.get(c.data_tgt.data(), c.data_tgt.size()) && pipe.get(c.data_dft.data(), c.data_dft.size()) &&
          pipe.get(c.data_spec.data(), c.data_spec.size()))) {
        info = "read failed";
        return false;
    }
    if (params_.verify) {
        uint64_t h = KC_HASH_SEED;
        h = kc_hash(h, c.data_tgt.data(), c.data_tgt.size());
        h = kc_hash(h, c.data_dft.data(), c.data_dft.size());
        h = kc_hash(h, c.data_spec.data(), c.data_spec.size());
        info = h == s.hash_data ? "verify: identical" : "verify: DIFFERS";
    }
    return true;
}

// [TAG_FN_L19_CONVSTORE_EVICT]
bool server_keepctx::evict_one(const server_keepctx_entry * keep) {
    auto victim = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (&*it == keep) {
            continue;
        }
        if (!params_.gdsf) {
            victim = it; // least recently stored first
            break;
        }
        if (victim == entries.end() || it->value < victim->value) {
            victim = it;
        }
    }
    if (victim == entries.end()) {
        return false;
    }
    if (params_.gdsf) {
        gdsf_L_ = std::max(gdsf_L_, victim->value);
        n_evicted_gdsf++;
    }
    entries.erase(victim);
    return true;
}

int server_keepctx::drop_prefixes_of(const server_tokens & tokens, const server_keepctx_entry * keep) {
    int n = 0;
    for (auto it = entries.begin(); it != entries.end();) {
        if (&*it != keep && it->prompt.tokens.size() <= tokens.size() &&
            it->prompt.tokens.get_common_prefix(tokens) == it->prompt.tokens.size()) {
            it = entries.erase(it);
            n++;
        } else {
            ++it;
        }
    }
    return n;
}

bool server_keepctx::save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft,
                          common_speculative * spec, llama_seq_id seq_id, const server_keepctx_entry * keep,
                          std::string & info, const server_keepctx_spill * spill, int64_t hits, int32_t sys_end) {
    const auto t0 = std::chrono::steady_clock::now();

    server_keepctx_entry e;
    e.size_tgt = llama_state_seq_get_size_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    e.size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    if (e.size_tgt == 0) {
        info = "the target state is empty";
        return false;
    }

    uint64_t need = e.size_tgt + e.size_dft;
    for (const auto & c : prompt.checkpoints) {
        need += c.data_tgt.size() + c.data_dft.size() + c.data_spec.size();
    }
    // [TAG_FN_L19_CONVSTORE_SPILL] the conversation's checkpoints on disk go with it (the same file, nothing written),
    // except the ones the ring holds as well
    uint64_t need_spilled = 0;
    if (spill) {
        for (const auto & sp : spill->list) {
            if (sp.n_tokens > prompt.n_tokens()) {
                continue;
            }
            bool in_ring = false;
            for (const auto & c : prompt.checkpoints) {
                in_ring = in_ring || c.n_tokens == sp.n_tokens;
            }
            if (!in_ring) {
                e.spilled.push_back(sp);
            }
        }
        keepctx_spill_validate(prompt.tokens, e.spilled);
        for (const auto & sp : e.spilled) {
            need_spilled += sp.bytes();
        }
    }
    e.hits    = hits;
    e.sys_end = sys_end;
    if (need + need_spilled > params_.budget) {
        e.spilled.clear(); // the state and the ring's checkpoints first
        need_spilled = 0;
    }
    if (need > params_.budget) {
        info = string_format("%.1f MiB is more than the store budget of %.1f MiB", need/1048576.0, params_.budget/1048576.0);
        return false;
    }

    // room: older states of this conversation, then the oldest entries (or [TAG_FN_L19_CONVSTORE_EVICT] the lowest
    // value per byte), never `keep`
    int n_evicted = drop_prefixes_of(prompt.tokens, keep);
    while ((n_bytes() + need + need_spilled > params_.budget || (int32_t) entries.size() >= params_.n_max) && evict_one(keep)) {
        n_evicted++;
    }
    if (n_bytes() + need + need_spilled > params_.budget || (int32_t) entries.size() >= params_.n_max) {
        info = string_format("no room beside the conversation the next prompt restores (%.1f MiB needed)", need/1048576.0);
        return false;
    }

    e.id = ++n_next_;
    const auto path = std::filesystem::path(params_.dir) /
            string_format("keepctx-%lu-%llu.kc", kc_pid(), (unsigned long long) e.id);
    std::string err;
    e.file = server_keepctx_file::create(path, err);
    if (!e.file) {
        info = "cannot create '" + path.string() + "': " + err;
        return false;
    }

    bool ok = true;
    {
        kc_pipe pipe(*e.file, true, 0);
        pipe.hashing = params_.verify;
        ok = ok && llama_state_seq_write_stream_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_write_cb, &pipe) == e.size_tgt;
        e.hash_tgt = pipe.hash;
        pipe.hash  = KC_HASH_SEED;
        if (ok && ctx_dft) {
            ok = llama_state_seq_write_stream_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_write_cb, &pipe) == e.size_dft;
        }
        e.hash_dft   = pipe.hash;
        pipe.hashing = false;
        for (const auto & c : prompt.checkpoints) {
            if (!ok) {
                break;
            }
            ok = pipe.put(c.data_tgt.data(), c.data_tgt.size()) && pipe.put(c.data_dft.data(), c.data_dft.size()) &&
                 pipe.put(c.data_spec.data(), c.data_spec.size());
            e.ckpt_sizes.push_back(c.data_tgt.size());
            e.ckpt_sizes.push_back(c.data_dft.size());
            e.ckpt_sizes.push_back(c.data_spec.size());
        }
        uint64_t n_written = 0;
        ok = pipe.finish(n_written) && ok;
        e.n_bytes = n_written;
    }
    if (!ok || e.n_bytes != need) {
        info = string_format("writing '%s' failed (%.1f of %.1f MiB)", path.string().c_str(), e.n_bytes/1048576.0, need/1048576.0);
        return false;
    }

    common_speculative_get_carry(spec, seq_id, e.carry);

    // [TAG_FN_L18_KEEPCTX_HOT] the counts before the new prompt fades them
    if (params_.hot) {
        llama_moe_hot_counts_get_ext(ctx_tgt, e.hot);
    }

    e.prompt.tokens = prompt.tokens.clone();
    for (const auto & c : prompt.checkpoints) {
        common_prompt_checkpoint m;
        m.n_tokens = c.n_tokens;
        m.id_task  = c.id_task;
        m.pos_min  = c.pos_min;
        m.pos_max  = c.pos_max;
        e.prompt.checkpoints.push_back(std::move(m));
    }

    // [TAG_FN_L19_CONVSTORE_EVICT]
    e.value = convstore_gdsf(gdsf_L_, e.hits, prompt.n_tokens(), e.n_bytes_all(), params_.fixed_tokens,
                             params_.budget / (uint64_t) std::max(1, params_.n_max));

    const double ms = kc_ms_since(t0);
    info = string_format("%d tokens, %u checkpoints, %.1f MiB (target %.1f, draft %.1f) in %.0f ms = %.2f GB/s, %d older evicted, "
                         "store %zu conversations / %.1f MiB", prompt.n_tokens(), (unsigned) prompt.checkpoints.size(),
                         need/1048576.0, e.size_tgt/1048576.0, e.size_dft/1048576.0, ms, need/(ms*1e6), n_evicted,
                         entries.size() + 1, (n_bytes() + e.n_bytes_all())/1048576.0);
    if (!e.spilled.empty() || params_.gdsf) {
        info += string_format(" [TAG_FN_L19_CONVSTORE] %zu checkpoints on disk (%.1f MiB), restores %lld, value %.0f, system block %d",
                              e.spilled.size(), need_spilled/1048576.0, (long long) e.hits, e.value, (int) e.sys_end);
    }
    entries.push_back(std::move(e));
    return true;
}

bool server_keepctx::load(std::list<server_keepctx_entry>::iterator it, server_prompt & prompt, llama_context * ctx_tgt,
                          llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq_id, bool keep, std::string & info,
                          server_keepctx_spill * spill, int64_t * hits, int32_t * sys_end) {
    const auto t0 = std::chrono::steady_clock::now();

    server_keepctx_entry e_own;
    if (!keep) {
        e_own = std::move(*it);
        entries.erase(it);
    }
    server_keepctx_entry & e = keep ? *it : e_own;

    // a kept entry that failed to restore is not offered again
    auto fail = [&](const char * why) {
        info = why;
        if (keep) {
            entries.erase(it);
        }
        return false;
    };

    if ((e.size_dft > 0) != (ctx_dft != nullptr)) {
        return fail("the draft context does not match the stored state");
    }

    std::list<common_prompt_checkpoint> ckpts;
    bool ok = true;
    {
        kc_pipe pipe(*e.file, false, e.n_bytes);
        ok = ok && llama_state_seq_read_stream_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_read_cb, &pipe) == e.size_tgt;
        if (ok && ctx_dft) {
            ok = llama_state_seq_read_stream_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, kc_read_cb, &pipe) == e.size_dft;
        }
        size_t k = 0;
        for (const auto & m : e.prompt.checkpoints) {
            if (!ok) {
                break;
            }
            common_prompt_checkpoint c;
            c.n_tokens = m.n_tokens;
            c.id_task  = m.id_task;
            c.pos_min  = m.pos_min;
            c.pos_max  = m.pos_max;
            c.data_tgt .resize(e.ckpt_sizes[k + 0]);
            c.data_dft .resize(e.ckpt_sizes[k + 1]);
            c.data_spec.resize(e.ckpt_sizes[k + 2]);
            k += 3;
            ok = pipe.get(c.data_tgt.data(), c.data_tgt.size()) && pipe.get(c.data_dft.data(), c.data_dft.size()) &&
                 pipe.get(c.data_spec.data(), c.data_spec.size());
            ckpts.push_back(std::move(c));
        }
    }
    if (!ok) {
        return fail("reading the stored state failed");
    }
    if (!common_speculative_set_carry(spec, seq_id, e.carry)) {
        return fail("the drafter carry does not match this drafter");
    }

    prompt.tokens      = keep ? e.prompt.tokens.clone() : std::move(e.prompt.tokens);
    prompt.checkpoints = std::move(ckpts);

    // [TAG_FN_L19_CONVSTORE] the checkpoints on disk go to the slot (the files are shared), the restore counts
    if (spill) {
        spill->list = e.spilled;
    }
    if (hits) {
        *hits = e.hits + 1;
    }
    if (sys_end) {
        *sys_end = e.sys_end;
    }
    if (keep) {
        e.hits++;
        e.value = convstore_gdsf(gdsf_L_, e.hits, e.prompt.n_tokens(), e.n_bytes_all(), params_.fixed_tokens,
                                 params_.budget / (uint64_t) std::max(1, params_.n_max));
    }

    const double ms = kc_ms_since(t0);

    // [TAG_FN_L18_KEEPCTX_HOT] the next pass moves the set back toward this conversation's experts
    std::string hot;
    if (!e.hot.empty()) {
        hot = llama_moe_hot_counts_set_ext(ctx_tgt, e.hot) ? ", hot-set counts back" : ", hot-set counts NOT set";
    }

    // the states as the contexts hold them now must be the bytes that were stored
    std::string ver;
    if (params_.verify) {
        auto state_hash = [&](llama_context * ctx, uint64_t & h) {
            std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
            const bool got = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == buf.size();
            h = kc_hash(KC_HASH_SEED, buf.data(), buf.size());
            return got;
        };
        uint64_t ht = 0;
        uint64_t hd = 0;
        const bool vt = state_hash(ctx_tgt, ht) && ht == e.hash_tgt;
        const bool vd = !ctx_dft || (state_hash(ctx_dft, hd) && hd == e.hash_dft);
        ver = string_format(", verify: target %s, draft %s", vt ? "identical" : "DIFFERS", ctx_dft ? (vd ? "identical" : "DIFFERS") : "none");
    }
    info = string_format("%d tokens, %zu checkpoints, %.1f MiB in %.0f ms = %.2f GB/s, %s, store %zu conversations / %.1f MiB%s%s",
                         prompt.n_tokens(), prompt.checkpoints.size(), e.n_bytes/1048576.0, ms, e.n_bytes/(ms*1e6),
                         keep ? "kept in the store (the prompt continues a part of it)" : "taken from the store",
                         entries.size(), n_bytes()/1048576.0, hot.c_str(), ver.c_str());
    if (!e.spilled.empty()) {
        info += string_format(" [TAG_FN_L19_CONVSTORE] %zu checkpoints on disk", e.spilled.size());
    }
    if (keep) {
        entries.splice(entries.end(), entries, it); // the most recently used
    }
    return true;
}
