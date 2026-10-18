// [TAG_FN_L22_SSDTIER] round 22, P5: direct I/O of mapped routed-expert ranges for the prompt stream (SSD-class experts,
// read into the caller's ring slots) and the pin fallback's arena fill. The engine is fn_res::dio_engine
// (llama-fn-res-core.h): unbuffered overlapped reads, LLAMA_FN_L22_SSDTIER_IO threads (default 8). Same bytes as the
// mapping: exact.

#include "llama-fn-l22.h"
#include "llama-fn-l22-p5.h"
#include "llama-fn-res-core.h"

#include "llama-fn-auto.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct dio_model {
    const llama_model *  model = nullptr;
    fn_res::dio_engine   eng;
};

std::mutex               g_dio_mtx;
std::atomic<dio_model *> g_dio{nullptr};
std::atomic<int>         g_dio_users{0}; // calls inside the engine now: the free waits for 0

struct dio_ref {
    dio_model * d       = nullptr;
    bool        counted = false;
    explicit dio_ref(const llama_model * m) {
        if (g_dio.load(std::memory_order_relaxed) == nullptr) {
            return; // off: no shared counter traffic
        }
        g_dio_users.fetch_add(1);
        counted = true;
        d = g_dio.load();
        if (d != nullptr && m != nullptr && d->model != m) {
            d = nullptr;
        }
    }
    ~dio_ref() {
        if (counted) {
            g_dio_users.fetch_sub(1);
        }
    }
    dio_ref(const dio_ref &) = delete;
    dio_ref & operator=(const dio_ref &) = delete;
};

// a caller waits at most this long for one ticket; then the rest is cancelled and it reads the mapping
constexpr uint32_t DIO_WAIT_MS = 20000;

} // namespace

void llama_fn_p5_on_build(const llama_model & model, llama_model_loader & ml) {
#ifdef _WIN32
    if (model.arch != LLM_ARCH_QWEN4EXP || !ml.use_mmap || llama_fn_l3_int(model, "LLAMA_FN_L22_SSDTIER", 0) < 1) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_dio_mtx);
    if (g_dio.load() != nullptr) {
        return; // one model per process uses it
    }
    auto d = std::make_unique<dio_model>();
    d->model = &model;
    size_t n_reg = 0;
    size_t bytes = 0;
    std::vector<const ggml_tensor *> regs;
    for (const auto & L : model.layers) {
        for (const ggml_tensor * t : { L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps }) {
            // a repacked tensor (extra set) holds other bytes than its file range
            if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer) || t->data == nullptr ||
                t->extra != nullptr) {
                continue;
            }
            const auto it = ml.weights_map.find(ggml_get_name(t));
            if (it == ml.weights_map.end() || it->second.idx >= ml.files_paths.size()) {
                continue;
            }
            d->eng.add_region(t->data, ggml_nbytes(t), (int) it->second.idx, (uint64_t) it->second.offs);
            regs.push_back(t);
            n_reg++;
            bytes += ggml_nbytes(t);
        }
    }
    if (n_reg == 0) {
        return;
    }
    const int n_thr = std::max(1, std::min(32, llama_fn_l3_int(model, "LLAMA_FN_L22_SSDTIER_IO", 8)));
    std::string err;
    if (!d->eng.open(ml.files_paths, n_thr, err)) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_SSDTIER] direct I/O is off: %s (SSD-class experts are read through the mapping)\n",
                __func__, err.c_str());
        return;
    }
    // the readers must give the mapped bytes: the first and last 4 KiB of every tensor are read and compared (a few
    // pages each; the load does not prefetch the routed experts)
    {
        std::vector<uint8_t> buf((size_t) 8192*regs.size());
        std::vector<fn_res::dio_req> rq;
        for (size_t i = 0; i < regs.size(); ++i) {
            const size_t n = ggml_nbytes(regs[i]);
            const size_t k = std::min<size_t>(n, 4096);
            rq.push_back({ regs[i]->data, k, buf.data() + 8192*i });
            rq.push_back({ (const char *) regs[i]->data + n - k, k, buf.data() + 8192*i + 4096 });
        }
        const int tk = d->eng.submit(rq.data(), (int) rq.size());
        bool same = tk >= 0 && d->eng.wait(tk, 60000);
        for (size_t i = 0; same && i < regs.size(); ++i) {
            const size_t n = ggml_nbytes(regs[i]);
            const size_t k = std::min<size_t>(n, 4096);
            same = memcmp(buf.data() + 8192*i, regs[i]->data, k) == 0 &&
                   memcmp(buf.data() + 8192*i + 4096, (const char *) regs[i]->data + n - k, k) == 0;
            if (!same) {
                LLAMA_LOG_WARN("%s: [TAG_FN_L22_SSDTIER] %s differs from its file range: direct I/O is off\n", __func__,
                        ggml_get_name(regs[i]));
            }
        }
        if (!same) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L22_SSDTIER] the readers' check failed: SSD-class experts are read through the mapping\n",
                    __func__);
            d->eng.close();
            return;
        }
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_SSDTIER] direct I/O ready: %zu expert tensors (%.2f GiB) in %zu files, %d reader threads\n",
            __func__, n_reg, bytes/1073741824.0, ml.files_paths.size(), n_thr);
    g_dio.store(d.release());
#else
    GGML_UNUSED(model);
    GGML_UNUSED(ml);
#endif
}

void llama_fn_p5_dio_free(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_dio_mtx);
    dio_model * d = g_dio.load();
    if (d == nullptr || d->model != &model) {
        return;
    }
    g_dio.store(nullptr);
    while (g_dio_users.load() != 0) {
        std::this_thread::yield();
    }
    d->eng.close(); // queued pieces fail, the readers stop, the files close
    delete d;
}

void llama_fn_p5_dio_stats(uint64_t * bytes, uint64_t * reads, uint64_t * fails) {
    dio_ref r(nullptr);
    if (r.d == nullptr) {
        *bytes = *reads = *fails = 0;
        return;
    }
    const auto s = r.d->eng.stats();
    *bytes = s.bytes;
    *reads = s.reads;
    *fails = s.fails;
}

int llama_fn_dio_submit(const llama_model * model, const llama_fn_dio_req * req, int n) {
    dio_ref r(model);
    if (r.d == nullptr || req == nullptr || n <= 0) {
        return -1;
    }
    std::vector<fn_res::dio_req> v((size_t) n);
    for (int i = 0; i < n; ++i) {
        v[i] = { req[i].src, req[i].len, req[i].dst };
    }
    return r.d->eng.submit(v.data(), n);
}

bool llama_fn_dio_wait(int ticket) {
    if (ticket < 0) {
        return false;
    }
    dio_ref r(nullptr);
    if (r.d == nullptr) {
        return false;
    }
    return r.d->eng.wait(ticket, DIO_WAIT_MS);
}
