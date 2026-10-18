// [TAG_FN_L22_DMASHARE] round 22 P14: the decode DMA share from pinned pages (D11), src/llama-fn-l22.h.
//
// LLAMA_FN_L22_DMASHARE=1 with LLAMA_FN_L22_PIN >= 1 (qwen4exp profile / environment; unset = nothing here runs: no bank,
// no fetch node, no flag): the bridge asks llama_fn_dmashare_wanted at its creation. Then
//   - banks: per group of bridged layers with the same expert types and shapes, one VRAM bank triple [ne0, ne1,
//     n_layers x S + 1] (S = LLAMA_FN_L22_DMASHARE_SLOTS, 1..8, default 4); layer k of the group owns slots
//     [k x S, k x S + S), the last slot stays zero. Made before the reserve and the hot set's fit (they see the VRAM).
//   - the device side (first llama_fn_dmashare_view, at the reserve): mapped side / done / edge areas per channel and the
//     device scratch, through the CUDA reg's procs (ggml/src/ggml-cuda/fn-l22-dmafetch.cu).
//   - the plan (bridge executor, under the bridge's graph_mtx, host memory only): bank hits, then up to k copies of cold
//     experts whose three slices are pinned (llama_fn_pin_query_dev: device addresses, tickets held), short unpinned
//     edges staged; the CPU pool skips the served experts. No copy while no pin is active (hits only).
//   - graph end (owner, the graph synced): the copies with a matching done record become bank hits, the rest are
//     emptied; every ticket goes back. PASSLOG kind=dma every 256 bridged graphs (32 at level 2).
// Placement class (GPU instead of CPU math for the served experts), KLD-gated.

#include "llama-fn-l22.h"
#include "llama-fn-dmashare-core.h"

#include "llama-fn-auto.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-bridge.h"
#include "llama-moe-gen5.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-fn-l22-dma.h"
#include "ggml-moe-bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint64_t DMA_MIB = 1ull << 20;

struct dma_group {
    ggml_tensor *      bank[3] = {}; // up, gate, down [ne0, ne1, n_layers x S + 1]
    llama_moe_dma_view view;
    int                n_layers   = 0;
    uint64_t           part_bytes[3] = {};
};

struct dma_layer {
    int                 il    = -1;
    int                 chan  = -1;
    int                 group = -1;
    const ggml_tensor * up    = nullptr; // the host expert tensors
    const ggml_tensor * gate  = nullptr;
    const ggml_tensor * down  = nullptr;
    fn_dma::layer_state slots;
    fn_dma::scratch     sc;
    fn_dma::pending_job pend;
};

struct dma_stats {
    uint64_t graphs = 0, jobs = 0, cpu_jobs = 0, hits = 0, copies = 0, cpu = 0, cold = 0, excl = 0, k = 0;
    uint64_t rej[fn_dma::REJ_N] = {};
    uint64_t nopin = 0, busy = 0, committed = 0, invalid = 0, errs = 0, bytes = 0;
    double   plan_ns = 0.0, fetch_ns = 0.0;
};

struct dma_state {
    const llama_model *      model = nullptr;
    const llama_moe_bridge * br    = nullptr;
    ggml_backend_dev_t       dev   = nullptr;
    int                      dev_index = -1;
    int                      S        = 4;
    int                      max_copy = 4;
    fn_dma::xrule            tail;
    fn_dma::xrule            defer;

    std::vector<dma_group>                       groups;
    std::vector<dma_layer>                       layers;
    std::vector<int>                             layer_of_chan; // chan -> layer, -1
    std::unordered_map<const ggml_tensor *, int> layer_of_up;
    ggml_context *          ctx = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    size_t                  bank_bytes = 0;

    // device side (owner thread, at the first view; read by the executor after the graph began)
    bool                           cuda_tried = false;
    bool                           cuda_ok    = false;
    int32_t                        bid        = -1;
    void *                         handle     = nullptr;
    ggml_backend_fn_l22_dma_free_t fn_free    = nullptr;
    ggml_fn_l22_dma_host           host       = {};
    uint64_t                       edge_bytes = 0;

    // plan / job / graph state: the executor (plan, job_done) and the owner (graph_end) under mtx
    std::mutex       mtx;
    fn_dma::rates    rates;
    uint64_t         tick = 0;
    std::vector<int> pend_layers;
    dma_stats        st;
    int64_t          open_tickets = 0;
    uint64_t         graphs_total = 0;
};

std::mutex               g_dma_mtx; // creation and free
std::atomic<dma_state *> g_dma{nullptr};

dma_state * dma_get(const llama_moe_bridge * br) {
    dma_state * s = g_dma.load(std::memory_order_acquire);
    return s != nullptr && s->br == br ? s : nullptr;
}

// plan 3.6: a device allocation with pins active first checks the DXGI LOCAL headroom and asks for an unpin once
bool dma_room(size_t bytes, bool * asked) {
    if (!llama_fn_pin_active()) {
        return true;
    }
    uint64_t budget = 0;
    uint64_t usage  = 0;
    if (llama_fn_dxgi_query(0, 0, &budget, &usage) && budget > usage && budget - usage >= (uint64_t) bytes + 512*DMA_MIB) {
        return true;
    }
    *asked = true;
    return llama_fn_pin_on_oom(bytes);
}

bool host_contig(const ggml_tensor * t) {
    return t != nullptr && t->data != nullptr && t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer) &&
           t->ne[3] == 1 && t->nb[1] == ggml_row_size(t->type, t->ne[0]) && t->nb[2] == t->nb[1]*(size_t) t->ne[1];
}

void dma_destroy(dma_state * s) {
    if (s == nullptr) {
        return;
    }
    for (auto & L : s->layers) {
        for (int32_t t : L.pend.tickets) {
            llama_fn_pin_release(t);
        }
        L.pend.tickets.clear();
        L.pend.on = false;
    }
    if (s->handle != nullptr && s->fn_free != nullptr) {
        s->fn_free(s->handle);
    }
    s->handle = nullptr;
    if (s->buf != nullptr) {
        ggml_backend_buffer_free(s->buf);
    }
    if (s->ctx != nullptr) {
        ggml_free(s->ctx);
    }
    delete s;
}

// the device side, at the first view (owner thread, no graph of this context runs)
void dma_cuda_init(dma_state * s) {
    s->cuda_tried = true;
    int32_t id   = -1;
    int32_t chan = -1;
    if (s->layers.empty() || !llama_moe_bridge_layer(s->br, s->layers[0].up, &id, &chan) || id < 0) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] the bridge has no id: no DMA share\n", __func__);
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(s->dev);
    auto fn_new = reg ? (ggml_backend_fn_l22_dma_new_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_l22_dma_new") : nullptr;
    s->fn_free  = reg ? (ggml_backend_fn_l22_dma_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_l22_dma_free") : nullptr;
    if (fn_new == nullptr || s->fn_free == nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] the device backend has no DMA share fetch: no DMA share\n", __func__);
        return;
    }
    ggml_fn_l22_dma_desc d;
    memset(&d, 0, sizeof(d));
    d.bridge_id = id;
    d.device    = s->dev_index;
    d.n_chan    = (int32_t) s->layer_of_chan.size();
    d.max_copy  = s->max_copy;
    d.edge_bytes = s->edge_bytes;
    for (const auto & L : s->layers) {
        d.slot_lo[L.chan] = L.slots.slot_lo;
        d.slot_n [L.chan] = L.slots.S;
    }
    const size_t mapped = (size_t) d.n_chan*(sizeof(ggml_fn_l22_dma_side) + sizeof(ggml_fn_l22_dma_done) + s->edge_bytes);
    bool asked = false;
    if (!dma_room(mapped, &asked)) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] no LOCAL headroom for %.1f MiB of mapped areas: no DMA share\n", __func__,
                mapped/(double) DMA_MIB);
        return;
    }
    ggml_fn_l22_dma_host h = {};
    s->handle = fn_new(&d, &h);
    if (s->handle == nullptr && llama_fn_pin_active() && !asked && llama_fn_pin_on_oom(mapped)) {
        s->handle = fn_new(&d, &h);
    }
    if (s->handle == nullptr || h.side == nullptr || h.done == nullptr || h.edge == nullptr || h.edge_dev == 0) {
        s->handle = nullptr;
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] the device side could not be made: no DMA share\n", __func__);
        return;
    }
    s->host    = h;
    s->bid     = id;
    s->cuda_ok = true;
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_DMASHARE] MoE bridge %d: fetch side for %zu layers, %.1f MiB mapped (edges %.0f KiB per layer)\n",
            __func__, id, s->layers.size(), mapped/(double) DMA_MIB, s->edge_bytes/1024.0);
}

void dma_passlog(dma_state * s) {
    dma_stats & t = s->st;
    const double ls = t.jobs > 0 ? (double) t.jobs : 1.0;
    const double cj = t.cpu_jobs > 0 ? (double) t.cpu_jobs : 1.0;
    const double gbs = t.fetch_ns > 0.0 ? (double) t.bytes/t.fetch_ns : 0.0;
    llama_fn_l22_log("dma", "graphs=%" PRIu64 " layer_steps=%" PRIu64 " per_layer_step: hit=%.2f dma=%.2f cpu=%.2f cold=%.2f dropped=%.2f "
            "k=%.2f dma_gbs=%.2f cpu_us=%.1f dma_us=%.1f g=%.2f plan_us=%.2f rejects: unpinned=%" PRIu64 " edge=%" PRIu64 " segs=%" PRIu64
            " devaddr=%" PRIu64 " range=%" PRIu64 " slots=%" PRIu64 " nopin_jobs=%" PRIu64 " busy_jobs=%" PRIu64 " committed=%" PRIu64
            " invalid=%" PRIu64 " errs=%" PRIu64 " open_tickets=%" PRId64, t.graphs, t.jobs, t.hits/ls, t.copies/ls, t.cpu/cj,
            t.cold/ls, t.excl/ls, t.k/ls, gbs, s->rates.cpu_us, s->rates.dma_us, s->rates.g, t.plan_ns/1e3/ls,
            t.rej[fn_dma::REJ_UNPINNED], t.rej[fn_dma::REJ_EDGE], t.rej[fn_dma::REJ_SEGS], t.rej[fn_dma::REJ_DEVADDR],
            t.rej[fn_dma::REJ_RANGE], t.rej[fn_dma::REJ_SLOTS], t.nopin, t.busy, t.committed, t.invalid, t.errs, s->open_tickets);
    t = dma_stats();
}

} // namespace

bool llama_fn_dmashare_wanted(const llama_model * model, const llama_moe_bridge * br) {
    if (model == nullptr || br == nullptr || model->arch != LLM_ARCH_QWEN4EXP) {
        return false;
    }
    const bool on = llama_fn_l22_switch_on(llama_fn_env(*model, "LLAMA_FN_L22_DMASHARE"));
    if (!on) {
        return false;
    }
    const fn_dma::cfg cfg = fn_dma::make_cfg(on, llama_fn_l3_int(*model, "LLAMA_FN_L22_PIN", 0),
            llama_fn_l3_int(*model, "LLAMA_FN_L22_DMASHARE_SLOTS", 4));
    if (!cfg.on) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] LLAMA_FN_L22_DMASHARE needs LLAMA_FN_L22_PIN >= 1 (it copies from pinned pages): off\n", __func__);
        return false;
    }
    std::lock_guard<std::mutex> lk(g_dma_mtx);
    if (g_dma.load() != nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] another bridge has the DMA share: off\n", __func__);
        return false;
    }

    std::unique_ptr<dma_state, void (*)(dma_state *)> s(new dma_state(), dma_destroy);
    s->model    = model;
    s->br       = br;
    s->S        = cfg.slots;
    s->max_copy = cfg.slots;
    // the levers' drop rules, read as the bridge reads them (P8: tail; P11: light rows)
    {
        s->tail.r = std::min(100, std::max(0, (int) llama_fn_l22_x_read(model).tailskip));
        const char * tw = llama_fn_env(*model, "LLAMA_FN_L22_X_TAILSKIP_W");
        s->tail.w = s->tail.r > 0 && tw && tw[0] ? std::min(1.0f, std::max(0.0f, (float) atof(tw))) : 0.0f;
        const char * dr = llama_fn_env(*model, "LLAMA_FN_L22_X_DEFER_R");
        const char * dw = llama_fn_env(*model, "LLAMA_FN_L22_X_DEFER_W");
        s->defer.r = std::min(9, std::max(0, dr && dr[0] ? atoi(dr) : 0));
        s->defer.w = dw && dw[0] ? std::min(1.0f, std::max(0.0f, (float) atof(dw))) : (s->defer.r > 0 ? 0.0f : 0.1f);
    }

    // the bridged layers, grouped by expert types and shapes
    struct gkey {
        ggml_type t[3];
        int64_t   ne[3][2];
        int64_t   n_exp;
        bool operator<(const gkey & o) const {
            return std::tie(t[0], t[1], t[2], ne[0][0], ne[0][1], ne[1][0], ne[1][1], ne[2][0], ne[2][1], n_exp) <
                   std::tie(o.t[0], o.t[1], o.t[2], o.ne[0][0], o.ne[0][1], o.ne[1][0], o.ne[1][1], o.ne[2][0], o.ne[2][1], o.n_exp);
        }
    };
    std::map<gkey, int> group_of;
    int max_chan = -1;
    for (size_t il = 0; il < model->layers.size(); ++il) {
        const auto & l = model->layers[il];
        int32_t id   = -1;
        int32_t chan = -1;
        if (l.ffn_up_exps == nullptr || !llama_moe_bridge_layer(br, l.ffn_up_exps, &id, &chan)) {
            continue;
        }
        const ggml_tensor * ts[3] = { l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps };
        if (!host_contig(ts[0]) || !host_contig(ts[1]) || !host_contig(ts[2]) || ts[1]->ne[2] != ts[0]->ne[2] ||
            ts[2]->ne[2] != ts[0]->ne[2] || chan < 0 || chan >= GGML_FN_L22_DMA_MAX_CHAN || l.ffn_gate_inp == nullptr ||
            l.ffn_gate_inp->buffer == nullptr) {
            continue;
        }
        ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer));
        if (d == nullptr || (s->dev != nullptr && d != s->dev)) {
            continue;
        }
        s->dev = d;
        gkey k;
        memset(&k, 0, sizeof(k));
        for (int i = 0; i < 3; ++i) {
            k.t[i]     = ts[i]->type;
            k.ne[i][0] = ts[i]->ne[0];
            k.ne[i][1] = ts[i]->ne[1];
        }
        k.n_exp = ts[0]->ne[2];
        auto it = group_of.find(k);
        if (it == group_of.end()) {
            it = group_of.emplace(k, (int) s->groups.size()).first;
            s->groups.emplace_back();
            for (int i = 0; i < 3; ++i) {
                s->groups.back().part_bytes[i] = ts[i]->nb[2];
            }
        }
        dma_layer L;
        L.il    = (int) il;
        L.chan  = chan;
        L.group = it->second;
        L.up    = ts[0];
        L.gate  = ts[1];
        L.down  = ts[2];
        s->groups[L.group].n_layers++;
        s->layers.push_back(std::move(L));
        max_chan = std::max(max_chan, (int) chan);
    }
    if (s->layers.empty() || s->dev == nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] no bridged layer the DMA share can take: off\n", __func__);
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(s->dev);
    for (size_t i = 0; reg != nullptr && i < ggml_backend_reg_dev_count(reg); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == s->dev) {
            s->dev_index = (int) i;
        }
    }
    if (s->dev_index < 0 || ggml_backend_reg_get_proc_address(reg, "ggml_backend_fn_l22_dma_new") == nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] the router's device has no DMA share fetch: off\n", __func__);
        return false;
    }

    // banks
    {
        ggml_init_params ip = { ggml_tensor_overhead()*(s->groups.size()*3 + 4), nullptr, true };
        s->ctx = ggml_init(ip);
        if (s->ctx == nullptr) {
            return false;
        }
        std::vector<int> seen(s->groups.size(), 0);
        for (const auto & L : s->layers) {
            dma_group & G = s->groups[L.group];
            if (seen[L.group]++) {
                continue;
            }
            const ggml_tensor * ts[3] = { L.up, L.gate, L.down };
            static const char * nm[3] = { "l22_dma_up.%d", "l22_dma_gate.%d", "l22_dma_down.%d" };
            for (int i = 0; i < 3; ++i) {
                G.bank[i] = ggml_new_tensor_3d(s->ctx, ts[i]->type, ts[i]->ne[0], ts[i]->ne[1], (int64_t) G.n_layers*s->S + 1);
                ggml_format_name(G.bank[i], nm[i], L.group);
                s->bank_bytes += ggml_nbytes(G.bank[i]);
            }
        }
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(s->dev);
        bool asked = false;
        if (buft == nullptr || !dma_room(s->bank_bytes, &asked)) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] no LOCAL headroom for %.0f MiB of banks: off\n", __func__,
                    s->bank_bytes/(double) DMA_MIB);
            return false;
        }
        s->buf = ggml_backend_alloc_ctx_tensors_from_buft(s->ctx, buft);
        if (s->buf == nullptr && llama_fn_pin_active() && !asked && llama_fn_pin_on_oom(s->bank_bytes)) {
            s->buf = ggml_backend_alloc_ctx_tensors_from_buft(s->ctx, buft);
        }
        if (s->buf == nullptr) {
            LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] %.0f MiB of banks could not be allocated: off\n", __func__,
                    s->bank_bytes/(double) DMA_MIB);
            return false;
        }
        ggml_backend_buffer_set_usage(s->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(s->buf, 0); // the zero slots stay zero
    }
    for (auto & G : s->groups) {
        G.view.up        = G.bank[0];
        G.view.gate      = G.bank[1];
        G.view.down      = G.bank[2];
        G.view.dev_table = nullptr;
        G.view.handle    = &G;
    }
    std::vector<int> k_in_group(s->groups.size(), 0);
    s->layer_of_chan.assign((size_t) max_chan + 1, -1);
    for (size_t i = 0; i < s->layers.size(); ++i) {
        dma_layer & L = s->layers[i];
        const dma_group & G = s->groups[L.group];
        const int n_exp = (int) L.up->ne[2];
        L.slots.init(s->S, (int32_t) (k_in_group[L.group]++*s->S), (int32_t) (G.n_layers*s->S), n_exp);
        L.sc.init(n_exp);
        L.pend.tickets.reserve(3*GGML_FN_L22_DMA_MAX_SEG*GGML_FN_L22_DMA_MAX_COPY);
        s->layer_of_chan[L.chan] = (int) i;
        s->layer_of_up[L.up]     = (int) i;
    }
    s->edge_bytes = (uint64_t) s->max_copy*3*GGML_FN_L22_DMA_EDGE_MAX;
    s->pend_layers.reserve(s->layers.size());

    LLAMA_LOG_WARN("%s: [TAG_FN_L22_DMASHARE] decode DMA share from pinned pages: %zu layers in %zu bank group(s), %d slots per layer "
            "(copies per job <= %d), %.0f MiB VRAM; tail rule r %d w %.3f, light rule r %d w %.3f (on flagged jobs)\n", __func__,
            s->layers.size(), s->groups.size(), s->S, s->max_copy, s->bank_bytes/(double) DMA_MIB, s->tail.r, s->tail.w,
            s->defer.r, s->defer.w);
    llama_fn_l22_log("dma", "banks layers=%zu groups=%zu slots=%d vram_mib=%.0f", s->layers.size(), s->groups.size(), s->S,
            s->bank_bytes/(double) DMA_MIB);
    g_dma.store(s.release(), std::memory_order_release);
    return true;
}

const llama_moe_dma_view * llama_fn_dmashare_view(const llama_moe_bridge * br, const ggml_tensor * up_exps, int64_t n_tokens) {
    dma_state * s = dma_get(br);
    if (s == nullptr || n_tokens < 1 || n_tokens > fn_dma::MAX_T) {
        return nullptr;
    }
    const auto it = s->layer_of_up.find(up_exps);
    if (it == s->layer_of_up.end()) {
        return nullptr;
    }
    if (!s->cuda_tried) {
        dma_cuda_init(s);
    }
    if (!s->cuda_ok) {
        return nullptr;
    }
    return &s->groups[s->layers[it->second].group].view;
}

bool llama_fn_dmashare_plan(llama_fn_dmashare_job * jv, int32_t * skip_table) {
    if (jv == nullptr || skip_table == nullptr || jv->job == nullptr || jv->job->plan == nullptr) {
        return false;
    }
    dma_state * s = dma_get(jv->br);
    const ggml_moe_bridge_job * j = jv->job;
    const int li = s != nullptr && jv->chan >= 0 && jv->chan < (int) s->layer_of_chan.size() ? s->layer_of_chan[jv->chan] : -1;
    // a fetch node exists only for a layer whose view the graph got (device side made, T <= 8): anything else is a job
    // this share cannot plan, so it fails (the ubatch is computed again without the bridge)
    if (li < 0 || !s->cuda_ok || j->n_tokens < 1 || j->n_tokens > fn_dma::MAX_T || j->n_used < 1 ||
        j->n_used*j->n_tokens > GGML_FN_L22_DMA_MAX_IDS || jv->up == nullptr) {
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    dma_layer & L = s->layers[li];
    if (jv->il >= 0 && jv->il != L.il) {
        return false;
    }

    fn_dma::job_in in;
    in.ids      = j->ids;
    in.w        = j->w;
    in.n_used   = j->n_used;
    in.T        = j->n_tokens;
    in.n_exp    = (int) jv->up->ne[2];
    in.hot      = jv->hot_table;
    in.hot_miss = jv->hot_miss;
    if (j->flags & GGML_MOE_BRIDGE_JOB_TAIL) {
        in.tail = s->tail;
    }
    if (j->flags & GGML_MOE_BRIDGE_JOB_DEFER) {
        in.defer = s->defer;
    }
    if (in.n_exp != (int) L.slots.mirror.size()) {
        return false;
    }

    ggml_fn_l22_dma_side * side = s->host.side + L.chan;
    fn_dma::edge_area ea;
    ea.host = s->host.edge + (size_t) L.chan*s->edge_bytes;
    ea.dev  = s->host.edge_dev + (uint64_t) L.chan*s->edge_bytes;
    ea.cap  = (size_t) s->edge_bytes;
    ea.used = 0;

    const llama_model * model = s->model;
    const int il = L.il;
    const fn_dma::query_fn q = [](const void * p, size_t n, fn_dma::seg_q * out) {
        llama_fn_pin_seg sg;
        uint64_t dev = 0;
        if (!llama_fn_pin_query_dev(p, n, &sg, &dev)) {
            return false;
        }
        out->src    = sg.src;
        out->len    = sg.len;
        out->pinned = sg.pinned;
        out->ticket = sg.ticket;
        out->dev    = dev;
        return true;
    };
    const fn_dma::release_fn rel = [](int32_t t) { llama_fn_pin_release(t); };
    const fn_dma::resolve_fn resolve = [&](int32_t e, ggml_fn_l22_dma_part parts[3], std::vector<int32_t> & tickets) {
        const uint8_t * p[3] = {};
        size_t          n[3] = {};
        for (int k = 0; k < 3; ++k) {
            if (!llama_fn_expert_host_range(model, il, e, k, &p[k], &n[k])) {
                return (int) fn_dma::REJ_RANGE;
            }
        }
        return fn_dma::resolve_expert(p, n, q, rel, ea, parts, tickets);
    };

    const bool pins = llama_fn_pin_active();
    std::lock_guard<std::mutex> lk(s->mtx);
    // a copy of this channel still waits for its commit (two jobs of one layer in a graph): this job takes hits only
    const bool busy = L.pend.on;
    const fn_dma::rates r = s->rates;
    const uint64_t tick = ++s->tick;
    std::vector<int32_t> tk;
    std::vector<int32_t> & tickets = busy ? tk : L.pend.tickets;
    if (!busy) {
        tickets.clear();
    }
    const fn_dma::plan_res res = fn_dma::plan_job(L.slots, in, L.sc, r, std::min(s->max_copy, std::max(0, jv->max_fetch)),
            pins && !busy, tick, resolve, side, j->seq, j->plan->slot_ids, skip_table, tickets);
    ggml_moe_bridge_plan * pl = j->plan;
    pl->n_copy = res.n_copy;
    for (int i = 0; i < res.n_copy; ++i) {
        pl->off[i]  = 0;
        pl->slot[i] = side->slot[i];
    }
    if (res.n_copy > 0) {
        L.pend.on  = true;
        L.pend.seq = j->seq;
        L.pend.n   = res.n_copy;
        for (int i = 0; i < res.n_copy; ++i) {
            L.pend.slot[i] = res.copy_slot[i];
            L.pend.e[i]    = res.copy_e[i];
        }
        s->pend_layers.push_back(li);
        s->open_tickets += (int64_t) L.pend.tickets.size();
    } else if (!busy) {
        for (int32_t t : tickets) { // none expected: rejected experts gave theirs back
            llama_fn_pin_release(t);
        }
        tickets.clear();
    }

    dma_stats & t = s->st;
    t.jobs++;
    t.hits   += (uint64_t) res.n_hit;
    t.copies += (uint64_t) res.n_copy;
    t.cold   += (uint64_t) res.n_cold;
    t.excl   += (uint64_t) res.n_excl;
    t.k      += (uint64_t) res.k;
    for (int k = 0; k < fn_dma::REJ_N; ++k) {
        t.rej[k] += (uint64_t) res.rej[k];
    }
    t.nopin += pins ? 0 : 1;
    t.busy  += busy ? 1 : 0;
    t.plan_ns += (double) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void llama_fn_dmashare_job_done(const llama_moe_bridge * br, int32_t chan, uint32_t seq, uint64_t cpu_ns, uint32_t n_cpu) {
    (void) chan;
    (void) seq;
    dma_state * s = dma_get(br);
    if (s == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    if (n_cpu >= 1 && cpu_ns > 0) {
        s->rates.cpu_us = fn_dma::ewma(s->rates.cpu_us, (double) cpu_ns/1e3/n_cpu);
    }
    s->st.cpu += n_cpu;
    s->st.cpu_jobs++;
}

void llama_fn_dmashare_graph_end(const llama_moe_bridge * br, uint32_t err, double dev_wait_us_per_layer) {
    dma_state * s = dma_get(br);
    if (s == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    s->st.graphs++;
    s->graphs_total++;
    s->st.errs += err != GGML_MOE_BRIDGE_ERR_NONE ? 1 : 0;
    s->rates.g = fn_dma::guard_step(s->rates.g, dev_wait_us_per_layer);
    const fn_dma::release_fn rel = [](int32_t t) { llama_fn_pin_release(t); };
    for (int li : s->pend_layers) {
        dma_layer & L = s->layers[li];
        if (!L.pend.on) {
            continue;
        }
        // the device wrote the record before the graph's end; the context synchronized before this call
        const volatile ggml_fn_l22_dma_done * dv = s->host.done + L.chan;
        ggml_fn_l22_dma_done d;
        memset(&d, 0, sizeof(d));
        d.seq      = dv->seq;
        d.n        = dv->n;
        d.ok       = dv->ok;
        d.fetch_ns = dv->fetch_ns;
        std::atomic_thread_fence(std::memory_order_acquire);
        const int n = L.pend.n;
        s->open_tickets -= (int64_t) L.pend.tickets.size();
        const int rc = fn_dma::commit(L.slots, L.pend, d, rel);
        if (rc == fn_dma::CM_OK) {
            const dma_group & G = s->groups[L.group];
            s->st.committed += (uint64_t) n;
            s->st.bytes     += (uint64_t) n*(G.part_bytes[0] + G.part_bytes[1] + G.part_bytes[2]);
            s->st.fetch_ns  += (double) d.fetch_ns;
            if (d.fetch_ns > 0 && n > 0) {
                s->rates.dma_us = fn_dma::ewma(s->rates.dma_us, (double) d.fetch_ns/1e3/n);
            }
        } else if (rc == fn_dma::CM_BAD) {
            s->st.invalid += (uint64_t) n;
        }
    }
    s->pend_layers.clear();
    const int lvl = llama_fn_l22_passlog_level();
    const uint64_t every = lvl >= 2 ? 32 : 256;
    if (lvl >= 1 && s->st.graphs >= every) {
        dma_passlog(s);
    } else if (lvl < 1 && s->st.graphs >= every) {
        s->st = dma_stats();
    }
}

void llama_fn_dmashare_free(const llama_moe_bridge * br) {
    std::lock_guard<std::mutex> lk(g_dma_mtx);
    dma_state * s = dma_get(br);
    if (s == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk2(s->mtx);
        if (s->st.graphs > 0 && llama_fn_l22_passlog_level() >= 1) {
            dma_passlog(s);
        }
        LLAMA_LOG_INFO("%s: [TAG_FN_L22_DMASHARE] %" PRIu64 " bridged graphs; %" PRId64 " tickets were open at the end (released now)\n",
                __func__, s->graphs_total, s->open_tickets);
    }
    g_dma.store(nullptr, std::memory_order_release);
    dma_destroy(s);
}
