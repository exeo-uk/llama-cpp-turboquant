// [TAG_FN_L22_LOGITDUMP] round 22 P15, see server-fn-l22-dump.h; the file format is in server-fn-l22-dump-io.h.
//
// Request: "l22": {"dump": NAME, "forced": [ids] | "forced_text": "...", "window": 3, "topk": 64, "ref": BASE_NAME}
// with env LLAMA_FN_L22_LOGITDUMP=<dir>. After the prompt: record 0 (the prompt's last row), then the forced tokens in
// windows of `window` rows (every row an output, decode phase GEN, no sampling, no drafting), one record per row.
// <dir>/NAME.l22d (written as .part, renamed at the end) and <dir>/NAME.json (sidecar). With "ref" the records also
// carry this run's log-probs at the ids of <dir>/BASE_NAME.l22d, so r22/test/kld_dump.py computes the exact sparse KLD.
// Runs on the server's main loop (a test tool: a 4,096-position dump blocks the loop for some seconds).

#include "server-fn-l22-dump.h"
#include "server-fn-l22-dump-io.h"

#include "../../src/llama-ext.h"

#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#ifndef _WIN32
extern char ** environ;
#endif

namespace fs = std::filesystem;

// every env key of the round's switches, for the sidecar
static common_json l22_dump_env() {
    static const char * prefixes[] = { "LLAMA_FN_", "GGML_CUDA_FN_", "GGML_SCHED_FN_", "LLAMA_MOE_", "LLAMA_KV_ROWS",
                                       "LLAMA_FLASHNEXT", "GGML_CUDA_MMQ_GLU_FUSION", "GGML_CUDA_GDN_" };
    common_json out = common_json::object();
#ifdef _WIN32
    char ** envp = _environ;
#else
    char ** envp = environ;
#endif
    for (char ** e = envp; e != nullptr && *e != nullptr; ++e) {
        const std::string kv = *e;
        const size_t eq = kv.find('=');
        if (eq == std::string::npos || eq == 0) {
            continue;
        }
        const std::string key = kv.substr(0, eq);
        for (const char * p : prefixes) {
            if (key.compare(0, strlen(p), p) == 0) {
                out[key] = kv.substr(eq + 1);
                break;
            }
        }
    }
    return out;
}

static std::string l22_hex(uint64_t v) {
    char buf[24];
    snprintf(buf, sizeof(buf), "%016" PRIx64, v);
    return buf;
}

static bool l22_write_text(const fs::path & p, const std::string & text) {
    FILE * f = fopen(p.string().c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    return fclose(f) == 0 && ok;
}

// rename over an existing file
static bool l22_replace(const fs::path & from, const fs::path & to) {
    std::error_code ec;
    fs::rename(from, to, ec);
    if (ec) {
        fs::remove(to, ec);
        ec.clear();
        fs::rename(from, to, ec);
    }
    return !ec;
}

bool server_fn_l22_dump_after_prompt(const server_fn_l22_dump_args & a) {
    using namespace l22dump;

    if (a.l22 == nullptr || !a.l22->is_object() || !a.l22->contains("dump")) {
        return false; // an "l22" object for something else
    }
    const llama_model * model = llama_get_model(a.ctx);
    const char * dir_env = llama_model_fn_env(model, "LLAMA_FN_L22_LOGITDUMP");
    if (dir_env == nullptr || dir_env[0] == '\0') {
        LOG_WRN("[TAG_FN_L22_LOGITDUMP] task %d: the request asks for a dump but LLAMA_FN_L22_LOGITDUMP is not set: served as usual\n", a.id_task);
        return false;
    }
    auto refuse = [&](const std::string & why) {
        LOG_WRN("[TAG_FN_L22_LOGITDUMP] task %d: no dump (%s): served as usual\n", a.id_task, why.c_str());
        return false;
    };

    // the request
    std::string name;
    std::string ref_name;
    std::vector<llama_token> forced;
    int64_t window_req = 3;
    int64_t topk_req   = 64;
    bool    from_text  = false;
    try {
        const common_json & o = *a.l22;
        if (!o.at("dump").is_string()) {
            return refuse("\"dump\" must be a string");
        }
        name = o.at("dump").get<std::string>();
        if (o.contains("ref") && !o.at("ref").is_null()) {
            ref_name = o.at("ref").get<std::string>();
        }
        if (o.contains("window")) {
            window_req = o.at("window").get<long long>();
        }
        if (o.contains("topk")) {
            topk_req = o.at("topk").get<long long>();
        }
        if (o.contains("forced") && !o.at("forced").is_null()) {
            const common_json & fj = o.at("forced");
            if (!fj.is_array()) {
                return refuse("\"forced\" must be an array of token ids");
            }
            for (size_t i = 0; i < fj.size(); ++i) {
                forced.push_back((llama_token) fj.at(i).get<long long>());
            }
            if (o.contains("forced_text") && !o.at("forced_text").is_null()) {
                LOG_WRN("[TAG_FN_L22_LOGITDUMP] task %d: both \"forced\" and \"forced_text\": \"forced\" is used\n", a.id_task);
            }
        } else if (o.contains("forced_text") && !o.at("forced_text").is_null()) {
            forced    = common_tokenize(a.vocab, o.at("forced_text").get<std::string>(), false, true);
            from_text = true;
        }
    } catch (const std::exception & e) {
        return refuse(std::string("bad \"l22\" object: ") + e.what());
    }
    if (!name_ok(name)) {
        return refuse("bad dump name \"" + name + "\" (1-96 of [A-Za-z0-9._-], no path)");
    }
    if (!ref_name.empty() && !name_ok(ref_name)) {
        return refuse("bad ref name \"" + ref_name + "\"");
    }
    if (a.bs_attached) {
        return refuse("the slot samples on the backend, so the row has no raw logits; send \"backend_sampling\": false");
    }
    if (a.multi) {
        return refuse("the task is part of an n_cmpl fan-out");
    }
    if (a.n_other_busy > 0) {
        return refuse(std::to_string(a.n_other_busy) + " other slot(s) are processing (they share the logits buffer)");
    }
    if (a.i_batch < 0) {
        return refuse("no output row");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(a.vocab);
    for (size_t i = 0; i < forced.size(); ++i) {
        if (forced[i] < 0 || forced[i] >= n_vocab) {
            return refuse("forced id " + std::to_string(forced[i]) + " at " + std::to_string(i) + " is outside the vocabulary");
        }
    }
    const int32_t n_forced_req = (int32_t) forced.size();
    const int32_t n_forced     = plan_forced(n_forced_req, (int64_t) a.n_ctx_slot - a.n_prompt - 1);
    forced.resize(n_forced);
    const int32_t win_max = std::max<int32_t>(1, std::min<int32_t>(WINDOW_MAX,
            (int32_t) std::min(llama_n_batch(a.ctx), llama_n_ubatch(a.ctx))));
    const int32_t window  = clamp_i(window_req, 1, win_max);
    const int32_t k       = clamp_i(topk_req, 1, std::min<int32_t>(K_MAX, n_vocab));

    header h;
    h.n_vocab  = n_vocab;
    h.k        = k;
    h.window   = window;
    h.n_forced = n_forced;
    h.n_prompt = a.n_prompt;
    h.flags    = from_text ? FLAG_FORCED_TEXT : 0;
    {
        const llama_tokens text = a.prompt != nullptr ? a.prompt->get_text_tokens() : llama_tokens();
        h.prompt_hash = hash_ids(text.data(), text.size(), a.n_prompt);
    }
    h.forced_hash = hash_ids(forced.data(), forced.size(), n_forced);

    const fs::path dir(dir_env);
    {
        std::error_code ec;
        fs::create_directories(dir, ec);
    }

    // the ref (exact KLD later): same vocabulary, k and forced ids
    header              rh;
    std::vector<record> rrecs;
    if (!ref_name.empty()) {
        std::vector<int32_t> rforced;
        std::string err;
        if (!read_dump((dir / (ref_name + ".l22d")).string(), rh, rforced, rrecs, err)) {
            return refuse("ref: " + err);
        }
        if (rh.n_vocab != n_vocab || rh.k != k || rh.forced_hash != h.forced_hash) {
            return refuse("ref " + ref_name + " differs (n_vocab " + std::to_string(rh.n_vocab) + ", k " + std::to_string(rh.k) +
                          ", forced ids " + (rh.forced_hash == h.forced_hash ? "same" : "different") + ")");
        }
        h.flags   |= FLAG_REF;
        h.ref_hash = ref_hash_of(rh, rrecs);
        if (rh.n_rec < n_forced + 1) {
            h.flags |= FLAG_REF_PARTIAL;
        }
    }
    const bool use_ref = (h.flags & FLAG_REF) != 0;
    auto ref_ids_of = [&](int32_t n) -> const int32_t * {
        return use_ref && n < (int32_t) rrecs.size() ? rrecs[n].ids.data() : nullptr;
    };

    // record 0: the prompt's last row (decoded in the PROMPT phase)
    const auto t0 = std::chrono::steady_clock::now();
    scratch s;
    record  r;
    const float * row0 = llama_get_logits_ith(a.ctx, a.i_batch);
    if (row0 == nullptr) {
        return refuse("no logits for the prompt's last row");
    }
    if (!make_record(row0, n_vocab, k, 0, n_forced > 0 ? forced[0] : -1, ref_ids_of(0), s, r)) {
        return refuse("the prompt's last row has non-finite logits");
    }
    if (use_ref && ref_ids_of(0) == nullptr) {
        set_ref_missing(r, k);
    }

    const fs::path path_part = dir / (name + ".l22d.part");
    const fs::path path_dump = dir / (name + ".l22d");
    const fs::path path_json = dir / (name + ".json");
    FILE * f = fopen(path_part.string().c_str(), "wb");
    if (f == nullptr) {
        return refuse("cannot write " + path_part.string());
    }
    if (!write_header(f, h, forced.data()) || !write_record(f, r, k, use_ref)) {
        fclose(f);
        std::error_code ec;
        fs::remove(path_part, ec);
        return refuse("write error on " + path_part.string());
    }

    // the forced rows: from here on the slot's memory holds rows it does not know, so the request is always taken
    int32_t     n_rec  = 1;
    std::string status = "ok";
    double      ms_dec = 0.0;
    int32_t     n_win  = 0;
    {
        common_batch b(a.ctx);
        for (int32_t j = 0; j < n_forced; j += window) {
            const int32_t m = std::min(window, n_forced - j);
            b.clear();
            for (int32_t i = 0; i < m; ++i) {
                b.add(forced[j + i], a.pos_next + j + i, a.seq_id, true);
            }
            llama_set_decode_phase_ext(a.ctx, LLAMA_FN_PHASE_GEN);
            const auto td = std::chrono::steady_clock::now();
            const int32_t ret = llama_process(a.ctx, LLAMA_PROCESS_TYPE_DECODE, b.get());
            ms_dec += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
            n_win++;
            if (ret != 0) {
                status = "aborted: llama_process returned " + std::to_string(ret) + " at forced row " + std::to_string(j);
                break;
            }
            bool ok = true;
            for (int32_t i = 0; i < m && ok; ++i) {
                const int32_t n    = j + i + 1;
                const int32_t next = n < n_forced ? forced[n] : -1;
                const float * row  = llama_get_logits_ith(a.ctx, i);
                if (row == nullptr || !make_record(row, n_vocab, k, n, next, ref_ids_of(n), s, r)) {
                    status = "aborted: no or non-finite logits at record " + std::to_string(n);
                    ok = false;
                    break;
                }
                if (use_ref && ref_ids_of(n) == nullptr) {
                    set_ref_missing(r, k);
                }
                if (!write_record(f, r, k, use_ref)) {
                    status = "aborted: write error at record " + std::to_string(n);
                    ok = false;
                    break;
                }
                n_rec++;
            }
            if (!ok) {
                break;
            }
        }
    }
    if (status != "ok") {
        h.flags |= FLAG_ABORTED;
    }
    h.n_rec = n_rec;
    bool file_ok = patch_header(f, n_rec, h.flags);
    file_ok = (fclose(f) == 0) && file_ok;
    file_ok = file_ok && l22_replace(path_part, path_dump);
    if (!file_ok && status == "ok") {
        status = "aborted: cannot finish " + path_dump.string();
    }
    const double ms_total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // the sidecar
    {
        char desc[256] = {};
        llama_model_desc(model, desc, sizeof(desc));
        const char * prof = llama_model_fn_profile(model);
        common_json sj = common_json::object();
        sj["format"]        = "L22DUMP1";
        sj["name"]          = name;
        sj["file"]          = path_dump.string();
        sj["status"]        = status;
        sj["model"]         = std::string(desc);
        sj["fn_profile"]    = std::string(prof ? prof : "");
        sj["n_vocab"]       = n_vocab;
        sj["k"]             = k;
        sj["window"]        = window;
        sj["n_rec"]         = n_rec;
        sj["n_forced"]      = n_forced;
        sj["n_forced_req"]  = n_forced_req;
        sj["forced_text"]   = from_text;
        sj["n_prompt"]      = a.n_prompt;
        sj["pos_next"]      = (int64_t) a.pos_next;
        sj["flags"]         = h.flags;
        sj["prompt_hash"]   = l22_hex(h.prompt_hash);
        sj["forced_hash"]   = l22_hex(h.forced_hash);
        sj["ref"]           = ref_name;
        sj["ref_hash"]      = l22_hex(h.ref_hash);
        sj["ref_partial"]   = (h.flags & FLAG_REF_PARTIAL) != 0;
        sj["ms_total"]      = ms_total;
        sj["ms_decode"]     = ms_dec;
        sj["n_windows"]     = n_win;
        sj["ms_per_window"] = n_win > 0 ? ms_dec/n_win : 0.0;
        sj["id_task"]       = a.id_task;
        sj["id_slot"]       = a.id_slot;
        sj["time"]          = (int64_t) time(nullptr);
        sj["env"]           = l22_dump_env();
        if (!l22_write_text(path_json, sj.dump(1))) {
            LOG_WRN("[TAG_FN_L22_LOGITDUMP] task %d: cannot write %s\n", a.id_task, path_json.string().c_str());
        }
    }

    LOG_WRN("[TAG_FN_L22_LOGITDUMP] task %d slot %d: %s: %d records (k %d, window %d, %d of %d forced tokens%s%s) in %.0f ms "
            "(%d windows, %.2f ms each): %s; the slot's cached prompt is dropped\n", a.id_task, a.id_slot,
            path_dump.string().c_str(), n_rec, k, window, n_forced, n_forced_req, use_ref ? ", ref " : "",
            use_ref ? ref_name.c_str() : "", ms_total, n_win, n_win > 0 ? ms_dec/n_win : 0.0, status.c_str());
    return true;
}
