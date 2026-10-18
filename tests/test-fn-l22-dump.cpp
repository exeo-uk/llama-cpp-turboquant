// [TAG_FN_L22_LOGITDUMP] round 22 P15: CPU unit test of the logit dump's pure parts (tools/server/server-fn-l22-dump-io.h):
// top-k and log-softmax against a naive full softmax, the tie order, the tail mass and its zero sentinel, the REF part,
// non-finite rows, the file round trip (n_rec / flags patch, ABORTED, REF), bad files, names, pinned hashes, the record
// plan. With argv[1] = <dir> it also writes the fixture dumps that r22/test/kld_dump.py --selftest --fixture <dir>
// recomputes in Python. No GPU, no model. Exit code 0 iff every check passes.

#include "../tools/server/server-fn-l22-dump-io.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace l22dump;

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

static bool near(double a, double b, double tol) {
    return std::fabs(a - b) <= tol*(1.0 + std::fabs(b));
}

// naive reference: full log-softmax in double, order by (logit desc, id asc)
struct naive {
    std::vector<double>  lp;
    std::vector<int32_t> order;
    explicit naive(const std::vector<float> & l) {
        const int n = (int) l.size();
        double mx = l[0];
        for (float v : l) {
            mx = std::max(mx, (double) v);
        }
        double s = 0.0;
        for (float v : l) {
            s += exp((double) v - mx);
        }
        lp.resize(n);
        for (int i = 0; i < n; ++i) {
            lp[i] = (double) l[i] - mx - log(s);
        }
        order.resize(n);
        for (int i = 0; i < n; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return l[a] > l[b] || (l[a] == l[b] && a < b);
        });
    }
    double log_outside(const std::vector<int32_t> & ids) const {
        std::vector<uint8_t> in(lp.size(), 0);
        for (int32_t id : ids) {
            in[id] = 1;
        }
        double t = 0.0;
        for (size_t i = 0; i < lp.size(); ++i) {
            t += in[i] ? 0.0 : exp(lp[i]);
        }
        return t > 0.0 ? log(t) : (double) LOG_ZERO;
    }
};

static void test_topk_vs_naive(int n_vocab, int k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float> l(n_vocab);
    for (float & v : l) {
        v = nd(rng);
    }
    const naive ref(l);
    scratch s;
    record  r;
    const int32_t next = n_vocab/3;
    TCHECK(make_record(l.data(), n_vocab, k, 7, next, nullptr, s, r), "make_record V %d", n_vocab);
    TCHECK(r.n == 7 && r.next == next, "n/next");
    TCHECK((int) r.ids.size() == k && (int) r.lps.size() == k && r.ref_lps.empty(), "sizes");
    bool ids_ok = true;
    bool lps_ok = true;
    for (int i = 0; i < k; ++i) {
        ids_ok = ids_ok && r.ids[i] == ref.order[i];
        lps_ok = lps_ok && near(r.lps[i], ref.lp[r.ids[i]], 2e-5);
    }
    TCHECK(ids_ok, "top-k ids differ from the naive order (V %d)", n_vocab);
    TCHECK(lps_ok, "top-k log-probs differ from the naive log-softmax (V %d)", n_vocab);
    TCHECK(near(r.lp_next, ref.lp[next], 2e-5), "lp_next %g vs %g", r.lp_next, ref.lp[next]);
    const double lt = ref.log_outside(r.ids);
    TCHECK(near(r.log_tail, lt, 2e-5), "log_tail %g vs %g (V %d)", r.log_tail, lt, n_vocab);
    // log_tail against log(1 - sum p) where that is accurate (a spread distribution)
    double sp = 0.0;
    for (int i = 0; i < k; ++i) {
        sp += exp((double) r.lps[i]);
    }
    TCHECK(near(r.log_tail, log(1.0 - sp), 1e-3), "log_tail %g vs log(1 - sum) %g", r.log_tail, log(1.0 - sp));
}

static void test_ties() {
    std::vector<float> l = { 1.0f, 3.0f, 3.0f, 2.0f, 3.0f, 0.5f };
    scratch s;
    record  r;
    TCHECK(make_record(l.data(), 6, 4, 0, -1, nullptr, s, r), "make_record");
    const int32_t want[4] = { 1, 2, 4, 3 };
    for (int i = 0; i < 4; ++i) {
        TCHECK(r.ids[i] == want[i], "tie order at %d: %d, want %d", i, r.ids[i], want[i]);
    }
    TCHECK(r.lp_next == 0.0f && r.next == -1, "no next");
    TCHECK(r.lps[0] == r.lps[1] && r.lps[1] == r.lps[2], "equal logits, equal log-probs");
}

static void test_tail_zero() {
    // k == V: nothing outside
    std::vector<float> l = { 0.1f, -0.4f, 2.0f, 1.0f };
    scratch s;
    record  r;
    TCHECK(make_record(l.data(), 4, 4, 0, 2, nullptr, s, r), "make_record");
    TCHECK(r.log_tail == LOG_ZERO, "k == V: log_tail %g", r.log_tail);
    double sp = 0.0;
    for (float lp : r.lps) {
        sp += exp((double) lp);
    }
    TCHECK(near(sp, 1.0, 1e-6), "sum p %g", sp);
    // the rest underflows in float exp: the sentinel, not log(0)
    std::vector<float> l2(100, -200.0f);
    l2[3] = 0.0f;
    l2[50] = 0.5f;
    TCHECK(make_record(l2.data(), 100, 2, 0, 3, nullptr, s, r), "make_record 2");
    TCHECK(r.ids[0] == 50 && r.ids[1] == 3, "ids");
    TCHECK(r.log_tail == LOG_ZERO && std::isfinite(r.log_tail), "underflow: log_tail %g", r.log_tail);
    // a tiny but representable tail is kept exactly (no 1 - sum cancellation)
    std::vector<float> l3(10, -30.0f);
    l3[0] = 0.0f;
    TCHECK(make_record(l3.data(), 10, 1, 0, -1, nullptr, s, r), "make_record 3");
    const double want = log(9.0*exp(-30.0)) - log(1.0 + 9.0*exp(-30.0));
    TCHECK(near(r.log_tail, want, 1e-5), "tiny tail %g vs %g", r.log_tail, want);
}

static void test_ref() {
    // 6 tokens, ref ids {2, 0}: ref_lps = logit - lse, ref_log_tail = log of p over {1, 3, 4, 5}
    std::vector<float> l = { 1.0f, 2.0f, 0.0f, -1.0f, 0.5f, 3.0f };
    double se = 0.0;
    for (float v : l) {
        se += exp((double) v);
    }
    const double lse = log(se);
    const int32_t ref_ids[2] = { 2, 0 };
    scratch s;
    record  r;
    TCHECK(make_record(l.data(), 6, 2, 3, 4, ref_ids, s, r), "make_record");
    TCHECK(r.ids[0] == 5 && r.ids[1] == 1, "own ids %d %d", r.ids[0], r.ids[1]);
    TCHECK((int) r.ref_lps.size() == 2, "ref size");
    TCHECK(near(r.ref_lps[0], 0.0 - lse, 1e-6) && near(r.ref_lps[1], 1.0 - lse, 1e-6), "ref_lps %g %g", r.ref_lps[0], r.ref_lps[1]);
    const double out = exp(2.0) + exp(-1.0) + exp(0.5) + exp(3.0);
    TCHECK(near(r.ref_log_tail, log(out) - lse, 1e-6), "ref_log_tail %g vs %g", r.ref_log_tail, log(out) - lse);
    TCHECK(near(r.log_tail, log(exp(1.0) + exp(0.0) + exp(-1.0) + exp(0.5)) - lse, 1e-6), "own tail");
    TCHECK(near(r.lp_next, 0.5 - lse, 1e-6), "lp_next");
    // a ref id outside the vocabulary is refused
    const int32_t bad[2] = { 2, 6 };
    record r2;
    TCHECK(!make_record(l.data(), 6, 2, 3, 4, bad, s, r2), "bad ref id accepted");
    set_ref_missing(r, 2);
    TCHECK(std::isnan(r.ref_lps[0]) && std::isnan(r.ref_lps[1]) && std::isnan(r.ref_log_tail), "missing ref part is NaN");
}

static void test_nonfinite() {
    scratch s;
    record  r;
    std::vector<float> l = { 0.0f, 1.0f, 2.0f };
    TCHECK(make_record(l.data(), 3, 2, 0, -1, nullptr, s, r), "finite row");
    const float bad[3] = { NAN, INFINITY, -INFINITY };
    for (float b : bad) {
        std::vector<float> lb = l;
        lb[1] = b;
        record rb;
        TCHECK(!make_record(lb.data(), 3, 2, 0, -1, nullptr, s, rb), "non-finite %g accepted", b);
        TCHECK(rb.ids.empty(), "nothing written on a refused row");
    }
    TCHECK(!make_record(nullptr, 3, 2, 0, -1, nullptr, s, r), "null row");
    TCHECK(!make_record(l.data(), 3, 4, 0, -1, nullptr, s, r), "k > V");
    TCHECK(!make_record(l.data(), 3, 0, 0, -1, nullptr, s, r), "k 0");
    TCHECK(!make_record(l.data(), 3, 2, 0, 3, nullptr, s, r), "next outside");
    TCHECK(!make_record(l.data(), 3, 2, 0, -2, nullptr, s, r), "next -2");
}

static std::string tmp_path(const char * name) {
    const char * t = getenv("TEMP");
    if (t == nullptr || t[0] == '\0') {
        t = getenv("TMPDIR");
    }
    std::string d = t && t[0] ? t : ".";
    return d + "/" + name;
}

static std::vector<float> fixture_logits(int n_vocab, int n, double shift) {
    std::vector<float> l(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        l[i] = (float) (4.0*sin(0.37*i + 1.3*n) + 0.25*((i*7 + n*3) % 11) + shift*cos(0.11*i + n));
    }
    return l;
}

// writes a dump of n_forced forced ids from fixture_logits; ref = the base's records or null. The file ends after
// n_write records (n_write <= n_forced + 1): fewer = ABORTED
static bool write_fixture(const std::string & path, int n_vocab, int k, const std::vector<int32_t> & forced, double shift,
                          const std::vector<record> * ref, uint64_t ref_hash, int n_write, std::vector<record> * out) {
    header h;
    h.n_vocab     = n_vocab;
    h.k           = k;
    h.window      = 3;
    h.n_forced    = (int32_t) forced.size();
    h.n_prompt    = 3;
    const int32_t prompt[3] = { 10, 11, 12 };
    h.prompt_hash = hash_ids(prompt, 3, 3);
    h.forced_hash = hash_ids(forced.data(), forced.size(), (int64_t) forced.size());
    h.flags       = ref ? FLAG_REF : 0;
    h.ref_hash    = ref ? ref_hash : 0;
    if (ref && (int) ref->size() < h.n_forced + 1) {
        h.flags |= FLAG_REF_PARTIAL;
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    bool ok = write_header(f, h, forced.data());
    scratch s;
    record  r;
    int32_t n_rec = 0;
    for (int n = 0; n < n_write && ok; ++n) {
        const std::vector<float> l = fixture_logits(n_vocab, n, shift);
        const int32_t next = n < h.n_forced ? forced[n] : -1;
        const int32_t * rid = ref && n < (int) ref->size() ? (*ref)[n].ids.data() : nullptr;
        ok = make_record(l.data(), n_vocab, k, n, next, rid, s, r);
        if (ok && ref && rid == nullptr) {
            set_ref_missing(r, k);
        }
        ok = ok && write_record(f, r, k, ref != nullptr);
        if (ok && out) {
            out->push_back(r);
        }
        n_rec += ok ? 1 : 0;
    }
    const int32_t flags = h.flags | (n_write < h.n_forced + 1 ? FLAG_ABORTED : 0);
    ok = ok && patch_header(f, n_rec, flags);
    ok = (fclose(f) == 0) && ok;
    return ok;
}

static void test_round_trip() {
    const std::string pb = tmp_path("test-fn-l22-dump-base.l22d");
    const std::string pt = tmp_path("test-fn-l22-dump-test.l22d");
    const std::vector<int32_t> forced = { 5, 7, 9, 11, 13 };
    std::vector<record> base_w;
    TCHECK(write_fixture(pb, 300, 16, forced, 0.0, nullptr, 0, (int) forced.size() + 1, &base_w), "write base");
    header hb;
    std::vector<int32_t> fb;
    std::vector<record>  rb;
    std::string err;
    TCHECK(read_dump(pb, hb, fb, rb, err), "read base: %s", err.c_str());
    TCHECK(hb.n_rec == 6 && hb.n_forced == 5 && hb.k == 16 && hb.n_vocab == 300 && hb.window == 3 && hb.flags == 0,
           "base header n_rec %d flags %d", hb.n_rec, hb.flags);
    TCHECK(fb == forced, "forced ids");
    bool same = rb.size() == base_w.size();
    for (size_t i = 0; same && i < rb.size(); ++i) {
        same = rb[i].ids == base_w[i].ids && rb[i].lps == base_w[i].lps && rb[i].log_tail == base_w[i].log_tail &&
               rb[i].lp_next == base_w[i].lp_next && rb[i].next == base_w[i].next && rb[i].n == (int) i;
    }
    TCHECK(same, "records round trip");
    TCHECK(rb.back().next == -1 && rb[0].next == 5, "next ids");
    TCHECK(near(rb[0].lp_next, naive(fixture_logits(300, 0, 0.0)).lp[5], 2e-5), "lp_next of record 0");

    // the test run with REF against the base, aborted after 4 records
    const uint64_t rh = ref_hash_of(hb, rb);
    TCHECK(write_fixture(pt, 300, 16, forced, 0.5, &rb, rh, 4, nullptr), "write test");
    header ht;
    std::vector<int32_t> ft;
    std::vector<record>  rt;
    TCHECK(read_dump(pt, ht, ft, rt, err), "read test: %s", err.c_str());
    TCHECK(ht.n_rec == 4 && (ht.flags & FLAG_REF) && (ht.flags & FLAG_ABORTED) && !(ht.flags & FLAG_REF_PARTIAL),
           "test header n_rec %d flags %d", ht.n_rec, ht.flags);
    TCHECK(ht.ref_hash == rh && ht.forced_hash == hb.forced_hash && ht.prompt_hash == hb.prompt_hash, "hashes");
    bool ref_ok = true;
    for (int n = 0; n < ht.n_rec; ++n) {
        const naive nv(fixture_logits(300, n, 0.5));
        for (int i = 0; i < 16; ++i) {
            ref_ok = ref_ok && near(rt[n].ref_lps[i], nv.lp[rb[n].ids[i]], 2e-5);
        }
        ref_ok = ref_ok && near(rt[n].ref_log_tail, nv.log_outside(rb[n].ids), 2e-5);
    }
    TCHECK(ref_ok, "REF part against the naive log-softmax at the base ids");

    // bad files: truncated, trailing bytes, bad magic
    {
        FILE * f = fopen(pb.c_str(), "rb");
        std::vector<char> all;
        if (f) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
                all.insert(all.end(), buf, buf + n);
            }
            fclose(f);
        }
        const std::string px = tmp_path("test-fn-l22-dump-bad.l22d");
        auto put = [&](const std::vector<char> & v) {
            FILE * g = fopen(px.c_str(), "wb");
            if (g) {
                fwrite(v.data(), 1, v.size(), g);
                fclose(g);
            }
        };
        header hx;
        std::vector<int32_t> fx;
        std::vector<record>  rx;
        std::vector<char> v = all;
        v.resize(v.size() - 5);
        put(v);
        TCHECK(!read_dump(px, hx, fx, rx, err), "truncated file accepted");
        v = all;
        v.push_back(0);
        put(v);
        TCHECK(!read_dump(px, hx, fx, rx, err), "trailing byte accepted");
        v = all;
        v[0] = 'X';
        put(v);
        TCHECK(!read_dump(px, hx, fx, rx, err), "bad magic accepted");
        v = all;
        const int32_t big = 9999;
        memcpy(v.data() + OFF_N_REC, &big, 4);
        put(v);
        TCHECK(!read_dump(px, hx, fx, rx, err), "n_rec > n_forced + 1 accepted");
        TCHECK(!read_dump(tmp_path("test-fn-l22-dump-missing.l22d"), hx, fx, rx, err), "missing file accepted");
        remove(px.c_str());
    }
    TCHECK(record_bytes(16, false) == 16 + 128 && record_bytes(16, true) == 16 + 128 + 68, "record_bytes");
    remove(pb.c_str());
    remove(pt.c_str());
}

static void test_names_hashes_plan() {
    TCHECK(name_ok("base_img2.off-1"), "plain name");
    TCHECK(name_ok("a"), "one char");
    TCHECK(name_ok(std::string(96, 'x')), "96 chars");
    const char * bad[] = { "", "../a", "a/b", "a\\b", ".x", "-x", "_x", "a b", "a:b", "nul", "NUL.l22d", "con", "COM1", "lpt9.x" };
    for (const char * b : bad) {
        TCHECK(!name_ok(b), "bad name \"%s\" accepted", b);
    }
    TCHECK(!name_ok(std::string(97, 'x')), "97 chars accepted");
    TCHECK(name_ok("COM10") && name_ok("console"), "device-like names that are not devices");

    TCHECK(fnv1a("", 0) == 0xcbf29ce484222325ULL, "fnv empty");
    TCHECK(fnv1a("a", 1) == 0xaf63dc4c8601ec8cULL, "fnv a");
    const int32_t ids[3] = { 1, 2, 3 };
    TCHECK(hash_ids(ids, 3, 3) == 0xd40cae13e35ed256ULL, "hash_ids 1 2 3: %llx", (unsigned long long) hash_ids(ids, 3, 3));
    TCHECK(hash_ids(nullptr, 0, 0) == 0xa8c7f832281a39c5ULL, "hash_ids empty");
    const int32_t ids2[2] = { -1, 248319 };
    TCHECK(hash_ids(ids2, 2, 7) == 0x45c740e83f39a0edULL, "hash_ids -1 248319");

    TCHECK(plan_forced(5000, 1000000) == 4095, "cap at 4095 forced (4096 records)");
    TCHECK(plan_forced(10, 5) == 5, "room");
    TCHECK(plan_forced(10, -3) == 0, "no room");
    TCHECK(plan_forced(0, 100) == 0, "none");
    TCHECK(clamp_i(0, 1, 64) == 1 && clamp_i(100, 1, 64) == 64 && clamp_i(3, 1, 64) == 3, "clamp");
    // windows: 4095 forced in windows of 3 = 1365 decodes, the last one full; 10 in 3s = 3,3,3,1
    int nw = 0;
    int last = 0;
    for (int j = 0; j < 10; j += 3) {
        last = std::min(3, 10 - j);
        nw++;
    }
    TCHECK(nw == 4 && last == 1, "window cut at the end");
}

// fixture for kld_dump.py --selftest --fixture <dir>: base (V 500, k 32, 7 forced, shift 0) and test (shift 0.4, REF)
static int write_fixtures(const std::string & dir) {
    const std::vector<int32_t> forced = { 3, 141, 59, 265, 358, 97, 93 };
    std::vector<record> rb;
    const std::string pb = dir + "/fixture_base.l22d";
    const std::string pt = dir + "/fixture_test.l22d";
    const std::string pn = dir + "/fixture_noref.l22d";
    if (!write_fixture(pb, 500, 32, forced, 0.0, nullptr, 0, (int) forced.size() + 1, &rb)) {
        fprintf(stderr, "cannot write %s\n", pb.c_str());
        return 1;
    }
    header hb;
    std::vector<int32_t> fb;
    std::vector<record>  rr;
    std::string err;
    if (!read_dump(pb, hb, fb, rr, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (!write_fixture(pt, 500, 32, forced, 0.4, &rr, ref_hash_of(hb, rr), (int) forced.size() + 1, nullptr) ||
        !write_fixture(pn, 500, 32, forced, 0.4, nullptr, 0, (int) forced.size() + 1, nullptr)) {
        fprintf(stderr, "cannot write the test fixtures\n");
        return 1;
    }
    printf("fixtures written: %s %s %s\n", pb.c_str(), pt.c_str(), pn.c_str());
    return 0;
}

int main(int argc, char ** argv) {
    test_topk_vs_naive(1000, 64, 1);
    test_topk_vs_naive(1000, 1, 2);
    test_topk_vs_naive(248320, 64, 3);
    test_topk_vs_naive(248320, 256, 4);
    test_ties();
    test_tail_zero();
    test_ref();
    test_nonfinite();
    test_round_trip();
    test_names_hashes_plan();

    int rc = g_fail == 0 ? 0 : 1;
    if (argc > 1) {
        rc = write_fixtures(argv[1]) || rc;
    }
    printf("test-fn-l22-dump: %d checks, %d failed\n", g_checks, g_fail);
    return rc;
}
