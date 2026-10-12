// [TAG_FN_L19_CONVSTORE] the policy of the agent conversation store (tools/server/server-convstore.h, header only): which
// checkpoint a full ring drops, the eviction value of a stored conversation, the thinning of checkpoints on disk. CPU only,
// no model.
//
// Checks: fixed cases; an agent-loop replay (a system + tools block, then turns that each add a checkpoint at the user
// message and one at the assistant-turn start) in which the old rule (min-step spacing, then the oldest) loses the
// system-block checkpoint and the new one keeps it, the last user message and the newest; random rings (the victim is
// never the incoming checkpoint, and never a protected one while an unprotected one exists); the greedy-dual value
// (monotone in tokens, restores and bytes; aging).

#include "../tools/server/server-convstore.h"

#include <algorithm>
#include <cstdio>
#include <random>
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

static void test_victim_fixed() {
    // ring 8000 (system block), 17000, 26000, 35000; incoming 36000; last user message at 35500
    {
        const std::vector<int64_t> ns = { 8000, 17000, 26000, 35000, 36000 };
        const int v = convstore_ckpt_victim(ns, 8000, 35500);
        TCHECK(v == 2, "victim %d, want 2 (26000: not the system block, not the last user message, not halfway)", v);
    }
    // the system block end unknown: the oldest is protected
    {
        const std::vector<int64_t> ns = { 100, 5000, 5100, 20000, 20004 };
        const int v = convstore_ckpt_victim(ns, -1, -1);
        // protected: 100 (oldest), 20004 (newest), halfway 10002 -> 5100; left 5000 and 20000: gaps 4900 and 14900
        TCHECK(v == 1, "victim %d, want 1", v);
    }
    // smallest gap among the unprotected, the older of equals
    {
        const std::vector<int64_t> ns = { 1000, 2000, 3000, 4000, 9000, 9004 };
        const int v = convstore_ckpt_victim(ns, 1000, 9000);
        // protected: 1000, 9000 (last user), halfway 4502 -> 4000, 9004; left 2000 (gap 1000), 3000 (gap 1000) -> 2000
        TCHECK(v == 1, "victim %d, want 1", v);
    }
    // a ring of one: its only entry goes
    {
        const std::vector<int64_t> ns = { 500, 900 };
        TCHECK(convstore_ckpt_victim(ns, 500, 500) == 0, "ring of one");
    }
    TCHECK(convstore_ckpt_victim({ 7 }, 0, 0) == -1, "nothing to drop");
    TCHECK(convstore_ckpt_victim({}, 0, 0) == -1, "empty");
}

// the old rule of create_checkpoint() for a full ring ([TAG_SYNC_CKPT_EVICT]): entries within min_step of an earlier one
// (not of the current task) go, then the oldest
static void old_rule(std::vector<int64_t> & ring, std::vector<int> & task, int cur_task, size_t n_max, int64_t min_step) {
    if (ring.size() + 1 >= n_max) {
        int64_t last = -1;
        for (size_t i = 0; i < ring.size();) {
            if (task[i] != cur_task && last >= 0 && ring[i] <= last + min_step) {
                ring.erase(ring.begin() + i);
                task.erase(task.begin() + i);
                continue;
            }
            last = ring[i];
            ++i;
        }
    }
    while (ring.size() >= n_max) {
        ring.erase(ring.begin());
        task.erase(task.begin());
    }
}

static void test_agent_loop() {
    const size_t  n_max    = 4;      // Flash-Next: 512 MiB / 112.6 MiB
    const int64_t min_step = 8192;
    const int64_t sys_end  = 9000;   // system + tools block

    std::vector<int64_t> ring_old;
    std::vector<int>     task_old;
    std::vector<int64_t> ring_new;

    std::mt19937 rng(42);
    int64_t n = sys_end;
    bool old_lost_sys = false;
    for (int turn = 0; turn < 60; ++turn) {
        // the turn: a user message (or a tool result) at n, then its tokens, then the assistant-turn start at the end
        const int64_t user = n;
        const int64_t len  = 300 + (int64_t) (rng() % 4000);
        const int64_t tail = user + len - 4;
        for (int64_t pos : { user, tail }) {
            if (ring_old.empty() || pos == user || pos > ring_old.back() + min_step || pos == tail) {
                old_rule(ring_old, task_old, turn, n_max, min_step);
                ring_old.push_back(pos);
                task_old.push_back(turn);
            }
            if (ring_new.size() >= n_max) {
                std::vector<int64_t> ns = ring_new;
                ns.push_back(pos);
                const int v = convstore_ckpt_victim(ns, sys_end, user);
                TCHECK(v >= 0 && v < (int) ring_new.size(), "turn %d: victim %d out of range", turn, v);
                ring_new.erase(ring_new.begin() + v);
            }
            ring_new.push_back(pos);
            TCHECK(std::is_sorted(ring_new.begin(), ring_new.end()), "turn %d: ring not ascending", turn);
            TCHECK(ring_new.size() <= n_max, "turn %d: ring of %zu", turn, ring_new.size());
        }
        // the first turn's user checkpoint is the end of the system block
        const bool new_has_sys = std::find(ring_new.begin(), ring_new.end(), sys_end) != ring_new.end();
        const bool new_has_user = std::find(ring_new.begin(), ring_new.end(), user) != ring_new.end();
        const bool new_has_tail = ring_new.back() == tail;
        TCHECK(new_has_sys, "turn %d: the new rule lost the system-block checkpoint", turn);
        TCHECK(new_has_user, "turn %d: the new rule lost the last user message", turn);
        TCHECK(new_has_tail, "turn %d: the new rule lost the newest", turn);
        // one more point, between about a quarter of the way and the end (a ladder with one rung)
        if (tail > 4 * sys_end) {
            TCHECK(ring_new.size() == n_max && ring_new[1] * 9 >= tail * 2 && ring_new[1] < user,
                   "turn %d: no point between a quarter of the way and the end (n = %lld, ring[1] = %lld)", turn,
                   (long long) tail, (long long) ring_new[1]);
        }
        old_lost_sys = old_lost_sys || std::find(ring_old.begin(), ring_old.end(), sys_end) == ring_old.end();
        n = user + len;
    }
    TCHECK(old_lost_sys, "the old rule was expected to lose the system-block checkpoint in a long loop");
    printf("agent loop: 60 turns to %lld tokens, the old rule %s the system-block checkpoint, the new keeps it\n",
           (long long) n, old_lost_sys ? "lost" : "kept");
}

static void test_victim_random() {
    std::mt19937 rng(7);
    for (int it = 0; it < 20000; ++it) {
        const int n = 2 + (int) (rng() % 8);
        std::vector<int64_t> ns;
        int64_t p = (int64_t) (rng() % 100);
        for (int i = 0; i < n; ++i) {
            ns.push_back(p);
            p += 1 + (int64_t) (rng() % 5000);
        }
        const int64_t sys_end   = (rng() % 3 == 0) ? -1 : ns[rng() % n];
        const int64_t last_user = (rng() % 3 == 0) ? -1 : ns[n - 1] - (int64_t) (rng() % 3000);
        const int v = convstore_ckpt_victim(ns, sys_end, last_user);
        TCHECK(v >= 0 && v < n - 1, "random %d: victim %d of %d", it, v, n);
        if (n >= 5) {
            // the system-block one (the newest at or before sys_end, or the oldest) survives a ring of 4 or more
            int s = 0;
            if (sys_end >= 0) {
                s = -1;
                for (int i = 0; i < n - 1; ++i) {
                    if (ns[i] <= sys_end) {
                        s = i;
                    }
                }
                if (s < 0) {
                    s = 0;
                }
            }
            TCHECK(v != s, "random %d: dropped the system-block checkpoint %d", it, s);
        }
    }
}

static void test_gdsf() {
    const uint64_t GiB = 1024ull * 1024 * 1024;
    const double a = convstore_gdsf(0.0, 0, 30000, GiB, 2048);
    const double b = convstore_gdsf(0.0, 1, 30000, GiB, 2048);
    const double c = convstore_gdsf(0.0, 0, 30000, 2 * GiB, 2048);
    const double d = convstore_gdsf(0.0, 0, 60000, GiB, 2048);
    const double e = convstore_gdsf(1e6, 0, 30000, GiB, 2048);
    TCHECK(b > a, "a restore raises the value");
    TCHECK(c < a, "more bytes lower it");
    TCHECK(d > a, "more tokens raise it");
    TCHECK(e > a + 9e5, "aging adds L");
    TCHECK(convstore_gdsf(0.0, 0, 0, 0, 0) == 0.0, "zero");
    // below the share of one entry, bytes do not count
    TCHECK(convstore_gdsf(0.0, 0, 2000, GiB / 8, 2048, 2 * GiB) < convstore_gdsf(0.0, 0, 30000, GiB, 2048, 2 * GiB),
           "a small side request is worth less than a session while the count binds");
    TCHECK(convstore_gdsf(0.0, 0, 2000, GiB / 8, 2048) > convstore_gdsf(0.0, 0, 30000, GiB, 2048),
           "without the share, per byte the small one is worth more");

    // replay: 3 sessions of ~1 GiB take turns with title requests of 2K tokens / 0.13 GiB; 6 entries of room (a 32 GiB
    // budget over 16 entries: 2 GiB each). The sessions must outlive the one-off title requests.
    struct ent { int id; double v; int64_t hits; int64_t n; uint64_t bytes; };
    std::vector<ent> store;
    double L = 0;
    auto put = [&](int id, int64_t hits, int64_t n, uint64_t bytes) {
        while (store.size() >= 6) {
            auto it = std::min_element(store.begin(), store.end(), [](const ent & x, const ent & y) { return x.v < y.v; });
            L = std::max(L, it->v);
            store.erase(it);
        }
        store.push_back({ id, convstore_gdsf(L, hits, n, bytes, 2048, 32 * GiB / 16), hits, n, bytes });
    };
    for (int round = 0; round < 20; ++round) {
        for (int s = 0; s < 3; ++s) {
            put(s, round, 30000 + 2000 * round, GiB + (uint64_t) round * 64 * 1024 * 1024);
            store.erase(std::remove_if(store.begin(), store.end(), [&](const ent & x) { return x.id == s && x.hits < round; }), store.end());
        }
        for (int t = 0; t < 4; ++t) {
            put(100 + round * 10 + t, 0, 2000, GiB / 8);
        }
        int alive = 0;
        for (const auto & x : store) {
            alive += x.id < 3;
        }
        TCHECK(alive == 3, "round %d: %d of the 3 sessions stored", round, alive);
    }
}

static void test_thin() {
    TCHECK(convstore_thin({ 1, 2, 3 }, 3) == -1, "within the limit");
    TCHECK(convstore_thin({ 1000, 1100, 5000, 9000 }, 3) == 1, "smallest gap");
    TCHECK(convstore_thin({ 1000, 2000, 3000, 3001 }, 3) == 0 || convstore_thin({ 1000, 2000, 3000, 3001 }, 3) == 1,
           "never the newest");
    TCHECK(convstore_thin({ 5, 6 }, 1) == 0, "the older of two");
    TCHECK(convstore_near({ 100, 5000 }, 900, 1024), "near");
    TCHECK(!convstore_near({ 100, 5000 }, 2000, 1024), "not near");
    TCHECK(!convstore_near({}, 2000, 1024), "nothing kept");
}

int main() {
    test_victim_fixed();
    test_agent_loop();
    test_victim_random();
    test_gdsf();
    test_thin();
    printf("test-fn-convstore: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
