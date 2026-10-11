#pragma once

// [TAG_FN_L18_FIRSTANS_STALL] diagnostics only, off unless LLAMA_FN_L18_STALL=<sample period in ms> (Windows; a no-op
// elsewhere). A sampler thread reads the state of every thread of this process (NtQuerySystemInformation: running, ready,
// or waiting and on what - user request, page in, push lock, ...) plus the process's CPU time, page faults, hard faults,
// read bytes and the RAM available, every <period> ms into a ring. Code marks (llama_fn_stall_mark) go into a second ring.
// llama_fn_stall_dump writes the last LLAMA_FN_L18_STALL_WIN ms (1500) of both to the log: one line per 50 ms of process
// counters, one line per named or busy thread (one letter per 50 ms), and the marks. A decode of <= 64 tokens that took
// longer than LLAMA_FN_L18_STALL_SLOW ms (250) dumps by itself (at most one dump per 2 s), as does a bridge wait timeout.
// Nothing here changes a value the model computes.

#include <cstdint>

bool llama_fn_stall_on();

// names the calling thread (SetThreadDescription) when the probe is on; k >= 0 is appended
void llama_fn_thread_name(const char * name, int k = -1);

// a mark: what must be a string literal (kept by pointer)
void llama_fn_stall_mark(const char * what, int64_t a = 0, int64_t b = 0);

// the last window of samples and marks to the log
void llama_fn_stall_dump(const char * why);

// a decode begins (returns its start time in us, 0 when off) / ends (dumps if a small one was slow)
int64_t llama_fn_stall_decode_begin(int n_tokens, int n_outputs);
void    llama_fn_stall_decode_end(int64_t t0_us, int n_tokens);

// a mark at the start and one at the end of a scope
struct llama_fn_stall_scope {
    const char * end;
    llama_fn_stall_scope(const char * begin, const char * end, int64_t a = 0, int64_t b = 0) : end(end) {
        llama_fn_stall_mark(begin, a, b);
    }
    ~llama_fn_stall_scope() {
        llama_fn_stall_mark(end);
    }
};
