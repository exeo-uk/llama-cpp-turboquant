#pragma once

// [TAG_FN_L22_PIN] round 22 P6: entry points beyond src/llama-fn-l22.h. L22-REQUEST (requests.md R-P6-1): the steward
// may move this declaration into the requests section of llama-fn-l22.h; until then its callers include this header.

#include <cstddef>
#include <cstdint>

struct llama_fn_pin_seg;

// R-P14-5 (P14 asks, P6 answers): llama_fn_pin_query plus the device address of seg->src for a pinned segment (0
// when unknown: no proc ggml_backend_fn_host_dev_ptr from P7, or an unpinned segment). Valid while the ticket is held
bool llama_fn_pin_query_dev(const void * p, size_t n, llama_fn_pin_seg * seg, uint64_t * dev);
