#pragma once

// [TAG_FN_L22_LOGITDUMP] round 22 P15: the server's logit dump. With LLAMA_FN_L22_LOGITDUMP=<dir>, a request whose "l22"
// object asks for a dump gets, after its prompt, the first-token top-k and then its forced tokens decoded in windows
// (all rows output, decode phase GEN, no sampling, no drafting), the top-k log-probs of every position written to <dir>.
// The server calls the hook once per prompt, after the prompt's last batch is decoded and before the first token is
// sampled, only for a task that carries an "l22" object.

#include "server-common.h"

#include "llama.h"

struct server_fn_l22_dump_args {
    llama_context *     ctx;      // the target context; its last decode holds the prompt's last row
    const llama_vocab * vocab;
    llama_seq_id        seq_id;   // the slot's sequence
    llama_pos           pos_next; // the position after the prompt
    int32_t             i_batch;  // the prompt's last output row in that decode (llama_get_logits_ith index)
    int                 id_task;
    int                 id_slot;
    const json *        l22;      // the task's "l22" object (not null)
    // guards (R-P15-1)
    bool                  bs_attached;  // the slot samples on the backend (no raw logits for the row)
    bool                  multi;        // the task is part of an n_cmpl fan-out (parent or child)
    int                   n_other_busy; // other slots that are processing (they share the logits buffer)
    int32_t               n_ctx_slot;   // the slot's context size
    int32_t               n_prompt;     // tokens of the prompt (cached + new)
    const server_tokens * prompt;       // the slot's prompt tokens (get_text_tokens for a hash)
};

// true: the dump took the request (it may have decoded forced rows into seq_id); the caller sends the final response
// with no generated tokens and drops the slot's cached prompt. false: generate as usual. P0 stub: false
bool server_fn_l22_dump_after_prompt(const server_fn_l22_dump_args & a);
