#pragma once

// [TAG_FN_L18_KEEPCTX] Conversation store of the slots (LLAMA_FN_L18_KEEPCTX=1).
//
// A request with another prompt prefix (a chat title, a second conversation) used to take the slot and drop the
// conversation in it: the next message of that conversation processed the whole prompt again (45,404 tokens, 23.6 s
// before the first token on Flash-Next). Before the slot's state is replaced, the store writes it to a file: the
// target and draft sequence states (llama_state_seq_write_stream_ext: the exact bytes llama_state_seq_get_data_ext
// gives), the slot's context checkpoints and the drafter's carry. A later prompt that continues a stored conversation
// further than the slot content gets it back, and only its new tokens are processed.
//
// The files bypass the OS file cache (Windows: FILE_FLAG_NO_BUFFERING), so a model bigger than RAM keeps its cached
// pages, and they are deleted when closed (FILE_FLAG_DELETE_ON_CLOSE), also when the process ends abnormally.

#include "server-task.h"

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

struct llama_context;
struct common_speculative;

struct server_keepctx_params {
    std::string dir;          // folder of the store files
    uint64_t    budget = 0;   // bytes on disk for all conversations
    int32_t     n_max  = 4;   // conversations kept
    bool        verify = false; // after a restore, compare a hash of the restored states with the stored bytes (tests)
};

class server_keepctx_file;

struct server_keepctx_entry {
    uint64_t id = 0;

    server_prompt prompt;               // tokens, and the checkpoint positions (their bytes are in the file)
    std::vector<uint64_t> ckpt_sizes;   // per checkpoint: target, draft and drafter bytes

    uint64_t size_tgt = 0;
    uint64_t size_dft = 0;
    uint64_t n_bytes  = 0;              // bytes in the file

    std::vector<uint8_t> carry;         // the drafter's carry (common_speculative_get_carry)

    uint64_t hash_tgt = 0;              // params.verify: hashes of the stored target and draft state bytes
    uint64_t hash_dft = 0;

    std::shared_ptr<server_keepctx_file> file;
};

struct server_keepctx {
    explicit server_keepctx(const server_keepctx_params & params);
    ~server_keepctx();

    bool ok() const { return ok_; }

    const server_keepctx_params & params() const { return params_; }

    std::list<server_keepctx_entry> entries; // least recently stored first

    uint64_t n_bytes() const;

    // store the sequence seq_id with its prompt (tokens and checkpoints). Older states of the same conversation (entries
    // whose tokens are a prefix of it) and then the oldest entries are dropped to stay within the budget and the count,
    // never `keep`. false: nothing stored (too big, no room beside `keep`, or an I/O error).
    bool save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
              llama_seq_id seq_id, const server_keepctx_entry * keep, std::string & info);

    // restore the entry into seq_id and prompt; the entry is removed in every case. false: the sequence and prompt are
    // in an unknown state and the caller must clear them.
    bool load(std::list<server_keepctx_entry>::iterator it, server_prompt & prompt, llama_context * ctx_tgt,
              llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq_id, std::string & info);

    // remove the entries whose tokens are a prefix of tokens (an older state of the same conversation), except keep
    int drop_prefixes_of(const server_tokens & tokens, const server_keepctx_entry * keep);

private:
    bool selftest(std::string & err);

    server_keepctx_params params_;
    bool                  ok_     = false;
    uint64_t              n_next_ = 0;
};
