# PetAI CPU KV checkpoints

The `petai/kv-checkpoint` branch is the native source used for PetAI's CPU KV
checkpoint integration. It starts at upstream commit
`a327b494f874a319605e6fd7e3439678daa4d07d`. The upstream license and existing source
comments are preserved. Native changes are limited to the files listed below.

The application pins an exact commit and archive checksum, not the moving branch
head. The fork owns state transfer; the application owns file I/O, model identity,
checksums, atomic replacement, retention, and Swift integration. Keeping those
policies out of the engine lets other clients choose their own storage contract.

## API

`litert_lm_session_transfer_state(session, transfer, user_data, reading)` transfers
an idle, exclusively owned CPU session synchronously. `reading=false` exports
state; `reading=true` imports into a fresh session. The callback borrows a mutable
buffer in chunks of at most 64 KiB and must finish each transfer before returning.
It must not retain the pointer, reenter the session, or start inference.

On export, read the buffer without modifying it. On import, fill the entire
buffer. `transfer(NULL, 0, user_data)` finalizes the transfer; the caller can verify
its checksum and EOF there. Return `false` on any failure. The API returns an
Abseil status code, with zero indicating success. Cleanup failure (code 10)
requires engine recreation before further inference.

State includes KV buffers, tensor layout, cursor, processed token IDs,
pending-token embeddings, and sampler RNG. Existing state buffers are transferred
directly rather than cloning the whole KV cache. Restore supports next-request
full-prompt reuse, not continuation of a half-generated response.

The caller must wait until the terminal generation callback has returned. Native
tasks and execution-manager ownership are checked by the engine as well. The
supported scope is CPU, batch one, in-place buffers, without LoRA or speculative
decoding. GPU, untrusted cache imports, cross-engine portability, and format
migration are outside this contract. The binary state format is version 2.

## Modified native files

State transfer:

- `c/engine.h`, `c/kv_checkpoint.cc`, `c/BUILD`
- `runtime/engine/engine.h`
- `runtime/core/session_advanced.h`, `runtime/core/kv_checkpoint_session.cc`, `runtime/core/BUILD`
- `runtime/framework/resource_management/execution_manager.h`
- `runtime/framework/resource_management/threaded_execution_manager.h`
- `runtime/framework/resource_management/kv_checkpoint_manager.cc`
- `runtime/framework/resource_management/resource_manager.cc`
- `runtime/framework/resource_management/BUILD`
- `runtime/executor/llm_executor_base.h`
- `runtime/executor/llm_litert_compiled_model_executor.h`
- `runtime/executor/kv_checkpoint.cc`, `runtime/executor/kv_checkpoint_io.h`
- `runtime/executor/BUILD`

Pending-token prefill:

- `runtime/executor/llm_litert_compiled_model_executor.cc`
- `runtime/executor/pending_prefill_embeddings.h`
- `runtime/executor/pending_prefill_embeddings_test.cc`
- `runtime/executor/BUILD` (shared with state transfer)

There are 20 distinct native files. `DeepCopy`, `CloneContext`, `RestoreContext`,
`Serialize`, and `Load` are unchanged. The separate prefill fix reconstructs
missing embeddings after rewind; it is not file persistence logic.

## Verification

The source-boundary checks require Python 3 and a Git checkout containing the
upstream base commit:

```bash
python3 tools/kv_checkpoint/check_contract_test.py
```

The prefill behavior tests use the repository's Bazel dependencies:

```bash
bazel test //runtime/executor:pending_prefill_embeddings_test \
  --define=LITERT_LM_FST_CONSTRAINTS_DISABLED=1
```

The iOS framework target is `//swift:CLiteRTLM`. PetAI's build uses Bazel 7.6.1
with `--define=LITERT_LM_FST_CONSTRAINTS_DISABLED=1`.

Application-level save/restore, interrupted file writes, cache identity, and
visible-output equality tests remain in the application repository. Source checks
and compilation do not prove device latency, memory safety, or app lifecycle
behavior. No new performance result is implied by publishing this branch.

## Releases

- `petai-ios-embedding-native-v1` contains LiteRT and SentencePiece frameworks
  for embeddings. It is an active application dependency, not the KV engine.
- `petai-v0.14.0-topk-poc.1` is a separate telemetry experiment, not the CPU KV
  checkpoint distribution.
- The KV engine is built from the pinned source commit. This branch does not
  replace either existing release or publish a new prebuilt engine binary.

Do not delete or repurpose an existing release while a consumer still pins it.
