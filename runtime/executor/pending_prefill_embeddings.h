#ifndef LITERT_LM_RUNTIME_EXECUTOR_PENDING_PREFILL_EMBEDDINGS_H_
#define LITERT_LM_RUNTIME_EXECUTOR_PENDING_PREFILL_EMBEDDINGS_H_

#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "runtime/executor/llm_executor_processed_tokens.h"

namespace litert::lm::internal {
template <typename Lookup>
absl::Status EnsurePendingPrefillEmbeddings(TokenData& token, Lookup* embedding,
                                         Lookup* per_layer, bool needs_per_layer) {
  auto ensure = [&](std::vector<float>& output, Lookup* lookup,
                    const char* name) -> absl::Status {
    if (!output.empty()) return absl::OkStatus();
    if (lookup == nullptr)
      return absl::FailedPreconditionError(std::string("Pending prefill token requires ") + name + " lookup");
    std::vector<float> resolved;
    auto status = lookup->LookupPrefill(token.id(), resolved);
    if (!status.ok()) return status;
    if (resolved.empty())
      return absl::FailedPreconditionError(std::string("Pending prefill token has empty ") + name + " after lookup");
    output = std::move(resolved);
    return absl::OkStatus();
  };
  auto status = ensure(token.mutable_embedding(), embedding, "embedding");
  if (!status.ok()) return status;
  if (needs_per_layer)
    return ensure(token.mutable_per_layer_embedding(), per_layer, "per-layer embedding");
  return absl::OkStatus();
}
}
#endif
