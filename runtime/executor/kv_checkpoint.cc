#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include <algorithm>
#include <sstream>
#include "runtime/executor/litert/state.h"
#include "runtime/executor/kv_checkpoint_io.h"
#include "absl/status/status_macros.h"
#include "litert/cc/litert_macros.h"

namespace litert::lm {
absl::Status LlmLiteRtCompiledModelExecutorBase::TransferState(
    bool (*transfer)(void*, size_t, void*), void* user_data, bool reading) {
  if (executor_settings_.GetBackend() != Backend::CPU || !state_ ||
      llm_context_->runtime_config().output_heads.value_or(1) != 1 ||
      llm_context_->processed_context().lora_id() || mtp_drafter_)
    return absl::UnimplementedError("KV checkpoint supports CPU batch1 without LoRA or drafter");
  if (reading && llm_context_->runtime_state().current_step != 0)
    return absl::FailedPreconditionError("restore requires fresh executor");
  checkpoint::Io io(transfer, user_data, reading);
  const auto original_random = llm_context_->runtime_state().rand_gen;
  auto run = [&]() -> absl::Status {
    if (!io.Expect(uint32_t{2}))
      return absl::DataLossError("checkpoint format mismatch");
    auto* state = static_cast<LitertState*>(state_.get());
    if (state->GetBatchSize() != 1) return absl::UnimplementedError("batch1 required");
    ABSL_ASSIGN_OR_RETURN(auto signature, GetPrefillSignatureKey());
    int32_t entries = state->GetNumEntries();
    if (!io.Value(entries) || entries < 1 || entries > executor_settings_.GetMaxNumTokens())
      return absl::DataLossError("invalid state capacity");
    if (reading && entries != state->GetNumEntries()) {
      ABSL_RETURN_IF_ERROR(state->Resize(*compiled_model_, signature, entries));
    }
    auto& runtime = llm_context_->runtime_state();
    int32_t step = runtime.current_step;
    uint8_t ran_decode = runtime.ran_decode;
    if (!io.Value(step) || !io.Value(ran_decode) || step < 0 || step > entries || ran_decode > 1)
      return absl::DataLossError("invalid runtime metadata");
    auto& tokens = llm_context_->processed_context().processed_tokens();
    auto copies = tokens.GetCopyOfTokens();
    if (copies.size() != 1) return absl::UnimplementedError("batch1 required");
    auto ids = std::move(copies.front());
    auto pending = tokens.GetNextUnprocessedToken();
    uint8_t has_pending = !pending.token.empty();
    if (!io.Vector(ids, 32769) || !io.Value(has_pending) || has_pending > 1 ||
        (has_pending && ids.empty())) return absl::DataLossError("invalid token metadata");
    std::vector<float> embedding, per_layer;
    if (!reading && has_pending) {
      embedding.assign(pending.token[0]->embedding().begin(), pending.token[0]->embedding().end());
      per_layer.assign(pending.token[0]->per_layer_embedding().begin(), pending.token[0]->per_layer_embedding().end());
    }
    if (!io.Vector(embedding, 1048576) || !io.Vector(per_layer, 1048576))
      return absl::DataLossError("invalid pending embedding");
    std::vector<char> random_state;
    if (!reading && runtime.rand_gen) {
      std::ostringstream out; out << *runtime.rand_gen;
      auto value = out.str(); random_state.assign(value.begin(), value.end());
    }
    if (!io.Vector(random_state, 4096)) return absl::DataLossError("invalid random state");
    ABSL_ASSIGN_OR_RETURN(auto buffers, state->GetStateBuffers(*compiled_model_, signature));
    if (buffers.input_buffers.empty() ||
        buffers.input_buffers.size() != buffers.output_buffers.size())
      return absl::UnimplementedError("KV checkpoint requires inplace state");
    if (!io.Expect(int32_t{state->GetBatchSize()}) || !io.Expect(int32_t{entries}) ||
        !io.Expect(uint64_t{buffers.input_buffers.size()}))
      return absl::DataLossError("state metadata mismatch or truncation");
    std::vector<std::string> names;
    for (const auto& [name, unused] : buffers.input_buffers) names.emplace_back(name);
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
      auto& buffer = buffers.input_buffers.at(name);
      auto output = buffers.output_buffers.find(name);
      if (output == buffers.output_buffers.end() || buffer.Get() != output->second.Get())
        return absl::UnimplementedError("KV checkpoint requires inplace state");
      LITERT_ASSIGN_OR_RETURN(auto buffer_type, buffer.BufferType());
      if (buffer_type != TensorBufferType::kHostMemory)
        return absl::UnimplementedError("KV checkpoint requires host memory");
      LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto size, buffer.PackedSize());
      if (type.Layout().HasStrides())
        return absl::UnimplementedError("KV checkpoint requires packed tensor layout");
      auto dims = type.Layout().Dimensions();
      if (!io.Name(name) || !io.Expect(int32_t{static_cast<int32_t>(type.ElementType())}) ||
          !io.Expect(uint64_t{dims.size()}))
        return absl::DataLossError("tensor metadata mismatch");
      for (auto dim : dims) if (!io.Expect(int32_t{dim}))
        return absl::DataLossError("tensor shape mismatch");
      if (!io.Expect(uint64_t{size})) return absl::DataLossError("tensor size mismatch");
      LITERT_ASSIGN_OR_RETURN(auto locked, TensorBufferScopedLock::Create(
          buffer, reading ? TensorBuffer::LockMode::kWrite : TensorBuffer::LockMode::kRead));
      if (!io.Bytes(locked.second, size)) return absl::DataLossError("tensor transfer failed");
    }
    if (!io.Finish()) return absl::DataLossError("state transfer finalization failed");
    if (reading) {
      ProcessedTokens restored;
      if (has_pending) {
        int id = ids.back(); ids.pop_back();
        restored.AddProcessedTokens(ids);
        ABSL_RETURN_IF_ERROR(restored.AddPendingInputToken({std::make_shared<TokenData>(
            id, std::move(embedding), std::move(per_layer))}));
      } else restored.AddProcessedTokens(ids);
      if (step > restored.TokenCount()) return absl::DataLossError("step exceeds tokens");
      tokens = std::move(restored);
      runtime.current_step = step; runtime.ran_decode = ran_decode;
      if (!random_state.empty()) {
        runtime.rand_gen = std::make_shared<std::default_random_engine>();
        std::istringstream in(std::string(random_state.begin(), random_state.end()));
        if (!(in >> *runtime.rand_gen)) return absl::DataLossError("random state parse failed");
      }
      force_prepare_needed_ = true;
    }
    return absl::OkStatus();
  };
  auto result = run();
  if (reading && !result.ok()) {
    auto cleared = state_->Clear();
    if (!cleared.ok()) return absl::AbortedError("checkpoint cleanup failed; discard engine");
    llm_context_->processed_context().processed_tokens() = ProcessedTokens();
    llm_context_->runtime_state().current_step = 0;
    llm_context_->runtime_state().ran_decode = false;
    llm_context_->runtime_state().rand_gen = original_random;
    force_prepare_needed_ = true;
  }
  return result;
}
}
