// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/top_k_telemetry.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/sampling_cpu_util.h"

namespace litert::lm {

absl::StatusOr<std::vector<TopKTelemetryEvent>> ComputeTopKTelemetry(
    absl::Span<const float> logits, int batch_size, int sequence_size,
    int top_k) {
  if (batch_size <= 0 || sequence_size <= 0) {
    return absl::InvalidArgumentError(
        "Telemetry batch and sequence sizes must be positive.");
  }
  if (logits.empty() ||
      logits.size() % (batch_size * sequence_size) != 0) {
    return absl::InvalidArgumentError(
        "Telemetry logits do not match batch and sequence sizes.");
  }
  if (top_k <= 0 || top_k > kMaxTopKTelemetryCandidates) {
    return absl::InvalidArgumentError(
        "Telemetry top-k must be between 1 and 16.");
  }

  const int vocab_size = logits.size() / (batch_size * sequence_size);
  const int actual_k = std::min(top_k, vocab_size);
  auto topk_ids =
      TopKTokenIds(logits, actual_k, batch_size, sequence_size);
  if (!topk_ids.ok()) {
    return topk_ids.status();
  }

  std::vector<int> flattened_ids(batch_size * sequence_size * actual_k);
  for (int batch = 0; batch < batch_size; ++batch) {
    std::copy((*topk_ids)[batch].begin(), (*topk_ids)[batch].end(),
              flattened_ids.begin() + batch * sequence_size * actual_k);
  }

  std::vector<std::vector<float>> max_logit_values;
  auto probabilities =
      Softmax(logits, flattened_ids, kTopKTelemetryMetricTemperature,
              batch_size, sequence_size, max_logit_values);
  if (!probabilities.ok()) {
    return probabilities.status();
  }

  std::vector<TopKTelemetryEvent> events;
  events.reserve(batch_size * sequence_size);
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int sequence = 0; sequence < sequence_size; ++sequence) {
      TopKTelemetryEvent event;
      event.batch_index = batch;
      event.sequence_index = sequence;
      event.candidates.reserve(actual_k);

      float entropy = 0.0f;
      const int vocab_offset =
          (batch * sequence_size + sequence) * vocab_size;
      const int topk_offset = sequence * actual_k;
      for (int rank = 0; rank < actual_k; ++rank) {
        const int token_id = (*topk_ids)[batch][topk_offset + rank];
        const float probability =
            (*probabilities)[batch][topk_offset + rank];
        event.candidates.push_back(TopKTelemetryCandidate{
            .token_id = token_id,
            .logit = logits[vocab_offset + token_id],
            .probability = probability,
        });
        if (probability > 0.0f) {
          entropy -= probability * std::log(probability);
        }
      }

      event.top_k_entropy = entropy;
      if (event.candidates.size() >= 2) {
        event.top1_top2_margin =
            event.candidates[0].probability -
            event.candidates[1].probability;
      } else if (!event.candidates.empty()) {
        event.top1_top2_margin = event.candidates[0].probability;
      }
      events.push_back(std::move(event));
    }
  }
  return events;
}

}  // namespace litert::lm
