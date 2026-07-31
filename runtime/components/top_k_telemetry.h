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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_TELEMETRY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_TELEMETRY_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

inline constexpr uint32_t kTopKTelemetryAbiVersion = 1;
inline constexpr int kMaxTopKTelemetryCandidates = 16;
inline constexpr float kTopKTelemetryMetricTemperature = 1.0f;

struct TopKTelemetryCandidate {
  int32_t token_id = -1;
  float logit = 0.0f;
  float probability = 0.0f;
};

// A compact diagnostic snapshot of the top-k distribution for one decoded
// position. Probabilities and entropy are normalized only across the reported
// top-k candidates at a fixed metric temperature of 1.0. They are not
// full-vocabulary probabilities or entropy.
struct TopKTelemetryEvent {
  int32_t batch_index = 0;
  int32_t sequence_index = 0;
  int32_t sampled_token_id = -1;
  float top_k_entropy = 0.0f;
  float top1_top2_margin = 0.0f;
  std::vector<TopKTelemetryCandidate> candidates;
};

class TopKTelemetryObserver {
 public:
  virtual ~TopKTelemetryObserver() = default;
  virtual void OnTopKTelemetry(const TopKTelemetryEvent& event) = 0;
};

struct TopKTelemetryConfig {
  int top_k = 0;
  std::shared_ptr<TopKTelemetryObserver> observer;

  bool enabled() const { return top_k > 0 && observer != nullptr; }
};

// Computes top-k diagnostic events from flattened logits with shape
// [batch_size, sequence_size, vocab_size]. This scans the full vocabulary
// locally, but only the compact top-k result is returned.
absl::StatusOr<std::vector<TopKTelemetryEvent>> ComputeTopKTelemetry(
    absl::Span<const float> logits, int batch_size, int sequence_size,
    int top_k);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_TELEMETRY_H_
