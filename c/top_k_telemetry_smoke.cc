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

#include "c/engine.h"

#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace {

using EngineSettingsPtr =
    std::unique_ptr<LiteRtLmEngineSettings,
                    decltype(&litert_lm_engine_settings_delete)>;
using EnginePtr =
    std::unique_ptr<LiteRtLmEngine, decltype(&litert_lm_engine_delete)>;
using SessionConfigPtr =
    std::unique_ptr<LiteRtLmSessionConfig,
                    decltype(&litert_lm_session_config_delete)>;
using ConversationConfigPtr =
    std::unique_ptr<LiteRtLmConversationConfig,
                    decltype(&litert_lm_conversation_config_delete)>;
using ConversationPtr =
    std::unique_ptr<LiteRtLmConversation,
                    decltype(&litert_lm_conversation_delete)>;
using JsonResponsePtr =
    std::unique_ptr<LiteRtLmJsonResponse,
                    decltype(&litert_lm_json_response_delete)>;

struct TelemetryState {
  std::atomic<int> event_count = 0;
  std::atomic<int> invalid_event_count = 0;
  std::atomic<int> release_count = 0;
  std::atomic<double> entropy_sum = 0.0;
  std::atomic<double> margin_sum = 0.0;
};

void ObserveTopK(void* user_data, const LiteRtLmTopKTelemetryEvent* event) {
  auto* state = static_cast<TelemetryState*>(user_data);
  if (state == nullptr || event == nullptr) {
    return;
  }

  const int event_index = state->event_count.fetch_add(1);
  bool valid = event->abi_version == 1 && event->candidate_count > 0 &&
               event->candidate_count <=
                   LITERT_LM_TOP_K_TELEMETRY_MAX_CANDIDATES &&
               event->candidates != nullptr &&
               std::isfinite(event->top_k_entropy) &&
               std::isfinite(event->top1_top2_margin) &&
               event->metric_temperature == 1.0f;
  double probability_sum = 0.0;
  for (int index = 0; valid && index < event->candidate_count; ++index) {
    const auto& candidate = event->candidates[index];
    valid = candidate.token_id >= 0 && std::isfinite(candidate.logit) &&
            std::isfinite(candidate.probability) &&
            candidate.probability >= 0.0f;
    probability_sum += candidate.probability;
    if (index > 0) {
      valid = valid &&
              event->candidates[index - 1].logit >= candidate.logit &&
              event->candidates[index - 1].probability >=
                  candidate.probability;
    }
  }
  valid = valid && std::abs(probability_sum - 1.0) < 0.001;
  if (!valid) {
    state->invalid_event_count.fetch_add(1);
  }
  state->entropy_sum.fetch_add(event->top_k_entropy);
  state->margin_sum.fetch_add(event->top1_top2_margin);

  if (event_index < 8) {
    std::cout << "TELEMETRY"
              << " step=" << event_index
              << " sampled=" << event->sampled_token_id
              << " entropy=" << std::fixed << std::setprecision(6)
              << event->top_k_entropy
              << " margin=" << event->top1_top2_margin
              << " probability_sum=" << probability_sum
              << " top1_token=" << event->candidates[0].token_id
              << " top1_logit=" << event->candidates[0].logit
              << " top1_probability=" << event->candidates[0].probability
              << '\n';
  }
}

void ReleaseTopKState(void* user_data) {
  auto* state = static_cast<TelemetryState*>(user_data);
  if (state != nullptr) {
    state->release_count.fetch_add(1);
  }
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

int Run(const std::string& model_path, const std::string& prompt,
        const std::string& backend) {
  TelemetryState telemetry;
  std::string response;
  {
    EngineSettingsPtr settings(
        litert_lm_engine_settings_create(model_path.c_str(), backend.c_str(),
                                         /*vision_backend_str=*/nullptr,
                                         /*audio_backend_str=*/nullptr),
        &litert_lm_engine_settings_delete);
    if (!settings) {
      std::cerr << "Failed to create engine settings.\n";
      return 1;
    }
    litert_lm_engine_settings_set_max_num_tokens(settings.get(), 2048);

    EnginePtr engine(litert_lm_engine_create(settings.get()),
                     &litert_lm_engine_delete);
    if (!engine) {
      std::cerr << "Failed to create engine.\n";
      return 1;
    }

    SessionConfigPtr config(litert_lm_session_config_create(),
                            &litert_lm_session_config_delete);
    if (!config) {
      std::cerr << "Failed to create session config.\n";
      return 1;
    }
    litert_lm_session_config_set_max_output_tokens(config.get(), 32);
    if (litert_lm_session_config_set_top_k_telemetry(
            config.get(), 8, ObserveTopK, ReleaseTopKState, &telemetry) != 0) {
      std::cerr << "Failed to enable Top-K telemetry.\n";
      return 1;
    }

    ConversationConfigPtr conversation_config(
        litert_lm_conversation_config_create(),
        &litert_lm_conversation_config_delete);
    if (!conversation_config) {
      std::cerr << "Failed to create conversation config.\n";
      return 1;
    }
    litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                     config.get());
    ConversationPtr conversation(
        litert_lm_conversation_create(engine.get(), conversation_config.get()),
        &litert_lm_conversation_delete);
    if (!conversation) {
      std::cerr << "Failed to create conversation.\n";
      return 1;
    }

    const std::string message =
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"" +
        JsonEscape(prompt) + "\"}]}";
    JsonResponsePtr response_json(
        litert_lm_conversation_send_message(
            conversation.get(), message.c_str(), /*extra_context=*/nullptr,
            /*optional_args=*/nullptr),
        &litert_lm_json_response_delete);
    if (!response_json) {
      std::cerr << "Conversation generation failed.\n";
      return 1;
    }
    const char* text = litert_lm_json_response_get_string(response_json.get());
    if (text == nullptr) {
      std::cerr << "Generation returned a null response.\n";
      return 1;
    }
    response = text;
  }

  const int count = telemetry.event_count.load();
  const int invalid = telemetry.invalid_event_count.load();
  const int released = telemetry.release_count.load();
  std::cout << "RESPONSE " << response << '\n';
  std::cout << "SUMMARY"
            << " events=" << count
            << " invalid_events=" << invalid
            << " release_count=" << released
            << " average_top_k_entropy="
            << (count == 0 ? 0.0 : telemetry.entropy_sum.load() / count)
            << " average_top1_top2_margin="
            << (count == 0 ? 0.0 : telemetry.margin_sum.load() / count)
            << '\n';

  if (count == 0 || invalid != 0 || released != 1) {
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " MODEL_PATH [PROMPT] [cpu|gpu]\n";
    return 2;
  }
  const std::string prompt =
      argc >= 3 ? argv[2] : "안녕? 한 문장으로 답해줘.";
  const std::string backend = argc >= 4 ? argv[3] : "gpu";
  return Run(argv[1], prompt, backend);
}
