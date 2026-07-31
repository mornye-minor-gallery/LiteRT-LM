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

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;

TEST(TopKTelemetryTest, ReturnsSortedTopKWithTemperatureOneProbabilities) {
  const std::vector<float> logits = {1.0f, 4.0f, 2.0f, 3.0f};

  auto result = ComputeTopKTelemetry(logits, /*batch_size=*/1,
                                     /*sequence_size=*/1, /*top_k=*/3);

  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), 1);
  const auto& event = result->front();
  ASSERT_EQ(event.candidates.size(), 3);
  EXPECT_THAT(
      std::vector<int32_t>({event.candidates[0].token_id,
                            event.candidates[1].token_id,
                            event.candidates[2].token_id}),
      ElementsAre(1, 3, 2));
  EXPECT_THAT(
      std::vector<float>({event.candidates[0].logit,
                          event.candidates[1].logit,
                          event.candidates[2].logit}),
      ElementsAre(4.0f, 3.0f, 2.0f));
  EXPECT_THAT(event.candidates[0].probability,
              FloatNear(0.66524096f, 1e-6f));
  EXPECT_THAT(event.candidates[1].probability,
              FloatNear(0.24472847f, 1e-6f));
  EXPECT_THAT(event.candidates[2].probability,
              FloatNear(0.09003057f, 1e-6f));
  EXPECT_THAT(event.top_k_entropy, FloatNear(0.83239558f, 1e-6f));
  EXPECT_THAT(event.top1_top2_margin, FloatNear(0.42051249f, 1e-6f));
}

TEST(TopKTelemetryTest, ProducesOneEventPerBatchAndSequence) {
  const std::vector<float> logits = {
      3.0f, 2.0f, 1.0f,  // batch 0, sequence 0
      1.0f, 3.0f, 2.0f,  // batch 0, sequence 1
      2.0f, 1.0f, 3.0f,  // batch 1, sequence 0
      3.0f, 1.0f, 2.0f,  // batch 1, sequence 1
  };

  auto result = ComputeTopKTelemetry(logits, /*batch_size=*/2,
                                     /*sequence_size=*/2, /*top_k=*/2);

  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), 4);
  EXPECT_EQ((*result)[0].candidates[0].token_id, 0);
  EXPECT_EQ((*result)[1].candidates[0].token_id, 1);
  EXPECT_EQ((*result)[2].candidates[0].token_id, 2);
  EXPECT_EQ((*result)[3].candidates[0].token_id, 0);
}

TEST(TopKTelemetryTest, RejectsInvalidTopK) {
  const std::vector<float> logits = {1.0f, 2.0f};

  EXPECT_EQ(ComputeTopKTelemetry(logits, 1, 1, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ComputeTopKTelemetry(logits, 1, 1, 17).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace litert::lm
