#include "runtime/executor/pending_prefill_embeddings.h"
#include <vector>
#include <gtest/gtest.h>

namespace litert::lm::internal {
namespace {
struct Lookup {
  int calls = 0;
  absl::Status result = absl::OkStatus();
  bool empty = false;
  absl::Status LookupPrefill(int id, std::vector<float>& out) {
    ++calls;
    if (!result.ok()) { out = {-99}; return result; }
    if (!empty) out = {float(id), 0.25f};
    return absl::OkStatus();
  }
};

TEST(PendingPrefillEmbeddings, RehydratesRewoundIdOnlyToken) {
  TokenData token(17); Lookup main, layer;
  ASSERT_TRUE(EnsurePendingPrefillEmbeddings(token, &main, &layer, true).ok());
  EXPECT_EQ(token.embedding(), (std::vector<float>{17, 0.25f}));
  EXPECT_EQ(token.per_layer_embedding(), (std::vector<float>{17, 0.25f}));
  EXPECT_EQ(main.calls, 1); EXPECT_EQ(layer.calls, 1);
  ASSERT_TRUE(EnsurePendingPrefillEmbeddings(token, &main, &layer, true).ok());
  EXPECT_EQ(main.calls, 1); EXPECT_EQ(layer.calls, 1);
}
TEST(PendingPrefillEmbeddings, PreservesPreparedOrMultimodalValues) {
  TokenData token(-1, {9, 8}, {7, 6});
  ASSERT_TRUE(EnsurePendingPrefillEmbeddings<Lookup>(token, nullptr, nullptr, true).ok());
  EXPECT_EQ(token.embedding(), (std::vector<float>{9, 8}));
  EXPECT_EQ(token.per_layer_embedding(), (std::vector<float>{7, 6}));
}
TEST(PendingPrefillEmbeddings, FillsOnlyMissingLayerEmbedding) {
  TokenData token(17, {9, 8}, {}); Lookup main, layer;
  ASSERT_TRUE(EnsurePendingPrefillEmbeddings(token, &main, &layer, true).ok());
  EXPECT_EQ(token.embedding(), (std::vector<float>{9, 8}));
  EXPECT_FALSE(token.per_layer_embedding().empty());
  EXPECT_EQ(main.calls, 0); EXPECT_EQ(layer.calls, 1);
}
TEST(PendingPrefillEmbeddings, DoesNotRequireUnusedLayerLookup) {
  TokenData token(17); Lookup main;
  ASSERT_TRUE(EnsurePendingPrefillEmbeddings<Lookup>(token, &main, nullptr, false).ok());
  EXPECT_FALSE(token.embedding().empty());
  EXPECT_TRUE(token.per_layer_embedding().empty());
}
TEST(PendingPrefillEmbeddings, RejectsMissingMainLookup) {
  TokenData token(17);
  EXPECT_EQ(EnsurePendingPrefillEmbeddings<Lookup>(token, nullptr, nullptr, false).code(), absl::StatusCode::kFailedPrecondition);
}
TEST(PendingPrefillEmbeddings, RejectsMissingRequiredLayerLookup) {
  TokenData token(17, {9, 8}, {}); Lookup main;
  EXPECT_EQ(EnsurePendingPrefillEmbeddings<Lookup>(token, &main, nullptr, true).code(), absl::StatusCode::kFailedPrecondition);
}
TEST(PendingPrefillEmbeddings, PropagatesMainLookupFailure) {
  TokenData token(17); Lookup main, layer;
  main.result = absl::UnavailableError("lookup failed");
  EXPECT_EQ(EnsurePendingPrefillEmbeddings(token, &main, &layer, true), main.result);
  EXPECT_TRUE(token.embedding().empty());
  EXPECT_EQ(layer.calls, 0);
}
TEST(PendingPrefillEmbeddings, PropagatesLayerFailureAndCanRetry) {
  TokenData token(17); Lookup main, layer;
  layer.result = absl::UnavailableError("layer lookup failed");
  EXPECT_EQ(EnsurePendingPrefillEmbeddings(token, &main, &layer, true), layer.result);
  EXPECT_TRUE(token.per_layer_embedding().empty());
  layer.result = absl::OkStatus();
  ASSERT_TRUE(EnsurePendingPrefillEmbeddings(token, &main, &layer, true).ok());
  EXPECT_EQ(main.calls, 1); EXPECT_EQ(layer.calls, 2);
}
TEST(PendingPrefillEmbeddings, RejectsEmptySuccessfulLookup) {
  TokenData token(17); Lookup main; main.empty = true;
  EXPECT_EQ(EnsurePendingPrefillEmbeddings<Lookup>(token, &main, nullptr, false).code(), absl::StatusCode::kFailedPrecondition);
}
TEST(PendingPrefillEmbeddings, RejectsEmptySuccessfulLayerLookup) {
  TokenData token(17); Lookup main, layer; layer.empty = true;
  EXPECT_EQ(EnsurePendingPrefillEmbeddings(token, &main, &layer, true).code(), absl::StatusCode::kFailedPrecondition);
}
}
}
