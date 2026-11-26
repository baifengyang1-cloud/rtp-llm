#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/normal_engine/CompactLogprobs.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

TEST(NormalOutputDispatcherTest, CompactLogprobsRetainSelectedTokensOutsideTopK) {
    auto probs    = torch::tensor({{0.1f, 0.6f, 0.3f, 0.0f}, {0.0f, 0.2f, 0.3f, 0.5f}});
    auto selected = torch::tensor({{0}, {1}}, torch::kInt32);
    auto result   = gatherCompactLogprobs(probs, selected, 2);
    EXPECT_TRUE(torch::equal(result.first, torch::tensor({{0, 1, 2}, {1, 3, 2}}, torch::kInt32)));
    EXPECT_TRUE(torch::allclose(result.second.exp(), torch::tensor({{0.1f, 0.6f, 0.3f}, {0.2f, 0.5f, 0.3f}})));
}

TEST(NormalOutputDispatcherTest, CompactLogprobsHandleZeroMassAndSelectedOnly) {
    auto probs    = torch::tensor({{0.0f, 1.0f, 0.0f}});
    auto selected = torch::tensor({{1}}, torch::kInt32);
    auto result   = gatherCompactLogprobs(probs, selected, 8);
    EXPECT_EQ(result.first.size(1), 4);
    EXPECT_FLOAT_EQ(result.second[0][0].item<float>(), 0.0f);
    EXPECT_EQ(torch::isneginf(result.second).sum().item<int>(), 2);
    auto selected_only = gatherCompactLogprobs(probs, selected, 0);
    EXPECT_EQ(selected_only.first.size(1), 1);
    EXPECT_FLOAT_EQ(selected_only.second[0][0].item<float>(), 0.0f);
}

TEST(NormalOutputDispatcherTest, ProcessorErrorsUseSamplerInputCoordinatesAfterBeamExpansion) {
    SamplerOutput sampler_output;
    sampler_output.processor_errors = {
        std::nullopt,
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second stream processor failed"),
    };

    // The first stream may expand from one input row to multiple output beams.
    // Its input slice must not consume the second stream's processor error.
    auto first_stream_error = collectStreamSamplerError(
        sampler_output, /*success_cpu=*/torch::Tensor(), /*batch_idx_in=*/0, /*cur_batch_size=*/1);
    EXPECT_FALSE(first_stream_error.has_value());

    auto second_stream_error = collectStreamSamplerError(
        sampler_output, /*success_cpu=*/torch::Tensor(), /*batch_idx_in=*/1, /*cur_batch_size=*/1);
    ASSERT_TRUE(second_stream_error.has_value());
    EXPECT_EQ(second_stream_error->code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(second_stream_error->ToString().find("second stream processor failed"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
