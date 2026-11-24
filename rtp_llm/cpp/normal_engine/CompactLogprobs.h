#pragma once

#include <algorithm>
#include <tuple>
#include <utility>
#include <torch/all.h>

namespace rtp_llm {

// Keep the sampler's probability semantics; reduce on device before any D2H.
inline std::pair<torch::Tensor, torch::Tensor>
gatherCompactLogprobs(const torch::Tensor& probabilities, const torch::Tensor& selected_tokens, int64_t top_k) {
    TORCH_CHECK(probabilities.dim() == 2 && probabilities.size(1) > 0, "expected [batch, vocab] probabilities");
    TORCH_CHECK(selected_tokens.numel() == probabilities.size(0), "expected one selected token per row");
    TORCH_CHECK(top_k >= 0 && top_k <= 1024, "invalid logprobs top-k");
    auto selected = selected_tokens.to(probabilities.device(), torch::kLong).reshape({-1, 1});
    auto top      = probabilities.topk(std::min(top_k, probabilities.size(1)), -1);
    auto ids      = torch::cat({selected, std::get<1>(top)}, -1).to(torch::kInt32);
    auto values   = torch::cat({probabilities.gather(1, selected), std::get<0>(top)}, -1).log();
    return {ids, values};
}

}  // namespace rtp_llm
