#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHash.h"

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#endif
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/multimodal_processor/FeatureHashKernel.h"
#endif

namespace rtp_llm {

torch::Tensor getMultimodalFeatureHash(const torch::Tensor& embedding) {
    TORCH_CHECK(embedding.defined() && embedding.dim() >= 1 && embedding.size(0) > 0,
                "multimodal feature tensor is empty");
    const int64_t rows      = embedding.size(0);
    const int64_t row_bytes = embedding.numel() / rows * embedding.element_size();
    TORCH_CHECK(row_bytes > 0, "multimodal feature row is empty");
#if USING_CUDA || USING_ROCM
    if (embedding.is_cuda()) {
#if USING_ROCM
        const c10::hip::HIPGuardMasqueradingAsCUDA guard(embedding.device());
        const auto stream = at::hip::getCurrentHIPStream(embedding.get_device()).stream();
#else
        const c10::cuda::CUDAGuard guard(embedding.device());
        const auto                 stream = c10::cuda::getCurrentCUDAStream(embedding.get_device()).stream();
#endif
        auto emb        = embedding.contiguous();
        auto gpu_hashes = torch::empty({rows}, emb.options().dtype(torch::kInt32));
        auto error      = invokeFeatureHash(emb.data_ptr(), rows, row_bytes, gpu_hashes.data_ptr<int32_t>(), stream);
#if USING_ROCM
        TORCH_CHECK(error == hipSuccess, "failed to hash multimodal features on GPU: ", hipGetErrorString(error));
#else
        TORCH_CHECK(error == cudaSuccess, "failed to hash multimodal features on GPU: ", cudaGetErrorString(error));
#endif
        // Blocking copy on the producer's current stream: only row token ids
        // leave the GPU, and callers can publish them immediately on return.
        return gpu_hashes.to(torch::kCPU);
    }
#endif
    auto        hashes = torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto        emb    = embedding.to(torch::kCPU).contiguous();
    const auto* bytes  = static_cast<const uint8_t*>(emb.data_ptr());
    auto*       output = hashes.data_ptr<int32_t>();
    for (int64_t row = 0; row < rows; ++row) {
        output[row] = featureHashToTokenId(hashFeatureRowCpu(bytes + row * row_bytes, row_bytes));
    }
    return hashes;
}

}  // namespace rtp_llm
