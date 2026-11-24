#pragma once

#include <cstdint>
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

#if USING_ROCM
using FeatureHashError  = hipError_t;
using FeatureHashStream = hipStream_t;
#else
using FeatureHashError  = cudaError_t;
using FeatureHashStream = cudaStream_t;
#endif

FeatureHashError
invokeFeatureHash(const void* input, int64_t num_rows, int64_t row_bytes, int32_t* output, FeatureHashStream stream);

}  // namespace rtp_llm
