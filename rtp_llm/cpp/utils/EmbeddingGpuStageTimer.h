#pragma once

#include "rtp_llm/cpp/utils/EmbeddingGpuTimingPool.h"

namespace rtp_llm {

enum class EmbeddingGpuPoint {
    Begin,
    BackboneBegin,
    BackboneEnd,
    ScoresReady
};

// Off by default. Requires the CPU stage log for batch identity. Events are
// allocated once per execution thread; no synchronization is added by timing.
class EmbeddingGpuStageScope {
public:
    explicit EmbeddingGpuStageScope(const EmbeddingGpuTimingSample& sample) noexcept;
    ~EmbeddingGpuStageScope() noexcept;
    EmbeddingGpuStageScope(const EmbeddingGpuStageScope&)            = delete;
    EmbeddingGpuStageScope& operator=(const EmbeddingGpuStageScope&) = delete;
    void                    finish(bool success) noexcept;

private:
    bool active_ = false;
};

// These functions share one implementation/context in the engine library.
// No active sampled embedding scope => no event or stream lookup.
void markEmbeddingGpuStage(EmbeddingGpuPoint point) noexcept;
void setEmbeddingGpuExecution(const char* path, int64_t bucket) noexcept;

}  // namespace rtp_llm
