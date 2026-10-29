#pragma once

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace rtp_llm {

bool embeddingStageLoggingEnabled();

// CPU diagnostics for the synchronous embedding execution thread. Timings can
// include waits inside PyTorch; none of these fields measures GPU kernel time.
class EmbeddingStageLog {
public:
    using Clock = std::chrono::steady_clock;

    explicit EmbeddingStageLog(int64_t request_count);
    ~EmbeddingStageLog() noexcept;
    EmbeddingStageLog(const EmbeddingStageLog&)            = delete;
    EmbeddingStageLog& operator=(const EmbeddingStageLog&) = delete;

    static EmbeddingStageLog* current();
    uint64_t                  batchId() const {
        return batch_id_;
    }
    void addRequest(int64_t request_id) noexcept;
    void setShape(int64_t items, int64_t tokens) noexcept;
    void setExecution(const char* path, int64_t bucket = 0) noexcept;
    void finish(bool success) noexcept {
        success_ = success;
    }
    void addTime(const char* name, double us) noexcept;

private:
    EmbeddingStageLog*                          previous_ = nullptr;
    Clock::time_point                           start_;
    uint64_t                                    batch_id_ = 0;
    int64_t                                     requests_ = 0;
    int64_t                                     items_    = -1;
    int64_t                                     tokens_   = -1;
    int64_t                                     bucket_   = 0;
    const char*                                 path_     = "unknown";
    bool                                        active_   = false;
    bool                                        success_  = false;
    std::vector<int64_t>                        request_ids_;
    std::vector<std::pair<const char*, double>> times_;
};

class EmbeddingStageSpan {
public:
    explicit EmbeddingStageSpan(const char* name);
    ~EmbeddingStageSpan() noexcept {
        finish();
    }
    EmbeddingStageSpan(const EmbeddingStageSpan&)            = delete;
    EmbeddingStageSpan& operator=(const EmbeddingStageSpan&) = delete;

    void finish() noexcept;

private:
    EmbeddingStageLog*                   log_;
    const char*                          name_;
    EmbeddingStageLog::Clock::time_point start_;
};

}  // namespace rtp_llm
