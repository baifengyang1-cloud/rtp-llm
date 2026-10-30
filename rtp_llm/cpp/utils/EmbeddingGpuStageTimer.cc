#include "rtp_llm/cpp/utils/EmbeddingGpuStageTimer.h"
#include "rtp_llm/cpp/utils/EmbeddingStageLog.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>
#include <unistd.h>

#if USING_CUDA || USING_ROCM
#include <c10/core/Event.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#endif

namespace rtp_llm {
namespace {
[[maybe_unused]] uint64_t timingEveryN() {
    static const uint64_t value = []() -> uint64_t {
        const char* text = std::getenv("RTP_LLM_EMBEDDING_GPU_TIMING_EVERY_N");
        if (text == nullptr || text[0] == '\0') {
            return 0;
        }
        char* end = nullptr;
        errno     = 0;
        auto n    = std::strtoull(text, &end, 10);
        if (errno != 0 || text[0] == '-' || end == text || *end != '\0') {
            RTP_LLM_LOG_WARNING("invalid RTP_LLM_EMBEDDING_GPU_TIMING_EVERY_N; GPU timing disabled");
            return 0;
        }
        if (n != 0 && !embeddingStageLoggingEnabled()) {
            RTP_LLM_LOG_WARNING("embedding GPU timing requires RTP_LLM_EMBEDDING_STAGE_LOG=1");
            return 0;
        }
        return n;
    }();
    return value;
}

#if USING_CUDA || USING_ROCM
#if USING_ROCM
constexpr auto timing_device_type = c10::DeviceType::HIP;
#else
constexpr auto timing_device_type = c10::DeviceType::CUDA;
#endif

class EventBackend {
public:
    using Stream = c10::Stream;
    explicit EventBackend(uint64_t every_n): every_n_(every_n), warm_stream_(currentStream()) {
        events_.reserve(16);
        for (int i = 0; i < 16; ++i) {
            // BACKEND_DEFAULT enables elapsed time on CUDA and HIP. The
            // PYTORCH_DEFAULT lifecycle events used by Graph disable timing.
            events_.emplace_back(timing_device_type, c10::EventFlag::BACKEND_DEFAULT);
            events_.back().record(warm_stream_);
        }
    }
    Stream currentStream() {
        c10::impl::VirtualGuardImpl impl(timing_device_type);
        return impl.getStream(impl.getDevice());
    }
    bool ready(const Stream& stream) {
        TORCH_CHECK(stream.device() == warm_stream_.device(), "embedding timing device changed");
        return warmed_ || (warmed_ = events_.back().query());
    }
    void record(size_t slot, int point, const Stream& stream) {
        events_[slot * 4 + point].record(stream);
    }
    bool query(size_t slot, int point) {
        return events_[slot * 4 + point].query();
    }
    double elapsed(size_t slot, int first, int last) {
        return events_[slot * 4 + first].elapsedTime(events_[slot * 4 + last]);
    }
    void skipped(const EmbeddingGpuTimingSample& sample, uint64_t count, const char* reason) {
        RTP_LLM_LOG_INFO("[embedding_stage] event=gpu_timing_skipped pid=%d batch_id=%llu valid=0 reason=%s "
                         "every_n=%llu skipped=%llu",
                         getpid(),
                         (unsigned long long)sample.batch_id,
                         reason,
                         (unsigned long long)every_n_,
                         (unsigned long long)count);
    }
    void report(const EmbeddingGpuTimingSample& sample,
                const Stream&                   stream,
                const char*                     reason,
                const std::array<double, 3>&    ms,
                uint64_t                        skipped_count) {
        for (double value : ms) {
            if (!std::isfinite(value) || value < 0) {
                reason = "invalid_elapsed_time";
            }
        }
        if (reason != nullptr) {
            RTP_LLM_LOG_INFO("[embedding_stage] event=gpu_timing pid=%d batch_id=%llu valid=0 reason=%s "
                             "path=%s every_n=%llu skipped=%llu",
                             getpid(),
                             (unsigned long long)sample.batch_id,
                             reason,
                             sample.path,
                             (unsigned long long)every_n_,
                             (unsigned long long)skipped_count);
            return;
        }
        RTP_LLM_LOG_INFO("[embedding_stage] event=gpu_timing clock=gpu_stream units=ms pid=%d batch_id=%llu "
                         "valid=1 device=%d stream=%lld requests=%lld items=%lld tokens=%lld path=%s bucket=%lld "
                         "prepare_stream_ms=%.6f backbone_stream_ms=%.6f post_stream_ms=%.6f "
                         "every_n=%llu skipped=%llu",
                         getpid(),
                         (unsigned long long)sample.batch_id,
                         (int)stream.device_index(),
                         (long long)stream.id(),
                         (long long)sample.requests,
                         (long long)sample.items,
                         (long long)sample.tokens,
                         sample.path,
                         (long long)sample.bucket,
                         ms[0],
                         ms[1],
                         ms[2],
                         (unsigned long long)every_n_,
                         (unsigned long long)skipped_count);
    }

private:
    uint64_t                every_n_;
    Stream                  warm_stream_;
    bool                    warmed_ = false;
    std::vector<c10::Event> events_;
};

struct Runtime {
    explicit Runtime(uint64_t every_n): backend(every_n), pool(backend) {}
    EventBackend                         backend;
    EmbeddingGpuTimingPool<EventBackend> pool;
    bool                                 active = false;
};
thread_local std::unique_ptr<Runtime> runtime;
thread_local bool                     disabled = false;

void disableTiming(const char* error) noexcept {
    disabled = true;
    if (runtime) {
        runtime->active = false;
    }
    RTP_LLM_LOG_WARNING(
        "[embedding_stage] event=gpu_timing_disabled pid=%d valid=0 reason=event_api_error error=%s", getpid(), error);
}
#endif
}  // namespace

EmbeddingGpuStageScope::EmbeddingGpuStageScope(const EmbeddingGpuTimingSample& sample) noexcept {
#if USING_CUDA || USING_ROCM
    const auto every_n = timingEveryN();
    if (disabled || every_n == 0 || sample.batch_id == 0) {
        return;
    }
    try {
        // Poll also on nonsampled batches, so pending events are reclaimed.
        if (runtime && !runtime->active) {
            runtime->pool.poll();
        }
        if ((sample.batch_id - 1) % every_n != 0 || (runtime && runtime->active)) {
            return;
        }
        if (!runtime) {
            runtime = std::make_unique<Runtime>(every_n);
        }
        active_         = runtime->pool.begin(sample);
        runtime->active = active_;
    } catch (const std::exception& error) {
        disableTiming(error.what());
    } catch (...) {
        disableTiming("unknown error");
    }
#else
    (void)sample;
#endif
}

EmbeddingGpuStageScope::~EmbeddingGpuStageScope() noexcept {
    finish(false);
}

void EmbeddingGpuStageScope::finish(bool success) noexcept {
#if USING_CUDA || USING_ROCM
    if (!active_ || disabled) {
        return;
    }
    active_         = false;
    runtime->active = false;
    try {
        runtime->pool.finish(success);
    } catch (const std::exception& error) {
        disableTiming(error.what());
    } catch (...) {
        disableTiming("unknown error");
    }
#else
    (void)success;
#endif
}

void markEmbeddingGpuStage(EmbeddingGpuPoint point) noexcept {
#if USING_CUDA || USING_ROCM
    if (!disabled && runtime && runtime->active) {
        try {
            runtime->pool.mark(static_cast<int>(point));
        } catch (const std::exception& error) {
            disableTiming(error.what());
        } catch (...) {
            disableTiming("unknown error");
        }
    }
#else
    (void)point;
#endif
}

void setEmbeddingGpuExecution(const char* path, int64_t bucket) noexcept {
#if USING_CUDA || USING_ROCM
    if (!disabled && runtime && runtime->active) {
        runtime->pool.setExecution(path, bucket);
    }
#else
    (void)path;
    (void)bucket;
#endif
}

}  // namespace rtp_llm
