#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <optional>

namespace rtp_llm {

struct EmbeddingGpuTimingSample {
    uint64_t    batch_id;
    int64_t     requests;
    int64_t     items;
    int64_t     tokens;
    const char* path   = "unknown";
    int64_t     bucket = 0;
};

// Backend supplies timing events and reporting. Kept independent of Torch so
// delayed completion, abort and stream changes can be tested without a GPU.
template<class Backend>
class EmbeddingGpuTimingPool {
public:
    using Stream = typename Backend::Stream;
    explicit EmbeddingGpuTimingPool(Backend& backend): backend_(backend) {}

    bool begin(const EmbeddingGpuTimingSample& sample) {
        poll();
        const auto stream = backend_.currentStream();
        if (active_ >= 0 || !backend_.ready(stream)) {
            backend_.skipped(sample, ++skipped_, "warming_or_nested_scope");
            return false;
        }
        for (size_t i = 0; i < slots_.size(); ++i) {
            auto& slot = slots_[i];
            if (slot.pending) {
                continue;
            }
            slot.sample = sample;
            slot.stream = stream;
            slot.last   = -1;
            slot.reason = nullptr;
            active_     = static_cast<int>(i);
            mark(0);
            return true;
        }
        backend_.skipped(sample, ++skipped_, "pool_full");
        return false;
    }

    void setExecution(const char* path, int64_t bucket) {
        if (active_ >= 0) {
            slots_[active_].sample.path   = path;
            slots_[active_].sample.bucket = bucket;
        }
    }

    void mark(int point) {
        if (active_ < 0) {
            return;
        }
        auto& slot = slots_[active_];
        if (slot.reason != nullptr) {
            return;
        }
        if (point != slot.last + 1 || point > 3) {
            slot.reason = "missing_or_repeated_boundary";
            return;
        }
        if (backend_.currentStream() != *slot.stream) {
            slot.reason = "device_or_stream_changed";
            return;
        }
        backend_.record(active_, point, *slot.stream);
        slot.last = point;
    }

    void finish(bool success) {
        if (active_ >= 0) {
            auto& slot = slots_[active_];
            if (!success) {
                slot.reason = "batch_failed";
            } else if (slot.reason == nullptr && slot.last != 3) {
                slot.reason = "missing_scores_ready_boundary";
            }
            slot.pending = true;
            active_      = -1;
        }
        poll();
    }

    void poll() {
        for (size_t i = 0; i < slots_.size(); ++i) {
            auto& slot = slots_[i];
            if (!slot.pending || (slot.last >= 0 && !backend_.query(i, slot.last))) {
                continue;
            }
            std::array<double, 3> elapsed{};
            if (slot.reason == nullptr) {
                for (int p = 0; p < 3; ++p) {
                    elapsed[p] = backend_.elapsed(i, p, p + 1);
                }
            }
            backend_.report(slot.sample, *slot.stream, slot.reason, elapsed, skipped_);
            slot.pending = false;
        }
    }

private:
    struct Slot {
        EmbeddingGpuTimingSample sample{};
        std::optional<Stream>    stream;
        int                      last    = -1;
        const char*              reason  = nullptr;
        bool                     pending = false;
    };
    Backend&            backend_;
    std::array<Slot, 4> slots_;
    int                 active_  = -1;
    uint64_t            skipped_ = 0;
};

}  // namespace rtp_llm
