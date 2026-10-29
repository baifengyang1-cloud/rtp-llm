#include "rtp_llm/cpp/utils/EmbeddingStageLog.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace rtp_llm {
namespace {
thread_local EmbeddingStageLog* current_log = nullptr;
std::atomic<uint64_t>           next_batch_id{0};
}  // namespace

bool embeddingStageLoggingEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("RTP_LLM_EMBEDDING_STAGE_LOG");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

EmbeddingStageLog::EmbeddingStageLog(int64_t request_count) {
    if (!embeddingStageLoggingEnabled()) {
        return;
    }
    // Allocate diagnostics before publishing the scope. A diagnostic allocation
    // failure must not affect model execution or leave a dangling context.
    try {
        request_ids_.reserve(request_count);
        times_.reserve(32);
    } catch (...) {
        return;
    }
    requests_   = request_count;
    batch_id_   = next_batch_id.fetch_add(1, std::memory_order_relaxed) + 1;
    previous_   = current_log;
    current_log = this;
    active_     = true;
    start_      = Clock::now();
}

EmbeddingStageLog::~EmbeddingStageLog() noexcept {
    if (!active_) {
        return;
    }
    current_log           = previous_;
    const double total_us = std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    try {
        std::ostringstream record;
        record << std::fixed << std::setprecision(3)
               << "[embedding_stage] event=batch_done clock=cpu_steady units=us pid=" << getpid()
               << " batch_id=" << batch_id_ << " success=" << (success_ ? 1 : 0) << " requests=" << requests_
               << " request_ids=[";
        for (size_t i = 0; i < request_ids_.size(); ++i) {
            record << (i == 0 ? "" : ",") << request_ids_[i];
        }
        record << "] items=" << items_ << " tokens=" << tokens_ << " path=" << path_ << " bucket=" << bucket_
               << " padding_tokens=" << (bucket_ > 0 ? bucket_ - tokens_ : 0) << " batch_cpu_us=" << total_us;
        for (const auto& entry : times_) {
            record << " " << entry.first << "=" << entry.second;
        }
        RTP_LLM_LOG_INFO("%s", record.str().c_str());
    } catch (...) {
        // Diagnostics must not throw while unwinding a failed batch.
    }
}

EmbeddingStageLog* EmbeddingStageLog::current() {
    return current_log;
}

void EmbeddingStageLog::addRequest(int64_t request_id) noexcept {
    if (active_) {
        try {
            request_ids_.push_back(request_id);
        } catch (...) {}
    }
}

void EmbeddingStageLog::setShape(int64_t items, int64_t tokens) noexcept {
    items_  = items;
    tokens_ = tokens;
}

void EmbeddingStageLog::setExecution(const char* path, int64_t bucket) noexcept {
    path_   = path;
    bucket_ = bucket;
}

void EmbeddingStageLog::addTime(const char* name, double us) noexcept {
    try {
        for (auto& entry : times_) {
            if (std::strcmp(entry.first, name) == 0) {
                entry.second += us;
                return;
            }
        }
        times_.emplace_back(name, us);
    } catch (...) {}
}

EmbeddingStageSpan::EmbeddingStageSpan(const char* name): log_(EmbeddingStageLog::current()), name_(name) {
    if (log_ != nullptr) {
        start_ = EmbeddingStageLog::Clock::now();
    }
}

void EmbeddingStageSpan::finish() noexcept {
    if (log_ != nullptr) {
        log_->addTime(name_,
                      std::chrono::duration<double, std::micro>(EmbeddingStageLog::Clock::now() - start_).count());
        log_ = nullptr;
    }
}

}  // namespace rtp_llm
