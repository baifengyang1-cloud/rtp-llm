#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/EmbeddingStageLog.h"
#include <memory>
#include <utility>

using namespace std;

namespace rtp_llm {

EmbeddingStream::EmbeddingStream(const shared_ptr<rtp_llm::EmbeddingInput>& query,
                                 EmbeddingProfileConfig                     profile_config):
    embedding_input_(query), profile_config_(std::move(profile_config)) {
    if (!query.get()) {
        return;
    }
    begin_time_       = autil::TimeUtility::currentTimeInMilliSeconds();
    embedding_output_ = make_shared<EmbeddingOutput>();
    stream_state_     = StreamState::WAITING;
    begin_time_us_    = autil::TimeUtility::currentTimeInMicroSeconds();
    if (embeddingStageLoggingEnabled()) {
        RTP_LLM_LOG_INFO("[embedding_stage] event=request_created request_id=%ld items=%ld tokens=%ld",
                         streamId(),
                         batchSize(),
                         inputLength());
    }
}

int64_t EmbeddingStream::streamId() const {
    return embedding_input_->request_id;
}

const std::optional<MultimodalFeature>& EmbeddingStream::multimodalFeature() const {
    return embedding_input_->multimodal_features;
}

int64_t EmbeddingStream::batchSize() const {
    return embedding_input_->input_lengths.size(0);
}

void EmbeddingStream::setMetricReporter(const kmonitor::MetricsReporterPtr& metric_reporter) {
    metrics_reporter_ = metric_reporter;
}

std::shared_ptr<EmbeddingInput> EmbeddingStream::embeddingInput() const {
    return embedding_input_;
}

std::shared_ptr<EmbeddingOutput> EmbeddingStream::embeddingOutput() const {
    return embedding_output_;
}

int64_t EmbeddingStream::inputLength() const {
    return embedding_input_->total_length;
}

void EmbeddingStream::waitFinish() {
    unique_lock<mutex> lock(lock_);
    while (stream_state_ != StreamState::FINISHED) {
        cond_.wait_for(lock, std::chrono::milliseconds(5));
    }
    if (!embedding_output_->error_info.ok()) {
        throw std::runtime_error("run stream failed: " + embedding_output_->error_info.ToString());
    }
}

void EmbeddingStream::reportMetrics() {
    const bool stage_logging = embeddingStageLoggingEnabled();
    if (metrics_reporter_ || stage_logging) {
        RtpEmbeddingStreamMetricsCollector collector;
        collector.input_token_length = inputLength();
        collector.wait_latency_us    = wait_time_us_;
        collector.total_latency_us   = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpEmbeddingStreamMetrics, RtpEmbeddingStreamMetricsCollector>(nullptr,
                                                                                                     &collector);
        }
        if (stage_logging) {
            RTP_LLM_LOG_INFO("[embedding_stage] event=request_done clock=wall units=us request_id=%ld "
                             "items=%ld tokens=%ld wait_us=%ld total_us=%ld success=%d",
                             streamId(),
                             batchSize(),
                             inputLength(),
                             collector.wait_latency_us,
                             collector.total_latency_us,
                             embedding_output_->error_info.ok() ? 1 : 0);
        }
    }
}

void EmbeddingStream::setError(const std::string& error_info) {
    embedding_output_->setError(ErrorCode::UNKNOWN_ERROR, error_info);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
}

void EmbeddingStream::setStart() {
    wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    stream_state_ = StreamState::RUNNING;
}

void EmbeddingStream::updateTensorOutput(torch::Tensor t) {
    lock_guard<mutex> lock(lock_);
    embedding_output_->setTensorOutput(t);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

void EmbeddingStream::updateMapOutput(std::vector<std::map<std::string, torch::Tensor>>& map) {
    lock_guard<mutex> lock(lock_);
    embedding_output_->setMapOutput(map);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

}  // namespace rtp_llm
