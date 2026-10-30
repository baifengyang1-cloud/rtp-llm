#include "rtp_llm/cpp/utils/EmbeddingGpuTimingPool.h"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace rtp_llm {
namespace {
void check(bool condition) {
    if (!condition) {
        std::abort();
    }
}
struct FakeBackend {
    using Stream = int;
    struct Report {
        EmbeddingGpuTimingSample sample;
        const char*              reason;
        std::array<double, 3>    elapsed;
    };
    int                                stream = 0;
    bool                               warm   = true;
    std::array<std::array<bool, 4>, 4> complete{};
    std::vector<Report>                reports;
    int                                records     = 0;
    uint64_t                           skips       = 0;
    const char*                        skip_reason = nullptr;

    Stream currentStream() {
        return stream;
    }
    bool ready(Stream) {
        return warm;
    }
    void record(size_t slot, int point, Stream) {
        complete[slot][point] = false;
        ++records;
    }
    bool query(size_t slot, int point) {
        return complete[slot][point];
    }
    double elapsed(size_t slot, int first, int last) {
        check(complete[slot][first] && complete[slot][last]);
        return 0.25 * (first + last);
    }
    void skipped(const EmbeddingGpuTimingSample&, uint64_t count, const char* reason) {
        skips       = count;
        skip_reason = reason;
    }
    void report(
        const EmbeddingGpuTimingSample& sample, Stream, const char* reason, const std::array<double, 3>& ms, uint64_t) {
        reports.push_back({sample, reason, ms});
    }
    void completeAll() {
        for (auto& slot : complete) {
            slot.fill(true);
        }
    }
};

void testDelayedCompletionAndCapacity() {
    FakeBackend                         backend;
    EmbeddingGpuTimingPool<FakeBackend> pool(backend);
    check(backend.records == 0);
    for (uint64_t id = 1; id <= 4; ++id) {
        check(pool.begin({id, 3, 160, 18000}));
        pool.setExecution(id == 1 ? "eager_fallback" : "graph", id == 1 ? 0 : 18432);
        pool.mark(1);
        pool.mark(2);
        pool.mark(3);
        pool.finish(true);
    }
    check(backend.records == 16 && backend.reports.empty());
    check(!pool.begin({5, 1, 32, 4096}));
    check(backend.records == 16 && backend.skips == 1);
    check(std::strcmp(backend.skip_reason, "pool_full") == 0);
    // Completing an earlier event cannot release a slot while E3 is pending.
    backend.complete[0][0] = backend.complete[0][1] = backend.complete[0][2] = true;
    pool.poll();
    check(backend.reports.empty());
    backend.completeAll();
    pool.poll();
    check(backend.reports.size() == 4);
    check(backend.reports[0].sample.batch_id == 1);
    check(std::strcmp(backend.reports[0].sample.path, "eager_fallback") == 0);
    check(backend.reports[0].sample.bucket == 0);
    check(backend.reports[1].sample.bucket == 18432);
    check(backend.reports[1].sample.items == 160);
    check(backend.reports[1].reason == nullptr);
    check((backend.reports[1].elapsed == std::array<double, 3>{0.25, 0.75, 1.25}));
    // Events are reusable only after last-record completion, even across streams.
    backend.stream = 3;
    check(pool.begin({6, 1, 198, 19000}));
    pool.mark(1);
    pool.mark(2);
    pool.mark(3);
    pool.finish(true);
    check(backend.reports.size() == 4);
    backend.completeAll();
    pool.poll();
    check(backend.reports.size() == 5);
}

void testInvalidSamplesAndWarmup() {
    FakeBackend                         backend;
    EmbeddingGpuTimingPool<FakeBackend> pool(backend);
    backend.warm = false;
    check(!pool.begin({1, 1, 1, 1}));
    check(backend.records == 0 && backend.skips == 1);
    backend.warm = true;
    for (int scenario = 0; scenario < 5; ++scenario) {
        check(pool.begin({static_cast<uint64_t>(scenario + 2), 1, 1, 1}));
        if (scenario == 0) {
            pool.mark(1);  // Exception before the backbone finishes.
        } else if (scenario == 1) {
            pool.mark(2);  // E1 was omitted.
        } else if (scenario == 2) {
            backend.stream = 1;
            pool.mark(1);
            backend.stream = 0;
        } else if (scenario == 3) {
            pool.mark(1);
            pool.mark(2);  // Handler did not provide E3.
        } else {
            pool.mark(1);
            pool.mark(1);  // Repeated boundary must invalidate the sample.
        }
        pool.finish(scenario != 0);
        check(backend.reports.size() == static_cast<size_t>(scenario));
        backend.completeAll();
        pool.poll();
        check(backend.reports.size() == static_cast<size_t>(scenario + 1));
        const char* expected[] = {"batch_failed",
                                  "missing_or_repeated_boundary",
                                  "device_or_stream_changed",
                                  "missing_scores_ready_boundary",
                                  "missing_or_repeated_boundary"};
        check(std::strcmp(backend.reports.back().reason, expected[scenario]) == 0);
    }
}
}  // namespace
}  // namespace rtp_llm

int main() {
    rtp_llm::testDelayedCompletionAndCapacity();
    rtp_llm::testInvalidSamplesAndWarmup();
}
