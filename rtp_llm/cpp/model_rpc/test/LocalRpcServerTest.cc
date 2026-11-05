#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/model_rpc/test/RoleCompatibilityFixtures.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RoleCompatibility.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

using namespace ::testing;

namespace rtp_llm {

TEST(RoleCompatibilityTest, EncodesNativeEnumsWithoutDependingOnTheirOrdinals) {
    const std::pair<SpeculativeType, SpeculativeCompatibilityPB::Mode> modes[] = {
        {SP_TYPE_NONE, SpeculativeCompatibilityPB::MODE_NONE},
        {SP_TYPE_VANILLA, SpeculativeCompatibilityPB::MODE_VANILLA},
        {SP_TYPE_MTP, SpeculativeCompatibilityPB::MODE_MTP},
        {SP_TYPE_EAGLE3, SpeculativeCompatibilityPB::MODE_EAGLE3},
        {SP_TYPE_EAGLE, SpeculativeCompatibilityPB::MODE_EAGLE},
        {SP_TYPE_DETERMINISTIC, SpeculativeCompatibilityPB::MODE_DETERMINISTIC},
        {SP_TYPE_DSPARK, SpeculativeCompatibilityPB::MODE_DSPARK},
    };
    for (const auto& [native, wire] : modes) {
        EngineInitParams params;
        params.sp_config.type = native;
        EXPECT_EQ(makeRoleCompatibility(params, nullptr).speculative().mode(), wire);
    }
    const std::pair<DataType, KVTransferGroupPB::DType> dtypes[] = {
        {TYPE_FP16, KVTransferGroupPB::DTYPE_FP16},
        {TYPE_BF16, KVTransferGroupPB::DTYPE_BF16},
        {TYPE_FP32, KVTransferGroupPB::DTYPE_FP32},
        {TYPE_FP64, KVTransferGroupPB::DTYPE_FP64},
        {TYPE_INT8, KVTransferGroupPB::DTYPE_INT8},
        {TYPE_UINT8, KVTransferGroupPB::DTYPE_UINT8},
        {TYPE_INT16, KVTransferGroupPB::DTYPE_INT16},
        {TYPE_UINT16, KVTransferGroupPB::DTYPE_UINT16},
        {TYPE_INT32, KVTransferGroupPB::DTYPE_INT32},
        {TYPE_UINT32, KVTransferGroupPB::DTYPE_UINT32},
        {TYPE_INT64, KVTransferGroupPB::DTYPE_INT64},
        {TYPE_UINT64, KVTransferGroupPB::DTYPE_UINT64},
        {TYPE_FP8_E4M3, KVTransferGroupPB::DTYPE_FP8_E4M3},
        {TYPE_FP8_E8M0, KVTransferGroupPB::DTYPE_FP8_E8M0},
    };
    for (const auto& [native, wire] : dtypes) {
        EXPECT_EQ(compatibilityDType(native), wire);
    }
    EXPECT_EQ(compatibilityDType(TYPE_INVALID), KVTransferGroupPB::DTYPE_UNSPECIFIED);
    EXPECT_EQ(compatibilityDType(TYPE_BOOL), KVTransferGroupPB::DTYPE_UNSPECIFIED);
    const std::pair<KVCacheSpecType, KVTransferGroupPB::CacheKind> kinds[] = {
        {MultiHeadAttention, KVTransferGroupPB::CACHE_KIND_MULTI_HEAD_ATTENTION},
        {MultiHeadLatentAttention, KVTransferGroupPB::CACHE_KIND_MULTI_HEAD_LATENT_ATTENTION},
        {LinearAttention, KVTransferGroupPB::CACHE_KIND_LINEAR_ATTENTION},
        {OpaqueKV, KVTransferGroupPB::CACHE_KIND_OPAQUE_KV},
        {OpaqueState, KVTransferGroupPB::CACHE_KIND_OPAQUE_STATE},
    };
    for (const auto& [native, wire] : kinds) {
        EXPECT_EQ(compatibilityCacheKind(native), wire);
    }
}

TEST(RoleCompatibilityTest, ReportsIndependentIdentitiesAndSpeculation) {
    EngineInitParams params;
    params.model_config_.model_identity                = "target-v2";
    params.model_config_.vision_model_identity         = "vision-v1";
    params.model_config_.mm_model_config.is_multimodal = true;
    params.model_config_.hidden_size                   = 128;
    params.sp_config.type                              = SP_TYPE_MTP;
    params.sp_config.model_identity                    = "draft-v3";
    params.sp_config.gen_num_per_cycle                 = 4;
    const auto descriptor                              = makeRoleCompatibility(params, nullptr);
    EXPECT_EQ(descriptor.schema_version(), 1);
    EXPECT_EQ(descriptor.target_identity(), "target-v2");
    EXPECT_EQ(descriptor.vision().identity(), "vision-v1");
    EXPECT_EQ(descriptor.speculative().mode(), SpeculativeCompatibilityPB::MODE_MTP);
    EXPECT_EQ(descriptor.speculative().draft_identity(), "draft-v3");
    EXPECT_EQ(descriptor.speculative().proposal_width(), 4);
    EXPECT_FALSE(descriptor.has_kv_transfer());
    SpeculativeExecutionConfig effective;
    const auto                 ordinary = makeRoleCompatibility(params, nullptr, nullptr, &effective);
    EXPECT_EQ(ordinary.speculative().mode(), SpeculativeCompatibilityPB::MODE_NONE);
    EXPECT_EQ(ordinary.speculative().proposal_width(), 1);
    EXPECT_TRUE(ordinary.speculative().draft_identity().empty());
}

TEST(RoleCompatibilityTest, ReportsNativeAsymmetricTpAndCpContracts) {
    EngineInitParams params;
    params.model_id                                   = 0;
    params.model_config_.model_identity               = "target";
    params.model_config_.num_layers                   = 2;
    params.model_config_.data_type                    = DataType::TYPE_BF16;
    params.model_config_.attn_config.kv_head_num      = 12;
    params.model_config_.attn_config.size_per_head    = 64;
    params.model_config_.attn_config.tokens_per_block = 64;
    KVCacheSpecDesc desc;
    desc.tag                                 = "default";
    desc.cache_type                          = KVCacheSpecType::MultiHeadAttention;
    desc.dtype                               = DataType::TYPE_BF16;
    params.model_config_.kv_cache_spec_descs = {{desc}, {desc}};
    params.pd_sep_config.role_type           = RoleType::PREFILL;
    params.parallelism_config.tp_size        = 4;
    auto       config = SingleConfigCreator::createSingleConfig(params.model_config_, params.parallelism_config);
    const auto p      = makeRoleCompatibility(params, &config);
    ASSERT_EQ(p.kv_transfer().groups_size(), 1);
    EXPECT_EQ(p.kv_transfer().groups(0).local_kv_heads(), 3);
    EXPECT_EQ(p.kv_transfer().groups(0).wire_bytes(0), 3 * 64 * 64 * 2);
    EXPECT_EQ(p.kv_transfer().groups(0).framing(), KVTransferGroupPB::FRAMING_SPLIT);

    params.pd_sep_config.role_type    = RoleType::DECODE;
    params.parallelism_config.tp_size = 2;
    config       = SingleConfigCreator::createSingleConfig(params.model_config_, params.parallelism_config);
    const auto d = makeRoleCompatibility(params, &config);
    EXPECT_EQ(d.kv_transfer().groups(0).local_kv_heads(), 6);
    EXPECT_EQ(d.kv_transfer().groups(0).wire_bytes(0), 2 * p.kv_transfer().groups(0).wire_bytes(0));
    config.use_mla                 = true;
    params.pd_sep_config.role_type = RoleType::PREFILL;
    EXPECT_EQ(makeRoleCompatibility(params, &config).kv_transfer().groups(0).framing(),
              KVTransferGroupPB::FRAMING_SPLIT);
    params.pd_sep_config.role_type = RoleType::DECODE;
    EXPECT_EQ(makeRoleCompatibility(params, &config).kv_transfer().groups(0).framing(), KVTransferGroupPB::FRAMING_KV);

    params.pd_sep_config.role_type                               = RoleType::PREFILL;
    params.parallelism_config.tp_size                            = 4;
    params.parallelism_config.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    params.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
    config          = SingleConfigCreator::createSingleConfig(params.model_config_, params.parallelism_config);
    const auto cp_p = makeRoleCompatibility(params, &config);
    params.pd_sep_config.role_type                              = RoleType::DECODE;
    params.parallelism_config.tp_size                           = 1;
    params.parallelism_config.prefill_cp_config.method          = CPRotateMethod::PREFILL_CP;
    params.parallelism_config.prefill_cp_config.prefill_cp_size = 4;
    config          = SingleConfigCreator::createSingleConfig(params.model_config_, params.parallelism_config);
    const auto cp_d = makeRoleCompatibility(params, &config);
    EXPECT_EQ(cp_p.kv_transfer().producer_cp_size(), 4);
    EXPECT_EQ(cp_d.kv_transfer().producer_cp_size(), 4);
    EXPECT_EQ(cp_p.kv_transfer().tp_size(), 1);
    EXPECT_EQ(cp_d.kv_transfer().tp_size(), 1);
    EXPECT_EQ(cp_p.kv_transfer().groups(0).SerializeAsString(), cp_d.kv_transfer().groups(0).SerializeAsString());
}

TEST(RoleCompatibilityTest, HybridMtpOmitsPlaceholderGroupsAndKeepsComponentLayers) {
    EngineInitParams params;
    auto&            model                                = params.model_config_;
    model.model_identity                                  = "target";
    model.num_layers                                      = 4;
    model.data_type                                       = DataType::TYPE_BF16;
    model.attn_config.head_num                            = 12;
    model.attn_config.kv_head_num                         = 12;
    model.attn_config.size_per_head                       = 64;
    model.attn_config.tokens_per_block                    = 64;
    model.hybrid_attention_config.enable_hybrid_attention = true;
    model.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::LINEAR, HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::NONE};
    model.linear_attention_config.linear_conv_kernel_dim = 2;
    model.linear_attention_config.linear_key_head_dim    = 8;
    model.linear_attention_config.linear_value_head_dim  = 8;
    model.linear_attention_config.linear_num_key_heads   = 2;
    model.linear_attention_config.linear_num_value_heads = 2;
    model.kv_cache_spec_descs                            = {{{"linear", KVCacheSpecType::LinearAttention}},
                                                            {{"linear", KVCacheSpecType::LinearAttention}},
                                                            {{"full", KVCacheSpecType::MultiHeadAttention}},
                                                            {{"full", KVCacheSpecType::MultiHeadAttention}}};
    EngineInitParams draft;
    draft.model_id                                                      = 1;
    draft.model_config_                                                 = model;
    draft.model_config_.num_layers                                      = 1;
    draft.model_config_.model_identity                                  = "draft";
    draft.model_config_.hybrid_attention_config.enable_hybrid_attention = false;
    draft.model_config_.hybrid_attention_config.hybrid_attention_types.clear();
    draft.model_config_.kv_cache_spec_descs = {{{"full", KVCacheSpecType::MultiHeadAttention}}};
    params.sp_config.type                   = SP_TYPE_MTP;
    params.sp_config.model_identity         = "draft";
    params.sp_config.gen_num_per_cycle      = 2;
    params.parallelism_config.tp_size       = 1;
    params.kv_cache_config.test_block_num   = 8;
    auto config                             = CacheConfigCreator::createSpConfig(model,
                                                     draft.model_config_,
                                                     params.parallelism_config,
                                                     params.runtime_config,
                                                     params.kv_cache_config,
                                                     params.sp_config,
                                                     std::nullopt,
                                                     true,
                                                     false);
    ASSERT_EQ(config.groupNums(), 2);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2);
    ASSERT_EQ(config.mtp_sub_configs.front()->groupNums(), 2);
    for (auto role : {RoleType::PREFILL, RoleType::DECODE}) {
        params.pd_sep_config.role_type = role;
        const auto descriptor          = makeRoleCompatibility(params, &config, &draft);
        ASSERT_EQ(descriptor.kv_transfer().groups_size(), 3);
        for (const auto& group : descriptor.kv_transfer().groups()) {
            ASSERT_GT(group.layers_size(), 0);
            if (group.tag() == "target/linear") {
                EXPECT_EQ(std::vector<int>(group.layers().begin(), group.layers().end()), (std::vector<int>{0, 1}));
            } else if (group.tag() == "target/full") {
                EXPECT_EQ(std::vector<int>(group.layers().begin(), group.layers().end()), (std::vector<int>{2, 3}));
            } else {
                EXPECT_EQ(group.tag(), "draft/0/full");
                EXPECT_EQ(std::vector<int>(group.layers().begin(), group.layers().end()), (std::vector<int>{0}));
            }
        }
        expectCompatibilityFixture(role == RoleType::PREFILL ? "hybrid_mtp_prefill.pbtxt" : "hybrid_mtp_decode.pbtxt",
                                   descriptor);
    }
}

TEST(RoleCompatibilityTest, ConsumerDescribesExpectedOpaqueCpPeerSlice) {
    auto describe = [](bool consumer) {
        EngineInitParams params;
        params.model_id                             = 0;
        params.model_config_.model_identity         = "target";
        params.pd_sep_config.role_type              = consumer ? RoleType::DECODE : RoleType::PREFILL;
        auto& parallel                              = params.parallelism_config;
        parallel.role_type                          = params.pd_sep_config.role_type;
        parallel.tp_size                            = consumer ? 1 : 4;
        parallel.prefill_cp_config.kv_cache_sharded = true;
        parallel.prefill_cp_config.prefill_cp_size  = 4;
        parallel.prefill_cp_config.method = consumer ? CPRotateMethod::PREFILL_CP : CPRotateMethod::ALL_GATHER;
        KVCacheSpecDesc desc;
        desc.tag                  = "state";
        desc.cache_type           = KVCacheSpecType::OpaqueState;
        desc.entry_dtype          = DataType::TYPE_UINT8;
        desc.entry_elems          = 8;
        desc.explicit_entry_count = 16;
        desc.cp.emplace();
        desc.cp->scale_seq_size                  = true;
        desc.cp->slice                           = CpBlockSliceMode::PAYLOAD_BYTES;
        desc.cp->prefill_slice_layout            = CpPrefillSliceLayout::PAYLOAD;
        params.model_config_.kv_cache_spec_descs = {{desc}};
        SpecBuildContext ctx;
        ctx.seq_size_per_block = 64;
        ctx.parallelism_config = &parallel;
        auto        spec       = SpecBuilder::build(desc, ctx);
        CacheConfig config;
        config.layer_num                 = 1;
        config.seq_size_per_block        = 64;
        config.use_opaque_kv_cache_store = true;
        config.fromGroupedSpecs({spec}, {{0}}, {CacheGroupType::SWA}, {"state"}, {SpecBuilder::groupPolicy(desc)});
        return makeRoleCompatibility(params, &config);
    };
    const auto p = describe(false);
    const auto d = describe(true);
    EXPECT_EQ(p.kv_transfer().groups(0).wire_bytes(0), 32);
    EXPECT_EQ(d.kv_transfer().groups(0).wire_bytes(0), 32);
    EXPECT_EQ(p.kv_transfer().groups(0).SerializeAsString(), d.kv_transfer().groups(0).SerializeAsString());
}

class MockGenerateStream: public GenerateStream {
public:
    MockGenerateStream(const std::shared_ptr<GenerateInput>& input,
                       const ModelConfig&                    model_config,
                       const RuntimeConfig&                  runtime_config):
        GenerateStream(input, model_config, runtime_config, ResourceContext{}, nullptr) {}

    MOCK_METHOD((ErrorResult<GenerateOutputs>), nextOutput, (int64_t), (override));
    MOCK_METHOD(void, updateOutput, (const StreamUpdateInfo&), (override));
};

class TestLocalRpcServer: public LocalRpcServer {
public:
    grpc::Status poll(std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(nullptr, "request", nullptr, stream);
    }

    grpc::Status poll(WriterInterface* writer, std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(nullptr, "request", writer, stream);
    }

    ErrorInfo collect(std::shared_ptr<GenerateStream>& stream) {
        GenerateOutputs last_outputs;
        return collectStreamOutput(nullptr, stream, nullptr, last_outputs);
    }

    void configureAllocatorDump(bool enabled, std::string auth_token, double cooldown_seconds = 0.0) {
        torch_allocator_dump_enabled_                = enabled;
        torch_allocator_dump_auth_token_             = std::move(auth_token);
        torch_allocator_dump_cooldown_seconds_       = cooldown_seconds;
        maga_init_params_.parallelism_config.tp_rank = 0;
        maga_init_params_.parallelism_config.tp_size = 1;
    }

    void configureInternalAllocatorDumpPeer(bool allowed) {
        internal_allocator_dump_peer_allowed_        = allowed;
        maga_init_params_.parallelism_config.tp_size = 2;
    }

    grpc::Status authorizeAllocatorDump(const TorchAllocatorDumpRequestPB& request) const {
        return authorizeTorchAllocatorDump(request);
    }

    grpc::Status beginAllocatorDump(const std::string& dump_id) {
        return beginTorchAllocatorDump(dump_id);
    }

    void finishAllocatorDump() {
        finishTorchAllocatorDump();
    }

    size_t allocatorDumpReplayHistorySize() const {
        return torch_allocator_dump_ids_.size();
    }

    grpc::Status aggregateAllocatorDumpResults(const std::string&                             dump_id,
                                               const std::vector<TorchAllocatorDumpResultPB>& results,
                                               TorchAllocatorDumpResponsePB*                  response) const {
        return aggregateTorchAllocatorDumpResults(dump_id, results, response);
    }

    void setAllocatorDumpCallback(std::function<TorchAllocatorDumpResultPB(const std::string&)> callback) {
        allocator_dump_callback_ = std::move(callback);
    }

    std::future<void> cancellationChecked() {
        return cancellation_checked_.get_future();
    }

    std::atomic<bool> cancelled{false};

protected:
    bool isCancelled(grpc::ServerContext*) const override {
        std::call_once(cancellation_check_once_, [this] { cancellation_checked_.set_value(); });
        return cancelled.load();
    }

    bool isTorchAllocatorDumpInternalPeer(grpc::ServerContext*) const override {
        return internal_allocator_dump_peer_allowed_;
    }

    TorchAllocatorDumpResultPB dumpTorchAllocatorOnCurrentProcess(const std::string& dump_id) override {
        if (allocator_dump_callback_) {
            return allocator_dump_callback_(dump_id);
        }
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    }

private:
    std::function<TorchAllocatorDumpResultPB(const std::string&)> allocator_dump_callback_;
    bool                                                          internal_allocator_dump_peer_allowed_{false};
    mutable std::once_flag                                        cancellation_check_once_;
    mutable std::promise<void>                                    cancellation_checked_;
};

class RecordingWriter: public LocalRpcServer::WriterInterface {
public:
    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions) override {
        outputs_.push_back(outputs);
        return true;
    }

    std::vector<GenerateOutputsPB> outputs_;
};

class FailingWriter: public LocalRpcServer::WriterInterface {
public:
    bool Write(const GenerateOutputsPB&, grpc::WriteOptions) override {
        ++write_count_;
        return false;
    }

    size_t write_count_{0};
};

enum class WakeReason {
    OUTPUT,
    FINISHED,
    STREAM_ERROR,
    TIMEOUT
};

std::shared_ptr<MockGenerateStream> createMockStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<MockGenerateStream>(input, model_config, RuntimeConfig{});
}

std::shared_ptr<NormalGenerateStream> createNormalStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
}

ErrorResult<GenerateOutputs> wakeResult(WakeReason reason) {
    switch (reason) {
        case WakeReason::OUTPUT: {
            GenerateOutputs outputs;
            return ErrorResult<GenerateOutputs>(std::move(outputs));
        }
        case WakeReason::FINISHED:
            return ErrorResult<GenerateOutputs>(ErrorCode::FINISHED, "finished");
        case WakeReason::STREAM_ERROR:
            return ErrorResult<GenerateOutputs>(ErrorCode::EXECUTION_EXCEPTION, "failed");
        case WakeReason::TIMEOUT:
            return ErrorResult<GenerateOutputs>(ErrorCode::GENERATE_TIMEOUT, "timeout");
    }
    return ErrorResult<GenerateOutputs>(ErrorCode::UNKNOWN_ERROR, "unknown wake reason");
}

void publishWakeError(MockGenerateStream* stream, WakeReason reason) {
    if (reason == WakeReason::STREAM_ERROR) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "failed");
    } else if (reason == WakeReason::TIMEOUT) {
        stream->reportError(ErrorCode::GENERATE_TIMEOUT, "timeout");
    }
}

ErrorCode expectedStreamError(WakeReason reason) {
    if (reason == WakeReason::STREAM_ERROR) {
        return ErrorCode::EXECUTION_EXCEPTION;
    }
    if (reason == WakeReason::TIMEOUT) {
        return ErrorCode::GENERATE_TIMEOUT;
    }
    return ErrorCode::CANCELLED;
}

TEST(LocalRpcServerTest, AllocatorDumpAuthorizationRequiresEnablementAndSecret) {
    TestLocalRpcServer          server;
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-123");

    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::PERMISSION_DENIED);

    server.configureAllocatorDump(true, "expected-secret");
    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::UNAUTHENTICATED);

    request.set_auth_token("expected-secret");
    EXPECT_TRUE(server.authorizeAllocatorDump(request).ok());
}

TEST(LocalRpcServerTest, AllocatorDumpAuthorizationRejectsUnsafeCorrelationId) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("../leak-path");

    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerTest, RejectedAllocatorDumpRpcReturnsNoBackendDetails) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "expected-secret");
    grpc::ServerContext          context;
    TorchAllocatorDumpRequestPB  request;
    TorchAllocatorDumpResponsePB response;
    TorchAllocatorDumpResultPB   internal_response;
    request.set_auth_token("wrong-secret");
    request.set_dump_id("dump-123");
    response.add_results()->set_file_path("/stale/private/path");
    internal_response.set_file_path("/stale/private/path");
    internal_response.set_pid(1234);

    const auto status          = server.DumpTorchAllocator(&context, &request, &response);
    const auto internal_status = server.DumpTorchAllocatorInternal(&context, &request, &internal_response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAUTHENTICATED);
    EXPECT_EQ(internal_status.error_code(), grpc::StatusCode::UNAUTHENTICATED);
    EXPECT_EQ(response.results_size(), 0);
    EXPECT_TRUE(internal_response.file_path().empty());
    EXPECT_EQ(internal_response.pid(), 0);
}

TEST(LocalRpcServerTest, DirectAllocatorDumpRpcRejectsReplayAndCooldown) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret", 60.0);
    grpc::ServerContext          context;
    TorchAllocatorDumpRequestPB  request;
    TorchAllocatorDumpResponsePB response;
    request.set_auth_token("secret");
    request.set_dump_id("dump-first");

    EXPECT_TRUE(server.DumpTorchAllocator(&context, &request, &response).ok());
    ASSERT_EQ(response.results_size(), 1);
    EXPECT_EQ(response.results(0).dump_id(), "dump-first");

    EXPECT_EQ(server.DumpTorchAllocator(&context, &request, &response).error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(response.results_size(), 0);

    request.set_dump_id("dump-second");
    EXPECT_EQ(server.DumpTorchAllocator(&context, &request, &response).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(LocalRpcServerTest, DirectAllocatorDumpRpcIsSingleFlight) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    std::promise<void> dump_started;
    std::promise<void> release_dump;
    auto               release_future = release_dump.get_future().share();
    std::atomic<int>   dump_calls{0};
    server.setAllocatorDumpCallback([&](const std::string& dump_id) {
        if (dump_calls.fetch_add(1) == 0) {
            dump_started.set_value();
            release_future.wait();
        }
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    });

    TorchAllocatorDumpRequestPB first_request;
    first_request.set_auth_token("secret");
    first_request.set_dump_id("dump-in-flight");
    grpc::ServerContext          first_context;
    TorchAllocatorDumpResponsePB first_response;
    auto                         first_call = std::async(
        std::launch::async, [&] { return server.DumpTorchAllocator(&first_context, &first_request, &first_response); });
    const auto dump_started_status = dump_started.get_future().wait_for(std::chrono::seconds(5));
    if (dump_started_status != std::future_status::ready) {
        release_dump.set_value();
        first_call.wait();
        ADD_FAILURE() << "first allocator dump did not start";
        return;
    }

    TorchAllocatorDumpRequestPB second_request;
    second_request.set_auth_token("secret");
    second_request.set_dump_id("dump-concurrent");
    grpc::ServerContext          second_context;
    TorchAllocatorDumpResponsePB second_response;
    EXPECT_EQ(server.DumpTorchAllocator(&second_context, &second_request, &second_response).error_code(),
              grpc::StatusCode::ABORTED);

    release_dump.set_value();
    ASSERT_EQ(first_call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(first_call.get().ok());
    EXPECT_EQ(dump_calls.load(), 1);
}

TEST(LocalRpcServerTest, InternalAllocatorDumpRequiresTrustedTpPeer) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(false);
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-internal");
    grpc::ServerContext        context;
    TorchAllocatorDumpResultPB response;

    const auto status = server.DumpTorchAllocatorInternal(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::PERMISSION_DENIED);
    EXPECT_TRUE(response.dump_id().empty());
}

TEST(LocalRpcServerTest, InternalAllocatorDumpEnforcesAdmissionAndReplayProtection) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(true);
    std::promise<void> dump_started;
    std::promise<void> release_dump;
    auto               release_future = release_dump.get_future().share();
    server.setAllocatorDumpCallback([&](const std::string& dump_id) {
        dump_started.set_value();
        release_future.wait();
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    });

    TorchAllocatorDumpRequestPB first_request;
    first_request.set_auth_token("secret");
    first_request.set_dump_id("dump-internal-first");
    grpc::ServerContext        first_context;
    TorchAllocatorDumpResultPB first_response;
    auto                       first_call          = std::async(std::launch::async, [&] {
        return server.DumpTorchAllocatorInternal(&first_context, &first_request, &first_response);
    });
    const auto                 dump_started_status = dump_started.get_future().wait_for(std::chrono::seconds(5));
    if (dump_started_status != std::future_status::ready) {
        release_dump.set_value();
        first_call.wait();
        ADD_FAILURE() << "internal allocator dump did not start";
        return;
    }

    TorchAllocatorDumpRequestPB second_request;
    second_request.set_auth_token("secret");
    second_request.set_dump_id("dump-internal-second");
    grpc::ServerContext        second_context;
    TorchAllocatorDumpResultPB second_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&second_context, &second_request, &second_response).error_code(),
              grpc::StatusCode::ABORTED);

    release_dump.set_value();
    ASSERT_EQ(first_call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(first_call.get().ok());
    EXPECT_EQ(first_response.dump_id(), first_request.dump_id());

    grpc::ServerContext        replay_context;
    TorchAllocatorDumpResultPB replay_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&replay_context, &first_request, &replay_response).error_code(),
              grpc::StatusCode::ALREADY_EXISTS);
}

TEST(LocalRpcServerTest, InternalAllocatorDumpEnforcesCooldownOnEachRank) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret", 60.0);
    server.configureInternalAllocatorDumpPeer(true);
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-internal-first");
    grpc::ServerContext        first_context;
    TorchAllocatorDumpResultPB first_response;
    ASSERT_TRUE(server.DumpTorchAllocatorInternal(&first_context, &request, &first_response).ok());

    request.set_dump_id("dump-internal-second");
    grpc::ServerContext        second_context;
    TorchAllocatorDumpResultPB second_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&second_context, &request, &second_response).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST(LocalRpcServerTest, LeaderAdmissionAllowsExactlyOneMatchingInternalFanout) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(true);
    ASSERT_TRUE(server.beginAllocatorDump("dump-fanout").ok());

    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-fanout");
    grpc::ServerContext        context;
    TorchAllocatorDumpResultPB response;
    EXPECT_TRUE(server.DumpTorchAllocatorInternal(&context, &request, &response).ok());
    EXPECT_EQ(response.dump_id(), request.dump_id());

    grpc::ServerContext        replay_context;
    TorchAllocatorDumpResultPB replay_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&replay_context, &request, &replay_response).error_code(),
              grpc::StatusCode::ALREADY_EXISTS);

    EXPECT_EQ(server.beginAllocatorDump("dump-other").error_code(), grpc::StatusCode::ABORTED);
    server.finishAllocatorDump();
}

TEST(LocalRpcServerTest, AllocatorDumpReplayHistoryIsBounded) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    for (int index = 0; index <= 1024; ++index) {
        ASSERT_TRUE(server.beginAllocatorDump("dump-" + std::to_string(index)).ok());
        server.finishAllocatorDump();
    }

    EXPECT_EQ(server.allocatorDumpReplayHistorySize(), 1024);
    EXPECT_TRUE(server.beginAllocatorDump("dump-0").ok());
    server.finishAllocatorDump();
    EXPECT_EQ(server.allocatorDumpReplayHistorySize(), 1024);
}

TEST(LocalRpcServerTest, AllocatorDumpAggregationRejectsMismatchedIds) {
    TestLocalRpcServer         server;
    TorchAllocatorDumpResultPB matching;
    matching.set_dump_id("expected-id");
    matching.set_world_rank(0);
    TorchAllocatorDumpResultPB mismatched;
    mismatched.set_dump_id("other-id");
    mismatched.set_world_rank(1);
    TorchAllocatorDumpResponsePB response;
    response.add_results()->set_dump_id("stale-id");

    const auto status = server.aggregateAllocatorDumpResults("expected-id", {matching, mismatched}, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::DATA_LOSS);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(LocalRpcServerTest, PollChecksCancellationBeforeHandlingEveryWakeReason) {
    for (const auto reason :
         std::array{WakeReason::OUTPUT, WakeReason::FINISHED, WakeReason::STREAM_ERROR, WakeReason::TIMEOUT}) {
        TestLocalRpcServer server;
        auto               mock_stream     = createMockStream();
        auto*              mock_stream_ptr = mock_stream.get();
        EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([&server, mock_stream_ptr, reason] {
            publishWakeError(mock_stream_ptr, reason);
            server.cancelled = true;
            return wakeResult(reason);
        }));
        std::shared_ptr<GenerateStream> stream = mock_stream;

        const auto status = server.poll(stream);

        EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().code(), expectedStreamError(reason));
    }
}

TEST(LocalRpcServerTest, CollectChecksCancellationBeforeHandlingEveryWakeReason) {
    for (const auto reason :
         std::array{WakeReason::OUTPUT, WakeReason::FINISHED, WakeReason::STREAM_ERROR, WakeReason::TIMEOUT}) {
        TestLocalRpcServer server;
        auto               mock_stream     = createMockStream();
        auto*              mock_stream_ptr = mock_stream.get();
        EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([&server, mock_stream_ptr, reason] {
            publishWakeError(mock_stream_ptr, reason);
            server.cancelled = true;
            return wakeResult(reason);
        }));
        std::shared_ptr<GenerateStream> stream = mock_stream;

        const auto status = server.collect(stream);

        EXPECT_EQ(status.code(), ErrorCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().code(), expectedStreamError(reason));
    }
}

TEST(LocalRpcServerTest, PollInterruptsBlockedNextOutputAfterClientCancellation) {
    TestLocalRpcServer              server;
    auto                            cancellation_checked = server.cancellationChecked();
    std::shared_ptr<GenerateStream> stream               = createNormalStream();
    auto poll_result = std::async(std::launch::async, [&server, &stream] { return server.poll(stream); });

    EXPECT_EQ(cancellation_checked.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    server.cancelled = true;

    const auto wait_status = poll_result.wait_for(std::chrono::seconds(5));
    if (wait_status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test poll cancellation timed out");
    }
    EXPECT_EQ(wait_status, std::future_status::ready);
    EXPECT_EQ(poll_result.get().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(LocalRpcServerTest, PollClassifiesClosedResponseStreamAsCancelled) {
    TestLocalRpcServer server;
    FailingWriter      writer;
    auto               mock_stream = createMockStream();
    EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([] {
        GenerateOutputs outputs;
        outputs.request_id = 1;
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }));
    std::shared_ptr<GenerateStream> stream = mock_stream;

    const auto status = server.poll(&writer, stream);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(status.error_message(), "request output consumer closed");
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(writer.write_count_, 1);
}

TEST(LocalRpcServerTest, CollectInterruptsBlockedNextOutputAfterClientCancellation) {
    TestLocalRpcServer              server;
    auto                            cancellation_checked = server.cancellationChecked();
    std::shared_ptr<GenerateStream> stream               = createNormalStream();
    auto collect_result = std::async(std::launch::async, [&server, &stream] { return server.collect(stream); });

    EXPECT_EQ(cancellation_checked.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    server.cancelled = true;

    const auto wait_status = collect_result.wait_for(std::chrono::seconds(5));
    if (wait_status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test collect cancellation timed out");
    }
    EXPECT_EQ(wait_status, std::future_status::ready);
    EXPECT_EQ(collect_result.get().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(LocalRpcServerTest, PollWritesFinalLocalOutputBeforeRemoteHandoff) {
    TestLocalRpcServer              server;
    RecordingWriter                 writer;
    auto                            normal_stream = createNormalStream();
    std::shared_ptr<GenerateStream> stream        = normal_stream;
    normal_stream->setNeedReleaseResource(true);
    normal_stream->generate_status_->status.store(StreamState::RUNNING);

    {
        std::lock_guard<std::mutex> lock(*normal_stream->mutex_);
        GenerateOutputs             outputs;
        outputs.request_id = 123;
        normal_stream->enqueueGenerateOutput(std::move(outputs));
        normal_stream->reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
    }

    const auto status = server.poll(&writer, stream);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(writer.outputs_.size(), 1);
    EXPECT_EQ(writer.outputs_[0].request_id(), 123);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::NeedRemoteGenerate));
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(normal_stream->stream_cache_resource_->isResourceReleased());
    EXPECT_FALSE(normal_stream->hasOutput());
}

}  // namespace rtp_llm
