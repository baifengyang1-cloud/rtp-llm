#pragma once

#include <algorithm>
#include <sstream>
#include <vector>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

inline const char* compatibilityDTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_FP16:
            return "fp16";
        case DataType::TYPE_BF16:
            return "bf16";
        case DataType::TYPE_FP32:
            return "fp32";
        case DataType::TYPE_FP64:
            return "fp64";
        case DataType::TYPE_INT8:
            return "int8";
        case DataType::TYPE_UINT8:
            return "uint8";
        case DataType::TYPE_INT16:
            return "int16";
        case DataType::TYPE_UINT16:
            return "uint16";
        case DataType::TYPE_INT32:
            return "int32";
        case DataType::TYPE_UINT32:
            return "uint32";
        case DataType::TYPE_INT64:
            return "int64";
        case DataType::TYPE_UINT64:
            return "uint64";
        case DataType::TYPE_FP8_E4M3:
            return "fp8_e4m3";
        case DataType::TYPE_FP8_E8M0:
            return "fp8_e8m0";
        default:
            return "invalid";
    }
}

inline KVTransferGroupPB::DType compatibilityDType(DataType value) {
    switch (value) {
        case DataType::TYPE_FP16:
            return KVTransferGroupPB::DTYPE_FP16;
        case DataType::TYPE_BF16:
            return KVTransferGroupPB::DTYPE_BF16;
        case DataType::TYPE_FP32:
            return KVTransferGroupPB::DTYPE_FP32;
        case DataType::TYPE_FP64:
            return KVTransferGroupPB::DTYPE_FP64;
        case DataType::TYPE_INT8:
            return KVTransferGroupPB::DTYPE_INT8;
        case DataType::TYPE_UINT8:
            return KVTransferGroupPB::DTYPE_UINT8;
        case DataType::TYPE_INT16:
            return KVTransferGroupPB::DTYPE_INT16;
        case DataType::TYPE_UINT16:
            return KVTransferGroupPB::DTYPE_UINT16;
        case DataType::TYPE_INT32:
            return KVTransferGroupPB::DTYPE_INT32;
        case DataType::TYPE_UINT32:
            return KVTransferGroupPB::DTYPE_UINT32;
        case DataType::TYPE_INT64:
            return KVTransferGroupPB::DTYPE_INT64;
        case DataType::TYPE_UINT64:
            return KVTransferGroupPB::DTYPE_UINT64;
        case DataType::TYPE_FP8_E4M3:
            return KVTransferGroupPB::DTYPE_FP8_E4M3;
        case DataType::TYPE_FP8_E8M0:
            return KVTransferGroupPB::DTYPE_FP8_E8M0;
        default:
            return KVTransferGroupPB::DTYPE_UNSPECIFIED;
    }
}

inline KVTransferGroupPB::CacheKind compatibilityCacheKind(KVCacheSpecType value) {
    switch (value) {
        case KVCacheSpecType::MultiHeadAttention:
            return KVTransferGroupPB::CACHE_KIND_MULTI_HEAD_ATTENTION;
        case KVCacheSpecType::MultiHeadLatentAttention:
            return KVTransferGroupPB::CACHE_KIND_MULTI_HEAD_LATENT_ATTENTION;
        case KVCacheSpecType::LinearAttention:
            return KVTransferGroupPB::CACHE_KIND_LINEAR_ATTENTION;
        case KVCacheSpecType::OpaqueKV:
            return KVTransferGroupPB::CACHE_KIND_OPAQUE_KV;
        case KVCacheSpecType::OpaqueState:
            return KVTransferGroupPB::CACHE_KIND_OPAQUE_STATE;
        default:
            return KVTransferGroupPB::CACHE_KIND_UNSPECIFIED;
    }
}

inline SpeculativeCompatibilityPB::Mode compatibilitySpeculativeMode(SpeculativeType value) {
    switch (value) {
        case SP_TYPE_NONE:
            return SpeculativeCompatibilityPB::MODE_NONE;
        case SP_TYPE_VANILLA:
            return SpeculativeCompatibilityPB::MODE_VANILLA;
        case SP_TYPE_MTP:
            return SpeculativeCompatibilityPB::MODE_MTP;
        case SP_TYPE_EAGLE3:
            return SpeculativeCompatibilityPB::MODE_EAGLE3;
        case SP_TYPE_EAGLE:
            return SpeculativeCompatibilityPB::MODE_EAGLE;
        case SP_TYPE_DETERMINISTIC:
            return SpeculativeCompatibilityPB::MODE_DETERMINISTIC;
        case SP_TYPE_DSPARK:
            return SpeculativeCompatibilityPB::MODE_DSPARK;
        default:
            return SpeculativeCompatibilityPB::MODE_UNSPECIFIED;
    }
}

inline void appendTransferGroups(KVTransferCompatibilityPB& transfer,
                                 const CacheConfig&         cfg,
                                 bool                       consumer,
                                 const std::string&         component,
                                 size_t                     model_id,
                                 const ModelConfig&         model) {
    const auto cp = transfer.producer_cp_size();
    for (size_t gid = 0; gid < static_cast<size_t>(cfg.groupNums()); ++gid) {
        std::vector<int> layers;
        for (int layer : cfg.layerIdsForGroup(gid)) {
            if (layer >= 0 && layer < static_cast<int>(cfg.layer_num)) {
                layers.push_back(layer);
            }
        }
        if (layers.empty()) {
            continue;
        }
        const auto& spec   = cfg.specForGroup(gid);
        const auto  policy = cfg.policyForGroup(gid);
        auto*       group  = transfer.add_groups();
        group->set_tag(component + cfg.tagForGroup(gid));
        for (int layer : layers) {
            group->add_layers(layer);
        }
        group->set_cache_kind(compatibilityCacheKind(spec->type));
        group->set_dtype(compatibilityDType(spec->memoryLayoutDType()));
        const bool whole = cfg.use_opaque_kv_cache_store || cfg.groupNums() > 1
                           || (consumer ? cfg.use_mla : spec->type == KVCacheSpecType::MultiHeadLatentAttention);
        group->set_framing(whole ? KVTransferGroupPB::FRAMING_KV : KVTransferGroupPB::FRAMING_SPLIT);
        group->set_block_tokens(cfg.seqSizePerBlockForGroup(gid));
        const auto kernel_tokens = cfg.kernelSeqSizePerBlockForGroup(gid);
        group->set_kernel_block_tokens(kernel_tokens == 0 ? cfg.seqSizePerBlockForGroup(gid) : kernel_tokens);
        group->set_local_kv_heads(cfg.localKvHeadNumForGroup(gid));
        size_t     kv_bytes    = cfg.kvBlockStrideBytesForGroup(gid);
        size_t     scale_bytes = cfg.kvScaleStrideBytesForGroup(gid);
        const auto slice =
            cp > 1 && policy.group_type != CacheGroupType::FULL ? policy.cp_slice : CpBlockSliceMode::NONE;
        if (consumer && slice != CpBlockSliceMode::NONE) {
            // Mirrors CPSlotMapper::sliceBlockForPeer: scales cannot be sliced
            // by that single-part transfer path.
            RTP_LLM_CHECK_WITH_INFO(whole && scale_bytes == 0, "unsupported CP scale/framing contract");
            const size_t full = slice == CpBlockSliceMode::EQUAL_BYTES ? kv_bytes : spec->k_block_payload_bytes();
            RTP_LLM_CHECK_WITH_INFO(full % cp == 0, "CP transfer payload is not divisible by CP size");
            kv_bytes = full / cp;
        }
        group->add_wire_bytes(whole ? kv_bytes : kv_bytes / 2);
        group->add_wire_bytes(whole ? 0 : kv_bytes / 2);
        group->add_wire_bytes(whole ? scale_bytes : scale_bytes / 2);
        group->add_wire_bytes(whole ? 0 : scale_bytes / 2);
        std::ostringstream projection;
        projection << "model=" << model_id << ";base=" << cfg.seq_size_per_block
                   << ";group=" << static_cast<int>(policy.group_type) << ";tail=" << policy.active_tail_blocks
                   << ";validate_tail=" << policy.validate_tail_blocks
                   << ";mapping=" << (cp > 1 ? static_cast<int>(policy.cp_mapping) : 0)
                   << ";slice=" << static_cast<int>(slice) << ";hybrid=" << (cfg.groupNums() > 1);
        group->set_projection(projection.str());
        std::ostringstream representation;
        representation << "kv-v1";
        if (spec->type == KVCacheSpecType::MultiHeadAttention) {
            representation << ";head_dim=" << model.attn_config.size_per_head;
        } else if (spec->type == KVCacheSpecType::MultiHeadLatentAttention) {
            representation << ";nope=" << model.attn_config.kv_lora_rank << ";rope=" << model.attn_config.rope_head_dim;
        } else if (spec->type == KVCacheSpecType::LinearAttention) {
            const auto& linear = model.linear_attention_config;
            representation << ";key_head_dim=" << linear.linear_key_head_dim
                           << ";value_head_dim=" << linear.linear_value_head_dim
                           << ";key_heads=" << linear.linear_num_key_heads
                           << ";value_heads=" << linear.linear_num_value_heads
                           << ";conv=" << linear.linear_conv_kernel_dim
                           << ";ssm_dtype=" << static_cast<int>(linear.ssm_state_dtype)
                           << ";conv_dtype=" << static_cast<int>(linear.conv_state_dtype);
        } else {
            // Declarative opaque packing parameters are shared across roles;
            // their materialized byte counts above account for CP slicing.
            for (const auto& layer : model.kv_cache_spec_descs) {
                auto found = std::find_if(layer.begin(), layer.end(), [&](const KVCacheSpecDesc& desc) {
                    return desc.tag == cfg.tagForGroup(gid);
                });
                if (found == layer.end()) {
                    continue;
                }
                const auto& desc = *found;
                representation << ";entry=" << desc.entry_elems << ";entry_dtype=" << static_cast<int>(desc.entry_dtype)
                               << ";count_mode=" << static_cast<int>(desc.entry_count_mode)
                               << ";count=" << desc.explicit_entry_count << ";compression=" << desc.compression_ratio
                               << ";ring=" << desc.state_ring_overlap
                               << ";ring_gen=" << desc.state_ring_include_gen_num_per_cycle;
                if (desc.cp.has_value()) {
                    representation << ";cp_scale_seq=" << desc.cp->scale_seq_size.value_or(false)
                                   << ";cp_align_payload=" << desc.cp->align_payload.value_or(false)
                                   << ";cp_prefill_layout="
                                   << static_cast<int>(
                                          desc.cp->prefill_slice_layout.value_or(CpPrefillSliceLayout::NONE));
                }
                break;
            }
        }
        group->set_representation(representation.str());
    }
}

inline RoleCompatibilityPB makeRoleCompatibility(const EngineInitParams&           params,
                                                 const CacheConfig*                cfg,
                                                 const EngineInitParams*           draft          = nullptr,
                                                 const SpeculativeExecutionConfig* effective_spec = nullptr) {
    RoleCompatibilityPB result;
    result.set_schema_version(1);
    const auto& model = params.model_config_;
    result.set_target_identity(model.model_identity.empty() ? model.ckpt_path : model.model_identity);
    if (model.mm_model_config.is_multimodal) {
        auto* vision = result.mutable_vision();
        vision->set_identity(model.vision_model_identity.empty() ? model.extra_data_path : model.vision_model_identity);
        vision->set_embedding_contract("tensor-v1:" + std::to_string(model.hidden_size) + ":"
                                       + compatibilityDTypeName(model.data_type) + ":"
                                       + std::to_string(model.mm_model_config.mm_position_ids_style));
    }
    auto*       speculative = result.mutable_speculative();
    const auto& sp          = effective_spec == nullptr ? params.sp_config : *effective_spec;
    speculative->set_mode(compatibilitySpeculativeMode(sp.type));
    if (sp.type != SP_TYPE_NONE && sp.type != SP_TYPE_DETERMINISTIC) {
        speculative->set_draft_identity(sp.model_identity.empty() ? sp.checkpoint_path : sp.model_identity);
    }
    speculative->set_proposal_width(sp.type == SP_TYPE_NONE ? 1 : sp.gen_num_per_cycle);
    if (cfg != nullptr && params.pd_sep_config.role_type != RoleType::PDFUSION) {
        auto* transfer = result.mutable_kv_transfer();
        transfer->set_backend(params.pd_sep_config.cache_store_rdma_mode ?
                                  KVTransferCompatibilityPB::BACKEND_CACHE_STORE_RDMA_V1 :
                                  KVTransferCompatibilityPB::BACKEND_CACHE_STORE_TCP_V1);
        transfer->set_key_format(KVTransferCompatibilityPB::KEY_FORMAT_MODEL_TOKEN_LAYER_TAG_V1);
        transfer->set_model_id(params.model_id);
        const auto& parallel  = params.parallelism_config;
        const auto& cp_config = parallel.prefill_cp_config;
        const bool  consumer  = params.pd_sep_config.role_type == RoleType::DECODE;
        const int   cp        = !cp_config.kv_cache_sharded ? 1 :
                                consumer ? (cp_config.is_prefill_enabled() ? cp_config.prefill_cp_size : 1) :
                                           (cp_config.is_enabled() ? parallel.tp_size : 1);
        transfer->set_producer_cp_size(cp);
        transfer->set_tp_size(parallel.get_attn_tp_size());
        transfer->set_worker_tp_size(parallel.tp_size);
        transfer->set_producer_layout(cp > 1 ? KVTransferCompatibilityPB::PRODUCER_LAYOUT_CP_SHARDED :
                                      (consumer ? cp_config.is_prefill_enabled() : cp_config.is_enabled()) ?
                                               KVTransferCompatibilityPB::PRODUCER_LAYOUT_CP_REPLICATED :
                                               KVTransferCompatibilityPB::PRODUCER_LAYOUT_TP);
        transfer->set_base_block_tokens(cfg->seq_size_per_block);
        appendTransferGroups(*transfer, *cfg, consumer, "target/", params.model_id, model);
        if (draft != nullptr && !cfg->mtp_sub_configs.empty() && cfg->mtp_sub_configs.front() != nullptr) {
            appendTransferGroups(
                *transfer, *cfg->mtp_sub_configs.front(), consumer, "draft/0/", draft->model_id, draft->model_config_);
        }
    }
    return result;
}

}  // namespace rtp_llm
