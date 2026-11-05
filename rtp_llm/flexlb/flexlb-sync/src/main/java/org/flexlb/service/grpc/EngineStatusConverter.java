package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatus.EngineObservation;
import org.flexlb.dao.master.WorkerStatus.StatusObservation;
import org.flexlb.dao.master.WorkerStatus.TaskObservation;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Converter between gRPC protobuf messages and Java objects for engine status
 */
public class EngineStatusConverter {

    /** Convert one protobuf response directly into the immutable status boundary. */
    public static StatusObservation convertToStatusObservation(
            WorkerStatus owner,
            EngineRpcService.WorkerStatusPB workerStatusPB) {
        Map<String, TaskObservation> runningTasks = convertTasks(
                workerStatusPB.getRunningTaskInfoList());
        Map<String, TaskObservation> finishedTasks = convertTasks(
                workerStatusPB.getFinishedTaskListList());
        EngineObservation engine = new EngineObservation(
                RoleTypeProtoConverter.fromWorkerStatus(workerStatusPB),
                (long) workerStatusPB.getAvailableConcurrency(),
                workerStatusPB.getAvailableKvCache(),
                workerStatusPB.getTotalKvCache(),
                runningTasks,
                workerStatusPB.getStepLatencyMs(),
                workerStatusPB.getIterateCount(),
                workerStatusPB.getDpSize(),
                workerStatusPB.getTpSize(),
                workerStatusPB.getDpRank(),
                workerStatusPB.getMaxSeqLen(),
                workerStatusPB.getMaxBatchTokensSize(),
                workerStatusPB.getRunningQueryLen(),
                workerStatusPB.getWaitingQueryLen());
        return owner.bindStatusObservation(
                engine,
                workerStatusPB.getAlive(),
                workerStatusPB.getStatusVersion(),
                workerStatusPB.getLatestFinishedVersion(),
                finishedTasks,
                workerStatusPB.hasCompatibility() ? convertCompatibility(workerStatusPB.getCompatibility()) : null);
    }

    private static RoleCompatibility convertCompatibility(EngineRpcService.RoleCompatibilityPB value) {
        RoleCompatibility.Vision vision = value.hasVision() ? new RoleCompatibility.Vision(
                value.getVision().getIdentity(), value.getVision().getEmbeddingContract()) : null;
        RoleCompatibility.Speculative speculative = value.hasSpeculative() ? new RoleCompatibility.Speculative(
                convertMode(value.getSpeculative().getMode()), value.getSpeculative().getDraftIdentity(),
                value.getSpeculative().getProposalWidth()) : null;
        RoleCompatibility.Transfer transfer = null;
        if (value.hasKvTransfer()) {
            var kv = value.getKvTransfer();
            transfer = new RoleCompatibility.Transfer(convertBackend(kv.getBackend()), convertKeyFormat(kv.getKeyFormat()),
                    kv.getModelId(),
                    kv.getTpSize(), kv.getProducerCpSize(), kv.getBaseBlockTokens(), kv.getGroupsList().stream()
                    .map(g -> new RoleCompatibility.Group(g.getTag(), g.getLayersList(), convertCacheKind(g.getCacheKind()),
                            convertDType(g.getDtype()), convertFraming(g.getFraming()),
                            g.getBlockTokens(), g.getKernelBlockTokens(),
                            g.getLocalKvHeads(), g.getWireBytesList(), g.getProjection(), g.getRepresentation())).toList(),
                    kv.getWorkerTpSize(), convertProducerLayout(kv.getProducerLayout()));
        }
        return new RoleCompatibility(value.getSchemaVersion(), value.getTargetIdentity(), vision, transfer, speculative);
    }

    // Keep the domain independent of generated protobuf classes. Known enums map
    // to interned literals; unspecified/future values remain invalid in validate().
    private static String convertMode(EngineRpcService.SpeculativeCompatibilityPB.Mode value) {
        return switch (value) {
            case MODE_NONE -> "none";
            case MODE_VANILLA -> "vanilla";
            case MODE_MTP -> "mtp";
            case MODE_EAGLE3 -> "eagle3";
            case MODE_EAGLE -> "eagle";
            case MODE_DETERMINISTIC -> "deterministic";
            case MODE_DSPARK -> "dspark";
            default -> "";
        };
    }

    private static String convertBackend(EngineRpcService.KVTransferCompatibilityPB.Backend value) {
        return switch (value) {
            case BACKEND_CACHE_STORE_RDMA_V1 -> "cache-store-rdma-v1";
            case BACKEND_CACHE_STORE_TCP_V1 -> "cache-store-tcp-v1";
            default -> "";
        };
    }

    private static String convertKeyFormat(EngineRpcService.KVTransferCompatibilityPB.KeyFormat value) {
        return switch (value) {
            case KEY_FORMAT_MODEL_TOKEN_LAYER_TAG_V1 -> "model-token-layer-tag-v1";
            default -> "";
        };
    }

    private static String convertProducerLayout(EngineRpcService.KVTransferCompatibilityPB.ProducerLayout value) {
        return switch (value) {
            case PRODUCER_LAYOUT_TP -> "tp";
            case PRODUCER_LAYOUT_CP_REPLICATED -> "cp-replicated";
            case PRODUCER_LAYOUT_CP_SHARDED -> "cp-sharded";
            default -> "";
        };
    }

    private static String convertCacheKind(EngineRpcService.KVTransferGroupPB.CacheKind value) {
        return switch (value) {
            case CACHE_KIND_MULTI_HEAD_ATTENTION -> "MultiHeadAttention";
            case CACHE_KIND_MULTI_HEAD_LATENT_ATTENTION -> "MultiHeadLatentAttention";
            case CACHE_KIND_LINEAR_ATTENTION -> "LinearAttention";
            case CACHE_KIND_OPAQUE_KV -> "OpaqueKV";
            case CACHE_KIND_OPAQUE_STATE -> "OpaqueState";
            default -> "";
        };
    }

    private static String convertDType(EngineRpcService.KVTransferGroupPB.DType value) {
        return switch (value) {
            case DTYPE_FP16 -> "fp16";
            case DTYPE_BF16 -> "bf16";
            case DTYPE_FP32 -> "fp32";
            case DTYPE_FP64 -> "fp64";
            case DTYPE_INT8 -> "int8";
            case DTYPE_UINT8 -> "uint8";
            case DTYPE_INT16 -> "int16";
            case DTYPE_UINT16 -> "uint16";
            case DTYPE_INT32 -> "int32";
            case DTYPE_UINT32 -> "uint32";
            case DTYPE_INT64 -> "int64";
            case DTYPE_UINT64 -> "uint64";
            case DTYPE_FP8_E4M3 -> "fp8_e4m3";
            case DTYPE_FP8_E8M0 -> "fp8_e8m0";
            default -> "";
        };
    }

    private static String convertFraming(EngineRpcService.KVTransferGroupPB.Framing value) {
        return switch (value) {
            case FRAMING_KV -> "kv";
            case FRAMING_SPLIT -> "split";
            default -> "";
        };
    }

    /**
     * Convert CacheStatusPB to CacheStatus
     */
    public static CacheStatus convertToCacheStatus(EngineRpcService.CacheStatusPB cacheStatusPB) {
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(cacheStatusPB.getAvailableKvCache());
        cacheStatus.setTotalKvCache(cacheStatusPB.getTotalKvCache());
        cacheStatus.setBlockSize(cacheStatusPB.getBlockSize());
        cacheStatus.setVersion(cacheStatusPB.getVersion());

        Map<Long, Boolean> cacheKeysMap = cacheStatusPB.getCacheKeysMap();
        // Copy the keySet: protobuf MapField#keySet() returns a VIEW that keeps
        // the whole WorkerStatusPB message graph reachable until the next sync.
        Set<Long> cachedKeysSet = new HashSet<>(cacheKeysMap.keySet());
        cacheStatus.setCachedKeys(cachedKeysSet);
        cacheStatus.setCacheKeySize(cacheKeysMap.size());
        return cacheStatus;
    }

    /**
     * Convert protobuf task values directly into immutable observations.
     */
    private static Map<String, TaskObservation> convertTasks(
            List<EngineRpcService.TaskInfoPB> taskInfoPBList) {
        if (taskInfoPBList == null || taskInfoPBList.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> tasks = new HashMap<>(
                taskInfoPBList.size());

        for (EngineRpcService.TaskInfoPB task : taskInfoPBList) {
            long errorCode = task.hasErrorInfo()
                    ? task.getErrorInfo().getErrorCode() : 0L;
            String errorMessage = errorCode == 0L
                    ? null : task.getErrorInfo().getErrorMessage();
            TaskObservation observation = new TaskObservation(
                    task.getRequestId(),
                    task.getPrefixLength(),
                    0L,
                    task.getInputLength(),
                    task.getWaitingTimeMs(),
                    task.getIterateCount(),
                    task.getEndTimeMs(),
                    task.getDpRank(),
                    errorCode,
                    errorMessage,
                    task.getBatchId(),
                    resolvePhase(task),
                    task.getExecutionTimeMs(),
                    switch (task.getPriorityPreemptionProgress()) {
                        case PRIORITY_PREEMPTION_CANCELING ->
                                PriorityPreemptionProgress.CANCELING;
                        case PRIORITY_PREEMPTION_CANCELED ->
                                PriorityPreemptionProgress.CANCELED;
                        case PRIORITY_PREEMPTION_NONE, UNRECOGNIZED ->
                                PriorityPreemptionProgress.NONE;
                    });
            tasks.put(String.valueOf(task.getRequestId()), observation);
        }
        return Map.copyOf(tasks);
    }

    private static TaskPhase resolvePhase(EngineRpcService.TaskInfoPB task) {
        if (task.getPhase() != EngineRpcService.TaskPhase.TASK_PHASE_PENDING) {
            // phase was added after the legacy is_waiting flag and is the
            // authoritative state whenever it carries a non-default value.
            // In particular, the immediately preceding e0 wire schema did not
            // contain field 9, so its RECEIVED/KV_ALLOCATED payloads are read
            // here with is_waiting=false. Treating that default as an explicit
            // value rejects valid status snapshots during a rolling upgrade.
            return convertPhase(task.getPhase());
        }
        // dsv4 writers only sent is_waiting. Because old proto3 writers omit
        // false on the wire, phase=0/is_waiting=false means legacy RUNNING.
        // New PENDING writers dual-write is_waiting=true.
        return task.getIsWaiting() ? TaskPhase.PENDING : TaskPhase.RUNNING;
    }

    private static TaskPhase convertPhase(EngineRpcService.TaskPhase protoPhase) {
        switch (protoPhase) {
            case TASK_PHASE_RECEIVED:
                return TaskPhase.RECEIVED;
            case TASK_PHASE_KV_ALLOCATED:
                return TaskPhase.KV_ALLOCATED;
            case TASK_PHASE_RUNNING:
                return TaskPhase.RUNNING;
            default:
                return TaskPhase.PENDING;
        }
    }
}
