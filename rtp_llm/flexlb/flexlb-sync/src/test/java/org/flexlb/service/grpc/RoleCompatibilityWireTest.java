package org.flexlb.service.grpc;

import com.google.protobuf.TextFormat;
import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.*;

class RoleCompatibilityWireTest {
    private EngineRpcService.RoleCompatibilityPB.Builder descriptor(String fixture) throws Exception {
        var descriptor = EngineRpcService.RoleCompatibilityPB.newBuilder();
        var stream = getClass().getResourceAsStream("/compatibility/" + fixture + ".pbtxt");
        assertNotNull(stream, fixture);
        try (var reader = new InputStreamReader(stream, StandardCharsets.UTF_8)) {
            TextFormat.getParser().merge(reader, descriptor);
        }
        return descriptor;
    }

    private RoleCompatibility convert(EngineRpcService.RoleCompatibilityPB.Builder descriptor, RoleType role) throws Exception {
        var encoded = EngineRpcService.WorkerStatusPB.newBuilder().setRole(role.name()).setAlive(true)
                .setStatusVersion(1).setCompatibility(descriptor).build().toByteArray();
        var response = EngineRpcService.WorkerStatusPB.parseFrom(encoded);
        var owner = WorkerStatus.createDiscovered(role, "blue", "127.0.0.1", 8000, 8001, "test");
        var converted = EngineStatusConverter.convertToStatusObservation(owner, response).compatibility();
        assertNotNull(converted);
        return converted;
    }

    private RoleCompatibility read(String fixture, RoleType role) throws Exception {
        var converted = convert(descriptor(fixture), role);
        converted.validate(role);
        return converted;
    }

    @Test
    void unspecifiedAndUnknownEnumsAreInvalidRatherThanLegacyOrDefaults() throws Exception {
        List<BiConsumer<EngineRpcService.RoleCompatibilityPB.Builder, Integer>> changes = List.of(
                (b, n) -> b.getSpeculativeBuilder().setModeValue(n),
                (b, n) -> b.getKvTransferBuilder().setBackendValue(n),
                (b, n) -> b.getKvTransferBuilder().setKeyFormatValue(n),
                (b, n) -> b.getKvTransferBuilder().setProducerLayoutValue(n),
                (b, n) -> b.getKvTransferBuilder().getGroupsBuilder(0).setCacheKindValue(n),
                (b, n) -> b.getKvTransferBuilder().getGroupsBuilder(0).setDtypeValue(n),
                (b, n) -> b.getKvTransferBuilder().getGroupsBuilder(0).setFramingValue(n));
        for (var change : changes) {
            for (int number : List.of(0, 127)) {
                var builder = descriptor("hybrid_mtp_prefill");
                change.accept(builder, number);
                var converted = convert(builder, RoleType.PREFILL);
                assertNotNull(converted);
                assertThrows(IllegalArgumentException.class, () -> converted.validate(RoleType.PREFILL));
            }
        }
    }

    @Test
    void nonSpeculativeModeIsExplicitAndSurvivesBinaryRoundTrip() throws Exception {
        var descriptor = descriptor("cp4_prefill").build();
        assertEquals(EngineRpcService.SpeculativeCompatibilityPB.Mode.MODE_NONE, descriptor.getSpeculative().getMode());
        assertNotEquals(0, descriptor.getSpeculative().getModeValue());
        var converted = read("cp4_prefill", RoleType.PREFILL);
        assertEquals("none", converted.speculative().mode());
        assertEquals("cache-store-rdma-v1", converted.transfer().backend());
        assertEquals("model-token-layer-tag-v1", converted.transfer().keyFormat());
        assertEquals("cp-sharded", converted.transfer().producerLayout());
        assertEquals("MultiHeadAttention", converted.transfer().groups().getFirst().cacheKind());
        assertEquals("bf16", converted.transfer().groups().getFirst().dtype());
        assertEquals("split", converted.transfer().groups().getFirst().framing());
    }

    @Test
    void nativeHybridMtpDescriptorsConvertAndValidateWithoutPlaceholderLayers() throws Exception {
        var prefill = read("hybrid_mtp_prefill", RoleType.PREFILL);
        var decode = read("hybrid_mtp_decode", RoleType.DECODE);
        assertEquals(3, prefill.transfer().groups().size());
        assertEquals(List.of("draft/0/full", "target/full", "target/linear"),
                prefill.transfer().groups().stream().map(RoleCompatibility.Group::tag).toList());
        assertEquals(List.of(List.of(0), List.of(2, 3), List.of(0, 1)),
                prefill.transfer().groups().stream().map(RoleCompatibility.Group::layers).toList());
        assertTrue(RoleCompatibility.matches(RoleType.PREFILL, prefill, RoleType.DECODE, decode));
    }

    @Test
    void nativeCp4PeerPayloadsPermitDecodeTp1AndRejectDecodeTp2() throws Exception {
        var prefill = read("cp4_prefill", RoleType.PREFILL);
        var decode1 = read("cp4_decode_tp1", RoleType.DECODE);
        var decode2 = read("cp4_decode_tp2", RoleType.DECODE);
        assertEquals(4, prefill.transfer().workerTpSize());
        assertEquals(1, prefill.transfer().tpSize());
        assertEquals(1, decode1.transfer().workerTpSize());
        assertEquals(2, decode2.transfer().tpSize());
        assertEquals(98304L, prefill.transfer().groups().getFirst().wireBytes().getFirst());
        assertEquals(49152L, decode2.transfer().groups().getFirst().wireBytes().getFirst());
        assertTrue(RoleCompatibility.matches(RoleType.PREFILL, prefill, RoleType.DECODE, decode1));
        assertTrue(RoleCompatibility.matches(RoleType.DECODE, decode1, RoleType.PREFILL, prefill));
        assertFalse(RoleCompatibility.matches(RoleType.PREFILL, prefill, RoleType.DECODE, decode2));
        assertFalse(RoleCompatibility.matches(RoleType.DECODE, decode2, RoleType.PREFILL, prefill));
    }
}
