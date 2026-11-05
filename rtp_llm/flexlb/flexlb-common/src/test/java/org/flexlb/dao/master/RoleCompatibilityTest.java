package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RoleCompatibilityTest {
    private static final RoleType P = RoleType.PREFILL;
    private static final RoleType D = RoleType.DECODE;

    static RoleCompatibility contract(int tp, int block, String frame, int cp) {
        int heads = 12 / tp;
        var group = new RoleCompatibility.Group("target/default", List.of(0, 1), "MultiHeadAttention", "bf16",
                frame, block, block, heads, List.of((long) heads * block * 128, (long) heads * block * 128, 0L, 0L),
                "full", "head-dim=64");
        return new RoleCompatibility(1, "target", null,
                new RoleCompatibility.Transfer("cache-store-tcp-v1", "model-token-layer-tag-v1", 0,
                        tp, cp, block, List.of(group), tp, cp > 1 ? "cp-sharded" : "tp"),
                new RoleCompatibility.Speculative("none", "", 1));
    }

    @Test
    void supportedAsymmetricTpIsNotTransitive() {
        var p4 = contract(4, 64, "split", 1);
        var p6 = contract(6, 64, "split", 1);
        var d2 = contract(2, 64, "split", 1);
        var d3 = contract(3, 64, "split", 1);
        for (var value : List.of(p4, p6, d2, d3)) { value.validate(P); }
        assertTrue(RoleCompatibility.matches(P, p4, D, d2));
        assertTrue(RoleCompatibility.matches(P, p6, D, d2));
        assertTrue(RoleCompatibility.matches(P, p6, D, d3));
        assertFalse(RoleCompatibility.matches(P, p4, D, d3));
    }

    @Test
    void framingCpAndBlockProjectionMustMatch() {
        var original = contract(2, 64, "split", 1);
        assertFalse(RoleCompatibility.matches(P, original, D, contract(2, 64, "kv", 1)));
        assertFalse(RoleCompatibility.matches(P, original, D, contract(2, 128, "split", 1)));
        assertFalse(RoleCompatibility.matches(P, original, D, contract(2, 64, "split", 4)));
        assertTrue(RoleCompatibility.matches(P, original, D, original));
    }

    @Test
    void legacyIsIsolatedAndInvalidIsNotLegacy() {
        assertTrue(RoleCompatibility.matches(P, null, D, null));
        assertFalse(RoleCompatibility.matches(P, contract(2, 64, "split", 1), D, null));
        assertThrows(IllegalArgumentException.class, () ->
                new RoleCompatibility(1, "target", null, null, null).validate(P));
        assertThrows(IllegalArgumentException.class, () ->
                new RoleCompatibility(2, "", new RoleCompatibility.Vision("v", "embedding"), null, null).validate(RoleType.VIT));
    }

    @Test
    void sharedVisionHasNoKvOrSpeculativeRequirements() {
        var vision = new RoleCompatibility.Vision("vision", "tensor-v1:128:3:0");
        var vit = new RoleCompatibility(1, "", vision, null, null);
        vit.validate(RoleType.VIT);
        for (int block : List.of(64, 128)) {
            var base = contract(2, block, "split", 1);
            var target = new RoleCompatibility(1, base.targetIdentity(), vision, base.transfer(), base.speculative());
            assertTrue(RoleCompatibility.matches(RoleType.VIT, vit, P, target));
        }
        assertThrows(IllegalArgumentException.class, () -> new RoleCompatibility(1, "", vision,
                contract(2, 64, "split", 1).transfer(), null).validate(RoleType.VIT));
    }

    @Test
    void speculativeModeDraftAndEffectiveWidthAreInvariants() {
        var base = contract(2, 64, "split", 1);
        var mtp = new RoleCompatibility(1, "target", null, base.transfer(),
                new RoleCompatibility.Speculative("mtp", "draft", 4));
        mtp.validate(P);
        assertFalse(RoleCompatibility.matches(P, base, D, mtp));
        assertFalse(RoleCompatibility.matches(P, mtp, D, new RoleCompatibility(1, "target", null,
                base.transfer(), new RoleCompatibility.Speculative("mtp", "draft", 1))));
        assertFalse(RoleCompatibility.matches(P, mtp, D, new RoleCompatibility(1, "target", null,
                base.transfer(), new RoleCompatibility.Speculative("mtp", "new-draft", 4))));
        assertThrows(IllegalArgumentException.class, () -> new RoleCompatibility(1, "target", null,
                base.transfer(), new RoleCompatibility.Speculative("mtp", "", 4)).validate(P));
    }

    @Test
    void shardedCpUsesDeclaredProducerLayoutRatherThanRawTpEquality() {
        var base = contract(1, 64, "split", 4);
        var p = new RoleCompatibility(1, "target", null, new RoleCompatibility.Transfer(
                "cache-store-tcp-v1", "model-token-layer-tag-v1", 0, 1, 4, 64,
                base.transfer().groups(), 4, "cp-sharded"), base.speculative());
        var d = new RoleCompatibility(1, "target", null, new RoleCompatibility.Transfer(
                "cache-store-tcp-v1", "model-token-layer-tag-v1", 0, 1, 4, 64,
                base.transfer().groups(), 1, "cp-sharded"), base.speculative());
        p.validate(P);
        d.validate(D);
        assertTrue(RoleCompatibility.matches(P, p, D, d));
    }

    @Test
    void shardedCpMhaCannotNormalizeDifferentPeerPayloadsByHeadCount() {
        var producer = contract(1, 64, "split", 4);
        var consumer = contract(2, 64, "split", 4);
        producer.validate(P);
        consumer.validate(D);
        assertFalse(RoleCompatibility.matches(P, producer, D, consumer));
        assertFalse(RoleCompatibility.matches(D, consumer, P, producer));
        var a = producer.transfer();
        var group = a.groups().getFirst();
        var mismatched = new RoleCompatibility.Group(group.tag(), group.layers(), group.cacheKind(), group.dtype(),
                group.framing(), group.blockTokens(), group.kernelBlockTokens(), group.localKvHeads(),
                List.of(group.wireBytes().getFirst() / 2, group.wireBytes().get(1) / 2, 0L, 0L),
                group.projection(), group.representation());
        var sameTp = new RoleCompatibility(1, "target", null,
                new RoleCompatibility.Transfer(a.backend(), a.keyFormat(), a.modelId(), a.tpSize(), a.producerCpSize(),
                        a.baseBlockTokens(), List.of(mismatched), a.workerTpSize(), a.producerLayout()), producer.speculative());
        sameTp.validate(D);
        assertFalse(RoleCompatibility.matches(P, producer, D, sameTp));
    }

    @Test
    void auxiliaryObservationsPreserveIdentityAndCannotReinterpretCommittedGeneration() {
        var worker = WorkerStatus.createDiscovered(RoleType.VIT, "blue", "127.0.0.1", 8080, 8081, "test");
        var response = new WorkerStatusResponse();
        response.setRole(RoleType.VIT);
        response.setStatusVersion(1L);
        response.setAlive(true);
        var first = new RoleCompatibility(1, "", new RoleCompatibility.Vision("vision1", "embedding"), null, null);
        response.setCompatibility(first);
        worker.lock.lock();
        try {
            worker.publishPreparedStatus(worker.prepareNewStatus(worker.freezeStatusResponse(response)));
            assertEquals(first, worker.compatibility());
            response.setStatusVersion(2L);
            response.setCompatibility(null);
            assertThrows(IllegalArgumentException.class, () -> worker.prepareNewStatus(worker.freezeStatusResponse(response)));
            response.setCompatibility(new RoleCompatibility(1, "", null, null, null));
            assertThrows(IllegalArgumentException.class, () -> worker.prepareNewStatus(worker.freezeStatusResponse(response)));
            assertEquals(first, worker.compatibility());
        } finally { worker.lock.unlock(); }
    }

    @Test
    void equalByteCountsDoNotOverrideLayerTailOrRepresentationMismatch() {
        var base = contract(2, 64, "split", 1);
        var g = base.transfer().groups().getFirst();
        for (var different : List.of(
                new RoleCompatibility.Group(g.tag(), List.of(0, 2), g.cacheKind(), g.dtype(), g.framing(),
                        g.blockTokens(), g.kernelBlockTokens(), g.localKvHeads(), g.wireBytes(), g.projection(), g.representation()),
                new RoleCompatibility.Group(g.tag(), g.layers(), g.cacheKind(), g.dtype(), g.framing(),
                        g.blockTokens(), g.kernelBlockTokens(), g.localKvHeads(), g.wireBytes(), "tail=2", g.representation()),
                new RoleCompatibility.Group(g.tag(), g.layers(), g.cacheKind(), g.dtype(), g.framing(),
                        g.blockTokens(), g.kernelBlockTokens(), g.localKvHeads(), g.wireBytes(), g.projection(), "different-packing"))) {
            var t = base.transfer();
            var other = new RoleCompatibility(1, "target", null,
                    new RoleCompatibility.Transfer(t.backend(), t.keyFormat(), t.modelId(), t.tpSize(), t.producerCpSize(),
                            t.baseBlockTokens(), List.of(different), t.workerTpSize(), t.producerLayout()), base.speculative());
            other.validate(D);
            assertFalse(RoleCompatibility.matches(P, base, D, other));
        }
    }
}
