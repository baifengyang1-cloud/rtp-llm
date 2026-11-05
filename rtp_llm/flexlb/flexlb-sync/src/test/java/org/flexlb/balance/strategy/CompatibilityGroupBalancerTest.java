package org.flexlb.balance.strategy;

import com.github.benmanes.caffeine.cache.Cache;
import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class CompatibilityGroupBalancerTest {
    private static final List<RoleType> PD = List.of(RoleType.PREFILL, RoleType.DECODE);
    private static final List<RoleType> PDV = List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT);

    @Test
    void fractionalAttributionConservesPhysicalWorkersAndExcludesIncompleteGroups() {
        var a = group("a", Map.of(RoleType.PREFILL, Set.of(1L), RoleType.DECODE, Set.of(10L),
                RoleType.VIT, new HashSet<>(List.of(21L, 21L, 22L))));
        var b = group("b", Map.of(RoleType.PREFILL, Set.of(2L), RoleType.DECODE, Set.of(11L), RoleType.VIT, Set.of(22L)));
        var incomplete = group("c", Map.of(RoleType.PREFILL, Set.of(3L), RoleType.VIT, Set.of(22L)));
        var weighted = CompatibilityGroup.weighted(List.of(a, a, b, incomplete), PDV);
        assertEquals(2, weighted.size());
        assertEquals(1.5, weighted.get(0).effectiveCounts().get(RoleType.VIT));
        assertEquals(0.5, weighted.get(1).effectiveCounts().get(RoleType.VIT));
        assertEquals(Math.cbrt(1.5), weighted.get(0).weight(), 1e-12);
        assertEquals(Math.cbrt(0.5), weighted.get(1).weight(), 1e-12);
        for (RoleType role : PDV) {
            assertEquals(2.0, weighted.stream().mapToDouble(g -> g.effectiveCounts().get(role)).sum(), 1e-12);
        }
        var onlyA = CompatibilityGroup.weighted(List.of(a), PDV).getFirst();
        assertEquals(2.0, onlyA.effectiveCounts().get(RoleType.VIT));
        assertEquals(Math.cbrt(2), onlyA.weight(), 1e-12);
        var completeC = group("c", Map.of(RoleType.PREFILL, Set.of(3L), RoleType.DECODE, Set.of(12L), RoleType.VIT, Set.of(22L)));
        var changed = CompatibilityGroup.weighted(List.of(a, b, completeC), PDV);
        assertEquals(1.0 + 1.0 / 3, changed.get(0).effectiveCounts().get(RoleType.VIT), 1e-12);
        assertEquals(1.0 / 3, changed.get(1).effectiveCounts().get(RoleType.VIT), 1e-12);
        assertEquals(1.0 / 3, changed.get(2).effectiveCounts().get(RoleType.VIT), 1e-12);
    }

    @Test
    void smoothRotationUsesFractionalGeometricWeightsAndAdvancesWithoutOutcomeCallbacks() {
        var a = group("a", Map.of(RoleType.PREFILL, Set.of(1L), RoleType.DECODE, Set.of(10L)));
        var b = group("b", Map.of(RoleType.PREFILL, Set.of(2L, 3L, 4L, 5L), RoleType.DECODE, Set.of(11L, 12L, 13L, 14L)));
        var groups = CompatibilityGroup.weighted(List.of(a, b), PD);
        assertEquals(1.0, groups.get(0).weight(), 1e-12);
        assertEquals(4.0, groups.get(1).weight(), 1e-12);
        var balancer = new CompatibilityGroupBalancer();
        Map<CompatibilityGroup.Key, Integer> counts = new HashMap<>();
        for (int i = 0; i < 500; i++) {
            // No admission/dispatch success is reported: every selection still consumes its turn.
            counts.merge(balancer.select("blue", PD, groups).key(), 1, Integer::sum);
        }
        assertEquals(100, counts.get(a.key()));
        assertEquals(400, counts.get(b.key()));
        var equallyWeighted = CompatibilityGroup.weighted(List.of(a, group("b", Map.of(
                RoleType.PREFILL, Set.of(2L), RoleType.DECODE, Set.of(11L)))), PD);
        var rotation = new CompatibilityGroupBalancer();
        assertNotEquals(rotation.select("blue", PD, equallyWeighted).key(), rotation.select("blue", PD, equallyWeighted).key());
    }

    @Test
    void positiveSmallWeightsSurviveAndPolicyRotationsAreIndependent() {
        assertEquals(1e-250, CompatibilityGroup.geometricMean(1e-200, 1e-300), 1e-262);
        var tiny = group("a", Map.of(RoleType.PREFILL, Set.of(1L), RoleType.DECODE, Set.of(10L)));
        tiny = new CompatibilityGroup(tiny.key(), tiny.members(), Map.of(), 0.001, tiny.id());
        var large = group("b", Map.of(RoleType.PREFILL, Set.of(2L), RoleType.DECODE, Set.of(11L)));
        large = new CompatibilityGroup(large.key(), large.members(), Map.of(), 1.0, large.id());
        var balancer = new CompatibilityGroupBalancer();
        var groups = List.of(tiny, large);
        int tinyTurns = 0;
        for (int i = 0; i < 2002; i++) {
            if (balancer.select("blue", PD, groups).key().equals(tiny.key())) { tinyTurns++; }
        }
        assertEquals(2, tinyTurns);
        assertEquals(new CompatibilityGroupBalancer().select("green", PD, groups).key(),
                balancer.select("green", PD, groups).key());
    }

    @Test
    void membershipsAndInputOrderingChangeWithoutDuplicatingOrResettingStableGroups() {
        var workers = new EnumMap<RoleType, List<WorkerStatus>>(RoleType.class);
        var p = worker(RoleType.PREFILL, "a", 2);
        var d = worker(RoleType.DECODE, "a", 2);
        workers.put(RoleType.PREFILL, new ArrayList<>(List.of(p, p)));
        workers.put(RoleType.DECODE, new ArrayList<>(List.of(d)));
        var first = CompatibilityGroups.build(PD, workers, ignored -> true);
        assertEquals(1, first.size());
        assertEquals(1.0, first.getFirst().effectiveCounts().get(RoleType.PREFILL));
        workers.get(RoleType.PREFILL).add(worker(RoleType.PREFILL, "a", 4));
        var changed = CompatibilityGroups.build(PD, workers, ignored -> true);
        assertEquals(1, changed.size(), "mutually compatible classes form one maximal group");
        assertEquals(2.0, changed.getFirst().effectiveCounts().get(RoleType.PREFILL));
        Collections.reverse(workers.get(RoleType.PREFILL));
        assertEquals(changed, CompatibilityGroups.build(PD, workers, ignored -> true));
        var extra = worker(RoleType.PREFILL, "a", 4);
        workers.get(RoleType.PREFILL).add(extra);
        var resized = CompatibilityGroups.build(PD, workers, ignored -> true);
        assertEquals(changed.getFirst().key(), resized.getFirst().key(), "worker count is not group identity");
        assertEquals(3.0, resized.getFirst().effectiveCounts().get(RoleType.PREFILL));
        assertEquals(changed, CompatibilityGroups.build(PD, workers, id -> id != extra.getGenerationId()));
    }

    @Test
    void nontransitiveClassesProduceOverlappingSoundGroupsWithoutEndpointCartesianProducts() {
        var p4 = worker(RoleType.PREFILL, "target", 4);
        var p6 = worker(RoleType.PREFILL, "target", 6);
        var d2 = worker(RoleType.DECODE, "target", 2);
        var d3 = worker(RoleType.DECODE, "target", 3);
        var vit = worker(RoleType.VIT, "", 1);
        var workers = Map.of(RoleType.PREFILL, List.of(p4, p6), RoleType.DECODE, List.of(d2, d3), RoleType.VIT, List.of(vit));
        var groups = CompatibilityGroups.build(PDV, workers, ignored -> true);
        assertEquals(2, groups.size());
        assertTrue(groups.stream().allMatch(g -> g.contains(RoleType.PREFILL, p6.getGenerationId())
                && g.contains(RoleType.DECODE, d2.getGenerationId()) && g.contains(RoleType.VIT, vit.getGenerationId())));
        assertTrue(groups.stream().noneMatch(g -> g.contains(RoleType.PREFILL, p4.getGenerationId())
                && g.contains(RoleType.DECODE, d3.getGenerationId())));
        for (RoleType role : PDV) {
            assertEquals(workers.get(role).size(), groups.stream().mapToDouble(g -> g.effectiveCounts().get(role)).sum(), 1e-12);
        }
        var manyIdentical = new ArrayList<WorkerStatus>();
        for (int i = 0; i < 500; i++) { manyIdentical.add(worker(RoleType.PREFILL, "target", 4)); }
        var one = CompatibilityGroups.build(PD, Map.of(RoleType.PREFILL, manyIdentical, RoleType.DECODE, List.of(d2)), ignored -> true);
        assertEquals(1, one.size());
        assertEquals(500, one.getFirst().members().get(RoleType.PREFILL).size());
    }

    @Test
    void concurrentSelectionsConsumeExactlyOneTurnEach() throws Exception {
        var groups = CompatibilityGroup.weighted(List.of(
                group("a", Map.of(RoleType.PREFILL, Set.of(1L), RoleType.DECODE, Set.of(10L))),
                group("b", Map.of(RoleType.PREFILL, Set.of(2L), RoleType.DECODE, Set.of(11L)))), PD);
        var balancer = new CompatibilityGroupBalancer();
        var counts = new java.util.concurrent.ConcurrentHashMap<CompatibilityGroup.Key, Integer>();
        try (var pool = Executors.newFixedThreadPool(8)) {
            for (int i = 0; i < 1000; i++) { pool.submit(() -> counts.merge(balancer.select("blue", PD, groups).key(), 1, Integer::sum)); }
            pool.shutdown();
            assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));
        }
        assertEquals(2, counts.size());
        counts.values().forEach(count -> assertEquals(500, count));
    }

    @Test
    void mutableInputAndChangedWeightsKeepRotationFresh() {
        var a = group("a", Map.of(RoleType.PREFILL, Set.of(1L), RoleType.DECODE, Set.of(10L)));
        var b = group("b", Map.of(RoleType.PREFILL, Set.of(2L), RoleType.DECODE, Set.of(11L)));
        var groups = new ArrayList<>(CompatibilityGroup.weighted(List.of(a, b), PD));
        var balancer = new CompatibilityGroupBalancer();
        var first = balancer.select("blue", PD, groups);
        int index = groups.indexOf(first);
        groups.set(index, new CompatibilityGroup(first.key(), first.members(), first.effectiveCounts(), 0.01, first.id()));
        int smallTurns = 0;
        for (int i = 0; i < 1010; i++) {
            if (balancer.select("blue", PD, groups).key().equals(first.key())) { smallTurns++; }
        }
        assertTrue(smallTurns >= 9 && smallTurns <= 11, "new weights must apply to a reused input list");
        groups.clear();
        assertNull(balancer.select("blue", PD, groups));
        groups.addAll(CompatibilityGroup.weighted(List.of(a, b), PD));
        assertNotEquals(balancer.select("blue", PD, groups).key(), balancer.select("blue", PD, groups).key());
    }

    @Test
    void resizingBothGroupsPreservesAlternationWithoutOldScaleBursts() {
        var large = rotationGroups(10.0, 10.0);
        var small = rotationGroups(1.0, 1.0);
        var balancer = new CompatibilityGroupBalancer();
        assertEquals(large.get(0).key(), balancer.select("blue", PD, large).key());
        for (int i = 0; i < 20; i++) {
            assertEquals(small.get((i + 1) % 2).key(), balancer.select("blue", PD, small).key());
        }
    }

    @Test
    void changedRatioRetainsCreditInSelectionsInsteadOfResettingRotation() {
        var initial = rotationGroups(3.0, 1.0);
        var balancer = new CompatibilityGroupBalancer();
        assertEquals(initial.get(0).key(), balancer.select("blue", PD, initial).key());
        var equal = rotationGroups(1.0, 1.0);
        assertEquals(equal.get(1).key(), balancer.select("blue", PD, equal).key());
        assertEquals(equal.get(0).key(), balancer.select("blue", PD, equal).key());
    }

    @Test
    void changingRatiosFollowCumulativeTargetSharesDespiteResizingAndInputReordering() {
        var identities = rotationGroups(1.0, 1.0, 1.0);
        long[] targetQuarters = new long[3];
        long[] selections = new long[3];
        var balancer = new CompatibilityGroupBalancer();
        for (int turn = 0; turn < 1200; turn++) {
            int preferred = (turn / 7) % 3;
            double scale = Math.scalb(1.0, (turn % 19 - 9) * 100);
            List<CompatibilityGroup> groups = new ArrayList<>();
            int expected = 0;
            long largestDeficit = Long.MIN_VALUE;
            for (int i = 0; i < 3; i++) {
                int weight = i == preferred ? 2 : 1;
                targetQuarters[i] += weight;
                long deficit = targetQuarters[i] - 4 * selections[i];
                if (deficit > largestDeficit) { largestDeficit = deficit; expected = i; }
                groups.add(withWeight(identities.get(i), weight * scale));
            }
            if (turn % 2 == 0) { Collections.reverse(groups); }
            assertEquals(identities.get(expected).key(), balancer.select("blue", PD, groups).key(), "turn " + turn);
            selections[expected]++;
        }
    }

    @Test
    void addingGroupStartsAtZeroWithoutResettingSurvivingCredits() {
        var groups = rotationGroups(1.0, 1.0, 1.0);
        var balancer = new CompatibilityGroupBalancer();
        assertEquals(groups.get(0).key(), balancer.select("blue", PD, groups.subList(0, 2)).key());
        assertEquals(groups.get(1).key(), balancer.select("blue", PD, groups).key());
        assertEquals(groups.get(2).key(), balancer.select("blue", PD, groups).key());
    }

    @Test
    void removingGroupsPreservesRelativeCreditOfSurvivors() {
        var groups = rotationGroups(2.0, 1.0, 1.0, 1.0);
        var balancer = new CompatibilityGroupBalancer();
        assertEquals(groups.get(0).key(), balancer.select("blue", PD, groups).key());
        assertEquals(groups.get(3).key(), balancer.select("blue", PD, List.of(groups.get(0), groups.get(3))).key());
    }

    @Test
    void replacingOtherGroupsCentersSurvivorsBeforeAddingNewcomer() {
        var groups = rotationGroups(1.0, 1.0, 1.0, 1.0);
        var balancer = new CompatibilityGroupBalancer();
        assertEquals(groups.get(0).key(), balancer.select("blue", PD, groups.subList(0, 3)).key());
        assertEquals(groups.get(0).key(), balancer.select("blue", PD, List.of(groups.get(0), groups.get(3))).key());
    }

    @Test
    void returningGroupStartsAtZeroAndRemovedCreditDoesNotBiasSurvivor() {
        var groups = rotationGroups(1.0, 1.0);
        var balancer = new CompatibilityGroupBalancer();
        assertEquals(groups.get(0).key(), balancer.select("blue", PD, groups).key());
        assertEquals(groups.get(1).key(), balancer.select("blue", PD, List.of(groups.get(1))).key());
        assertEquals(groups.get(0).key(), balancer.select("blue", PD, groups).key());
    }

    @Test
    void overflowingWeightSumStillProducesFiniteNormalizedShares() {
        var groups = rotationGroups(Double.MAX_VALUE, Double.MAX_VALUE);
        var balancer = new CompatibilityGroupBalancer();
        for (int i = 0; i < 20; i++) {
            assertEquals(groups.get(i % 2).key(), balancer.select("blue", PD, groups).key());
        }
    }

    @Test
    void telemetryIdentityDistinguishesAbsentPolicyFromLiteralNullAndEscapesIdentityFields() {
        var a = new CompatibilityGroup.Key(null, Map.of(RoleType.PREFILL, Set.of(descriptor(RoleType.PREFILL, "a|b=c", 2))));
        var b = new CompatibilityGroup.Key("null", a.descriptors());
        assertNotEquals(a.canonical(), b.canonical());
        assertNotEquals(CompatibilityGroup.unweighted(a, Map.of()).id(), CompatibilityGroup.unweighted(b, Map.of()).id());
        assertEquals(a.canonical(), new CompatibilityGroup.Key(null, a.descriptors()).canonical());
    }

    @Test
    void legacyIsolationAlsoAppliesToSingleRoleTopologies() {
        var legacy = WorkerStatus.createDiscovered(RoleType.PDFUSION, "blue", "legacy", 8080, 8081, "test");
        var reporting = worker(RoleType.PDFUSION, "target", 2);
        var roles = List.of(RoleType.PDFUSION);
        var groups = CompatibilityGroups.build(roles, Map.of(RoleType.PDFUSION, List.of(legacy, reporting)), ignored -> true);
        assertEquals(2, groups.size());
        assertTrue(groups.stream().allMatch(g -> g.members().get(RoleType.PDFUSION).size() == 1));
        assertTrue(groups.stream().allMatch(g -> g.weight() == 1.0));
    }

    private static List<CompatibilityGroup> rotationGroups(double... weights) {
        List<CompatibilityGroup> groups = new ArrayList<>();
        for (int i = 0; i < weights.length; i++) {
            groups.add(group("rotation-" + i, Map.of(RoleType.PREFILL, Set.of(2L * i + 1),
                    RoleType.DECODE, Set.of(2L * i + 2))));
        }
        groups.sort(Comparator.comparing(CompatibilityGroup::id));
        for (int i = 0; i < weights.length; i++) { groups.set(i, withWeight(groups.get(i), weights[i])); }
        return List.copyOf(groups);
    }

    @Test
    void policyChurnReleasesOldPlansWhileKeepingRecentlyUsedCredit() {
        var groups = rotationGroups(1.0, 1.0);
        var retired = scopedGroups("retired", groups);
        var balancer = new CompatibilityGroupBalancer();
        assertEquals(retired.getFirst(), balancer.select("retired", PD, retired));
        for (int i = 0; i < 192; i++) {
            assertEquals(groups.get(i % 2), balancer.select("blue", PD, groups), "active scope must retain credit");
            String policy = "policy-" + i;
            var next = scopedGroups(policy, groups);
            assertEquals(next.getFirst(), balancer.select(policy, PD, next));
            assertTrue(rotations(balancer).size() <= 64, "old policy plans must be evicted");
        }
        assertEquals(64, rotations(balancer).size());
        assertTrue(rotations(balancer).values().stream()
                .noneMatch(rotation -> ReflectionTestUtils.getField(rotation, "source") == retired));
        assertEquals(retired.getFirst(), balancer.select("retired", PD, retired), "returning scope starts at zero");
    }

    @Test
    void emptyScopeReleasesItsPlanAndReturningScopeStartsAtZero() {
        var groups = rotationGroups(1.0, 1.0);
        var balancer = new CompatibilityGroupBalancer();
        assertNull(balancer.select(null, PD, List.of()));
        assertTrue(rotations(balancer).isEmpty());
        assertEquals(groups.getFirst(), balancer.select(null, PD, groups));
        assertEquals(groups.getFirst(), balancer.select("blue", PD, groups));
        assertNull(balancer.select(null, PD, List.of()));
        assertEquals(1, rotations(balancer).size());
        assertEquals(groups.get(1), balancer.select("blue", PD, groups), "other scope retains credit");
        assertEquals(groups.getFirst(), balancer.select(null, PD, groups), "empty scope discarded credit");
    }

    private static Map<?, ?> rotations(CompatibilityGroupBalancer balancer) {
        Object retained = ReflectionTestUtils.getField(balancer, "rotations");
        if (retained instanceof Cache<?, ?> cache) {
            // Caffeine applies its size bound during maintenance, as in the planner.
            cache.cleanUp();
            return cache.asMap();
        }
        return (Map<?, ?>) retained;
    }

    private static List<CompatibilityGroup> scopedGroups(String policy, List<CompatibilityGroup> groups) {
        return groups.stream().map(group -> withWeight(CompatibilityGroup.unweighted(
                new CompatibilityGroup.Key(policy, group.key().descriptors()), group.members()), group.weight()))
                .sorted(Comparator.comparing(CompatibilityGroup::id)).toList();
    }

    private static CompatibilityGroup withWeight(CompatibilityGroup group, double weight) {
        return new CompatibilityGroup(group.key(), group.members(), group.effectiveCounts(), weight, group.id());
    }

    private static CompatibilityGroup group(String target, Map<RoleType, Set<Long>> members) {
        return CompatibilityGroup.unweighted(new CompatibilityGroup.Key("blue", Map.of(
                RoleType.PREFILL, Set.of(descriptor(RoleType.PREFILL, target, 2)))), members);
    }

    private static WorkerStatus worker(RoleType role, String target, int tp) {
        var status = WorkerStatus.createDiscovered(role, "blue", "worker", 8080, 8081, "test");
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setStatusVersion(1L);
        response.setCompatibility(descriptor(role, target, tp));
        status.lock.lock();
        try { status.publishPreparedStatus(status.prepareNewStatus(status.freezeStatusResponse(response))); }
        finally { status.lock.unlock(); }
        return status;
    }

    private static RoleCompatibility descriptor(RoleType role, String target, int tp) {
        var vision = new RoleCompatibility.Vision("vision", "embedding");
        if (role == RoleType.VIT) { return new RoleCompatibility(1, "", vision, null, null); }
        int heads = 12 / tp;
        var group = new RoleCompatibility.Group("target/default", List.of(0), "MultiHeadAttention", "bf16", "split",
                64, 64, heads, List.of((long) heads * 8192, (long) heads * 8192, 0L, 0L), "full", "head-dim=64");
        return new RoleCompatibility(1, target, vision, new RoleCompatibility.Transfer("cache-store-tcp-v1",
                "model-token-layer-tag-v1", 0, tp, 1, 64, List.of(group), tp, "tp"),
                new RoleCompatibility.Speculative("none", "", 1));
    }
}
