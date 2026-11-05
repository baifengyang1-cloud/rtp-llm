package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CompatibilityCandidates;
import org.flexlb.balance.strategy.CompatibilityPlanner;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.CompatibilityGroupReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class CompatibilityRoutingTest {
    private static final List<RoleType> ROLES = List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT);
    private static final RoleCompatibility.Vision VISION = new RoleCompatibility.Vision("vision", "embedding-v1");

    private static RoleCompatibility descriptor(RoleType role, String target, int block, int tp) {
        if (role == RoleType.VIT) { return new RoleCompatibility(1, "", VISION, null, null); }
        int heads = 12 / tp;
        var group = new RoleCompatibility.Group("target/default", List.of(0), "MultiHeadAttention", "bf16", "split",
                block, block, heads, List.of((long) block * heads * 128, (long) block * heads * 128, 0L, 0L),
                "full", "head-dim=64");
        return new RoleCompatibility(1, target, VISION,
                new RoleCompatibility.Transfer("cache-store-tcp-v1", "model-token-layer-tag-v1", 0,
                        tp, 1, block, List.of(group), tp, "tp"), new RoleCompatibility.Speculative("none", "", 1));
    }

    @ParameterizedTest
    @CsvSource({"true,false,false,false", "false,false,false,false", "false,true,false,false",
            "false,false,true,false", "false,true,true,false", "false,false,false,true", "false,true,false,true",
            "false,false,true,true", "false,true,true,true"})
    void onlyCompleteCompatibleRoutesAreCommittedAndCancellationSettlesOwnership(
            boolean direct, boolean priority, boolean window, boolean batch) throws Exception {
        try (Fleet fleet = new Fleet(direct, priority, window, batch)) {
            var wrongP = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "incomplete", 64, 2));
            var p = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "target", 128, 4));
            var d = fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 128, 2));
            var v = fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            // A shared ViT cannot connect incompatible P/D descriptors.
            fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "different", 64, 2));
            var ctx = RequestLifecycleTestSupport.context(fleet.config, 1001L);
            ctx.getRequest().setSeqLen(32L);
            ctx.getRequest().setMaxNewTokens(8);
            var response = fleet.scheduler.submit(ctx).get(3, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), response.getErrorMessage());
            assertEquals(List.of(p.ipPort(), d.ipPort(), v.ipPort()), response.getServerStatus().stream()
                    .map(status -> status.getServerIp() + ":" + status.getHttpPort()).toList());
            assertEquals(0, ((PrefillEndpoint) wrongP).observedRequestCount());
            assertEquals(1, fleet.requests.liveRequestCount());
            var state = fleet.requests.getRequestState(1001L, 0L);
            assertNotNull(state);
            fleet.requests.cancelRequest(1001L, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(0, fleet.scheduler.getQueuedRequestCount());
        }
    }

    @Test
    void legacyAndReportedWorkersCannotMixOrEscapeSelectedPolicyGroup() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            fleet.add(RoleType.PREFILL, "blue", null);
            fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 64, 2));
            fleet.add(RoleType.VIT, "blue", null);
            for (RoleType role : ROLES) { fleet.add(role, "green", null); }
            var ctx = RequestLifecycleTestSupport.context(fleet.config, 1002L);
            var selection = fleet.router.select(ctx, "blue");
            assertEquals(PlacementResult.Status.BLOCKED, selection.status());
            assertEquals(3, selection.blocker().dependencies().size());
            assertEquals(0, fleet.requests.liveRequestCount());
        }
    }

    @Test
    void textOnlyInitialPlacementStillRequiresConfiguredVit() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            fleet.add(RoleType.PREFILL, "blue", null);
            fleet.add(RoleType.DECODE, "blue", null);
            var ctx = RequestLifecycleTestSupport.context(fleet.config, 1003L);
            assertEquals(PlacementResult.Status.BLOCKED, fleet.router.select(ctx, "blue").status());
            fleet.add(RoleType.VIT, "blue", null);
            try (RouteAdmission admission = fleet.router.select(ctx, "blue").value()) {
                assertNotNull(admission);
            }
        }
    }

    @Test
    void dispatchReplacementKeepsVisionIdentityAndPolicyBoundary() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            var original = descriptor(RoleType.VIT, "", 0, 0);
            var retired = fleet.add(RoleType.VIT, "blue", original);
            fleet.add(RoleType.VIT, "blue", new RoleCompatibility(1, "",
                    new RoleCompatibility.Vision("new-vision", "embedding-v1"), null, null));
            fleet.add(RoleType.VIT, "blue", null);
            fleet.add(RoleType.VIT, "green", original);
            retired.close();
            var strategy = new RandomStrategy(fleet.workers);
            var context = RequestLifecycleTestSupport.context(fleet.config, 3001L);
            assertNull(strategy.reselectCompatible(context, "blue", original));
            var replacement = fleet.add(RoleType.VIT, "blue", original);
            try (var selected = strategy.reselectCompatible(context, "blue", original)) {
                assertNotNull(selected);
                assertSame(replacement, selected.endpoint());
            }
        }
    }

    @Test
    void retirementBetweenSelectionAndAdmissionRejectsRouteAndReleasesProvisionalPins() throws Exception {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            var p = fleet.add(RoleType.PREFILL, "blue", null);
            fleet.add(RoleType.DECODE, "blue", null);
            fleet.add(RoleType.VIT, "blue", null);
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            var selector = spy(new CostBasedPrefillStrategy(fleet.workers, cache, mock(EngineHealthReporter.class)));
            var detached = new java.util.concurrent.atomic.AtomicReference<EndpointRegistry.DetachedGeneration>();
            doAnswer(call -> {
                var selected = call.callRealMethod();
                p.getStatus().lock.lock();
                try { detached.set(fleet.workers.beginRetirement(RoleType.PREFILL, p.ipPort(), p.getStatus())); }
                finally { p.getStatus().lock.unlock(); }
                return selected;
            }).when(selector).select(any(), any(), any(), any());
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(ROLES);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(fleet.config);
            var router = new DefaultRouter(selector, new DecodeSelector(fleet.workers), new RandomStrategy(fleet.workers),
                    service, model, fleet.workers);
            assertEquals(PlacementResult.Status.BLOCKED,
                    router.select(RequestLifecycleTestSupport.context(fleet.config, 3002L), "blue").status());
            java.util.concurrent.CompletableFuture.runAsync(() -> fleet.workers.completeRetirement(RoleType.PREFILL,
                    p.ipPort(), p.getStatus(), detached.get(), cache, org.slf4j.LoggerFactory.getLogger(getClass())))
                    .get(3, TimeUnit.SECONDS);
            assertEquals(0, ((PrefillEndpoint) p).observedRequestCount());
            assertEquals(0, fleet.requests.liveRequestCount());
        }
    }

    @Test
    void overlappingMembershipRetainsAlternativesWithoutDuplicatingSharedVit() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            var p4 = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "target", 64, 4));
            var p6 = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "target", 64, 6));
            var d2 = fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 64, 2));
            var d3 = fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 64, 3));
            var vit = fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            var candidates = new CompatibilityCandidates(fleet.workers, List.of(RoleType.VIT, RoleType.PREFILL, RoleType.DECODE), "blue");
            assertTrue(candidates.select(RoleType.VIT, vit.getStatus()));
            assertTrue(candidates.filter(RoleType.PREFILL).test(p4.getStatus().getGenerationId()));
            assertTrue(candidates.filter(RoleType.PREFILL).test(p6.getStatus().getGenerationId()));
            assertTrue(candidates.select(RoleType.PREFILL, p4.getStatus()));
            assertTrue(candidates.filter(RoleType.DECODE).test(d2.getStatus().getGenerationId()));
            assertFalse(candidates.filter(RoleType.DECODE).test(d3.getStatus().getGenerationId()));
            assertEquals(1, fleet.workers.routingStatusSnapshot(RoleType.VIT, "blue").size());
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false,false", "true,false,true", "false,true,true", "true,true,false"})
    void waitingForCompatibilityRecoversOnAnotherRoleAndCancellationCleansWaiters(
            boolean priority, boolean window, boolean batch) throws Exception {
        try (Fleet fleet = new Fleet(false, priority, window, batch)) {
            var p = fleet.add(RoleType.PREFILL, "blue", null);
            fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 64, 2));
            fleet.add(RoleType.VIT, "blue", null);
            var waiting = fleet.scheduler.submit(RequestLifecycleTestSupport.context(fleet.config, 2001L));
            assertFalse(waiting.isDone());
            var d = fleet.add(RoleType.DECODE, "blue", null);
            var route = waiting.get(3, TimeUnit.SECONDS);
            assertTrue(route.isSuccess(), route.getErrorMessage());
            assertEquals(d.getIp(), route.getServerStatus().get(1).getServerIp());
            assertEquals(p.getIp(), route.getServerStatus().get(0).getServerIp());
            var blocked = RequestLifecycleTestSupport.context(fleet.config, 2002L);
            // A separate policy boundary has no complete group and may be cancelled.
            var policy = mock(org.flexlb.config.TrafficPolicyConfig.class);
            when(policy.resolveTargetGroup(blocked.getRequest())).thenReturn(java.util.Optional.of("empty"));
            blocked.getConfig().getRouter().setGroupSelector(policy);
            var cancelled = fleet.scheduler.submit(blocked);
            fleet.requests.cancelRequest(2002L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(cancelled.get(3, TimeUnit.SECONDS).isSuccess());
        }
    }

    @ParameterizedTest
    @CsvSource({"true,false,false,false", "false,false,false,false", "false,true,false,false",
            "false,false,true,false", "false,true,true,false", "false,false,false,true", "false,true,false,true",
            "false,false,true,true", "false,true,true,true"})
    void coldGroupsReceiveTurnsBeforeWorkerCostAndSharedVitKeepsPhysicalAccounting(
            boolean direct, boolean priority, boolean window, boolean batch) throws Exception {
        try (Fleet fleet = new Fleet(direct, priority, window, batch)) {
            fleet.config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens)");
            var warm = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "warm", 64, 2));
            var cold = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "cold", 64, 2));
            var warmD = (DecodeEndpoint) fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "warm", 64, 2));
            var coldD = (DecodeEndpoint) fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "cold", 64, 2));
            fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            when(fleet.cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(warm.ipPort(), 7));
            Map<String, Integer> turns = new HashMap<>();
            for (int i = 0; i < 20; i++) {
                var context = RequestLifecycleTestSupport.context(fleet.config, 4000L + i);
                context.getRequest().setSeqLen(512L);
                context.getRequest().setMaxNewTokens(8);
                context.getRequest().setBlockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L));
                context.getRequest().setCacheKeyBlockSize(64L);
                try (var admission = fleet.router.select(context, "blue").value()) {
                    assertNotNull(admission);
                    var status = admission.createScheduledRequest(context, new java.util.concurrent.CompletableFuture<>(),
                            System.currentTimeMillis()).prefill();
                    turns.merge(admission.prefillEndpoint().ipPort(), 1, Integer::sum);
                    boolean cached = admission.prefillEndpoint() == warm;
                    assertEquals(cached ? 64L : 512L, status.getPrefillTime());
                    assertEquals(cached ? 448L : 0L, status.getDebugInfo().getHitCacheLen());
                    assertSame(cached ? warmD : coldD, admission.decodeEndpoint());
                }
            }
            assertEquals(Map.of(warm.ipPort(), 10, cold.ipPort(), 10), turns);
            assertEquals(0, warmD.resourceSnapshot().queuedCount());
            assertEquals(0, coldD.resourceSnapshot().queuedCount());
            assertEquals(0, fleet.requests.liveRequestCount());
            // Actual admission and delivery still charge each selected physical endpoint once.
            for (int i = 0; i < 2; i++) {
                var context = RequestLifecycleTestSupport.context(fleet.config, 4100L + i);
                context.getRequest().setMaxNewTokens(8);
                var response = fleet.scheduler.submit(context).get(3, TimeUnit.SECONDS);
                assertTrue(response.isSuccess(), response.getErrorMessage());
                var selectedD = response.getServerStatus().get(1).getServerIp().equals(warmD.getIp()) ? warmD : coldD;
                assertNotNull(selectedD.reservationHandle(4100L + i));
                assertNull((selectedD == warmD ? coldD : warmD).reservationHandle(4100L + i));
            }
            assertEquals(1, fleet.workers.routingStatusSnapshot(RoleType.VIT, "blue").size());
            verify(fleet.monitor, times(22)).report(eq(CompatibilityGroupReporter.SELECTION), any(), eq(1.0));
            verify(fleet.monitor, times(22)).report(eq(CompatibilityGroupReporter.WORKER_SELECTION), any(), eq(1.0));
            verify(fleet.monitor, timeout(2000).times(22)).report(eq(CompatibilityGroupReporter.ADMISSION), any(), eq(1.0));
            verify(fleet.monitor, timeout(2000).times(2)).report(eq(CompatibilityGroupReporter.DISPATCH), any(), eq(1.0));
        }
    }

    @Test
    void workerSelectionFailureConsumesTurnWithoutAttemptingAnotherGroup() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            var pA = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "a", 64, 2));
            var pB = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "b", 64, 2));
            fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "a", 64, 2));
            fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "b", 64, 2));
            fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            var selector = spy(fleet.prefillSelector);
            var failed = new java.util.concurrent.atomic.AtomicReference<WorkerEndpoint>();
            doAnswer(call -> {
                if (failed.get() != null) { return call.callRealMethod(); }
                var scope = call.getArgument(3, java.util.function.LongPredicate.class);
                failed.set(scope.test(pA.getStatus().getGenerationId()) ? pA : pB);
                return PlacementResult.blocked(RoleType.PREFILL);
            }).when(selector).select(any(), any(), any(), any());
            var router = fleet.newRouter(selector);
            assertEquals(PlacementResult.Status.BLOCKED,
                    router.select(RequestLifecycleTestSupport.context(fleet.config, 4200L), "blue").status());
            verify(selector, times(1)).select(any(), any(), any(), any());
            try (var admission = router.select(RequestLifecycleTestSupport.context(fleet.config, 4201L), "blue").value()) {
                assertNotNull(admission);
                assertNotSame(failed.get(), admission.prefillEndpoint());
            }
            assertEquals(0, ((PrefillEndpoint) pA).observedRequestCount());
            assertEquals(0, ((PrefillEndpoint) pB).observedRequestCount());
            verify(fleet.monitor, times(2)).report(eq(CompatibilityGroupReporter.SELECTION), any(), eq(1.0));
            verify(fleet.monitor, never()).report(eq(CompatibilityGroupReporter.DISPATCH), any(), anyDouble());
        }
    }

    @Test
    void admissionFailureDoesNotRefundTurnOrReleaseAnotherRequestsPhysicalReservation() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            fleet.config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            for (String target : List.of("a", "b")) {
                fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, target, 64, 2));
                fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, target, 64, 2));
            }
            fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            var context = RequestLifecycleTestSupport.context(fleet.config, 4300L);
            context.getRequest().setMaxNewTokens(8);
            context.setFuture(fleet.requests.register(context));
            PrefillEndpoint first;
            try (var admission = fleet.router.select(context, "blue").value()) {
                assertNotNull(admission);
                first = admission.prefillEndpoint();
                var decode = admission.decodeEndpoint();
                DecodeEndpoint.ReservationHandle occupant;
                try (var pin = decode.tryPinGeneration()) { occupant = decode.reserve(pin, 999L, 16L, 24L, 50); }
                assertNotNull(occupant);
                var acquired = decode.acquireDispatchPermit(occupant, admission.decodeBinding().capacity());
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
                try {
                    assertEquals(PlacementResult.Status.BLOCKED, admission.tryCommitDirectRoute(context, fleet.requests).status());
                    admission.close();
                    assertNull(decode.reservationHandle(4300L));
                    assertEquals(occupant, decode.reservationHandle(999L));
                    assertEquals(1, decode.routingView().engineCapacityUsed());
                    assertEquals(0, first.observedRequestCount());
                } finally {
                    acquired.permit().release();
                    decode.release(occupant, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
                }
            }
            fleet.requests.cancelRequest(4300L, 0L, CancelReason.CLIENT_CANCELLED);
            try (var admission = fleet.router.select(RequestLifecycleTestSupport.context(fleet.config, 4301L), "blue").value()) {
                assertNotNull(admission);
                assertNotSame(first, admission.prefillEndpoint());
            }
            verify(fleet.monitor, times(2)).report(eq(CompatibilityGroupReporter.SELECTION), any(), eq(1.0));
            verify(fleet.monitor, times(2)).report(eq(CompatibilityGroupReporter.ADMISSION), any(), eq(1.0));
            verify(fleet.monitor, never()).report(eq(CompatibilityGroupReporter.DISPATCH), any(), anyDouble());
            assertEquals(0, fleet.requests.liveRequestCount());
        }
    }

    @ParameterizedTest
    @CsvSource({"true,false", "false,false", "false,true"})
    void groupEligibilityUsesExistingImmediateQueuedAndPreemptibleDecodeModes(boolean direct, boolean preempt) {
        try (Fleet fleet = new Fleet(direct, preempt, false, false)) {
            if (preempt) { SchedulingTestConfig.preemption(fleet.config); }
            fleet.config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "a", 64, 2));
            var busy = (DecodeEndpoint) fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "a", 64, 2));
            fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "b", 64, 2));
            fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "b", 64, 2));
            fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            var context = RequestLifecycleTestSupport.context(fleet.config, 4400L);
            context.getRequest().setMaxNewTokens(8);
            var binding = ScheduledRequest.DecodeBinding.capture(context);
            assertEquals(direct ? ScheduledRequest.DecodeMode.IMMEDIATE : preempt
                    ? ScheduledRequest.DecodeMode.PREEMPT_AT_PLACEMENT : ScheduledRequest.DecodeMode.WAIT_AT_PLACEMENT,
                    binding.mode());
            DecodeEndpoint.ReservationHandle occupant;
            try (var pin = busy.tryPinGeneration()) { occupant = busy.reserve(pin, 999L, 16L, 24L, 10); }
            var acquired = busy.acquireDispatchPermit(occupant, binding.capacity());
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
            try {
                var candidates = new CompatibilityCandidates(fleet.workers, ROLES, "blue");
                var eligible = candidates.eligibleGenerations(context, binding);
                var groups = candidates.completeGroups(eligible::contains);
                assertEquals(direct ? 1 : 2, groups.size());
                assertEquals(!direct, eligible.contains(busy.getStatus().getGenerationId()));
                assertTrue(groups.stream().allMatch(g -> g.effectiveCounts().get(RoleType.VIT) == (direct ? 1.0 : 0.5)));
                int busyTurns = 0;
                for (int i = 0; i < 10; i++) {
                    try (var admission = fleet.router.select(context, "blue").value()) {
                        assertNotNull(admission);
                        if (admission.decodeEndpoint() == busy) { busyTurns++; }
                    }
                }
                assertEquals(direct ? 0 : 5, busyTurns);
                assertEquals(occupant, busy.reservationHandle(999L));
                assertEquals(1, busy.routingView().engineCapacityUsed());
            } finally {
                acquired.permit().release();
                busy.release(occupant, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
            }
        }
    }

    @Test
    void decodeFirstRoutingRetainsRoleOrderAndRotatesDespiteConsistentlyWorseDecodeCost() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            for (String target : List.of("warm", "cold")) {
                fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, target, 64, 2));
            }
            var warm = fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "warm", 64, 2));
            var cold = fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "cold", 64, 2), 50000L);
            fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            var order = List.of(RoleType.DECODE, RoleType.PREFILL, RoleType.VIT);
            var router = fleet.newRouter(fleet.prefillSelector, order);
            Map<String, Integer> turns = new HashMap<>();
            for (int i = 0; i < 20; i++) {
                var context = RequestLifecycleTestSupport.context(fleet.config, 4500L + i);
                context.getRequest().setMaxNewTokens(8);
                try (var admission = router.select(context, "blue").value()) {
                    assertNotNull(admission);
                    turns.merge(admission.decodeEndpoint().ipPort(), 1, Integer::sum);
                    var item = admission.createScheduledRequest(context, new java.util.concurrent.CompletableFuture<>(),
                            System.currentTimeMillis());
                    assertEquals(order, item.routeResponse().getServerStatus().stream().map(status -> status.getRole()).toList());
                }
            }
            assertEquals(Map.of(warm.ipPort(), 10, cold.ipPort(), 10), turns);
        }
    }

    @Test
    void cachedGroupsRecomputeCompletenessAndSharedContributionsFromLiveEligibility() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            fleet.config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            var p4 = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "target", 64, 4));
            var p6 = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "target", 64, 6));
            var d2 = (DecodeEndpoint) fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 64, 2));
            var d3 = fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 64, 3));
            fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            var planner = new CompatibilityPlanner(fleet.workers, ROLES);
            var context = RequestLifecycleTestSupport.context(fleet.config, 4600L);
            var binding = ScheduledRequest.DecodeBinding.capture(context);
            var original = planner.prepare(context, binding, "blue").completeGroups();
            assertEquals(2, original.size());
            assertEquals(1.0, original.stream().mapToDouble(g -> g.effectiveCounts().get(RoleType.VIT)).sum());
            DecodeEndpoint.ReservationHandle occupant;
            try (var pin = d2.tryPinGeneration()) { occupant = d2.reserve(pin, 999L, 16L, 24L, 50); }
            var acquired = d2.acquireDispatchPermit(occupant, binding.capacity());
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
            try {
                var candidates = planner.prepare(context, binding, "blue");
                var groups = candidates.completeGroups();
                assertEquals(1, groups.size());
                var group = groups.getFirst();
                assertFalse(group.contains(RoleType.PREFILL, p4.getStatus().getGenerationId()));
                assertTrue(group.contains(RoleType.PREFILL, p6.getStatus().getGenerationId()));
                assertTrue(group.contains(RoleType.DECODE, d3.getStatus().getGenerationId()));
                assertEquals(1.0, group.weight());
                assertEquals(1.0, group.effectiveCounts().get(RoleType.VIT));
                candidates.restrictTo(group);
                assertFalse(candidates.filter(RoleType.DECODE).test(d2.getStatus().getGenerationId()));
                assertEquals(occupant, d2.reservationHandle(999L));
            } finally {
                acquired.permit().release();
                d2.release(occupant, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
            }
            assertEquals(original, planner.prepare(context, binding, "blue").completeGroups());
        }
    }

    @Test
    void cachedScopesTrackIncomingPolicyLabelsAndRejectReplacedGenerations() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            var p = fleet.add(RoleType.PREFILL, "blue", descriptor(RoleType.PREFILL, "target", 64, 2));
            var incoming = fleet.add(RoleType.PREFILL, "green", descriptor(RoleType.PREFILL, "target", 64, 2));
            var d = fleet.add(RoleType.DECODE, "blue", descriptor(RoleType.DECODE, "target", 64, 2));
            fleet.add(RoleType.VIT, "blue", descriptor(RoleType.VIT, "", 0, 0));
            var planner = new CompatibilityPlanner(fleet.workers, ROLES);
            var context = RequestLifecycleTestSupport.context(fleet.config, 4700L);
            var binding = ScheduledRequest.DecodeBinding.capture(context);
            var old = planner.prepare(context, binding, "blue");
            var original = old.completeGroups().getFirst();
            assertEquals(old.completeGroups(), old.completeGroups(generation -> true));
            assertTrue(old.eligibleGenerations(context, binding).contains(d.getStatus().getGenerationId()));
            assertTrue(old.anyEligible(RoleType.PREFILL, java.util.Set.of(p.getStatus().getGenerationId())));
            old.restrictTo(original);
            assertEquals(1.0, original.effectiveCounts().get(RoleType.PREFILL));
            updateGroup(incoming, "blue");
            var expanded = planner.prepare(context, binding, "blue").completeGroups().getFirst();
            assertEquals(2.0, expanded.effectiveCounts().get(RoleType.PREFILL));
            assertEquals(original.key(), expanded.key());
            assertFalse(old.filter(RoleType.PREFILL).test(incoming.getStatus().getGenerationId()));
            updateGroup(p, "green");
            assertFalse(old.filter(RoleType.PREFILL).test(p.getStatus().getGenerationId()));
            updateGroup(incoming, "green");
            assertTrue(planner.prepare(context, binding, "blue").completeGroups().isEmpty());
            updateGroup(incoming, "blue");
            assertEquals(1.0, planner.prepare(context, binding, "blue").completeGroups().getFirst()
                    .effectiveCounts().get(RoleType.PREFILL));
            EndpointRegistry.DetachedGeneration detached;
            d.getStatus().lock.lock();
            try { detached = fleet.workers.beginRetirement(RoleType.DECODE, d.ipPort(), d.getStatus()); }
            finally { d.getStatus().lock.unlock(); }
            assertFalse(old.filter(RoleType.DECODE).test(d.getStatus().getGenerationId()));
            assertTrue(planner.prepare(context, binding, "blue").completeGroups().isEmpty());
            fleet.workers.completeRetirement(RoleType.DECODE, d.ipPort(), d.getStatus(), detached, fleet.cache,
                    org.slf4j.LoggerFactory.getLogger(getClass()));
            var replacement = fleet.publish(WorkerStatus.createDiscovered(RoleType.DECODE, "blue", d.getIp(),
                    8080, 8081, "test"), descriptor(RoleType.DECODE, "target", 64, 2), 100000L);
            var restored = planner.prepare(context, binding, "blue").completeGroups().getFirst();
            assertTrue(restored.contains(RoleType.DECODE, replacement.getStatus().getGenerationId()));
            assertFalse(restored.contains(RoleType.DECODE, d.getStatus().getGenerationId()));
            assertFalse(old.filter(RoleType.DECODE).test(replacement.getStatus().getGenerationId()));
        }
    }

    @Test
    void manyEligibilityPatternsKeepPhysicalMembershipAndWeightsFresh() {
        try (Fleet fleet = new Fleet(true, false, false, false)) {
            fleet.config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            fleet.add(RoleType.PREFILL, "blue", null);
            var decodes = new java.util.ArrayList<DecodeEndpoint>();
            for (int i = 0; i < 8; i++) { decodes.add((DecodeEndpoint) fleet.add(RoleType.DECODE, "blue", null)); }
            fleet.add(RoleType.VIT, "blue", null);
            var planner = new CompatibilityPlanner(fleet.workers, ROLES);
            var context = RequestLifecycleTestSupport.context(fleet.config, 4800L);
            var binding = ScheduledRequest.DecodeBinding.capture(context);
            for (int pattern = 0; pattern < 130; pattern++) {
                int mask = pattern % 128; // one Decode always remains eligible
                var handles = new java.util.ArrayList<DecodeEndpoint.ReservationHandle>();
                var permits = new java.util.ArrayList<DecodeEndpoint.EngineDispatchPermit>();
                try {
                    for (int i = 0; i < 7; i++) {
                        if ((mask & (1 << i)) == 0) { continue; }
                        DecodeEndpoint.ReservationHandle handle;
                        try (var pin = decodes.get(i).tryPinGeneration()) {
                            handle = decodes.get(i).reserve(pin, 100000L + pattern * 10L + i, 16L, 24L, 50);
                        }
                        var acquired = decodes.get(i).acquireDispatchPermit(handle, binding.capacity());
                        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
                        handles.add(handle); permits.add(acquired.permit());
                    }
                    var group = planner.prepare(context, binding, "blue").completeGroups().getFirst();
                    int expected = 8 - Integer.bitCount(mask);
                    assertEquals(expected, group.members().get(RoleType.DECODE).size());
                    assertEquals((double) expected, group.effectiveCounts().get(RoleType.DECODE));
                    assertEquals(Math.cbrt(expected), group.weight(), 1e-12);
                } finally {
                    for (int i = 0; i < permits.size(); i++) {
                        permits.get(i).release();
                        int worker = Integer.numberOfTrailingZeros(mask);
                        decodes.get(worker).release(handles.get(i), DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
                        mask &= mask - 1;
                    }
                }
            }
        }
    }

    private static void updateGroup(WorkerEndpoint endpoint, String group) {
        var status = endpoint.getStatus();
        status.lock.lock();
        try { status.updateDiscoveryLabels("test", group); }
        finally { status.lock.unlock(); }
    }

    private static final class Fleet implements AutoCloseable {
        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        private final RequestRegistry requests;
        private final EndpointRegistry endpoints;
        private final WorkerDirectory workers;
        private final DefaultRouter router;
        private final RequestScheduler scheduler;
        private final SchedulerRuntime runtime;
        private final ConfigService service;
        private final CacheAwareService cache = mock(CacheAwareService.class);
        private final CostBasedPrefillStrategy prefillSelector;
        private final FlexMonitor monitor = mock(FlexMonitor.class);
        private int address;

        Fleet(boolean direct, boolean priority, boolean window, boolean batch) {
            if (priority) { SchedulingTestConfig.usePriorityQueue(config); }
            else { SchedulingTestConfig.useFifoQueue(config); }
            if (window) {
                SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(2);
                config.queueScheduler().getDecision().setMaxCollectionWaitMs(5L);
            } else { SchedulingTestConfig.useSingleDecision(config); }
            if (batch) { SchedulingTestConfig.useBatchDispatcher(config); }
            else { SchedulingTestConfig.useNonBatchDispatcher(config); }
            if (direct) { config.setScheduler(SchedulerConfig.direct()); }
            service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(BatchSchedulerReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            requests = new RequestRegistry(service, reporter, requestReporter);
            var availability = new PlacementAvailability();
            DeliveryStrategy delivery = batch ? new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.accepted(
                    new BatchDeliveryStrategy.PreparedSubmission() {
                        public void submit(BatchDeliveryStrategy.Delivery delivery) {
                            delivery.run((items, id, predicted, reason, observer) ->
                                    items.forEach(item -> observer.accept(item, DeliveryResult.delivered())));
                        }
                        public void close() { }
                    }), new AtomicLong()::incrementAndGet, requests, new DeliveryMetrics(reporter))
                    : new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter));
            endpoints = new EndpointRegistry(service, new EndpointEventProjector(requests), reporter, delivery, availability);
            workers = new WorkerDirectory(endpoints);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            prefillSelector = new CostBasedPrefillStrategy(workers, cache, mock(EngineHealthReporter.class));
            router = newRouter(prefillSelector);
            scheduler = new RequestScheduler(service, router, endpoints, reporter, mock(EvictionManager.class), requests, availability);
            runtime = new SchedulerRuntime(requests, endpoints, reporter, requestReporter, scheduler);
        }

        DefaultRouter newRouter(CostBasedPrefillStrategy selector) {
            return newRouter(selector, ROLES);
        }

        DefaultRouter newRouter(CostBasedPrefillStrategy selector, List<RoleType> roles) {
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(roles);
            var result = new DefaultRouter(selector,
                    new DecodeSelector(workers), new RandomStrategy(workers), service, model, workers);
            result.setCompatibilityGroupReporter(new CompatibilityGroupReporter(monitor));
            return result;
        }

        WorkerEndpoint add(RoleType role, String group, RoleCompatibility descriptor) {
            return add(role, group, descriptor, 100000L);
        }

        WorkerEndpoint add(RoleType role, String group, RoleCompatibility descriptor, long availableKv) {
            WorkerStatus status = WorkerStatus.createDiscovered(role, group, "127.0.0." + (++address), 8080, 8081, "test");
            return publish(status, descriptor, availableKv);
        }

        private WorkerEndpoint publish(WorkerStatus status, RoleCompatibility descriptor, long availableKv) {
            RoleType role = status.getRole();
            workers.currentOrDiscover(role, status.getIpPort(), () -> status);
            var response = new WorkerStatusResponse();
            response.setRole(role);
            response.setAlive(true);
            response.setStatusVersion(1L);
            response.setTotalKvCacheTokens(100000L);
            response.setAvailableKvCacheTokens(availableKv);
            response.setMaxSeqLen(10000L);
            response.setMaxBatchTokensSize(10000L);
            response.setCompatibility(descriptor);
            EndpointRegistry.EndpointPublication published;
            status.lock.lock();
            try {
                published = workers.publishPreparedEndpoint(status.getIpPort(), status,
                        status.prepareNewStatus(status.freezeStatusResponse(response)));
            } finally { status.lock.unlock(); }
            published.statusProjection().run();
            return published.endpoint();
        }

        public void close() { runtime.shutdown(); }
    }
}
