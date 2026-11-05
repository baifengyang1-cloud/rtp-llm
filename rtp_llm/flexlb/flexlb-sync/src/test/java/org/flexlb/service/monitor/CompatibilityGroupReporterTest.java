package org.flexlb.service.monitor;

import org.flexlb.balance.strategy.CompatibilityGroup;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.springframework.test.util.ReflectionTestUtils;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class CompatibilityGroupReporterTest {
    private static CompatibilityGroupReporter.Attempt select(CompatibilityGroupReporter reporter, CompatibilityGroup group) {
        return reporter.selected(group.key().policyGroup(), List.copyOf(group.effectiveCounts().keySet()), List.of(group), group);
    }

    private static CompatibilityGroup group() {
        return new CompatibilityGroup(new CompatibilityGroup.Key("blue", Map.of()),
                Map.of(RoleType.VIT, Set.of(1L)), Map.of(RoleType.VIT, 0.5), Math.cbrt(0.5), "group-a");
    }

    @Test
    void selectionAdmissionAndDispatchAreIndependentAndKeepFractionalGauges() {
        var monitor = mock(FlexMonitor.class);
        var reporter = new CompatibilityGroupReporter(monitor);
        reporter.init();
        for (String metric : new String[]{CompatibilityGroupReporter.SELECTION, CompatibilityGroupReporter.WORKER_SELECTION,
                CompatibilityGroupReporter.ADMISSION, CompatibilityGroupReporter.DISPATCH}) {
            verify(monitor).register(metric, FlexMetricType.QPS);
        }
        var selected = select(reporter, group());
        var tags = FlexMetricTags.of("policy_group", "blue", "compatibility_group", "group-a", "outcome", "selected");
        verify(monitor).report(CompatibilityGroupReporter.SELECTION, tags, 1.0);
        verify(monitor).report(CompatibilityGroupReporter.WEIGHT, tags, Math.cbrt(0.5));
        verify(monitor).report(CompatibilityGroupReporter.EFFECTIVE_COUNT,
                FlexMetricTags.of("policy_group", "blue", "compatibility_group", "group-a", "role", "VIT"), 0.5);
        verify(monitor, never()).report(eq(CompatibilityGroupReporter.ADMISSION), any(), anyDouble());
        verify(monitor, never()).report(eq(CompatibilityGroupReporter.DISPATCH), any(), anyDouble());
        selected.workerSelection("selected");
        selected.admission(true);
        selected.admission(false); // RouteAdmission.close after commit cannot rewrite the outcome.
        selected.dispatch("DELIVERED");
        selected.dispatch("DELIVERY_TIMEOUT");
        verify(monitor).report(CompatibilityGroupReporter.WORKER_SELECTION, tags, 1.0);
        verify(monitor).report(CompatibilityGroupReporter.ADMISSION,
                FlexMetricTags.of("policy_group", "blue", "compatibility_group", "group-a", "outcome", "committed"), 1.0);
        verify(monitor).report(CompatibilityGroupReporter.DISPATCH,
                FlexMetricTags.of("policy_group", "blue", "compatibility_group", "group-a", "outcome", "DELIVERED"), 1.0);
        verify(monitor, times(1)).report(eq(CompatibilityGroupReporter.ADMISSION), any(), anyDouble());
        verify(monitor, times(1)).report(eq(CompatibilityGroupReporter.DISPATCH), any(), anyDouble());
    }

    @Test
    void duplicateAsynchronousOutcomeCallbacksOnlyReportOnce() throws Exception {
        var monitor = mock(FlexMonitor.class);
        var attempt = select(new CompatibilityGroupReporter(monitor), group());
        try (var pool = Executors.newFixedThreadPool(8)) {
            for (int i = 0; i < 100; i++) {
                pool.submit(() -> { attempt.admission(false); attempt.dispatch("DELIVERY_TIMEOUT"); });
            }
            pool.shutdown();
            assertTrue(pool.awaitTermination(3, TimeUnit.SECONDS));
        }
        verify(monitor, times(1)).report(eq(CompatibilityGroupReporter.ADMISSION), any(), anyDouble());
        verify(monitor, times(1)).report(eq(CompatibilityGroupReporter.DISPATCH), any(), anyDouble());
    }

    @Test
    void reusedIdentityKeepsLiveGaugesAndIndependentRequestOutcomes() {
        var monitor = mock(FlexMonitor.class);
        var reporter = new CompatibilityGroupReporter(monitor);
        var original = group();
        var first = select(reporter, original);
        var changed = new CompatibilityGroup(original.key(), original.members(), Map.of(RoleType.VIT, 1.0), 1.0, original.id());
        var second = select(reporter, changed);
        first.admission(false);
        second.admission(true);
        first.dispatch("DELIVERY_TIMEOUT");
        second.dispatch("DELIVERED");
        var selected = FlexMetricTags.of("policy_group", "blue", "compatibility_group", "group-a", "outcome", "selected");
        verify(monitor).report(CompatibilityGroupReporter.WEIGHT, selected, Math.cbrt(0.5));
        verify(monitor).report(CompatibilityGroupReporter.WEIGHT, selected, 1.0);
        var role = FlexMetricTags.of("policy_group", "blue", "compatibility_group", "group-a", "role", "VIT");
        verify(monitor).report(CompatibilityGroupReporter.EFFECTIVE_COUNT, role, 0.5);
        verify(monitor).report(CompatibilityGroupReporter.EFFECTIVE_COUNT, role, 1.0);
        verify(monitor, times(2)).report(eq(CompatibilityGroupReporter.ADMISSION), any(), anyDouble());
        verify(monitor, times(2)).report(eq(CompatibilityGroupReporter.DISPATCH), any(), anyDouble());
    }

    @Test
    void observedGroupsHaveCurrentGaugesAndRetirementFencesLateCallbacks() {
        var registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var reporter = new CompatibilityGroupReporter(monitor);
            reporter.init();
            var a = group();
            var b = new CompatibilityGroup(a.key(), a.members(), Map.of(RoleType.VIT, 1.5), 1.5, "group-b");
            var first = reporter.selected("blue", List.of(RoleType.VIT), List.of(a, b), a);
            assertEquals(1.5, registry.get("flexlb." + CompatibilityGroupReporter.WEIGHT)
                    .tag("compatibility_group", b.id()).gauge().value());
            reporter.selected("blue", List.of(RoleType.VIT), List.of(b), b);
            first.workerSelection("blocked");
            first.admission(false);
            first.dispatch("DELIVERY_TIMEOUT");
            assertTrue(registry.getMeters().stream().noneMatch(m -> a.id().equals(m.getId().getTag("compatibility_group"))));
            assertEquals(1.5, registry.get("flexlb." + CompatibilityGroupReporter.EFFECTIVE_COUNT)
                    .tag("compatibility_group", b.id()).gauge().value());
            var replacement = reporter.selected("blue", List.of(RoleType.VIT), List.of(a), a);
            first.workerSelection("rejected");
            first.dispatch("DELIVERED");
            assertNull(registry.find("flexlb." + CompatibilityGroupReporter.WORKER_SELECTION).counter());
            replacement.dispatch("DELIVERED");
            assertEquals(1.0, registry.get("flexlb." + CompatibilityGroupReporter.DISPATCH).counter().count());
            reporter.selected("blue", List.of(RoleType.VIT), List.of(), null);
            replacement.workerSelection("blocked");
            assertTrue(registry.getMeters().isEmpty());
            assertMonitorCachesEmpty(monitor);
        } finally { registry.close(); }
    }

    @Test
    void registryAndDownstreamCachesStayBoundedBeyondPreparedTagLimit() {
        var registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var reporter = new CompatibilityGroupReporter(monitor);
            reporter.init();
            var base = group();
            List<CompatibilityGroup> groups = new ArrayList<>();
            for (int i = 0; i < CompatibilityGroupReporter.MAX_GROUPS + 20; i++) {
                groups.add(new CompatibilityGroup(base.key(), base.members(), base.effectiveCounts(), base.weight(), "group-" + i));
            }
            var evicted = reporter.selected("blue", List.of(RoleType.VIT), List.of(groups.getFirst()), groups.getFirst());
            reporter.selected("blue", List.of(RoleType.VIT), groups, groups.getLast());
            evicted.workerSelection("blocked");
            evicted.admission(false);
            evicted.dispatch("DELIVERED");
            long exposedGroups = registry.getMeters().stream()
                    .map(m -> m.getId().getTag("compatibility_group")).distinct().count();
            assertEquals(CompatibilityGroupReporter.MAX_GROUPS, exposedGroups);
            assertEquals(2 * CompatibilityGroupReporter.MAX_GROUPS + 1, registry.getMeters().size());
            assertTrue(registry.getMeters().stream().noneMatch(m -> groups.getFirst().id().equals(m.getId().getTag("compatibility_group"))));
            for (String cache : List.of("gaugeCache", "counterCache", "timerCache", "tagsCache")) {
                var entries = (Map<?, ?>) ReflectionTestUtils.getField(monitor, cache);
                assertNotNull(entries);
                assertTrue(entries.size() <= 2 * CompatibilityGroupReporter.MAX_GROUPS + 1, cache);
            }
            reporter.close();
            assertTrue(registry.getMeters().isEmpty());
            assertMonitorCachesEmpty(monitor);
        } finally { registry.close(); }
    }

    @Test
    void scopeRetirementDoesNotRemoveOtherPolicyOrRoleSeries() {
        var registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var reporter = new CompatibilityGroupReporter(monitor);
            reporter.init();
            var a = group();
            reporter.selected("blue", List.of(RoleType.VIT), List.of(a), a);
            for (int i = 0; i < 70; i++) {
                var b = new CompatibilityGroup(new CompatibilityGroup.Key("policy-" + i, Map.of()),
                        a.members(), a.effectiveCounts(), a.weight(), "scope-" + i);
                reporter.selected("policy-" + i, List.of(RoleType.VIT), List.of(b), b);
            }
            assertEquals(64, registry.getMeters().stream().map(m -> m.getId().getTag("compatibility_group")).distinct().count());
            reporter.selected("policy-69", List.of(RoleType.VIT), List.of(), null);
            assertEquals(63, registry.getMeters().stream().map(m -> m.getId().getTag("compatibility_group")).distinct().count());
            reporter.close();
            assertMonitorCachesEmpty(monitor);
        } finally { registry.close(); }
    }

    private void assertMonitorCachesEmpty(MicrometerFlexMonitor monitor) {
        for (String cache : List.of("gaugeCache", "counterCache", "timerCache", "tagsCache")) {
            assertEquals(Map.of(), ReflectionTestUtils.getField(monitor, cache), cache);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void sharedSeriesSurviveEitherScopeBecomingEmpty(boolean removeUnscoped) {
        var registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var reporter = new CompatibilityGroupReporter(monitor);
            reporter.init();
            var group = group();
            var unscoped = reporter.selected(null, List.of(RoleType.VIT), List.of(group), group);
            var scoped = reporter.selected("blue", List.of(RoleType.VIT), List.of(group), group);
            reporter.selected(removeUnscoped ? null : "blue", List.of(RoleType.VIT), List.of(), null);
            assertEquals(group.weight(), registry.get("flexlb." + CompatibilityGroupReporter.WEIGHT).gauge().value());
            assertEquals(0.5, registry.get("flexlb." + CompatibilityGroupReporter.EFFECTIVE_COUNT).gauge().value());
            unscoped.admission(true);
            scoped.admission(true);
            unscoped.dispatch("DELIVERED");
            scoped.dispatch("DELIVERED");
            assertEquals(2.0, registry.get("flexlb." + CompatibilityGroupReporter.ADMISSION).counter().count());
            assertEquals(2.0, registry.get("flexlb." + CompatibilityGroupReporter.DISPATCH).counter().count());

            reporter.selected(removeUnscoped ? "blue" : null, List.of(RoleType.VIT), List.of(), null);
            unscoped.workerSelection("blocked");
            scoped.workerSelection("blocked");
            assertTrue(registry.getMeters().isEmpty());
            assertMonitorCachesEmpty(monitor);
        } finally { registry.close(); }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void sharedSeriesSurviveEvictionOfEitherOwningScope(boolean evictUnscoped) {
        var registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var reporter = new CompatibilityGroupReporter(monitor);
            reporter.init();
            var group = group();
            var unscoped = reporter.selected(null, List.of(RoleType.VIT), List.of(group), group);
            var scoped = reporter.selected("blue", List.of(RoleType.VIT), List.of(group), group);
            // Make the departing scope oldest, then exceed the 64-scope limit.
            reporter.selected(evictUnscoped ? "blue" : null, List.of(RoleType.VIT), List.of(group), null);
            for (int i = 0; i < 63; i++) {
                reporter.selected("policy-" + i, List.of(RoleType.VIT), List.of(), null);
            }
            assertEquals(group.weight(), registry.get("flexlb." + CompatibilityGroupReporter.WEIGHT).gauge().value());
            assertEquals(0.5, registry.get("flexlb." + CompatibilityGroupReporter.EFFECTIVE_COUNT).gauge().value());
            unscoped.dispatch("DELIVERED");
            scoped.dispatch("DELIVERED");
            assertEquals(2.0, registry.get("flexlb." + CompatibilityGroupReporter.DISPATCH).counter().count());
            reporter.selected(evictUnscoped ? "blue" : null, List.of(RoleType.VIT), List.of(), null);
            unscoped.workerSelection("blocked");
            scoped.workerSelection("blocked");
            assertTrue(registry.getMeters().isEmpty());
            assertMonitorCachesEmpty(monitor);
        } finally { registry.close(); }
    }

    @Test
    void globalGroupEvictionDetachesEveryScopeAndFencesOldAttemptsOnReturn() {
        var registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var reporter = new CompatibilityGroupReporter(monitor);
            reporter.init();
            var group = group();
            var unscoped = reporter.selected(null, List.of(RoleType.VIT), List.of(group), group);
            var scoped = reporter.selected("blue", List.of(RoleType.VIT), List.of(group), group);
            var others = new ArrayList<CompatibilityGroup>();
            for (int i = 0; i < CompatibilityGroupReporter.MAX_GROUPS; i++) {
                others.add(new CompatibilityGroup(group.key(), group.members(), group.effectiveCounts(), group.weight(), "other-" + i));
            }
            reporter.selected("other", List.of(RoleType.VIT), others, null);
            var replacement = reporter.selected(null, List.of(RoleType.VIT), List.of(group), group);
            reporter.selected("blue", List.of(RoleType.VIT), List.of(), null);
            unscoped.dispatch("DELIVERED");
            scoped.dispatch("DELIVERED");
            assertNull(registry.find("flexlb." + CompatibilityGroupReporter.DISPATCH).counter());
            replacement.dispatch("DELIVERED");
            assertEquals(1.0, registry.get("flexlb." + CompatibilityGroupReporter.DISPATCH).counter().count());
            reporter.close();
            assertTrue(registry.getMeters().isEmpty());
            assertMonitorCachesEmpty(monitor);
        } finally { registry.close(); }
    }

    @Test
    void telemetryFailureCannotChangeRoutingOrOwnership() {
        var monitor = mock(FlexMonitor.class);
        doThrow(new IllegalStateException("monitor unavailable")).when(monitor).report(anyString(), any(), anyDouble());
        var reporter = new CompatibilityGroupReporter(monitor);
        var attempt = assertDoesNotThrow(() -> select(reporter, group()));
        assertDoesNotThrow(() -> {
            attempt.workerSelection("blocked");
            attempt.admission(false);
            attempt.dispatch("DELIVERY_TIMEOUT");
        });
    }
}
