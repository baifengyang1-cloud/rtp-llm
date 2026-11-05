package org.flexlb.service.monitor;

import org.flexlb.balance.strategy.CompatibilityGroup;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/** Attempt outcomes are telemetry only; they cannot change rotation or request ownership. */
@Component
public class CompatibilityGroupReporter {
    public static final String SELECTION = "router.compatibility_group.selection_count";
    public static final String WORKER_SELECTION = "router.compatibility_group.worker_selection_count";
    public static final String ADMISSION = "router.compatibility_group.admission_count";
    public static final String DISPATCH = "router.compatibility_group.dispatch_count";
    public static final String WEIGHT = "router.compatibility_group.weight";
    public static final String EFFECTIVE_COUNT = "router.compatibility_group.effective_instances";
    static final int MAX_GROUPS = 1024;
    private static final int MAX_SCOPES = 64;
    private static final String[] OUTCOME_METRICS = {SELECTION, WORKER_SELECTION, ADMISSION, DISPATCH};
    private final FlexMonitor monitor;
    private final Map<Scope, Observation> observations = new LinkedHashMap<>(16, 0.75f, true);
    private final Map<String, PreparedTags> prepared = new LinkedHashMap<>(16, 0.75f, true);

    private record Scope(String policy, Set<RoleType> roles) { }
    private static final class Observation {
        private List<CompatibilityGroup> source;
        private final Map<String, PreparedTags> tags = new HashMap<>();
    }

    private static final class PreparedTags {
        private final Set<Observation> owners = new HashSet<>();
        private final String policy;
        private final String id;
        private final Map<String, FlexMetricTags> outcomes = new HashMap<>();
        private final Map<RoleType, FlexMetricTags> roles = new EnumMap<>(RoleType.class);
        private boolean active = true;

        PreparedTags(CompatibilityGroup group) {
            policy = String.valueOf(group.key().policyGroup()); id = group.id();
            for (var role : group.effectiveCounts().keySet()) {
                roles.put(role, FlexMetricTags.of("policy_group", policy, "compatibility_group", id, "role", role.name()));
            }
        }

        FlexMetricTags outcome(String value) {
            return outcomes.computeIfAbsent(value,
                    key -> FlexMetricTags.of("policy_group", policy, "compatibility_group", id, "outcome", key));
        }
    }

    public CompatibilityGroupReporter(FlexMonitor monitor) { this.monitor = monitor; }

    @PostConstruct
    public void init() {
        for (String metric : OUTCOME_METRICS) {
            monitor.register(metric, FlexMetricType.QPS);
        }
        monitor.register(WEIGHT, FlexMetricType.GAUGE);
        monitor.register(EFFECTIVE_COUNT, FlexMetricType.GAUGE);
    }

    public synchronized Attempt selected(String policyGroup, List<RoleType> roles,
                                         List<CompatibilityGroup> groups, CompatibilityGroup chosen) {
        Observation observation = observations.computeIfAbsent(new Scope(policyGroup, Set.copyOf(roles)),
                ignored -> new Observation());
        if (observation.source != groups) {
            Set<String> present = new HashSet<>();
            for (CompatibilityGroup group : groups) { present.add(group.id()); }
            for (PreparedTags tags : List.copyOf(observation.tags.values())) {
                if (!present.contains(tags.id)) { release(observation, tags); }
            }
            for (CompatibilityGroup group : groups) { gauges(prepare(observation, group), group); }
            observation.source = groups;
        }
        while (observations.size() > MAX_SCOPES) {
            var iterator = observations.values().iterator();
            Observation retired = iterator.next();
            iterator.remove();
            for (PreparedTags tags : List.copyOf(retired.tags.values())) { release(retired, tags); }
        }
        if (chosen == null) { return null; }
        boolean observed = observation.tags.containsKey(chosen.id());
        PreparedTags tags = prepare(observation, chosen);
        if (!observed) { gauges(tags, chosen); }
        Attempt attempt = new Attempt(this, tags);
        report(SELECTION, attempt, "selected");
        return attempt;
    }

    private PreparedTags prepare(Observation observation, CompatibilityGroup group) {
        PreparedTags tags = prepared.get(group.id());
        if (tags == null) {
            while (prepared.size() >= MAX_GROUPS) { retire(prepared.values().iterator().next()); }
            tags = new PreparedTags(group);
            prepared.put(group.id(), tags);
        }
        // An unscoped plan can expose the same series as a named policy plan.
        tags.owners.add(observation);
        observation.tags.put(group.id(), tags);
        return tags;
    }

    private void gauges(PreparedTags tags, CompatibilityGroup group) {
        try {
            monitor.report(WEIGHT, tags.outcome("selected"), group.weight());
            for (var entry : tags.roles.entrySet()) {
                monitor.report(EFFECTIVE_COUNT, entry.getValue(), group.effectiveCounts().get(entry.getKey()));
            }
        } catch (Throwable failure) { Logger.warn("Compatibility group weight telemetry failed", failure); }
    }

    private void release(Observation observation, PreparedTags tags) {
        observation.tags.remove(tags.id);
        tags.owners.remove(observation);
        if (tags.owners.isEmpty()) { retire(tags); }
    }

    private void retire(PreparedTags tags) {
        tags.active = false;
        prepared.remove(tags.id);
        // Global capacity eviction and shutdown withdraw the series from every scope.
        for (Observation owner : tags.owners) { owner.tags.remove(tags.id); }
        tags.owners.clear();
        retireSeries(WEIGHT, tags.outcome("selected"), true);
        for (var role : tags.roles.values()) { retireSeries(EFFECTIVE_COUNT, role, true); }
        for (FlexMetricTags outcome : tags.outcomes.values()) {
            for (String metric : OUTCOME_METRICS) { retireSeries(metric, outcome, false); }
        }
        tags.outcomes.clear();
        tags.roles.clear();
    }

    private void retireSeries(String metric, FlexMetricTags tags, boolean gauge) {
        if (gauge) {
            try { monitor.report(metric, tags, 0.0); }
            catch (Throwable failure) { Logger.warn("Compatibility group retirement telemetry failed", failure); }
        }
        try { monitor.remove(metric, tags); }
        catch (Throwable failure) { Logger.warn("Compatibility group retirement telemetry failed", failure); }
    }

    @PreDestroy
    public synchronized void close() {
        for (PreparedTags tags : List.copyOf(prepared.values())) { retire(tags); }
        observations.clear();
    }

    private synchronized void report(String metric, Attempt attempt, String outcome) {
        if (!attempt.tags.active) { return; }
        try { monitor.report(metric, attempt.tags.outcome(outcome), 1.0); }
        catch (Throwable failure) { Logger.warn("Compatibility group outcome telemetry failed", failure); }
    }

    public static final class Attempt {
        private final CompatibilityGroupReporter reporter;
        private final PreparedTags tags;
        private final AtomicBoolean admission = new AtomicBoolean();
        private final AtomicBoolean dispatch = new AtomicBoolean();

        private Attempt(CompatibilityGroupReporter reporter, PreparedTags tags) {
            this.reporter = reporter;
            this.tags = tags;
        }

        public void workerSelection(String outcome) { reporter.report(WORKER_SELECTION, this, outcome); }
        public void admission(boolean committed) {
            if (admission.compareAndSet(false, true)) { reporter.report(ADMISSION, this, committed ? "committed" : "not_committed"); }
        }
        public void dispatch(String outcome) {
            if (dispatch.compareAndSet(false, true)) { reporter.report(DISPATCH, this, outcome); }
        }
    }
}
