package org.flexlb.balance.strategy;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.LongPredicate;

/**
 * Caches immutable generation/descriptor structure, never admission observations.
 * Every request reads live eligibility. Class-presence and physical-eligibility
 * caches are bounded and scoped to one published topology and policy boundary.
 */
public final class CompatibilityPlanner {
    private final WorkerDirectory directory;
    private final List<RoleType> roles;
    private final Cache<String, Index> named = Caffeine.newBuilder().maximumSize(64).build();
    private volatile Index unscoped;

    public CompatibilityPlanner(WorkerDirectory directory, List<RoleType> roles) {
        this.directory = directory;
        this.roles = List.copyOf(roles);
    }

    public CompatibilityCandidates prepare(BalanceContext context, DecodeBinding decode, String group) {
        Map<RoleType, List<WorkerEndpoint>> topology = directory.routingTopologySnapshot();
        if (topology == null || topology.isEmpty()) {
            // Directly constructed/mock directories retain the existing snapshot contract.
            return uncached(context, decode, group);
        }
        Index index = group == null ? unscoped : named.getIfPresent(group);
        for (int retry = 0; retry < 2; retry++) {
            if (index == null || index.topology != topology) {
                index = new Index(topology, group, roles);
                if (group == null) { unscoped = index; } else { named.put(group, index); }
            }
            Plan plan = index.plan(context, decode);
            if (plan != null) { return new CompatibilityCandidates(directory, roles, group, plan); }
            // Discovery labels may change without replacing a generation.
            index = null;
        }
        return uncached(context, decode, group);
    }

    private CompatibilityCandidates uncached(BalanceContext context, DecodeBinding decode, String group) {
        var candidates = new CompatibilityCandidates(directory, roles, group);
        candidates.prepare(context, decode);
        return candidates;
    }

    private record DescriptorClass(RoleType role, String group, RoleCompatibility descriptor) { }

    private static final class Index {
        final Map<RoleType, List<WorkerEndpoint>> topology;
        final WorkerEndpoint[] endpoints;
        final WorkerStatus[] statuses;
        final String[] labels;
        final WorkerStatus.TopologySnapshot[] labelVersions;
        final int[] classes;
        final Map<RoleType, List<WorkerStatus>> workers;
        final List<RoleType> roles;
        final long[] generationKeys;
        final int[] generationValues;
        final Cache<BitSet, List<CompatibilityGroup>> structures = Caffeine.newBuilder().maximumSize(64).build();
        final Cache<BitSet, Plan> plans = Caffeine.newBuilder().maximumSize(64).build();
        final ThreadLocal<BitSet> scratch = ThreadLocal.withInitial(BitSet::new);

        Index(Map<RoleType, List<WorkerEndpoint>> topology, String boundary, List<RoleType> roles) {
            this.topology = topology;
            this.roles = roles;
            var all = new ArrayList<WorkerEndpoint>();
            for (RoleType role : roles) { all.addAll(topology.getOrDefault(role, List.of())); }
            endpoints = all.toArray(WorkerEndpoint[]::new);
            statuses = new WorkerStatus[endpoints.length];
            labels = new String[endpoints.length];
            labelVersions = new WorkerStatus.TopologySnapshot[endpoints.length];
            classes = new int[endpoints.length];
            workers = new EnumMap<>(RoleType.class);
            Map<DescriptorClass, Integer> descriptorClasses = new HashMap<>();
            int capacity = 2;
            while (capacity < endpoints.length * 2) { capacity *= 2; }
            generationKeys = new long[capacity];
            generationValues = new int[capacity];
            for (int i = 0; i < endpoints.length; i++) {
                WorkerStatus status = endpoints[i].getStatus();
                statuses[i] = status;
                labelVersions[i] = status.topologySnapshot();
                labels[i] = labelVersions[i].group();
                int slot = slot(status.getGenerationId());
                while (generationKeys[slot] != 0L) { slot = (slot + 1) & (capacity - 1); }
                generationKeys[slot] = status.getGenerationId(); generationValues[slot] = i;
                classes[i] = -1;
                if (boundary == null || boundary.equals(labels[i])) {
                    var key = new DescriptorClass(status.getRole(), labels[i], status.compatibility());
                    classes[i] = descriptorClasses.computeIfAbsent(key, ignored -> descriptorClasses.size());
                    workers.computeIfAbsent(status.getRole(), ignored -> new ArrayList<>()).add(status);
                }
            }
        }

        int slot(long generation) {
            long mixed = generation * 0x9E3779B97F4A7C15L;
            return (int) (mixed ^ (mixed >>> 32)) & (generationKeys.length - 1);
        }

        int find(long generation) {
            int slot = slot(generation);
            while (generationKeys[slot] != 0L) {
                if (generationKeys[slot] == generation) { return generationValues[slot]; }
                slot = (slot + 1) & (generationKeys.length - 1);
            }
            return -1;
        }

        Plan plan(BalanceContext context, DecodeBinding decode) {
            BitSet eligible = scratch.get();
            eligible.clear();
            for (int i = 0; i < endpoints.length; i++) {
                WorkerStatus status = statuses[i];
                if (labelVersions[i] != status.topologySnapshot()) { return null; }
                if (classes[i] < 0 || !status.isActiveGeneration()) { continue; }
                WorkerEndpoint endpoint = endpoints[i];
                boolean accepts = endpoint instanceof PrefillEndpoint p ? CostBasedPrefillStrategy.eligible(context, p)
                        : endpoint instanceof DecodeEndpoint d ? DecodeSelector.eligible(decode, d.routingView()) : true;
                if (accepts) { eligible.set(i); }
            }
            Plan cached = plans.getIfPresent(eligible);
            if (cached != null) { return cached; }
            return plans.get((BitSet) eligible.clone(), this::buildPlan);
        }

        Plan buildPlan(BitSet eligible) {
            BitSet present = new BitSet();
            for (int i = eligible.nextSetBit(0); i >= 0; i = eligible.nextSetBit(i + 1)) { present.set(classes[i]); }
            List<CompatibilityGroup> structural = structures.get(present, mask -> CompatibilityGroups.build(roles, workers,
                    generation -> { int i = find(generation); return i >= 0 && classes[i] >= 0 && mask.get(classes[i]); }));
            List<CompatibilityGroup> narrowed = new ArrayList<>(structural.size());
            for (CompatibilityGroup group : structural) {
                Map<RoleType, Set<Long>> members = new EnumMap<>(RoleType.class);
                for (RoleType role : roles) {
                    Set<Long> actual = new HashSet<>();
                    for (long generation : group.members().get(role)) {
                        int i = find(generation);
                        if (i >= 0 && eligible.get(i)) { actual.add(generation); }
                    }
                    members.put(role, actual);
                }
                narrowed.add(new CompatibilityGroup(group.key(), members, Map.of(), 0.0, group.id()));
            }
            return new Plan(this, eligible, CompatibilityGroup.weighted(narrowed, roles));
        }
    }

    static final class Plan {
        final List<CompatibilityGroup> groups;
        final IdentityHashMap<CompatibilityGroup, LongPredicate[]> filters = new IdentityHashMap<>();
        final Index index;
        final BitSet eligible;

        Plan(Index index, BitSet eligible, List<CompatibilityGroup> groups) {
            this.index = index; this.eligible = eligible; this.groups = groups;
            for (CompatibilityGroup group : groups) {
                BitSet membership = new BitSet();
                for (Set<Long> ids : group.members().values()) {
                    for (long generation : ids) { membership.set(index.find(generation)); }
                }
                LongPredicate[] byRole = new LongPredicate[RoleType.values().length];
                for (RoleType role : index.roles) {
                    byRole[role.ordinal()] = generation -> {
                        int i = index.find(generation);
                        return i >= 0 && membership.get(i) && index.statuses[i].getRole() == role
                                && index.statuses[i].isActiveGeneration()
                                && Objects.equals(index.labels[i], index.statuses[i].getGroup());
                    };
                }
                filters.put(group, byRole);
            }
        }

        Set<Long> eligibleGenerations() {
            Set<Long> result = new HashSet<>();
            for (int i = eligible.nextSetBit(0); i >= 0; i = eligible.nextSetBit(i + 1)) {
                result.add(index.statuses[i].getGenerationId());
            }
            return result;
        }
    }
}
