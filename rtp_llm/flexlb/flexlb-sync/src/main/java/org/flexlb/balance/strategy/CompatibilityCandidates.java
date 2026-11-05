package org.flexlb.balance.strategy;

import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.dao.BalanceContext;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.LongPredicate;

/** Cached request membership, with descriptor-class search for uncached and diagnostic paths. */
public final class CompatibilityCandidates {
    private record Member(String policyGroup, RoleCompatibility descriptor) { }
    private final List<RoleType> roles;
    private final Map<RoleType, Set<Member>> classes;
    private final Map<Long, Member> generations;
    private final Map<Long, WorkerStatus> statuses;
    private final Map<RoleType, Member> selected;
    private final Map<RoleType, List<WorkerStatus>> workers;
    private CompatibilityGroup chosenGroup;
    private final WorkerDirectory directory;
    private final String policyGroup;
    private final CompatibilityPlanner.Plan fast;
    private CompatibilityCandidates diagnostic;
    private LongPredicate[] chosenFilters;
    private List<CompatibilityGroup> preparedGroups;
    private Set<Long> preparedEligible;

    CompatibilityCandidates(WorkerDirectory directory, List<RoleType> roles, String group, CompatibilityPlanner.Plan fast) {
        this.directory = directory; this.roles = roles; this.policyGroup = group; this.fast = fast;
        classes = null; generations = null; statuses = null; selected = null; workers = null;
        preparedGroups = fast.groups;
    }

    public CompatibilityCandidates(WorkerDirectory directory, List<RoleType> roles, String group) {
        this.roles = List.copyOf(roles);
        this.directory = directory;
        this.policyGroup = group;
        this.fast = null;
        classes = new EnumMap<>(RoleType.class); generations = new HashMap<>(); statuses = new HashMap<>();
        selected = new EnumMap<>(RoleType.class); workers = new EnumMap<>(RoleType.class);
        for (RoleType role : roles) {
            Set<Member> members = new HashSet<>();
            List<WorkerStatus> snapshot = directory.routingStatusSnapshot(role, group);
            workers.put(role, snapshot);
            for (WorkerStatus status : snapshot) {
                Member member = new Member(status.getGroup(), status.compatibility());
                members.add(member);
                generations.put(status.getGenerationId(), member);
                statuses.put(status.getGenerationId(), status);
            }
            classes.put(role, members);
        }
    }

    public List<CompatibilityGroup> completeGroups(LongPredicate eligible) {
        if (fast != null) { return diagnostic().completeGroups(eligible); }
        return CompatibilityGroups.build(roles, workers, eligible);
    }

    void prepare(BalanceContext context, DecodeBinding decode) {
        preparedEligible = eligibleGenerations(context, decode);
        preparedGroups = completeGroups(preparedEligible::contains);
    }

    public List<CompatibilityGroup> completeGroups() { return preparedGroups; }

    public boolean anyEligible(RoleType role) {
        if (fast == null) { return anyEligible(role, preparedEligible); }
        return diagnostic().anyEligible(role, fast.eligibleGenerations());
    }

    private CompatibilityCandidates diagnostic() {
        if (diagnostic == null) { diagnostic = new CompatibilityCandidates(directory, roles, policyGroup); }
        return diagnostic;
    }

    /** Existing worker admission eligibility, before cost preference and exact reservations. */
    public Set<Long> eligibleGenerations(BalanceContext context, DecodeBinding decode) {
        if (fast != null) { return diagnostic().eligibleGenerations(context, decode); }
        Set<Long> eligible = new HashSet<>();
        for (RoleType role : roles) {
            if (role.supportsPrefill()) {
                for (var entry : directory.prefillRoutingSnapshot(role)) {
                    long generation = entry.endpoint().getStatus().getGenerationId();
                    if (generations.containsKey(generation) && CostBasedPrefillStrategy.eligible(context, entry.endpoint())) {
                        eligible.add(generation);
                    }
                }
            } else if (role == RoleType.DECODE) {
                for (var view : directory.decodeRoutingSnapshot(null)) {
                    if (generations.containsKey(view.generationId()) && DecodeSelector.eligible(decode, view)) {
                        eligible.add(view.generationId());
                    }
                }
            } else if (role == RoleType.VIT) {
                for (WorkerStatus worker : workers.get(role)) { eligible.add(worker.getGenerationId()); }
            }
        }
        return eligible;
    }

    public boolean anyEligible(RoleType role, Set<Long> eligible) {
        if (fast != null) {
            LongPredicate allowed = filter(role);
            return eligible.stream().anyMatch(generation -> allowed.test(generation));
        }
        LongPredicate compatible = filter(role);
        return workers.getOrDefault(role, List.of()).stream().anyMatch(worker ->
                eligible.contains(worker.getGenerationId()) && compatible.test(worker.getGenerationId()));
    }

    public void restrictTo(CompatibilityGroup group) {
        if (fast != null) {
            if (chosenFilters != null || diagnostic != null && !diagnostic.selected.isEmpty()) {
                throw new IllegalStateException("group already selected");
            }
            chosenFilters = fast.filters.get(group);
            if (chosenFilters == null) { throw new IllegalArgumentException("group belongs to another snapshot"); }
            return;
        }
        if (!selected.isEmpty() || chosenGroup != null) { throw new IllegalStateException("group already selected"); }
        chosenGroup = Objects.requireNonNull(group, "group");
    }

    public LongPredicate filter(RoleType role) {
        if (fast != null) { return chosenFilters == null ? diagnostic().filter(role) : chosenFilters[role.ordinal()]; }
        if (chosenGroup != null) {
            return generation -> {
                WorkerStatus status = statuses.get(generation);
                return status != null && status.getRole() == role && status.isActiveGeneration()
                        && chosenGroup.contains(role, generation)
                        && Objects.equals(status.getGroup(), generations.get(generation).policyGroup());
            };
        }
        Set<Member> viable = new HashSet<>();
        for (Member member : classes.getOrDefault(role, Set.of())) {
            Map<RoleType, Member> assignment = new EnumMap<>(selected);
            assignment.put(role, member);
            if (extend(assignment, 0)) { viable.add(member); }
        }
        return generation -> {
            Member member = generations.get(generation);
            WorkerStatus status = statuses.get(generation);
            return member != null && viable.contains(member) && status.isActiveGeneration()
                    && (chosenGroup == null || chosenGroup.contains(role, generation))
                    && Objects.equals(status.getGroup(), member.policyGroup());
        };
    }

    public boolean select(RoleType role, WorkerStatus status) {
        if (fast != null && chosenFilters == null) { return diagnostic().select(role, status); }
        if (!filter(role).test(status.getGenerationId())) { return false; }
        if (fast != null) { return true; }
        selected.put(role, generations.get(status.getGenerationId()));
        return true;
    }

    private boolean extend(Map<RoleType, Member> assignment, int index) {
        if (index == roles.size()) { return compatible(assignment); }
        RoleType role = roles.get(index);
        if (assignment.containsKey(role)) { return compatible(assignment) && extend(assignment, index + 1); }
        for (Member member : classes.getOrDefault(role, Set.of())) {
            assignment.put(role, member);
            if (compatible(assignment) && extend(assignment, index + 1)) {
                assignment.remove(role);
                return true;
            }
        }
        assignment.remove(role);
        return false;
    }

    private static boolean compatible(Map<RoleType, Member> assignment) {
        var entries = new ArrayList<>(assignment.entrySet());
        for (int i = 0; i < entries.size(); i++) {
            for (int j = 0; j < i; j++) {
                var a = entries.get(i);
                var b = entries.get(j);
                if (!Objects.equals(a.getValue().policyGroup(), b.getValue().policyGroup())
                        || !RoleCompatibility.matches(a.getKey(), a.getValue().descriptor(),
                        b.getKey(), b.getValue().descriptor())) { return false; }
            }
        }
        return true;
    }
}
