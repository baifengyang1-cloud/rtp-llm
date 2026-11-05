package org.flexlb.balance.strategy;

import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.JsonUtils;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.LongPredicate;

/**
 * Each vertex is a (role, policy group, immutable descriptor) class. Edges join
 * same-role classes or matching producer/consumer classes, with legacy isolated.
 * A maximal clique covering all required roles is one group: every cross-role
 * combination in its worker pools is valid, and classes may occur in several groups.
 * Endpoint multiplicity lives only in membership sets, never in the clique search.
 */
final class CompatibilityGroups {
    private record ClassKey(RoleType role, String policyGroup, RoleCompatibility descriptor) { }
    private record Node(ClassKey key, Set<Long> generations, String order) { }

    static List<CompatibilityGroup> build(List<RoleType> roles, Map<RoleType, List<WorkerStatus>> workers,
                                         LongPredicate eligible) {
        Map<ClassKey, Set<Long>> classes = new LinkedHashMap<>();
        for (RoleType role : roles) {
            for (WorkerStatus worker : workers.getOrDefault(role, List.of())) {
                if (worker.isActiveGeneration() && eligible.test(worker.getGenerationId())) {
                    classes.computeIfAbsent(new ClassKey(role, worker.getGroup(), worker.compatibility()),
                            ignored -> new HashSet<>()).add(worker.getGenerationId());
                }
            }
        }
        List<Node> nodes = classes.entrySet().stream().map(entry -> new Node(entry.getKey(), entry.getValue(),
                        JsonUtils.toString(entry.getKey().descriptor())))
                .sorted(Comparator.comparing((Node n) -> n.key().role())
                        .thenComparing(n -> String.valueOf(n.key().policyGroup()))
                        .thenComparing(Node::order)).toList();
        BitSet[] neighbors = new BitSet[nodes.size()];
        for (int i = 0; i < nodes.size(); i++) { neighbors[i] = new BitSet(nodes.size()); }
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = 0; j < i; j++) {
                ClassKey a = nodes.get(i).key();
                ClassKey b = nodes.get(j).key();
                if (Objects.equals(a.policyGroup(), b.policyGroup())
                        && (a.descriptor() == null) == (b.descriptor() == null) && (a.role() == b.role()
                        || RoleCompatibility.matches(a.role(), a.descriptor(), b.role(), b.descriptor()))) {
                    neighbors[i].set(j);
                    neighbors[j].set(i);
                }
            }
        }
        BitSet available = new BitSet(nodes.size());
        available.set(0, nodes.size());
        List<CompatibilityGroup> groups = new ArrayList<>();
        enumerate(new BitSet(), available, new BitSet(), neighbors, nodes, roles, groups);
        return CompatibilityGroup.weighted(groups, roles);
    }

    // Same-role classes are adjacent within the legacy/reporting boundary: a
    // maximal clique includes all classes that can safely share its cross-role combinations.
    private static void enumerate(BitSet chosen, BitSet available, BitSet excluded, BitSet[] neighbors,
                                  List<Node> nodes, List<RoleType> roles, List<CompatibilityGroup> groups) {
        BitSet possible = (BitSet) chosen.clone();
        possible.or(available);
        for (RoleType role : roles) {
            boolean present = false;
            for (int i = possible.nextSetBit(0); i >= 0; i = possible.nextSetBit(i + 1)) {
                if (nodes.get(i).key().role() == role) { present = true; break; }
            }
            if (!present) { return; }
        }
        if (available.isEmpty() && excluded.isEmpty()) {
            var descriptors = new EnumMap<RoleType, Set<RoleCompatibility>>(RoleType.class);
            var members = new EnumMap<RoleType, Set<Long>>(RoleType.class);
            String policyGroup = null;
            for (int i = chosen.nextSetBit(0); i >= 0; i = chosen.nextSetBit(i + 1)) {
                Node node = nodes.get(i);
                policyGroup = node.key().policyGroup();
                descriptors.computeIfAbsent(node.key().role(), ignored -> new HashSet<>()).add(node.key().descriptor());
                members.computeIfAbsent(node.key().role(), ignored -> new HashSet<>()).addAll(node.generations());
            }
            groups.add(CompatibilityGroup.unweighted(new CompatibilityGroup.Key(policyGroup, descriptors), members));
            return;
        }
        BitSet pivots = (BitSet) available.clone();
        pivots.or(excluded);
        int pivot = -1;
        int maximum = -1;
        for (int i = pivots.nextSetBit(0); i >= 0; i = pivots.nextSetBit(i + 1)) {
            BitSet adjacent = (BitSet) available.clone();
            adjacent.and(neighbors[i]);
            if (adjacent.cardinality() > maximum) { pivot = i; maximum = adjacent.cardinality(); }
        }
        BitSet branches = (BitSet) available.clone();
        if (pivot >= 0) { branches.andNot(neighbors[pivot]); }
        for (int i = branches.nextSetBit(0); i >= 0; i = branches.nextSetBit(i + 1)) {
            BitSet next = (BitSet) available.clone();
            next.and(neighbors[i]);
            BitSet previouslyConsidered = (BitSet) excluded.clone();
            previouslyConsidered.and(neighbors[i]);
            chosen.set(i);
            enumerate(chosen, next, previouslyConsidered, neighbors, nodes, roles, groups);
            chosen.clear(i);
            available.clear(i);
            excluded.set(i);
        }
    }
}
