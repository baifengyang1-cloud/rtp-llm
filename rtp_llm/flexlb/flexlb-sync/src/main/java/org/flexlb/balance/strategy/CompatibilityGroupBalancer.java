package org.flexlb.balance.strategy;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.dao.route.RoleType;

import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Smooth weighted round-robin local to one model router, scoped by policy boundary
 * and required roles. Credit counts request selections using current normalized
 * weights. Resizing preserves credit; changing keys centers survivors before new
 * groups start at zero. Selecting consumes a turn without a success/refund callback.
 * Bound retained scopes to 64 using the planner's cache policy. Empty scopes are removed immediately;
 * a removed or evicted scope starts with zero credit if it returns.
 */
public final class CompatibilityGroupBalancer {
    private static final int MAX_SCOPES = 64;
    private record Scope(String policyGroup, Set<RoleType> roles) { }
    private static final class Rotation {
        private List<CompatibilityGroup> source;
        private CompatibilityGroup[] observed = new CompatibilityGroup[0];
        private List<CompatibilityGroup> ordered = List.of();
        private double[] current = new double[0];
        private double[] shares = new double[0];

        void configure(List<CompatibilityGroup> groups) {
            if (source == groups && observed.length == groups.size()) {
                boolean unchanged = true;
                for (int i = 0; i < observed.length; i++) { unchanged &= observed[i] == groups.get(i); }
                if (unchanged) { return; }
            }
            var next = groups.stream().sorted(Comparator.comparing(CompatibilityGroup::id)).toList();
            Set<CompatibilityGroup.Key> present = new HashSet<>();
            double sum = 0.0;
            for (CompatibilityGroup group : next) {
                if (!(group.weight() > 0.0) || !Double.isFinite(group.weight())) {
                    throw new IllegalArgumentException("group weight must be positive and finite");
                }
                if (!present.add(group.key())) { throw new IllegalArgumentException("duplicate compatibility group"); }
                sum += group.weight();
            }
            double[] normalized = shares.length == next.size() ? shares : new double[next.size()];
            if (Double.isFinite(sum)) {
                for (int i = 0; i < next.size(); i++) { normalized[i] = next.get(i).weight() / sum; }
            } else {
                double maximumWeight = 0.0;
                for (CompatibilityGroup group : next) { maximumWeight = Math.max(maximumWeight, group.weight()); }
                double scaledSum = 0.0;
                for (int i = 0; i < next.size(); i++) {
                    normalized[i] = next.get(i).weight() / maximumWeight;
                    scaledSum += normalized[i];
                }
                for (int i = 0; i < normalized.length; i++) { normalized[i] /= scaledSum; }
            }
            boolean sameKeys = ordered.size() == next.size();
            for (int i = 0; sameKeys && i < next.size(); i++) {
                sameKeys = ordered.get(i).key().equals(next.get(i).key());
            }
            double[] credit = current;
            if (!sameKeys) {
                Map<CompatibilityGroup.Key, Double> previous = new HashMap<>();
                for (int i = 0; i < ordered.size(); i++) { previous.put(ordered.get(i).key(), current[i]); }
                credit = new double[next.size()];
                double survivingCredit = 0.0;
                int survivors = 0;
                for (int i = 0; i < next.size(); i++) {
                    Double retained = previous.get(next.get(i).key());
                    if (retained != null) {
                        credit[i] = retained;
                        survivingCredit += retained;
                        survivors++;
                    }
                }
                if (survivors > 0) {
                    double mean = survivingCredit / survivors;
                    for (int i = 0; i < next.size(); i++) {
                        if (previous.containsKey(next.get(i).key())) { credit[i] -= mean; }
                    }
                }
            }
            observed = groups.toArray(CompatibilityGroup[]::new);
            ordered = next; current = credit; shares = normalized; source = groups;
        }
    }
    private final Cache<Scope, Rotation> rotations = Caffeine.newBuilder().maximumSize(MAX_SCOPES).build();

    public CompatibilityGroup select(String policyGroup, List<RoleType> roles, List<CompatibilityGroup> groups) {
        Scope scope = new Scope(policyGroup, Set.copyOf(roles));
        if (groups.isEmpty()) {
            rotations.invalidate(scope);
            return null;
        }
        Rotation rotation = rotations.get(scope, ignored -> new Rotation());
        // An in-flight selection may finish on an evicted rotation. Its plan is
        // released when that call returns; subsequent calls use a fresh rotation.
        synchronized (rotation) {
            rotation.configure(groups);
            int selected = -1;
            double maximum = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < rotation.ordered.size(); i++) {
                double current = rotation.current[i] += rotation.shares[i];
                if (current > maximum) { maximum = current; selected = i; }
            }
            if (selected < 0) { return null; }
            rotation.current[selected] -= 1.0;
            return rotation.ordered.get(selected);
        }
    }
}
