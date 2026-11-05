package org.flexlb.balance.strategy;

import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.JsonUtils;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One complete cross-role clique of descriptor classes, with deduplicated physical generations.
 * The key excludes endpoint counts so resizing a class preserves rotation credit.
 * The canonical JSON hash identifies telemetry; routing equality uses the full key.
 */
public record CompatibilityGroup(Key key, Map<RoleType, Set<Long>> members,
                                 Map<RoleType, Double> effectiveCounts, double weight, String id) {
    public record Key(String policyGroup, Map<RoleType, Set<RoleCompatibility>> descriptors) {
        public Key {
            var copy = new EnumMap<RoleType, Set<RoleCompatibility>>(RoleType.class);
            descriptors.forEach((role, values) -> copy.put(role,
                    Collections.unmodifiableSet(new HashSet<>(values)))); // null is the legacy class
            descriptors = Collections.unmodifiableMap(copy);
        }

        String canonical() {
            List<Object> result = new ArrayList<>();
            result.add(policyGroup);
            descriptors.forEach((role, values) -> result.add(List.of(role.name(),
                    values.stream().map(JsonUtils::toString).sorted().toList())));
            return JsonUtils.toString(result);
        }
    }

    public CompatibilityGroup {
        var copy = new EnumMap<RoleType, Set<Long>>(RoleType.class);
        members.forEach((role, values) -> copy.put(role, Set.copyOf(values)));
        members = Collections.unmodifiableMap(copy);
        effectiveCounts = Map.copyOf(effectiveCounts);
    }

    static CompatibilityGroup unweighted(Key key, Map<RoleType, Set<Long>> members) {
        return new CompatibilityGroup(key, members, Map.of(), 0.0, identifier(key));
    }

    public boolean contains(RoleType role, long generation) {
        return members.getOrDefault(role, Set.of()).contains(generation);
    }

    /** Exclude incomplete groups before splitting each worker's contribution across its memberships. */
    public static List<CompatibilityGroup> weighted(List<CompatibilityGroup> candidates, List<RoleType> roles) {
        if (roles.isEmpty()) { return List.of(); }
        List<CompatibilityGroup> complete = candidates.stream().filter(group -> roles.stream()
                .allMatch(role -> !group.members().getOrDefault(role, Set.of()).isEmpty())).distinct().toList();
        Map<Long, Integer> memberships = new HashMap<>();
        for (CompatibilityGroup group : complete) {
            for (RoleType role : roles) {
                for (long generation : group.members().get(role)) { memberships.merge(generation, 1, Integer::sum); }
            }
        }
        return complete.stream().map(group -> {
            var effective = new EnumMap<RoleType, Double>(RoleType.class);
            for (RoleType role : roles) {
                double count = 0.0;
                for (long generation : group.members().get(role)) { count += 1.0 / memberships.get(generation); }
                effective.put(role, count);
            }
            return new CompatibilityGroup(group.key(), group.members(), effective,
                    geometricMean(effective.values().stream().mapToDouble(Double::doubleValue).toArray()), group.id());
        }).toList();
    }

    static double geometricMean(double... counts) {
        if (counts.length == 0) { throw new IllegalArgumentException("required roles are empty"); }
        double logarithm = 0.0;
        for (double count : counts) {
            if (!(count > 0.0) || !Double.isFinite(count)) {
                throw new IllegalArgumentException("effective counts must be positive and finite");
            }
            logarithm += Math.log(count) / counts.length;
        }
        return Math.exp(logarithm);
    }

    private static String identifier(Key key) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(key.canonical().getBytes(StandardCharsets.UTF_8))).substring(0, 24);
        } catch (NoSuchAlgorithmException impossible) { throw new IllegalStateException(impossible); }
    }
}
