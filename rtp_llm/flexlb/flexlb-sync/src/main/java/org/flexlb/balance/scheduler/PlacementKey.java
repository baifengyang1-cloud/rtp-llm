package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import java.util.Objects;
import java.util.Collection;
import java.util.Set;

/**
 * Logical capacity domain. A null group means any group for the role; a
 * non-null endpoint identifies one exact worker within the group.
 */
public record PlacementKey(RoleType role, String group, String endpoint, Set<PlacementKey> alternatives) {

    public PlacementKey(RoleType role, String group, String endpoint) {
        this(role, group, endpoint, Set.of());
    }

    public PlacementKey(RoleType role, String group) {
        this(role, group, null);
    }

    public PlacementKey {
        Objects.requireNonNull(role, "role");
        alternatives = Set.copyOf(alternatives);
        if (group != null && group.isBlank()) {
            group = null;
        }
        if (endpoint != null && endpoint.isBlank()) {
            endpoint = null;
        }
    }

    public static PlacementKey anyOf(Collection<PlacementKey> keys) {
        Set<PlacementKey> alternatives = Set.copyOf(keys);
        if (alternatives.isEmpty()) { throw new IllegalArgumentException("empty placement dependencies"); }
        if (alternatives.size() == 1) { return alternatives.iterator().next(); }
        PlacementKey first = keys.iterator().next();
        return new PlacementKey(first.role(), first.group(), null, alternatives);
    }

    public Set<PlacementKey> dependencies() { return alternatives.isEmpty() ? Set.of(this) : alternatives; }

    public static PlacementKey anyGroup(RoleType role) {
        return new PlacementKey(role, null);
    }

    public static PlacementKey exact(
            RoleType role,
            String group,
            String endpoint) {
        return new PlacementKey(
                role, group, Objects.requireNonNull(endpoint, "endpoint"));
    }
}
