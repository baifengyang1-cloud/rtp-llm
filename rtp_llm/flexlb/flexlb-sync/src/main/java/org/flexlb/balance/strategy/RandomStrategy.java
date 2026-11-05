package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import org.flexlb.service.VitCacheDirectory;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.LongPredicate;

/** Random selection for the stateless VIT role. */
@Component
public final class RandomStrategy {
    private static final Logger LOGGER =
            LoggerFactory.getLogger(RandomStrategy.class);

    private final WorkerDirectory workerDirectory;
    @Autowired
    private VitCacheDirectory vitCacheDirectory;

    public RandomStrategy(WorkerDirectory workerDirectory) {
        this.workerDirectory = workerDirectory;
    }

    public SelectedRole select(
            BalanceContext context, RoleType role, String group) {
        return select(context, role, group, true);
    }

    public SelectedRole select(BalanceContext context, RoleType role, String group, LongPredicate eligible) {
        return select(context, role, group, true, eligible);
    }

    /** Select a live replacement without mutating the frontend's original hint. */
    public SelectedRole reselect(BalanceContext context, String group) {
        return select(context, RoleType.VIT, group, false);
    }

    public SelectedRole reselect(BalanceContext context, String group, LongPredicate eligible) {
        return select(context, RoleType.VIT, group, false, eligible);
    }

    public SelectedRole reselectCompatible(BalanceContext context, String group, RoleCompatibility expected) {
        var generations = workerDirectory.routingStatusSnapshot(RoleType.VIT, group).stream()
                .filter(worker -> Objects.equals(expected, worker.compatibility()))
                .map(WorkerStatus::getGenerationId).collect(java.util.stream.Collectors.toSet());
        return reselect(context, group, generations::contains);
    }

    private SelectedRole select(BalanceContext context, RoleType role, String group, boolean honorSelected) {
        return select(context, role, group, honorSelected, ignored -> true);
    }

    private SelectedRole select(BalanceContext context, RoleType role, String group,
                                boolean honorSelected, LongPredicate eligible) {
        if (role != RoleType.VIT) {
            throw new IllegalArgumentException(
                    "RANDOM endpoint selection is supported only for VIT");
        }
        boolean pinned = honorSelected && context.getRequest().getSelectedVit() != null;
        if (vitCacheDirectory != null && (pinned
                || (context.getRequest().getMediaKeys() != null && !context.getRequest().getMediaKeys().isEmpty()))) {
            ServerStatus route = pinned ? vitCacheDirectory.validate(context, group, eligible)
                    : honorSelected ? vitCacheDirectory.select(context, group, eligible)
                    : vitCacheDirectory.selectReplacement(context, group, eligible);
            if (!route.isSuccess()) {
                return null;
            }
            String address = route.getServerIp() + ":" + route.getHttpPort();
            WorkerEndpoint.GenerationPin pin = workerDirectory.captureEndpoint(RoleType.VIT, address);
            if (pin == null) {
                return null;
            }
            WorkerStatus status = pin.endpoint().getStatus();
            if (!eligible.test(status.getGenerationId())
                    || (group != null || !honorSelected) && !Objects.equals(group, status.topologySnapshot().group())) {
                pin.close();
                return null;
            }
            return SelectedRole.stateless(pin, route);
        }
        List<String> addresses =
                workerDirectory.endpointAddressSnapshot(RoleType.VIT);
        if (addresses.isEmpty()) {
            return null;
        }

        int start = ThreadLocalRandom.current().nextInt(addresses.size());
        for (int offset = 0; offset < addresses.size(); offset++) {
            String address = addresses.get((start + offset) % addresses.size());
            WorkerEndpoint.GenerationPin pin =
                    workerDirectory.captureEndpoint(RoleType.VIT, address);
            if (pin == null) {
                continue;
            }
            try {
                WorkerStatus status = pin.endpoint().getStatus();
                WorkerStatus.TopologySnapshot topology =
                        status.topologySnapshot();
                if (!eligible.test(status.getGenerationId())) { continue; }
                if ((group != null || !honorSelected) && !Objects.equals(group, topology.group())) {
                    continue;
                }
                WorkerStatus.EngineObservation engine =
                        status.committedEngineObservation();
                WorkerEndpoint.GenerationPin selectedPin = pin;
                pin = null;
                return selected(
                        selectedPin,
                        topology,
                        engine,
                        context.getRequestId());
            } finally {
                if (pin != null) {
                    pin.close();
                }
            }
        }
        LOGGER.warn(
                "No VIT worker available out of {} registered workers",
                addresses.size());
        return null;
    }

    private static SelectedRole selected(
            WorkerEndpoint.GenerationPin pin,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.EngineObservation engine,
            long requestId) {
        try {
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(RoleType.VIT);
            result.setRequestId(requestId);
            result.setGroup(topology.group());
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(engine.dpRank());

            WorkerEndpoint.GenerationPin owned = pin;
            pin = null;
            return SelectedRole.stateless(owned, result);
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }
}
