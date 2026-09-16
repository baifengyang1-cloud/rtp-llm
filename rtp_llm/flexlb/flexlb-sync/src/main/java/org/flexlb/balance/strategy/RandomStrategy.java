package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;
import org.flexlb.service.VitCacheDirectory;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.concurrent.ThreadLocalRandom;

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
        if (role != RoleType.VIT) {
            throw new IllegalArgumentException(
                    "RANDOM endpoint selection is supported only for VIT");
        }
        if (vitCacheDirectory != null && (context.getRequest().getSelectedVit() != null
                || (context.getRequest().getMediaKeys() != null && !context.getRequest().getMediaKeys().isEmpty()))) {
            ServerStatus route = context.getRequest().getSelectedVit() == null
                    ? vitCacheDirectory.select(context, group) : vitCacheDirectory.validate(context, group);
            if (!route.isSuccess()) {
                return null;
            }
            String address = route.getServerIp() + ":" + route.getHttpPort();
            WorkerEndpoint.GenerationPin pin = workerDirectory.captureEndpoint(RoleType.VIT, address);
            if (pin == null) {
                return null;
            }
            WorkerStatus status = pin.endpoint().getStatus();
            if (group != null && !group.equals(status.topologySnapshot().group())) {
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
                if (group != null && !group.equals(topology.group())) {
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
