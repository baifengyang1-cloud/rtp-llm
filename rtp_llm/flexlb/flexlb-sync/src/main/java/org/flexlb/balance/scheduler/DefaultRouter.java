package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.balance.strategy.CompatibilityCandidates;
import org.flexlb.balance.strategy.CompatibilityGroupBalancer;
import org.flexlb.balance.strategy.CompatibilityPlanner;
import org.flexlb.service.monitor.CompatibilityGroupReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Component
public class DefaultRouter {

    private final CostBasedPrefillStrategy prefillSelector;
    private final DecodeSelector decodeSelector;
    private final RandomStrategy vitSelector;
    private final ConfigService configService;
    private final List<RoleType> requiredRoles;
    private final WorkerDirectory workerDirectory;
    private final CompatibilityGroupBalancer groupBalancer = new CompatibilityGroupBalancer();
    private final CompatibilityPlanner compatibilityPlanner;
    private CompatibilityGroupReporter groupReporter;

    @Autowired(required = false)
    public void setCompatibilityGroupReporter(CompatibilityGroupReporter reporter) { this.groupReporter = reporter; }

    @Autowired
    public DefaultRouter(
            CostBasedPrefillStrategy prefillSelector,
            DecodeSelector decodeSelector,
            RandomStrategy vitSelector,
            ConfigService configService,
            ModelMetaConfig modelMetaConfig,
            WorkerDirectory workerDirectory) {
        this.prefillSelector = Objects.requireNonNull(
                prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(
                decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(
                vitSelector, "vitSelector");
        this.configService = Objects.requireNonNull(
                configService, "configService");
        this.workerDirectory = Objects.requireNonNull(workerDirectory, "workerDirectory");
        this.requiredRoles = List.copyOf(
                Objects.requireNonNull(
                        modelMetaConfig, "modelMetaConfig").requiredRoles());
        this.compatibilityPlanner = new CompatibilityPlanner(workerDirectory, requiredRoles);
    }

    public Response routeVit(BalanceContext context) {
        Response invalid = validateRequest(context);
        if (invalid != null) {
            return invalid;
        }
        try (SelectedRole selected = vitSelector.select(context, RoleType.VIT, resolvePolicyGroup(context))) {
            if (selected == null) {
                return Response.error(context.getRequest().getSelectedVit() == null
                        ? StrategyErrorType.NO_VIT_WORKER : StrategyErrorType.VIT_ROUTE_STALE);
            }
            return Response.buildSuccessResponse(
                    buildSuccessResponse(List.of(selected.serverStatus())), false);
        }
    }

    public PlacementResult<RouteAdmission, PlacementKey> select(BalanceContext context) {
        return select(context, resolvePolicyGroup(context));
    }

    public PlacementResult<RouteAdmission, PlacementKey> select(BalanceContext context, String policyGroup) {
        Response validationFailure = validateRequest(context);
        if (validationFailure != null) { return PlacementResult.rejected(validationFailure); }
        DecodeBinding decodeAdmission = DecodeBinding.capture(context);
        try (PinnedRouting routing = selectAll(context, requiredRoles, policyGroup, decodeAdmission)) {
            if (routing.attempt != null) {
                routing.attempt.workerSelection(routing.blocker() != null ? "blocked" : routing.failure() != null ? "rejected" : "selected");
            }
            if (routing.blocker() != null) {
                return PlacementResult.blocked(routing.blocker(), routing.failure(), routing.diagnostics);
            }
            if (routing.failure() != null) {
                return PlacementResult.rejected(routing.failure(), routing.diagnostics);
            }
            try {
                RouteAdmission admission = RouteAdmission.prepare(context, routing.selections(),
                        buildSuccessResponse(routing.serverStatuses()), decodeAdmission);
                admission.observeCompatibilityGroup(routing.attempt);
                return PlacementResult.success(admission);
            } catch (RuntimeException | Error failure) {
                if (routing.attempt != null) { routing.attempt.admission(false); }
                throw failure;
            }
        }
    }

    private Response validateRequest(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            Logger.error("masterRequest is null");
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        return null;
    }

    private PinnedRouting selectAll(BalanceContext context, List<RoleType> roles, String policyGroup,
                                   DecodeBinding decodeAdmission) {
        List<SelectedRole> selected = new ArrayList<>(roles.size());
        CompatibilityCandidates compatibility = compatibilityPlanner.prepare(context, decodeAdmission, policyGroup);
        var complete = compatibility.completeGroups();
        var chosen = groupBalancer.select(policyGroup, roles, complete);
        CompatibilityGroupReporter.Attempt attempt = groupReporter == null ? null
                : groupReporter.selected(policyGroup, roles, complete, chosen);
        if (chosen == null) {
            for (RoleType role : roles) {
                if (!compatibility.anyEligible(role)) {
                    var failure = switch (role) {
                        case PREFILL, PDFUSION -> prefillSelector.unavailable(context, role, policyGroup, compatibility.filter(role));
                        case DECODE -> decodeSelector.unavailable(decodeAdmission, policyGroup, compatibility.filter(role));
                        default -> PlacementResult.<SelectedRole, RoleType>blocked(role);
                    };
                    if (failure != null && failure.status() == PlacementResult.Status.REJECTED) {
                        return new PinnedRouting(selected, null, failure.failure(), failure.diagnostics());
                    }
                    return new PinnedRouting(selected, compatibilityBlocker(roles, role, policyGroup),
                            failure == null ? null : failure.failure(), failure == null ? null : failure.diagnostics());
                }
            }
            return new PinnedRouting(selected, compatibilityBlocker(roles, roles.getFirst(), policyGroup), null, null);
        }
        compatibility.restrictTo(chosen);
        String group = policyGroup;
        if (StringUtils.isNotBlank(policyGroup)) {
            Logger.info(
                    "Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.getRequestId(),
                    "trafficPolicy",
                    group);
        }

        try {
            for (RoleType role : roles) {
                PlacementResult<SelectedRole, RoleType> result =
                        selectRole(context, role, group, decodeAdmission, compatibility);
                if (role == RoleType.VIT && result.status() == PlacementResult.Status.REJECTED
                        && result.failure().getCode() == StrategyErrorType.VIT_ROUTE_STALE.getErrorCode()) {
                    SelectedRole replacement = vitSelector.reselect(context, group, compatibility.filter(role));
                    if (replacement != null) {
                        result = PlacementResult.success(replacement);
                    }
                }
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    Logger.debug(
                            "Failed to select {} worker for request {}",
                            role.getCode(), context.getRequestId());
                    if (result.status() == PlacementResult.Status.REJECTED) {
                        return new PinnedRouting(
                                selected, null, result.failure(), result.diagnostics(), attempt);
                    }
                    if (result.status() == PlacementResult.Status.BLOCKED) {
                        return new PinnedRouting(
                                selected,
                                compatibilityBlocker(roles, result.blocker(), group),
                                result.failure(), result.diagnostics(), attempt);
                    }
                    throw new IllegalStateException("unexpected selector result: "
                            + result.status());
                }
                SelectedRole selection = result.value();
                try {
                    selected.add(selection);
                } catch (RuntimeException | Error appendFailure) {
                    closeSelection(selection, appendFailure);
                    throw appendFailure;
                }
                if (!compatibility.select(role, selection.endpoint().getStatus())) {
                    return new PinnedRouting(selected, compatibilityBlocker(roles, role, group), null, null, attempt);
                }
                if (StringUtils.isBlank(policyGroup)) {
                    group = selection.serverStatus().getGroup();
                }
            }
            return new PinnedRouting(selected, null, null, null, attempt);
        } catch (RuntimeException | Error failure) {
            if (attempt != null) { attempt.workerSelection("error"); }
            closeSelections(selected, failure);
            throw failure;
        }
    }

    String resolvePolicyGroup(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return null;
        }
        FlexlbConfig config = context.getConfig() != null
                ? context.getConfig()
                : configService.loadBalanceConfig();
        if (config == null || config.getRouter().getGroupSelector() == null) {
            return null;
        }
        return config.getRouter().getGroupSelector()
                .resolveTargetGroup(context.getRequest())
                .orElse(null);
    }

    private static PlacementKey compatibilityBlocker(List<RoleType> roles, RoleType failedRole, String group) {
        List<PlacementKey> keys = new ArrayList<>();
        keys.add(new PlacementKey(failedRole, group));
        for (RoleType role : roles) { keys.add(new PlacementKey(role, group)); }
        return PlacementKey.anyOf(keys);
    }

    private PlacementResult<SelectedRole, RoleType> selectRole(
            BalanceContext context, RoleType role, String group, DecodeBinding decodeAdmission,
            CompatibilityCandidates compatibility) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    prefillSelector.select(context, role, group, compatibility.filter(role));
            case DECODE -> decodeSelector.select(decodeAdmission, group, compatibility.filter(role));
            case VIT -> {
                SelectedRole selected = vitSelector.select(context, role, group, compatibility.filter(role));
                if (selected == null && context.getRequest().getSelectedVit() != null) {
                    yield PlacementResult.rejected(Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                yield selectedOrBlocked(selected, role);
            }
            case FRONTEND -> throw new IllegalArgumentException(
                    "Endpoint selection is not supported for FRONTEND");
        };
    }

    private static PlacementResult<SelectedRole, RoleType> selectedOrBlocked(
            SelectedRole selected, RoleType role) {
        return selected == null
                ? PlacementResult.blocked(role)
                : PlacementResult.success(selected);
    }

    private static List<ServerStatus> serverStatuses(List<SelectedRole> selections) {
        List<ServerStatus> statuses = new ArrayList<>(selections.size());
        for (SelectedRole selection : selections) {
            statuses.add(selection.serverStatus());
        }
        return statuses;
    }

    private static Throwable closeSelections(
            List<SelectedRole> selections,
            Throwable primaryFailure) {
        Throwable failure = primaryFailure;
        for (int index = selections.size() - 1; index >= 0; index--) {
            failure = closeSelection(selections.get(index), failure);
        }
        return failure;
    }

    private static Throwable closeSelection(
            SelectedRole selection,
            Throwable primaryFailure) {
        try {
            selection.close();
        } catch (Throwable closeFailure) {
            return appendFailure(primaryFailure, closeFailure);
        }
        return primaryFailure;
    }

    private static Throwable appendFailure(
            Throwable primaryFailure,
            Throwable cleanupFailure) {
        if (primaryFailure == null) {
            return cleanupFailure;
        }
        if (primaryFailure != cleanupFailure) {
            primaryFailure.addSuppressed(cleanupFailure);
        }
        return primaryFailure;
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "route selection cleanup failed", failure);
    }

    private static Response buildSuccessResponse(
            List<ServerStatus> statuses) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(statuses);
        return response;
    }

    private static final class PinnedRouting implements AutoCloseable {
        private final List<SelectedRole> selections;
        private final PlacementKey blocker;
        private final Response failure;
        private final Map<String, Object> diagnostics;
        private final CompatibilityGroupReporter.Attempt attempt;

        private PinnedRouting(
                List<SelectedRole> selections,
                PlacementKey blocker,
                Response failure, Map<String, Object> diagnostics) {
            this(selections, blocker, failure, diagnostics, null);
        }

        private PinnedRouting(List<SelectedRole> selections, PlacementKey blocker, Response failure,
                              Map<String, Object> diagnostics, CompatibilityGroupReporter.Attempt attempt) {
            this.selections = selections;
            this.blocker = blocker;
            this.failure = failure;
            this.diagnostics = diagnostics;
            this.attempt = attempt;
        }

        private PlacementKey blocker() {
            return blocker;
        }

        private Response failure() {
            return failure;
        }

        private List<SelectedRole> selections() {
            return selections;
        }

        private List<ServerStatus> serverStatuses() {
            return DefaultRouter.serverStatuses(selections);
        }

        @Override
        public void close() {
            Throwable failure = closeSelections(selections, null);
            if (failure != null) {
                throw propagate(failure);
            }
        }
    }

}
