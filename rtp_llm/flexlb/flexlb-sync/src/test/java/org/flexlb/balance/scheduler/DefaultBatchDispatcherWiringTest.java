package org.flexlb.balance.scheduler;

import org.flexlb.balance.composition.DeliveryBindingConfiguration;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.VitCacheDirectory;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DefaultBatchDispatcherWiringTest {
    @Test
    void replacementSelectorDoesNotCreateStartupDependencyCycle() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        ModelMetaConfig topology = mock(ModelMetaConfig.class);
        when(topology.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT));
        try (var context = new AnnotationConfigApplicationContext()) {
            context.registerBean(ConfigService.class, () -> config);
            context.registerBean(ModelMetaConfig.class, () -> topology);
            context.registerBean(EngineGrpcClient.class, () -> mock(EngineGrpcClient.class));
            context.registerBean(RequestRegistry.class, () -> mock(RequestRegistry.class));
            context.registerBean(EndpointEventProjector.class, () -> mock(EndpointEventProjector.class));
            context.registerBean(BatchSchedulerReporter.class, () -> mock(BatchSchedulerReporter.class));
            context.registerBean(VitCacheDirectory.class, () -> mock(VitCacheDirectory.class));
            context.register(PlacementAvailability.class, DeliveryBindingConfiguration.class,
                    DefaultBatchDispatcher.class, EndpointRegistry.class, WorkerDirectory.class, RandomStrategy.class);
            context.refresh();
            assertInstanceOf(BatchDeliveryStrategy.class, context.getBean(DeliveryStrategy.class));
            assertNotNull(context.getBean(RandomStrategy.class));
            assertNotNull(context.getBean(DefaultBatchDispatcher.class));
        }
    }
}
