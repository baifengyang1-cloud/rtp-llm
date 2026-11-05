package org.flexlb.balance.scheduler;

import ch.qos.logback.classic.Level;
import io.netty.channel.EventLoopGroup;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.RoleCompatibility;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.CompatibilityGroupReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.JsonUtils;
import org.slf4j.LoggerFactory;
import reactor.netty.resources.LoopResources;

import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Opt-in standalone benchmark, outside Surefire's test-name patterns. Measures the
 * real DefaultRouter.select + RouteAdmission.close path with real endpoint indexes,
 * generation pins, selectors, predictors, and a no-op metrics sink. No Engine RPC or
 * capacity reservation is part of this routing-only measurement.
 */
public final class CompatibilityRoutingBenchmark {
    private static final List<RoleType> ROLES = List.of(RoleType.DECODE, RoleType.PREFILL, RoleType.VIT);
    private static final com.sun.management.ThreadMXBean THREADS =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    private static final int SAMPLE_CAPACITY = 262144;

    private record Scenario(String name, int workersPerPdRole, int targets, int threads,
                            boolean legacy, boolean overlap, boolean cold, boolean incomplete) { }

    private record Sample(long operations, long bytes, long cpuNanos, long ended,
                          long[] latencies, long[] targetSelections, long checksum) { }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) { throw new IllegalArgumentException("variant output.json"); }
        for (String name : List.of("ROOT", "flexlbLogger", "syncLogger", "pvLogger")) {
            ((ch.qos.logback.classic.Logger) LoggerFactory.getLogger(name)).setLevel(Level.ERROR);
        }
        THREADS.setThreadAllocatedMemoryEnabled(true);
        THREADS.setThreadCpuTimeEnabled(true);
        var scenarios = List.of(
                new Scenario("legacy_1", 1, 1, 1, true, false, false, false),
                new Scenario("reported_1", 1, 1, 1, false, false, false, false),
                new Scenario("legacy_16", 16, 1, 1, true, false, false, false),
                new Scenario("reported_16", 16, 1, 1, false, false, false, false),
                new Scenario("reported_128", 128, 1, 1, false, false, false, false),
                new Scenario("reported_750", 750, 1, 1, false, false, false, false),
                new Scenario("rollout4_128", 128, 4, 1, false, false, false, false),
                new Scenario("rollout16_128", 128, 16, 1, false, false, false, false),
                new Scenario("overlap16_128", 128, 16, 1, false, true, false, false),
                new Scenario("incomplete16_128", 128, 16, 1, false, false, false, true),
                new Scenario("cold2_128", 128, 2, 1, false, false, true, false),
                new Scenario("varying128_128", 128, 1, 1, false, false, false, false),
                new Scenario("topology16_128", 128, 16, 1, false, false, false, false),
                new Scenario("policy4_128", 128, 4, 1, false, false, false, false),
                new Scenario("concurrent_reported_128", 128, 1, 8, false, false, false, false),
                new Scenario("concurrent_rollout16_128", 128, 16, 8, false, false, false, false));
        List<Map<String, Object>> results = new ArrayList<>();
        String only = System.getProperty("flexlb.benchmark.scenarios", "");
        for (Scenario scenario : scenarios) {
            if (!only.isBlank() && !Arrays.asList(only.split(",")).contains(scenario.name())) { continue; }
            try (Fixture fixture = new Fixture(scenario); ExecutorService pool = Executors.newFixedThreadPool(scenario.threads())) {
                var state = ThreadLocal.withInitial(() -> new WorkerState(scenario.targets()));
                phase(fixture, pool, state, Long.getLong("flexlb.benchmark.warmupMs", 2000L), false);
                for (int round = 0; round < Integer.getInteger("flexlb.benchmark.rounds", 3); round++) {
                    Map<String, Object> row = phase(fixture, pool, state,
                            Long.getLong("flexlb.benchmark.measureMs", 1000L), true);
                    row.put("variant", args[0]);
                    row.put("scenario", scenario.name());
                    row.put("workers_per_pd_role", scenario.workersPerPdRole());
                    row.put("targets", scenario.targets());
                    row.put("threads", scenario.threads());
                    row.put("round", round);
                    results.add(row);
                    Files.writeString(Path.of(args[1]), JsonUtils.toFormattedString(results) + "\n");
                    System.out.println(JsonUtils.toString(row));
                }
            }
        }
    }

    private static Map<String, Object> phase(Fixture fixture, ExecutorService pool, ThreadLocal<WorkerState> state,
                                             long milliseconds, boolean record) throws Exception {
        var ready = new CountDownLatch(fixture.scenario.threads());
        var start = new CountDownLatch(1);
        long[] deadline = new long[1]; // published by the start latch
        List<Future<Sample>> futures = new ArrayList<>();
        for (int i = 0; i < fixture.scenario.threads(); i++) {
            futures.add(pool.submit(() -> {
                WorkerState worker = state.get();
                ready.countDown();
                start.await();
                return worker.run(fixture, deadline[0], record);
            }));
        }
        ready.await();
        long gcBefore = gcCount();
        long gcMillisBefore = gcMillis();
        long began = System.nanoTime();
        deadline[0] = began + TimeUnit.MILLISECONDS.toNanos(milliseconds);
        start.countDown();
        long operations = 0, bytes = 0, cpu = 0, ended = began, checksum = 0;
        List<long[]> arrays = new ArrayList<>();
        long[] selected = new long[fixture.scenario.targets()];
        for (var future : futures) {
            Sample sample = future.get();
            operations += sample.operations(); bytes += sample.bytes(); cpu += sample.cpuNanos();
            ended = Math.max(ended, sample.ended()); checksum += sample.checksum();
            arrays.add(sample.latencies());
            for (int i = 0; i < selected.length; i++) { selected[i] += sample.targetSelections()[i]; }
        }
        long gcDelta = gcCount() - gcBefore;
        long gcMillisDelta = gcMillis() - gcMillisBefore;
        long[] latencies = new long[arrays.stream().mapToInt(a -> a.length).sum()];
        int offset = 0;
        for (long[] array : arrays) { System.arraycopy(array, 0, latencies, offset, array.length); offset += array.length; }
        Arrays.sort(latencies);
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("operations", operations);
        row.put("elapsed_seconds", (ended - began) / 1e9);
        row.put("qps", operations * 1e9 / (ended - began));
        row.put("cpu_us_per_op", cpu / 1000.0 / operations);
        row.put("bytes_per_op", bytes / (double) operations);
        row.put("p50_us", percentile(latencies, 0.50));
        row.put("p95_us", percentile(latencies, 0.95));
        row.put("p99_us", percentile(latencies, 0.99));
        row.put("latency_samples", latencies.length);
        row.put("target_selections", selected);
        row.put("gc_collections", gcDelta);
        row.put("gc_millis", gcMillisDelta);
        row.put("checksum", checksum);
        return row;
    }

    private static double percentile(long[] values, double p) {
        return values.length == 0 ? 0.0 : values[Math.min(values.length - 1, (int) Math.ceil(values.length * p) - 1)] / 1000.0;
    }

    private static long gcCount() {
        return ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(bean -> Math.max(0, bean.getCollectionCount())).sum();
    }

    private static long gcMillis() {
        return ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(bean -> Math.max(0, bean.getCollectionTime())).sum();
    }

    private static final class WorkerState {
        private final long[] latencies = new long[SAMPLE_CAPACITY];
        private final long[] selected;
        private long requestId;

        WorkerState(int targets) { selected = new long[targets]; }

        Sample run(Fixture fixture, long deadline, boolean record) {
            Arrays.fill(selected, 0L);
            long count = 0, checksum = 0;
            int samples = 0;
            long id = Thread.currentThread().threadId();
            long bytes = THREADS.getThreadAllocatedBytes(id);
            long cpu = THREADS.getCurrentThreadCpuTime();
            long ended = System.nanoTime();
            while (ended < deadline) {
                long began = System.nanoTime();
                if (fixture.scenario.name().startsWith("topology") && (requestId & 127L) == 127L) {
                    fixture.replaceDecode();
                }
                var request = new Request();
                request.setRequestId(++requestId);
                request.setSeqLen(512L);
                request.setMaxNewTokens(32);
                if (fixture.scenario.name().startsWith("varying")) {
                    request.setMaxNewTokens((int) (requestId % 128) * 600);
                }
                if (fixture.scenario.cold()) {
                    request.setBlockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L));
                    request.setCacheKeyBlockSize(64L);
                }
                var context = new BalanceContext(fixture.config);
                context.setRequest(request);
                context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
                var result = fixture.router.select(context, fixture.scenario.name().startsWith("policy") ? "blue" : null);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    throw new IllegalStateException("routing failed: " + result.status());
                }
                try (var admission = result.value()) {
                    int target = fixture.prefillTargets.get(admission.prefillEndpoint());
                    selected[target]++;
                    checksum += admission.decodeEndpoint().getStatus().getGenerationId();
                }
                ended = System.nanoTime();
                if (record && samples < latencies.length) { latencies[samples++] = ended - began; }
                count++;
            }
            cpu = THREADS.getCurrentThreadCpuTime() - cpu;
            bytes = THREADS.getThreadAllocatedBytes(id) - bytes;
            return new Sample(count, bytes, cpu, ended, Arrays.copyOf(latencies, samples), selected.clone(), checksum);
        }
    }

    private static final class Fixture implements AutoCloseable {
        private final Scenario scenario;
        private final FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final RequestRegistry requests;
        private final EndpointRegistry endpoints;
        private final WorkerDirectory workers;
        private final DefaultRouter router;
        private final IdentityHashMap<WorkerEndpoint, Integer> prefillTargets = new IdentityHashMap<>();
        private WorkerEndpoint replaceableDecode;
        private final CacheAwareService cache;
        private final SimpleMeterRegistry meters;
        private int address;

        Fixture(Scenario scenario) throws ReflectiveOperationException {
            this.scenario = scenario;
            config.setScheduler(SchedulerConfig.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens)");
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            boolean realMetrics = System.getProperty("flexlb.benchmark.metrics", "noop").equals("micrometer");
            meters = realMetrics ? new SimpleMeterRegistry() : null;
            FlexMonitor monitor = realMetrics ? new MicrometerFlexMonitor(meters) : new NoOpFlexMonitor();
            var reporter = new BatchSchedulerReporter(monitor);
            var requestReporter = new RequestSchedulerReporter(monitor);
            reporter.init(); requestReporter.init();
            requests = new RequestRegistry(service, reporter, requestReporter);
            endpoints = new EndpointRegistry(service, new EndpointEventProjector(requests), reporter,
                    new RouteDeliveryStrategy(requests, new DeliveryMetrics(reporter)), new PlacementAvailability());
            workers = new WorkerDirectory(endpoints);
            Map<String, Integer> cacheHits = new java.util.HashMap<>();
            for (int i = 0; i < scenario.workersPerPdRole(); i++) {
                int target = i % scenario.targets();
                int tp = scenario.overlap() ? (i / scenario.targets() % 2 == 0 ? 4 : 6) : 2;
                var p = add(RoleType.PREFILL, target, tp, 100000L);
                prefillTargets.put(p, target);
                if (scenario.cold() && target == 0) { cacheHits.put(p.ipPort(), 7); }
                int dTarget = scenario.incomplete() ? 0 : target;
                int dTp = scenario.overlap() ? (i / scenario.targets() % 2 == 0 ? 2 : 3) : 2;
                long available = scenario.cold() && target > 0 ? 50000L : 100000L;
                if (scenario.name().startsWith("varying") && i < scenario.workersPerPdRole() - 1) {
                    available = 10000L + i * 600L + 512L;
                }
                var decode = add(RoleType.DECODE, dTarget, dTp, available);
                if (i == 0) { replaceableDecode = decode; }
            }
            for (int i = 0; i < Math.min(16, scenario.targets()); i++) { add(RoleType.VIT, 0, 1, 100000L); }
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(ROLES);
            cache = new CacheAwareService() {
                public Map<String, Integer> findMatchingEngines(List<Long> keys, RoleType role, List<String> candidates) { return cacheHits; }
                public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus status) { return null; }
                public void removeEngineBlockCache(String address) { }
            };
            var engine = mock(EngineGrpcClient.class);
            var loops = mock(LoopResources.class);
            var group = mock(EventLoopGroup.class);
            when(engine.getEventLoopGroup()).thenReturn(group);
            when(loops.onServer(true)).thenReturn(group);
            when(loops.onServerSelect(true)).thenReturn(group);
            var cacheMetrics = new CacheMetricsReporter();
            var metricField = CacheMetricsReporter.class.getDeclaredField("monitor");
            metricField.setAccessible(true);
            metricField.set(cacheMetrics, monitor);
            cacheMetrics.init();
            var health = new EngineHealthReporter(monitor, cacheMetrics, engine, loops, workers);
            health.init();
            router = new DefaultRouter(new CostBasedPrefillStrategy(workers, cache, health), new DecodeSelector(workers),
                    new RandomStrategy(workers), service, model, workers);
            var groupReporter = new CompatibilityGroupReporter(monitor);
            groupReporter.init();
            router.setCompatibilityGroupReporter(groupReporter);
        }

        private WorkerEndpoint add(RoleType role, int target, int tp, long available) {
            var status = WorkerStatus.createDiscovered(role, "blue", "worker-" + (++address), 8080, 8081, "benchmark");
            return publish(status, target, tp, available);
        }

        private WorkerEndpoint publish(WorkerStatus status, int target, int tp, long available) {
            RoleType role = status.getRole();
            workers.currentOrDiscover(role, status.getIpPort(), () -> status);
            var response = new WorkerStatusResponse();
            response.setRole(role); response.setAlive(true); response.setStatusVersion(1L);
            response.setTotalKvCacheTokens(100000L); response.setAvailableKvCacheTokens(available);
            response.setMaxSeqLen(100000L); response.setMaxBatchTokensSize(100000L);
            response.setCompatibility(scenario.legacy() ? null : descriptor(role, "target-" + target, tp));
            EndpointRegistry.EndpointPublication publication;
            status.lock.lock();
            try { publication = workers.publishPreparedEndpoint(status.getIpPort(), status,
                    status.prepareNewStatus(status.freezeStatusResponse(response))); }
            finally { status.lock.unlock(); }
            publication.statusProjection().run();
            return publication.endpoint();
        }

        private void replaceDecode() {
            var old = replaceableDecode.getStatus();
            EndpointRegistry.DetachedGeneration detached;
            old.lock.lock();
            try { detached = workers.beginRetirement(RoleType.DECODE, old.getIpPort(), old); }
            finally { old.lock.unlock(); }
            workers.completeRetirement(RoleType.DECODE, old.getIpPort(), old, detached, cache,
                    LoggerFactory.getLogger(CompatibilityRoutingBenchmark.class));
            var status = WorkerStatus.createDiscovered(RoleType.DECODE, "blue", old.getIp(), 8080, 8081, "benchmark");
            replaceableDecode = publish(status, 0, 2, 100000L);
        }

        public void close() {
            endpoints.close();
            requests.closeAdmissionAndAwaitMutations();
            requests.closeOutstandingAndTerminalize();
            requests.closeExpiration();
            requests.closePublisher();
            if (meters != null) { meters.close(); }
        }
    }

    private static RoleCompatibility descriptor(RoleType role, String target, int tp) {
        var vision = new RoleCompatibility.Vision("vision", "embedding-v1");
        if (role == RoleType.VIT) { return new RoleCompatibility(1, "", vision, null, null); }
        int heads = 12 / tp;
        var group = new RoleCompatibility.Group("target/default", List.of(0), "MultiHeadAttention", "bf16", "split",
                64, 64, heads, List.of((long) heads * 8192, (long) heads * 8192, 0L, 0L), "full", "head-dim=64");
        return new RoleCompatibility(1, target, vision, new RoleCompatibility.Transfer("cache-store-tcp-v1",
                "model-token-layer-tag-v1", 0, tp, 1, 64, List.of(group), tp, "tp"),
                new RoleCompatibility.Speculative("none", "", 1));
    }
}
