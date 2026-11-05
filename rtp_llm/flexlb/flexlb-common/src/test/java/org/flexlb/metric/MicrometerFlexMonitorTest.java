package org.flexlb.metric;

import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class MicrometerFlexMonitorTest {

    @AfterEach
    void resetAllowedMetrics() {
        MicrometerFlexMonitor.setAllowedMetrics(null);
    }

    @Test
    void removalReleasesOnlyTheRequestedSeriesAndAllowsFreshRegistration() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            var tags = FlexMetricTags.of("compatibility_group", "retired");
            var other = FlexMetricTags.of("compatibility_group", "current");
            for (var type : new FlexMetricType[]{FlexMetricType.GAUGE, FlexMetricType.COUNTER, FlexMetricType.TIMER}) {
                monitor.register(type.name(), type);
                monitor.report(type.name(), tags, 3.0);
                monitor.report(type.name(), other, 5.0);
                monitor.remove(type.name(), tags);
                monitor.remove(type.name(), tags);
                assertEquals(1, registry.find("flexlb." + type.name()).meters().size());
                monitor.report(type.name(), tags, 7.0);
                assertEquals(2, registry.find("flexlb." + type.name()).meters().size());
            }
            assertEquals(7.0, registry.get("flexlb.GAUGE").tag("compatibility_group", "retired").gauge().value());
            assertEquals(7.0, registry.get("flexlb.COUNTER").tag("compatibility_group", "retired").counter().count());
            assertEquals(1L, registry.get("flexlb.TIMER").tag("compatibility_group", "retired").timer().count());
            assertEquals(5.0, registry.get("flexlb.GAUGE").tag("compatibility_group", "current").gauge().value());
        } finally { registry.close(); }
    }

    @Test
    void prepareTimerRegistersWithoutRecordingAndReportReusesIt() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            FlexMetricTags tags = FlexMetricTags.of("role", "PREFILL", "engineIp", "127.0.0.1");
            monitor.register("test.timer", FlexMetricType.TIMER);

            monitor.prepare("test.timer", tags);

            Timer prepared = registry.find("flexlb.test.timer")
                    .tags("role", "PREFILL", "engineIp", "127.0.0.1")
                    .timer();
            assertNotNull(prepared);
            assertEquals(0L, prepared.count());

            monitor.report("test.timer", tags, 12.0D);

            Timer reported = registry.find("flexlb.test.timer")
                    .tags("role", "PREFILL", "engineIp", "127.0.0.1")
                    .timer();
            assertEquals(prepared, reported);
            assertEquals(1L, reported.count());
            assertEquals(12.0D, reported.totalTime(TimeUnit.MILLISECONDS), 0.001D);
        } finally {
            monitor.close();
            registry.close();
        }
    }
}
