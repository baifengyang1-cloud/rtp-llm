import socket
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.multimodal import vit_proxy_start_server as startup
from rtp_llm.server.vit_proxy_server import VitProxyServer


class VitProxyStartupTest(TestCase):
    def run_startup(self, failing_stage=None):
        events = []
        proxy = MagicMock()
        collector = MagicMock()
        collector.collect.return_value = 7
        collector.get_freeze_count.return_value = 12345
        collector.isenabled.return_value = True
        collector.get_threshold.return_value = (700, 10, 10)
        config = MagicMock()
        config.vit_config.mm_timeout_ms = 120000

        def stage(name, result=None):
            def invoke(*args, **kwargs):
                events.append(name)
                if name == failing_stage:
                    raise RuntimeError(name)
                return result

            return invoke

        proxy.prepare.side_effect = stage("prepare")
        proxy.start.side_effect = stage("grpc_start")
        proxy.stop.side_effect = stage("stop")
        collector.collect.side_effect = stage("collect", 7)
        collector.freeze.side_effect = stage("freeze")
        app = object()
        with (
            patch.object(startup, "VitProxyServer", return_value=proxy),
            patch.object(startup, "create_proxy_app", side_effect=stage("app", app)),
            patch.object(startup, "start_http_server", side_effect=stage("http_start")),
            patch.object(startup, "setproctitle"),
            patch.object(startup, "gc", collector),
            patch.object(startup.logging, "info") as log,
        ):
            if failing_stage:
                with self.assertRaisesRegex(RuntimeError, failing_stage):
                    startup.vit_proxy_start_server(
                        config, ["worker:1"], 1234, 1235, ["worker:2"]
                    )
            else:
                startup.vit_proxy_start_server(
                    config, ["worker:1"], 1234, 1235, ["worker:2"]
                )
        return events, proxy, collector, log

    def test_freeze_finishes_before_either_interface_accepts_requests(self):
        events, proxy, collector, log = self.run_startup()
        self.assertEqual(
            events,
            ["prepare", "app", "collect", "freeze", "grpc_start", "http_start", "stop"],
        )
        collector.collect.assert_called_once_with()
        collector.freeze.assert_called_once_with()
        collector.disable.assert_not_called()
        collector.set_threshold.assert_not_called()
        records = [
            c.args for c in log.call_args_list if "vit_proxy_gc_freeze" in c.args[0]
        ]
        self.assertEqual(len(records), 1)
        fmt, *args = records[0]
        message = fmt % tuple(args)
        for expected in (
            "collected=7",
            "frozen=12345",
            "gc_enabled=True",
            "gc_threshold=(700, 10, 10)",
        ):
            self.assertIn(expected, message)
        proxy.stop.assert_called_once_with()

    def test_initialization_failure_closes_resources_without_serving(self):
        for failure in ("prepare", "app", "collect", "freeze"):
            with self.subTest(failure=failure):
                events, proxy, _, _ = self.run_startup(failure)
                self.assertEqual(events[-1], "stop")
                proxy.start.assert_not_called()
                proxy.stop.assert_called_once_with()
                self.assertNotIn("http_start", events)

    def test_serving_failure_closes_resources(self):
        for failure in ("grpc_start", "http_start"):
            with self.subTest(failure=failure):
                events, proxy, _, _ = self.run_startup(failure)
                self.assertEqual(events[-1], "stop")
                proxy.stop.assert_called_once_with()
                if failure == "grpc_start":
                    self.assertNotIn("http_start", events)


class VitProxyPreparationTest(TestCase):
    @patch("rtp_llm.server.vit_proxy_server.kmonitor.init")
    def test_prepare_is_reusable_and_start_still_prepares_implicitly(self, _init):
        for explicitly_prepare in (False, True):
            with self.subTest(explicitly_prepare=explicitly_prepare):
                with patch("rtp_llm.server.vit_proxy_server.grpc.server") as factory:
                    proxy = VitProxyServer(["worker:1"], 1234)
                    if explicitly_prepare:
                        proxy.prepare()
                        proxy.prepare()
                        factory.return_value.start.assert_not_called()
                    proxy.start()
                    factory.assert_called_once()
                    factory.return_value.start.assert_called_once_with()
                    proxy.stop()

    @patch("rtp_llm.server.vit_proxy_server.kmonitor.init")
    def test_real_grpc_server_accepts_connections_only_after_start(self, _init):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        proxy = VitProxyServer(["worker:1"], port)
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        ready = grpc.channel_ready_future(channel)
        try:
            proxy.prepare()
            with self.assertRaises(grpc.FutureTimeoutError):
                ready.result(timeout=0.1)
            proxy.start()
            ready.result(timeout=5)
        finally:
            ready.cancel()
            channel.close()
            proxy.stop()

    @patch("rtp_llm.server.vit_proxy_server.kmonitor.init")
    def test_real_grpc_server_can_stop_before_start(self, _init):
        proxy = VitProxyServer(["worker:1"], 0)
        proxy.prepare()
        proxy.stop()


if __name__ == "__main__":
    main()
