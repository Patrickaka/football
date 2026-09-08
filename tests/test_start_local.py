import argparse
from contextlib import contextmanager, ExitStack
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import socket
import sys
import threading
import types
import unittest
from unittest import mock

import start_local


def _football_schema():
    return {
        "info": {"title": "Football 预测服务"},
        "paths": {"/api/matches": {}, "/api/predictions": {}, "/healthz": {}},
    }


@contextmanager
def _http_service(document=None, *, status=200, raw=None, protocol="HTTP/1.0"):
    requests = []
    body = raw if raw is not None else json.dumps(document).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = protocol

        def do_GET(self):
            requests.append(self.path)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def _bound_socket(host="127.0.0.1", *, family=socket.AF_INET, port=0):
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((host, port))
        listener.listen()
        yield listener
    finally:
        listener.close()


class LauncherArgumentsTests(unittest.TestCase):
    def test_status_and_ipv6_host_are_accepted(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = start_local._arguments(["--status", "--host", "::1", "--port", "9100"])
        self.assertTrue(args.status)
        self.assertEqual((args.host, args.port), ("::1", 9100))

    def test_default_and_boundary_ports(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(start_local._arguments([]).port, 9004)
            for value in ("1", "65535"):
                with self.subTest(port=value):
                    self.assertEqual(start_local._arguments(["--port", value]).port, int(value))

    def test_invalid_command_line_ports_are_argparse_errors(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            for value in ("0", "-1", "65536", "not-a-port"):
                with self.subTest(port=value), mock.patch.object(sys, "stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        start_local._arguments(["--port", value])
                    self.assertEqual(caught.exception.code, 2)

    def test_invalid_environment_ports_are_argparse_errors(self):
        for value in ("0", "-1", "65536", "not-a-port", ""):
            with self.subTest(port=value), mock.patch.dict(os.environ, {"FOOTBALL_PORT": value}, clear=True):
                with mock.patch.object(sys, "stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        start_local._arguments([])
                self.assertEqual(caught.exception.code, 2)

    def test_display_urls_use_connectable_wildcards_and_ipv6_brackets(self):
        expected = {
            "0.0.0.0": "http://127.0.0.1:9004",
            "::": "http://[::1]:9004",
            "::1": "http://[::1]:9004",
            "2001:db8::1": "http://[2001:db8::1]:9004",
            "localhost": "http://localhost:9004",
        }
        for host, url in expected.items():
            with self.subTest(host=host):
                self.assertEqual(start_local._display_url(host, 9004), url)


class ListenerReservationTests(unittest.TestCase):
    def test_real_occupied_port_is_rejected(self):
        with _bound_socket() as occupied:
            with self.assertRaises(OSError):
                start_local._reserve_listeners("127.0.0.1", occupied.getsockname()[1])

    def test_reservation_listens_and_retains_port_until_closed(self):
        listeners = start_local._reserve_listeners("127.0.0.1", 0)
        self.addCleanup(lambda: [listener.close() for listener in listeners])
        self.assertEqual(len(listeners), 1)
        listener = listeners[0]
        port = listener.getsockname()[1]
        self.assertEqual(listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)
        with self.assertRaises(OSError):
            start_local._reserve_listeners("127.0.0.1", port)
        listener.close()
        with _bound_socket(port=port):
            pass

    def test_ipv6_loopback_is_reserved_when_available(self):
        if not socket.has_ipv6:
            self.skipTest("IPv6 is unavailable")
        try:
            with _bound_socket("::1", family=socket.AF_INET6):
                pass
        except OSError as exc:
            self.skipTest(f"IPv6 loopback is unavailable: {exc}")
        listeners = start_local._reserve_listeners("::1", 0)
        self.addCleanup(lambda: [listener.close() for listener in listeners])
        self.assertTrue(listeners)
        for listener in listeners:
            self.assertEqual(listener.family, socket.AF_INET6)
            self.assertEqual(listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)
            with self.assertRaises(OSError):
                start_local._reserve_listeners("::1", listener.getsockname()[1])

    def test_partial_hostname_reservation_failure_releases_prior_listener(self):
        try:
            occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.addCleanup(occupied.close)
            if os.name == "nt":
                occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            occupied.bind(("127.0.0.2", 0))
            occupied.listen()
        except OSError as exc:
            self.skipTest(f"Second loopback address is unavailable: {exc}")
        port = occupied.getsockname()[1]
        addresses = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.2", port)),
        ]
        with mock.patch.object(socket, "getaddrinfo", return_value=addresses):
            with self.assertRaises(OSError):
                start_local._reserve_listeners("multiple-addresses.test", port)
        # The first address was free, but must be released when the second fails.
        with _bound_socket(port=port):
            pass

    def test_unavailable_ipv6_falls_back_to_a_real_ipv4_listener(self):
        addresses = [
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", 0, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 0)),
        ]
        real_socket = socket.socket
        for error_code in (errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT, errno.EADDRNOTAVAIL):
            with self.subTest(error=error_code):
                unavailable = mock.Mock()
                unavailable.bind.side_effect = OSError(error_code, "IPv6 is unavailable")

                def make_socket(family, *args):
                    if family == socket.AF_INET6:
                        if error_code != errno.EADDRNOTAVAIL:
                            raise OSError(error_code, "IPv6 is unavailable")
                        return unavailable
                    return real_socket(family, *args)

                with mock.patch.object(socket, "getaddrinfo", return_value=addresses):
                    with mock.patch.object(socket, "socket", side_effect=make_socket):
                        listeners = start_local._reserve_listeners("localhost", 0)
                self.addCleanup(lambda sockets=listeners: [listener.close() for listener in sockets])
                self.assertEqual(len(listeners), 1)
                self.assertEqual(listeners[0].family, socket.AF_INET)
                self.assertEqual(listeners[0].getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)
                if error_code == errno.EADDRNOTAVAIL:
                    unavailable.close.assert_called_once()

    def test_all_resolved_addresses_unavailable_fails_and_closes_created_sockets(self):
        addresses = [
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", 0, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 0)),
        ]
        unavailable = mock.Mock()
        unavailable.bind.side_effect = OSError(errno.EADDRNOTAVAIL, "Address is unavailable")
        with mock.patch.object(socket, "getaddrinfo", return_value=addresses):
            with mock.patch.object(socket, "socket", side_effect=[OSError(errno.EAFNOSUPPORT, "No IPv6"), unavailable]):
                with self.assertRaises(OSError):
                    start_local._reserve_listeners("localhost", 0)
        unavailable.close.assert_called_once()

    def test_occupied_or_forbidden_address_still_aborts_and_releases_prior_listener(self):
        addresses = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", 0, 0, 0)),
        ]
        real_socket = socket.socket
        for error_code in (errno.EADDRINUSE, errno.EACCES, errno.EPERM):
            with self.subTest(error=error_code):
                created = []
                rejected = mock.Mock()
                rejected.bind.side_effect = OSError(error_code, "Binding is forbidden")

                def make_socket(family, *args):
                    if family == socket.AF_INET6:
                        return rejected
                    listener = real_socket(family, *args)
                    created.append(listener)
                    self.addCleanup(listener.close)
                    return listener

                with mock.patch.object(socket, "getaddrinfo", return_value=addresses):
                    with mock.patch.object(socket, "socket", side_effect=make_socket):
                        with self.assertRaises(OSError) as caught:
                            start_local._reserve_listeners("localhost", 0)
                self.assertEqual(caught.exception.errno, error_code)
                self.assertTrue(created)
                self.assertTrue(all(listener.fileno() == -1 for listener in created))
                rejected.close.assert_called_once()


class ExistingServiceTests(unittest.TestCase):
    def test_recognizes_project_openapi_even_with_a_broken_proxy_environment(self):
        proxy_env = {
            "http_proxy": "http://127.0.0.1:1",
            "https_proxy": "http://127.0.0.1:1",
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "",
            "no_proxy": "",
        }
        with _http_service(_football_schema()) as (port, requests):
            with mock.patch.dict(os.environ, proxy_env):
                self.assertTrue(start_local._existing_football_service("127.0.0.1", port))
            self.assertEqual(requests, ["/openapi.json"])

    def test_rejects_other_http_200_services_and_incomplete_schemas(self):
        cases = {
            "generic response": {"status": "ok"},
            "wrong title": {"info": {"title": "Unrelated app"}, "paths": _football_schema()["paths"]},
            "missing route": {"info": _football_schema()["info"], "paths": {"/api/matches": {}}},
            "non-object response": ["Football 预测服务"],
            "invalid info type": {"info": None, "paths": _football_schema()["paths"]},
            "invalid paths type": {"info": _football_schema()["info"], "paths": list(_football_schema()["paths"])},
        }
        for name, document in cases.items():
            with self.subTest(case=name), _http_service(document) as (port, _):
                self.assertFalse(start_local._existing_football_service("127.0.0.1", port))

    def test_rejects_malformed_json_and_http_errors(self):
        with _http_service(raw=b"<html>hello</html>") as (port, _):
            self.assertFalse(start_local._existing_football_service("127.0.0.1", port))
        with _http_service(_football_schema(), status=503) as (port, _):
            self.assertFalse(start_local._existing_football_service("127.0.0.1", port))

    def test_non_http_response_is_not_a_project_service(self):
        with _http_service(_football_schema(), protocol="SSH-2.0") as (port, requests):
            self.assertFalse(start_local._existing_football_service("127.0.0.1", port))
            self.assertEqual(requests, ["/openapi.json"])


class LauncherMainTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(mock.patch.object(start_local, "_configure_console"))
        self.output = self.stack.enter_context(mock.patch.object(sys, "stdout", io.StringIO()))
        self.stack.enter_context(mock.patch.object(start_local.os, "chdir"))
        self.venv = self.stack.enter_context(mock.patch.object(start_local, "_create_or_enter_venv"))
        self.missing = self.stack.enter_context(mock.patch.object(start_local, "_missing_modules", return_value=[]))
        self.install = self.stack.enter_context(mock.patch.object(start_local, "_install_dependencies"))
        self.args = argparse.Namespace(host="127.0.0.1", port=0, no_install=False, startup_tasks=False, status=False)
        self.stack.enter_context(mock.patch.object(start_local, "_arguments", return_value=self.args))
        self.service = types.ModuleType("main")
        self.service.main = mock.Mock()
        self.stack.enter_context(mock.patch.dict(sys.modules, {"main": self.service}))
        self.listeners = None
        reserve = start_local._reserve_listeners

        def capture_listeners(host, port):
            self.listeners = reserve(host, port)
            return self.listeners

        self.reserve = self.stack.enter_context(mock.patch.object(start_local, "_reserve_listeners", side_effect=capture_listeners))

    def _assert_no_startup_work(self):
        self.missing.assert_not_called()
        self.install.assert_not_called()
        self.service.main.assert_not_called()

    def test_other_service_occupancy_stops_before_dependency_checks_or_service_import(self):
        original_import = __import__

        def reject_service_import(name, *args, **kwargs):
            if name == "main":
                raise AssertionError("The occupied-port path must not import the application")
            return original_import(name, *args, **kwargs)

        with _http_service({"status": "ok"}) as (port, _):
            self.args.port = port
            with mock.patch("builtins.__import__", side_effect=reject_service_import):
                with self.assertRaises(SystemExit) as caught:
                    start_local.main()
            self.assertNotIn(caught.exception.code, (None, 0))
        self._assert_no_startup_work()

    def test_existing_football_service_is_a_successful_noop(self):
        with _http_service(_football_schema()) as (port, _):
            self.args.port = port
            self.assertIn(start_local.main(), (None, 0))
        self._assert_no_startup_work()
        self.assertIn(str(port), self.output.getvalue())

    def test_status_checks_running_service_without_environment_or_startup_work(self):
        self.args.status = True
        with _http_service(_football_schema()) as (port, _):
            self.args.port = port
            self.assertIn(start_local.main(), (None, 0))
        self.venv.assert_not_called()
        self.reserve.assert_not_called()
        self._assert_no_startup_work()

    def test_status_rejects_an_unrelated_service_without_startup_work(self):
        self.args.status = True
        with _http_service({"status": "ok"}) as (port, _):
            self.args.port = port
            with self.assertRaises(SystemExit) as caught:
                start_local.main()
        self.assertNotIn(caught.exception.code, (None, 0))
        self.venv.assert_not_called()
        self.reserve.assert_not_called()
        self._assert_no_startup_work()

    def test_only_reserved_sockets_are_handed_to_the_service_and_closed_on_return(self):
        def run_service(*, sockets):
            self.assertIs(sockets, self.listeners)
            self.assertTrue(sockets)
            self.assertEqual(sockets[0].getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)
            with self.assertRaises(OSError):
                with _bound_socket(port=sockets[0].getsockname()[1]):
                    pass

        self.service.main.side_effect = run_service
        start_local.main()
        self.service.main.assert_called_once()
        self.assertTrue(all(listener.fileno() == -1 for listener in self.listeners))

    def test_port_is_reserved_after_venv_but_before_dependency_work(self):
        def check_dependencies():
            self.venv.assert_called_once()
            self.assertTrue(self.listeners)
            self.assertTrue(all(listener.fileno() >= 0 for listener in self.listeners))
            return []

        self.missing.side_effect = check_dependencies
        start_local.main()

    def test_dependency_failure_releases_reserved_sockets(self):
        self.missing.return_value = ["fastapi"]
        self.install.side_effect = SystemExit("installation failed")
        with self.assertRaises(SystemExit):
            start_local.main()
        self.assertTrue(self.listeners)
        self.assertTrue(all(listener.fileno() == -1 for listener in self.listeners))
        self.service.main.assert_not_called()

    def test_no_install_dependency_failure_releases_reserved_sockets(self):
        self.args.no_install = True
        self.missing.return_value = ["fastapi"]
        with self.assertRaises(SystemExit):
            start_local.main()
        self.assertTrue(self.listeners)
        self.assertTrue(all(listener.fileno() == -1 for listener in self.listeners))
        self.install.assert_not_called()
        self.service.main.assert_not_called()

    def test_application_failure_releases_reserved_sockets(self):
        self.service.main.side_effect = RuntimeError("startup failed")
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            start_local.main()
        self.assertTrue(self.listeners)
        self.assertTrue(all(listener.fileno() == -1 for listener in self.listeners))

    def test_service_import_failure_releases_reserved_sockets(self):
        original_import = __import__

        def fail_service_import(name, *args, **kwargs):
            if name == "main":
                raise ModuleNotFoundError("No module named fastapi", name="fastapi")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fail_service_import):
            with self.assertRaises(SystemExit):
                start_local.main()
        self.assertTrue(self.listeners)
        self.assertTrue(all(listener.fileno() == -1 for listener in self.listeners))
        self.service.main.assert_not_called()

    def test_keyboard_interrupt_releases_reserved_sockets(self):
        self.service.main.side_effect = KeyboardInterrupt()
        start_local.main()
        self.assertTrue(self.listeners)
        self.assertTrue(all(listener.fileno() == -1 for listener in self.listeners))


if __name__ == "__main__":
    unittest.main()
