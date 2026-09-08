import unittest
import socket
from unittest import mock


class EntrypointTests(unittest.TestCase):
    def test_reads_host_and_port_from_env(self):
        from main import server_config

        with mock.patch.dict(
            'os.environ', {'FOOTBALL_HOST': '127.0.0.1', 'FOOTBALL_PORT': '9100'}, clear=True
        ):
            config = server_config()
        self.assertEqual(config['host'], '127.0.0.1')
        self.assertEqual(config['port'], 9100)

    def test_defaults_match_previous_server(self):
        from main import server_config

        with mock.patch.dict('os.environ', {}, clear=True):
            config = server_config()
        self.assertEqual(config['host'], '0.0.0.0')
        self.assertEqual(config['port'], 9004)

    def test_single_worker_due_to_memory_limit(self):
        from main import server_config

        with mock.patch.dict('os.environ', {}, clear=True):
            self.assertEqual(server_config()['workers'], 1)

    def test_no_socket_argument_preserves_uvicorn_run_contract(self):
        import main

        with mock.patch.dict('os.environ', {}, clear=True):
            with mock.patch.object(main.sys, 'stdout'), mock.patch.object(main.uvicorn, 'run') as run:
                main.main()
        run.assert_called_once_with(
            'main:app', host='0.0.0.0', port=9004, workers=1, log_level='info'
        )

    def test_reserved_sockets_are_passed_unchanged_to_uvicorn_server(self):
        import main

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        sockets = [listener]
        env = {'FOOTBALL_HOST': '127.0.0.1', 'FOOTBALL_PORT': str(listener.getsockname()[1])}
        with mock.patch.dict('os.environ', env, clear=True):
            with mock.patch.object(main.sys, 'stdout'), mock.patch.object(main.uvicorn, 'run') as run:
                with mock.patch.object(main.uvicorn, 'Config') as config, mock.patch.object(main.uvicorn, 'Server') as server:
                    main.main(sockets=sockets)
        run.assert_not_called()
        config.assert_called_once()
        config_options = config.call_args.kwargs
        self.assertEqual(config_options['host'], env['FOOTBALL_HOST'])
        self.assertEqual(config_options['port'], int(env['FOOTBALL_PORT']))
        self.assertEqual(config_options['workers'], 1)
        server.assert_called_once_with(config.return_value)
        server.return_value.run.assert_called_once_with(sockets=sockets)
        self.assertIs(server.return_value.run.call_args.kwargs['sockets'], sockets)
        self.assertGreaterEqual(listener.fileno(), 0)

    def test_uvicorn_startup_failure_returns_a_failure_exit_status(self):
        import main

        with mock.patch.object(main.sys, 'stdout'), mock.patch.object(main.uvicorn, 'Config'):
            with mock.patch.object(main.uvicorn, 'Server') as server:
                server.return_value.started = False
                with self.assertRaises(SystemExit) as caught:
                    main.main(sockets=[])
        self.assertNotIn(caught.exception.code, (None, 0))


if __name__ == '__main__':
    unittest.main()
