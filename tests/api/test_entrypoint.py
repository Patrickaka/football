import unittest
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


if __name__ == '__main__':
    unittest.main()
