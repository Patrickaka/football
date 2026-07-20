import unittest
from unittest.mock import patch

import server


class ServerSsqRouteTests(unittest.TestCase):
    def setUp(self):
        server._CACHE['ssq']['data'] = None
        server._CACHE['ssq']['timestamp'] = 0
        self.handler = server.Handler.__new__(server.Handler)
        self.handler._log = server.log

    def test_subpath_route_normalizes_to_ssq_api(self):
        self.assertEqual(
            server.Handler._normalize_path('/football/api/ssq'),
            '/api/ssq',
        )

    @patch('server._serve_cached')
    def test_ssq_payload_returns_cached_prediction(self, serve_cached):
        serve_cached.return_value = ({'latest_period': '26080'}, None)
        payload = self.handler._ssq_payload()
        self.assertEqual(payload['result']['latest_period'], '26080')
        serve_cached.assert_called_once_with('ssq', server.ssq_run_prediction)

    @patch('server.ssq_clear_cache')
    @patch('server.ssq_run_prediction')
    def test_ssq_refresh_repopulates_server_cache(self, predict, clear):
        predict.return_value = {'latest_period': '26081'}
        payload = self.handler._ssq_refresh_payload()
        clear.assert_called_once_with()
        predict.assert_called_once_with(force_refresh=True)
        self.assertEqual(payload['result']['latest_period'], '26081')
        self.assertEqual(server._CACHE['ssq']['data']['latest_period'], '26081')


if __name__ == '__main__':
    unittest.main()
