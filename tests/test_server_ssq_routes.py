from src.api.services import lottery as service
import logging
import unittest
from unittest.mock import patch

from src.api.runtime import caching, lazy_modules


class ServerSsqRouteTests(unittest.TestCase):
    def setUp(self):
        caching._CACHE['ssq']['data'] = None
        caching._CACHE['ssq']['timestamp'] = 0

    # 服务层走的是 `_caching_mod._serve_cached`（模块属性访问），
    # 所以 patch 要打在 `src.api.runtime.caching` 上。**打错地方不会报错**，
    # 只是什么也没替换掉，测试看着通过其实测的是真实调用。
    @patch('src.api.runtime.caching._serve_cached')
    def test_ssq_payload_returns_cached_prediction(self, serve_cached):
        serve_cached.return_value = ({'latest_period': '26080'}, None)
        payload = service.ssq_payload()
        self.assertEqual(payload['result']['latest_period'], '26080')
        serve_cached.assert_called_once_with('ssq', lazy_modules.ssq_run_prediction)

    @patch('src.api.runtime.lazy_modules.ssq_clear_cache')
    @patch('src.api.runtime.lazy_modules.ssq_run_prediction')
    def test_ssq_refresh_repopulates_server_cache(self, predict, clear):
        predict.return_value = {'latest_period': '26081'}
        payload = service.ssq_refresh_payload()
        clear.assert_called_once_with()
        predict.assert_called_once_with(force_refresh=True)
        self.assertEqual(payload['result']['latest_period'], '26081')
        self.assertEqual(caching._CACHE['ssq']['data']['latest_period'], '26081')


if __name__ == '__main__':
    unittest.main()
