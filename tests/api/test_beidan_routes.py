# -*- coding: utf-8 -*-
"""北单接口在新入口上的行为。

业务逻辑住在 `src.api.services.beidan`，**新旧两个入口共用同一份**
（判据 11）——旧的 `src/webapp/beidan_api.py` 从 142 行缩到 23 行转发。

迁移当时跑过 17 条新旧双跑差分，只有 3 条不一致，全是非法参数
（`limit=abc` / `threshold=abc`）：旧入口把参数错误伪装成 `200` 里的
error 串，新入口返回 422 并同时给 `error` 与 `detail`。这一处是有意的。

**首轮差分曾多出 2 条 `total_matches` 305 vs 306**——那不是代码差异，
是首次调用抓了一次新数据。同一个函数连调四次都稳定在 306、缓存热了以后
重跑差分那两条就消失了。**别急着把数据漂当成回归**。
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.services import beidan as service


def make_client():
    return TestClient(create_app(auth_settings=AuthSettings(credentials={})))


class ParameterPassing(unittest.TestCase):

    def _captured(self, url, function):
        with mock.patch.object(service, function, return_value={}) as spy:
            with make_client() as client:
                client.get(url)
        return spy.call_args[0][0] if spy.call_args else None

    def test_defaults_match_the_old_entry_point(self):
        self.assertEqual(
            self._captured('/api/beidan', 'beidan_payload'),
            {'date': [None], 'types': ['spf,rqspf,zjq'], 'source': ['okooo'],
             'force_refresh': ['false']})

    def test_the_history_limit_defaults_to_two_hundred(self):
        self.assertEqual(self._captured('/api/beidan/history', 'beidan_history_payload'),
                         {'limit': [200]})

    def test_the_value_defaults_match(self):
        self.assertEqual(self._captured('/api/beidan/value', 'beidan_value_payload'),
                         {'date': [None], 'source': ['okooo'], 'threshold': [0.05]})


class ForceRefreshStaysAString(unittest.TestCase):
    """服务层判的是 `.lower() == 'true'`，只有字面量 `true` 算真。

    改用 FastAPI 的 `bool` 会让 `1` / `yes` / `on` 也变成真——那是
    **悄悄放宽**，不是等价迁移：一个手滑的 `?force_refresh=1` 会绕过缓存
    去真抓一遍。
    """

    def _forced(self, query):
        with mock.patch.object(service, 'beidan_payload', return_value={}) as spy:
            with make_client() as client:
                client.get(f'/api/beidan?{query}')
        return spy.call_args[0][0]['force_refresh'][0]

    def test_only_the_literal_true_is_forwarded_as_true(self):
        self.assertEqual(self._forced('force_refresh=true'), 'true')

    def test_other_truthy_looking_values_pass_through_unchanged(self):
        """**它们必须原样传下去**，由服务层按旧规则判——不能在路由层就转成 bool。"""
        for value in ('1', 'yes', 'on', 'TRUE', 'True'):
            with self.subTest(value=value):
                self.assertEqual(self._forced(f'force_refresh={value}'), value)

    def test_the_service_only_honours_a_lowercase_true(self):
        """**反方向**：服务层这条规则本身也要有测试，否则上面测的是空气。"""
        import inspect
        source = inspect.getsource(service.beidan_payload)
        self.assertIn(".lower() == 'true'", source)


class ValidationErrors(unittest.TestCase):

    def test_a_bad_limit_is_a_422(self):
        with make_client() as client:
            response = client.get('/api/beidan/history?limit=abc')
            self.assertEqual(response.status_code, 422)
            self.assertIn('limit', response.json()['error'])

    def test_a_bad_threshold_is_a_422(self):
        with make_client() as client:
            response = client.get('/api/beidan/value?threshold=abc')
            self.assertEqual(response.status_code, 422)
            self.assertIn('threshold', response.json()['error'])

    def test_valid_values_are_not_rejected(self):
        with mock.patch.object(service, 'beidan_history_payload', return_value={'result': []}):
            with make_client() as client:
                self.assertEqual(client.get('/api/beidan/history?limit=5').status_code, 200)


class Routing(unittest.TestCase):

    EXPECTED = {'/api/beidan', '/api/beidan/matches',
                '/api/beidan/value', '/api/beidan/history'}

    def test_all_four_old_routes_exist(self):
        """清单取自 OpenAPI schema——`app.routes` 看不到 `include_router`
        进来的路由，那样的断言会永远通过。
        """
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertEqual(self.EXPECTED - set(app.openapi()['paths']), set())

    def test_the_old_mixin_still_shares_the_implementation(self):
        from src.webapp.beidan_api import BeidanApiMixin
        with mock.patch.object(service, 'beidan_history_payload',
                               return_value={'sentinel': True}):
            self.assertEqual(
                BeidanApiMixin()._beidan_history_payload({}), {'sentinel': True})

    def test_beidan_routes_are_not_public(self):
        client = TestClient(create_app(auth_settings=AuthSettings(credentials={'a': 'b'})))
        with client:
            self.assertEqual(
                client.get('/api/beidan/history',
                           headers={'accept': 'application/json'}).status_code, 401)


if __name__ == '__main__':
    unittest.main()
