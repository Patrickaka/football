# -*- coding: utf-8 -*-
"""篮球接口在新入口上的行为。

业务逻辑住在 `src.api.services.basketball`，**新旧两个入口共用同一份**
（判据 11）——旧的 `src/webapp/basketball_api.py` 已改成 25 行转发。
所以这里测的不是业务，是**参数怎么走到服务层**、以及错误长什么样。

迁移当时跑过 21 条新旧双跑差分（同一个 URL 分别喂给旧 mixin 与新路由），
只有两条不一致，都是非法 `threshold`：旧入口把参数错误伪装成 `200` 里的
一句 error 字符串，新入口返回 422。**这一处没有照搬旧行为**——422 才是
正确语义。但网页整套是按 `data.error` 判错的，所以 422 的响应体里
两个字段都给：机器读 `detail`，网页读 `error`。
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.services import basketball as service


def make_client():
    return TestClient(create_app(auth_settings=AuthSettings(credentials={})))


class ParameterPassing(unittest.TestCase):
    """路由把具名参数包回 `parse_qs` 的 `{键: [值]}` 形状交给服务层。

    形状保持不变是为了让新旧双跑差分能喂同一批输入；切换完成、旧入口
    删除后再换成具名参数。
    """

    def _captured(self, url, function):
        with mock.patch.object(service, function, return_value={}) as spy:
            with make_client() as client:
                client.get(url)
        return spy.call_args[0][0] if spy.call_args else None

    def test_defaults_match_the_old_entry_point(self):
        """默认值必须和旧入口一样，否则不带参数的请求会静静地换个行为。"""
        self.assertEqual(
            self._captured('/api/basketball', 'basketball_payload'),
            {'date': [None], 'types': ['spf,rqspf,dx'], 'source': ['okooo']})

    def test_the_value_threshold_defaults_to_five_percent(self):
        self.assertEqual(
            self._captured('/api/basketball/value', 'basketball_value_payload'),
            {'date': [None], 'threshold': [0.05]})

    def test_query_parameters_reach_the_service(self):
        self.assertEqual(
            self._captured('/api/basketball?date=2026-08-29&types=spf&source=500',
                           'basketball_payload'),
            {'date': ['2026-08-29'], 'types': ['spf'], 'source': ['500']})

    def test_the_match_id_reaches_the_movement_service(self):
        self.assertEqual(
            self._captured('/api/basketball/movement?match_id=abc',
                           'basketball_movement_payload'),
            {'match_id': ['abc']})


class ValidationErrors(unittest.TestCase):

    def test_a_bad_number_is_a_422_not_a_200(self):
        """**没有照搬旧行为**：旧入口把参数错误伪装成 200 里的 error 串。"""
        with make_client() as client:
            self.assertEqual(
                client.get('/api/basketball/value?threshold=abc').status_code, 422)

    def test_the_response_carries_both_shapes(self):
        """网页按 `data.error` 判错，只给 `detail` 的话它会以为请求成功、
        然后拿不到数据，页面空白且没有任何提示。
        """
        with make_client() as client:
            payload = client.get('/api/basketball/value?threshold=abc').json()
            self.assertIn('threshold', payload['error'])
            self.assertTrue(payload['detail'])

    def test_the_error_names_the_offending_field(self):
        with make_client() as client:
            payload = client.get('/api/basketball/value?threshold=').json()
            self.assertIn('threshold', payload['error'])

    def test_a_valid_number_is_not_rejected(self):
        """**反方向**：合法值不能被校验挡住，否则这个接口就废了。"""
        with mock.patch.object(service, 'basketball_value_payload', return_value={'result': []}):
            with make_client() as client:
                self.assertEqual(
                    client.get('/api/basketball/value?threshold=0.3').status_code, 200)


class Routing(unittest.TestCase):

    EXPECTED = {'/api/basketball', '/api/basketball/matches', '/api/basketball/value',
                '/api/basketball/track', '/api/basketball/movement'}

    def test_all_five_old_routes_exist(self):
        """旧入口的五条篮球路由**一条都不能少**——少一条就是切换当天才发现。

        清单取自 OpenAPI schema，不是 `app.routes`：`include_router` 进来的
        路由包在 `_IncludedRouter` 里，遍历 `app.routes` 一条也看不到，
        断言会**永远通过**（判据：看着在测、其实什么也没测）。
        """
        app = create_app(auth_settings=AuthSettings(credentials={}))
        paths = set(app.openapi()['paths'])
        self.assertEqual(self.EXPECTED - paths, set())

    def test_the_route_listing_itself_works(self):
        """守卫的守卫：清单必须真的列得出东西，否则上一条是空转。"""
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertIn('/healthz', app.openapi()['paths'])


    def test_basketball_routes_need_a_session_when_auth_is_on(self):
        """业务路由**不在豁免清单里**——加路由时别顺手把它放进去。"""
        client = TestClient(create_app(auth_settings=AuthSettings(credentials={'a': 'b'})))
        with client:
            response = client.get('/api/basketball/matches',
                                  headers={'accept': 'application/json'})
            self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()
