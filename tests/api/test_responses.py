# -*- coding: utf-8 -*-
"""新入口的 JSON 序列化。

**和旧入口用同一套清洗**（判据 11）：`_sanitize_json` + `_json_default`
住在 `src/webapp/http_util.py`。

这一条是切换入口当天线上炸出来的：`/api/predict/batch` 全部 500，
`TypeError: vars() argument must have __dict__ attribute`。前端拿到 500
退回逐场重试、逐场同样失败，界面就停在「正在分析比赛数据（24/58）」不动。

**为什么迁移时没发现**：路由测试全都 mock 掉了服务层、返回
`{'sentinel': ...}` 这种纯字面量；双跑差分比的是**服务函数的返回值**，
在进 FastAPI 的序列化之前就比完了。两层都绕开了真正出问题的那一步。
"""
import json
import math
import unittest

import numpy as np
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.responses import SanitizedJSONResponse


class Serialisation(unittest.TestCase):

    def _rendered(self, content):
        return json.loads(SanitizedJSONResponse(content=content).body.decode('utf-8'))

    def test_numpy_scalars_become_plain_numbers(self):
        """模型算出来的概率是 numpy 标量——**FastAPI 默认那套遇到就 500**。"""
        self.assertEqual(self._rendered({'p': np.float64(0.25)}), {'p': 0.25})
        self.assertEqual(self._rendered({'n': np.int64(3)}), {'n': 3})

    def test_numpy_arrays_become_lists(self):
        self.assertEqual(self._rendered({'a': np.array([1, 2, 3])}), {'a': [1, 2, 3]})

    def test_objects_with_to_dict_are_unwrapped(self):
        class SteamSignalLike:
            def to_dict(self):
                return {'kind': 'steam'}

        self.assertEqual(self._rendered({'s': SteamSignalLike()}), {'s': {'kind': 'steam'}})

    def test_infinities_become_null(self):
        """**浏览器的 `JSON.parse` 解析不了 `Infinity` / `NaN` 字面量。**

        `distance=inf` 这类业务哨兵值属于此类，输出原样会让整页 JSON 解析失败。
        """
        self.assertEqual(self._rendered({'d': math.inf, 'e': -math.inf, 'n': math.nan}),
                         {'d': None, 'e': None, 'n': None})

    def test_nested_infinities_are_cleaned_too(self):
        self.assertEqual(
            self._rendered({'a': [{'b': math.inf}], 'c': (1, math.nan)}),
            {'a': [{'b': None}], 'c': [1, None]})

    def test_chinese_is_not_escaped(self):
        """`ensure_ascii=False` 照抄旧入口——转义后体积翻倍且日志没法看。"""
        self.assertIn('中文', SanitizedJSONResponse(content={'k': '中文'}).body.decode('utf-8'))

    def test_the_output_is_parseable_json(self):
        body = SanitizedJSONResponse(content={'p': np.float64(0.1), 'd': math.inf}).body
        json.loads(body.decode('utf-8'))


class FastApiDefaultWouldHaveFailed(unittest.TestCase):
    """**证明这个响应类不是多余的。**

    没有它，同样的 payload 会让 FastAPI 抛错、接口 500——线上就是这么挂的。
    """

    def _app(self, response_class=None):
        app = FastAPI(**({'default_response_class': response_class}
                         if response_class else {}))
        router = APIRouter()

        @router.get('/probe')
        async def probe():
            return {'p': np.float64(0.25), 'd': math.inf}

        app.include_router(router)
        return app

    def test_the_default_encoder_cannot_handle_this_payload(self):
        with TestClient(self._app(), raise_server_exceptions=False) as client:
            self.assertEqual(client.get('/probe').status_code, 500)

    def test_ours_handles_it(self):
        with TestClient(self._app(SanitizedJSONResponse)) as client:
            response = client.get('/probe')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'p': 0.25, 'd': None})


class WiredIntoTheApp(unittest.TestCase):

    def test_the_app_uses_it_by_default(self):
        """**设成默认响应类，而不是逐个路由包一层。**

        漏包一个路由不会有任何报错——只会在那条接口恰好返回 numpy 或 inf
        的时候才 500，而那可能是上线几天后的某个特定比赛。
        """
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertIs(app.router.default_response_class, SanitizedJSONResponse)

    def test_a_real_route_goes_through_it(self):
        from unittest import mock

        from src.api.services import football as service
        payload = {'result': {'p': np.float64(0.5), 'd': math.inf}}
        with mock.patch.object(service, 'sync_status_payload', return_value=payload):
            with TestClient(create_app(auth_settings=AuthSettings(credentials={}))) as client:
                response = client.get('/api/sync/status')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'result': {'p': 0.5, 'd': None}})

    def test_every_business_route_survives_a_numpy_payload(self):
        """**逐条路由**用带 numpy 与 inf 的 payload 打一遍。

        这是唯一能挡住这类问题的测试。此前的路由测试全都 mock 成
        `{'sentinel': ...}` 这种纯字面量，双跑差分比的又是服务函数的返回值
        （在进 FastAPI 序列化之前就比完了）——**两层都绕开了真正会炸的
        那一步**，于是测试全绿、线上每条接口 500。
        """
        import importlib
        from unittest import mock

        payload = {'result': {'p': np.float64(0.5), 'd': math.inf,
                              'arr': np.array([1, 2])}}
        checked = 0
        for module_name, service_name in (
                ('basketball', 'basketball_matches_payload'),
                ('beidan', 'beidan_history_payload'),
                ('lottery', 'lottery_cycles_payload'),
                ('kl8', 'kl8_conflicts_payload'),
                ('football', 'sync_status_payload')):
            service = importlib.import_module(f'src.api.services.{module_name}')
            path = {
                'basketball': '/api/basketball/matches',
                'beidan': '/api/beidan/history',
                'lottery': '/api/lottery/cycles',
                'kl8': '/api/kl8/conflicts',
                'football': '/api/sync/status',
            }[module_name]
            with self.subTest(path=path):
                with mock.patch.object(service, service_name, return_value=payload):
                    with TestClient(create_app(
                            auth_settings=AuthSettings(credentials={}))) as client:
                        response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(),
                                 {'result': {'p': 0.5, 'd': None, 'arr': [1, 2]}})
                checked += 1
        self.assertEqual(checked, 5)

    def test_no_business_route_returns_a_bare_dict(self):
        """**路由必须返回 Response，不能 `return payload`。**

        `return payload` 会让 FastAPI 先跑一遍 `jsonable_encoder`——
        `default_response_class` 挡不住那一步，它只管最后的渲染。
        所以业务路由一律走 `json_result(...)`，漏一处就是一条会 500 的接口。
        """
        import ast
        import pathlib

        for module in ('basketball', 'beidan', 'lottery', 'kl8', 'football'):
            source = pathlib.Path(f'src/api/routers/{module}.py').read_text(encoding='utf-8')
            with self.subTest(module=module):
                self.assertNotIn('await run_blocking(', source,
                                 'run_blocking 的返回值是裸 dict，要用 json_result')

    def test_it_is_the_same_cleaning_as_the_old_entry_point(self):
        """两边共用一份清洗函数——各写一份必然会漂（判据 11）。"""
        import inspect

        from src.api import responses
        source = inspect.getsource(responses)
        self.assertIn('from src.webapp.http_util import _json_default, _sanitize_json',
                      source)


if __name__ == '__main__':
    unittest.main()
