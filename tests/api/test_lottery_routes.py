# -*- coding: utf-8 -*-
"""彩票接口在新入口上的行为。

业务逻辑住在 `src.api.services.lottery`，**新旧两个入口共用同一份**
（判据 11）——旧的 `src/webapp/lottery_api.py` 从 345 行缩到 62 行转发。

迁移当时跑过 17 条新旧双跑差分，只有 2 条不一致（`top_n=abc`、
`periods=abc`），与前两批同类：旧入口把参数错误伪装成 `200` 里的 error 串，
新入口返回 422 并同时给 `error` 与 `detail`。

**差分刻意不比 `*-refresh` / `fetch` / `recommend`**：它们会真的触发重算
或抓取（3D 的 ML 训练约 22 秒），跑两遍既慢，又会因为"第二遍读到第一遍
的结果"制造假差异。那几条的参数传递由本文件单独覆盖。
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.services import lottery as service

#: 旧入口的十七条彩票路由。**一条都不能少**——网页里的地址是写死的，
#: 少一条或改一个字都是切换当天才发现的故障。
OLD_PATHS = {
    '/api/3d', '/api/3d-ml', '/api/3d-refresh',
    '/api/ssq', '/api/ssq-refresh',
    '/api/lottery', '/api/lottery-refresh', '/api/lottery/task-status',
    '/api/lottery/recommend', '/api/lottery/rank', '/api/lottery/ensemble',
    '/api/lottery/cycles', '/api/lottery/contribution', '/api/lottery/backtest',
    '/api/lottery/fetch', '/api/lottery/ml', '/api/lottery/ml-refresh',
}


def make_client():
    return TestClient(create_app(auth_settings=AuthSettings(credentials={})))


class Routing(unittest.TestCase):

    def test_every_old_path_still_exists(self):
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertEqual(OLD_PATHS - set(app.openapi()['paths']), set())

    def test_the_paths_are_not_all_under_one_prefix(self):
        """`/api/3d` 与 `/api/lottery` 不同前缀是**旧入口的既有形状**，
        照抄不是疏忽。统一前缀会让网页里所有写死的地址失效。
        """
        self.assertIn('/api/3d', OLD_PATHS)
        self.assertIn('/api/lottery', OLD_PATHS)

    def test_lottery_routes_are_not_public(self):
        client = TestClient(create_app(auth_settings=AuthSettings(credentials={'a': 'b'})))
        with client:
            self.assertEqual(
                client.get('/api/lottery', headers={'accept': 'application/json'}
                           ).status_code, 401)


class ParameterPassing(unittest.TestCase):

    def _captured(self, url, function):
        with mock.patch.object(service, function, return_value={}) as spy:
            with make_client() as client:
                client.get(url)
        return spy.call_args[0][0] if spy.call_args and spy.call_args[0] else None

    def test_the_rank_top_n_defaults_to_ten(self):
        self.assertEqual(self._captured('/api/lottery/rank', 'lottery_rank_payload'),
                         {'top_n': [10]})

    def test_the_backtest_defaults_match_the_old_entry_point(self):
        self.assertEqual(self._captured('/api/lottery/backtest', 'lottery_backtest_payload'),
                         {'method': ['balanced'], 'periods': [30]})

    def test_query_parameters_reach_the_service(self):
        self.assertEqual(
            self._captured('/api/lottery/backtest?method=aggressive&periods=5',
                           'lottery_backtest_payload'),
            {'method': ['aggressive'], 'periods': [5]})


class BacktestFlagStaysAString(unittest.TestCase):
    """`/api/3d-refresh?backtest=` 认的是 `1/true/yes/on` 四个字面量。

    换成 FastAPI 的 `bool` 会连 `y`、`t`、`on` 之外的写法一起认，
    那是**悄悄放宽**——3D 的回测很重，多跑一次要二十几秒。
    """

    def _forwarded(self, value):
        with mock.patch.object(service, 'lottery_3d_refresh_payload',
                               return_value={}) as spy:
            with make_client() as client:
                client.get(f'/api/3d-refresh?backtest={value}')
        return spy.call_args[0][0]['backtest'][0]

    def test_values_pass_through_unchanged(self):
        for value in ('1', 'true', 'yes', 'on', 'y', 't', '0', ''):
            with self.subTest(value=value):
                self.assertEqual(self._forwarded(value), value)

    def test_the_default_is_off(self):
        with mock.patch.object(service, 'lottery_3d_refresh_payload',
                               return_value={}) as spy:
            with make_client() as client:
                client.get('/api/3d-refresh')
        self.assertEqual(spy.call_args[0][0], {'backtest': ['0']})

    def test_the_service_only_honours_four_literals(self):
        """**反方向**：服务层这条规则本身也要有测试，否则上面测的是空气。"""
        import inspect
        source = inspect.getsource(service.lottery_3d_refresh_payload)
        self.assertIn("('1', 'true', 'yes', 'on')", source)

    def test_the_web_pages_post_method_is_supported(self):
        """网页刷新按钮使用 POST；GET 仅为兼容旧调用保留。"""
        with mock.patch.object(service, 'lottery_3d_refresh_payload',
                               return_value={'processing': True}) as spy:
            with make_client() as client:
                response = client.post('/api/3d-refresh?backtest=1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'processing': True})
        self.assertEqual(spy.call_args[0][0], {'backtest': ['1']})


class RefreshRequestMethods(unittest.TestCase):

    def test_lottery_refresh_accepts_the_web_pages_post(self):
        with mock.patch.object(service, 'lottery_refresh_payload',
                               return_value={'processing': True}) as spy:
            with make_client() as client:
                response = client.post('/api/lottery-refresh')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'processing': True})
        spy.assert_called_once_with()


class NoArgumentRoutes(unittest.TestCase):
    """十一条不收参数的路由——**转发时漏传或多传都不会有编译期错误**。"""

    CASES = (
        ('/api/3d', 'lottery_3d_payload'),
        ('/api/3d-ml', 'lottery_3d_ml_payload'),
        ('/api/ssq', 'ssq_payload'),
        ('/api/ssq-refresh', 'ssq_refresh_payload'),
        ('/api/lottery', 'lottery_payload'),
        ('/api/lottery-refresh', 'lottery_refresh_payload'),
        ('/api/lottery/task-status', 'lottery_task_status_payload'),
        ('/api/lottery/ensemble', 'lottery_ensemble_payload'),
        ('/api/lottery/cycles', 'lottery_cycles_payload'),
        ('/api/lottery/contribution', 'lottery_contribution_payload'),
        ('/api/lottery/fetch', 'lottery_fetch_payload'),
        ('/api/lottery/ml', 'lottery_ml_payload'),
        ('/api/lottery/ml-refresh', 'lottery_ml_refresh_payload'),
    )

    def test_each_route_calls_its_own_service_function(self):
        """接错一条不会报错，只会安静地返回另一个接口的数据。"""
        for path, function in self.CASES:
            with self.subTest(path=path):
                with mock.patch.object(service, function,
                                       return_value={'sentinel': path}) as spy:
                    with make_client() as client:
                        response = client.get(path)
                self.assertTrue(spy.called)
                self.assertEqual(response.json(), {'sentinel': path})


class ValidationErrors(unittest.TestCase):

    def test_bad_integers_are_422_with_both_error_shapes(self):
        with make_client() as client:
            for url, field in (('/api/lottery/rank?top_n=abc', 'top_n'),
                               ('/api/lottery/backtest?periods=abc', 'periods')):
                with self.subTest(url=url):
                    response = client.get(url)
                    self.assertEqual(response.status_code, 422)
                    self.assertIn(field, response.json()['error'])
                    self.assertTrue(response.json()['detail'])


class SharedImplementation(unittest.TestCase):


    def test_the_lifted_functions_carry_no_self(self):
        """机械提升最容易留下的痕迹是一个悬空的 `self` 形参。"""
        import ast
        import pathlib
        tree = ast.parse(pathlib.Path('src/api/services/lottery.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                with self.subTest(function=node.name):
                    self.assertNotIn('self', [a.arg for a in node.args.args])


if __name__ == '__main__':
    unittest.main()
