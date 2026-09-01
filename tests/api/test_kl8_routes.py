# -*- coding: utf-8 -*-
"""快乐 8 接口在新入口上的行为。

业务逻辑住在 `src.api.services.kl8`，**新旧两个入口共用同一份**（判据 11）
——旧的 `src/webapp/kl8_api.py` 从 682 行缩到 68 行转发。

**这一批的参数原样透传，不加 FastAPI 类型标注**，所以双跑差分 18 条
**零差异**——连非法值的行为都一致。前三批之所以有 2~3 条差异，正是因为
给参数标了类型：标了就会在非法时返回 422，而旧入口是把错误吞成 `200` 里的
一句 error 串。kl8 这一族的参数全是字符串语义（服务层自己转类型、自己定
默认值，`window_size_str` 这类名字就是证据），逐个引入那种差异不值得。

差分不覆盖 `fetch` / `kl8-refresh` / `parameter-search` / `backtest`：
它们会真的抓取、清缓存或起长任务。**`backtest` 实测单次 >40 秒**，
跑两遍就是一分半。它们的参数传递在本文件用 mock 单独测。
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.deps import query_params
from src.api.services import kl8 as service

OLD_PATHS = {
    '/api/kl8', '/api/kl8-refresh', '/api/kl8-refresh/start', '/api/kl8/fetch',
    '/api/kl8-refresh/status',
    '/api/kl8/exclude-recalculate', '/api/kl8/snapshots', '/api/kl8/records',
    '/api/kl8/settle', '/api/kl8/backtest', '/api/kl8/parameter-search',
    '/api/kl8/parameter-search/start', '/api/kl8/parameter-search/status',
    '/api/kl8/integrity', '/api/kl8/conflicts', '/api/kl8/activate',
}


def make_client():
    return TestClient(create_app(auth_settings=AuthSettings(credentials={})))


class Routing(unittest.TestCase):

    def test_every_old_path_still_exists(self):
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertEqual(OLD_PATHS - set(app.openapi()['paths']), set())

    def test_the_nested_parameter_search_paths_are_distinct(self):
        """`/parameter-search`、`/start`、`/status` 是三条不同的路由。
        少一条不会报错——只会静静地落到最短的那条上。
        """
        app = create_app(auth_settings=AuthSettings(credentials={}))
        paths = set(app.openapi()['paths'])
        for suffix in ('', '/start', '/status'):
            with self.subTest(suffix=suffix):
                self.assertIn(f'/api/kl8/parameter-search{suffix}', paths)

    def test_kl8_routes_are_not_public(self):
        client = TestClient(create_app(auth_settings=AuthSettings(credentials={'a': 'b'})))
        with client:
            self.assertEqual(
                client.get('/api/kl8', headers={'accept': 'application/json'}
                           ).status_code, 401)


class QueryPassThrough(unittest.TestCase):
    """`query_params` 把 `request.query_params` 还原成 `parse_qs` 形状。"""

    def _captured(self, url, function):
        with mock.patch.object(service, function, return_value={}) as spy:
            with make_client() as client:
                client.get(url)
        return spy.call_args[0][0] if spy.call_args and spy.call_args[0] else None

    def test_values_arrive_as_single_item_lists(self):
        self.assertEqual(
            self._captured('/api/kl8/settle?issue=2026001', 'kl8_settle_payload'),
            {'issue': ['2026001']})

    def test_records_pagination_reaches_the_service(self):
        self.assertEqual(
            self._captured(
                '/api/kl8/records?page=2&page_size=8',
                'kl8_records_payload',
            ),
            {'page': ['2'], 'page_size': ['8']},
        )

    def test_refresh_status_job_id_reaches_the_service(self):
        self.assertEqual(
            self._captured(
                '/api/kl8-refresh/status?job_id=refresh-123',
                'kl8_refresh_status_payload',
            ),
            {'job_id': ['refresh-123']},
        )

    def test_repeated_keys_keep_every_value(self):
        """`?x=1&x=2` 在 `parse_qs` 下是 `{'x': ['1', '2']}`——
        用 `dict(request.query_params)` 会**只留最后一个**，静静丢数据。
        """
        self.assertEqual(
            self._captured('/api/kl8/settle?numbers=1&numbers=2', 'kl8_settle_payload'),
            {'numbers': ['1', '2']})

    def test_no_query_means_an_empty_dict(self):
        self.assertEqual(self._captured('/api/kl8/settle', 'kl8_settle_payload'), {})

    def test_blank_values_are_kept_not_dropped(self):
        """`?issue=` 要留成 `{'issue': ['']}`——丢掉它会让服务层走进
        「没传这个参数」的分支，那是另一条路。
        """
        self.assertEqual(
            self._captured('/api/kl8/settle?issue=', 'kl8_settle_payload'),
            {'issue': ['']})

    def test_unknown_parameters_are_forwarded_too(self):
        """透传就是透传——不认识的键也得带上，服务层自己决定理不理。"""
        self.assertEqual(
            self._captured('/api/kl8/activate?没见过的键=1', 'kl8_activate_payload'),
            {'没见过的键': ['1']})

    def test_all_thirteen_activate_parameters_get_through(self):
        """activate 的十三个参数**一个都不能漏**，漏掉的会静静用回默认值。"""
        keys = ('play_type', 'feature_weights', 'model_weights', 'window_size',
                'repeat_direction', 'repeat_avoid_score', 'repeat_non_avoid_score',
                'repeat_follow_score', 'repeat_non_follow_score', 'pool_diversify',
                'pool_max_last_numbers', 'auto_activate', 'n_permutations')
        query = '&'.join(f'{key}=v{i}' for i, key in enumerate(keys))
        captured = self._captured(f'/api/kl8/activate?{query}', 'kl8_activate_payload')
        self.assertEqual(captured, {key: [f'v{i}'] for i, key in enumerate(keys)})


class SlowRoutesStillWired(unittest.TestCase):
    """差分跑不动的那几条——**接线本身仍要有测试**，否则它们是唯一
    没人看过的路由。
    """

    CASES = (
        ('/api/kl8/backtest?periods=5', 'kl8_backtest_payload', {'periods': ['5']}),
        ('/api/kl8/parameter-search?async=1', 'kl8_parameter_search_payload',
         {'async': ['1']}),
        ('/api/kl8/parameter-search/start?top_n=3',
         'kl8_parameter_search_start_payload', {'top_n': ['3']}),
        ('/api/kl8/fetch', 'kl8_fetch_payload', None),
        ('/api/kl8-refresh', 'kl8_refresh_payload', None),
    )

    def test_each_slow_route_reaches_its_own_service_function(self):
        for path, function, expected in self.CASES:
            with self.subTest(path=path):
                with mock.patch.object(service, function,
                                       return_value={'sentinel': path}) as spy:
                    with make_client() as client:
                        response = client.get(path)
                self.assertEqual(response.json(), {'sentinel': path})
                if expected is not None:
                    self.assertEqual(spy.call_args[0][0], expected)

    def test_refresh_accepts_the_web_pages_post(self):
        with mock.patch.object(service, 'kl8_refresh_payload',
                               return_value={
                                   'success': True,
                                   'result': {'marker': 'finished'},
                               }) as spy:
            with make_client() as client:
                response = client.post('/api/kl8-refresh')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            'success': True,
            'result': {'marker': 'finished'},
        })
        spy.assert_called_once_with()

    def test_refresh_start_post_reaches_the_job_service(self):
        queued = {
            'success': True,
            'result': {'job_id': 'refresh-123', 'status': 'queued'},
        }
        with mock.patch.object(
                service, 'kl8_refresh_start_payload', return_value=queued,
        ) as spy:
            with make_client() as client:
                response = client.post('/api/kl8-refresh/start')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), queued)
        spy.assert_called_once_with()


class SharedImplementation(unittest.TestCase):



    def test_no_lifted_function_kept_a_self_parameter(self):
        import ast
        import pathlib
        tree = ast.parse(pathlib.Path('src/api/services/kl8.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                with self.subTest(function=node.name):
                    self.assertNotIn('self', [a.arg for a in node.args.args])


if __name__ == '__main__':
    unittest.main()
