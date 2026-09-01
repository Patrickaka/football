# -*- coding: utf-8 -*-
"""足球接口在新入口上的行为。

业务逻辑住在 `src.api.services.football`，**新旧两个入口共用同一份**
（判据 11）——旧的 `src/webapp/football_api.py` 从 742 行缩到 125 行。

**`_serve_report_file` 没有提升**：它直接往 `self.wfile` 写响应流，是 HTTP
层的文件服务而不是业务装配。新入口用 `FileResponse` 另做一条。

双跑差分 16 条零差异（参数原样透传，与 kl8 同策略）。

## 差分里排掉的两条，排掉的理由本身值得记

- `/api/football/review` **真的执行结算**：首轮结算了 11/12 场，第二轮
  只剩 1 场待结算，于是报"差异"。
- `/api/calibrate?league=` 会训练并写 `last_updated` 时间戳。

两条都不是回归。**差分报差异时先判断是不是副作用或时钟。**
它们的接线改用 mock 测。
"""
import pathlib
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.services import football as service

OLD_PATHS = {
    '/api/matches', '/api/predict', '/api/predict/batch',
    '/api/football/clear_cache', '/api/football/prepare_ml_data',
    '/api/football/diagnostics', '/api/football/review',
    '/api/football/professional-status',
    '/api/calibrate', '/api/calibrate/list', '/api/calibrate/clear',
    '/api/backtest', '/api/backtest/threshold',
    '/api/model/status', '/api/model/backtest_stats',
    '/api/predictions', '/api/predictions/export',
    '/api/sync/status', '/api/sync/trigger', '/api/sync/hide_failed',
}


def make_client():
    return TestClient(create_app(auth_settings=AuthSettings(credentials={})))


class Routing(unittest.TestCase):

    def test_every_old_path_still_exists(self):
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertEqual(OLD_PATHS - set(app.openapi()['paths']), set())

    def test_the_report_file_route_exists(self):
        app = create_app(auth_settings=AuthSettings(credentials={}))
        self.assertIn('/reports/{name}', app.openapi()['paths'])

    def test_football_routes_are_not_public(self):
        client = TestClient(create_app(auth_settings=AuthSettings(credentials={'a': 'b'})))
        with client:
            self.assertEqual(
                client.get('/api/predictions', headers={'accept': 'application/json'}
                           ).status_code, 401)

    def test_reports_are_not_public_either(self):
        """报告文件里有预测内容——**豁免它等于把分析结果公开**。"""
        client = TestClient(create_app(auth_settings=AuthSettings(credentials={'a': 'b'})))
        with client:
            self.assertEqual(
                client.get('/reports/anything.html',
                           headers={'accept': 'application/json'}).status_code, 401)


class ReportFileServing(unittest.TestCase):
    """`name` 来自 URL，拼接前必须挡住路径穿越。"""

    def test_traversal_attempts_are_refused(self):
        """**只检查字符串里有没有 `..` 挡不住编码变体**——
        `resolve()` 之后比对父目录才是可靠的判断。
        """
        with make_client() as client:
            for attempt in ('../../../etc/passwd',
                            '..%2f..%2f..%2fetc%2fpasswd',
                            '....//....//etc/passwd',
                            '/etc/passwd',
                            '../src/api/app.py'):
                with self.subTest(attempt=attempt):
                    self.assertEqual(
                        client.get(f'/reports/{attempt}').status_code, 404)

    def test_a_missing_report_is_a_404(self):
        with make_client() as client:
            self.assertEqual(client.get('/reports/没有这个文件.html').status_code, 404)

    def test_a_real_report_is_served(self):
        """**反方向**：挡住穿越之后，正常文件还得能取到，否则这条路由就废了。"""
        from src.api.routers.football import REPORTS_DIR
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        probe = REPORTS_DIR / '_test_probe.html'
        probe.write_text('<p>探针</p>', encoding='utf-8')
        self.addCleanup(probe.unlink)
        with make_client() as client:
            response = client.get('/reports/_test_probe.html')
            self.assertEqual(response.status_code, 200)
            self.assertIn('探针', response.text)



class BatchPrediction(unittest.TestCase):

    def test_the_body_reaches_the_service_unchanged(self):
        """**body 原样交给服务层**——形状校验是它的契约。在路由层用
        pydantic 模型挡一道会把服务层的中文错误换成 422 的 detail。
        """
        body = {'matches': [{'match_id': 'm1'}]}
        with mock.patch.object(service, 'predict_batch_payload', return_value={}) as spy:
            with make_client() as client:
                client.post('/api/predict/batch', json=body)
        self.assertEqual(spy.call_args[0][0], body)

    def test_a_malformed_body_gets_the_services_own_error(self):
        with make_client() as client:
            response = client.post('/api/predict/batch', json={'matches': []})
            self.assertEqual(response.status_code, 200)
            self.assertIn('matches', response.json()['error'])

    def test_an_empty_body_is_not_a_crash(self):
        with make_client() as client:
            response = client.post('/api/predict/batch')
            self.assertEqual(response.status_code, 200)
            self.assertIn('error', response.json())

    def test_batch_is_a_post_not_a_get(self):
        with make_client() as client:
            self.assertEqual(client.get('/api/predict/batch').status_code, 405)


class PredictionRecordsIndependentOfSchedule(unittest.TestCase):

    def test_records_route_reaches_the_records_service(self):
        payload = {'result': {'records': [{'match_id': 'history-1'}], 'count': 1}}
        with mock.patch.object(
                service, 'predictions_payload', return_value=payload,
        ) as records:
            with make_client() as client:
                response = client.get('/api/predictions')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        records.assert_called_once_with()

    def test_records_service_does_not_require_a_current_match_list(self):
        stored_records = [{'match_id': 'history-1'}]
        with mock.patch(
                'src.football.result_sync.get_prediction_records',
                return_value=stored_records,
        ) as records, mock.patch.object(
                service, 'matches_payload',
                side_effect=AssertionError('预测记录不应抓取当前赛程'),
        ) as matches:
            payload = service.predictions_payload()

        # storage_degraded 只在 MySQL 读失败时出现（无库的开发机上就会出现），
        # 这条用例只关心记录本身不依赖赛程。
        self.assertEqual(payload['result']['records'], stored_records)
        self.assertEqual(payload['result']['count'], 1)
        records.assert_called_once_with(include_hidden=False)
        matches.assert_not_called()


class SideEffectRoutesStillWired(unittest.TestCase):
    """差分排掉的那几条——**接线仍要有测试**，否则它们没人看过。"""

    CASES = (
        ('/api/football/review?date=2026-08-29', 'football_review_payload',
         {'date': ['2026-08-29']}),
        ('/api/calibrate?league=英超', 'calibrate_payload', {'league': ['英超']}),
        ('/api/matches', 'matches_payload', None),
        ('/api/football/clear_cache', 'football_clear_cache_payload', None),
        ('/api/football/prepare_ml_data', 'prepare_ml_history_data_payload', None),
        ('/api/predictions/export', 'predictions_export_payload', None),
        ('/api/sync/trigger', 'sync_trigger_payload', None),
        ('/api/sync/hide_failed', 'sync_hide_failed_payload', None),
        ('/api/calibrate/clear', 'calibrate_clear_payload', None),
    )

    def test_each_route_reaches_its_own_service_function(self):
        for path, function, expected in self.CASES:
            with self.subTest(path=path):
                with mock.patch.object(service, function,
                                       return_value={'sentinel': path}) as spy:
                    with make_client() as client:
                        response = client.get(path)
                self.assertEqual(response.json(), {'sentinel': path})
                if expected is not None:
                    self.assertEqual(spy.call_args[0][0], expected)


class SharedImplementation(unittest.TestCase):


    def test_no_lifted_function_kept_a_self_parameter(self):
        import ast
        tree = ast.parse(pathlib.Path('src/api/services/football.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                with self.subTest(function=node.name):
                    self.assertNotIn('self', [a.arg for a in node.args.args])


if __name__ == '__main__':
    unittest.main()
