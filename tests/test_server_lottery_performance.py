import unittest
from unittest.mock import patch

import server
from src.webapp import caching as webapp_caching
from src.webapp import jobs as webapp_jobs
from src.webapp import lazy_modules as webapp_lazy


class ServerLotteryPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.handler = server.Handler.__new__(server.Handler)
        self.handler._log = server.log
        server._CACHE['lottery']['data'] = None
        server._CACHE['lottery']['timestamp'] = 0

    def test_normal_lottery_load_uses_fast_prediction_options(self):
        result = {'data_quality': {'issues': 100}, 'recommendations': {}}
        with patch.object(webapp_lazy, 'lottery_run_prediction', return_value=result) as run:
            payload = self.handler._lottery_payload()

        self.assertEqual(payload['result'], result)
        run.assert_called_once_with(
            force_refresh=False,
            enable_backtest=False,
            enable_ml=False,
            enable_fusion=False,
            compute_weights=False,
        )

    def test_refresh_endpoint_returns_background_task_immediately(self):
        job = {
            'task_id': 'job-1',
            'status': 'processing',
            'message': 'started',
        }
        with patch.object(webapp_jobs, '_start_lottery_refresh_job', return_value=job):
            payload = self.handler._lottery_refresh_payload()

        self.assertTrue(payload['processing'])
        self.assertEqual(payload['task_id'], 'job-1')

    def test_3d_refresh_endpoint_returns_background_task_immediately(self):
        job = {
            'task_id': '3d-job-1',
            'status': 'processing',
            'message': 'started',
        }
        with patch.object(webapp_jobs, '_start_3d_refresh_job', return_value=job) as start:
            payload = self.handler._lottery_3d_refresh_payload({'backtest': ['1']})

        self.assertTrue(payload['processing'])
        self.assertEqual(payload['task_id'], '3d-job-1')
        self.assertTrue(payload['backtest_enabled'])
        start.assert_called_once_with(enable_backtest=True)

    def test_3d_refresh_reuses_active_job(self):
        with server.LOTTERY_BACKGROUND_LOCK:
            server.LOTTERY_BACKGROUND_JOBS.clear()
            server.LOTTERY_BACKGROUND_JOBS['3d-existing'] = {
                'task_id': '3d-existing',
                'kind': '3d_refresh',
                'status': 'processing',
                'created_at': server.time.time(),
            }
        with patch.object(server.threading, 'Thread') as thread:
            job = server._start_3d_refresh_job(enable_backtest=False)

        self.assertEqual(job['task_id'], '3d-existing')
        thread.assert_not_called()

    def test_fetch_endpoint_uses_same_background_fast_path(self):
        job = {'task_id': 'job-3', 'status': 'processing', 'message': 'started'}
        with patch.object(webapp_jobs, '_start_lottery_refresh_job', return_value=job):
            payload = self.handler._lottery_fetch_payload()

        self.assertTrue(payload['processing'])
        self.assertEqual(payload['task_id'], 'job-3')

    def test_recommend_endpoint_reuses_prediction_snapshot(self):
        snapshot = {
            'recommendations': {
                'balanced': {
                    'front': [1, 2, 3, 4, 5],
                    'back': [1, 2],
                    'method': 'balanced',
                    'label': '均衡',
                },
            },
            'portfolio_policy': {'ticket_count': 1},
            'back_coverage_profile': {'unique_number_count': 2},
            'version': 'test-version',
        }
        with patch.object(webapp_lazy, 'lottery_run_prediction', return_value=snapshot) as run:
            payload = self.handler._lottery_recommend_payload({})

        self.assertEqual(payload['result']['recommendations'][0]['front'], [1, 2, 3, 4, 5])
        self.assertEqual(payload['result']['version'], 'test-version')
        run.assert_called_once_with(
            force_refresh=False,
            enable_backtest=False,
            enable_ml=False,
            enable_fusion=False,
            compute_weights=False,
        )

    def test_task_status_exposes_background_registry(self):
        with server.LOTTERY_BACKGROUND_LOCK:
            server.LOTTERY_BACKGROUND_JOBS.clear()
            server.LOTTERY_BACKGROUND_JOBS['job-2'] = {
                'task_id': 'job-2',
                'kind': 'lottery_refresh',
                'status': 'done',
                'created_at': server.time.time(),
            }

        payload = self.handler._lottery_task_status_payload()
        self.assertEqual(payload['job-2']['status'], 'done')


if __name__ == '__main__':
    unittest.main()
