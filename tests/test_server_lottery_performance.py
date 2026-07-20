import unittest
from unittest.mock import patch

import server


class ServerLotteryPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.handler = server.Handler.__new__(server.Handler)
        self.handler._log = server.log
        server._CACHE['lottery']['data'] = None
        server._CACHE['lottery']['timestamp'] = 0

    def test_normal_lottery_load_uses_fast_prediction_options(self):
        result = {'data_quality': {'issues': 100}, 'recommendations': {}}
        with patch.object(server, 'lottery_run_prediction', return_value=result) as run:
            payload = self.handler._lottery_payload()

        self.assertEqual(payload['result'], result)
        run.assert_called_once_with(
            force_refresh=False,
            enable_backtest=False,
            enable_ml=True,
            enable_fusion=True,
            compute_weights=False,
        )

    def test_refresh_endpoint_returns_background_task_immediately(self):
        job = {
            'task_id': 'job-1',
            'status': 'processing',
            'message': 'started',
        }
        with patch.object(server, '_start_lottery_refresh_job', return_value=job):
            payload = self.handler._lottery_refresh_payload()

        self.assertTrue(payload['processing'])
        self.assertEqual(payload['task_id'], 'job-1')

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
