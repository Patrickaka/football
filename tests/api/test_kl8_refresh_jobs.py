# -*- coding: utf-8 -*-
"""快乐8手动刷新必须异步执行，并把完成结果写回统一缓存。"""

import threading
import time
import unittest
from unittest import mock

from src.api.runtime import jobs as runtime_jobs
from src.api.runtime import kl8_cache, shared_cache
from src.api.services import kl8 as service
from src.foundation.cache import Cache, MemoryBackend


class KL8RefreshJobTests(unittest.TestCase):

    def setUp(self):
        self.issue = f'job-test-{id(self)}'
        self.version = 'kl8-refresh-job-test-version'
        self.config_fingerprint = 'kl8-refresh-config-test'
        self.old_result = {'marker': 'old'}
        self.new_result = {
            'marker': 'new',
            'snapshot_file': 'snapshot_new-version.json',
            'strategy_config_fingerprint': self.config_fingerprint,
            'statistics': {
                'version': self.version,
                'based_on_issue': self.issue,
                'prediction_generated_at': '2026-09-01T12:00:00',
            },
        }
        self.cache = Cache(
            l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=60,
        )
        self.cache.set(
            kl8_cache.cache_key(
                self.issue,
                service._kl8_cache_version(
                    self.version, self.config_fingerprint,
                ),
            ),
            self.old_result,
        )
        shared_cache.set_cache(self.cache)
        self.addCleanup(shared_cache.reset)
        self.cache_state = mock.patch.dict(
            service._CACHE['kl8'],
            {'data': self.old_result, 'timestamp': 1, 'expire_seconds': 86400},
            clear=True,
        )
        self.cache_state.start()
        self.addCleanup(self.cache_state.stop)

        # 与现有参数搜索任务相同，预计刷新任务注册表放在 runtime.jobs。
        # 每例清空，避免已完成任务影响 singleflight 判定。
        registry = getattr(runtime_jobs, 'KL8_REFRESH_JOBS', None)
        if isinstance(registry, dict):
            self.registry_state = mock.patch.dict(registry, {}, clear=True)
            self.registry_state.start()
            self.addCleanup(self.registry_state.stop)
        threads = getattr(runtime_jobs, 'KL8_REFRESH_THREADS', None)
        if isinstance(threads, dict):
            self.thread_state = mock.patch.dict(threads, {}, clear=True)
            self.thread_state.start()
            self.addCleanup(self.thread_state.stop)

    def _contexts(self, calculate):
        """返回 patch，保持后台线程整个生命周期内有效。"""
        return (
            mock.patch.object(service, 'kl8_latest_issue', return_value=self.issue),
            mock.patch.object(
                service, '_current_kl8_predictor_version', return_value=self.version,
            ),
            mock.patch.object(
                service,
                '_current_kl8_config_fingerprint',
                return_value=self.config_fingerprint,
            ),
            mock.patch.object(service, 'kl8_run_prediction', side_effect=calculate),
        )

    def _status(self, job_id):
        return service.kl8_refresh_status_payload({'job_id': [job_id]})

    def _wait_for_status(self, job_id, expected, timeout=3):
        deadline = time.monotonic() + timeout
        waiter = threading.Event()
        last = None
        while time.monotonic() < deadline:
            payload = self._status(job_id)
            if payload.get('error'):
                self.fail(payload['error'])
            last = payload.get('result') or {}
            if last.get('status') == expected:
                return last
            waiter.wait(min(0.01, max(0, deadline - time.monotonic())))
        self.fail(f'任务未进入 {expected}，最后状态: {last}')

    def test_start_returns_queued_without_waiting_for_prediction(self):
        entered = threading.Event()
        release = threading.Event()
        returned = threading.Event()
        box = {}

        def calculate(force_refresh=False):
            self.assertTrue(force_refresh)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('测试未释放预测任务')
            return self.new_result

        def invoke():
            box['payload'] = service.kl8_refresh_start_payload()
            returned.set()

        patches = self._contexts(calculate)
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(release.set)

        caller = threading.Thread(target=invoke, name='KL8RefreshCaller')
        caller.start()
        try:
            self.assertTrue(entered.wait(2), '后台预测任务没有启动')
            returned_before_prediction = returned.wait(2)
        finally:
            release.set()
            caller.join(timeout=3)

        self.assertTrue(
            returned_before_prediction,
            '刷新请求等待预测完成，会再次触发网关 504',
        )
        job = box['payload']['result']
        self.assertTrue(box['payload']['success'])
        self.assertEqual(job['status'], 'queued')
        self.assertTrue(job['job_id'])

        running = self._status(job['job_id'])['result']
        self.assertIn(running['status'], {'running', 'completed'})
        completed = self._wait_for_status(job['job_id'], 'completed')
        self.assertEqual(completed['result'], self.new_result)

    def test_status_exposes_running_then_completed_and_writes_shared_cache(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def calculate(force_refresh=False):
            calls.append(force_refresh)
            if not force_refresh:
                raise AssertionError('完成后读取不应再次计算')
            entered.set()
            if not release.wait(5):
                raise RuntimeError('测试未释放预测任务')
            return self.new_result

        patches = self._contexts(calculate)
        clear = mock.patch.object(service, 'kl8_clear_cache')
        for patcher in (*patches, clear):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(release.set)

        start = service.kl8_refresh_start_payload()
        job_id = start['result']['job_id']
        self.assertTrue(entered.wait(2), '后台预测任务没有启动')
        running = self._status(job_id)['result']
        self.assertEqual(running['status'], 'running')
        self.assertIsNone(running.get('finished_at'))

        release.set()
        completed = self._wait_for_status(job_id, 'completed')
        self.assertEqual(completed['result'], self.new_result)
        self.assertIsNotNone(completed.get('started_at'))
        self.assertIsNotNone(completed.get('finished_at'))
        self.assertGreaterEqual(completed.get('elapsed_seconds', 0), 0)

        # job 完成后普通 GET 必须直接读到新值；只更新兼容 _CACHE 而漏掉
        # foundation/cache 会在这里重新返回 Redis 中的旧预测。
        self.assertEqual(service.kl8_payload(), {'result': self.new_result})
        self.assertEqual(calls, [True])
        self.assertEqual(service._CACHE['kl8']['data'], self.new_result)
        cached = self.cache.peek(kl8_cache.cache_key(
            self.issue,
            service._kl8_cache_version(
                self.version, self.config_fingerprint,
            ),
        ))
        self.assertIsNotNone(cached)
        self.assertEqual(cached.value, self.new_result)

    def test_exception_marks_job_failed(self):
        def calculate(force_refresh=False):
            raise RuntimeError('prediction exploded')

        patches = self._contexts(calculate)
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        start = service.kl8_refresh_start_payload()
        failed = self._wait_for_status(start['result']['job_id'], 'failed')

        self.assertTrue(failed.get('error'))
        self.assertIsNotNone(failed.get('finished_at'))
        self.assertNotIn('result', failed)

    def test_error_payload_is_a_failed_job_not_a_completed_result(self):
        patches = self._contexts(lambda force_refresh=False: {'error': '历史数据不足'})
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        start = service.kl8_refresh_start_payload()
        failed = self._wait_for_status(start['result']['job_id'], 'failed')

        self.assertTrue(failed.get('error'))
        self.assertNotIn('result', failed)

    def test_stale_result_cannot_overwrite_newer_issue_cache(self):
        stale_result = {
            'marker': 'stale',
            'statistics': {
                'based_on_issue': 'older-issue',
                'version': self.version,
            },
        }
        with mock.patch.object(service, 'kl8_latest_issue', return_value=self.issue), \
                mock.patch.object(service, 'kl8_clear_cache'), \
                mock.patch.object(
                    service, 'kl8_run_prediction', return_value=stale_result,
                ):
            payload = service.kl8_refresh_payload()

        self.assertIn('error', payload)
        self.assertEqual(service._CACHE['kl8']['data'], self.old_result)
        cached = self.cache.peek(kl8_cache.cache_key(
            self.issue,
            service._kl8_cache_version(
                self.version, self.config_fingerprint,
            ),
        ))
        self.assertIsNotNone(cached)
        self.assertEqual(cached.value, self.old_result)

    def test_late_get_cannot_roll_back_compatibility_cache(self):
        service._sync_kl8_compat_cache(self.new_result)
        stale_get = {
            'marker': 'stale-get',
            'snapshot_file': 'snapshot_old.json',
            'statistics': {
                'version': self.version,
                'based_on_issue': self.issue,
                'prediction_generated_at': '2026-09-01T11:00:00',
            },
        }

        service._sync_kl8_compat_cache(stale_get)

        self.assertIs(service._CACHE['kl8']['data'], self.new_result)

    def test_concurrent_refresh_requests_join_the_same_inflight_job(self):
        entered = threading.Event()
        release = threading.Event()
        guard = threading.Lock()
        calls = [0]

        def calculate(force_refresh=False):
            with guard:
                calls[0] += 1
            entered.set()
            if not release.wait(5):
                raise RuntimeError('测试未释放预测任务')
            return self.new_result

        patches = self._contexts(calculate)
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(release.set)

        first = service.kl8_refresh_start_payload()['result']
        self.assertTrue(entered.wait(2), '首个后台预测任务没有启动')
        second = service.kl8_refresh_start_payload()['result']
        try:
            self.assertEqual(second['job_id'], first['job_id'])
            self.assertIn(second['status'], {'queued', 'running'})
            with guard:
                self.assertEqual(calls[0], 1)
        finally:
            release.set()
        self._wait_for_status(first['job_id'], 'completed')

    def test_status_rejects_missing_and_unknown_job_ids(self):
        self.assertIn('error', service.kl8_refresh_status_payload({}))
        self.assertIn(
            'error',
            service.kl8_refresh_status_payload({'job_id': ['does-not-exist']}),
        )

    def test_stale_running_job_is_failed_and_does_not_block_retry(self):
        stale_id = 'stale-refresh-job'
        with runtime_jobs.KL8_REFRESH_LOCK:
            runtime_jobs.KL8_REFRESH_JOBS[stale_id] = {
                'job_id': stale_id,
                'status': 'running',
                'created_at': time.time() - 1000,
                'started_at': time.time() - 1000,
            }

        with mock.patch.object(
            service,
            'kl8_refresh_payload',
            return_value={'success': True, 'result': self.new_result},
        ):
            retried = service.kl8_refresh_start_payload()['result']

        self.assertNotEqual(retried['job_id'], stale_id)
        self.assertEqual(self._status(stale_id)['result']['status'], 'failed')
        self._wait_for_status(retried['job_id'], 'completed')

    def test_live_long_running_job_is_not_falsely_failed_or_duplicated(self):
        stale_id = 'live-stale-refresh-job'
        live_worker = mock.Mock()
        live_worker.is_alive.return_value = True
        with runtime_jobs.KL8_REFRESH_LOCK:
            runtime_jobs.KL8_REFRESH_JOBS[stale_id] = {
                'job_id': stale_id,
                'status': 'running',
                'created_at': time.time() - 1000,
                'started_at': time.time() - 1000,
            }
            runtime_jobs.KL8_REFRESH_THREADS[stale_id] = live_worker

        refresh_fn = mock.Mock()
        returned = runtime_jobs._start_kl8_refresh_job(refresh_fn)

        self.assertEqual(returned['job_id'], stale_id)
        self.assertEqual(returned['status'], 'running')
        self.assertIn('仍在后台执行', returned['message'])
        refresh_fn.assert_not_called()

    def test_thread_start_failure_marks_job_failed(self):
        with mock.patch.object(
            runtime_jobs.threading.Thread,
            'start',
            side_effect=RuntimeError('thread unavailable'),
        ):
            job = service.kl8_refresh_start_payload()['result']

        self.assertEqual(job['status'], 'failed')
        self.assertIn('thread unavailable', job['error'])
        self.assertEqual(self._status(job['job_id'])['result']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
