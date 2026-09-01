# -*- coding: utf-8 -*-
"""快乐8预测记录页面必须使用服务端分页和短期页面缓存。"""

import unittest
from pathlib import Path


HTML = Path('web/index.html').read_text(encoding='utf-8')


class KL8RecordsFrontend(unittest.TestCase):

    def test_records_request_contains_server_side_pagination(self):
        self.assertIn("fetchJson('/api/kl8/records?page='", HTML)
        self.assertIn("'&page_size=' + KL8_RECORDS_PAGE_SIZE", HTML)

    def test_loaded_pages_are_cached_briefly(self):
        self.assertIn('const KL8_RECORDS_CACHE_TTL = 30000;', HTML)
        self.assertIn('const kl8RecordsPageCache = new Map();', HTML)
        self.assertIn('Date.now() - cached.loadedAt < KL8_RECORDS_CACHE_TTL', HTML)

    def test_record_mutations_invalidate_the_page_cache(self):
        self.assertGreaterEqual(HTML.count('invalidateKL8RecordsCache();'), 4)

    def test_refresh_uses_background_job_contract(self):
        self.assertIn("fetchJsonPost('/api/kl8-refresh/start')", HTML)
        self.assertIn("'/api/kl8-refresh/status?job_id='", HTML)
        self.assertIn("job.status === 'completed'", HTML)
        self.assertIn("job.status === 'failed'", HTML)
        self.assertIn("job.status !== 'queued' && job.status !== 'running'", HTML)

    def test_completed_refresh_renders_job_result_without_second_prediction_get(self):
        polling_code = HTML.split('async function pollKL8Refresh(', 1)[1].split(
            'async function refreshKL8()', 1,
        )[0]
        refresh_code = HTML.split('async function refreshKL8()', 1)[1].split(
            'function buildKL8ShapeProfile', 1,
        )[0]
        self.assertIn('renderKL8(job.result);', polling_code)
        self.assertIn('kl8Loaded = true;', polling_code)
        self.assertIn('invalidateKL8RecordsCache();', polling_code)
        self.assertNotIn('loadKL8()', polling_code + refresh_code)
        self.assertNotIn("fetchJsonPost('/api/kl8-refresh')", HTML)

    def test_refresh_polling_is_bounded_and_stops_for_removed_page(self):
        self.assertIn('const KL8_REFRESH_TIMEOUT_MS = 180000;', HTML)
        self.assertIn('Date.now() - startedAt < KL8_REFRESH_TIMEOUT_MS', HTML)
        self.assertIn('appKL8.isConnected', HTML)
        self.assertIn("document.getElementById('app-kl8') === appKL8", HTML)
        self.assertIn('if (!isCurrentKL8Container(appKL8)) return;', HTML)
        self.assertIn('fetchJsonWithTimeout(', HTML)
        self.assertIn('consecutivePollErrors >= 3', HTML)

    def test_failed_refresh_keeps_previous_result_and_retry_controls(self):
        refresh_code = HTML.split('async function refreshKL8()', 1)[1].split(
            'function buildKL8ShapeProfile', 1,
        )[0]
        self.assertIn('const previousData = kl8LastData;', refresh_code)
        self.assertIn('renderKL8(previousData);', refresh_code)
        self.assertIn('kl8Loaded = true;', refresh_code)
        self.assertIn('已保留上一次结果', refresh_code)
        self.assertIn('kl8Loaded = false;', refresh_code)
        self.assertIn('onclick="refreshKL8()">重试</button>', refresh_code)

    def test_manual_recalculation_carries_exact_rendered_prediction_context(self):
        self.assertIn('function buildKL8SourceContextQuery(option)', HTML)
        for field in (
            'source_snapshot_id',
            'source_version',
            'source_target_issue',
            'source_based_on_issue',
            'source_config_fingerprint',
        ):
            self.assertIn(field, HTML)
        self.assertGreaterEqual(HTML.count('buildKL8SourceContextQuery(option)'), 3)


if __name__ == '__main__':
    unittest.main()
