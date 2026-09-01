# -*- coding: utf-8 -*-
"""doc 表整表读取不得让 MySQL 排序大 JSON 列，且降级必须可见。

线上事故：`football_prediction` 的 doc 列单条约 40KB、共 742 行，
`SELECT doc ... ORDER BY created_at, match_id` 触发
`ERROR 1038 (HY001): Out of sort memory`，load_all 静默回落到几天前的
fallback 快照，页面上凭空少了中间五天的预测记录且毫无提示。
"""

import json
import unittest

from src.common import doc_store


class DocStoreLoadAllSorting(unittest.TestCase):

    def setUp(self):
        self._orig_query = doc_store.db.query
        self._orig_fallback = doc_store._fallback_load_all
        self.executed = []
        doc_store.clear_degradation('t')

    def tearDown(self):
        doc_store.db.query = self._orig_query
        doc_store._fallback_load_all = self._orig_fallback
        doc_store.clear_degradation('t')

    def _fake_query(self, rows):
        def query(sql, params=None):
            self.executed.append(sql)
            return rows
        return query

    def test_sql_never_asks_mysql_to_sort(self):
        rows = [{'created_at': '2026-09-01', 'match_id': 'b', 'doc': '{"id": 1}'}]
        doc_store.db.query = self._fake_query(rows)

        doc_store.load_all('t', order_by='created_at, match_id')

        self.assertEqual(len(self.executed), 1)
        self.assertNotIn('ORDER BY', self.executed[0].upper())

    def test_rows_are_ordered_in_python_by_the_requested_columns(self):
        rows = [
            {'created_at': '2026-09-01', 'match_id': 'b', 'doc': json.dumps({'n': 3})},
            {'created_at': '2026-08-30', 'match_id': 'z', 'doc': json.dumps({'n': 1})},
            {'created_at': '2026-09-01', 'match_id': 'a', 'doc': json.dumps({'n': 2})},
        ]
        doc_store.db.query = self._fake_query(rows)

        loaded = doc_store.load_all('t', order_by='created_at, match_id')

        self.assertEqual([r['n'] for r in loaded], [1, 2, 3])

    def test_null_sort_values_do_not_break_ordering(self):
        rows = [
            {'created_at': '2026-09-01', 'match_id': 'a', 'doc': json.dumps({'n': 2})},
            {'created_at': None, 'match_id': 'a', 'doc': json.dumps({'n': 1})},
        ]
        doc_store.db.query = self._fake_query(rows)

        loaded = doc_store.load_all('t', order_by='created_at, match_id')

        self.assertEqual([r['n'] for r in loaded], [1, 2])

    def test_order_columns_are_selected_alongside_doc(self):
        doc_store.db.query = self._fake_query([])

        doc_store.load_all('t', order_by='id')

        self.assertIn('SELECT id, doc FROM t', self.executed[0])


class DocStoreDegradationVisibility(unittest.TestCase):

    def setUp(self):
        self._orig_query = doc_store.db.query
        self._orig_fallback = doc_store._fallback_load_all
        self._orig_connection = doc_store.db.get_connection
        self._orig_upsert_fallback = doc_store._fallback_upsert_one
        doc_store.clear_degradation('t')

    def tearDown(self):
        doc_store.db.query = self._orig_query
        doc_store._fallback_load_all = self._orig_fallback
        doc_store.db.get_connection = self._orig_connection
        doc_store._fallback_upsert_one = self._orig_upsert_fallback
        doc_store.clear_degradation('t')

    def test_load_failure_records_a_visible_degradation(self):
        def boom(sql, params=None):
            raise RuntimeError('Out of sort memory')
        doc_store.db.query = boom
        doc_store._fallback_load_all = lambda table: [{'n': 1}]

        loaded = doc_store.load_all('t', order_by='id')

        self.assertEqual(loaded, [{'n': 1}])
        state = doc_store.degradation('t')
        self.assertTrue(state)
        self.assertIn('Out of sort memory', state['error'])
        self.assertTrue(state['at'])
        self.assertEqual(state['source'], 'fallback')

    def test_successful_load_clears_a_previous_degradation(self):
        doc_store._record_degradation('t', RuntimeError('boom'))
        doc_store.db.query = lambda sql, params=None: []

        doc_store.load_all('t', order_by='id')

        self.assertIsNone(doc_store.degradation('t'))

    def test_upsert_fallback_is_recorded_too(self):
        doc_store.db.get_connection = lambda: (_ for _ in ()).throw(RuntimeError('down'))
        recorded = {}
        doc_store._fallback_upsert_one = lambda *a, **k: recorded.setdefault('called', True)

        backend = doc_store.upsert_one('t', ['match_id', 'doc'], ('m1', '{}'), ['match_id'])

        self.assertEqual(backend, 'fallback')
        self.assertTrue(recorded.get('called'))
        self.assertTrue(doc_store.degradation('t'))


class DegradationSurfacesInPredictionRecords(unittest.TestCase):
    """读到的是过期快照时，接口和页面都必须说出来。"""

    def tearDown(self):
        doc_store.clear_degradation('football_prediction')

    def test_payload_carries_the_degradation_when_storage_fell_back(self):
        from unittest import mock
        from src.api.services import football as service
        import src.football.result_sync  # 模块导入时会读一次整表，先让它发生

        doc_store._record_degradation('football_prediction', RuntimeError('Out of sort memory'))
        with mock.patch('src.football.result_sync.get_prediction_records', return_value=[]):
            payload = service.predictions_payload()

        degraded = payload['result']['storage_degraded']
        self.assertIn('Out of sort memory', degraded['error'])

    def test_payload_stays_clean_when_storage_is_healthy(self):
        from unittest import mock
        from src.api.services import football as service

        import src.football.result_sync  # 模块导入时会读一次整表，先让它发生

        doc_store.clear_degradation('football_prediction')
        with mock.patch('src.football.result_sync.get_prediction_records', return_value=[]):
            payload = service.predictions_payload()

        self.assertNotIn('storage_degraded', payload['result'])

    def test_frontend_warns_about_the_stale_snapshot(self):
        from pathlib import Path

        html = Path('web/index.html').read_text(encoding='utf-8')
        loader = html.split('async function loadPredictions()', 1)[1].split(
            'async function exportPredictionRecords()', 1,
        )[0]
        self.assertIn('storage_degraded', loader)
        self.assertIn('本地快照', loader)


if __name__ == '__main__':
    unittest.main()
