"""预测历史加载时逐行精简时间线，整表完整对象不得同时驻留内存。

线上 919 条记录 185 MB JSON 整表解析后再裁剪，Python 堆高水位不会回落，
进程启动即 1.75 GB。裁剪必须发生在每一行反序列化之后、追加进列表之前。
"""
import json
import unittest
from datetime import datetime
from unittest import mock

from src.common import doc_store, repositories
from src.football import result_sync
from src.football.result_sync import PredictionHistory

NOW = datetime(2026, 9, 11, 10, 0)


def _timeline(n):
    return [{'captured_at': f'2026-08-20T1{i}:00:00', 'odds': {'i': i}} for i in range(n)]


def _record(match_id, match_time, settled, timeline_len=5):
    return {
        'match_id': match_id, 'match_time': match_time, 'settled': settled,
        'home': 'A', 'away': 'B', 'market_timeline': _timeline(timeline_len),
    }


def _row(record, created_at='2026-08-20'):
    return {'created_at': created_at, 'match_id': record['match_id'],
            'doc': json.dumps(record, ensure_ascii=False)}


class _FakeCursor:

    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql = sql

    def __iter__(self):
        return iter(self.rows)


class _FakeConnection:

    def __init__(self, rows):
        self.rows = rows
        self.cursor_classes = []

    def cursor(self, cursor_class=None):
        self.cursor_classes.append(cursor_class)
        return _FakeCursor(self.rows)


class IterQueryTests(unittest.TestCase):

    def test_iter_query_streams_rows_through_an_unbuffered_dict_cursor(self):
        from src.common import db
        conn = _FakeConnection([{'doc': '1'}, {'doc': '2'}])
        with mock.patch.object(db, 'get_connection', return_value=conn):
            rows = db.iter_query('SELECT doc FROM t')
            self.assertEqual([r['doc'] for r in rows], ['1', '2'])

        self.assertEqual(conn.cursor_classes, [db.SSDictCursor])


class LoadAllTransformTests(unittest.TestCase):

    def setUp(self):
        self._orig_iter = doc_store.db.iter_query
        self._orig_fallback = doc_store._fallback_load_all
        doc_store.clear_degradation('t')

    def tearDown(self):
        doc_store.db.iter_query = self._orig_iter
        doc_store._fallback_load_all = self._orig_fallback
        doc_store.clear_degradation('t')

    def test_transform_is_applied_to_every_row_before_it_is_returned(self):
        rows = [_row({'match_id': 'b', 'n': 2}, '2026-09-02'),
                _row({'match_id': 'a', 'n': 1}, '2026-09-01')]
        doc_store.db.iter_query = lambda sql, params=None: iter(rows)

        loaded = doc_store.load_all('t', order_by='created_at, match_id',
                                    transform=lambda r: {**r, 'n': r['n'] * 10})

        self.assertEqual([r['n'] for r in loaded], [10, 20])

    def test_each_row_is_transformed_before_the_next_row_is_fetched(self):
        pulled = []

        def stream(sql, params=None):
            for n in (1, 2):
                pulled.append(n)
                yield _row({'match_id': str(n), 'n': n})

        seen_when_transforming = []

        def transform(record):
            seen_when_transforming.append(list(pulled))
            return record

        doc_store.db.iter_query = stream
        doc_store.load_all('t', order_by='created_at, match_id', transform=transform)

        self.assertEqual(seen_when_transforming, [[1], [1, 2]])

    def test_a_transform_error_propagates_instead_of_degrading_to_the_snapshot(self):
        doc_store.db.iter_query = lambda sql, params=None: iter([_row({'match_id': '1'})])
        doc_store._fallback_load_all = lambda table: [{'n': 'snapshot'}]

        def corrupt(record):
            raise ValueError('corrupt archive')

        with self.assertRaisesRegex(ValueError, 'corrupt archive'):
            doc_store.load_all('t', order_by='created_at, match_id', transform=corrupt)
        self.assertIsNone(doc_store.degradation('t'))

    def test_an_error_in_the_middle_of_the_stream_still_falls_back_visibly(self):
        def stream(sql, params=None):
            yield _row({'match_id': '1', 'n': 1})
            raise RuntimeError('Lost connection')

        doc_store.db.iter_query = stream
        doc_store._fallback_load_all = lambda table: [{'n': 'snapshot'}]

        loaded = doc_store.load_all('t', order_by='created_at, match_id')

        self.assertEqual(loaded, [{'n': 'snapshot'}])
        self.assertIn('Lost connection', doc_store.degradation('t')['error'])


class RepositoryTransformTests(unittest.TestCase):

    def setUp(self):
        self._orig_iter = doc_store.db.iter_query
        doc_store.clear_degradation('football_prediction')

    def tearDown(self):
        doc_store.db.iter_query = self._orig_iter
        doc_store.clear_degradation('football_prediction')

    def test_transform_runs_after_archive_decoding(self):
        record = _record('old', '2026-08-20 20:00', True)
        doc_store.db.iter_query = lambda sql, params=None: iter([_row(record)])
        seen = []

        def transform(decoded):
            seen.append(len(decoded['market_timeline']))
            decoded['market_timeline'] = decoded['market_timeline'][-1:]
            return decoded

        loaded = repositories.football_prediction_load(transform=transform)

        self.assertEqual(seen, [5])
        self.assertEqual(len(loaded[0]['market_timeline']), 1)


class HistoryLeanLoadTests(unittest.TestCase):

    def test_history_offloads_stale_timelines_row_by_row_during_load(self):
        with mock.patch.object(result_sync.repositories, 'football_prediction_load',
                               return_value=[]) as load, \
             mock.patch.object(result_sync, 'datetime', wraps=datetime) as dt:
            dt.now.return_value = NOW
            PredictionHistory()

        transform = load.call_args.kwargs['transform']
        trimmed_old = transform(_record('old', '2026-08-20 20:00', True))
        trimmed_recent = transform(_record('recent', '2026-09-10 20:00', True))
        trimmed_pending = transform(_record('pending', '2026-08-20 20:00', False))

        self.assertEqual(trimmed_old['market_timeline'], _timeline(5)[-1:])
        self.assertTrue(trimmed_old[result_sync.TIMELINE_OFFLOADED])
        self.assertEqual(len(trimmed_recent['market_timeline']), 5)
        self.assertEqual(len(trimmed_pending['market_timeline']), 5)


if __name__ == '__main__':
    unittest.main()
