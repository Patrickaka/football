# -*- coding: utf-8 -*-
"""高频写入的持久化必须是增量的，不能整表 DELETE + 全量 INSERT。

线上三天写出 11GB binlog，而业务数据本体只有 290MB：每场比赛结算都把
similar_market（19762 行）和 elo_rating/elo_history 整表重写一遍。
row-based binlog 会把删掉的行和插入的行都记下来，写放大成百倍。
"""

import unittest
from unittest import mock

from src.common import doc_store, repositories


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sink.append(('execute', ' '.join(sql.split()), params))

    def executemany(self, sql, rows):
        self.sink.append(('executemany', ' '.join(sql.split()), list(rows)))


class FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return FakeCursor(self.sink)

    def begin(self):
        self.sink.append(('begin', '', None))

    def commit(self):
        self.sink.append(('commit', '', None))

    def rollback(self):
        self.sink.append(('rollback', '', None))


def _statements(sink):
    return [entry[1] for entry in sink if entry[0] in ('execute', 'executemany')]


def _rows_written(sink):
    written = 0
    for kind, _sql, payload in sink:
        if kind == 'executemany':
            written += len(payload)
    return written


class SimilarMarketAppendsOnly(unittest.TestCase):
    """相似盘口库只增不改，保存时只写新增的那几行。"""

    def _save(self, stored_count, records):
        sink = []
        with mock.patch.object(doc_store.db, 'get_connection', return_value=FakeConn(sink)), \
                mock.patch.object(doc_store.db, 'query', return_value=[{'c': stored_count}]):
            repositories.similar_market_save({'records': records})
        return sink

    def _record(self, asian):
        return {col: None for col in repositories.SIMILAR_COLS} | {'asian': asian}

    def test_only_the_new_tail_is_inserted(self):
        sink = self._save(5, [self._record(i) for i in range(7)])

        self.assertEqual(_rows_written(sink), 2)
        self.assertFalse([s for s in _statements(sink) if s.startswith('DELETE')])

    def test_unchanged_database_is_not_written_at_all(self):
        sink = self._save(7, [self._record(i) for i in range(7)])

        self.assertEqual(_rows_written(sink), 0)
        self.assertFalse([s for s in _statements(sink) if s.startswith('DELETE')])

    def test_shrinking_record_set_falls_back_to_a_full_rewrite(self):
        sink = self._save(9, [self._record(i) for i in range(3)])

        self.assertIn('DELETE FROM similar_market', _statements(sink))
        self.assertEqual(_rows_written(sink), 3)


class EloSaveWritesOnlyWhatChanged(unittest.TestCase):

    RATINGS_SQL = "SELECT team FROM elo_rating"
    HISTORY_SQL = "SELECT team, rating, date, event FROM elo_history ORDER BY id"

    def _save(self, stored_teams, stored_history, data):
        sink = []

        def fake_query(sql, params=None):
            if 'elo_rating' in sql:
                return [{'team': t} for t in stored_teams]
            return [
                {'team': team, 'rating': e[0], 'date': e[1], 'event': e[2]}
                for team, entries in stored_history.items() for e in entries
            ]

        with mock.patch.object(repositories.db, 'get_connection', return_value=FakeConn(sink)), \
                mock.patch.object(repositories.db, 'query', side_effect=fake_query):
            repositories.elo_save(data)
        return sink

    def test_ratings_are_upserted_not_wiped(self):
        sink = self._save(
            ['A', 'B'], {},
            {'ratings': {'A': 1500.0, 'B': 1600.0}, 'history': {}, 'updated_at': 'now'},
        )

        statements = _statements(sink)
        self.assertTrue(any('ON DUPLICATE KEY UPDATE' in s for s in statements))
        self.assertFalse([s for s in statements if s.startswith('DELETE FROM elo_rating')])

    def test_teams_that_disappeared_are_removed(self):
        sink = self._save(
            ['A', 'B', 'GONE'], {},
            {'ratings': {'A': 1500.0, 'B': 1600.0}, 'history': {}, 'updated_at': 'now'},
        )

        deletes = [(s, p) for kind, s, p in sink if s.startswith('DELETE FROM elo_rating')]
        self.assertEqual(len(deletes), 1)
        self.assertIn('GONE', deletes[0][1])

    def test_history_of_untouched_teams_is_left_alone(self):
        entries = [{'rating': 1500.0, 'date': '2026-08-01', 'event': 'x'}]
        stored = {'A': [(1500.0, '2026-08-01', 'x')], 'B': [(1600.0, '2026-08-02', 'y')]}
        data = {
            'ratings': {'A': 1500.0, 'B': 1600.0},
            'history': {
                'A': entries,
                'B': [{'rating': 1600.0, 'date': '2026-08-02', 'event': 'y'}],
            },
            'updated_at': 'now',
        }

        sink = self._save(['A', 'B'], stored, data)

        self.assertFalse([s for s in _statements(sink) if 'elo_history' in s])

    def test_only_the_changed_team_history_is_rewritten(self):
        stored = {'A': [(1500.0, '2026-08-01', 'x')], 'B': [(1600.0, '2026-08-02', 'y')]}
        data = {
            'ratings': {'A': 1510.0, 'B': 1600.0},
            'history': {
                'A': [
                    {'rating': 1500.0, 'date': '2026-08-01', 'event': 'x'},
                    {'rating': 1510.0, 'date': '2026-09-01', 'event': 'match'},
                ],
                'B': [{'rating': 1600.0, 'date': '2026-08-02', 'event': 'y'}],
            },
            'updated_at': 'now',
        }

        sink = self._save(['A', 'B'], stored, data)

        history_writes = [(s, p) for kind, s, p in sink if 'elo_history' in s]
        self.assertEqual(len(history_writes), 2)  # 只有 A：一次 DELETE + 一次批量插入
        self.assertTrue(history_writes[0][0].startswith('DELETE FROM elo_history'))
        self.assertEqual(history_writes[0][1], ('A',))
        self.assertEqual(len(history_writes[1][1]), 2)


if __name__ == '__main__':
    unittest.main()
