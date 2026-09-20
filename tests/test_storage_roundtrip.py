# -*- coding: utf-8 -*-
"""存储层往返测试。

这套测试以前挂着 `skipUnless(_can_connect())`——本地和 CI 都没有 MySQL，
所以它**从来没真正跑过**，是一直绿着的假绿。换成 SQLite 之后不依赖任何
外部服务，每次都真跑。

库文件建在临时目录，绝不碰 data/ 下的真实库。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import db, kv_store, repositories as repo


class StorageRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls._prev = os.environ.get('FOOTBALL_DB_PATH')
        os.environ['FOOTBALL_DB_PATH'] = str(Path(cls._dir.name) / 'roundtrip.db')
        db.close_connection()
        db.init_db()

    @classmethod
    def tearDownClass(cls):
        db.close_connection()
        if cls._prev is None:
            os.environ.pop('FOOTBALL_DB_PATH', None)
        else:
            os.environ['FOOTBALL_DB_PATH'] = cls._prev
        cls._dir.cleanup()

    def test_kv_roundtrip(self):
        obj = {'a': 1, 'b': [1, 2, 3], '中文': '值'}
        kv_store.save('t_kv', obj)
        self.assertEqual(kv_store.load('t_kv'), obj)
        self.assertTrue(kv_store.exists('t_kv'))
        kv_store.delete('t_kv')
        self.assertIsNone(kv_store.load('t_kv'))

    def test_cache_date_semantics(self):
        kv_store.save_cache('t_cache', [1, 2, 3])
        self.assertEqual(kv_store.load_cache('t_cache'), [1, 2, 3])
        db.execute("UPDATE kv_store SET cache_date='2000-01-01' WHERE k='t_cache'")
        self.assertIsNone(kv_store.load_cache('t_cache'))

    def test_football_prediction_roundtrip(self):
        records = [
            {'match_id': 'm1', 'league': '英超', 'settled': True, 'sync_status': 'done',
             'created_at': '2026-01-01T00:00:00', 'updated_at': '2026-01-02T00:00:00',
             'predicted_scores': {'1-1': 0.2}, 'extra_open_key': [1, 2]},
            {'match_id': 'm2', 'league': '西甲', 'settled': False, 'sync_status': 'pending',
             'created_at': '2026-01-03T00:00:00', 'updated_at': '2026-01-03T00:00:00'},
        ]
        repo.football_prediction_save(records)
        got = {r['match_id']: r for r in repo.football_prediction_load()}
        self.assertEqual(got['m1'], records[0])  # 开放式键完整保留
        self.assertEqual(got['m2'], records[1])

    def test_elo_roundtrip(self):
        data = {
            'ratings': {'中国': 1500.0, '泰国': 1480.5},
            'history': {
                '中国': [{'rating': 1500, 'date': '2026-06-11T15:26:18', 'event': 'initialized'}],
                '泰国': [{'rating': 1480.5, 'date': '2026-06-12T00:00:00', 'event': 'match'}],
            },
            'updated_at': '2026-06-12T00:00:00',
        }
        repo.elo_save(data)
        got = repo.elo_load()
        self.assertEqual(got['ratings'], data['ratings'])
        self.assertEqual(got['history'], data['history'])

    def test_similar_market_roundtrip(self):
        data = {'records': [
            {'asian': -2.0, 'asian_odds_home': 0.0, 'asian_odds_away': 0.0, 'total': 2.5,
             'total_over': 0.0, 'total_under': 0.0, 'euro_home': 1.08, 'euro_draw': 11.0,
             'euro_away': 21.0, 'result': 'H', 'goals_home': 3, 'goals_away': 0,
             'date': '2025-08-15', 'league': 'E0', 'home_team': 'Liverpool', 'away_team': 'Bournemouth'},
        ], 'version': '1.0', 'count': 1}
        repo.similar_market_save(data)
        got = repo.similar_market_load()
        self.assertEqual(got['count'], 1)
        self.assertEqual(got['records'][0], data['records'][0])


if __name__ == '__main__':
    unittest.main()
