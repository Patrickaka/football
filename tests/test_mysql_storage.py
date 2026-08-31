# -*- coding: utf-8 -*-
"""MySQL 存储层测试。

非破坏性：在独立的 `<MYSQL_DB>_test` 库上建表/读写/清理，不触碰业务库。
无法连接 MySQL 时整体 skip，便于本地无库环境跑其余测试。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql

from src.common import db, kv_store, doc_store, repositories as repo


def _can_connect():
    cfg = {
        'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
        'port': int(os.getenv('MYSQL_PORT', '3306')),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'connect_timeout': 2,
    }
    try:
        conn = pymysql.connect(**cfg)
        conn.close()
        return True
    except Exception:
        return False


@unittest.skipUnless(_can_connect(), "无可用 MySQL，跳过存储层测试")
class MySQLStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_db = os.environ.get('MYSQL_DB', 'football')
        test_db = cls._orig_db + '_test'
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST', '127.0.0.1'),
            port=int(os.getenv('MYSQL_PORT', '3306')),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
        )
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {test_db} CHARACTER SET utf8mb4")
        conn.close()
        os.environ['MYSQL_DB'] = test_db
        cls._test_db = test_db
        db._local.conn = None
        db.init_db()

    @classmethod
    def tearDownClass(cls):
        try:
            db.execute(f"DROP DATABASE IF EXISTS {cls._test_db}")
        finally:
            os.environ['MYSQL_DB'] = cls._orig_db
            db._local.conn = None

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
