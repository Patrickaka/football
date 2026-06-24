# -*- coding: utf-8 -*-
"""match_store 测试。纯函数无需 DB；往返测在 <MYSQL_DB>_test 库，无库则跳过。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import match_store as ms


class SeasonFromDateTest(unittest.TestCase):
    def test_august_is_new_season(self):
        self.assertEqual(ms.season_from_date('15/08/2025'), '2526')

    def test_spring_belongs_to_prev_year_season(self):
        self.assertEqual(ms.season_from_date('03/03/2026'), '2526')

    def test_may_is_old_season(self):
        self.assertEqual(ms.season_from_date('31/05/2025'), '2425')

    def test_july_boundary_is_old_season(self):
        self.assertEqual(ms.season_from_date('01/07/2025'), '2425')

    def test_june_boundary_is_old_season(self):
        self.assertEqual(ms.season_from_date('30/06/2025'), '2425')


class RecordToCsvRowTest(unittest.TestCase):
    def _rec(self, **over):
        rec = {
            'league_code': 'E0',
            'match_date': '15/08/2025',
            'match_time': '20:00',
            'home_team': 'Liverpool',
            'away_team': 'Bournemouth',
            'fthg': 4, 'ftag': 2, 'ftr': 'H',
            'hthg': 1, 'htag': 1, 'htr': 'D',
            'odds': '{"B365H": 1.35, "AHh": -1.5, "Avg>2.5": 1.9}',
            'stats': '{"referee": "A Taylor", "hs": 19, "as": 10}',
        }
        rec.update(over)
        return rec

    def test_core_fields_use_original_csv_keys_as_strings(self):
        row = ms.record_to_csv_row(self._rec())
        self.assertEqual(row['Div'], 'E0')
        self.assertEqual(row['Date'], '15/08/2025')
        self.assertEqual(row['Time'], '20:00')
        self.assertEqual(row['HomeTeam'], 'Liverpool')
        self.assertEqual(row['FTHG'], '4')
        self.assertEqual(row['FTAG'], '2')
        self.assertEqual(row['FTR'], 'H')
        self.assertTrue(row['FTHG'].isdigit())

    def test_odds_keys_preserved_values_stringified(self):
        row = ms.record_to_csv_row(self._rec())
        self.assertEqual(row['B365H'], '1.35')
        self.assertEqual(row['AHh'], '-1.5')
        self.assertEqual(row['Avg>2.5'], '1.9')

    def test_stats_keys_uppercased_referee_capitalized(self):
        row = ms.record_to_csv_row(self._rec())
        self.assertEqual(row['HS'], '19')
        self.assertEqual(row['AS'], '10')
        self.assertEqual(row['Referee'], 'A Taylor')

    def test_missing_values_become_empty_string(self):
        row = ms.record_to_csv_row(self._rec(match_time=None, hthg=None, htr=None))
        self.assertEqual(row['Time'], '')
        self.assertEqual(row['HTHG'], '')
        self.assertEqual(row['HTR'], '')


class BuildMatchRowTest(unittest.TestCase):
    def _csv_row(self, **over):
        row = {
            'Div': 'E0', 'Date': '15/08/2025', 'Time': '20:00',
            'HomeTeam': 'Liverpool', 'AwayTeam': 'Bournemouth',
            'FTHG': '4', 'FTAG': '2', 'FTR': 'H',
            'HTHG': '1', 'HTAG': '1', 'HTR': 'D',
            'B365H': '1.35', 'AHh': '-1.5', 'HS': '19', 'Referee': 'A Taylor',
        }
        row.update(over)
        return row

    def test_returns_tuple_aligned_with_cols(self):
        built = ms.build_match_row(self._csv_row(), 'E0', '2026-06-24T00:00:00')
        self.assertIsNotNone(built)
        self.assertEqual(len(built), len(ms.MATCHES_COLS))
        d = dict(zip(ms.MATCHES_COLS, built))
        self.assertEqual(d['match_id'], 'E0_2025-08-15_Liverpool_Bournemouth')
        self.assertEqual(d['match_date'], '2025-08-15')
        self.assertEqual(d['settled'], 1)

    def test_missing_date_returns_none(self):
        self.assertIsNone(ms.build_match_row(self._csv_row(Date=''), 'E0', 'now'))


import pymysql
from src.common import db


def _can_connect():
    try:
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST', '127.0.0.1'),
            port=int(os.getenv('MYSQL_PORT', '3306')),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


@unittest.skipUnless(_can_connect(), "无可用 MySQL，跳过 match_store DB 测试")
class MatchStoreDbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_db = os.environ.get('MYSQL_DB', 'football')
        os.environ['MYSQL_DB'] = cls._orig_db + '_test'
        import importlib
        importlib.reload(db)
        from src.common import match_store
        importlib.reload(match_store)
        cls.ms = match_store
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST', '127.0.0.1'),
            port=int(os.getenv('MYSQL_PORT', '3306')),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
        )
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {cls._orig_db}_test "
                        "DEFAULT CHARSET=utf8mb4")
        conn.close()
        db.init_db()
        db.execute("DELETE FROM matches")

    @classmethod
    def tearDownClass(cls):
        db.execute("DELETE FROM matches")
        os.environ['MYSQL_DB'] = cls._orig_db
        import importlib
        importlib.reload(db)

    def test_upsert_then_iter_roundtrip(self):
        row = {
            'Div': 'E0', 'Date': '15/08/2025', 'Time': '20:00',
            'HomeTeam': 'Liverpool', 'AwayTeam': 'Bournemouth',
            'FTHG': '4', 'FTAG': '2', 'FTR': 'H',
            'HTHG': '1', 'HTAG': '1', 'HTR': 'D',
            'B365H': '1.35', 'AHh': '-1.5', 'Avg>2.5': '1.9',
            'HS': '19', 'Referee': 'A Taylor',
        }
        built = self.ms.build_match_row(row, 'E0', '2026-06-24T00:00:00')
        self.ms.upsert_rows([built])

        rows = list(self.ms.iter_csv_rows('E0'))
        self.assertEqual(len(rows), 1)
        got = rows[0]
        self.assertEqual(got['Date'], '15/08/2025')
        self.assertEqual(got['Time'], '20:00')
        self.assertEqual(got['HomeTeam'], 'Liverpool')
        self.assertEqual(got['FTHG'], '4')
        self.assertEqual(got['B365H'], '1.35')
        self.assertEqual(got['AHh'], '-1.5')
        self.assertEqual(got['HS'], '19')
        self.assertEqual(got['Referee'], 'A Taylor')

    def test_upsert_is_idempotent(self):
        row = {
            'Div': 'D1', 'Date': '20/09/2025',
            'HomeTeam': 'Bayern', 'AwayTeam': 'Mainz',
            'FTHG': '3', 'FTAG': '0', 'B365H': '1.2',
        }
        built = self.ms.build_match_row(row, 'D1', 'now')
        self.ms.upsert_rows([built])
        self.ms.upsert_rows([built])
        rows = list(self.ms.iter_csv_rows('D1'))
        self.assertEqual(len(rows), 1)

    def test_iter_returns_chronological_order(self):
        """验证 iter_csv_rows 按真实日期升序而非字符串序排序。"""
        league_code = 'SP1'
        # 先插入较晚的日期，再插入较早的日期，验证返回按真实日期排序
        row1 = {
            'Div': league_code, 'Date': '15/08/2025',
            'HomeTeam': 'Real Madrid', 'AwayTeam': 'Barcelona',
            'FTHG': '3', 'FTAG': '2',
        }
        row2 = {
            'Div': league_code, 'Date': '20/09/2024',
            'HomeTeam': 'Atletico Madrid', 'AwayTeam': 'Valencia',
            'FTHG': '1', 'FTAG': '1',
        }
        built1 = self.ms.build_match_row(row1, league_code, 'now')
        built2 = self.ms.build_match_row(row2, league_code, 'now')
        self.ms.upsert_rows([built1, built2])

        rows = list(self.ms.iter_csv_rows(league_code))
        self.assertEqual(len(rows), 2)
        # 验证第一行是较早的日期（20/09/2024）
        self.assertEqual(rows[0]['Date'], '20/09/2024')
        self.assertEqual(rows[0]['HomeTeam'], 'Atletico Madrid')
        # 验证第二行是较晚的日期（15/08/2025）
        self.assertEqual(rows[1]['Date'], '15/08/2025')
        self.assertEqual(rows[1]['HomeTeam'], 'Real Madrid')


if __name__ == '__main__':
    unittest.main()
