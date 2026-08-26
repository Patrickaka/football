"""表结构依据线上 kv_store 的真实数据（2026-08-26 读取），非迁移前的猜测。

样例值直接取自线上：Elo 初始分 1500、盘口快照的九个字段、
match_key 的 '日期_主队_客队' 格式。
"""
import unittest

from src.domain.sports.basketball.repository import (
    EloHistoryRepository, EloRatingRepository, EloRecentFormRepository,
    OddsSnapshotRepository, create_all,
)
from src.foundation.store import Database, make_engine


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)


class EloRatingRepositoryTests(_Base):
    def test_upsert_then_read(self):
        repo = EloRatingRepository(self.db)
        repo.upsert({'team': '火花', 'rating': 1500.0,
                     'updated_at': '2026-08-20T15:19:51.145713'}, key_cols=['team'])
        rows = repo.find_by(team='火花')
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['rating'], 1500.0)

    def test_upsert_updates_existing_team(self):
        repo = EloRatingRepository(self.db)
        row = {'team': '火花', 'rating': 1500.0, 'updated_at': '2026-08-20T15:19:51'}
        repo.upsert(row, key_cols=['team'])
        repo.upsert({**row, 'rating': 1523.5}, key_cols=['team'])
        rows = repo.find_all()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['rating'], 1523.5)

    def test_unknown_column_is_rejected(self):
        """源数据里没有 games 字段，误加会被拦下而不是静默丢弃。"""
        repo = EloRatingRepository(self.db)
        with self.assertRaises(ValueError):
            repo.upsert({'team': '火花', 'rating': 1500.0,
                         'updated_at': 'x', 'games': 3}, key_cols=['team'])


class EloHistoryRepositoryTests(_Base):
    def test_same_team_keeps_multiple_entries(self):
        repo = EloHistoryRepository(self.db)
        for at, event in (('2026-07-13T11:49:48.763564', 'initialized'),
                          ('2026-07-20T10:00:00.000000', 'match')):
            repo.upsert({'team': '火花', 'recorded_at': at, 'rating': 1500.0,
                         'event': event}, key_cols=['team', 'recorded_at'])
        self.assertEqual(repo.count(), 2)

    def test_history_is_ordered_by_time(self):
        repo = EloHistoryRepository(self.db)
        for at in ('2026-07-20T10:00:00', '2026-07-13T11:49:48'):
            repo.upsert({'team': '火花', 'recorded_at': at, 'rating': 1500.0,
                         'event': 'x'}, key_cols=['team', 'recorded_at'])
        rows = repo.find_by(order_by='recorded_at', team='火花')
        self.assertEqual([r['recorded_at'] for r in rows],
                         ['2026-07-13T11:49:48', '2026-07-20T10:00:00'])


class EloRecentFormRepositoryTests(_Base):
    """近 N 场胜负记录。源数据是无时间戳的数值列表，故用 seq 保序。"""

    def test_keeps_order_by_seq(self):
        repo = EloRecentFormRepository(self.db)
        for seq, result in enumerate([1.0, 0.0, 1.0]):
            repo.upsert({'team': '火花', 'seq': seq, 'result': result},
                        key_cols=['team', 'seq'])
        rows = repo.find_by(order_by='seq', team='火花')
        self.assertEqual([r['result'] for r in rows], [1.0, 0.0, 1.0])

    def test_teams_are_isolated(self):
        repo = EloRecentFormRepository(self.db)
        repo.upsert({'team': '火花', 'seq': 0, 'result': 1.0}, key_cols=['team', 'seq'])
        repo.upsert({'team': '梦想', 'seq': 0, 'result': 0.0}, key_cols=['team', 'seq'])
        self.assertEqual(len(repo.find_by(team='火花')), 1)
        self.assertEqual(repo.count(), 2)

    def test_replacing_a_team_form_removes_old_entries(self):
        """近 N 场是截断列表而非追加流水：新列表变短时旧条目必须消失，
        否则会残留出一条比实际更长的历史。"""
        repo = EloRecentFormRepository(self.db)
        for seq in range(5):
            repo.upsert({'team': '火花', 'seq': seq, 'result': 1.0},
                        key_cols=['team', 'seq'])
        repo.delete_by(team='火花')
        for seq, result in enumerate([0.0, 1.0]):
            repo.upsert({'team': '火花', 'seq': seq, 'result': result},
                        key_cols=['team', 'seq'])
        rows = repo.find_by(order_by='seq', team='火花')
        self.assertEqual([r['result'] for r in rows], [0.0, 1.0])


class OddsSnapshotRepositoryTests(_Base):
    SAMPLE = {
        'match_key': '2026-07-23_水星_火花',
        'captured_at': '2026-07-22T11:38:12.250497',
        'spf_home': 1.81, 'spf_away': 1.6,
        'rqspf_home': 1.7, 'rqspf_away': 1.7,
        'dx_over': 1.66, 'dx_under': 1.74,
        'handicap': '-1.5', 'total_line': 177.5,
    }

    def test_stores_all_three_market_types(self):
        repo = OddsSnapshotRepository(self.db)
        repo.upsert(self.SAMPLE, key_cols=['match_key', 'captured_at'])
        row = repo.find_all()[0]
        self.assertAlmostEqual(row['spf_home'], 1.81)
        self.assertAlmostEqual(row['rqspf_home'], 1.7)
        self.assertAlmostEqual(row['dx_over'], 1.66)

    def test_handicap_stays_a_string(self):
        """让分盘可能出现非纯数值写法，转数值会丢信息。"""
        repo = OddsSnapshotRepository(self.db)
        repo.upsert(self.SAMPLE, key_cols=['match_key', 'captured_at'])
        self.assertEqual(repo.find_all()[0]['handicap'], '-1.5')

    def test_composite_key_allows_multiple_snapshots(self):
        repo = OddsSnapshotRepository(self.db)
        for captured in ('2026-07-22T11:38:12.250497', '2026-07-22T14:39:56.285547'):
            repo.upsert({**self.SAMPLE, 'captured_at': captured},
                        key_cols=['match_key', 'captured_at'])
        self.assertEqual(repo.count(), 2)

    def test_same_key_overwrites(self):
        repo = OddsSnapshotRepository(self.db)
        repo.upsert(self.SAMPLE, key_cols=['match_key', 'captured_at'])
        repo.upsert({**self.SAMPLE, 'spf_home': 1.86},
                    key_cols=['match_key', 'captured_at'])
        self.assertEqual(repo.count(), 1)
        self.assertAlmostEqual(repo.find_all()[0]['spf_home'], 1.86)

    def test_missing_market_columns_are_allowed(self):
        """并非每次快照三类盘口都齐全，缺的列应可为空而不是报错。"""
        repo = OddsSnapshotRepository(self.db)
        repo.upsert({'match_key': 'k', 'captured_at': 't',
                     'spf_home': 1.8, 'spf_away': 2.0},
                    key_cols=['match_key', 'captured_at'])
        row = repo.find_all()[0]
        self.assertIsNone(row['dx_over'])


class AllRepositoriesTests(_Base):
    def test_every_table_is_created_and_empty(self):
        for cls in (EloRatingRepository, EloHistoryRepository,
                    EloRecentFormRepository, OddsSnapshotRepository):
            self.assertEqual(cls(self.db).count(), 0, f'{cls.__name__} 建表失败')

    def test_delete_by_without_filters_is_rejected(self):
        """继承自 Repository 的护栏：零参数 delete_by 会删空整表。"""
        repo = EloRatingRepository(self.db)
        with self.assertRaises(ValueError):
            repo.delete_by()


if __name__ == '__main__':
    unittest.main()
