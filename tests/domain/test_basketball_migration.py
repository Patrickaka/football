"""迁移测试。_FAKE_KV 的结构逐字取自线上 kv_store（2026-08-26 读取），
不是按预期编造的——照假设写迁移脚本、上线才发现结构对不上，是这类任务
最典型的翻车方式。
"""
import unittest

from scripts.migrate.basketball_kv_to_store import migrate, verify
from src.domain.sports.basketball.repository import (
    EloHistoryRepository, EloRatingRepository, OddsSnapshotRepository, create_all,
)
from src.foundation.store import Database, make_engine

_FAKE_KV = {
    'basketball_elo_ratings': {
        'ratings': {'火花': 1500, '梦想': 1500, '水星': 1523.5},
        'history': {
            '火花': [
                {'rating': 1500, 'date': '2026-07-13T11:49:48.763564',
                 'event': 'initialized'},
            ],
            '水星': [
                {'rating': 1500, 'date': '2026-07-13T11:49:48.778493',
                 'event': 'initialized'},
                {'rating': 1523.5, 'date': '2026-07-20T10:00:00.000000',
                 'event': 'match'},
            ],
        },
        'recent_form': {'火花': [], '梦想': [], '水星': []},
        'updated_at': '2026-08-20T15:19:51.145713',
    },
    'basketball_odds_history': {
        '2026-07-23_水星_火花': [
            {'ts': '2026-07-22T11:38:12.250497', 'spf_home': 1.81, 'spf_away': 1.6,
             'rqspf_home': 1.7, 'rqspf_away': 1.7, 'dx_over': 1.66, 'dx_under': 1.74,
             'handicap': '-1.5', 'total_line': 177.5},
            {'ts': '2026-07-22T14:39:56.285547', 'spf_home': 1.86, 'spf_away': 1.56,
             'rqspf_home': 1.72, 'rqspf_away': 1.68, 'dx_over': 1.7, 'dx_under': 1.7,
             'handicap': '-1.5', 'total_line': 177.5},
        ],
        '2026-07-24_天猫_太阳': [
            {'ts': '2026-07-23T09:00:00.000000', 'spf_home': 2.1, 'spf_away': 1.7,
             'rqspf_home': 1.9, 'rqspf_away': 1.9, 'dx_over': 1.8, 'dx_under': 1.95,
             'handicap': '+2.5', 'total_line': 165.0},
        ],
    },
}


def _loader(key, default=None):
    return _FAKE_KV.get(key, default)


def _empty_loader(key, default=None):
    return default


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)

    def test_migrates_elo_ratings(self):
        stats = migrate(_loader, self.db)
        self.assertEqual(stats['bb_elo_rating']['migrated'], 3)
        self.assertEqual(EloRatingRepository(self.db).count(), 3)

    def test_migrates_elo_history_flattened(self):
        """history 是 {队名: [条目]}，摊平后共 3 条（火花 1 + 水星 2）。"""
        stats = migrate(_loader, self.db)
        self.assertEqual(stats['bb_elo_history']['migrated'], 3)
        self.assertEqual(EloHistoryRepository(self.db).count(), 3)

    def test_migrates_odds_snapshots_flattened(self):
        """odds_history 是 {match_key: [快照]}，摊平后共 3 条。"""
        stats = migrate(_loader, self.db)
        self.assertEqual(stats['bb_odds_snapshot']['migrated'], 3)
        self.assertEqual(OddsSnapshotRepository(self.db).count(), 3)

    def test_rating_uses_global_updated_at(self):
        """ratings 里只有数值，时间戳取顶层的 updated_at。"""
        migrate(_loader, self.db)
        row = EloRatingRepository(self.db).find_by(team='火花')[0]
        self.assertEqual(row['updated_at'], '2026-08-20T15:19:51.145713')
        self.assertAlmostEqual(row['rating'], 1500.0)

    def test_snapshot_content_matches_source(self):
        migrate(_loader, self.db)
        rows = OddsSnapshotRepository(self.db).find_by(
            match_key='2026-07-23_水星_火花', order_by='captured_at')
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['spf_home'], 1.81)
        self.assertEqual(rows[0]['handicap'], '-1.5')
        self.assertAlmostEqual(rows[1]['spf_home'], 1.86)

    def test_history_event_is_preserved(self):
        migrate(_loader, self.db)
        rows = EloHistoryRepository(self.db).find_by(
            team='水星', order_by='recorded_at')
        self.assertEqual([r['event'] for r in rows], ['initialized', 'match'])

    def test_verify_passes_after_migration(self):
        migrate(_loader, self.db)
        self.assertEqual(verify(_loader, self.db), [])

    def test_verify_detects_missing_row(self):
        migrate(_loader, self.db)
        EloRatingRepository(self.db).delete_by(team='火花')
        problems = verify(_loader, self.db)
        self.assertTrue(problems)
        self.assertTrue(any('bb_elo_rating' in p for p in problems))

    def test_verify_detects_content_mismatch(self):
        """行数对得上但内容被改过，也必须查出来。"""
        migrate(_loader, self.db)
        EloRatingRepository(self.db).upsert(
            {'team': '火花', 'rating': 9999.0, 'updated_at': 'x'}, key_cols=['team'])
        problems = verify(_loader, self.db)
        self.assertTrue(any('火花' in p for p in problems))

    def test_empty_source_is_not_an_error(self):
        stats = migrate(_empty_loader, self.db)
        self.assertEqual(stats['bb_elo_rating']['migrated'], 0)
        self.assertEqual(EloRatingRepository(self.db).count(), 0)

    def test_dry_run_does_not_write(self):
        stats = migrate(_loader, self.db, dry_run=True)
        self.assertEqual(stats['bb_elo_rating']['migrated'], 3)
        self.assertEqual(EloRatingRepository(self.db).count(), 0)

    def test_migration_is_idempotent(self):
        migrate(_loader, self.db)
        migrate(_loader, self.db)
        self.assertEqual(EloRatingRepository(self.db).count(), 3)
        self.assertEqual(OddsSnapshotRepository(self.db).count(), 3)


if __name__ == '__main__':
    unittest.main()
