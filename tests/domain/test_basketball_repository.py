import unittest

from src.domain.sports.basketball.repository import (
    CalibrationRepository, EloRepository, MatchResultRepository,
    OddsHistoryRepository, PredictionHistoryRepository,
    PredictionRecordRepository, create_all,
)
from src.foundation.store import Database, make_engine


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)


class EloRepositoryTests(_Base):
    def test_upsert_then_read(self):
        repo = EloRepository(self.db)
        repo.upsert({'team': 'Lakers', 'rating': 1520.0, 'games': 3,
                     'updated_at': '2026-08-26T10:00:00'}, key_cols=['team'])
        rows = repo.find_by(team='Lakers')
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['rating'], 1520.0)

    def test_upsert_updates_existing_team(self):
        repo = EloRepository(self.db)
        row = {'team': 'Lakers', 'rating': 1500.0, 'games': 1,
               'updated_at': '2026-08-26T10:00:00'}
        repo.upsert(row, key_cols=['team'])
        repo.upsert({**row, 'rating': 1540.0, 'games': 2}, key_cols=['team'])
        rows = repo.find_all()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['rating'], 1540.0)

    def test_unknown_column_is_rejected(self):
        repo = EloRepository(self.db)
        with self.assertRaises(ValueError):
            repo.upsert({'team': 'Lakers', 'rating': 1500.0, 'games': 1,
                         'updated_at': 'x', 'typo': 1}, key_cols=['team'])


class OddsHistoryRepositoryTests(_Base):
    def test_composite_key_allows_multiple_snapshots(self):
        repo = OddsHistoryRepository(self.db)
        for captured in ('2026-08-26T10:00:00', '2026-08-26T10:15:00'):
            repo.upsert({'match_id': 'bb-1', 'captured_at': captured,
                         'home_odds': 1.8, 'away_odds': 2.0, 'source': '500'},
                        key_cols=['match_id', 'captured_at'])
        self.assertEqual(repo.count(), 2)

    def test_same_key_overwrites(self):
        repo = OddsHistoryRepository(self.db)
        row = {'match_id': 'bb-1', 'captured_at': '2026-08-26T10:00:00',
               'home_odds': 1.8, 'away_odds': 2.0, 'source': '500'}
        repo.upsert(row, key_cols=['match_id', 'captured_at'])
        repo.upsert({**row, 'home_odds': 1.9}, key_cols=['match_id', 'captured_at'])
        self.assertEqual(repo.count(), 1)
        self.assertAlmostEqual(repo.find_all()[0]['home_odds'], 1.9)

    def test_find_by_match_returns_all_snapshots_in_order(self):
        repo = OddsHistoryRepository(self.db)
        for captured in ('2026-08-26T10:15:00', '2026-08-26T10:00:00'):
            repo.upsert({'match_id': 'bb-1', 'captured_at': captured,
                         'home_odds': 1.8, 'away_odds': 2.0, 'source': '500'},
                        key_cols=['match_id', 'captured_at'])
        rows = repo.find_by(order_by='captured_at', match_id='bb-1')
        self.assertEqual([r['captured_at'] for r in rows],
                         ['2026-08-26T10:00:00', '2026-08-26T10:15:00'])


class PredictionHistoryRepositoryTests(_Base):
    def test_payload_roundtrips_as_json_string(self):
        import json
        repo = PredictionHistoryRepository(self.db)
        payload = json.dumps({'picks': ['home'], 'prob': 0.61}, ensure_ascii=False)
        repo.upsert({'match_id': 'bb-1', 'predicted_at': '2026-08-26T10:00:00',
                     'payload': payload, 'league': 'NBA'},
                    key_cols=['match_id', 'predicted_at'])
        row = repo.find_all()[0]
        self.assertEqual(json.loads(row['payload'])['prob'], 0.61)


class AllRepositoriesTests(_Base):
    def test_every_table_is_created_and_empty(self):
        for cls in (EloRepository, CalibrationRepository, OddsHistoryRepository,
                    PredictionRecordRepository, MatchResultRepository,
                    PredictionHistoryRepository):
            self.assertEqual(cls(self.db).count(), 0, f'{cls.__name__} 建表失败')

    def test_delete_by_without_filters_is_rejected(self):
        """继承自 Repository 的护栏：零参数 delete_by 会删空整表。"""
        repo = EloRepository(self.db)
        with self.assertRaises(ValueError):
            repo.delete_by()


if __name__ == '__main__':
    unittest.main()
