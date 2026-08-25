import unittest

from sqlalchemy import Column, Integer, MetaData, String, Table

from src.foundation.store.engine import Database, make_engine
from src.foundation.store.repository import Repository

_META = MetaData()
_DRAWS = Table(
    'draws', _META,
    Column('period', String(16), primary_key=True),
    Column('numbers', String(64), nullable=False),
    Column('score', Integer, nullable=False, default=0),
)


class DrawRepository(Repository):
    table = _DRAWS


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        _META.create_all(self.db.engine)
        self.repo = DrawRepository(self.db)

    def test_insert_many_and_count(self):
        self.repo.insert_many([
            {'period': '2026001', 'numbers': '1,2,3', 'score': 10},
            {'period': '2026002', 'numbers': '4,5,6', 'score': 20},
        ])
        self.assertEqual(self.repo.count(), 2)

    def test_insert_many_with_empty_list_is_noop(self):
        self.assertEqual(self.repo.insert_many([]), 0)
        self.assertEqual(self.repo.count(), 0)

    def test_find_all_respects_order(self):
        self.repo.insert_many([
            {'period': '2026002', 'numbers': 'b', 'score': 2},
            {'period': '2026001', 'numbers': 'a', 'score': 1},
        ])
        rows = self.repo.find_all(order_by='period')
        self.assertEqual([r['period'] for r in rows], ['2026001', '2026002'])

    def test_find_by_filters(self):
        self.repo.insert_many([
            {'period': '2026001', 'numbers': 'a', 'score': 1},
            {'period': '2026002', 'numbers': 'b', 'score': 2},
        ])
        rows = self.repo.find_by(period='2026002')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['numbers'], 'b')

    def test_upsert_inserts_when_absent(self):
        self.repo.upsert({'period': '2026001', 'numbers': 'a', 'score': 1}, key_cols=['period'])
        self.assertEqual(self.repo.count(), 1)

    def test_upsert_updates_when_present(self):
        self.repo.upsert({'period': '2026001', 'numbers': 'a', 'score': 1}, key_cols=['period'])
        self.repo.upsert({'period': '2026001', 'numbers': 'z', 'score': 9}, key_cols=['period'])
        self.assertEqual(self.repo.count(), 1)
        self.assertEqual(self.repo.find_by(period='2026001')[0]['numbers'], 'z')

    def test_delete_by(self):
        self.repo.insert_many([
            {'period': '2026001', 'numbers': 'a', 'score': 1},
            {'period': '2026002', 'numbers': 'b', 'score': 2},
        ])
        self.assertEqual(self.repo.delete_by(period='2026001'), 1)
        self.assertEqual(self.repo.count(), 1)

    def test_rows_are_plain_dicts(self):
        self.repo.insert_many([{'period': '2026001', 'numbers': 'a', 'score': 1}])
        row = self.repo.find_all()[0]
        self.assertIsInstance(row, dict)
        self.assertEqual(row['score'], 1)

    def test_delete_by_without_filters_raises(self):
        self.repo.insert_many([
            {'period': '2026001', 'numbers': 'a', 'score': 1},
            {'period': '2026002', 'numbers': 'b', 'score': 2},
        ])
        with self.assertRaises(ValueError):
            self.repo.delete_by()
        self.assertEqual(self.repo.count(), 2)

    def test_delete_all_clears_table(self):
        self.repo.insert_many([
            {'period': '2026001', 'numbers': 'a', 'score': 1},
            {'period': '2026002', 'numbers': 'b', 'score': 2},
        ])
        self.assertEqual(self.repo.delete_all(), 2)
        self.assertEqual(self.repo.count(), 0)

    def test_insert_many_rejects_unknown_column(self):
        with self.assertRaises(ValueError):
            self.repo.insert_many([
                {'period': '2026001', 'numbers': 'a', 'score': 1, 'extraTypo': 'zz'},
            ])
        self.assertEqual(self.repo.count(), 0)

    def test_upsert_rejects_unknown_column_when_absent(self):
        with self.assertRaises(ValueError):
            self.repo.upsert(
                {'period': '2026001', 'numbers': 'a', 'score': 1, 'extraTypo': 'zz'},
                key_cols=['period'],
            )
        self.assertEqual(self.repo.count(), 0)

    def test_upsert_rejects_unknown_column_when_present(self):
        self.repo.upsert({'period': '2026001', 'numbers': 'a', 'score': 1}, key_cols=['period'])
        with self.assertRaises(ValueError):
            self.repo.upsert(
                {'period': '2026001', 'numbers': 'z', 'score': 9, 'extraTypo': 'zz'},
                key_cols=['period'],
            )
        self.assertEqual(self.repo.find_by(period='2026001')[0]['numbers'], 'a')


if __name__ == '__main__':
    unittest.main()
