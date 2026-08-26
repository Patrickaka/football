"""EloStore 的往返测试。

接口形状刻意与迁移前的 kv_store 调用一致（load 返回含三部分的 dict、
save 接收三部分），这样 BasketballELORatingSystem 只需替换 _load/_save
两个方法，其余 500 行算法逻辑逐字不动。
"""
import unittest

from src.domain.sports.basketball.elo_store import EloStore
from src.domain.sports.basketball.repository import create_all
from src.foundation.store import Database, make_engine


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = EloStore(self.db)


class EmptyStoreTests(_Base):
    def test_load_from_empty_store_returns_empty_parts(self):
        data = self.store.load()
        self.assertEqual(data['ratings'], {})
        self.assertEqual(data['history'], {})
        self.assertEqual(data['recent_form'], {})


class RoundTripTests(_Base):
    RATINGS = {'火花': 1523.5, '梦想': 1476.5}
    HISTORY = {
        '火花': [
            {'rating': 1500.0, 'date': '2026-07-13T11:49:48', 'event': 'initialized'},
            {'rating': 1523.5, 'date': '2026-07-20T10:00:00', 'event': 'match'},
        ],
        '梦想': [
            {'rating': 1500.0, 'date': '2026-07-13T11:49:48', 'event': 'initialized'},
        ],
    }
    RECENT_FORM = {'火花': [1.0, 1.0, 0.0], '梦想': [0.0]}

    def test_saved_ratings_come_back_unchanged(self):
        self.store.save(self.RATINGS, self.HISTORY, self.RECENT_FORM)
        self.assertEqual(self.store.load()['ratings'], self.RATINGS)

    def test_saved_history_keeps_order_and_fields(self):
        self.store.save(self.RATINGS, self.HISTORY, self.RECENT_FORM)
        history = self.store.load()['history']
        self.assertEqual([h['event'] for h in history['火花']],
                         ['initialized', 'match'])
        self.assertAlmostEqual(history['火花'][1]['rating'], 1523.5)
        self.assertEqual(history['火花'][1]['date'], '2026-07-20T10:00:00')

    def test_saved_recent_form_keeps_order(self):
        self.store.save(self.RATINGS, self.HISTORY, self.RECENT_FORM)
        self.assertEqual(self.store.load()['recent_form'], self.RECENT_FORM)

    def test_second_save_replaces_rather_than_accumulates(self):
        """整体保存语义：第二次 save 的内容应完全取代第一次，不是叠加。"""
        self.store.save(self.RATINGS, self.HISTORY, self.RECENT_FORM)
        self.store.save({'火花': 1600.0}, {'火花': []}, {'火花': [1.0]})
        data = self.store.load()
        self.assertEqual(data['ratings'], {'火花': 1600.0})
        self.assertEqual(data['recent_form'], {'火花': [1.0]})

    def test_shrinking_recent_form_drops_stale_entries(self):
        """近 N 场变短时旧条目必须消失，否则会读出比实际更长的历史。"""
        self.store.save(self.RATINGS, self.HISTORY, {'火花': [1.0, 1.0, 0.0, 1.0, 1.0]})
        self.store.save(self.RATINGS, self.HISTORY, {'火花': [0.0]})
        self.assertEqual(self.store.load()['recent_form']['火花'], [0.0])

    def test_team_removed_from_ratings_disappears(self):
        self.store.save(self.RATINGS, self.HISTORY, self.RECENT_FORM)
        self.store.save({'火花': 1523.5}, {}, {})
        self.assertNotIn('梦想', self.store.load()['ratings'])


class UpdatedAtTests(_Base):
    def test_save_stamps_updated_at(self):
        self.store.save({'火花': 1500.0}, {}, {}, updated_at='2026-08-26T12:00:00')
        self.assertEqual(self.store.updated_at(), '2026-08-26T12:00:00')

    def test_updated_at_is_empty_before_any_save(self):
        self.assertEqual(self.store.updated_at(), '')


if __name__ == '__main__':
    unittest.main()
