"""迁移后的 Elo 系统：算法逻辑逐字未动，只换了持久化后端。

这些测试同时是回归判据——若 _load/_save 的改写破坏了原有语义，
评分更新、冷启动门控、状态因子都会露馅。
"""
import unittest

from src.domain.sports.basketball.elo import BasketballELORatingSystem
from src.domain.sports.basketball.elo_store import EloStore
from src.domain.sports.basketball.repository import create_all
from src.foundation.store import Database, make_engine


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = EloStore(self.db)

    def _system(self):
        return BasketballELORatingSystem(store=self.store)


class PersistenceTests(_Base):
    def test_new_system_starts_empty(self):
        self.assertEqual(self._system().ratings, {})

    def test_ratings_survive_reload(self):
        first = self._system()
        first.update_ratings('火花', '梦想', 100, 90, league='NBA')
        first._save()

        second = self._system()
        self.assertAlmostEqual(second.get_rating('火花'),
                               first.get_rating('火花'), places=6)

    def test_history_survives_reload(self):
        first = self._system()
        first.update_ratings('火花', '梦想', 100, 90, league='NBA')
        first._save()

        second = self._system()
        self.assertEqual(second.games_played('火花'), first.games_played('火花'))

    def test_recent_form_survives_reload(self):
        """近 N 场直接影响预测概率，丢了不会报错但结果会悄悄变。"""
        first = self._system()
        for _ in range(3):
            first.update_ratings('火花', '梦想', 100, 90, league='NBA')
        first._save()

        second = self._system()
        self.assertEqual(second.recent_form.get('火花'),
                         first.recent_form.get('火花'))


class BehaviourTests(_Base):
    def test_winner_gains_loser_loses(self):
        system = self._system()
        before = system.get_rating('火花')
        system.update_ratings('火花', '梦想', 110, 90, league='NBA')
        self.assertGreater(system.get_rating('火花'), before)
        self.assertLess(system.get_rating('梦想'), before)

    def test_games_played_excludes_initialization(self):
        """冷启动门控依赖它：初始化事件不能被算作已打过的比赛。"""
        system = self._system()
        system.get_rating('火花')  # 触发 _init_team
        self.assertEqual(system.games_played('火花'), 0)
        system.update_ratings('火花', '梦想', 100, 90, league='NBA')
        self.assertEqual(system.games_played('火花'), 1)

    def test_win_prob_favours_higher_rated_team(self):
        system = self._system()
        for _ in range(5):
            system.update_ratings('火花', '梦想', 120, 90, league='NBA')
        result = system.predict_win_prob('火花', '梦想', league='NBA')
        self.assertGreater(result['home_prob'], 0.5)
        self.assertAlmostEqual(result['home_prob'] + result['away_prob'], 1.0, places=6)
        self.assertGreater(result['rating_diff'], 0)


class NoStoreTests(unittest.TestCase):
    def test_without_store_it_degrades_to_memory_only(self):
        """不注入 store 时退化为纯内存，不应抛异常——便于离线分析与测试。"""
        system = BasketballELORatingSystem()
        system.update_ratings('火花', '梦想', 100, 90, league='NBA')
        self.assertGreater(system.get_rating('火花'), 0)
        system._save()  # 不抛异常


if __name__ == '__main__':
    unittest.main()
