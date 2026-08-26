"""盘口与走势解读的回归判据。

原本是 `tests/test_basketball_market_analysis.py`，直接测 `src/basketball`。
这些用例编码的是领域约定，不该随旧模块一起消失：让分用「主队加分值」、
大小分升盘指向大分、水位与盘口冲突时禁止反推、盘口没动时只推进
observed_ts 而不改变化时刻。
"""
import unittest
from datetime import datetime

from src.domain.sports.basketball.movement import (
    apply_market_inference, describe_market_movement,
    infer_market_from_movement, movement_from_snapshots, normalize_okooo_trend,
)
from src.domain.sports.basketball.odds_history import (
    OddsHistoryStore, OddsTracker,
)
from src.domain.sports.basketball.repository import create_all
from src.foundation.store import Database, make_engine

NOW = datetime(2026, 8, 20, 12, 0, 0)


class TrendNormalisationTests(unittest.TestCase):
    def test_total_market_uses_over_under_sides_and_keeps_line_move(self):
        movement = normalize_okooo_trend({
            'direction': 'over_backing', 'strength': .12,
            'home_move': -.08, 'away_move': .06, 'line_move': 2.5,
            'samples': 5,
        }, 'ou')
        self.assertEqual(movement['side'], 'over')
        self.assertEqual(movement['line_move'], 2.5)


class JointAnalysisTests(unittest.TestCase):
    def test_reports_confirmation(self):
        result = describe_market_movement(
            {'rqspf': {'available': True, 'side': 'home', 'strength': .7,
                       'steam': True, 'stale': False, 'samples': 6,
                       'line_move': -1.5},
             'dx': {'available': True, 'side': 'under', 'strength': .4,
                    'steam': False, 'stale': False, 'samples': 4,
                    'line_move': -2}},
            {'rqspf': {'line_movement': {'confirmed': True}},
             'dx': {'line_movement': {'confirmed': True}}})
        self.assertTrue(result['available'])
        self.assertEqual(result['level'], 'strong')
        self.assertEqual(result['aligned_count'], 2)
        self.assertIn('盘口降1.5分', result['signals'][0]['summary'])

    def test_warns_on_model_market_conflict(self):
        result = describe_market_movement(
            {'spf': {'available': True, 'side': 'away', 'strength': .5,
                     'steam': False, 'stale': False, 'samples': 3}},
            {'spf': {'line_movement': {'confirmed': False}}})
        self.assertEqual(result['level'], 'warning')


class SnapshotMovementTests(unittest.TestCase):
    def test_spread_and_total_line_changes(self):
        """让分数值下降代表主队变强；大小分升盘指向大分。方向搞反不会报错，
        只会让推荐一直站在错的一边。"""
        snapshots = [
            {'ts': '2026-08-12T10:00:00', 'h': 1.80, 'a': 1.80,
             'handicap': -3.5, 'total': 218.5},
            {'ts': '2026-08-12T10:10:00', 'h': 1.80, 'a': 1.80,
             'handicap': -5.5, 'total': 221.5},
        ]
        spread = movement_from_snapshots(snapshots, 'h', 'a', 'handicap', 'ah',
                                         now_fn=lambda: NOW)
        total = movement_from_snapshots(snapshots, 'h', 'a', 'total', 'ou',
                                        now_fn=lambda: NOW)
        self.assertEqual(spread['line_move'], -2.0)
        self.assertEqual(spread['side'], 'home')
        self.assertEqual(total['line_move'], 3.0)
        self.assertEqual(total['side'], 'over')

    def test_water_and_deeper_home_line_form_joint_signal(self):
        snapshots = [
            {'ts': '2026-08-20T10:00:00', 'h': 1.92, 'a': 1.72,
             'handicap': -3.5},
            {'ts': '2026-08-20T10:20:00', 'h': 1.70, 'a': 1.95,
             'handicap': -5.5},
        ]
        movement = movement_from_snapshots(snapshots, 'h', 'a', 'handicap', 'ah',
                                           now_fn=lambda: NOW)
        self.assertEqual(movement['water_side'], 'home')
        self.assertEqual(movement['line_side'], 'home')
        self.assertTrue(movement['signal_agreement'])
        self.assertFalse(movement['signal_conflict'])


class InferenceGuardTests(unittest.TestCase):
    def test_conflicting_water_and_line_cannot_drive_inference(self):
        inference = infer_market_from_movement({
            'available': True, 'side': 'home', 'strength': .9, 'samples': 5,
            'stale': False, 'steam': True, 'water_side': 'home',
            'line_side': 'away', 'signal_conflict': True}, 'rqspf')
        self.assertFalse(inference['actionable'])
        self.assertEqual(inference['reason'], 'water_line_conflict')

    def test_strong_water_signal_can_reverse_a_weak_model(self):
        over, under, inference = apply_market_inference(.52, .48, {
            'available': True, 'side': 'under', 'strength': .8, 'samples': 5,
            'stale': False, 'steam': False, 'water_side': 'under',
            'line_side': 'under', 'signal_agreement': True,
            'signal_conflict': False}, 'dx')
        self.assertGreater(under, over)
        self.assertTrue(inference['reversed_model'])


class UnchangedPollTests(unittest.TestCase):
    def test_unchanged_poll_does_not_refresh_last_move_timestamp(self):
        """盘口没动时只推进 observed_ts。改写 ts 会把一个几小时没动过的旧
        信号伪装成刚刚发生的变化，走势的新鲜度判断随之全线失真。"""
        odds = {'rqspf_home': 1.80, 'rqspf_away': 1.80, 'handicap': -3.5,
                'dx_over': 1.80, 'dx_under': 1.80, 'total_line': 218.5,
                'spf_home': 1.5, 'spf_away': 2.5}
        old_ts = '2026-08-20T08:00:00'

        db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(db)
        store = OddsHistoryStore(db)
        store.save({'m1': [{'ts': old_ts, **odds}]})

        OddsTracker(schedule_fetcher=lambda date=None: [{'id': 'm1', **odds}],
                    store=store, now_fn=lambda: NOW).track('2026-08-20')

        snapshot = store.load()['m1'][0]
        self.assertEqual(snapshot['ts'], old_ts)
        self.assertEqual(snapshot['observed_ts'], NOW.isoformat())


if __name__ == '__main__':
    unittest.main()
