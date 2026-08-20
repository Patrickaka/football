import unittest
from unittest.mock import patch

from src.basketball.odds_movement import (
    apply_market_inference,
    infer_market_from_movement,
    track_basketball_odds,
    _movement_from_snapshots,
    _normalize_okooo_trend,
    describe_market_movement,
)


class BasketballMarketAnalysisTests(unittest.TestCase):
    def test_total_market_uses_over_under_sides_and_keeps_line_move(self):
        movement = _normalize_okooo_trend({
            "direction": "over_backing", "strength": .12,
            "home_move": -.08, "away_move": .06, "line_move": 2.5,
            "samples": 5,
        }, "ou")
        self.assertEqual(movement["side"], "over")
        self.assertEqual(movement["line_move"], 2.5)

    def test_joint_analysis_reports_confirmation(self):
        movements = {
            "rqspf": {"available": True, "side": "home", "strength": .7,
                       "steam": True, "stale": False, "samples": 6,
                       "line_move": -1.5},
            "dx": {"available": True, "side": "under", "strength": .4,
                   "steam": False, "stale": False, "samples": 4,
                   "line_move": -2},
        }
        bets = {
            "rqspf": {"line_movement": {"confirmed": True}},
            "dx": {"line_movement": {"confirmed": True}},
        }
        result = describe_market_movement(movements, bets)
        self.assertTrue(result["available"])
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["aligned_count"], 2)
        self.assertIn("盘口降1.5分", result["signals"][0]["summary"])

    def test_joint_analysis_warns_on_model_market_conflict(self):
        movements = {
            "spf": {"available": True, "side": "away", "strength": .5,
                    "steam": False, "stale": False, "samples": 3},
        }
        bets = {"spf": {"line_movement": {"confirmed": False}}}
        result = describe_market_movement(movements, bets)
        self.assertEqual(result["level"], "warning")

    def test_snapshot_movement_uses_spread_and_total_line_changes(self):
        snapshots = [
            {"ts": "2026-08-12T10:00:00", "h": 1.80, "a": 1.80,
             "handicap": -3.5, "total": 218.5},
            {"ts": "2026-08-12T10:10:00", "h": 1.80, "a": 1.80,
             "handicap": -5.5, "total": 221.5},
        ]
        spread = _movement_from_snapshots(
            snapshots, "h", "a", "handicap", "ah"
        )
        total = _movement_from_snapshots(
            snapshots, "h", "a", "total", "ou"
        )
        self.assertEqual(spread["line_move"], -2.0)
        self.assertEqual(spread["side"], "home")
        self.assertEqual(total["line_move"], 3.0)
        self.assertEqual(total["side"], "over")

    def test_spread_water_and_deeper_home_line_form_joint_signal(self):
        snapshots = [
            {"ts": "2026-08-20T10:00:00", "h": 1.92, "a": 1.72,
             "handicap": -3.5},
            {"ts": "2026-08-20T10:20:00", "h": 1.70, "a": 1.95,
             "handicap": -5.5},
        ]
        movement = _movement_from_snapshots(
            snapshots, "h", "a", "handicap", "ah"
        )
        self.assertEqual(movement["water_side"], "home")
        self.assertEqual(movement["line_side"], "home")
        self.assertTrue(movement["signal_agreement"])
        self.assertFalse(movement["signal_conflict"])

    def test_conflicting_water_and_line_cannot_drive_inference(self):
        movement = {
            "available": True, "side": "home", "strength": .9,
            "samples": 5, "stale": False, "steam": True,
            "water_side": "home", "line_side": "away",
            "signal_conflict": True,
        }
        inference = infer_market_from_movement(movement, "rqspf")
        self.assertFalse(inference["actionable"])
        self.assertEqual(inference["reason"], "water_line_conflict")

    def test_strong_water_signal_can_reverse_a_weak_model(self):
        movement = {
            "available": True, "side": "under", "strength": .8,
            "samples": 5, "stale": False, "steam": False,
            "water_side": "under", "line_side": "under",
            "signal_agreement": True, "signal_conflict": False,
        }
        over, under, inference = apply_market_inference(
            .52, .48, movement, "dx"
        )
        self.assertGreater(under, over)
        self.assertTrue(inference["reversed_model"])

    def test_unchanged_poll_does_not_refresh_last_move_timestamp(self):
        old_ts = "2026-08-20T08:00:00"
        history = {"m1": [{
            "ts": old_ts, "rqspf_home": 1.80, "rqspf_away": 1.80,
            "handicap": -3.5, "dx_over": 1.80, "dx_under": 1.80,
            "total_line": 218.5, "spf_home": 1.5, "spf_away": 2.5,
        }]}
        matches = [{
            "id": "m1", "rqspf_home": 1.80, "rqspf_away": 1.80,
            "handicap": -3.5, "dx_over": 1.80, "dx_under": 1.80,
            "total_line": 218.5, "spf_home": 1.5, "spf_away": 2.5,
        }]
        saved = {}
        with patch('src.basketball.fetch_basketball_schedule', return_value=matches), \
                patch('src.basketball.odds_movement.kv_store.load', return_value=history), \
                patch('src.basketball.odds_movement.kv_store.save', side_effect=lambda _k, value: saved.update(value)):
            track_basketball_odds('2026-08-20')
        self.assertEqual(saved['m1'][0]['ts'], old_ts)
        self.assertIn('observed_ts', saved['m1'][0])


if __name__ == "__main__":
    unittest.main()
