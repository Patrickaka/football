import unittest

from src.basketball.odds_movement import (
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


if __name__ == "__main__":
    unittest.main()
