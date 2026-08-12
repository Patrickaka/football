import unittest
from unittest.mock import patch

import src.beidan as beidan


class BeidanMarketTimelineTests(unittest.TestCase):
    def test_rqspf_requires_same_match_market_confirmation(self):
        section = {"prediction": "让胜"}
        asian = {"history": [
            {"handicap": -0.5, "home_odds": 1.0, "away_odds": .9},
            {"handicap": 0.0, "home_odds": .82, "away_odds": 1.08},
        ]}
        result = beidan.build_beidan_market_admission(section, "rqspf", asian, None)
        self.assertTrue(result["official"])
        self.assertTrue(result["aligned"])

    def test_zjq_conflicting_total_movement_is_not_official(self):
        section = {"prediction": "4"}
        goals = {"history": [
            {"line": 2.75, "over_odds": .90, "under_odds": .96},
            {"line": 2.25, "over_odds": 1.05, "under_odds": .80},
        ]}
        result = beidan.build_beidan_market_admission(section, "zjq", None, goals)
        self.assertFalse(result["official"])
        self.assertEqual(result["skip_reason"], "market_conflicts_with_model")

    def test_repeated_save_appends_only_changed_market_layers(self):
        stored = []
        match = {
            "match_id": "m1", "date": "2026-08-12", "num": "001",
            "home": "H", "away": "A", "league": "L", "time": "20:00",
            "handicap": -1,
            "asian": {"history": [{"handicap": -.5, "home_odds": .9, "away_odds": 1.0}]},
            "goals": {"history": [{"line": 2.5, "over_odds": .9, "under_odds": 1.0}]},
            "rqspf": {"prediction": "让胜", "market_admission": {"official": True}},
            "zjq": {"prediction": "3", "market_admission": {"official": True}},
        }
        payload = {"source": "okooo", "recommendations": [match]}

        def save(_key, rows):
            stored[:] = rows

        with patch('src.beidan.kv_store.load', side_effect=lambda _k, default=None: list(stored)), \
                patch('src.beidan.kv_store.save', side_effect=save):
            beidan.save_beidan_prediction_snapshot(payload)
            beidan.save_beidan_prediction_snapshot(payload)

        self.assertEqual(len(stored), 1)
        self.assertEqual(len(stored[0]["market_layers"]), 1)


if __name__ == "__main__":
    unittest.main()
