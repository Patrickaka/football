import unittest
from unittest.mock import patch

import src.football as football
from src.football.prediction_policy import select_diverse_score_scenarios
from src.football.result_sync import PredictionHistory


class ScoreScenarioDiversityTests(unittest.TestCase):
    def test_prediction_logic_version_invalidates_old_diversified_ranking_cache(self):
        self.assertIn('accuracy-ranking', football.FOOTBALL_PREDICTION_LOGIC_VERSION)

    def test_primary_score_follows_aggregate_result_not_global_draw_mode(self):
        candidates = [
            ((1, 1), .16),
            ((1, 0), .14),
            ((2, 0), .13),
            ((2, 1), .12),
            ((0, 0), .10),
            ((0, 1), .08),
        ]
        selected = select_diverse_score_scenarios(candidates, limit=4)
        self.assertEqual(selected[0][0], (1, 0))
        self.assertIn(((1, 1), .16), selected)
        self.assertGreaterEqual(len({sum(score) for score, _ in selected[:3]}), 3)

    def test_goal_count_distribution_is_persisted_for_future_backtests(self):
        history = PredictionHistory.__new__(PredictionHistory)
        history.records = []
        with patch.object(history, "_save_record"):
            history.add_prediction(
                "m1", "Test", "A", "B", "07-25 20:00",
                {"1-1": .2}, {"H": .4, "D": .35, "A": .25},
                goal_count={"distribution_dict": {2: .4, 3: .3}},
            )
        self.assertEqual(
            history.records[0]["goal_count"]["distribution_dict"][2],
            .4,
        )


if __name__ == "__main__":
    unittest.main()
