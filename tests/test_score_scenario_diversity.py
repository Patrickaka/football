import unittest
from unittest.mock import patch

import src.football as football
from src.football import assess_football_upset
from src.football.prediction_policy import select_diverse_score_scenarios
from src.football.result_sync import PredictionHistory


class ScoreScenarioDiversityTests(unittest.TestCase):
    def test_high_upset_risk_prefers_outsider_win_over_another_draw(self):
        candidates = [
            ((1, 1), .20), ((0, 0), .15), ((1, 0), .14),
            ((0, 1), .10), ((1, 2), .08),
        ]
        result = assess_football_upset(
            {'favor': 'home'},
            {'close': {'home': .42, 'draw': .30, 'away': .28}},
            {}, candidates,
        )
        self.assertEqual(result['level'], 'high')
        self.assertEqual(result['candidates'][0]['score'], '0-1')
        self.assertEqual(result['candidates'][0]['scenario'], 'outright_upset')
        self.assertEqual(result['draw_candidates'][0]['score'], '1-1')

    def test_medium_upset_risk_keeps_draw_cover_first(self):
        candidates = [((1, 1), .20), ((0, 1), .12), ((1, 0), .18)]
        result = assess_football_upset(
            {'favor': 'home'},
            {'close': {'home': .48, 'draw': .29, 'away': .23}},
            {}, candidates,
        )
        self.assertEqual(result['level'], 'medium')
        self.assertEqual(result['candidates'][0]['scenario'], 'draw_cover')

    def test_prediction_logic_version_invalidates_old_diversified_ranking_cache(self):
        self.assertIn('context-fusion', football.FOOTBALL_PREDICTION_LOGIC_VERSION)

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
