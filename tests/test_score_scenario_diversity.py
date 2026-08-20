import unittest
from unittest.mock import patch

import src.football as football
from src.football import assess_football_upset
from src.football.prediction_policy import blend_score_matrices, select_diverse_score_scenarios
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
        self.assertIn('fair-price-joint-matrix', football.FOOTBALL_PREDICTION_LOGIC_VERSION)

    def test_score_matrix_ensemble_is_normalized(self):
        blended = blend_score_matrices(
            {(1, 0): .7, (1, 1): .3},
            {(1, 0): .4, (1, 1): .4, (0, 1): .2},
            .5,
        )
        self.assertAlmostEqual(sum(blended.values()), 1.0)
        self.assertAlmostEqual(blended[(1, 0)], .55)

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

    def test_market_timeline_keeps_changed_timestamped_snapshots(self):
        history = PredictionHistory.__new__(PredictionHistory)
        history.records = []
        scores = {"1-0": .2}
        result = {"H": .5, "D": .3, "A": .2}
        with patch.object(history, "_save_record"):
            history.add_prediction(
                "timeline-1", "Test", "A", "B", "2099-07-25 20:00",
                scores, result, odds_data={"euro": {"close": {"home": 2.0}}},
            )
            history.add_prediction(
                "timeline-1", "Test", "A", "B", "2099-07-25 20:00",
                scores, result, odds_data={"euro": {"close": {"home": 1.9}}},
            )
            history.add_prediction(
                "timeline-1", "Test", "A", "B", "2099-07-25 20:00",
                scores, result, odds_data={"euro": {"close": {"home": 1.9}}},
            )

        timeline = history.records[0]["market_timeline"]
        self.assertEqual(len(timeline), 2)
        self.assertTrue(all(item["is_prematch"] for item in timeline))
        self.assertEqual(timeline[0]["layer"], "T-24h")
        self.assertNotEqual(timeline[0]["signature"], timeline[1]["signature"])

    def test_professional_gate_snapshot_is_persisted_with_prediction(self):
        history = PredictionHistory.__new__(PredictionHistory)
        history.records = []
        snapshot = {
            "decision_gate": {"mode": "research_only", "official_bet_allowed": False},
            "validation": {"production_ready": False},
        }
        with patch.object(history, "_save_record") as save_record:
            history.add_prediction(
                "audit-1", "Test", "A", "B", "2099-07-25 20:00",
                {"1-0": .2}, {"H": .5, "D": .3, "A": .2},
                professional_snapshot=snapshot,
            )

        saved = save_record.call_args.args[0]
        self.assertEqual(saved["professional_snapshot"]["decision_gate"]["mode"], "research_only")
        self.assertEqual(
            saved["market_timeline"][0]["professional_snapshot"]["validation"]["production_ready"],
            False,
        )


if __name__ == "__main__":
    unittest.main()
