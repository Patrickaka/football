import unittest
from unittest.mock import patch

import src.football as football
from src.football import pipeline
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

    def test_away_favorite_weakening_generates_auditable_cold_cover(self):
        candidates = [((0, 1), .18), ((1, 1), .15), ((1, 0), .10)]
        result = assess_football_upset(
            {
                'favor': 'away', 'open_handicap': -.75, 'handicap': -.5,
                'open_water': {'away': .86}, 'close_water': {'away': 1.01},
            },
            {
                'open': {'home': .22, 'draw': .16, 'away': .62},
                'close': {'home': .29, 'draw': .17, 'away': .54},
                'kelly': {'hardest': 'away', 'favored': 'home'},
            },
            {}, candidates,
            total={'open_line': 3.0, 'close_line': 2.5},
            anomaly={'euro_asian_deviation': {'abs_deviation': .6}},
        )

        self.assertTrue(result['alert'])
        self.assertEqual(result['favorite'], '负')
        self.assertEqual(result['recommended_cover'], '胜/平')
        self.assertIn('欧赔与亚盘明显背离', result['signals'])
        self.assertIn('热门方向升水+0.15', result['signals'])

    def test_prediction_logic_version_invalidates_old_cache(self):
        """版本对不上的缓存必须判为过期。

        **原先这里断言的是版本字符串里含 `fair-price-joint-matrix`**，那测的
        不是它想保护的行为：缓存失效靠的是相等性比较，版本改成任何别的值都
        照样失效；而版本一演进（现在是 `2026-08-21-...-v34`）这条就红了——
        保护没落到实处，噪声倒是留下了。改成直接测那个判断本身。
        """
        current = football.FOOTBALL_PREDICTION_LOGIC_VERSION
        self.assertTrue(pipeline._is_prediction_cache_current(
            {'model': {'prediction_logic_version': current}}))
        self.assertFalse(pipeline._is_prediction_cache_current(
            {'model': {'prediction_logic_version': 'some-older-version'}}))

    def test_prediction_cache_without_version_is_stale(self):
        """连版本都没有的旧缓存同样算过期——**默认必须是「不新鲜」**：
        认下一份来路不明的缓存，比重算一次贵得多。"""
        self.assertFalse(pipeline._is_prediction_cache_current({}))
        self.assertFalse(pipeline._is_prediction_cache_current(
            {'model': {}, 'model_status': {}}))
        self.assertFalse(pipeline._is_prediction_cache_current('not-a-dict'))

    def test_prediction_version_falls_back_to_model_status(self):
        """版本写在 `model` 或 `model_status` 里都认——历史记录两种都有。"""
        current = football.FOOTBALL_PREDICTION_LOGIC_VERSION
        self.assertTrue(pipeline._is_prediction_cache_current(
            {'model_status': {'prediction_logic_version': current}}))

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

    def test_database_decision_snapshot_uses_league_accuracy_gate(self):
        history = PredictionHistory.__new__(PredictionHistory)
        history.records = []
        professional = {"accuracy_gate": {"spf": {
            "selected": False,
            "candidate": "胜",
            "probability": .72,
            "market_probability": .68,
            "margin": .40,
            "market_margin": .30,
            "minimum_probability": .70,
            "threshold_scope": "global",
            "validation_status": "chronological_holdout_near_target",
            "reasons": ["官方赔率去水概率低于70%"],
        }}}
        with patch.object(history, "_save_record"):
            history.add_prediction(
                "audit-gate", "意甲", "A", "B", "2099-07-25 20:00",
                {"1-0": .2}, {"H": .72, "D": .17, "A": .11},
                professional_snapshot=professional,
            )

        decision = history.records[0]["decision_snapshot"]
        self.assertFalse(decision["eligible"])
        self.assertEqual(decision["prediction"], "H")
        self.assertEqual(decision["policy_version"], "league-validated-spf-v1")


if __name__ == "__main__":
    unittest.main()
