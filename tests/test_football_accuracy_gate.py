import unittest

from src.football.accuracy_gate import (
    build_accuracy_gate,
    build_total_goals_gate,
    prediction_reliability,
)


class FootballAccuracyGateTests(unittest.TestCase):
    def test_spf_selects_only_strong_market_aligned_pick(self):
        result = build_accuracy_gate({
            "standard": {
                "probabilities": {"胜": 0.84, "平": 0.09, "负": 0.07},
                "market_probabilities": {"胜": 0.82, "平": 0.10, "负": 0.08},
            }
        })
        self.assertTrue(result["spf"]["selected"])
        self.assertEqual(result["spf"]["decision"], "胜")

    def test_prediction_reliability_combines_probability_and_information(self):
        reliability = prediction_reliability(0.72, 0.88)
        self.assertAlmostEqual(reliability, 0.6736, places=4)
        self.assertLess(reliability, 0.80)

    def test_validated_market_threshold_is_not_blocked_by_impossible_reliability_gate(self):
        result = build_accuracy_gate({
            "standard": {
                "probabilities": {"胜": 0.72, "平": 0.19, "负": 0.09},
                "market_probabilities": {"胜": 0.70, "平": 0.20, "负": 0.10},
            },
        }, confidence={"score": 0.88})
        self.assertTrue(result["spf"]["selected"])
        self.assertAlmostEqual(result["spf"]["information_completeness"], 0.88)
        self.assertAlmostEqual(result["spf"]["prediction_reliability"], 0.6736)
        self.assertNotIn("预测可信度低于60%", result["spf"]["reasons"])

    def test_spf_abstains_below_threshold(self):
        result = build_accuracy_gate({
            "standard": {
                "probabilities": {"胜": 0.62, "平": 0.22, "负": 0.16},
                "market_probabilities": {"胜": 0.60, "平": 0.23, "负": 0.17},
            }
        })
        self.assertFalse(result["spf"]["selected"])
        self.assertEqual(result["spf"]["decision"], "观望")

    def test_spf_uses_official_market_probability_for_validated_threshold(self):
        result = build_accuracy_gate({
            "standard": {
                "probabilities": {"胜": 0.76, "平": 0.14, "负": 0.10},
                "market_probabilities": {"胜": 0.62, "平": 0.23, "负": 0.15},
            }
        })
        self.assertFalse(result["spf"]["selected"])
        self.assertIn("官方赔率去水概率低于70%", result["spf"]["reasons"])

    def test_sp1_uses_dual_season_validated_lower_threshold(self):
        lottery = {"standard": {
            "probabilities": {"胜": 0.72, "平": 0.17, "负": 0.11},
            "market_probabilities": {"胜": 0.66, "平": 0.20, "负": 0.14},
        }}
        spain = build_accuracy_gate(lottery, league="西甲")
        global_gate = build_accuracy_gate(lottery)

        self.assertTrue(spain["spf"]["selected"])
        self.assertEqual(spain["spf"]["minimum_probability"], 0.65)
        self.assertEqual(spain["spf"]["threshold_scope"], "SP1")
        self.assertFalse(global_gate["spf"]["selected"])

    def test_volatile_league_does_not_receive_unvalidated_override(self):
        lottery = {"standard": {
            "probabilities": {"胜": 0.73, "平": 0.16, "负": 0.11},
            "market_probabilities": {"胜": 0.68, "平": 0.19, "负": 0.13},
        }}
        result = build_accuracy_gate(lottery, league="意甲")

        self.assertFalse(result["spf"]["selected"])
        self.assertEqual(result["spf"]["threshold_scope"], "global")

    def test_rqspf_requires_stricter_probability_and_market_agreement(self):
        result = build_accuracy_gate({
            "handicap": {
                "probabilities": {"让胜": 0.83, "让平": 0.09, "让负": 0.08},
                "market_probabilities": {"让胜": 0.81, "让平": 0.10, "让负": 0.09},
            }
        })
        self.assertTrue(result["rqspf"]["selected"])
        self.assertEqual(
            result["rqspf"]["validation_status"],
            "pending_independent_rqspf_validation",
        )

    def test_conflict_forces_abstention(self):
        result = build_accuracy_gate(
            {
                "standard": {
                    "probabilities": {"胜": 0.82, "平": 0.10, "负": 0.08},
                    "market_probabilities": {"胜": 0.80, "平": 0.11, "负": 0.09},
                }
            },
            anomaly={"euro_asian_deviation": {"abs_deviation": 0.6}},
        )
        self.assertFalse(result["spf"]["selected"])
        self.assertIn("欧赔与亚盘明显冲突", result["spf"]["reasons"])

    def test_upset_alert_downgrades_chalk_and_exposes_defensive_watch(self):
        result = build_accuracy_gate(
            {
                "standard": {
                    "probabilities": {"胜": .76, "平": .14, "负": .10},
                    "market_probabilities": {"胜": .72, "平": .17, "负": .11},
                },
                "handicap": {},
            },
            confidence={"score": 1.0},
            upset={
                "alert": True,
                "level": "high",
                "risk_score": .72,
                "recommended_cover": "平/负",
                "signals": ["热门降盘", "热门升水"],
                "defensive_selections": [{"result": "平"}, {"result": "负"}],
            },
        )

        self.assertFalse(result["spf"]["selected"])
        self.assertIn("爆冷信号触发", result["spf"]["reasons"][-1])
        self.assertTrue(result["upset"]["watch"])
        self.assertEqual(result["upset"]["candidate"], "平/负")

    def test_d1_total_goals_gate_uses_frozen_high_precision_threshold(self):
        result = build_total_goals_gate({
            "close_line": 2.5,
            "close_prob": {"over": .66, "under": .34},
        }, league="德甲")

        self.assertTrue(result["selected"])
        self.assertEqual(result["decision"], "over")
        self.assertEqual(result["minimum_probability"], .65)
        self.assertAlmostEqual(result["validation"]["holdout"]["accuracy"], .7927)

    def test_total_goals_gate_rejects_unvalidated_league(self):
        result = build_total_goals_gate({
            "close_line": 2.5,
            "close_prob": {"over": .70, "under": .30},
        }, league="意甲")

        self.assertFalse(result["selected"])
        self.assertIn("该联赛大小球规则未通过冻结跨赛季验证", result["reasons"])

    def test_total_goals_gate_does_not_extrapolate_beyond_two_point_five(self):
        result = build_total_goals_gate({
            "close_line": 2.75,
            "close_prob": {"over": .70, "under": .30},
        }, league="德甲")

        self.assertFalse(result["selected"])
        self.assertIn("当前验证仅覆盖2.5球盘口", result["reasons"])


if __name__ == "__main__":
    unittest.main()
