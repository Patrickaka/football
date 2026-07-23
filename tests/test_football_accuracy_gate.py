import unittest

from src.football.accuracy_gate import build_accuracy_gate, prediction_reliability


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

    def test_information_complete_but_reliability_below_80_abstains(self):
        result = build_accuracy_gate({
            "standard": {
                "probabilities": {"胜": 0.72, "平": 0.19, "负": 0.09},
                "market_probabilities": {"胜": 0.70, "平": 0.20, "负": 0.10},
            },
        }, confidence={"score": 0.88})
        self.assertFalse(result["spf"]["selected"])
        self.assertAlmostEqual(result["spf"]["information_completeness"], 0.88)
        self.assertAlmostEqual(result["spf"]["prediction_reliability"], 0.6736)
        self.assertIn("预测可信度低于80%", result["spf"]["reasons"])

    def test_spf_abstains_below_threshold(self):
        result = build_accuracy_gate({
            "standard": {
                "probabilities": {"胜": 0.62, "平": 0.22, "负": 0.16},
                "market_probabilities": {"胜": 0.60, "平": 0.23, "负": 0.17},
            }
        })
        self.assertFalse(result["spf"]["selected"])
        self.assertEqual(result["spf"]["decision"], "观望")

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


if __name__ == "__main__":
    unittest.main()
