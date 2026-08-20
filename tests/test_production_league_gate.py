import unittest
from unittest.mock import patch

from src.football.accuracy_gate import build_accuracy_gate
from src.football import production_league_gate
from src.football.production_league_gate import (
    load_production_league_spf_policy,
    validate_league_spf_policy,
)


def _record(index, *, degraded_holdout=False):
    high_confidence = index % 2 == 0
    cycle = index % 20
    if high_confidence:
        hit = cycle != 18
    else:
        hit = cycle in {1, 3, 5, 7, 9, 11}
    if degraded_holdout and index >= 130 and high_confidence:
        hit = False
    probability = .75 if high_confidence else .62
    return {
        "match_id": f"m-{index:03d}",
        "match_time": f"2026-{index:03d}",
        "league": "测试杯赛",
        "settled": True,
        "actual_score": "1-0" if hit else "0-1",
        "actual_result": "H" if hit else "A",
        "professional_snapshot": {
            "accuracy_gate": {
                "spf": {
                    "candidate": "胜",
                    "market_probability": probability,
                    "reasons": [],
                }
            }
        },
    }


class ProductionLeagueGateTests(unittest.TestCase):
    def test_threshold_is_selected_then_verified_chronologically(self):
        policy = validate_league_spf_policy([_record(i) for i in range(200)], "测试杯赛")

        self.assertTrue(policy["supported"])
        self.assertEqual(policy["minimum_probability"], .65)
        self.assertGreaterEqual(policy["training"]["accuracy"], .80)
        self.assertGreaterEqual(policy["holdout"]["accuracy"], .80)
        self.assertGreaterEqual(policy["holdout"]["sample_count"], 30)

    def test_later_period_degradation_keeps_league_closed(self):
        policy = validate_league_spf_policy(
            [_record(i, degraded_holdout=True) for i in range(200)],
            "测试杯赛",
        )

        self.assertFalse(policy["supported"])
        self.assertEqual(policy["reason"], "冻结阈值未通过后续时间段验证")

    def test_insufficient_database_history_fails_closed(self):
        policy = validate_league_spf_policy([_record(i) for i in range(40)], "测试杯赛")

        self.assertFalse(policy["supported"])
        self.assertIn("样本不足100场", policy["reason"])

    def test_supported_production_policy_can_extend_unknown_league(self):
        lottery = {
            "standard": {
                "probabilities": {"胜": .72, "平": .15, "负": .13},
                "market_probabilities": {"胜": .68, "平": .17, "负": .15},
            },
            "handicap": {},
        }
        unsupported = build_accuracy_gate(lottery, confidence={"score": 1.0}, league="测试杯赛")
        supported = build_accuracy_gate(
            lottery,
            confidence={"score": 1.0},
            league="测试杯赛",
            production_spf_policy={
                "supported": True,
                "league": "测试杯赛",
                "minimum_probability": .65,
                "sample_count": 200,
                "training": {"accuracy": .88, "sample_count": 65},
                "holdout": {"accuracy": .86, "sample_count": 35},
                "target_accuracy": .80,
                "selection_rule": "frozen_chronological",
            },
        )

        self.assertFalse(unsupported["spf"]["selected"])
        self.assertTrue(supported["spf"]["selected"])
        self.assertEqual(supported["spf"]["threshold_scope"], "production:测试杯赛")
        self.assertEqual(
            supported["spf"]["validation_status"],
            "production_chronological_holdout_supported",
        )

    def test_live_batch_reads_full_database_only_once_per_cache_window(self):
        records = [_record(i) for i in range(200)]
        second_league = [
            {**_record(i), "league": "第二测试联赛", "match_id": f"s-{i:03d}"}
            for i in range(200)
        ]
        production_league_gate._POLICY_CACHE = None
        with patch(
            "src.common.repositories.football_prediction_load",
            return_value=records + second_league,
        ) as load:
            first = load_production_league_spf_policy("测试杯赛")
            second = load_production_league_spf_policy("第二测试联赛")

        self.assertTrue(first["supported"])
        self.assertTrue(second["supported"])
        load.assert_called_once_with()
        production_league_gate._POLICY_CACHE = None


if __name__ == "__main__":
    unittest.main()
