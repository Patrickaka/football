import unittest

from src.football.professional_readiness import (
    build_match_evidence_profile,
    build_system_gap_assessment,
)


class ProfessionalReadinessTests(unittest.TestCase):
    def test_match_evidence_reports_missing_live_and_market_data(self):
        profile = build_match_evidence_profile({
            "euro": {"close": {"home": .5}},
            "asian": {"handicap": .5},
            "total": {"close_line": 2.5},
            "team": {"home_recent": {"form_pts": 2}},
            "lottery": {"standard": {"probabilities": {"胜": .6, "平": .25, "负": .15}}},
            "model": {"ml": {"ml_available": False}},
        })
        self.assertGreater(profile["coverage_score"], 0)
        self.assertLess(profile["coverage_score"], .70)
        self.assertIn("未取得可核验伤停", profile["blockers"])
        self.assertIn("未取得确认首发", profile["blockers"])
        self.assertEqual(profile["model_market_agreement"], "unavailable")

    def test_report_live_context_counts_injuries_and_lineup_separately(self):
        profile = build_match_evidence_profile({
            "euro": {"close": {}},
            "asian": {"handicap": 0},
            "total": {"close_line": 2.5},
            "live_context": {
                "injuries": [{"team": "A", "player": "P"}],
                "lineup": {},
            },
        })
        checks = {item["key"]: item["available"] for item in profile["checks"]}
        self.assertTrue(checks["injuries"])
        self.assertFalse(checks["confirmed_lineup"])
        self.assertNotIn("未取得可核验伤停", profile["blockers"])
        self.assertIn("未取得确认首发", profile["blockers"])

    def test_match_evidence_quantifies_model_market_conflict(self):
        profile = build_match_evidence_profile({
            "euro": {"close": {}},
            "asian": {"handicap": 0},
            "total": {"close_line": 2.5},
            "lottery": {"standard": {
                "model_probabilities": {"胜": .70, "平": .20, "负": .10},
                "market_probabilities": {"胜": .45, "平": .30, "负": .25},
            }},
        })
        self.assertEqual(profile["model_market_agreement"], "conflict")
        self.assertIn("模型与市场概率分歧较大", profile["blockers"])

    def test_system_gaps_keep_unprofitable_model_blocked(self):
        assessment = build_system_gap_assessment({
            "model_metrics": {"logloss": 1.0},
            "market_baseline_metrics": {"logloss": .98},
            "strategy": {"roi": -.01, "mean_clv": -.005},
        })
        p0_names = [item["name"] for item in assessment["gaps"] if item["priority"] == "P0"]
        self.assertIn("模型尚未跑赢市场概率", p0_names)
        self.assertIn("样本外ROI尚未转正", p0_names)
        self.assertIn("平均CLV尚未转正", p0_names)


if __name__ == "__main__":
    unittest.main()
