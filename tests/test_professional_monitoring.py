import unittest

from src.football.professional_monitoring import (
    build_professional_monitoring,
    calibration_report,
    wilson_interval,
)


class ProfessionalMonitoringTests(unittest.TestCase):
    def test_wilson_interval_is_not_naive_point_accuracy(self):
        low, high = wilson_interval(8, 10)
        self.assertLess(low, .80)
        self.assertGreater(high, .80)

    def test_spf_and_rqspf_are_calibrated_independently(self):
        records = [{
            "settled": True,
            "actual_score": "2-1",
            "actual_result": "H",
            "lottery_handicap": -1,
            "predicted_1x2": {"H": .80, "D": .12, "A": .08},
            "predicted_rqspf": {"让胜": .20, "让平": .65, "让负": .15},
        }]
        self.assertEqual(calibration_report(records, "spf")["accuracy"], 1.0)
        self.assertEqual(calibration_report(records, "rqspf")["accuracy"], 1.0)
        self.assertEqual(calibration_report(records, "rqspf")["n"], 1)

    def test_monitoring_counts_only_real_time_layers(self):
        records = [{
            "settled": True,
            "actual_score": "1-0",
            "actual_result": "H",
            "predicted_1x2": {"H": .6, "D": .25, "A": .15},
            "odds_layers": {"final": {"euro": {}}, "T-15min": None},
        }, {
            "settled": True,
            "actual_score": "0-0",
            "actual_result": "D",
            "predicted_1x2": {"H": .3, "D": .4, "A": .3},
            "odds_layers": {"T-1h": {"euro": {}}, "T-15min": {"euro": {}}},
        }]
        report = build_professional_monitoring(records)
        self.assertEqual(report["market_timing"]["timed_snapshot_samples"], 1)
        self.assertEqual(report["market_timing"]["closing_odds_samples"], 1)
        self.assertEqual(report["spf"]["n"], 2)

    def test_drift_detects_material_logloss_deterioration(self):
        records = []
        for index in range(60):
            actual = "H" if index % 2 == 0 else "A"
            records.append({
                "settled": True, "match_id": str(index), "match_time": f"2026-01-{index + 1:03d}",
                "actual_score": "1-0" if actual == "H" else "0-1", "actual_result": actual,
                "predicted_1x2": {actual: .8, "D": .1, ("A" if actual == "H" else "H"): .1},
            })
        for index in range(50):
            actual = "H" if index % 2 == 0 else "A"
            wrong = "A" if actual == "H" else "H"
            records.append({
                "settled": True, "match_id": f"r{index}", "match_time": f"2026-12-{index + 1:03d}",
                "actual_score": "1-0" if actual == "H" else "0-1", "actual_result": actual,
                "predicted_1x2": {wrong: .8, "D": .1, actual: .1},
            })
        report = build_professional_monitoring(records, recent_window=50, baseline_window=60)
        self.assertTrue(report["drift"]["detected"])
        self.assertIn("近期LogLoss显著恶化", report["drift"]["reasons"])


if __name__ == "__main__":
    unittest.main()
