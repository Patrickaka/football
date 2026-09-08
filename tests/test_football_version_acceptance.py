"""Version acceptance rejects attractive but unauditable historical results."""

from copy import deepcopy
from datetime import datetime, timezone
import unittest

from src.domain.sports.football.acceptance import assess_model_acceptance


NOW = datetime(2026, 9, 8, 8, tzinfo=timezone.utc)


def accepted_report():
    return {
        "schema_version": "football-model-acceptance-v1",
        "evaluation_scope": "production_pipeline",
        "model_version": "model-current",
        "prediction_logic_version": "logic-current",
        "generated_at": "2026-09-08T08:00:00Z",
        "training_cutoff_at": "2026-07-01T00:00:00Z",
        "evaluation_start_at": "2026-07-02T00:00:00Z",
        "evaluation_end_at": "2026-09-07T22:00:00Z",
        "data_cutoff_at": "2026-09-08T07:00:00Z",
        "out_of_sample_n": 1200,
        "model_metrics": {"n": 1200, "accuracy": .60, "brier": .54, "logloss": .91},
        "market_baseline_metrics": {"n": 1200, "accuracy": .56, "brier": .57, "logloss": .95},
        "paired_comparison": {
            "n": 1200,
            "independent_days": 60,
            "accuracy_difference": {"estimate": .04, "ci95": [.01, .07]},
            "logloss_improvement": {"estimate": .04, "ci95": [.01, .07]},
            "brier_improvement": {"estimate": .03, "ci95": [.01, .05]},
        },
        "audit": {"immutable_prematch": True, "same_snapshot_market": True, "unique_matches": True},
    }


class FootballVersionAcceptanceTests(unittest.TestCase):
    def assess(self, report, *, now=NOW):
        return assess_model_acceptance(
            report, model_version="model-current", prediction_logic_version="logic-current", now=now,
        )

    def assert_rejected(self, report, key):
        result = self.assess(report)
        self.assertFalse(result["prediction_ready"])
        self.assertFalse(result["checks"][key])
        self.assertTrue(result["reasons"])
        return result

    def test_current_production_report_with_paired_evidence_passes(self):
        report = accepted_report()
        original = deepcopy(report)
        result = self.assess(report)
        self.assertTrue(result["prediction_ready"], result["reasons"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["policy"]["readiness_scope"], "prediction_accuracy_only")
        self.assertEqual(report, original)

    def test_missing_or_malformed_report_fails_closed(self):
        for report in (None, {}, [], "old report", 42):
            with self.subTest(report=report):
                self.assert_rejected(report, "report_available")

    def test_protocol_scope_and_both_versions_are_bound(self):
        cases = (
            ("schema_version", "football-model-acceptance-v0", "schema_current"),
            ("evaluation_scope", "offline_submodel", "production_pipeline"),
            ("model_version", "previous-model", "current_model_version"),
            ("prediction_logic_version", "previous-logic", "current_prediction_logic_version"),
        )
        for field, value, key in cases:
            for candidate in (None, value):
                with self.subTest(field=field, value=candidate):
                    report = accepted_report()
                    report[field] = candidate
                    self.assert_rejected(report, key)

    def test_each_timestamp_is_required_and_timezone_aware(self):
        for field in ("generated_at", "training_cutoff_at", "evaluation_start_at", "evaluation_end_at", "data_cutoff_at"):
            for value in (None, "2026-09-08T08:00:00", "invalid", 1788854400):
                with self.subTest(field=field, value=value):
                    report = accepted_report()
                    report[field] = value
                    self.assert_rejected(report, "timestamps_present")

    def test_old_data_cannot_be_refreshed_by_new_report_timestamp(self):
        report = accepted_report()
        report["evaluation_end_at"] = "2026-08-20T00:00:00Z"
        report["data_cutoff_at"] = "2026-08-20T12:00:00Z"
        self.assert_rejected(report, "data_cutoff_recent")
        report["generated_at"] = "2026-08-20T13:00:00Z"
        self.assert_rejected(report, "generated_at_recent")

    def test_recently_settled_old_matches_do_not_renew_acceptance(self):
        report = accepted_report()
        report["evaluation_end_at"] = "2026-08-24T23:00:00Z"
        report["paired_comparison"]["independent_days"] = 30
        result = self.assert_rejected(report, "evaluation_end_recent")
        self.assertTrue(result["checks"]["data_cutoff_recent"])
        self.assertTrue(result["checks"]["generated_at_recent"])

    def test_future_timestamp_and_small_clock_skew(self):
        report = accepted_report()
        report["generated_at"] = "2026-09-08T08:05:00Z"
        self.assertTrue(self.assess(report)["prediction_ready"])
        report["generated_at"] = "2026-09-08T08:05:01Z"
        self.assert_rejected(report, "not_future")

    def test_all_time_boundaries_enforce_no_leakage(self):
        for field, value in (
            ("training_cutoff_at", "2026-07-02T00:00:00Z"),
            ("evaluation_start_at", "2026-09-08T06:00:00Z"),
            ("evaluation_end_at", "2026-09-08T07:01:00Z"),
            ("data_cutoff_at", "2026-09-08T08:01:00Z"),
        ):
            with self.subTest(field=field):
                report = accepted_report()
                report[field] = value
                self.assert_rejected(report, "time_order")

    def test_naive_now_fails_closed_and_offset_now_is_supported(self):
        self.assertFalse(self.assess(accepted_report(), now=NOW.replace(tzinfo=None))["prediction_ready"])
        self.assertTrue(self.assess(accepted_report(), now="2026-09-08T16:00:00+08:00")["prediction_ready"])

    def test_missing_pairing_and_independent_days_are_rejected(self):
        report = accepted_report()
        del report["paired_comparison"]
        self.assert_rejected(report, "paired_sample_count")
        for days in (None, 29, True, 60.5, 1201):
            report = accepted_report()
            report["paired_comparison"]["independent_days"] = days
            self.assert_rejected(report, "independent_evaluation_days")

    def test_claimed_independent_days_cannot_exceed_evaluation_period(self):
        report = accepted_report()
        report["evaluation_start_at"] = "2026-09-07T00:00:00Z"
        self.assert_rejected(report, "independent_evaluation_days")

    def test_report_and_data_age_boundary_is_inclusive(self):
        report = accepted_report()
        report["evaluation_end_at"] = "2026-08-25T08:00:00Z"
        report["data_cutoff_at"] = report["generated_at"] = "2026-08-25T08:00:00Z"
        report["paired_comparison"]["independent_days"] = 30
        self.assertTrue(self.assess(report)["prediction_ready"])
        report["evaluation_end_at"] = "2026-08-25T07:59:58Z"
        report["data_cutoff_at"] = "2026-08-25T07:59:59Z"
        self.assert_rejected(report, "data_cutoff_recent")

    def test_counts_must_be_positive_matching_integers(self):
        for n in (None, 999, 1200.0, True, float("nan")):
            report = accepted_report()
            report["out_of_sample_n"] = n
            self.assert_rejected(report, "enough_out_of_sample")
        for container in ("model_metrics", "market_baseline_metrics", "paired_comparison"):
            report = accepted_report()
            report[container]["n"] = 1201
            key = "paired_sample_count" if container == "paired_comparison" else "matched_sample_counts"
            self.assert_rejected(report, key)

    def test_invalid_metrics_and_malformed_nested_objects_fail_closed(self):
        for container, key in (("model_metrics", "valid_model_metrics"), ("market_baseline_metrics", "valid_market_metrics")):
            for metric, values in (
                ("accuracy", [float("nan"), float("inf"), -.1, 1.1, "0.6", True]),
                ("brier", [float("nan"), float("inf"), -.1, 2.1, None]),
                ("logloss", [float("nan"), float("inf"), -.1, None, 10 ** 1000]),
            ):
                for value in values:
                    with self.subTest(container=container, metric=metric, value=str(value)[:30]):
                        report = accepted_report()
                        report[container][metric] = value
                        self.assert_rejected(report, key)
            report = accepted_report()
            report[container] = ["bad"]
            self.assert_rejected(report, key)

    def test_point_estimate_improvements_without_ci_support_do_not_pass(self):
        for name in ("accuracy_difference", "logloss_improvement", "brier_improvement"):
            report = accepted_report()
            report["paired_comparison"][name]["ci95"][0] = -.001
            self.assert_rejected(report, f"supported_{name}")

    def test_logloss_requires_strict_improvement_other_lower_bounds_allow_zero(self):
        report = accepted_report()
        for name in ("accuracy_difference", "brier_improvement"):
            report["paired_comparison"][name]["ci95"][0] = 0
        self.assertTrue(self.assess(report)["prediction_ready"])
        report["paired_comparison"]["logloss_improvement"]["ci95"][0] = 0
        self.assert_rejected(report, "supported_logloss_improvement")

    def test_invalid_intervals_and_inconsistent_estimates_are_rejected(self):
        for name in ("accuracy_difference", "logloss_improvement", "brier_improvement"):
            for interval in (None, {}, {"estimate": float("nan"), "ci95": [0, .1]},
                             {"estimate": .04, "ci95": [0, float("inf")]},
                             {"estimate": .04, "ci95": [.1, 0]},
                             {"estimate": .04, "ci95": [.05, .1]},
                             {"estimate": .04, "ci95": [0, .1, .2]}):
                report = accepted_report()
                report["paired_comparison"][name] = interval
                self.assert_rejected(report, f"valid_{name}_interval")
            report = accepted_report()
            report["paired_comparison"][name]["estimate"] = .02
            self.assert_rejected(report, f"consistent_{name}_estimate")

    def test_all_accuracy_and_probability_metrics_must_not_regress(self):
        for metric, worse in (("accuracy", .55), ("logloss", .96), ("brier", .58)):
            report = accepted_report()
            report["model_metrics"][metric] = worse
            self.assert_rejected(report, f"{metric}_no_regression")

    def test_audit_flags_require_literal_true(self):
        for flag in ("immutable_prematch", "same_snapshot_market", "unique_matches"):
            for value in (False, None, 1, "true"):
                report = accepted_report()
                report["audit"][flag] = value
                self.assert_rejected(report, flag)


if __name__ == "__main__":
    unittest.main()
