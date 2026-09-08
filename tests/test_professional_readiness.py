import unittest

from src.football.professional_readiness import (
    build_match_evidence_profile,
    build_professional_decision_gate,
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
                "injuries": [{
                    "team": "A", "player": "P", "source": "official", "ts": "2026-08-12T00:00:00Z"
                }],
                "lineup": {},
            },
        })
        checks = {item["key"]: item["available"] for item in profile["checks"]}
        self.assertTrue(checks["injuries"])
        self.assertFalse(checks["confirmed_lineup"])
        self.assertNotIn("未取得可核验伤停", profile["blockers"])
        self.assertIn("未取得确认首发", profile["blockers"])

    def test_elo_estimates_do_not_count_as_event_xg_or_team_form(self):
        profile = build_match_evidence_profile({
            "team": {"elo_xg_home": 1.8, "elo_xg_away": 1.1},
            "asian": {"bookmaker_consensus": {"available": True}},
            "similar_market": {"count": 35},
        })
        checks = {item["key"]: item["available"] for item in profile["checks"]}
        self.assertFalse(checks["expected_goals"])
        self.assertFalse(checks["team_form"])
        self.assertEqual(profile["data_provenance"]["elo_expected_goals"]["kind"], "elo_estimate")
        self.assertTrue(checks["bookmaker_consensus"])
        self.assertTrue(checks["historical_analogs"])

    def test_elo_copied_into_total_remains_an_estimate(self):
        profile = build_match_evidence_profile({
            "team": {"elo_xg_home": 1.8, "elo_xg_away": 1.1},
            "total": {"xg_home": 1.8, "xg_away": 1.1},
        })
        audit = profile["data_provenance"]
        self.assertFalse(audit["expected_goals"]["verified"])
        self.assertEqual(audit["total_expected_goals"]["kind"], "elo_estimate")

    def test_unlabelled_total_xg_is_never_assumed_to_be_event_data(self):
        audit = build_match_evidence_profile({"total": {"xg_home": 1.8, "xg_away": 1.1}})["data_provenance"]
        self.assertFalse(audit["expected_goals"]["verified"])
        self.assertEqual(audit["total_expected_goals"]["kind"], "model_estimate")

    def test_sourced_timestamped_event_xg_and_history_count_separately(self):
        profile = build_match_evidence_profile({"team": {
            "home_xg_last5": 0.0, "away_xg_last5": 5.5,
            "home_recent": {"games": 5, "gf": 0, "ga": 7},
            "away_recent": {"games": 5, "gf": 6, "ga": 3},
            "data_provenance": {
                "expected_goals": {"kind": "event_xg", "source": "event-provider",
                                   "collected_at": "2026-09-08T02:00:00Z"},
                "team_form": {"kind": "historical_results", "source": "results-provider",
                              "collected_at": "2026-09-08T02:00:00Z"},
            },
        }})
        checks = {item["key"]: item["available"] for item in profile["checks"]}
        self.assertTrue(checks["expected_goals"])
        self.assertTrue(checks["team_form"])
        self.assertEqual(profile["data_provenance"]["expected_goals"]["kind"], "event_xg")

    def test_numeric_history_without_provenance_is_not_verified(self):
        audit = build_match_evidence_profile({"team": {
            "home_recent": {"games": 5, "gf": 6, "ga": 2},
            "away_recent": {"games": 5, "gf": 6, "ga": 3},
        }})["data_provenance"]["team_form"]
        self.assertEqual(audit["kind"], "historical_results")
        self.assertFalse(audit["verified"])
        self.assertIn("source_missing", audit["reasons"])

    def test_partial_invalid_and_default_xg_cannot_pass(self):
        metadata = {"kind": "event_xg", "source": "provider", "collected_at": "2026-09-08T02:00:00Z"}
        cases = [
            {"home_xg_last5": 2.0},
            {"home_xg_last5": float("nan"), "away_xg_last5": 2.0},
            {"home_xg_last5": -1, "away_xg_last5": 2.0},
            {"home_xg_last5": True, "away_xg_last5": 2.0},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                audit = build_match_evidence_profile({"team": {
                    **fields, "data_provenance": {"expected_goals": metadata},
                }})["data_provenance"]["expected_goals"]
                self.assertFalse(audit["verified"])
        audit = build_match_evidence_profile({"team": {
            "home_xg_last5": 2.0, "away_xg_last5": 2.0,
            "data_provenance": {"expected_goals": {**metadata, "is_default": True}},
        }})["data_provenance"]["expected_goals"]
        self.assertFalse(audit["verified"])
        self.assertEqual(audit["kind"], "default")

    def test_claimed_event_xg_requires_real_source_and_explicit_collection_time(self):
        for source, timestamp in [("UNAVAILABLE", "2026-09-08T02:00:00Z"),
                                  ("provider", "bad"), ("provider", "2026-09-08T02:00:00"),
                                  ("provider", None)]:
            with self.subTest(source=source, timestamp=timestamp):
                audit = build_match_evidence_profile({"team": {
                    "home_xg_last5": 6.0, "away_xg_last5": 4.0,
                    "data_provenance": {"expected_goals": {
                        "kind": "event_xg", "source": source, "collected_at": timestamp,
                    }},
                }})["data_provenance"]["expected_goals"]
                self.assertFalse(audit["verified"])

    def test_nonempty_default_strength_is_not_historical_form(self):
        audit = build_match_evidence_profile({"team": {
            "attack_home": 1.4, "defense_home": 1.4, "league_profile": {"source": "static"},
        }})["data_provenance"]["team_form"]
        self.assertFalse(audit["verified"])
        self.assertEqual(audit["kind"], "default")

    def test_zero_game_fallback_cannot_be_counted_as_observed_history(self):
        profile = build_match_evidence_profile({"team": {
            "home_recent": {"games": 1, "gf": 0, "ga": 0, "wins": 0, "draws": 0, "losses": 0},
            "away_recent": {"games": 5, "gf": 6, "ga": 3},
            "data_provenance": {"team_form": {
                "kind": "historical_results", "source": "provider", "collected_at": "2026-09-08T02:00:00Z",
            }},
        }})
        self.assertFalse(profile["data_provenance"]["team_form"]["verified"])

    def test_lineup_needs_confirmation_and_stale_live_audit_cannot_turn_green(self):
        live = {"lineup": {"source": "official", "ts": "2026-09-08T02:00:00Z"},
                "injuries": [{"source": "official", "ts": "2026-09-08T02:00:00Z"}]}
        profile = build_match_evidence_profile({"live_context": live,
            "live_context_quality": {"checks": {"injuries": "unverified", "lineup": "unverified"}}})
        checks = {item["key"]: item["available"] for item in profile["checks"]}
        self.assertFalse(checks["confirmed_lineup"])
        self.assertFalse(checks["injuries"])

    def test_unverifiable_live_context_does_not_turn_green(self):
        profile = build_match_evidence_profile({
            "live_context": {
                "injuries": [{"team": "A", "player": "P"}],
                "lineup": {"home": ["P1"]},
            },
        })
        checks = {item["key"]: item["available"] for item in profile["checks"]}
        self.assertFalse(checks["injuries"])
        self.assertFalse(checks["confirmed_lineup"])

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

    def test_system_gaps_reject_positive_roi_from_tiny_bet_sample(self):
        assessment = build_system_gap_assessment({
            "model_metrics": {"logloss": .97},
            "market_baseline_metrics": {"logloss": .98},
            "strategy": {"bets": 16, "roi": .15, "mean_clv": .01},
        })
        p0_names = [item["name"] for item in assessment["gaps"] if item["priority"] == "P0"]
        self.assertIn("独立策略下注样本不足100笔", p0_names)

    def test_professional_gate_requires_system_match_and_supported_market(self):
        gate = build_professional_decision_gate(
            {"production_ready": True},
            {"coverage_score": .82},
            {"official_bet_allowed": True},
            {"spf": {
                "selected": True,
                "validation_status": "independent_chronological_holdout_supported",
            }},
        )
        self.assertTrue(gate["official_bet_allowed"])
        self.assertEqual(gate["supported_markets"], ["spf"])

    def test_professional_gate_rejects_market_proxy_validation(self):
        gate = build_professional_decision_gate(
            {"production_ready": True},
            {"coverage_score": .82},
            {"official_bet_allowed": True},
            {"spf": {
                "selected": True,
                "validation_status": "dual_season_market_proxy_supported",
            }},
        )
        self.assertFalse(gate["official_bet_allowed"])
        self.assertFalse(gate["market_gate_passed"])

    def test_professional_gate_fails_closed_for_unprofitable_validation(self):
        gate = build_professional_decision_gate(
            {"production_ready": False},
            {"coverage_score": .90},
            {"official_bet_allowed": True},
            {"spf": {"selected": True, "validation_status": "chronological_holdout_near_target"}},
        )
        self.assertFalse(gate["official_bet_allowed"])
        self.assertEqual(gate["mode"], "research_only")
        self.assertTrue(any("样本外验证" in reason for reason in gate["reasons"]))

    def test_professional_gate_exposes_version_acceptance_failure(self):
        gate = build_professional_decision_gate({
            "production_ready": False,
            "acceptance": {"reasons": ["验证报告与当前模型版本不一致"]},
        })
        self.assertFalse(gate["official_bet_allowed"])
        self.assertIn("验证报告与当前模型版本不一致", gate["reasons"])


if __name__ == "__main__":
    unittest.main()
