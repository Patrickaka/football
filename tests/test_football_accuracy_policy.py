import unittest

from src.football.result_sync import (
    PRODUCTION_MODEL_VERSION,
    PredictionHistory,
    _prediction_decision_snapshot,
    check_ml_fusion_eligibility,
)


class FootballAccuracyPolicyTests(unittest.TestCase):
    def test_actionable_rule_is_frozen_from_prematch_probabilities(self):
        strong = _prediction_decision_snapshot({'H': 0.67, 'D': 0.20, 'A': 0.13})
        weak = _prediction_decision_snapshot({'H': 0.43, 'D': 0.32, 'A': 0.25})

        self.assertTrue(strong['eligible'])
        self.assertEqual(strong['prediction'], 'H')
        self.assertFalse(weak['eligible'])
        self.assertEqual(strong['policy_version'], 'selective-1x2-v4-accuracy-first')
        self.assertEqual(strong['min_probability'], 0.65)

    def test_stats_separate_all_predictions_from_actionable_coverage(self):
        history = PredictionHistory.__new__(PredictionHistory)
        common = {
            'settled': True,
            'actual_score': '1-0',
            'predicted_scores': {'1-0': 0.3, '1-1': 0.2},
            'time_layers': {},
        }
        history.records = [
            {
                **common,
                'actual_result': 'H',
                'predicted_1x2': {'H': 0.60, 'D': 0.25, 'A': 0.15},
                'decision_snapshot': {'eligible': True},
                'model_version': 'v-new',
            },
            {
                **common,
                'actual_result': 'A',
                'predicted_1x2': {'H': 0.42, 'D': 0.31, 'A': 0.27},
                'decision_snapshot': {'eligible': False},
            },
        ]

        stats = history.get_stats()

        self.assertEqual(stats['valid_1x2_predictions'], 2)
        self.assertEqual(stats['hit_rate_1x2'], 0.5)
        self.assertEqual(stats['actionable_1x2']['total'], 1)
        self.assertEqual(stats['actionable_1x2']['hit_rate'], 1.0)
        self.assertEqual(stats['actionable_1x2']['coverage'], 0.5)
        self.assertEqual(stats['by_model_version']['v-new']['hit_rate_1x2'], 1.0)
        self.assertEqual(stats['by_model_version']['legacy-unversioned']['hit_rate_1x2'], 0.0)

    def test_production_version_is_explicit(self):
        self.assertTrue(PRODUCTION_MODEL_VERSION.startswith('football-v'))

    def test_ml_fusion_requires_probability_improvement_not_only_samples(self):
        common = {
            'sample_count': 120,
            'base_1x2_logloss': .98,
            'base_1x2_brier': .58,
            'ml_1x2_logloss': 1.00,
            'ml_1x2_brier': .60,
            'fused_5pct_logloss': .981,
            'fused_5pct_brier': .581,
        }
        rejected = check_ml_fusion_eligibility({'overall': common}, test_set_samples=300)
        accepted = check_ml_fusion_eligibility({'overall': {
            **common,
            'fused_5pct_logloss': .979,
            'fused_5pct_brier': .579,
        }}, test_set_samples=300)

        self.assertFalse(rejected['eligible'])
        self.assertFalse(rejected['metrics_passed'])
        self.assertTrue(accepted['eligible'])
        self.assertTrue(accepted['metrics_passed'])


if __name__ == '__main__':
    unittest.main()
