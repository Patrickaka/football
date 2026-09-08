from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from src.domain.sports.football.release_evaluation import build_release_report, paired_date_comparison
from src.domain.sports.football.acceptance import assess_model_acceptance
from src.domain.sports.football.settlement import PRODUCTION_MODEL_VERSION
from src.football.config import FOOTBALL_PREDICTION_LOGIC_VERSION
from src.football import bayes_report
from tests.test_football_version_acceptance import accepted_report, NOW


class ReleaseEvaluationTests(unittest.TestCase):
    def payload(self, count=1200):
        samples = []
        training = '2026-07-01T00:00:00+00:00'
        for index in range(count):
            kickoff = datetime(2026, 7, 20, 12, tzinfo=timezone.utc) + timedelta(days=index//30)
            samples.append({
                'match_id': str(index), 'model_version': 'test-model', 'prediction_logic_version': 'test-logic',
                'training_cutoff_at': training, 'frozen': True,
                'kickoff_at': kickoff.isoformat(),
                'predicted_at': (kickoff-timedelta(hours=1)).isoformat(),
                'market_captured_at': (kickoff-timedelta(hours=1)).isoformat(),
                'settled_at': (kickoff+timedelta(hours=2)).isoformat(),
                'probabilities': {'H': .8, 'D': .1, 'A': .1},
                'market_probabilities': {'H': .4, 'D': .3, 'A': .3}, 'actual': 'H',
            })
        return {'schema_version': 'football-frozen-prediction-events-v1',
                'evaluation_scope': 'production_pipeline',
                'model_version': 'test-model', 'prediction_logic_version': 'test-logic',
                'training_cutoff_at': training, 'samples': samples}

    def test_generator_produces_auditable_paired_report(self):
        payload = self.payload()
        original = deepcopy(payload)
        report = build_release_report(payload, now=NOW)
        self.assertEqual(payload, original)
        self.assertEqual(report['out_of_sample_n'], 1200)
        self.assertEqual(report['paired_comparison']['independent_days'], 40)
        self.assertEqual(report['paired_comparison']['accuracy_difference']['estimate'], 0)
        self.assertGreater(report['paired_comparison']['logloss_improvement']['ci95'][0], 0)
        acceptance = assess_model_acceptance(report, model_version='test-model',
                                             prediction_logic_version='test-logic', now=NOW)
        self.assertTrue(acceptance['prediction_ready'], acceptance['reasons'])

    def test_invalid_or_leaking_events_are_excluded_with_reasons(self):
        for field, value, reason in (
            ('frozen', False, 'prediction_not_frozen'),
            ('model_version', 'old', 'model_version_mismatch'),
            ('predicted_at', '2026-07-20T11:00:00', 'missing_timezone_aware_timestamp'),
            ('market_captured_at', '2026-07-20T11:05:00Z', 'market_not_available_at_prediction_time'),
            ('training_cutoff_at', '2026-08-01T00:00:00Z', 'training_cutoff_mismatch_or_leakage'),
            ('probabilities', {'H': float('nan'), 'D': .2, 'A': .3}, 'invalid_probabilities_or_result'),
        ):
            with self.subTest(field=field):
                payload = self.payload(1)
                payload['samples'][0][field] = value
                report = build_release_report(payload, now=NOW)
                self.assertEqual(report['out_of_sample_n'], 0)
                self.assertEqual(report['audit']['rejection_reasons'], {reason: 1})

    def test_duplicate_events_cannot_inflate_independent_sample_count(self):
        payload = self.payload(1)
        payload['samples'] *= 3
        report = build_release_report(payload, now=NOW)
        self.assertEqual(report['out_of_sample_n'], 0)
        self.assertFalse(report['audit']['unique_matches'])

    def test_fixed_prediction_horizon_is_enforced(self):
        payload = self.payload(1)
        payload['samples'][0]['predicted_at'] = '2026-07-20T11:30:00Z'
        report = build_release_report(payload, now=NOW)
        self.assertEqual(report['audit']['rejection_reasons'], {'outside_fixed_decision_horizon': 1})

    def test_shadow_scope_is_not_promoted_to_production(self):
        payload = self.payload(1)
        payload['evaluation_scope'] = 'catboost_shadow'
        report = build_release_report(payload, now=NOW)
        self.assertEqual(report['evaluation_scope'], 'catboost_shadow')


class ValidationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.cache = deepcopy(bayes_report._PRO_VALIDATION_CACHE)
        bayes_report._PRO_VALIDATION_CACHE.update({'value': {}, 'checked_at': 0, 'mtime': None})

    def tearDown(self):
        bayes_report._PRO_VALIDATION_CACHE.clear()
        bayes_report._PRO_VALIDATION_CACHE.update(self.cache)

    def test_cached_report_is_rechecked_for_expiry_and_current_version(self):
        report = accepted_report()
        report.update(model_version=PRODUCTION_MODEL_VERSION,
                      prediction_logic_version=FOOTBALL_PREDICTION_LOGIC_VERSION)
        report['strategy'] = {'bets': 200, 'roi': .1, 'mean_clv': .01}
        with patch.object(bayes_report.os.path, 'exists', return_value=False), \
             patch('src.common.kv_store.load_with_backend', return_value=(report, 'mysql')) as load:
            first = bayes_report.load_professional_validation_summary(now=NOW)
            self.assertTrue(first['prediction_ready'])
            self.assertTrue(first['production_ready'])
            later = bayes_report.load_professional_validation_summary(now=NOW+timedelta(days=15))
            self.assertFalse(later['prediction_ready'])
            self.assertFalse(later['production_ready'])
            with patch('src.domain.sports.football.settlement.PRODUCTION_MODEL_VERSION', 'new-release'):
                changed = bayes_report.load_professional_validation_summary(now=NOW)
            self.assertFalse(changed['prediction_ready'])
            self.assertEqual(load.call_count, 1)

    def test_attractive_legacy_baseline_cannot_certify_current_release(self):
        legacy = {'out_of_sample_n': 5000, 'model_metrics': {'logloss': .8},
                  'market_baseline_metrics': {'logloss': 1.0},
                  'strategy': {'bets': 500, 'roi': .2, 'mean_clv': .1}}
        with patch.object(bayes_report.os.path, 'exists', return_value=False), \
             patch('src.common.kv_store.load_with_backend', return_value=(legacy, 'mysql')):
            result = bayes_report.load_professional_validation_summary(now=NOW)
        self.assertTrue(result['available'])
        self.assertFalse(result['prediction_ready'])
        self.assertFalse(result['production_ready'])
        self.assertTrue(result['acceptance']['reasons'])

    def test_status_endpoint_uses_the_same_release_gate(self):
        from src.api.services import football as service
        summary = {'available': True, 'production_ready': False, 'prediction_ready': False,
                   'checks': {'model_beats_market': True, 'positive_roi': True,
                              'positive_clv': True, 'enough_samples': True,
                              'enough_strategy_bets': True, 'current_release_validated': False},
                   'model': {'logloss': .8}, 'market': {'logloss': 1.0},
                   'strategy': {'bets': 500, 'roi': .2, 'mean_clv': .1},
                   'out_of_sample_n': 5000, 'acceptance': {'reasons': ['模型版本不匹配']}}
        with patch.object(bayes_report, 'load_professional_validation_summary', return_value=summary), \
             patch('src.common.maintenance.disk_status', return_value={'under_pressure': False}), \
             patch('src.football.result_sync.get_prediction_export', return_value={'records': []}), \
             patch('src.football.professional_monitoring.build_professional_monitoring', return_value={}):
            status = service.football_professional_status_payload()['result']
        self.assertFalse(status['production_ready'])
        self.assertFalse(status['official_betting_allowed'])
        self.assertEqual(status['acceptance'], summary['acceptance'])


if __name__ == '__main__':
    unittest.main()
