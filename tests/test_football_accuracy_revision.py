import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.football import backtest, result_sync
from src.football.ml import predict_goal_counts_from_candidates
from src.domain.sports.football.accuracy_gate import build_accuracy_gate
from src.api.services import football as service


class AccuracyRevisionTests(unittest.TestCase):
    def record(self, day):
        return {
            'match_id': str(day), 'created_at': f'2025-01-{day:02d}T10:00:00',
            'match_time': f'2025-01-{day:02d} 12:00', 'league': '西甲',
            'settled': True, 'sync_status': 'synced', 'actual_score': '1-0',
            'actual_result': 'H', 'predicted_scores': {'1-0': .7, '0-0': .2, '0-1': .1},
            'predicted_1x2': {'H': .7, 'D': .2, 'A': .1}, 'asian': .5,
            'total_line': 2.5, 'odds_snapshot': {'euro': {'close': {}}},
            'result_quality': {'grade': 'high'},
        }

    def test_rolling_uses_full_snapshots_and_latest_matches(self):
        records = [self.record(3), self.record(1), self.record(2)]
        original = copy.deepcopy(records)
        with patch.object(result_sync, 'get_history', return_value=SimpleNamespace(records=records)), \
             patch.object(result_sync, 'get_prediction_records', side_effect=AssertionError('page payload')):
            report = backtest.rolling_backtest_from_history(limit=2, windows=(2,))
            selected = backtest._settled_history_records(limit=2)
        self.assertNotIn('error', report)
        self.assertEqual(report['available_samples'], 2)
        self.assertEqual([r['match_id'] for r in selected], ['2', '3'])
        self.assertEqual(records, original)

    def test_stats_settle_frozen_direction_instead_of_later_top_pick(self):
        history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
        r = self.record(1)
        r['decision_snapshot'] = {'eligible': True, 'prediction': 'A'}
        history.records = [r]
        stats = history.get_stats()
        self.assertEqual(stats['hit_rate_1x2'], 1)
        self.assertEqual(stats['actionable_1x2']['total'], 1)
        self.assertEqual(stats['actionable_1x2']['correct'], 0)

    def test_legacy_history_uses_creation_year_and_future_results_are_excluded(self):
        old = self.record(1)
        old.update(match_time='12-31 12:00', created_at='2024-12-30T10:00:00')
        future = self.record(2)
        future['match_time'] = '2099-01-02 12:00'
        with patch.object(result_sync, 'get_history', return_value=SimpleNamespace(records=[old, future])):
            self.assertEqual(backtest._settled_history_records(), [old])

    def test_export_retains_audit_fields_and_does_not_mutate_history(self):
        history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
        r = self.record(1)
        r['decision_snapshot'] = {'eligible': True, 'prediction': 'H'}
        r['exclude_from_calibration'] = True
        history.records = [r]
        original = copy.deepcopy(r)
        with patch.object(result_sync, '_global_history', history):
            export = result_sync.get_prediction_export()
        for key in ('result_quality', 'decision_snapshot', 'exclude_from_calibration'):
            self.assertEqual(export['records'][0][key], original[key])
        self.assertEqual(export['schema_version'], 'football-prediction-export-v3')
        self.assertEqual(r, original)

    def test_total_recommendation_is_a_marginal_of_final_scores(self):
        candidates = [((0, 0), .1), ((1, 0), .2), ((0, 1), .15), ((2, 1), .55)]
        with patch('src.football.market_db.MarketScoreDB', side_effect=AssertionError('second fit')):
            result = predict_goal_counts_from_candidates(
                candidates, max_goals=3, asian={'handicap': .5},
                total={'close_line': 2.5}, use_history=False)
        self.assertEqual(result['sample_info']['blend_weight'], 0)
        self.assertAlmostEqual(result['distribution_dict'][1], .35)
        self.assertAlmostEqual(result['distribution_dict'][3], .55)
        self.assertEqual(result['recommendations'][0]['goals'], 3)
        for goals, probability in result['distribution_dict'].items():
            marginal = sum(p for h, row in enumerate(result['matrix'])
                           for a, p in enumerate(row) if h + a == goals)
            self.assertAlmostEqual(probability, marginal)

    def test_market_confidence_cannot_promote_a_weak_model(self):
        lottery = {'standard': {
            'probabilities': {'胜': .62, '平': .23, '负': .15},
            'market_probabilities': {'胜': .75, '平': .15, '负': .10},
        }}
        gate = build_accuracy_gate(lottery, confidence={'score': 1})['spf']
        self.assertFalse(gate['selected'])
        self.assertIn('模型最高概率低于65%', gate['reasons'])
        lottery['standard']['probabilities'] = {'胜': .67, '平': .20, '负': .13}
        self.assertTrue(build_accuracy_gate(lottery, confidence={'score': 1})['spf']['selected'])

    def test_diagnostics_explain_empty_samples(self):
        report = {'error': 'no settled records with actual_score', 'available_samples': 0,
                  'sample_quality': {'input_count': 3, 'kept_count': 0}}
        with patch.object(backtest, 'rolling_backtest_from_history', return_value=report), \
             patch.object(result_sync, 'audit_prediction_history', return_value={}), \
             patch.object(result_sync, 'get_sync_status_summary', return_value={}):
            result = service.football_diagnostics_payload({})['result']
        self.assertEqual(result['error'], report['error'])
        self.assertEqual(result['sample_quality'], report['sample_quality'])


if __name__ == '__main__':
    unittest.main()
