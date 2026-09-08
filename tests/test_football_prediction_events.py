"""No network or real persistence: prove the frozen observation/settlement boundaries."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import unittest
from unittest.mock import Mock, patch

from src.football.prediction_events import append_prediction_event
from src.domain.sports.football.prediction_evaluation import (
    evaluate_frozen_events, evaluate_frozen_ml, matrix_outcomes, score_metrics, validate_event,
)
from src.football import result_sync


UTC = timezone.utc
AS_OF = datetime(2026, 1, 1, 8, tzinfo=UTC)
MATRIX = {'0-0': .2, '0-1': .2, '1-0': .5, '1-1': .1}


def payload(as_of=AS_OF, *, model_version='production-v1', ml_version='ml-v1', agent_applied=False):
    def variant(matrix=MATRIX, applied=True):
        return {'probabilities': matrix_outcomes(matrix), 'score_probabilities': deepcopy(matrix),
                'applied': applied, 'status': 'applied' if applied else 'fallback',
                'fallback_reason': None if applied else 'insufficient_research_evidence'}
    agent = {'0-0': .1, '0-1': .2, '1-0': .6, '1-1': .1} if agent_applied else MATRIX
    return {
        'schema_version': 'football-prediction-event-v1',
        'as_of': as_of.isoformat(), 'kickoff_at': (as_of+timedelta(hours=2)).isoformat(),
        'model_version': model_version, 'prediction_logic_version': 'logic-v1',
        'training_cutoff_at': None,
        'variants': {'statistical': variant(), 'market_adjusted': variant(),
                     'agent_adjusted': variant(agent, agent_applied), 'production': variant(),
                     'market_only': {'probabilities': {'H': .4, 'D': .3, 'A': .3},
                                     'score_probabilities': None, 'captured_at': as_of.isoformat(),
                                     'status': 'applied', 'applied': True}},
        'context': {'intelligence': {'items': [{'id': 'evidence-1'}]}},
        'execution_trace': {'ml_candidate': {
            'probabilities': {'H': .7, 'D': .2, 'A': .1}, 'base_probabilities': matrix_outcomes(MATRIX),
            'model_version': ml_version, 'training_cutoff_at': (as_of-timedelta(days=1)).isoformat(),
            'applied': False, 'status': 'shadow_only', 'feature_audit': {'complete': True}}},
    }


def record(match_id='m1', as_of=AS_OF, **kwargs):
    result = {'match_id': match_id, 'home': 'Home', 'away': 'Away', 'league': 'Test',
              'match_time': (as_of+timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
              'predicted_scores': deepcopy(MATRIX), 'predicted_1x2': matrix_outcomes(MATRIX),
              'asian': -.5, 'total_line': 2.5}
    status = append_prediction_event(result, payload(as_of, **kwargs), now=as_of+timedelta(seconds=1))
    assert status['appended'], status
    result.update(settled=True, actual_score='1-0', actual_result='H', result_quality={'grade': 'high'},
                  settled_at=(as_of+timedelta(hours=5)).isoformat())
    return result


class FrozenEventTests(unittest.TestCase):
    def test_first_event_is_immutable_idempotent_and_copies_context(self):
        row, original = {'match_id': 'm1'}, payload()
        first = append_prediction_event(row, original, now=AS_OF+timedelta(seconds=1))
        sealed = deepcopy(row['prediction_events'][0])
        retry = append_prediction_event(row, original, now=AS_OF+timedelta(seconds=20))
        self.assertEqual(retry['status'], 'duplicate')
        self.assertEqual(len(row['prediction_events']), 1)
        original['context']['intelligence']['items'][0]['id'] = 'mutated'
        original['variants']['statistical']['score_probabilities']['1-0'] = .8
        self.assertEqual(row['prediction_events'][0], sealed)
        next_payload = payload(AS_OF+timedelta(minutes=10))
        # Kickoff is the same match even though observation time moved.
        next_payload['kickoff_at'] = sealed['kickoff_at']
        append_prediction_event(row, next_payload, now=AS_OF+timedelta(minutes=10, seconds=1))
        self.assertEqual(len(row['prediction_events']), 2)
        self.assertEqual(row['selected_prediction_event_id'], first['event_id'])
        self.assertEqual(row['prediction_events'][0], sealed)

    def test_no_legacy_backfill_or_post_kickoff_recording(self):
        row = {'match_id': 'm1'}
        self.assertEqual(append_prediction_event(row, None)['status'], 'not_provided')
        self.assertNotIn('prediction_events', row)
        result = append_prediction_event(row, payload(), now=AS_OF+timedelta(hours=2))
        self.assertEqual(result['reason'], 'not_recorded_before_kickoff')
        self.assertNotIn('prediction_events', row)

    def test_timestamp_and_probability_fail_closed(self):
        for field, value in [('as_of', '2026-01-01T08:00:00'),
                             ('as_of', (AS_OF+timedelta(minutes=1)).isoformat()),
                             ('training_cutoff_at', AS_OF.isoformat())]:
            with self.subTest(field=field, value=value):
                data = payload()
                data[field] = value
                result = append_prediction_event({'match_id': 'm1'}, data, now=AS_OF)
                self.assertEqual(result['status'], 'rejected')
        for bad in ({'0-0': .5, '1-0': .4}, {'0-0': float('nan')}, {'1-0': 1.0}):
            data = payload()
            data['variants']['statistical']['score_probabilities'] = bad
            self.assertEqual(append_prediction_event({'match_id': 'm1'}, data, now=AS_OF)['status'], 'rejected')
        data = payload()
        data['variants']['statistical']['probabilities'] = {'H': .6, 'D': .2, 'A': .2}
        self.assertIn('inconsistent_score_and_1x2', append_prediction_event({'match_id': 'm1'}, data, now=AS_OF)['reason'])
        data = payload()
        data['variants']['production']['score_probabilities'] = None
        self.assertEqual(append_prediction_event({'match_id': 'm1'}, data, now=AS_OF)['reason'],
                         'missing_complete_score_matrix:production')

    def test_event_hash_detects_modified_evidence_and_never_falls_forward(self):
        row = record()
        row['prediction_events'][0]['context']['intelligence']['items'] = []
        self.assertEqual(validate_event(row['prediction_events'][0]), 'event_hash_mismatch')
        report = evaluate_frozen_events([row])
        self.assertEqual(report['n'], 0)
        self.assertEqual(report['exclusions']['event_hash_mismatch'], 1)

    def test_unknown_training_cutoff_is_descriptive_only(self):
        report = evaluate_frozen_events([record()])
        self.assertEqual(report['n'], 1)
        self.assertEqual(report['trained_provenance_n'], 0)
        self.assertFalse(report['release_qualified'])


class FrozenEvaluationTests(unittest.TestCase):
    def test_actual_score_metrics_and_missing_are_distinct(self):
        metrics = score_metrics(MATRIX, '1-0')
        self.assertEqual(metrics['actual_score_rank'], 1)
        self.assertAlmostEqual(metrics['score_logloss'], -math.log(.5))
        self.assertAlmostEqual(metrics['score_brier'], .34)
        self.assertTrue(metrics['hit_top10'])
        partial = score_metrics({'1-0': .4, '0-0': .2}, '2-0')
        self.assertIsNone(partial['actual_score_prob'])
        self.assertIsNone(partial['score_logloss'])
        self.assertIsNone(partial['hit_top10'])
        zero = score_metrics(MATRIX, '5-0')
        self.assertEqual(zero['actual_score_prob'], 0)
        self.assertGreater(zero['score_logloss'], 30)
        missing = score_metrics({}, '1-0')
        self.assertIsNone(missing['hit_top1'])

    def test_comparisons_pair_same_events_and_separate_actual_agent_execution(self):
        rows = [record('m1', agent_applied=True),
                record('m2', AS_OF+timedelta(days=1), agent_applied=False)]
        report = evaluate_frozen_events(rows)
        self.assertEqual(report['n'], 2)
        self.assertEqual(report['agent_actual_applied_n'], 1)
        self.assertEqual(report['agent_fallback_n'], 1)
        uplift = report['comparisons']['agent_adjusted_vs_market_adjusted']
        self.assertEqual(uplift['n'], 2)
        self.assertAlmostEqual(uplift['logloss_improvement']['estimate'], math.log(.6/.5)/2)
        self.assertIsNotNone(uplift['logloss_improvement']['ci95'])
        actual = report['agent_research_uplift_applied_only']
        self.assertEqual(actual['n'], 1)
        self.assertAlmostEqual(actual['logloss_improvement']['estimate'], math.log(.6/.5))
        self.assertIsNone(actual['logloss_improvement']['ci95'])
        self.assertEqual(report['variants']['market_only']['score']['n'], 0)
        self.assertIsNone(report['variants']['market_only']['score']['brier'])
        self.assertEqual(uplift['score']['n'], 2)

    def test_version_mixing_and_old_evaluation_cannot_create_ml_qualification(self):
        legacy = {'match_id': 'old', 'settled': True, 'actual_result': 'H',
                  'evaluation': {'ml_1x2_logloss': .001, 'base_1x2_logloss': 5}}
        rows = [record('m1'), record('m2', ml_version='ml-v2'), legacy]
        report = evaluate_frozen_ml(rows, min_samples=2, model_version='ml-v1')
        self.assertEqual(report['overall']['sample_count'], 1)
        self.assertFalse(report['overall']['qualified'])
        self.assertAlmostEqual(report['overall']['base_1x2_logloss'], -math.log(.5))
        self.assertAlmostEqual(report['overall']['ml_1x2_logloss'], -math.log(.7))
        self.assertEqual(report['exclusions']['ml_model_version_mismatch'], 1)
        self.assertEqual(report['exclusions']['no_frozen_event'], 1)
        self.assertEqual(evaluate_frozen_ml(rows)['overall']['sample_count'], 0)

    def test_ml_cutoff_missing_or_asof_before_settlement_is_not_eligible(self):
        row = {'match_id': 'm1'}
        data = payload()
        data['execution_trace']['ml_candidate']['training_cutoff_at'] = None
        append_prediction_event(row, data, now=AS_OF)
        row.update(settled=True, actual_score='1-0', actual_result='H', settled_at=(AS_OF+timedelta(hours=5)).isoformat())
        report = evaluate_frozen_ml([row], min_samples=1, model_version='ml-v1')
        self.assertEqual(report['overall']['sample_count'], 0)
        self.assertIsNone(report['overall']['ml_1x2_brier'])
        self.assertEqual(evaluate_frozen_ml([record()], as_of=AS_OF)['overall']['sample_count'], 0)
        with self.assertRaises(ValueError):
            evaluate_frozen_ml([record()], as_of='2026-01-01')

    def test_research_production_versions_are_not_pooled(self):
        report = evaluate_frozen_events([record('m1'), record('m2', model_version='production-v2')])
        self.assertEqual(report['status'], 'version_filter_required')
        self.assertEqual(report['n'], 0)
        self.assertEqual(report['by_model_version']['production-v2']['n'], 1)

    def test_hot_path_skips_bootstrap_without_changing_paired_estimate(self):
        rows = [record('m1'), record('m2', AS_OF+timedelta(days=1))]
        fast = evaluate_frozen_events(rows, include_confidence_intervals=False)
        full = evaluate_frozen_events(rows)
        a, b = (report['comparisons']['statistical_vs_market_only'] for report in (fast, full))
        self.assertEqual(a['logloss_improvement']['estimate'], b['logloss_improvement']['estimate'])
        self.assertIsNone(a['logloss_improvement']['ci95'])
        self.assertIsNotNone(b['logloss_improvement']['ci95'])
        self.assertEqual(a['bootstrap_draws'], 0)


class PersistenceAndSettlementTests(unittest.TestCase):
    def history(self, rows=None):
        history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
        history.records = rows or []
        history._save_record = Mock(return_value='memory')
        history._hydrate_timeline = lambda row: row
        history._persistable = lambda row: row
        for name in ('_update_calibrator', '_update_market_db', '_update_score_frequency_db',
                     '_update_elo_ratings', '_update_half_time_stats', '_update_goal_count_stats', '_update_market_change_db'):
            setattr(history, name, Mock())
        return history

    def add(self, history, event=None):
        return history.add_prediction('m1', 'Test', 'Home', 'Away', '2035-01-01 10:00',
                                      deepcopy(MATRIX), matrix_outcomes(MATRIX), prediction_event=event)

    def test_settled_reanalysis_and_time_layers_cannot_rewrite_original_prediction(self):
        original = record()
        history = self.history([original])
        before = deepcopy(original)
        self.add(history, payload())
        self.assertEqual(original, before)
        self.assertFalse(history.update_time_layer('m1', 'final', {'9-0': 1}))
        self.assertEqual(original, before)

    def test_failed_prediction_persistence_is_retryable_without_fake_frozen_success(self):
        history = self.history()
        history._save_record.side_effect = ['failed', 'memory']
        now = datetime.now(UTC)-timedelta(seconds=2)
        first = self.add(history, payload(now))
        self.assertFalse(first['saved'])
        self.assertEqual(first['prediction_event']['status'], 'persistence_failed')
        self.assertEqual(history.records, [])
        second = self.add(history, payload(now))
        self.assertTrue(second['saved'])
        self.assertEqual(len(history.records[0]['prediction_events']), 1)

    @patch.object(result_sync, '_is_match_settle_due', return_value=True)
    @patch.object(result_sync, '_assess_result_quality', return_value={'grade': 'high'})
    @patch.object(result_sync, '_is_result_quality_usable', return_value=True)
    def test_repeated_settlement_ingests_once_and_claim_is_persisted_first(self, *_):
        row = record()
        row['settled'] = False
        history = self.history([row])
        stored = []
        history._save_record.side_effect = lambda r: stored.append(deepcopy(r)) or 'memory'
        self.assertTrue(history.update_result('m1', '1-0', 'H', now=AS_OF+timedelta(hours=5)))
        self.assertEqual(stored[0]['training_ingest']['status'], 'claimed')
        self.assertIsNotNone(datetime.fromisoformat(row['settled_at']).tzinfo)
        self.assertTrue(history.update_result('m1', '1-0', 'H'))
        self.assertEqual(history._update_calibrator.call_count, 1)
        self.assertEqual(history._update_elo_ratings.call_count, 1)
        self.assertEqual(row['training_ingest']['status'], 'attempted')

    @patch.object(result_sync, '_is_match_settle_due', return_value=True)
    @patch.object(result_sync, '_assess_result_quality', return_value={'grade': 'high'})
    @patch.object(result_sync, '_is_result_quality_usable', return_value=True)
    def test_failed_claim_never_runs_derived_updates_and_can_retry(self, *_):
        row = record()
        row['settled'] = False
        history = self.history([row])
        history._save_record.return_value = 'failed'
        self.assertFalse(history.update_result('m1', '1-0', 'H'))
        self.assertFalse(row['settled'])
        self.assertNotIn('training_ingest', row)
        history._update_elo_ratings.assert_not_called()

    @patch.object(result_sync, '_is_match_settle_due', return_value=True)
    @patch.object(result_sync, '_assess_result_quality', return_value={'grade': 'high'})
    @patch.object(result_sync, '_is_result_quality_usable', return_value=True)
    def test_interrupted_derived_write_cannot_be_replayed(self, *_):
        row = record()
        row['settled'] = False
        history = self.history([row])
        history._update_market_db.side_effect = RuntimeError('test interruption')
        self.assertTrue(history.update_result('m1', '1-0', 'H'))
        self.assertEqual(row['training_ingest']['status'], 'interrupted')
        self.assertTrue(row['training_ingest_requires_rebuild'])
        self.assertTrue(history.update_result('m1', '1-0', 'H'))
        history._update_calibrator.assert_called_once()
        history._update_elo_ratings.assert_not_called()

    @patch.object(result_sync, '_is_match_settle_due', return_value=True)
    @patch.object(result_sync, '_assess_result_quality', return_value={'grade': 'high'})
    def test_result_without_ingestion_claim_rolls_back_when_save_fails(self, *_):
        row = record()
        row.update(settled=False, skip_training_ingest=True)
        history = self.history([row])
        before = deepcopy(row)
        history._save_record.return_value = 'failed'
        self.assertFalse(history.update_result('m1', '1-0', 'H'))
        self.assertEqual(row, before)
        history._save_record.return_value = 'memory'
        self.assertTrue(history.update_result('m1', '1-0', 'H'))

    @patch.object(result_sync, '_is_match_settle_due', return_value=True)
    @patch.object(result_sync, '_assess_result_quality', return_value={'grade': 'high'})
    def test_legacy_settled_corrections_require_rebuild_instead_of_second_elo(self, *_):
        row = record()
        row.update(actual_half_score='0-0', half_time_data_quality='real')
        history = self.history([row])
        self.assertTrue(history.update_result('m1', '2-0', 'H'))
        self.assertEqual(row['actual_half_score'], '0-0')
        self.assertEqual(row['half_time_data_quality'], 'real')
        self.assertTrue(row['training_ingest_requires_rebuild'])
        self.assertEqual(len(row['settlement_corrections']), 1)
        history._update_calibrator.assert_not_called()
        history._update_elo_ratings.assert_not_called()

    def test_main_stats_and_export_have_valid_denominators_and_do_not_backfill_events(self):
        row = {'match_id': 'legacy', 'settled': True, 'actual_score': '1-0', 'actual_result': 'H',
               'predicted_1x2': {'H': .7, 'D': .2, 'A': .1}, 'predicted_scores': {}}
        history = self.history([row])
        stats = history.get_stats()
        self.assertEqual(stats['valid_1x2_predictions'], 1)
        self.assertEqual(stats['hit_rate_1x2'], 1)
        self.assertEqual(stats['valid_top10_predictions'], 0)
        self.assertIsNone(stats['hit_rate_top10'])
        self.assertIsNone(stats['score_brier'])
        before = deepcopy(row)
        with patch.object(result_sync, '_global_history', history):
            exported = result_sync.get_prediction_export()
        self.assertIsNone(exported['records'][0]['actual_score_prob'])
        self.assertIn('hit_top10', exported['records'][0])
        self.assertNotIn('prediction_events', exported['records'][0])
        self.assertEqual(row, before)

    @patch.object(result_sync.time, 'time', return_value=120)
    def test_point_cache_reuses_work_but_invalidates_for_result_or_event_mutation(self, *_):
        row = record()
        history = self.history([row])
        with patch.object(result_sync, 'evaluate_frozen_events', wraps=evaluate_frozen_events) as calculate:
            first = history.get_frozen_evaluation_stats(include_confidence_intervals=False)
            first['variants']['production']['one_x_two']['accuracy'] = 99
            again = history.get_frozen_evaluation_stats(include_confidence_intervals=False)
            self.assertEqual(again['variants']['production']['one_x_two']['accuracy'], 1)
            self.assertEqual(calculate.call_count, 1)
            row.update(actual_score='0-1', actual_result='A')
            changed = history.get_frozen_evaluation_stats(include_confidence_intervals=False)
            self.assertEqual(changed['variants']['production']['one_x_two']['accuracy'], 0)
            self.assertEqual(calculate.call_count, 2)
            row['prediction_events'][0]['context']['intelligence'] = {}
            invalid = history.get_frozen_evaluation_stats(include_confidence_intervals=False)
            self.assertEqual(invalid['n'], 0)
            self.assertEqual(invalid['exclusions']['event_hash_mismatch'], 1)
            self.assertEqual(calculate.call_count, 3)


if __name__ == '__main__':
    unittest.main()
