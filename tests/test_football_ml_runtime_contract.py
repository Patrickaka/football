import copy
import hashlib
import json
import os
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.football import ml, ml_trainer
from src.football.ml_feature_schema import get_feature_defaults, get_feature_names
from src.football.ml_features import build_prediction_features, feature_vector, histories_from_records


class FittedModel:
    n_features_in_ = 40
    classes_ = [2, 0, 1]

    def __init__(self):
        self.calls = 0
        self.probabilities = [[0.2, 0.5, 0.3]]

    def is_fitted(self):
        return True

    def predict_proba(self, values):
        self.calls += 1
        self.seen = values.copy()
        return self.probabilities


def metadata():
    return {'feature_version': 'v2', 'feature_builder_version': 'v2-strict-2',
            'features': get_feature_names(), 'model_version': 'test-v2',
            'train_count': 300, 'validation_count': 200, 'test_count': 200, 'classes': [2, 0, 1],
            'training_cutoff_at': '2026-01-04T00:00:00+00:00',
            'split_dates': {'train_end': '2026-01-01T00:00:00+00:00',
                            'validation_end': '2026-01-02T00:00:00+00:00',
                            'test_end': '2026-01-03T00:00:00+00:00'}}


def history():
    return [{'match_id': str(day), 'date': f'2026-01-{day:02d}T10:00:00+00:00',
             'observed_at': f'2026-01-{day:02d}T12:00:00+00:00',
             'goals_for': day % 4, 'goals_against': 1, 'venue': 'home' if day % 2 else 'away'}
            for day in range(1, 13)]


def valid_features():
    return {**get_feature_defaults(), 'home_matches_count': 10, 'away_matches_count': 10}


class RuntimeFeatureTests(unittest.TestCase):
    def test_real_history_and_markets_form_the_same_training_vector(self):
        rows = history()
        result = build_prediction_features(
            euro={'close': {'home': .5, 'draw': .3, 'away': .2}},
            asian={'handicap': .75, 'close_prob': {'home_give': .55, 'away_recv': .45}},
            total={'close_line': 2.5, 'close_prob': {'over': .6, 'under': .4}},
            team={'elo_home': 1600, 'elo_away': 1500, 'home_history': rows, 'away_history': rows},
            league_profile={'history': rows}, match_time='2026-02-02T00:00:00+00:00',
            as_of='2026-02-01T00:00:00+00:00')
        self.assertTrue(result['available'], result['audit'])
        self.assertEqual(result['features']['asian_handicap'], -.75)
        self.assertEqual(result['features']['elo_diff'], 100)
        self.assertAlmostEqual(result['features']['home_attack_5'], sum(row['goals_for'] for row in rows[-5:]) / 5)
        x, _ = ml_trainer.prepare_features_target(
            [{'features': result['features'], 'target': {'result': 'H'}}], get_feature_names())
        self.assertEqual(x[0].tolist(), feature_vector(result['features']))

    def test_aggregate_form_is_not_relabelled_as_observed_five_match_history(self):
        result = build_prediction_features(team={'home_recent': {'games': 10, 'attack': 1.5}, 'elo_home': 1600})
        self.assertFalse(result['available'])
        self.assertNotIn('home_attack_5', result['features'])
        self.assertNotIn('elo_away', result['features'])

    def test_proxy_market_is_explicitly_missing(self):
        result = build_prediction_features(asian={'source': 'model_proxy', 'handicap': 0,
                                                  'close_prob': {'home': .5, 'away': .5}})
        self.assertEqual(result['features']['has_asian_odds'], 0)
        self.assertNotIn('asian_home_prob', result['features'])

    def test_local_records_require_exact_team_quality_and_observed_result_cutoff(self):
        base = {'match_id': 'valid', 'settled': True, 'home': 'Home', 'away': 'Away', 'league': 'League',
                'match_time': '2026-01-01T10:00:00+00:00', 'actual_score': '2-1',
                'result_quality': {'grade': 'high'}, 'settled_at': '2026-01-01T12:00:00+00:00'}
        records = [base, {**base, 'match_id': 'future', 'settled_at': '2026-03-01T12:00:00+00:00'},
                   {**base, 'match_id': 'unverified', 'result_quality': {}},
                   {**base, 'match_id': 'unknown_time', 'settled_at': None},
                   {**base, 'match_id': 'other', 'home': 'Home Reserves', 'away': 'Away Reserves'}]
        result = build_prediction_features(history_records=records, home='Home', away='Away', league='League',
                                           match_time='2026-02-02T00:00:00+00:00', as_of='2026-02-01T00:00:00+00:00')
        self.assertEqual(result['features']['home_matches_count'], 1)
        self.assertEqual(result['features']['away_matches_count'], 1)
        self.assertNotIn('home_attack_5', result['features'])
        self.assertFalse(result['available'])

    def test_short_schedule_uses_only_valid_selected_frozen_kickoff(self):
        from tests.test_football_prediction_events import record
        row = record()
        row['match_time'] = '01-01 18:00'
        as_of = '2026-02-01T00:00:00Z'
        result = histories_from_records([row], 'Home', 'Away', 'Test', as_of=as_of)
        self.assertEqual(result['home_history'][0]['date'], row['prediction_events'][0]['kickoff_at'])
        for field, value in [('selected_prediction_event_id', 'different'),
                             ('prediction_events', [{**row['prediction_events'][0], 'kickoff_at': '2025-01-01T00:00:00Z'}])]:
            with self.subTest(field=field):
                candidate = {**row, field: value, 'match_time': '2026-01-01T10:00:00Z'}
                self.assertEqual(histories_from_records([candidate], 'Home', 'Away', 'Test', as_of=as_of)['home_history'], [])

    def test_observation_and_prediction_times_require_explicit_timezones(self):
        for field in ('as_of', 'observed_at', 'match_time'):
            with self.subTest(field=field):
                rows = history()
                args = {'match_time': '2026-02-02T00:00:00Z', 'as_of': '2026-02-01T00:00:00Z'}
                if field == 'observed_at':
                    rows = [{**row, 'observed_at': row['observed_at'][:19]} for row in rows]
                else:
                    args[field] = args[field][:19]
                result = build_prediction_features(team={'home_history': rows}, **args)
                self.assertNotIn('home_attack_5', result['features'])
        rows = [{**row, 'date': row['date'][:10]} for row in history()]
        result = build_prediction_features(team={'home_history': rows},
                                           match_time='2026-02-02T00:00:00Z', as_of='2026-02-01T00:00:00Z')
        self.assertEqual(result['features']['home_matches_count'], 12)

    def test_windows_and_venue_features_require_the_advertised_support(self):
        args = {'match_time': '2026-02-02T00:00:00Z', 'as_of': '2026-02-01T00:00:00Z'}
        for count in (1, 4, 5, 9):
            with self.subTest(count=count):
                result = build_prediction_features(team={'home_history': history()[:count]}, **args)
                self.assertNotIn('home_goals_for_10', result['features'])
                self.assertFalse(result['available'])
                self.assertEqual('home_attack_5' in result['features'], count >= 5)
        rows = [{**row, 'venue': 'home' if i < 4 else 'away'} for i, row in enumerate(history())]
        result = build_prediction_features(team={'home_history': rows}, **args)
        self.assertIn('home_goals_for_10', result['features'])
        self.assertNotIn('home_h_goals_for_5', result['features'])

    def test_live_precomputed_dict_cannot_bypass_observed_history_requirements(self):
        result = build_prediction_features(team={'ml_features': valid_features()},
                                           match_time='2026-02-02T00:00:00Z', as_of='2026-02-01T00:00:00Z')
        self.assertFalse(result['available'])
        self.assertNotIn('home_attack_5', result['features'])

    def test_low_sample_snapshot_is_rejected_by_training_and_inference(self):
        features = valid_features()
        features['home_matches_count'] = 9
        model = FittedModel()
        self.assertFalse(ml.predict_with_model(model, metadata(), features)['available'])
        self.assertEqual(model.calls, 0)
        with self.assertRaisesRegex(ValueError, 'at_least_10_observed_matches_required'):
            ml_trainer.prepare_features_target([{'features': features, 'target': {'result': 'H'}}], get_feature_names())

    def test_current_future_and_late_observed_games_are_excluded(self):
        rows = history()
        result = build_prediction_features(team={'home_history': rows},
                                           match_time='2026-01-10T10:00:00+00:00',
                                           as_of='2026-01-09T11:00:00+00:00')
        self.assertEqual(result['features']['home_matches_count'], 8)

    def test_history_without_result_observation_or_prediction_cutoff_stays_missing(self):
        rows = [{**row, 'observed_at': None} for row in history()]
        result = build_prediction_features(team={'home_history': rows},
                                           match_time='2026-02-02T00:00:00+00:00',
                                           as_of='2026-02-01T00:00:00+00:00')
        self.assertNotIn('home_attack_5', result['features'])
        result = build_prediction_features(team={'home_history': history()},
                                           match_time='2026-02-02T00:00:00+00:00')
        self.assertNotIn('home_attack_5', result['features'])

    def test_invalid_inputs_never_reach_model(self):
        for value in (None, float('nan'), float('inf'), True, '1500'):
            with self.subTest(value=value):
                features = valid_features()
                features['elo_home'] = value
                model = FittedModel()
                self.assertFalse(ml.predict_with_model(model, metadata(), features)['available'])
                self.assertEqual(model.calls, 0)
        features = valid_features()
        features.pop('home_attack_5')
        self.assertFalse(ml.predict_with_model(FittedModel(), metadata(), features)['available'])

    def test_unknown_training_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid_training_result'):
            ml_trainer.prepare_features_target(
                [{'features': valid_features(), 'target': {'result': '?'}}], get_feature_names())

    def test_catboost_column_predictions_do_not_broadcast_accuracy(self):
        result = ml_trainer.calculate_metrics(np.array([0, 1, 2]), np.array([[0], [1], [2]]))
        self.assertEqual(result['accuracy'], 1.0)

    def test_training_boundaries_keep_same_date_together(self):
        rows = [{'match_date': f'2026-01-{day:02d}', 'id': (day, duplicate)}
                for day in range(1, 11) for duplicate in range(3)]
        train, validation, test = ml_trainer.split_by_time(list(reversed(rows)), .45, .25)
        self.assertLess(max(row['match_date'] for row in train), min(row['match_date'] for row in validation))
        self.assertLess(max(row['match_date'] for row in validation), min(row['match_date'] for row in test))
        self.assertEqual(len(train) + len(validation) + len(test), len(rows))


class RuntimeArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model_path = os.path.join(self.temp.name, 'model.pkl')
        self.meta_path = os.path.join(self.temp.name, 'model.json')

    def write_artifact(self, meta=None, model=None):
        blob = pickle.dumps(model or FittedModel())
        with open(self.model_path, 'wb') as handle:
            handle.write(blob)
        meta = copy.deepcopy(meta or metadata())
        meta['model_sha256'] = hashlib.sha256(blob).hexdigest()
        with open(self.meta_path, 'w', encoding='utf-8') as handle:
            json.dump(meta, handle)

    def test_complete_artifact_loads_and_classes_are_mapped_by_label(self):
        self.write_artifact()
        model, meta = ml.read_model_artifact(self.model_path, self.meta_path)
        result = ml.predict_with_model(model, meta, valid_features())
        self.assertTrue(result['available'])
        self.assertEqual((result['H'], result['D'], result['A']), (.5, .3, .2))
        self.assertEqual(result['training_cutoff_at'], meta['training_cutoff_at'])

    def test_missing_sidecar_does_not_use_unrelated_kv_metadata_or_stale_model(self):
        with mock.patch.object(ml, '_trained_ml_model', FittedModel()), mock.patch.object(ml.kv_store, 'load') as load:
            self.assertFalse(ml.load_trained_ml_model(self.model_path, self.meta_path))
            self.assertIsNone(ml._trained_ml_model)
        load.assert_not_called()

    def test_changed_pickle_does_not_load_under_old_metadata(self):
        self.write_artifact()
        with open(self.model_path, 'ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            ml.read_model_artifact(self.model_path, self.meta_path)

    def test_incompatible_or_unproven_metadata_is_rejected(self):
        bad_values = {'feature_version': 'v1', 'feature_builder_version': 'legacy',
                      'features': list(reversed(get_feature_names())), 'classes': [0, 1, 2],
                      'train_count': 0, 'training_cutoff_at': '2026-01-02T00:00:00+00:00'}
        for key, value in bad_values.items():
            with self.subTest(key=key):
                meta = metadata()
                meta[key] = value
                self.write_artifact(meta)
                with self.assertRaises(ValueError):
                    ml.read_model_artifact(self.model_path, self.meta_path)

    def test_untrained_model_is_not_invoked(self):
        model = FittedModel()
        with mock.patch.object(model, 'is_fitted', return_value=False):
            self.assertFalse(ml.predict_with_model(model, metadata(), valid_features())['available'])
        self.assertEqual(model.calls, 0)

    def test_invalid_probability_outputs_are_rejected(self):
        for probabilities in ([[float('nan'), .5, .5]], [[-.2, .7, .5]], [[0, 0, 0]], [[.5, .5]], [.2, .5, .3]):
            with self.subTest(probabilities=probabilities):
                model = FittedModel()
                model.probabilities = probabilities
                result = ml.predict_with_model(model, metadata(), valid_features())
                self.assertFalse(result['available'])
                self.assertEqual(result['reason'], 'invalid_model_probabilities')


if __name__ == '__main__':
    unittest.main()
