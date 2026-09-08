"""Exclusion-only strategy changes need new identities and replayable snapshots."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.kl8 import analyzer as analyzer_module
from src.kl8 import config
from src.kl8 import fetch
from src.kl8 import records
from src.kl8 import strategies
from src.kl8.analyzer import KL8Analyzer


def strategy_fixture():
    return {
        'strategy_id': 'fixture',
        'feature_weights': {'frequency': 1.0},
        'model_weights': {'rank': 1.0},
        'window_size': 50,
        'final_selection_mode': 'concentrated',
        'prediction_mode': 'reference_unvalidated',
        'is_validated': False,
    }


def exclusion_fixture():
    return {
        'exclusion_selection_mode': 'balanced',
        'first_exclusion_strategy': {
            'strategy_id': 'first-round-fixture',
            'feature_weights': {'frequency': .7, 'gap': .3},
            'model_weights': {'rank': 1.0},
            'window_size': 100,
            'exclusion_selection_mode': 'concentrated',
        },
    }


def analyzer_fixture():
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_data = [
        {'issue': str(2026200 - i), 'numbers': list(range(1, 21)), 'date': '2026-08-01'}
        for i in range(config.KL8_MIN_PREDICTION_PERIODS)
    ]
    analyzer.using_simulated_data = False
    analyzer.statistics = {'last_numbers': set(range(1, 21))}
    return analyzer


class ExclusionStrategyFingerprintTests(unittest.TestCase):
    def test_unconfigured_strategy_keeps_legacy_fingerprint(self):
        # Recorded before exclusion options entered the hash; pin the code
        # version so future model releases do not change this compatibility check.
        with mock.patch.object(records, 'KL8_PREDICTOR_VERSION', 'fingerprint-compat-v1'):
            self.assertEqual(records._strategy_fingerprint(strategy_fixture()), '324b8986f020')

    def test_selection_mode_and_nested_override_each_change_identity(self):
        baseline = {**strategy_fixture(), **exclusion_fixture()}
        variants = []
        changed_mode = deepcopy(baseline)
        changed_mode['exclusion_selection_mode'] = 'concentrated'
        variants.append(changed_mode)
        changed_weights = deepcopy(baseline)
        changed_weights['first_exclusion_strategy']['feature_weights'] = {'frequency': .2, 'gap': .8}
        variants.append(changed_weights)
        changed_window = deepcopy(baseline)
        changed_window['first_exclusion_strategy']['window_size'] = 150
        variants.append(changed_window)
        changed_nested_mode = deepcopy(baseline)
        changed_nested_mode['first_exclusion_strategy']['exclusion_selection_mode'] = 'balanced'
        variants.append(changed_nested_mode)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(records._strategy_fingerprint(baseline),
                                    records._strategy_fingerprint(variant))
                for play in ('select_6', 'fu_shi_7'):
                    self.assertNotEqual(
                        records._resolved_strategies_fingerprint({play: baseline}),
                        records._resolved_strategies_fingerprint({play: variant}),
                    )
        reordered = json.loads(json.dumps(baseline, sort_keys=True))
        self.assertEqual(records._strategy_fingerprint(baseline),
                         records._strategy_fingerprint(reordered))

    def test_config_cache_fingerprint_tracks_both_exclusion_options(self):
        baseline = strategy_fixture()
        with mock.patch.object(config, 'ACTIVE_STRATEGIES', {'select_6': baseline}):
            old_fingerprint = records._prediction_config_fingerprint()
        for field, value in exclusion_fixture().items():
            with self.subTest(field=field), mock.patch.object(
                    config, 'ACTIVE_STRATEGIES', {'select_6': {**baseline, field: value}}):
                self.assertNotEqual(old_fingerprint, records._prediction_config_fingerprint())

    def test_exclusion_only_changes_do_not_reuse_same_issue_snapshot_identity(self):
        analyzer = analyzer_fixture()
        baseline = strategy_fixture()
        with_mode = {**baseline, 'exclusion_selection_mode': 'balanced'}
        with_override = {**with_mode, 'first_exclusion_strategy': exclusion_fixture()['first_exclusion_strategy']}
        snapshots = []
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(config, 'KL8_SNAPSHOT_DIR', temp_dir):
            for strategy in (baseline, with_mode, with_override, deepcopy(with_override)):
                filename = analyzer._save_prediction_snapshot({
                    'resolved_strategies': {'select_6': strategy, 'fu_shi_7': deepcopy(strategy)},
                    'strategy_config_fingerprint': 'fixed-config-to-isolate-strategy-identity',
                    'select_6': {'numbers': [21, 22, 23, 24, 25, 26]},
                    'fu_shi_7': {'top7_numbers': [21, 22, 23, 24, 25, 26, 27]},
                })
                self.assertIsNotNone(filename)
                snapshots.append(json.loads((Path(temp_dir) / filename).read_text(encoding='utf-8')))
        self.assertEqual([row['is_experiment'] for row in snapshots], [False, False, False, True])
        self.assertEqual(len({row['strategy_fingerprint'] for row in snapshots[:3]}), 3)
        for row in snapshots:
            self.assertEqual(len(row['select_6']), 6)
            self.assertEqual(len(row['fu_shi_7']), 7)
        for play in ('select_6', 'fu_shi_7'):
            saved = snapshots[2]['resolved_strategies'][play]
            self.assertEqual(saved['first_exclusion_strategy'], with_override['first_exclusion_strategy'])
            self.assertEqual(saved['exclusion_selection_mode'], 'balanced')


class ExclusionStrategyMetadataTests(unittest.TestCase):
    def predict_with_strategy(self, strategy):
        analyzer = analyzer_fixture()
        candidates = [(num, 1.0 - num / 100) for num in range(21, 81)]
        with mock.patch.object(strategies, 'resolve_play_strategy', return_value=strategy), \
                mock.patch.object(analyzer_module, '_load_last_snapshot', return_value=None), \
                mock.patch.object(analyzer_module, '_build_recent_settlement_performance', return_value={}), \
                mock.patch.object(fetch, 'count_valid_history_periods', return_value=len(analyzer.history_data)), \
                mock.patch.object(analyzer, 'build_pool_by_strategy', return_value={
                    'selected': [num for num, _ in candidates[:20]], 'candidates': candidates,
                }), \
                mock.patch.object(analyzer, '_calculate_select_recalculation', return_value=(
                    {'numbers': [41, 42, 43, 44, 45, 46]}, None,
                )), \
                mock.patch.object(analyzer, '_save_prediction_snapshot', return_value=None):
            return analyzer, analyzer.predict_all()

    def test_predict_all_preserves_configured_options_in_single_and_linked_pools(self):
        strategy = {**strategy_fixture(), **exclusion_fixture()}
        analyzer, result = self.predict_with_strategy(strategy)
        resolved = result['resolved_strategies']
        for play in ('select_6', 'fu_shi_7', 'fu_shi_10_11'):
            with self.subTest(play=play):
                self.assertEqual(resolved[play]['exclusion_selection_mode'], 'balanced')
                self.assertEqual(resolved[play]['first_exclusion_strategy'], strategy['first_exclusion_strategy'])
        # Configuration edits after prediction must not rewrite the saved provenance.
        strategy['first_exclusion_strategy']['feature_weights']['gap'] = .99
        self.assertEqual(resolved['select_6']['first_exclusion_strategy']['feature_weights']['gap'], .3)
        resolved['select_6']['first_exclusion_strategy']['window_size'] = 120
        self.assertEqual(resolved['fu_shi_7']['first_exclusion_strategy']['window_size'], 100)
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(config, 'KL8_SNAPSHOT_DIR', temp_dir):
            filename = analyzer._save_prediction_snapshot(result)
            saved = json.loads((Path(temp_dir) / filename).read_text(encoding='utf-8'))
        self.assertEqual(saved['resolved_strategies'], resolved)
        self.assertEqual(saved['strategy_fingerprint'], records._resolved_strategies_fingerprint(resolved))

    def test_unconfigured_prediction_does_not_invent_exclusion_overrides(self):
        _, result = self.predict_with_strategy(strategy_fixture())
        for play in ('select_6', 'fu_shi_7', 'fu_shi_10_11'):
            self.assertNotIn('exclusion_selection_mode', result['resolved_strategies'][play])
            self.assertNotIn('first_exclusion_strategy', result['resolved_strategies'][play])
        self.assertEqual(len(result['select_6']['numbers']), 6)
        self.assertEqual(len(result['fu_shi_7']['top7_numbers']), 7)


if __name__ == '__main__':
    unittest.main()
