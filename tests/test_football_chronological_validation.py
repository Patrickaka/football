"""Model fitting and policy fitting must both exclude every test-date result."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.backtest import backtest_football_professional as backtest
from src.domain.sports.football import validation


def dated_rows():
    rows = []
    for day, count in enumerate((3, 2, 4, 1, 2), 1):
        for _ in range(count):
            rows.append({
                'date': f'2025-01-{day:02d}',
                'match_id': f'm{len(rows):02d}',
                'actual': 'H',
                'probabilities': {'H': .60, 'D': .20, 'A': .20},
                'odds': {'H': 2.0, 'D': 3.5, 'A': 4.0},
            })
    return rows


class DateGroupedValidationTests(unittest.TestCase):
    def test_warmup_and_test_boundaries_include_the_entire_final_day(self):
        rows = dated_rows()
        bounds = validation.chronological_fold_bounds(
            [row['date'] for row in rows], initial_train=2, test_size=2,
        )
        self.assertEqual(bounds, [(3, 5), (5, 9), (9, 12)])
        for start, stop in bounds:
            self.assertLess(rows[start - 1]['date'], rows[start]['date'])
            if stop < len(rows):
                self.assertLess(rows[stop - 1]['date'], rows[stop]['date'])
        self.assertEqual([i for start, stop in bounds for i in range(start, stop)], list(range(3, 12)))

    def test_residual_and_threshold_fits_see_only_strictly_earlier_dates(self):
        rows = dated_rows()
        with patch.object(validation, 'select_market_residual_weight', wraps=validation.select_market_residual_weight) as residual, \
                patch.object(validation, 'select_threshold', wraps=validation.select_threshold) as threshold:
            report = validation.walk_forward_evaluate(
                list(reversed(rows)), initial_train=2, test_size=2, min_training_bets=1,
            )
        self.assertEqual(report['out_of_sample_n'], 9)
        self.assertTrue(report['date_grouped'])
        self.assertEqual([fold['test_n'] for fold in report['folds']], [2, 4, 3])
        for calls in (residual.call_args_list, threshold.call_args_list):
            self.assertEqual(len(calls), len(report['folds']))
            for call, fold in zip(calls, report['folds']):
                training = call.args[0]
                self.assertEqual(len(training), fold['train_n'])
                self.assertEqual(max(row['date'] for row in training), fold['train_end'])
                self.assertLess(fold['train_end'], fold['test_start'])

    def test_single_day_and_short_history_produce_no_oos_claim(self):
        for rows, warmup in ((dated_rows()[:3], 2), (dated_rows(), 12), ([], 1)):
            with self.subTest(size=len(rows), warmup=warmup):
                with patch.object(validation, 'select_market_residual_weight') as fit:
                    report = validation.walk_forward_evaluate(rows, warmup, 2)
                self.assertEqual(report['out_of_sample_n'], 0)
                self.assertEqual(report['folds'], [])
                fit.assert_not_called()

    def test_missing_or_invalid_dates_are_not_reported_as_time_validated(self):
        for invalid in (None, '', 'not-a-date', '2025-02-30'):
            rows = dated_rows()
            rows[1]['date'] = invalid
            with self.subTest(date=invalid), self.assertRaisesRegex(ValueError, 'valid ISO date'):
                validation.walk_forward_evaluate(rows, 2, 2)

    def test_nonpositive_sizes_fail_instead_of_looping_or_using_no_training(self):
        for warmup, size in ((0, 2), (-1, 2), (2, 0), (2, -1)):
            with self.subTest(warmup=warmup, size=size), self.assertRaisesRegex(ValueError, 'positive integer'):
                validation.walk_forward_evaluate(dated_rows(), warmup, size)


class ModelChronologicalValidationTests(unittest.TestCase):
    def test_classifier_training_and_prediction_never_share_a_date(self):
        rows = dated_rows()
        training_batches, test_batches = [], []

        class RecordingModel:
            def __init__(self, **kwargs):
                pass

            def fit(self, features, labels):
                training_batches.append(features[:, 0].astype(int).tolist())

            def predict_proba(self, features):
                test_batches.append(features[:, 0].astype(int).tolist())
                return np.tile([.60, .20, .20], (len(features), 1))

        samples = [{
            'match_id': row['match_id'], 'match_date': row['date'], 'league': 'E0',
            'features': {'row_index': index}, 'target': {'result': row['actual']},
        } for index, row in enumerate(rows)]
        prices = {row['match_id']: {'odds': row['odds']} for row in rows}
        with patch.dict(sys.modules, {'catboost': types.SimpleNamespace(CatBoostClassifier=RecordingModel)}), \
                patch.object(backtest, 'get_feature_names', return_value=['row_index']):
            output = backtest.generate_oos_predictions(list(reversed(samples)), prices, 2, 2)
        self.assertEqual([len(batch) for batch in training_batches], [3, 5, 9])
        self.assertEqual([len(batch) for batch in test_batches], [2, 4, 3])
        self.assertEqual([row['match_id'] for row in output], [row['match_id'] for row in rows[3:]])
        for training, test in zip(training_batches, test_batches):
            self.assertLess(max(rows[index]['date'] for index in training), min(rows[index]['date'] for index in test))

    def test_insufficient_distinct_dates_do_not_train_a_classifier(self):
        samples = [{'match_date': '2025-01-01', 'match_id': str(index)} for index in range(4)]
        self.assertEqual(backtest.generate_oos_predictions(samples, {}, warmup=2, fold_size=2), [])
        self.assertEqual(backtest.generate_oos_predictions([], {}, warmup=2, fold_size=2), [])

    def test_cli_does_not_write_a_report_without_a_later_validation_date(self):
        oos = dated_rows()[:3]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'report.json'
            with patch.object(sys, 'argv', ['backtest', '--threshold-train', '2', '--out', str(out)]), \
                    patch.object(backtest, 'load_samples', return_value=[{}]), \
                    patch.object(backtest, 'load_prices', return_value={}), \
                    patch.object(backtest, 'generate_oos_predictions', return_value=oos):
                with self.assertRaisesRegex(RuntimeError, 'distinct dates'):
                    backtest.main()
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
