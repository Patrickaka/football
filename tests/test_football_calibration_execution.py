import unittest
from unittest import mock

from src.domain.sports.football import calibration
from src.football import scoring


class CalibrationExecutionTests(unittest.TestCase):
    def test_insufficient_data_platt_preserves_probabilities_exactly(self):
        matrix = {(0, 0): .1, (1, 0): .8, (0, 1): .1}
        params = calibration.fit_platt_scaling([])
        self.assertEqual(calibration.calibrate_with_platt(matrix, {'platt_params': params}), matrix)
        self.assertEqual(calibration.calibrate_with_platt(matrix, {'platt_params': (2, -1), 'trained_on': 0}), matrix)

    def test_invalid_platt_parameters_preserve_distribution(self):
        matrix = {(0, 0): .2, (1, 0): .8}
        for params in (None, (float('nan'), 0), (1,), ('bad', 0)):
            with self.subTest(params=params):
                self.assertEqual(calibration.calibrate_with_platt(matrix, {'platt_params': params}), matrix)

    def test_ensemble_calibrates_once_after_blending_and_reports_actual_method(self):
        matrix = [((1, 0), .6), ((0, 1), .4)]
        with mock.patch.object(scoring, 'predict_scores', return_value=(matrix, 1.2, .8, {})) as member:
            with mock.patch.object(scoring, 'calibrate_predictions', return_value={'1-0': .55, '0-1': .45}) as calibrate:
                result, _, _, meta = scoring.ensemble_predict_scores(
                    {'handicap': 0}, {}, {'close_line': 2.5}, enable_calibration=True,
                    enable_draw_calibration=False)
        self.assertEqual(member.call_count, 2)
        self.assertTrue(all(call.kwargs['enable_calibration'] is False for call in member.call_args_list))
        self.assertTrue(all(call.kwargs['enable_draw_calibration'] is False for call in member.call_args_list))
        calibrate.assert_called_once()
        self.assertEqual(dict(result), {(1, 0): .55, (0, 1): .45})
        self.assertTrue(meta['calibrated'])
        self.assertEqual(meta['calibration_method'], 'bayesian')

    def test_disabled_or_identity_calibration_is_not_reported_as_applied(self):
        matrix = [((1, 0), .6), ((0, 1), .4)]
        for enabled in (False, True):
            with self.subTest(enabled=enabled), mock.patch.object(scoring, 'predict_scores', return_value=(matrix, 1.2, .8, {})):
                with mock.patch.object(scoring, 'calibrate_predictions', side_effect=lambda values, *args: values) as calibrate:
                    _, _, _, meta = scoring.ensemble_predict_scores({'handicap': 0}, {}, {'close_line': 2.5}, enable_calibration=enabled)
                self.assertFalse(meta['calibrated'])
                self.assertIsNone(meta['calibration_method'])
                self.assertEqual(calibrate.call_count, int(enabled))


if __name__ == '__main__':
    unittest.main()
