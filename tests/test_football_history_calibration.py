import unittest

from src.football.history_calibration import (
    apply_history_calibration,
    estimate_history_calibration,
)


class FootballHistoryCalibrationTests(unittest.TestCase):
    @staticmethod
    def _record(index, actual_score='3-1'):
        return {
            'match_id': str(index),
            'created_at': f'2026-06-{(index % 28) + 1:02d}T12:00:00',
            'settled': True,
            'sync_status': 'synced',
            'actual_score': actual_score,
            'predicted_scores': {
                '0-0': 0.20,
                '1-0': 0.30,
                '1-1': 0.30,
                '2-1': 0.20,
            },
            'predicted_1x2': {'H': 0.50, 'D': 0.50, 'A': 0.0},
            'asian': 0.0,
            'total_line': 2.5,
            'odds_snapshot': {'euro': {'close': {}}},
        }

    def test_profile_waits_for_enough_settled_samples(self):
        profile = estimate_history_calibration([self._record(i) for i in range(20)])
        self.assertFalse(profile['applied'])
        self.assertEqual(profile['reason'], 'insufficient_history')

    def test_underestimated_goal_history_produces_guarded_positive_tilt(self):
        profile = estimate_history_calibration([self._record(i) for i in range(100)])
        self.assertTrue(profile['applied'])
        self.assertGreater(profile['goal_beta'], 0)
        self.assertLessEqual(profile['goal_beta'], 0.18)
        self.assertGreater(profile['actual_goal_mean'], profile['predicted_goal_mean'])

    def test_applying_positive_tilt_increases_expected_goals_and_normalizes(self):
        candidates = [((0, 0), 0.30), ((1, 0), 0.30), ((1, 1), 0.25), ((3, 1), 0.15)]
        adjusted, meta = apply_history_calibration(candidates, {
            'applied': True,
            'source': 'test',
            'sample_count': 100,
            'effective_weight': 70,
            'goal_beta': 0.15,
            'outcome_weights': {'H': 1.0, 'D': 1.0, 'A': 1.0},
        })
        self.assertTrue(meta['applied'])
        self.assertAlmostEqual(sum(probability for _, probability in adjusted), 1.0)
        self.assertGreater(meta['expected_goals_after'], meta['expected_goals_before'])
        before_home = sum(probability for (home, away), probability in candidates if home > away)
        after_home = sum(probability for (home, away), probability in adjusted if home > away)
        before_draw = sum(probability for (home, away), probability in candidates if home == away)
        after_draw = sum(probability for (home, away), probability in adjusted if home == away)
        self.assertAlmostEqual(before_home, after_home)
        self.assertAlmostEqual(before_draw, after_draw)
        self.assertTrue(meta['preserved_1x2'])


if __name__ == '__main__':
    unittest.main()
