import unittest

from src.kl8 import (
    KL8RollingBacktest,
    KL8Analyzer,
    _clean_pick_numbers,
    _compute_next_issue,
    normalize_record,
)


def _record(issue: int):
    base = ((issue - 1) % 60) + 1
    nums = [((base + i - 1) % 80) + 1 for i in range(20)]
    return {'issue': str(2026000 + issue), 'numbers': sorted(set(nums))}


class KL8PredictionGuardTests(unittest.TestCase):
    def test_normalize_record_strips_issue_and_rejects_bad_numbers(self):
        record = normalize_record({'issue': ' 2026001 ', 'numbers': list(range(1, 21))})
        self.assertEqual(record['issue'], '2026001')

        self.assertIsNone(normalize_record({'issue': '2026001', 'numbers': list(range(1, 20))}))
        self.assertIsNone(normalize_record({'issue': '2026001', 'numbers': [1] * 20}))
        self.assertIsNone(normalize_record({'issue': '2026001', 'numbers': list(range(62, 82))}))

    def test_clean_pick_numbers_requires_exact_unique_range(self):
        self.assertEqual(_clean_pick_numbers([1, 2, 3], 3), [1, 2, 3])
        self.assertEqual(_clean_pick_numbers([1, 1, 2], 3), [])
        self.assertEqual(_clean_pick_numbers([1, 2, 81], 3), [])
        self.assertEqual(_clean_pick_numbers([1, 2], 3), [])

    def test_compute_next_issue_uses_recent_diffs(self):
        history = [
            {'issue': '2026001', 'numbers': list(range(1, 21))},
            {'issue': '2026010', 'numbers': list(range(1, 21))},
            {'issue': '2026011', 'numbers': list(range(1, 21))},
            {'issue': '2026012', 'numbers': list(range(1, 21))},
        ]
        self.assertEqual(_compute_next_issue('2026012', history), '2026013')

    def test_backtest_passes_repeat_configuration_to_voting(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False

        captured = []
        original = KL8Analyzer.multi_model_voting

        def fake_voting(self, **kwargs):
            captured.append(kwargs)
            return {
                'selected': list(range(1, 21)),
                'candidates': [(n, float(21 - n)) for n in range(1, 21)],
                'votes': {},
            }

        try:
            KL8Analyzer.multi_model_voting = fake_voting
            result = KL8RollingBacktest(analyzer)._rolling_backtest_parametric(
                {'frequency': 1.0},
                {'rank': 1.0},
                start_idx=55,
                end_idx=70,
                min_train=50,
                window_size=50,
                repeat_direction='avoid',
                repeat_avoid_score=0.12,
                repeat_non_avoid_score=0.88,
            )
        finally:
            KL8Analyzer.multi_model_voting = original

        self.assertNotIn('error', result)
        self.assertTrue(captured)
        self.assertTrue(all(c['repeat_direction'] == 'avoid' for c in captured))
        self.assertTrue(all(c['repeat_avoid_score'] == 0.12 for c in captured))
        self.assertTrue(all(c['repeat_non_avoid_score'] == 0.88 for c in captured))


if __name__ == '__main__':
    unittest.main()
