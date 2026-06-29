import unittest

from src.kl8 import (
    KL8RollingBacktest,
    KL8Analyzer,
    KL8_MIN_PREDICTION_PERIODS,
    _clean_pick_numbers,
    _diversify_candidate_pool,
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

    def test_diversify_candidate_pool_limits_basic_concentration(self):
        candidates = [
            (1, 100.0), (2, 99.0), (3, 98.0), (4, 97.0), (5, 96.0),
            (6, 95.0), (7, 94.0), (11, 93.0), (21, 92.0), (31, 91.0),
            (41, 90.0), (51, 89.0), (61, 88.0), (71, 87.0),
        ]
        diversified = _diversify_candidate_pool(candidates, 7, set(range(1, 21)))
        nums = [n for n, _ in diversified]

        self.assertEqual(len(nums), 7)
        self.assertLessEqual(sum(1 for n in nums if n <= 20), 3)
        self.assertLessEqual(max(nums.count(n) for n in nums), 1)

    def test_diversify_candidate_pool_accepts_repeat_cap(self):
        candidates = [
            (1, 100.0), (2, 99.0), (3, 98.0), (4, 97.0), (5, 96.0),
            (21, 95.0), (31, 94.0), (41, 93.0), (51, 92.0),
        ]
        diversified = _diversify_candidate_pool(
            candidates,
            5,
            set(range(1, 21)),
            max_last_numbers=1,
        )
        nums = [n for n, _ in diversified]

        self.assertEqual(len(nums), 5)
        self.assertLessEqual(sum(1 for n in nums if n <= 20), 1)

    def test_multi_model_voting_uses_broader_diversified_pool(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original = KL8Analyzer._model_rank

        def fake_rank(self, top_n=20, **kwargs):
            return list(range(1, top_n + 1))

        try:
            KL8Analyzer._model_rank = fake_rank
            result = analyzer.multi_model_voting(
                pick_n=7,
                top_n=7,
                feature_weights={'frequency': 1.0},
                model_weights={'rank': 1.0},
            )
        finally:
            KL8Analyzer._model_rank = original

        self.assertTrue(result['diversified'])
        self.assertEqual(result['raw_candidate_count'], 40)
        self.assertEqual(len(result['selected']), 7)
        self.assertLessEqual(sum(1 for n in result['selected'] if n <= 20), 3)

    def test_predict_all_blocks_tiny_history(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(10, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {}

        result = analyzer.predict_all()

        self.assertIn('error', result)
        self.assertEqual(result['data_quality']['min_required'], KL8_MIN_PREDICTION_PERIODS)
        self.assertEqual(result['data_quality']['reason'], 'insufficient_history')

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
                repeat_direction='follow',
                repeat_follow_score=0.92,
                repeat_non_follow_score=0.55,
                pool_diversify=False,
                pool_max_last_numbers=1,
            )
        finally:
            KL8Analyzer.multi_model_voting = original

        self.assertNotIn('error', result)
        self.assertTrue(captured)
        self.assertTrue(all(c['repeat_direction'] == 'follow' for c in captured))
        self.assertTrue(all(c['repeat_follow_score'] == 0.92 for c in captured))
        self.assertTrue(all(c['repeat_non_follow_score'] == 0.55 for c in captured))
        self.assertTrue(all(c['pool_diversify'] is False for c in captured))
        self.assertTrue(all(c['pool_max_last_numbers'] == 1 for c in captured))


if __name__ == '__main__':
    unittest.main()
