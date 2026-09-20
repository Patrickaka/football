from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from scripts.diagnose.compare_kl8_early_consensus import candidates, low_hit_rate, summary
from scripts.backtest.backtest_kl8_early_rounds import live_groups
from src.kl8.analyzer import KL8Analyzer
from src.kl8.exclusion import exclusion_candidates
from src.kl8.records import _strategy_fingerprint
from src.kl8.strategies import resolve_exclusion_strategy


class EarlyConsensusTests(unittest.TestCase):
    def test_equal_scores_remain_equal_and_windows_have_equal_weight(self):
        analyzer = Mock()
        ascending = [(n, float(n)) for n in range(1, 81)]
        descending = [(n, float(81 - n)) for n in range(1, 81)]
        analyzer.build_pool_by_strategy.side_effect = [
            {'candidates': ascending}, {'candidates': descending},
        ]
        result = exclusion_candidates(analyzer, {'exclusion_windows': [50, 100]}, 80)
        self.assertEqual(len(result), 80)
        for _, score in result:
            self.assertAlmostEqual(score, .5)
        analyzer.build_pool_by_strategy.side_effect = None
        analyzer.build_pool_by_strategy.return_value = {'candidates': [(n, 1.) for n in range(80, 0, -1)]}
        self.assertEqual(exclusion_candidates(analyzer, {'exclusion_windows': [100]}, 6),
                         [(n, .5) for n in range(1, 7)])

    def test_invalid_or_incomplete_consensus_fails_explicitly(self):
        analyzer = Mock()
        for windows in ([], [True], [49], [100, 100], '100'):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                exclusion_candidates(analyzer, {'exclusion_windows': windows}, 40)
        analyzer.build_pool_by_strategy.return_value = {'candidates': [(1, 1.)]}
        with self.assertRaises(ValueError):
            exclusion_candidates(analyzer, {'exclusion_windows': [100]}, 40)

    def test_baseline_passes_through_and_does_not_rerank(self):
        analyzer = Mock()
        original = [(8, .9), (1, .3)]
        analyzer.build_pool_by_strategy.return_value = {'candidates': original}
        self.assertIs(exclusion_candidates(analyzer, {}, 40), original)
        analyzer.build_pool_by_strategy.assert_called_once_with({}, pool_size=40)

    def test_only_two_rounds_change_and_configuration_is_replayable(self):
        slate = candidates()
        strategy = slate['consensus_50_100_200']
        before = deepcopy(strategy)
        for count in (6, 12):
            resolved = resolve_exclusion_strategy(strategy, 'select_6', range(count))
            self.assertEqual(resolved['exclusion_windows'], [50, 100, 200])
            self.assertFalse(resolved['is_validated'])
        for count in (0, 18, 24):
            self.assertNotIn('exclusion_windows', resolve_exclusion_strategy(strategy, 'select_6', range(count)))
        self.assertEqual(strategy, before)
        changed = deepcopy(strategy)
        changed['early_exclusion_strategy']['exclusion_windows'][0] = 75
        self.assertNotEqual(_strategy_fingerprint(strategy), _strategy_fingerprint(changed))

    def test_live_linked_chain_preserves_primary_sizes_and_exclusions(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [{'issue': '2026001', 'numbers': list(range(1, 21))}]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}
        slate = candidates()

        def ranking(strategy, pool_size=80):
            order = range(80, 0, -1) if strategy['window_size'] != 100 else range(1, 81)
            return {'candidates': [(n, float(81-i)) for i, n in enumerate(order)][:pool_size]}

        with patch.object(analyzer, 'build_pool_by_strategy', side_effect=ranking):
            baseline = live_groups(analyzer, slate['current'])
            candidate = live_groups(analyzer, slate['consensus_50_100_200'])
        for play, size in (('select_6', 6), ('fu_shi_7', 7)):
            self.assertEqual(baseline[play][0], candidate[play][0])
            self.assertNotEqual(baseline[play][1], candidate[play][1])
            self.assertEqual([len(g) for g in candidate[play]], [size] * 3)
            self.assertEqual(len(set(sum(candidate[play], []))), size * 3)
        for r in (1, 2):
            already_used = set(sum(candidate['fu_shi_7'][:r], []))
            self.assertTrue((set(candidate['select_6'][r]) - already_used)
                            <= set(candidate['fu_shi_7'][r]))

    def test_low_hit_metric_cannot_hide_two_hit_rounds_in_a_union(self):
        row = {'select_6': [6, 2, 3], 'fu_shi_7': [7, 2, 4]}
        self.assertEqual(low_hit_rate(row), .5)
        result = summary([row])
        self.assertEqual(result['select_6'][1]['hit_0_to_2_rate'], 1)
        self.assertEqual(result['select_6'][2]['hit_0_to_2_rate'], 0)
        low = {'select_6': [6, 2, 1], 'fu_shi_7': [7, 2, 2]}
        result = summary([low, low, row, low])
        self.assertEqual(result['select_6_both_early_rounds_low'],
                         {'rate': .75, 'longest_streak': 2})
