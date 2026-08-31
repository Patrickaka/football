import unittest
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from src.domain.numeric.kl8 import pools, scoring

import src.kl8 as kl8_module
from src.kl8 import config as kl8_config
from src.kl8 import records as kl8_records
from src.kl8 import snapshots as kl8_snapshots
from src.kl8 import strategies as kl8_strategies
from src.kl8 import (
    KL8RollingBacktest,
    KL8Analyzer,
    KL8_MIN_PREDICTION_PERIODS,
    _clean_pick_numbers,
    _diversify_candidate_pool,
    _compute_next_issue,
    normalize_record,
    validate_and_activate_strategy,
    _adaptive_repeat_cap,
    _hit_rate_priority_score,
    _hit_rate_priority_thresholds,
    _practical_validation_score,
    _shape_balanced_candidate_pool,
    _shape_profile,
    resolve_play_strategy,
    _simulate_multi_slip_coverage,
)


def _voting_analyzer():
    """只为投票而生的分析器：统计量与期号手工给定，不碰历史加载。"""
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.statistics = {'last_numbers': set(range(1, 21))}
    analyzer.history_data = []
    return analyzer


@contextmanager
def _fake_ranking(rank_fn):
    """把排名换成给定的号码顺序。

    排名本身有黄金文件与专门的用例（`tests/domain/numeric/kl8/test_scoring.py`），
    这里打桩是为了把断言钉在投票与整形上，而不是跟着评分曲线一起漂。
    """
    original = scoring.ensemble_ranking
    scoring.ensemble_ranking = lambda statistics, weights, top_n=20, **kwargs: [
        {'num': num} for num in rank_fn(top_n)]
    try:
        yield
    finally:
        scoring.ensemble_ranking = original


class CandidateVariantGuardTests(unittest.TestCase):
    """两个入口展示的候选形态。

    迁移前这两处各自重写了一遍「每种模式的重号上限该加 1 还是减 1」，还把
    `zone_spread` 整段抄了一份。现在都走 `pools.build_pool`，**这组用例是它们
    唯一的守卫**——之前一条都没有，改错了不会报错，只会让两个入口给出
    不一样的推荐。
    """

    def setUp(self):
        self.analyzer = KL8Analyzer.__new__(KL8Analyzer)
        self.analyzer.history_data = [_record(i) for i in range(1, 61)]
        self.analyzer.using_simulated_data = False
        self.analyzer.history_file = ''
        self.analyzer._data_mtime = 0
        self.analyzer.update_statistics()
        self.candidates = [(n, 0.9 - i * 0.01) for i, n in enumerate(range(1, 41))]

    # 模式名写死在这里，不引用被测常量——引用的话从常量里删掉一种模式，
    # 断言会跟着一起少一项，改坏了照样全绿。
    EXPECTED_VARIANTS = {'high_tier_chase', 'balanced', 'concentrated', 'low_repeat',
                         'repeat_follow', 'zone_spread', 'prize_floor', 'shape_balanced'}
    EXPECTED_RECALC_VARIANTS = ['concentrated', 'balanced', 'repeat_follow',
                                'low_repeat', 'prize_floor', 'zone_spread',
                                'shape_balanced']

    def test_every_named_variant_is_built(self):
        variants = self.analyzer._candidate_variants(self.candidates, 6, 3)
        self.assertEqual(set(variants), self.EXPECTED_VARIANTS)

    def test_each_variant_has_the_requested_pick_size(self):
        for label, nums in self.analyzer._candidate_variants(self.candidates, 6, 3).items():
            with self.subTest(variant=label):
                self.assertEqual(len(nums), 6)
                self.assertEqual(sorted(nums), nums)

    def test_variants_match_the_shared_pool_builders(self):
        """与 `pools.build_pool` 逐个对齐——两处走偏了不会报错。"""
        last = self.analyzer.statistics.get('last_numbers', set())
        variants = self.analyzer._candidate_variants(self.candidates, 6, 3)
        for label in sorted(self.EXPECTED_VARIANTS):
            with self.subTest(variant=label):
                expected = sorted(num for num, _ in pools.build_pool(
                    label, self.candidates, 6, last, 3))
                self.assertEqual(variants[label], expected)

    def test_concentrated_variant_is_the_plain_top_of_the_pool(self):
        variants = self.analyzer._candidate_variants(self.candidates, 6, 3)
        self.assertEqual(variants['concentrated'],
                         sorted(num for num, _ in self.candidates[:6]))

    def test_low_repeat_keeps_no_more_repeats_than_repeat_follow(self):
        """这两个接反了，推荐会系统性偏向或偏离上期的号，而且不报错。"""
        last = set(self.analyzer.statistics.get('last_numbers', set()))
        variants = self.analyzer._candidate_variants(self.candidates, 8, 3)
        low = sum(1 for n in variants['low_repeat'] if n in last)
        follow = sum(1 for n in variants['repeat_follow'] if n in last)
        self.assertLessEqual(low, follow)

    def test_empty_candidates_yield_no_variants(self):
        self.assertEqual(self.analyzer._candidate_variants([], 6, 3), {})
        self.assertEqual(self.analyzer._candidate_variants(self.candidates, 0, 3), {})

    def test_exclude_recalculation_tries_every_named_variant(self):
        seen = []
        original = pools.build_pool
        try:
            pools.build_pool = lambda mode, *a, **k: (seen.append(mode)
                                                      or original(mode, *a, **k))
            self.analyzer._best_exclude_recalculation_pool(self.candidates, 6, 3)
        finally:
            pools.build_pool = original
        self.assertEqual(seen, self.EXPECTED_RECALC_VARIANTS)

    def test_exclude_recalculation_returns_a_full_pick(self):
        pool, quality = self.analyzer._best_exclude_recalculation_pool(
            self.candidates, 6, 3)
        self.assertEqual(len(pool), 6)
        self.assertIn(quality.get('selection_mode'), self.EXPECTED_RECALC_VARIANTS)

    def test_exclude_recalculation_honours_an_explicit_mode(self):
        pool, quality = self.analyzer._best_exclude_recalculation_pool(
            self.candidates, 6, 3, selection_mode='zone_spread')
        self.assertEqual(quality.get('requested_selection_mode'), 'zone_spread')
        self.assertEqual(len(pool), 6)


def _record(issue: int):
    base = ((issue - 1) % 60) + 1
    nums = [((base + i - 1) % 80) + 1 for i in range(20)]
    return {'issue': str(2026000 + issue), 'numbers': sorted(set(nums)), 'date': '2026-01-01'}


class KL8PredictionGuardTests(unittest.TestCase):
    def setUp(self):
        self._recalculation_tmp = tempfile.TemporaryDirectory()
        self._original_recalculation_dir = kl8_config.KL8_RECALCULATION_DIR
        kl8_config.KL8_RECALCULATION_DIR = self._recalculation_tmp.name

    def tearDown(self):
        kl8_config.KL8_RECALCULATION_DIR = self._original_recalculation_dir
        self._recalculation_tmp.cleanup()

    def test_next_transition_uses_older_to_newer_draws_with_shrinkage(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.using_simulated_data = False
        analyzer.history_data = []
        for idx in range(60):
            if idx % 2 == 0:
                numbers = list(range(61, 81))  # follows the draw containing 1
            else:
                numbers = list(range(1, 21))
            analyzer.history_data.append({'issue': str(206000 - idx), 'numbers': numbers})
        analyzer.update_statistics()

        probabilities = analyzer.statistics['next_transition_probability']
        self.assertGreater(probabilities[1], probabilities[40])
        self.assertGreater(probabilities[1], 0.25)
        score = analyzer._calculate_feature_score(1)['next_transition']
        self.assertGreater(score, 0.50)

    def test_unvalidated_numbers_use_dynamic_reference_by_default(self):
        self.assertFalse(kl8_config.VERIFY_ONLY_MODE)
        select5 = resolve_play_strategy('select_5')
        select6 = resolve_play_strategy('select_6')
        self.assertEqual(select5['baseline_type'], 'adaptive_pattern_reference')
        self.assertEqual(select6['baseline_type'], 'adaptive_pattern_reference')
        self.assertNotEqual(select5['feature_weights'], {'seeded_random': 1.0})
        self.assertGreater(select5['feature_weights']['trend'], 0)
        self.assertGreater(select6['feature_weights']['road_residual'], 0)

    def test_multi_slip_coverage_accounts_for_identical_ticket_overlap(self):
        slips = [[1, 2, 3, 4, 5, 6]] * 8
        profile = _simulate_multi_slip_coverage(slips, simulations=20000, seed_key='test')

        # Eight identical tickets must behave like one ticket, not eight
        # independent tickets. The fair P(6-number ticket hits >=4) is ~3.18%.
        self.assertAlmostEqual(profile['at_least_one_ge4'], 0.0318, delta=0.006)
        self.assertEqual(profile['unique_number_count'], 6)
        self.assertEqual(profile['max_pair_overlap'], 6)

    def test_reference_select_5_and_6_use_play_specific_ranking(self):
        select5 = resolve_play_strategy('select_5', allow_reference=True)
        select6 = resolve_play_strategy('select_6', allow_reference=True)

        self.assertEqual(select5['final_selection_mode'], 'concentrated')
        self.assertEqual(select5['window_size'], 100)
        self.assertEqual(select6['final_selection_mode'], 'concentrated')
        self.assertEqual(select6['window_size'], 100)
        self.assertEqual(select6['chain_objective'], 'primary_accuracy_then_early_exclusion')
        self.assertEqual(select6['chain_audit_rounds'], 5)
        for strategy in (select5, select6):
            self.assertFalse(strategy['pool_diversify'])
            self.assertGreater(strategy['feature_weights']['frequency'], 0.0)
            self.assertGreater(strategy['feature_weights']['pair_cooccurrence'], 0.0)
            self.assertEqual(strategy['baseline_type'], 'adaptive_pattern_reference')
            self.assertFalse(strategy['is_validated'])

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

    def test_adaptive_repeat_cap_relaxes_when_recent_overlap_is_high(self):
        history = [_record(i) for i in range(40, 0, -1)]

        self.assertEqual(_adaptive_repeat_cap(history, 5), 3)
        self.assertEqual(_adaptive_repeat_cap(history, 7), 4)

    def test_build_pool_by_strategy_uses_adaptive_repeat_cap(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(40, 0, -1)]
        analyzer.history_file = ''
        analyzer._data_mtime = 0
        analyzer.using_simulated_data = False

        captured = {}

        def fake_voting(**kwargs):
            captured.update(kwargs)
            return {
                'selected': list(range(1, 8)),
                'candidates': [(n, float(80 - n)) for n in range(1, 21)],
                'votes': {},
            }

        original_build = KL8Analyzer._build_window_analyzer
        try:
            KL8Analyzer._build_window_analyzer = lambda self, window_size: type(
                'TempAnalyzer',
                (),
                {
                    'history_data': analyzer.history_data[:window_size],
                    'multi_model_voting': staticmethod(fake_voting),
                },
            )()
            analyzer.build_pool_by_strategy(
                {
                    'feature_weights': {'frequency': 1.0},
                    'model_weights': {'rank': 1.0},
                    'window_size': 40,
                    'repeat_direction': 'neutral',
                },
                pool_size=7,
            )
        finally:
            KL8Analyzer._build_window_analyzer = original_build

        self.assertEqual(captured['pool_max_last_numbers'], 4)

    def test_recalculate_play_excluding_removes_current_numbers(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original_build = KL8Analyzer.build_pool_by_strategy
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, min(pool_size, 40) + 1)),
                'candidates': [(n, float(100 - n)) for n in range(1, 41)],
                'votes': {},
            }
            result = analyzer.recalculate_play_excluding('select_5', [1, 2, 3, 4, 5])
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertNotIn('error', result)
        self.assertEqual(len(result['numbers']), 5)
        self.assertEqual(result['numbers'], sorted(result['numbers']))
        self.assertFalse(set(result['numbers']) & {1, 2, 3, 4, 5})
        self.assertEqual(result['excluded_numbers'], [1, 2, 3, 4, 5])

    def test_select6_recalculation_rounds_are_persisted_and_deduplicated(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original_build = KL8Analyzer.build_pool_by_strategy
        original_dir = kl8_config.KL8_RECALCULATION_DIR
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                kl8_config.KL8_RECALCULATION_DIR = tmp
                kl8_config.VERIFY_ONLY_MODE = False
                KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                    'selected': list(range(1, 51)),
                    'candidates': [(n, float(100 - n)) for n in range(1, 51)],
                    'votes': {},
                }
                first = analyzer.recalculate_play_excluding('select_6', [1, 2, 3, 4, 5, 6])
                duplicate = analyzer.recalculate_play_excluding('select_6', [1, 2, 3, 4, 5, 6])
                second = analyzer.recalculate_play_excluding(
                    'select_6',
                    [1, 2, 3, 4, 5, 6] + first['numbers'],
                )
                stored = kl8_module.list_exclude_recalculations()
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.KL8_RECALCULATION_DIR = original_dir
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(first['recalculation_record']['round'], 1)
        self.assertEqual(duplicate['recalculation_record']['record_id'], first['recalculation_record']['record_id'])
        self.assertEqual(second['recalculation_record']['round'], 2)
        self.assertEqual(len(stored), 2)

    def test_select6_recalculation_chain_runs_until_candidates_are_exhausted(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}
        original_build = KL8Analyzer.build_pool_by_strategy
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, 19)),
                'candidates': [(n, float(100 - n)) for n in range(1, 19)],
                'votes': {},
            }
            chain = analyzer.generate_exclude_recalculation_chain(
                'select_6',
                [1, 2, 3, 4, 5, 6],
            )
            stored = kl8_module.list_exclude_recalculations()
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(chain['generated_rounds'], 2)
        self.assertTrue(chain['exhausted'])
        self.assertEqual(chain['terminal']['remaining_count'], 0)
        self.assertEqual(len(stored), 3)
        self.assertEqual([row['status'] for row in reversed(stored)], ['generated', 'generated', 'exhausted'])

    def test_automatic_select6_chain_matches_manual_click_sequence(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}
        initial = [1, 2, 3, 4, 5, 6]
        candidates = [(n, float(100 - n)) for n in range(1, 31)]

        original_build = KL8Analyzer.build_pool_by_strategy
        original_dir = kl8_config.KL8_RECALCULATION_DIR
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': [n for n, _ in candidates],
                'candidates': candidates,
                'votes': {},
            }
            with tempfile.TemporaryDirectory() as automatic_dir:
                kl8_config.KL8_RECALCULATION_DIR = automatic_dir
                chain = analyzer.generate_exclude_recalculation_chain(
                    'select_6',
                    initial,
                    source_snapshot_id='same-snapshot',
                    source_version='same-version',
                )
                automatic_numbers = [row['numbers'] for row in chain['records']]
                self.assertTrue(all(
                    row['generation_mode'] == 'automatic'
                    for row in chain['records']
                ))

            with tempfile.TemporaryDirectory() as manual_dir:
                kl8_config.KL8_RECALCULATION_DIR = manual_dir
                excluded = set(initial)
                manual_numbers = []
                current = initial
                while current:
                    result = analyzer.recalculate_play_excluding(
                        'select_6',
                        sorted(excluded),
                        record_context={
                            'source_snapshot_id': 'same-snapshot',
                            'source_version': 'same-version',
                            'generation_mode': 'manual',
                            'initial_numbers': initial,
                        },
                    )
                    if result.get('error'):
                        break
                    current = result['numbers']
                    manual_numbers.append(current)
                    excluded.update(current)
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.KL8_RECALCULATION_DIR = original_dir
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(automatic_numbers, manual_numbers)

    def test_manual_replay_reuses_automatic_record_without_duplication(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}
        candidates = [(n, float(100 - n)) for n in range(1, 41)]

        original_build = KL8Analyzer.build_pool_by_strategy
        original_dir = kl8_config.KL8_RECALCULATION_DIR
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                kl8_config.KL8_RECALCULATION_DIR = tmp
                kl8_config.VERIFY_ONLY_MODE = False
                KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                    'selected': [n for n, _ in candidates],
                    'candidates': candidates,
                    'votes': {},
                }
                automatic = analyzer.generate_exclude_recalculation_chain(
                    'select_6',
                    [1, 2, 3, 4, 5, 6],
                    max_rounds=1,
                    source_snapshot_id='dedupe-snapshot',
                )['records'][0]
                manual = analyzer.recalculate_play_excluding(
                    'select_6',
                    [1, 2, 3, 4, 5, 6],
                    record_context={
                        'source_snapshot_id': 'dedupe-snapshot',
                        'generation_mode': 'manual',
                        'initial_numbers': [1, 2, 3, 4, 5, 6],
                    },
                )['recalculation_record']
                stored = kl8_module.list_exclude_recalculations()
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.KL8_RECALCULATION_DIR = original_dir
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(automatic['record_id'], manual['record_id'])
        self.assertEqual(automatic['numbers'], manual['numbers'])
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]['generation_mode'], 'automatic')

    def test_fushi7_exclude_recalculation_is_saved_with_seven_numbers(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}
        candidates = [(n, float(100 - n)) for n in range(1, 41)]

        original_build = KL8Analyzer.build_pool_by_strategy
        original_dir = kl8_config.KL8_RECALCULATION_DIR
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                kl8_config.KL8_RECALCULATION_DIR = temp_dir
                kl8_config.VERIFY_ONLY_MODE = False
                KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                    'selected': [n for n, _ in candidates],
                    'candidates': candidates,
                    'votes': {},
                }
                result = analyzer.recalculate_play_excluding(
                    'fu_shi_7',
                    [1, 2, 3],
                    record_context={
                        'source_snapshot_id': 'fushi7-snapshot',
                        'generation_mode': 'manual',
                        'initial_numbers': [1, 2, 3, 4, 5, 6, 7],
                    },
                )
                stored = kl8_module.list_exclude_recalculations()
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.KL8_RECALCULATION_DIR = original_dir
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(len(result['top7_numbers']), 7)
        self.assertEqual(result['total_combinations'], 21)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]['play_type'], 'fu_shi_7')
        self.assertEqual(stored[0]['numbers'], result['top7_numbers'])
        self.assertEqual(stored[0]['source_snapshot_id'], 'fushi7-snapshot')

    def test_fushi7_recalculation_chain_runs_until_candidates_are_exhausted(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set()}

        original_build = KL8Analyzer.build_pool_by_strategy
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, 22)),
                'candidates': [(n, float(100 - n)) for n in range(1, 22)],
                'votes': {},
            }
            chain = analyzer.generate_exclude_recalculation_chain(
                'fu_shi_7',
                [1, 2, 3, 4, 5, 6, 7],
                source_snapshot_id='fushi7-auto-snapshot',
            )
            stored = kl8_module.list_exclude_recalculations()
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(chain['generated_rounds'], 2)
        self.assertTrue(chain['exhausted'])
        self.assertEqual(chain['terminal']['remaining_count'], 0)
        self.assertEqual(len(stored), 3)
        self.assertTrue(all(row['play_type'] == 'fu_shi_7' for row in stored))
        self.assertTrue(all(
            len(row['numbers']) == 7
            for row in stored
            if row['status'] == 'generated'
        ))

    def test_recalculation_identity_is_scoped_to_source_snapshot(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        result = {
            'play_type': 'select_6',
            'excluded_numbers': [1, 2, 3, 4, 5, 6],
            'numbers': [7, 8, 9, 10, 11, 12],
        }

        first = analyzer._save_exclude_recalculation(
            result,
            record_context={'source_snapshot_id': 'snapshot-a', 'initial_numbers': [1, 2, 3, 4, 5, 6]},
        )
        second = analyzer._save_exclude_recalculation(
            result,
            record_context={'source_snapshot_id': 'snapshot-b', 'initial_numbers': [2, 3, 4, 5, 6, 7]},
        )

        self.assertNotEqual(first['record_id'], second['record_id'])
        self.assertEqual(first['source_snapshot_id'], 'snapshot-a')
        self.assertEqual(second['source_snapshot_id'], 'snapshot-b')

    def test_recalculate_play_excluding_supports_select_10(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original_build = KL8Analyzer.build_pool_by_strategy
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            unsorted_candidates = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11] + list(range(21, 51))
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, min(pool_size, 50) + 1)),
                'candidates': [(n, float(100 - i)) for i, n in enumerate(unsorted_candidates)],
                'votes': {},
            }
            result = analyzer.recalculate_play_excluding('select_10', list(range(1, 11)))
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertNotIn('error', result)
        self.assertEqual(len(result['numbers']), 10)
        self.assertEqual(result['numbers'], sorted(result['numbers']))
        self.assertFalse(set(result['numbers']) & set(range(1, 11)))
        self.assertEqual(result['excluded_numbers'], list(range(1, 11)))
        self.assertNotEqual(result['quality']['selection_mode'], 'low_repeat')
        self.assertEqual(result['quality']['requested_selection_mode'], 'concentrated')
        self.assertEqual(result['quality']['selection_mode'], 'concentrated')
        self.assertEqual(result['numbers'], sorted(unsorted_candidates[:10]))

    def test_recalculate_play_excluding_respects_strategy_selection_mode(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original_build = KL8Analyzer.build_pool_by_strategy
        original_resolve = kl8_strategies.resolve_play_strategy
        try:
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, min(pool_size, 50) + 1)),
                'candidates': [(n, float(100 - n)) for n in range(1, 51)],
                'votes': {},
            }
            kl8_strategies.resolve_play_strategy = lambda play_type: {
                'strategy_id': f'{play_type}_forced_prize_floor',
                'feature_weights': {'frequency': 1.0},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
                'final_selection_mode': 'prize_floor',
                'prediction_mode': 'reference_unvalidated',
                'is_validated': False,
            }
            result = analyzer.recalculate_play_excluding('select_10', list(range(1, 11)))
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_strategies.resolve_play_strategy = original_resolve

        self.assertNotIn('error', result)
        self.assertEqual(result['quality']['requested_selection_mode'], 'prize_floor')
        self.assertEqual(result['quality']['selection_mode'], 'prize_floor')

    def test_multi_model_voting_uses_broader_diversified_pool(self):
        # 打桩打在排名上：排名有自己的黄金文件与用例，这里要测的是
        # 「拿到一份排名之后」投票和整形做了什么。
        with _fake_ranking(lambda top_n: list(range(1, top_n + 1))):
            result = _voting_analyzer().multi_model_voting(
                pick_n=7,
                top_n=7,
                feature_weights={'frequency': 1.0},
                model_weights={'rank': 1.0},
            )

        self.assertTrue(result['diversified'])
        self.assertEqual(result['raw_candidate_count'], 40)
        self.assertEqual(len(result['selected']), 7)
        self.assertLessEqual(sum(1 for n in result['selected'] if n <= 20), 3)

    def test_multi_model_voting_raw_rank_primary_keeps_true_top_numbers(self):
        ranked = [41, 42, 43, 44, 45] + list(range(1, 41))

        with _fake_ranking(lambda top_n: ranked[:top_n]):
            result = _voting_analyzer().multi_model_voting(
                pick_n=5,
                top_n=20,
                feature_weights={'frequency': 1.0},
                model_weights={'rank': 1.0},
                pool_diversify=False,
                final_selection_mode='concentrated',
            )

        self.assertFalse(result['diversified'])
        self.assertEqual(result['selected'], ranked[:5])
        self.assertEqual([num for num, _ in result['candidates'][:5]], ranked[:5])

    def test_shape_balanced_candidate_pool_controls_zone_and_odd_even(self):
        candidates = [
            (1, 100.0), (3, 99.0), (5, 98.0), (7, 97.0), (9, 96.0),
            (22, 95.0), (24, 94.0), (26, 93.0),
            (41, 92.0), (43, 91.0), (62, 90.0), (64, 89.0),
        ]

        pool = _shape_balanced_candidate_pool(
            candidates,
            8,
            last_numbers={1, 3, 5, 7, 9},
            max_last_numbers=2,
        )
        nums = [num for num, _ in pool]
        profile = _shape_profile(nums, {1, 3, 5, 7, 9})

        self.assertEqual(len(nums), 8)
        self.assertLessEqual(max(profile['zone20']), 3)
        self.assertLessEqual(abs(profile['odd_even']['odd'] - profile['odd_even']['even']), 2)
        self.assertLessEqual(profile['repeat_from_last'], 3)

    def test_predict_all_blocks_tiny_history(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(10, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {}

        result = analyzer.predict_all()

        self.assertIn('error', result)
        self.assertEqual(result['data_quality']['min_required'], KL8_MIN_PREDICTION_PERIODS)
        self.assertEqual(result['data_quality']['reason'], 'insufficient_history')

    def test_strategy_activation_rejects_invalid_repeat_configuration(self):
        result = validate_and_activate_strategy(
            'select_5',
            {'frequency': 1.0},
            {'rank': 1.0},
            50,
            repeat_direction='sideways',
        )
        self.assertIn('repeat_direction', result['error'])

        result = validate_and_activate_strategy(
            'select_5',
            {'frequency': 1.0},
            {'rank': 1.0},
            50,
            pool_max_last_numbers=-1,
        )
        self.assertIn('pool_max_last_numbers', result['error'])

        result = validate_and_activate_strategy(
            'select_5',
            {'frequency': 1.0},
            {'rank': 1.0},
            50,
            pool_max_last_numbers=6,
        )
        self.assertIn('pick count', result['error'])

    def test_reference_plays_use_distinct_adaptive_strategies(self):
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            select5 = kl8_strategies.resolve_play_strategy('select_5')
            select6 = kl8_strategies.resolve_play_strategy('select_6')
            select10 = kl8_strategies.resolve_play_strategy('select_10')
        finally:
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(select5['final_selection_mode'], 'concentrated')
        self.assertEqual(select6['final_selection_mode'], 'concentrated')
        self.assertEqual(select10['final_selection_mode'], 'concentrated')
        self.assertFalse(select5['pool_diversify'])
        self.assertFalse(select6['pool_diversify'])
        self.assertFalse(select10['pool_diversify'])
        self.assertEqual(select5['strategy_id'], 'select_5_ref_transition_repeat_v3')
        self.assertEqual(select6['strategy_id'], 'select_6_ref_transition_primary_v5')
        self.assertEqual(select10['strategy_id'], 'select_10_ref_trend100_shape_balanced')
        self.assertNotEqual(select5['feature_weights'], select6['feature_weights'])
        self.assertEqual(select5['target_hits'], 4)
        self.assertEqual(select6['target_hits'], 5)

    def test_select6_hit_rate_priority_targets_prize_floor(self):
        self.assertEqual(_hit_rate_priority_thresholds('select_5'), ['>=4', '>=3'])
        self.assertEqual(_hit_rate_priority_thresholds('select_6'), ['>=5', '>=4'])

        select6_score, select6_detail = _hit_rate_priority_score(
            {
                'probabilities': {'>=2': 0.9, '>=3': 0.22, '>=4': 0.05},
                'theoretical_probs': {'>=2': 0.47, '>=3': 0.17, '>=4': 0.04},
            },
            'select_6',
        )

        self.assertGreater(select6_score, 0)
        self.assertNotIn('>=2', select6_detail)
        self.assertIn('>=5', select6_detail)
        self.assertIn('>=4', select6_detail)

    def test_predict_all_includes_select_8_9_10_and_fushi_10_11(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.history_file = ''
        analyzer._data_mtime = 0
        analyzer.statistics = {}
        analyzer.update_statistics()

        original_save = KL8Analyzer._save_prediction_snapshot
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            KL8Analyzer._save_prediction_snapshot = lambda self, prediction_result: None
            result = analyzer.predict_all()
        finally:
            KL8Analyzer._save_prediction_snapshot = original_save
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        for pick in range(3, 11):
            self.assertNotIn('multi_slips', result[f'select_{pick}'])
        self.assertEqual(
            result['select_5']['strategy_id'],
            'select_5_ref_transition_repeat_v3',
        )
        self.assertEqual(result['select_5']['final_selection_mode'], 'concentrated')
        self.assertEqual(result['select_5']['baseline_type'], 'adaptive_pattern_reference')

        for pick in [8, 9, 10]:
            key = f'select_{pick}'
            self.assertIn(key, result)
            self.assertEqual(result[key]['pick'], pick)
            self.assertEqual(len(result[key]['numbers']), pick)
            self.assertEqual(result[key]['numbers'], sorted(result[key]['numbers']))

        for key in [f'select_{pick}' for pick in range(3, 11)]:
            expected_mode = 'concentrated'
            self.assertEqual(
                result['resolved_strategies'][key]['final_selection_mode'],
                expected_mode,
            )
            self.assertFalse(result['resolved_strategies'][key]['pool_diversify'])
            self.assertNotIn('variants', result[key])

        self.assertEqual(result['resolved_strategies']['select_5']['pool_max_last_numbers'], 2)
        self.assertEqual(result['resolved_strategies']['select_6']['pool_max_last_numbers'], 3)
        self.assertEqual(
            result['resolved_strategies']['select_6']['chain_objective'],
            'primary_accuracy_then_early_exclusion',
        )
        self.assertEqual(result['resolved_strategies']['select_6']['chain_audit_rounds'], 5)
        last_numbers = set(analyzer.history_data[0]['numbers'])
        self.assertIn('repeat_profile', result['select_5'])
        self.assertGreaterEqual(result['select_5']['repeat_profile']['sample_size'], 1)
        self.assertFalse(result['select_5']['repeat_profile']['constraint_applied'])
        self.assertFalse(result['select_6']['repeat_profile']['constraint_applied'])
        self.assertEqual(result['select_5']['prize_hit_thresholds'], ['>=4', '>=3'])
        self.assertEqual(result['select_6']['prize_hit_thresholds'], ['>=5', '>=4'])
        self.assertEqual(result['select_6']['hit_rate_priority_thresholds'], ['>=5', '>=4'])
        self.assertEqual(result['select_5']['accuracy_profile']['expected_hits_random'], 1.25)
        self.assertEqual(result['select_5']['accuracy_profile']['key_thresholds'], ['>=4', '>=3'])
        self.assertEqual(result['select_5']['accuracy_profile']['target_hits'], 4)
        self.assertEqual(result['select_6']['accuracy_profile']['expected_hits_random'], 1.5)
        self.assertEqual(result['select_6']['accuracy_profile']['key_thresholds'], ['>=5', '>=4'])
        self.assertEqual(result['select_6']['accuracy_profile']['target_hits'], 5)
        self.assertEqual(result['select_6']['accuracy_profile']['selected_mode'], 'concentrated')
        self.assertNotIn('variants', result['select_6'])
        self.assertEqual(
            result['resolved_strategies']['select_10']['final_selection_mode'],
            'concentrated',
        )
        self.assertIsNone(result['resolved_strategies']['select_10']['pool_max_last_numbers'])

        self.assertEqual(
            result['resolved_strategies']['fu_shi_7']['final_selection_mode'],
            'concentrated',
        )
        self.assertNotIn('variants', result['fu_shi_7'])
        self.assertEqual(len(result['fu_shi_7']['top7_numbers']), 7)
        self.assertEqual(result['fu_shi_7']['pool_size'], 7)
        self.assertEqual(result['fu_shi_7']['total_combinations'], 21)
        self.assertEqual(result['fu_shi_7']['prize_hit_thresholds'], ['>=3'])
        self.assertEqual(result['resolved_strategies']['fu_shi_7']['pool_max_last_numbers'], 4)

        self.assertIn('fu_shi_10_11', result)
        self.assertEqual(
            result['resolved_strategies']['fu_shi_10_11']['final_selection_mode'],
            'concentrated',
        )
        self.assertEqual(len(result['fu_shi_10_11']['top11_numbers']), 11)
        self.assertEqual(
            result['fu_shi_10_11']['top11_numbers'],
            sorted(result['fu_shi_10_11']['top11_numbers']),
        )
        self.assertEqual(result['fu_shi_10_11']['combo_pick'], 10)
        self.assertEqual(result['fu_shi_10_11']['pool_size'], 11)
        self.assertEqual(result['fu_shi_10_11']['total_combinations'], 11)

    def test_predict_all_automatically_generates_select6_and_fushi7_chains(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.history_file = ''
        analyzer._data_mtime = 0
        analyzer.statistics = {}
        analyzer.update_statistics()
        captured = []

        original_save = KL8Analyzer._save_prediction_snapshot
        original_chain = KL8Analyzer.generate_exclude_recalculation_chain
        original_verify_only = kl8_config.VERIFY_ONLY_MODE
        try:
            kl8_config.VERIFY_ONLY_MODE = False
            KL8Analyzer._save_prediction_snapshot = (
                lambda self, prediction_result: 'snapshot_auto-chain-id.json'
            )

            def fake_chain(self, play_type, initial_numbers, **kwargs):
                captured.append({
                    'play_type': play_type,
                    'initial_numbers': list(initial_numbers),
                    **kwargs,
                })
                return {
                    'play_type': play_type,
                    'generation_mode': 'automatic',
                    'generated_rounds': 12,
                }

            KL8Analyzer.generate_exclude_recalculation_chain = fake_chain
            result = analyzer.predict_all()
        finally:
            KL8Analyzer._save_prediction_snapshot = original_save
            KL8Analyzer.generate_exclude_recalculation_chain = original_chain
            kl8_config.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual([item['play_type'] for item in captured], ['select_6', 'fu_shi_7'])
        self.assertEqual(captured[0]['initial_numbers'], result['select_6']['numbers'])
        self.assertEqual(captured[1]['initial_numbers'], result['fu_shi_7']['top7_numbers'])
        self.assertTrue(all(
            item['source_snapshot_id'] == 'auto-chain-id'
            for item in captured
        ))
        self.assertTrue(all(
            item['source_version'] == kl8_module.KL8_PREDICTOR_VERSION
            for item in captured
        ))
        self.assertEqual(
            result['select_6_recalculation_chain']['generation_mode'],
            'automatic',
        )
        self.assertEqual(
            result['fu_shi_7_recalculation_chain']['generation_mode'],
            'automatic',
        )

    def test_snapshot_records_select6_and_exactly_seven_fushi_numbers(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        select6 = [1, 2, 3, 4, 5, 6]
        fushi7 = [11, 12, 13, 14, 15, 16, 17]

        with tempfile.TemporaryDirectory() as temp_dir:
            original_snapshot_dir = kl8_config.KL8_SNAPSHOT_DIR
            try:
                kl8_config.KL8_SNAPSHOT_DIR = temp_dir
                filename = analyzer._save_prediction_snapshot({
                    'resolved_strategies': {},
                    'select_6': {'numbers': select6},
                    'fu_shi_7': {'top7_numbers': fushi7},
                })
                saved = json.loads(
                    (Path(temp_dir) / filename).read_text(encoding='utf-8')
                )
            finally:
                kl8_config.KL8_SNAPSHOT_DIR = original_snapshot_dir

        self.assertEqual(saved['select_6'], select6)
        self.assertEqual(saved['fu_shi_7'], fushi7)
        self.assertEqual(len(saved['fu_shi_7']), 7)

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

    def test_permutation_passes_pool_configuration_to_voting(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(90, 0, -1)]
        analyzer.using_simulated_data = False

        backtest = KL8RollingBacktest(analyzer)
        captured = []

        def fake_backtest(*args, **kwargs):
            return {'select_5': {'lift': 0.1, 'mean_hits': 1.4}}

        original_voting = KL8Analyzer.multi_model_voting

        def fake_voting(self, **kwargs):
            captured.append(kwargs)
            return {
                'selected': list(range(1, 21)),
                'candidates': [(n, float(21 - n)) for n in range(1, 21)],
                'votes': {},
            }

        try:
            backtest._rolling_backtest_parametric = fake_backtest
            KL8Analyzer.multi_model_voting = fake_voting
            result = backtest._permutation_test(
                {'frequency': 1.0},
                {'rank': 1.0},
                start_idx=55,
                end_idx=70,
                pick_n=5,
                n_permutations=1,
                window_size=50,
                repeat_direction='follow',
                repeat_follow_score=0.91,
                repeat_non_follow_score=0.54,
                pool_diversify=False,
                pool_max_last_numbers=2,
            )
        finally:
            KL8Analyzer.multi_model_voting = original_voting

        self.assertNotIn('error', result)
        self.assertTrue(captured)
        self.assertTrue(all(c['repeat_direction'] == 'follow' for c in captured))
        self.assertTrue(all(c['repeat_follow_score'] == 0.91 for c in captured))
        self.assertTrue(all(c['repeat_non_follow_score'] == 0.54 for c in captured))
        self.assertTrue(all(c['pool_diversify'] is False for c in captured))
        self.assertTrue(all(c['pool_max_last_numbers'] == 2 for c in captured))

    def test_parameter_search_ranks_best_candidate_per_play(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(820, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        backtest._build_parameter_search_candidates = lambda max_candidates=80: {
            'candidate_a': {
                'strategy_id': 'candidate_a',
                'feature_weights': {'marker': 'a'},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
            },
            'candidate_b': {
                'strategy_id': 'candidate_b',
                'feature_weights': {'marker': 'b'},
                'model_weights': {'rank': 1.0},
                'window_size': 100,
            },
        }

        def fake_rolling(feature_weights, model_weights, **kwargs):
            marker = feature_weights['marker']
            is_final = kwargs['start_idx'] >= 600
            if marker == 'a':
                select_lift = 0.20 if not is_final else 0.04
                fushi_mean = 3.1 if not is_final else 2.9
            else:
                select_lift = 0.10 if not is_final else 0.30
                fushi_mean = 3.6 if not is_final else 3.4
            return {
                'select_5': {
                    'lift': select_lift,
                    'mean_hits': 1.25,
                    'expected_random': 1.25,
                    'profit_roi': -0.2,
                    'random_profit_roi': -0.3,
                    'return_multiple': 0.8,
                    'n_tests': 100,
                },
                'fu_shi_10_11': {
                    'pool_mean_hits': fushi_mean,
                    'pool_expected_random': 2.75,
                    'profit_roi': -0.1,
                    'random_profit_roi': -0.4,
                    'return_multiple': 0.9,
                    'n_tests': 100,
                },
            }

        backtest._rolling_backtest_parametric = fake_rolling

        result = backtest.run_parameter_search(
            play_types=['select_5', 'fu_shi_10_11'],
            max_candidates=2,
            top_n=2,
        )

        self.assertNotIn('error', result)
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(result['best_by_play']['select_5']['candidate'], 'candidate_a')
        self.assertEqual(result['best_by_play']['fu_shi_10_11']['candidate'], 'candidate_b')
        self.assertEqual(len(result['rankings']['select_5']), 2)

    def test_parameter_search_prioritizes_key_hit_rates(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(820, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        backtest._build_parameter_search_candidates = lambda max_candidates=80: {
            'mean_lift_only': {
                'strategy_id': 'mean_lift_only',
                'feature_weights': {'marker': 'mean'},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
            },
            'hit_rate_first': {
                'strategy_id': 'hit_rate_first',
                'feature_weights': {'marker': 'hit'},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
            },
        }

        def fake_rolling(feature_weights, model_weights, **kwargs):
            marker = feature_weights['marker']
            if marker == 'hit':
                return {
                    'select_5': {
                        'lift': 0.05,
                        'mean_hits': 1.31,
                        'expected_random': 1.25,
                        'probabilities': {'>=2': 0.35, '>=3': 0.08},
                        'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                        'profit_roi': -0.2,
                        'random_profit_roi': -0.3,
                        'return_multiple': 0.8,
                        'n_tests': 100,
                    },
                }
            return {
                'select_5': {
                    'lift': 0.10,
                    'mean_hits': 1.38,
                    'expected_random': 1.25,
                    'probabilities': {'>=2': 0.31, '>=3': 0.07},
                    'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                    'profit_roi': -0.2,
                    'random_profit_roi': -0.3,
                    'return_multiple': 0.8,
                    'n_tests': 100,
                },
            }

        backtest._rolling_backtest_parametric = fake_rolling

        result = backtest.run_parameter_search(
            play_types=['select_5'],
            max_candidates=2,
            top_n=2,
        )

        self.assertNotIn('error', result)
        best = result['best_by_play']['select_5']
        self.assertEqual(best['candidate'], 'hit_rate_first')
        self.assertGreater(best['validation_hit_rate_score'], 0)

    def test_practical_score_prioritizes_prize_thresholds_over_mean_lift(self):
        hit_score, hit_detail = _practical_validation_score(
            {
                'lift': 0.03,
                'probabilities': {'>=2': 0.36, '>=3': 0.09},
                'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                'profit_roi': -0.2,
                'random_profit_roi': -0.3,
                'return_multiple': 0.8,
            },
            'select_5',
        )
        mean_score, mean_detail = _practical_validation_score(
            {
                'lift': 0.12,
                'probabilities': {'>=2': 0.30, '>=3': 0.07},
                'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                'profit_roi': -0.2,
                'random_profit_roi': -0.3,
                'return_multiple': 0.8,
            },
            'select_5',
        )

        self.assertGreater(hit_score, mean_score)
        self.assertGreater(hit_detail['hit_rate_score'], mean_detail['hit_rate_score'])

    def test_seeded_random_validation_candidates_are_present(self):
        candidates = kl8_module.VALIDATION_CANDIDATES

        self.assertIn('random_shape_50', candidates)
        self.assertIn('random_prize_floor_100', candidates)
        self.assertEqual(candidates['random_shape_50']['feature_weights'], {'seeded_random': 1.0})
        self.assertEqual(candidates['random_prize_floor_100']['final_selection_mode'], 'prize_floor')

    def test_per_play_tournament_final_test_uses_strategy_pool_config(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(820, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        captured = []

        def fake_rolling(feature_weights, model_weights, **kwargs):
            captured.append(kwargs)
            return {
                'select_5': {
                    'lift': 0.1,
                    'mean_hits': 1.4,
                    'probabilities': {'>=3': 0.12, '>=4': 0.03},
                    'theoretical_probs': {'>=3': 0.1, '>=4': 0.02},
                    'profit_roi': -0.4,
                    'random_profit_roi': -0.5,
                },
                'fu_shi_7': {
                    'pool_mean_hits': 1.9,
                    'pool_expected_random': 1.75,
                    'probabilities': {'>=3': 0.25},
                    'theoretical_probs': {'>=3': 0.2},
                    'profit_roi': -0.4,
                    'random_profit_roi': -0.5,
                },
            }

        def fake_permutation(*args, **kwargs):
            return {'p_value': 0.01}

        original_activate = kl8_snapshots.activate_verified_strategy
        original_persist = kl8_records._persist_trial_results
        original_trials = kl8_config.STRATEGY_TRIAL_RESULTS

        try:
            backtest._rolling_backtest_parametric = fake_rolling
            backtest._permutation_test = fake_permutation
            kl8_snapshots.activate_verified_strategy = lambda *args, **kwargs: None
            kl8_records._persist_trial_results = lambda: None
            kl8_config.STRATEGY_TRIAL_RESULTS = []

            result = backtest.run_candidate_tournament_per_play_type(
                'select_5',
                candidate_strategies={
                    'pool_limited': {
                        'strategy_id': 'pool_limited',
                        'feature_weights': {'frequency': 1.0},
                        'model_weights': {'rank': 1.0},
                        'window_size': 50,
                        'pool_diversify': False,
                        'pool_max_last_numbers': 2,
                    },
                },
                n_permutations=1,
            )
        finally:
            kl8_snapshots.activate_verified_strategy = original_activate
            kl8_records._persist_trial_results = original_persist
            kl8_config.STRATEGY_TRIAL_RESULTS = original_trials

        self.assertTrue(result.get('activated'))
        final_calls = [c for c in captured if c['start_idx'] == 600 and c['end_idx'] == 820]
        self.assertTrue(final_calls)
        self.assertTrue(all(c['pool_diversify'] is False for c in final_calls))
        self.assertTrue(all(c['pool_max_last_numbers'] == 2 for c in final_calls))

    def test_legacy_tournament_validation_uses_each_strategy_pool_config(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(700, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        validation_calls = []

        def fake_rolling(feature_weights, model_weights, **kwargs):
            if kwargs['start_idx'] == 200 and kwargs['end_idx'] == 500:
                validation_calls.append((feature_weights.get('marker'), kwargs))
            return {
                f'select_{pick}': {
                    'lift': 0.1,
                    'mean_hits': 1.0,
                    'probabilities': {},
                    'theoretical_probs': {},
                    'profit_roi': -0.4,
                    'random_profit_roi': -0.5,
                }
                for pick in [3, 4, 5, 6, 7]
            }

        def fake_permutation(*args, **kwargs):
            return {'p_value': 1.0}

        original_persist = kl8_records._persist_trial_results
        original_trials = kl8_config.STRATEGY_TRIAL_RESULTS

        try:
            backtest._rolling_backtest_parametric = fake_rolling
            backtest._permutation_test = fake_permutation
            kl8_records._persist_trial_results = lambda: None
            kl8_config.STRATEGY_TRIAL_RESULTS = []

            result = backtest.run_candidate_tournament(
                candidate_strategies={
                    'first': {
                        'strategy_id': 'first',
                        'feature_weights': {'frequency': 1.0, 'marker': 'first'},
                        'model_weights': {'rank': 1.0},
                        'window_size': 50,
                        'pool_diversify': False,
                        'pool_max_last_numbers': 2,
                    },
                    'second': {
                        'strategy_id': 'second',
                        'feature_weights': {'frequency': 1.0, 'marker': 'second'},
                        'model_weights': {'rank': 1.0},
                        'window_size': 50,
                        'pool_diversify': True,
                        'pool_max_last_numbers': 7,
                    },
                },
                n_permutations=1,
            )
        finally:
            kl8_records._persist_trial_results = original_persist
            kl8_config.STRATEGY_TRIAL_RESULTS = original_trials

        self.assertNotIn('error', result)
        by_marker = {marker: kwargs for marker, kwargs in validation_calls}
        self.assertFalse(by_marker['first']['pool_diversify'])
        self.assertEqual(by_marker['first']['pool_max_last_numbers'], 2)
        self.assertTrue(by_marker['second']['pool_diversify'])
        self.assertEqual(by_marker['second']['pool_max_last_numbers'], 7)

    def test_settlement_includes_select_8_9_10_and_fushi_10_11(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(20, 0, -1)]
        analyzer.using_simulated_data = False

        snapshot = {
            'snapshot_id': 'settle-new-plays',
            'based_on_issue': '2026001',
            'select_8': list(range(1, 9)),
            'select_9': list(range(1, 10)),
            'select_10': list(range(1, 11)),
            'fu_shi_10_11': list(range(1, 12)),
            'play_strategies': {},
            'prediction_modes': {},
            'resolved_strategies': {},
        }

        original_snapshot_dir = kl8_config.KL8_SNAPSHOT_DIR
        original_settlement_dir = kl8_config.KL8_SETTLEMENT_DIR

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / 'snapshots'
            settlement_dir = Path(tmp) / 'settlements'
            snapshot_dir.mkdir()
            settlement_dir.mkdir()
            snapshot_path = snapshot_dir / 'snapshot_new_plays.json'
            snapshot_path.write_text(json.dumps(snapshot), encoding='utf-8')

            try:
                kl8_config.KL8_SNAPSHOT_DIR = str(snapshot_dir)
                kl8_config.KL8_SETTLEMENT_DIR = str(settlement_dir)

                result = analyzer.settle_prediction(
                    snapshot_path.name,
                    '2026002',
                    list(range(1, 21)),
                )
            finally:
                kl8_config.KL8_SNAPSHOT_DIR = original_snapshot_dir
                kl8_config.KL8_SETTLEMENT_DIR = original_settlement_dir

        self.assertTrue(result['success'])
        settlement = result['settlement']

        self.assertEqual(settlement['hit_select_8'], 8)
        self.assertEqual(settlement['hit_select_9'], 9)
        self.assertEqual(settlement['hit_select_10'], 10)
        self.assertEqual(settlement['prize_settlement']['select_8']['prize'], 50000)
        self.assertEqual(settlement['prize_settlement']['select_9']['prize'], 250000)
        self.assertEqual(settlement['prize_settlement']['select_10']['prize'], 5000000)

        fushi = settlement['fushi_settlement']['fu_shi_10_11']
        self.assertTrue(fushi['placed'])
        self.assertEqual(fushi['pool_hits'], 11)
        self.assertEqual(fushi['max_combo_hits'], 10)
        self.assertEqual(fushi['total_combinations'], 11)
        self.assertEqual(fushi['total_bet'], 22)
        self.assertEqual(fushi['total_prize'], 55000000)
        self.assertEqual(fushi['hit_distribution'], {10: 11})

    def test_recent_settlement_performance_compares_random_baseline(self):
        original_settlement_dir = kl8_config.KL8_SETTLEMENT_DIR

        with tempfile.TemporaryDirectory() as tmp:
            settlement_dir = Path(tmp) / 'settlements'
            settlement_dir.mkdir()
            rows = [
                {
                    'snapshot_id': 'perf-1',
                    'settled_at': '2026-01-02T00:00:00',
                    'prize_settlement': {
                        'select_5': {'placed': True, 'hits': 3, 'bet': 2, 'prize': 10},
                    },
                    'fushi_settlement': {
                        'fu_shi_7': {'placed': True, 'pool_hits': 2, 'total_bet': 42, 'total_prize': 20},
                    },
                },
                {
                    'snapshot_id': 'perf-2',
                    'settled_at': '2026-01-01T00:00:00',
                    'prize_settlement': {
                        'select_5': {'placed': True, 'hits': 1, 'bet': 2, 'prize': 0},
                    },
                    'fushi_settlement': {
                        'fu_shi_7': {'placed': True, 'pool_hits': 4, 'total_bet': 42, 'total_prize': 100},
                    },
                },
            ]
            for row in rows:
                path = settlement_dir / f"settlement_{row['snapshot_id']}.json"
                path.write_text(json.dumps(row), encoding='utf-8')

            try:
                kl8_config.KL8_SETTLEMENT_DIR = str(settlement_dir)
                result = kl8_module._build_recent_settlement_performance(windows=(2,))
            finally:
                kl8_config.KL8_SETTLEMENT_DIR = original_settlement_dir

        self.assertEqual(result['available_count'], 2)
        window = result['windows'][0]
        self.assertEqual(window['settled_count'], 2)
        select5 = window['play_stats']['select_5']
        self.assertEqual(select5['settled_count'], 2)
        self.assertEqual(select5['avg_hits'], 2.0)
        self.assertEqual(select5['random_expected_hits'], 1.25)
        self.assertEqual(select5['hit_delta_vs_random'], 0.75)
        self.assertEqual(select5['profit_roi'], 1.5)
        fushi7 = window['play_stats']['fu_shi_7']
        self.assertEqual(fushi7['avg_hits'], 3.0)
        # 期望命中 = 号码个数 x 20/80。fu_shi_7 是选5复式 7 码 → 7*0.25=1.75
        # （1b57890 之前配的是 8 码，名字与配置对不上，那时这里是 2.0）
        self.assertEqual(fushi7['random_expected_hits'], 1.75)
        self.assertEqual(fushi7['hit_delta_vs_random'], 1.25)

    def test_strategy_health_combines_validation_and_recent_settlements(self):
        original_strategies = kl8_config.ACTIVE_STRATEGIES
        try:
            kl8_config.ACTIVE_STRATEGIES = {
                key: {'strategy_id': '', 'feature_weights': {}, 'model_weights': {}, 'window_size': 0}
                for key in list(kl8_module.SELECT_PLAY_KEYS) + list(kl8_module.FUSHI_PLAY_KEYS)
            }
            kl8_config.ACTIVE_STRATEGIES['select_5'] = {
                'strategy_id': 'select_5_good',
                'is_validated': True,
                'validation_report': {'validation_lift': 0.2, 'final_test_lift': 0.1},
            }
            kl8_config.ACTIVE_STRATEGIES['select_6'] = {
                'strategy_id': 'select_6_weak',
                'is_validated': True,
                'validation_report': {'validation_lift': -0.1, 'final_test_lift': -0.05},
            }

            performance = {
                'available_count': 30,
                'windows': [
                    {
                        'window_size': 30,
                        'settled_count': 30,
                        'play_stats': {
                            'select_5': {
                                'settled_count': 30,
                                'hit_delta_vs_random': 0.35,
                                'profit_roi': 0.1,
                            },
                            'select_6': {
                                'settled_count': 30,
                                'hit_delta_vs_random': -0.35,
                                'profit_roi': -0.9,
                            },
                        },
                    }
                ],
            }

            result = kl8_module._build_strategy_health(performance)
        finally:
            kl8_config.ACTIVE_STRATEGIES = original_strategies

        health = result['health_by_play']
        self.assertEqual(health['select_5']['status'], 'healthy')
        self.assertEqual(health['select_6']['status'], 'cool_down')
        self.assertEqual(health['select_10']['status'], 'unverified')
        self.assertEqual(result['window_size'], 30)


if __name__ == '__main__':
    unittest.main()
