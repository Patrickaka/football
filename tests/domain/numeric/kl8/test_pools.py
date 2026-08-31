"""kl8 的候选池：形态、重号上限、七种挑法、最终选池、多注覆盖。

这一层决定最终推荐哪几个号，**算错了不会报错，只会换一组号**。参照物是从
迁移前的实现生成的黄金文件（`tests/fixtures/golden/kl8_pools.json.gz`，
24459 条），语料按「每个函数各自的边界」铺开：五种形状的候选序列（真实投票
产物、均匀递减、全部并列、短序列、空）× 四种上期号码 × 十档选号数 ×
七档重号上限 × 十三种模式（含 `None` 与一个拼错的名字）。

另有一组手写用例守住语义：三种 diversify 变体的差别只在重号上限、
惰性建池不改变结果、`best_variant` 排除 `low_repeat`、形态惩罚两侧对称。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.numeric.kl8 import pools, portfolio, shape
from src.domain.numeric.kl8.space import DRAW_COUNT, SPACE
from tests.domain.golden import as_json

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/kl8_pools.json.gz')
HISTORY = _load('numeric/kl8_history.json.gz')['results']

# 四种形状的候选序列。真实的那份在 setUpModule 里从投票管道取。
EVEN = [(n, 1.0 - n * 0.01) for n in range(1, 41)]
TIED = [(n, 0.5) for n in range(1, 41)]
SHORT = [(n, 0.9 - n * 0.02) for n in (3, 7, 11, 19, 23, 31, 44, 52, 61, 70)]

LASTS = {'none': set(), 'first20': set(range(1, 21)),
         'odd': {n for n in range(1, 81) if n % 2}}
SIZES = (0, 1, 3, 5, 6, 7, 8, 11, 20, 25)
CAPS = (None, 0, 1, 2, 3, 6, 99)
MODES = tuple(pools.MODE_BUILDERS) + (pools.BEST_VARIANT, None, 'typo_mode')


def real_candidates():
    """真实投票产物：票数分布与人造语料完全不同，两者都要覆盖。"""
    from src.kl8.analyzer import KL8Analyzer
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_data = HISTORY
    analyzer.using_simulated_data = False
    analyzer.history_file = ''
    analyzer._data_mtime = 0
    analyzer.update_statistics()
    vote = analyzer.multi_model_voting(
        pick_n=20, top_n=40, model_weights={'rank': 1.0},
        feature_weights={'frequency': 0.45, 'gap': 0.20, 'trend': 0.20,
                         'pair_cooccurrence': 0.10, 'position_residual': 0.05})
    return vote['candidates'], set(analyzer.statistics.get('last_numbers', set()))


POOL_NAMES = ('real', 'even', 'tied', 'short', 'empty')


class GoldenTests(unittest.TestCase):
    """迁移前后逐条比对。任何一条对不上，都意味着推荐的号变了。"""

    @classmethod
    def setUpClass(cls):
        real, real_last = real_candidates()
        cls.pools = {'real': real, 'even': EVEN, 'tied': TIED,
                     'short': SHORT, 'empty': []}
        cls.lasts = dict(LASTS, real=real_last)

    def _check(self, key, value):
        with self.subTest(case=key):
            self.assertEqual(as_json(value), GOLDEN[key])

    def test_shape_matches_golden(self):
        samples = {'spread': [3, 18, 25, 41, 57, 66, 72, 79],
                   'clustered': [1, 2, 3, 4, 5, 6],
                   'all_odd': [1, 3, 5, 7, 9, 11],
                   'all_small': [2, 4, 6, 8, 10, 12],
                   'single': [40], 'empty': []}
        for size in SIZES:
            self._check(f'shape_targets:{size}', shape.targets(size))
        for name, nums in samples.items():
            for lname, last in self.lasts.items():
                self._check(f'shape_profile:{name}:{lname}', shape.profile(nums, last))
                for cap in (0, 2, 6):
                    self._check(f'shape_penalty:{name}:{lname}:{len(nums)}:{cap}',
                                shape.penalty(nums, len(nums), last, cap))
                self._check(f'shape_penalty_wrongsize:{name}:{lname}',
                            shape.penalty(nums, len(nums) + 1, last, 3))

    def test_repeat_caps_match_golden(self):
        hists = {'full': HISTORY, 'recent20': HISTORY[:20], 'two': HISTORY[:2],
                 'one': HISTORY[:1], 'none': []}
        for hname, hist in hists.items():
            for size in SIZES:
                self._check(f'default_cap:{size}', pools.default_repeat_cap(size))
                self._check(f'adaptive_cap:{hname}:{size}',
                            pools.adaptive_repeat_cap(hist, size))
                for minimum in (0, 1, 3):
                    self._check(f'adaptive_target:{hname}:{size}:{minimum}',
                                pools.adaptive_repeat_target(hist, size, minimum))

    def _builders_for(self, pname):
        pool = self.pools[pname]
        for size in SIZES:
            self._check(f'zone_spread:{pname}:{size}', pools.zone_spread(pool, size))
            for lname, last in self.lasts.items():
                for cap in CAPS:
                    for label, fn in (('diversify', pools.diversify),
                                      ('prize_floor', pools.prize_floor),
                                      ('high_tier', pools.high_tier_chase),
                                      ('shape_balanced', pools.shape_balanced)):
                        self._check(f'{label}:{pname}:{size}:{lname}:{cap}',
                                    fn(pool, size, last, max_last_numbers=cap))

    def _select_final_for(self, pname):
        pool = self.pools[pname]
        for size in SIZES:
            for lname, last in self.lasts.items():
                for cap in CAPS:
                    for mode in MODES:
                        self._check(
                            f'select_final:{pname}:{size}:{lname}:{cap}:{mode}',
                            pools.select_final(pool, size, last,
                                               max_last_numbers=cap,
                                               selection_mode=mode))

    def test_scoring_and_minimum_repeats_match_golden(self):
        for pname, pool in self.pools.items():
            for lname, last in self.lasts.items():
                for size in (5, 6, 11):
                    for cap in (0, 3):
                        self._check(f'score_selection:{pname}:{size}:{lname}:{cap}',
                                    pools.score_selection(pool[:size], pool, size, last, cap))
                for minimum in (0, 1, 2, 5):
                    self._check(f'min_repeats:{pname}:{lname}:{minimum}',
                                pools.enforce_minimum_repeats(pool[:6], pool, last, minimum))

    def test_portfolio_matches_golden(self):
        slips = {'disjoint': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
                 'overlap': [[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]],
                 'single': [[1, 2, 3, 4, 5]], 'empty': [],
                 'with_blank': [[], [1, 2, 3, 4, 5]]}
        for name, group in slips.items():
            for seed in ('', 'abc', '2026228'):
                self._check(f'coverage:{name}:{seed}',
                            portfolio.simulate_coverage(group, simulations=2000, seed_key=seed))
            self._check(f'coverage_zero:{name}',
                        portfolio.simulate_coverage(group, simulations=0))


# 这两族原本各是**一个**测试方法，内部把 5 个候选池 × 10 个尺寸 × 若干上次
# 开奖 × 7 个上限 × 13 种模式全跑一遍——select_final 那条单独 20 秒，占全量
# 总时长四分之一强。pytest 的调度粒度是测试方法，**一个方法无论多慢都只能
# 待在一个进程里**，所以它同时也是并行化的天花板。
#
# 按候选池拆成每池一个方法：覆盖一条不少（黄金键名完全没变），但最慢的单条
# 从 20 秒降到 4 秒上下，并行时能摊到不同 worker 上。
def _bind_per_pool(stem, impl):
    for name in POOL_NAMES:
        def method(self, _pool_name=name, _impl=impl):
            _impl(self, _pool_name)
        method.__name__ = f'{stem}_{name}'
        method.__doc__ = f'候选池 {name!r} 的黄金比对。'
        setattr(GoldenTests, method.__name__, method)


_bind_per_pool('test_builders', GoldenTests._builders_for)
_bind_per_pool('test_select_final', GoldenTests._select_final_for)


class ModeRegistryTests(unittest.TestCase):
    """三种 diversify 变体的差别只有重号上限——写成一处，接反的可能就只剩一处。"""

    def setUp(self):
        self.candidates = EVEN
        self.last = set(range(1, 21))

    def _pool(self, mode, size=8, cap=3):
        return pools.build_pool(mode, self.candidates, size, self.last, cap)

    def test_low_repeat_allows_fewer_repeats_than_balanced(self):
        low = sum(1 for n, _ in self._pool('low_repeat') if n in self.last)
        balanced = sum(1 for n, _ in self._pool('balanced') if n in self.last)
        self.assertLessEqual(low, balanced)

    def test_repeat_follow_allows_more_repeats_than_balanced(self):
        follow = sum(1 for n, _ in self._pool('repeat_follow') if n in self.last)
        balanced = sum(1 for n, _ in self._pool('balanced') if n in self.last)
        self.assertGreaterEqual(follow, balanced)

    def test_balanced_and_diversified_are_the_same_pool(self):
        self.assertEqual(self._pool('balanced'), self._pool('diversified'))

    def test_low_repeat_cap_never_goes_negative(self):
        """上限减到 0 以下会让 min() 比较出乱七八糟的结果。"""
        self.assertEqual(pools._shifted_cap(0, -1, 8), 0)

    def test_repeat_follow_cap_never_exceeds_pick_size(self):
        self.assertEqual(pools._shifted_cap(8, +1, 8), 8)

    def test_unknown_mode_is_rejected_by_build_pool(self):
        """`build_pool` 是内部调用，名字必须先查过表——这里报错才不会静默建错池。"""
        with self.assertRaises(KeyError):
            pools.build_pool('typo_mode', self.candidates, 6, self.last, 3)


class SelectFinalTests(unittest.TestCase):

    def setUp(self):
        self.candidates = EVEN
        self.last = set(range(1, 21))

    def _select(self, mode, size=6, cap=3):
        return pools.select_final(self.candidates, size, self.last,
                                  max_last_numbers=cap, selection_mode=mode)

    def test_only_the_named_pool_is_built(self):
        """迁移前十种全建再挑一种，`shape_balanced` 一家就占一次预测的 15%。"""
        built = []
        original = pools.build_pool
        try:
            pools.build_pool = lambda mode, *a, **k: (built.append(mode)
                                                      or original(mode, *a, **k))
            self._select('concentrated')
        finally:
            pools.build_pool = original
        self.assertEqual(built, ['concentrated'])

    def test_best_variant_builds_every_pool_but_low_repeat(self):
        built = []
        original = pools.build_pool
        try:
            pools.build_pool = lambda mode, *a, **k: (built.append(mode)
                                                      or original(mode, *a, **k))
            self._select(pools.BEST_VARIANT)
        finally:
            pools.build_pool = original
        self.assertNotIn('low_repeat', built)
        self.assertEqual(set(built), set(pools.MODE_BUILDERS) - {'low_repeat'})

    def test_unknown_mode_falls_back_to_scoring_including_low_repeat(self):
        """迁移前就有的行为，且线上可达：2433 条试验记录的模式字段是 None。"""
        built = []
        original = pools.build_pool
        try:
            pools.build_pool = lambda mode, *a, **k: (built.append(mode)
                                                      or original(mode, *a, **k))
            _, resolved = self._select(None)
        finally:
            pools.build_pool = original
        self.assertIn('low_repeat', built)
        self.assertIn(resolved, pools.MODE_BUILDERS)

    def test_named_mode_is_reported_back_unchanged(self):
        self.assertEqual(self._select('zone_spread')[1], 'zone_spread')

    def test_empty_candidates_report_the_requested_mode(self):
        """报回请求值而不是解析值：什么都没建，就没有「实际用的模式」可报。"""
        self.assertEqual(pools.select_final([], 6, self.last, selection_mode='balanced'),
                         ([], 'balanced'))

    def test_zero_target_gives_an_empty_pool(self):
        self.assertEqual(self._select('balanced', size=0)[0], [])


class RepeatCapTests(unittest.TestCase):

    def _history(self, overlap):
        """造一段相邻两期恰好重合 overlap 个号的历史。"""
        base = list(range(1, 21))
        other = list(range(21, 41))
        records = []
        for idx in range(6):
            nums = base[:overlap] + other[:DRAW_COUNT - overlap] if idx % 2 else base
            records.append({'issue': str(2026000 + 100 - idx), 'numbers': sorted(set(nums))})
        return records

    def test_high_overlap_raises_the_cap(self):
        high = pools.adaptive_repeat_cap(self._history(15), 10)
        low = pools.adaptive_repeat_cap(self._history(1), 10)
        self.assertGreater(high, low)

    def test_cap_stays_within_the_allowed_ratio_band(self):
        """界限写死在断言里，不引用被测常量——引用的话改坏常量断言会跟着挪。"""
        for overlap in (0, 1, 5, 10, 20):
            cap = pools.adaptive_repeat_cap(self._history(overlap), 20)
            self.assertGreaterEqual(cap, 5)    # 20 * 0.25
            self.assertLessEqual(cap, 11)      # 20 * 0.55

    def test_ratio_band_clamps_both_ends(self):
        """当前档位表产不出界外的值，所以只能直接喂——它守的是档位表被改坏。"""
        self.assertEqual(pools.ratio_within_band(0.10), 0.25)
        self.assertEqual(pools.ratio_within_band(0.90), 0.55)

    def test_ratio_band_leaves_in_band_values_alone(self):
        self.assertEqual(pools.ratio_within_band(0.40), 0.40)

    def test_too_little_history_falls_back_to_the_static_cap(self):
        for history in ([], HISTORY[:1]):
            self.assertEqual(pools.adaptive_repeat_cap(history, 8),
                             pools.default_repeat_cap(8))

    def test_target_without_samples_uses_the_theoretical_rate(self):
        """没有样本时退回 80 选 20 的期望重合数，而不是 0。"""
        self.assertEqual(pools.adaptive_repeat_target([], 8)['mean_draw_overlap'],
                         DRAW_COUNT * 0.25)

    def test_minimum_floor_wins_over_a_lower_computed_target(self):
        self.assertGreaterEqual(pools.adaptive_repeat_target(self._history(0), 8, 3)['target'], 3)

    def test_target_never_exceeds_the_cap(self):
        for overlap in (0, 10, 20):
            profile = pools.adaptive_repeat_target(self._history(overlap), 8)
            self.assertLessEqual(profile['target'], profile['cap'])

    def test_blank_draws_are_not_counted_as_zero_overlap(self):
        """缺数据算成「重合 0 个」会把上限一路压到底。"""
        blanks = [{'issue': '1', 'numbers': []}, {'issue': '2', 'numbers': []}]
        self.assertEqual(pools.adaptive_repeat_target(blanks, 8)['sample_size'], 0)


class CleanPickTests(unittest.TestCase):

    def test_accepts_a_well_formed_pick(self):
        self.assertEqual(pools.clean_pick_numbers([3, 1, 2], 3), [3, 1, 2])

    def test_rejects_duplicates_wrong_length_and_out_of_range(self):
        for bad in ([1, 1, 2], [1, 2], [1, 2, 3, 4], [1, 2, SPACE.high + 1],
                    [0, 2, 3], 'abc', None):
            with self.subTest(pick=bad):
                self.assertEqual(pools.clean_pick_numbers(bad, 3), [])

    def test_both_range_ends_are_accepted(self):
        """只测上界的话，下界写错了也发现不了。"""
        self.assertEqual(pools.clean_pick_numbers([SPACE.low, SPACE.high], 2),
                         [SPACE.low, SPACE.high])


class ShapeSwapWindowTests(unittest.TestCase):
    """替补只在靠前的一段候选里找。放开窗口，低分号会仅凭形态挤进来。"""

    def _candidates(self):
        """靠前的号全挤在头两个大区，形态好的号故意排在窗口之外。"""
        head = [(n, 0.90 - i * 0.0001) for i, n in enumerate(range(1, 41))]
        tail = [(n, 0.80 - i * 0.0001) for i, n in enumerate(range(41, 81))]
        return head + tail

    def test_swaps_stay_inside_the_window(self):
        pool = pools.shape_balanced(self._candidates(), 6, set(), max_last_numbers=3)
        self.assertEqual([num for num, _ in pool], [1, 2, 21, 22, 41, 42])

    def test_window_widens_with_the_pick_size(self):
        """窗口是 max(选号数*5, 30)：选 8 时能够到第 40 名，选 6 时够不到。"""
        pool = pools.shape_balanced(self._candidates(), 8, set(), max_last_numbers=3)
        self.assertEqual([num for num, _ in pool], [1, 2, 4, 21, 22, 41, 42, 43])

    def test_window_multiplier_reaches_past_the_flat_minimum(self):
        """选 12 时 5 倍能够到第 60 名、3 倍只到第 36 名——倍数改小就够不着了。"""
        pool = pools.shape_balanced(self._candidates(), 12, set(), max_last_numbers=3)
        self.assertEqual([num for num, _ in pool],
                         [1, 2, 4, 21, 22, 23, 41, 42, 43, 44, 46, 61])


class ShapePenaltyTests(unittest.TestCase):

    def test_wrong_size_is_marked_malformed_not_merely_bad(self):
        self.assertEqual(shape.penalty([1, 2, 3], 4, set(), 3), shape.MALFORMED_PENALTY)

    def test_penalty_is_symmetric_around_the_neutral_range(self):
        """奇多与偶多要罚得一样重，只罚一侧会让形态系统性偏斜。"""
        all_odd = shape.penalty([1, 3, 5, 7], 4, set(), 4)
        all_even = shape.penalty([2, 4, 6, 8], 4, set(), 4)
        self.assertEqual(all_odd, all_even)

    def test_balanced_shape_beats_a_clustered_one(self):
        balanced = shape.penalty([5, 25, 45, 65], 4, set(), 4)
        clustered = shape.penalty([1, 2, 3, 4], 4, set(), 4)
        self.assertLess(balanced, clustered)

    def test_only_excess_repeats_are_penalised(self):
        """重号偏少不是毛病，偏多才是。两边都罚会逼着推荐去追上期的号。"""
        last = {1, 2, 3, 4}
        few = shape.penalty([5, 25, 45, 65], 4, last, 2)
        many = shape.penalty([1, 2, 3, 4], 4, last, 2)
        self.assertLess(few, many)

    def test_numbers_outside_the_space_are_not_bucketed(self):
        """空间外的号码硬塞进首尾桶，会凭空造出一个形态。"""
        self.assertEqual(shape.profile([0, 81, 200])['zone20'], [0, 0, 0, 0])

    def test_every_bucket_has_a_slot_even_when_empty(self):
        profile = shape.profile([1])
        self.assertEqual(len(profile['zone20']), 4)
        self.assertEqual(len(profile['zone10']), 8)


class PortfolioTests(unittest.TestCase):

    def test_first_slip_matches_the_top_of_the_ranking(self):
        """第一注若与主推号码不一致，用户会看到两组不同的「推荐」。"""
        ranked = list(range(1, 81))
        slips = portfolio.coverage_slips(ranked, 4, 6)
        self.assertEqual(slips[0], sorted(ranked[:6]))

    def test_slips_are_disjoint_while_numbers_last(self):
        """13 注 × 6 码 = 78 个槽位，还没到 80，所以应当一个号都不重复。"""
        slips = portfolio.coverage_slips(list(range(1, 81)), 13, 6)
        flat = [n for slip in slips for n in slip]
        self.assertEqual(len(flat), 78)
        self.assertEqual(len(set(flat)), 78)

    def test_numbers_are_reused_once_the_space_runs_out(self):
        slips = portfolio.coverage_slips(list(range(1, 81)), 14, 6)
        self.assertEqual(len(slips), 14)
        self.assertEqual(slips[13], sorted(range(1, 5)) + [79, 80])

    def test_ranking_shorter_than_one_slip_yields_nothing(self):
        self.assertEqual(portfolio.coverage_slips([1, 2, 3], 2, 6), [])

    def test_simulation_is_deterministic_for_the_same_seed(self):
        slips = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        first = portfolio.simulate_coverage(slips, simulations=500, seed_key='k')
        second = portfolio.simulate_coverage(slips, simulations=500, seed_key='k')
        self.assertEqual(first, second)

    def test_different_seeds_give_different_estimates(self):
        """种子不参与抽样的话，「确定性」就变成了「恒定」。"""
        slips = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        first = portfolio.simulate_coverage(slips, simulations=500, seed_key='a')
        second = portfolio.simulate_coverage(slips, simulations=500, seed_key='b')
        self.assertNotEqual(first['average_best_hits'], second['average_best_hits'])

    def test_disjoint_slips_cover_more_than_overlapping_ones(self):
        disjoint = portfolio.simulate_coverage(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], simulations=3000, seed_key='s')
        overlapping = portfolio.simulate_coverage(
            [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]], simulations=3000, seed_key='s')
        self.assertGreater(disjoint['at_least_one_ge3'], overlapping['at_least_one_ge3'])
        self.assertGreater(disjoint['unique_number_count'],
                           overlapping['unique_number_count'])

    def test_tiers_are_nested(self):
        """中 4 的注一定也中了 3，比例反过来就说明档位算错了。"""
        result = portfolio.simulate_coverage(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], simulations=2000, seed_key='n')
        self.assertGreaterEqual(result['at_least_one_ge3'], result['at_least_one_ge4'])
        self.assertGreaterEqual(result['at_least_one_ge4'], result['at_least_one_ge5'])
        self.assertGreaterEqual(result['at_least_one_ge5'], result['at_least_one_ge6'])

    def test_no_simulations_yields_nothing(self):
        self.assertEqual(portfolio.simulate_coverage([[1, 2, 3]], simulations=0), {})


if __name__ == '__main__':
    unittest.main()
