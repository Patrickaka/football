"""福彩 3D 的回测层：滚动回测、随机对照、置换检验、权重搜索、四个分项回测。

这一层不出号，**算错了只会让「这套策略行不行」这个结论反过来**——而那个
结论决定 ML 要不要进实盘融合、权重要不要换一组。参照物是从迁移前的实现
生成的黄金文件（`tests/fixtures/golden/lottery3d_backtest.json.gz`，75 条），
语料按三种长度的历史铺开，其中 `s80` 短于「回测期数 + 最长窗口 + 缓冲」，
专门用来走 `resolve_trials` 把期数压回去的那条分支。

**黄金值与迁移前逐条相同**（75 条零差异）。有两处语义是有意改的，但在这份
语料上表现不出差异，因此各由手写用例单独守着：

1. 比率的分母改成**实际评估的期数**，迁移前一律用请求的期数。历史比请求的
   期数还短时，迁移前算出来的命中率会被系统性低估，报出来的 `trials` 也与
   实际跑的期数对不上。
2. `rolling_slices` 的起点夹到 0。迁移前 `len(numbers) - trials` 为负时
   `range` 会从负数开始，实际跑的期数比请求的多，且前几期的训练集与开奖号
   对不上——**全程不报错**。

另有一处签名变更：`_mutate_weights` 删掉了 `base` 参数。它在迁移前的函数体
里从头到尾没出现过，而调用方一直在认真地传（§五·1 那一类）。
"""
import gzip
import importlib
import inspect
import json
import pathlib
import random
import unittest

from src.domain.numeric.lottery3d import backtest as bt
from src.domain.numeric.lottery3d import component_backtest as cbt
from src.domain.numeric.lottery3d import weight_search as ws
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_backtest.json.gz')
NUMBERS = [tuple(r['digits'])
           for r in _load('numeric/lottery3d_history.json.gz')['results']]

# s80 比 trials + max(RECENT_WINDOWS) + 5 短，走 resolve_trials 的压缩分支
SERIES = {'s300': NUMBERS[-300:], 's150': NUMBERS[-150:], 's80': NUMBERS[-80:]}
# 迁移当时 config.py 里生效的那组值，写死不 import——import 的话配置一改，
# 期望值会跟着挪，黄金文件就白设了（判据 12）
FIXED_WW = {30: 0.3, 45: 0.25, 60: 0.25, 90: 0.2}


def golden_entries():
    """按 (键, 值) 逐条产出全部语料，测试与重生成脚本共用。"""
    yield from _rolling_entries()
    yield from _component_entries()
    yield from _permutation_entries()
    yield from _search_entries()


def _rolling_entries():
    for name, series in SERIES.items():
        for trials in (5, 8):
            yield f'backtest:{name}:{trials}:dyn', adapter.backtest(series, trials=trials)
            yield (f'backtest:{name}:{trials}:fixed',
                   adapter.backtest(series, trials=trials, window_weights=FIXED_WW))
        for trials in (20, 50):
            for top_n in (30, 100):
                for seed in (42, 7):
                    yield (f'random:{name}:{trials}:{top_n}:{seed}',
                           adapter.random_baseline_backtest(
                               series, trials=trials, top_n=top_n, seed=seed))


def _component_entries():
    for name, series in SERIES.items():
        for trials in (5, 20):
            yield f'dan_kill:{name}:{trials}', scoring.backtest_dan_kill(series, trials=trials)
            yield (f'form_pred:{name}:{trials}',
                   scoring.backtest_form_prediction(series, trials=trials))
            yield (f'sum_span:{name}:{trials}',
                   scoring.backtest_sum_span_interval(series, trials=trials))
        for trials in (20, 50):
            yield f'slope:{name}:{trials}', features.backtest_slope_patterns(series, trials=trials)


def _permutation_entries():
    # shuffles 压到 3：线上默认 200 次，每次一整轮回测
    for name in ('s150', 's80'):
        yield (f'perm:{name}',
               adapter.permutation_test(SERIES[name], 0.05, trials=5,
                                        window_weights=FIXED_WW, shuffles=3, seed=20))
    sample = adapter.backtest(SERIES['s150'], trials=5, window_weights=FIXED_WW)
    for metric in ('top3_rate', 'top30_rate', 'top_rate', 'ge2_digit_rate', 'composite'):
        yield f'objective:{metric}', adapter.backtest_objective(sample, metric)


def _search_entries():
    base = config.default_weights()
    yield 'base_weights', base
    for seed in (1, 42):
        rng = random.Random(seed)
        yield f'sample_w:{seed}', [adapter._sample_random_weights(base, rng) for _ in range(4)]
        rng = random.Random(seed)
        weights, seq = dict(base), []
        for _ in range(4):
            weights = adapter._mutate_weights(weights, rng)
            seq.append(dict(weights))
        yield f'mutate_w:{seed}', seq
    yield 'eval_w', adapter.evaluate_weights(SERIES['s150'], base, trials=5,
                                             window_weights=FIXED_WW, metric='top3_rate')
    yield 'search', adapter.search_weights(
        numbers=SERIES['s150'], iterations=2, backtest_trials=5, metric='top3_rate',
        seed=42, refine_rounds=2, verbose=False, test_ratio=0.15)
    # test_ratio 放大且 backtest_trials 够大，才走得到测试集验收那条分支
    yield 'search_wide', adapter.search_weights(
        numbers=SERIES['s150'], iterations=2, backtest_trials=20, metric='composite',
        seed=7, refine_rounds=1, verbose=False, test_ratio=0.3)


class GoldenTests(unittest.TestCase):
    """与迁移前的实现逐条比对。"""

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class RollingSliceTests(unittest.TestCase):
    """滚动切片：回测里唯一「错了也不报错」的地方。"""

    SERIES = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]

    def test_train_is_strictly_before_actual(self):
        """训练集只到 actual 的前一期。**方向反了命中率会好得不真实。**"""
        for train, actual in bt.rolling_slices(self.SERIES, 3):
            self.assertNotIn(actual, train)
            self.assertEqual(train, self.SERIES[:self.SERIES.index(actual)])

    def test_yields_exactly_trials_periods(self):
        self.assertEqual(len(list(bt.rolling_slices(self.SERIES, 3))), 3)

    def test_start_clamped_when_trials_exceed_history(self):
        """请求的期数超过历史长度时，跑满整段就停，**不会从负索引开始**。"""
        pairs = list(bt.rolling_slices(self.SERIES, 99))
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs[0], ([], (1, 1, 1)))

    def test_first_slice_of_full_run_has_empty_train(self):
        self.assertEqual(list(bt.rolling_slices(self.SERIES, 5))[0][0], [])


class ResolveTrialsTests(unittest.TestCase):

    def test_keeps_request_when_history_is_long_enough(self):
        self.assertEqual(bt.resolve_trials(300, 8, 90), 8)

    def test_boundary_is_exactly_request_plus_window_plus_guard(self):
        """刚好够（103 = 8 + 90 + 5）时不压缩，差一期就压缩——两侧都断言，
        只测一侧的话把不等号方向写反也发现不了。"""
        self.assertEqual(bt.resolve_trials(103, 8, 90), 8)
        self.assertEqual(bt.resolve_trials(102, 8, 90), 20)

    def test_compresses_to_floor_when_history_is_short(self):
        self.assertEqual(bt.resolve_trials(80, 60, 90), 20)

    def test_floor_can_exceed_total(self):
        """压缩结果可能比总期数还大——所以 rolling_slices 还得自己夹一次。"""
        self.assertEqual(bt.resolve_trials(15, 60, 90), 20)


class RankOfTests(unittest.TestCase):

    def test_rank_starts_at_one(self):
        self.assertEqual(bt.rank_of('123', ['123', '456']), 1)
        self.assertEqual(bt.rank_of('456', ['123', '456']), 2)

    def test_missing_gets_miss_rank(self):
        """没排进来记 1001，比排最后（1000）更差且可区分。"""
        self.assertEqual(bt.rank_of('789', ['123', '456']), 1001)

    def test_miss_rank_is_above_full_space(self):
        self.assertEqual(bt.MISS_RANK, 1001)


class RankingBacktestTests(unittest.TestCase):
    """命中判定与聚合。"""

    def _accumulator(self, rng=None):
        return bt.RankingBacktest(top3_size=3, recommend_size=30,
                                  zu6_four_size=4, zu6_pool_size=6,
                                  rng=rng or random.Random(42))

    def test_denominator_is_evaluated_periods(self):
        """分母是**实际喂进来的期数**。迁移前用请求的期数，短历史上偏低。"""
        acc = self._accumulator()
        for _ in range(4):
            acc.observe((1, 2, 3), ['123'], ['123'], ['123'], ['123'])
        self.assertEqual(acc.trials, 4)
        self.assertEqual(acc.summarise(0.0, 0, {})['top3_rate'], 1.0)

    def test_raw_and_served_counted_separately(self):
        """raw 中了 served 没中时，两个数字必须不同——合成一个就分不清
        是模型退步还是那些后处理伤了它。"""
        acc = self._accumulator()
        acc.observe((1, 2, 3), ['123'], [], ['123'], ['999'])
        result = acc.summarise(0.0, 0, {})
        self.assertEqual(result['raw_top30_rate'], 1.0)
        self.assertEqual(result['served_top30_rate'], 0.0)
        self.assertEqual(result['top30_rate'], 0.0, '主指标取 served')

    def test_served_hit_alone_also_distinguished(self):
        """反过来的那一侧：只测一个方向，把两者写反照样全绿。"""
        acc = self._accumulator()
        acc.observe((1, 2, 3), ['123'], [], ['999'], ['123'])
        result = acc.summarise(0.0, 0, {})
        self.assertEqual(result['raw_top30_rate'], 0.0)
        self.assertEqual(result['served_top30_rate'], 1.0)

    def test_recent_rates_use_exactly_last_window(self):
        """近 N 期取的是**恰好** N 期。断言窗口外那期确实被排除掉了：
        只断言「窗口内的算进去了」，把窗口改大也发现不了。"""
        acc = self._accumulator()
        acc.observe((1, 2, 3), ['123'], [], ['123'], ['123'])   # 命中
        for _ in range(3):
            acc.observe((9, 9, 9), ['123'], [], ['123'], ['123'])  # 未命中
        self.assertEqual(acc.recent_rates(last_window=3), (0.0, 0.0))
        self.assertEqual(acc.recent_rates(last_window=4), (0.25, 0.25))

    def test_ge2_counts_shared_digits_with_served_pool(self):
        acc = self._accumulator()
        acc.observe((1, 2, 3), ['123'], [], [], ['129'])
        self.assertEqual(acc.summarise(0.0, 0, {})['ge2_digit_rate'], 1.0)

    def test_ge2_needs_two_not_one(self):
        acc = self._accumulator()
        acc.observe((1, 2, 3), ['123'], [], [], ['189'])
        self.assertEqual(acc.summarise(0.0, 0, {})['ge2_digit_rate'], 0.0)

    def test_zu6_random_source_advances_across_periods(self):
        """随机对照的抽样**跨期连续**。每期重置随机源的话，「随机」会变成
        同一组数字抽很多遍，方差被压掉，对照就失真了。"""
        acc = self._accumulator(rng=random.Random(1))
        first = []
        for _ in range(6):
            before = acc.rng.getstate()
            acc.observe_zu6((1, 2, 3), [1, 2, 3, 4], [1, 2, 3, 4, 5, 6])
            first.append(before != acc.rng.getstate())
        self.assertTrue(all(first), '每期都该消费随机源')

    def test_zu6_pool_and_four_are_separate_hits(self):
        acc = self._accumulator()
        acc.observe_zu6((1, 2, 9), [1, 2, 3, 4], [1, 2, 9, 4, 5, 6])
        result = acc.summarise(0.0, 0, {})
        self.assertEqual(result['zu6_four_rate'], 0.0)
        self.assertEqual(result['zu6_pool_rate'], 1.0)
        self.assertEqual(result['zu6_ge2_rate'], 1.0, '1 与 2 都在四码里')

    def test_zu6_rates_are_zero_without_zu6_draws(self):
        """一期组六都没有时分母是 0，不能崩也不能算成 100%。"""
        acc = self._accumulator()
        acc.observe((1, 1, 1), ['111'], ['111'], ['111'], ['111'])
        self.assertEqual(acc.summarise(0.0, 0, {})['zu6_four_rate'], 0.0)

    def test_is_zu6_matches_form(self):
        acc = self._accumulator()
        self.assertTrue(acc.is_zu6((1, 2, 3)))
        self.assertFalse(acc.is_zu6((1, 1, 3)))
        self.assertFalse(acc.is_zu6((1, 1, 1)))

    def test_median_rank_from_sorted_ranks(self):
        acc = self._accumulator()
        for key in ('999', '123', '555'):
            acc.observe(tuple(int(c) for c in key), ['123', '555', '999'], [], [], [])
        self.assertEqual(acc.summarise(0.0, 0, {})['actual_rank_median'], 2)

    def test_baselines_come_from_pool_sizes(self):
        """基准率是池子大小除以 1000，不是写死的 0.003/0.03。"""
        result = self._accumulator().summarise(0.0, 0, {})
        self.assertEqual(result['top3_rate_baseline'], 0.003)
        self.assertEqual(result['top30_rate_baseline'], 0.03)

    def test_empty_run_does_not_divide_by_zero(self):
        result = self._accumulator().summarise(0.0, 0, {})
        self.assertEqual(result['trials'], 0)
        self.assertEqual(result['top3_rate'], 0.0)
        self.assertEqual(result['actual_rank_avg'], 0.0)


class PermutationTests(unittest.TestCase):

    def test_pvalue_never_zero(self):
        """分子分母都加 1：观测自己也是一个排列。全部低于观测时 p 也不是 0,
        而 0 意味着「不可能」。"""
        result = bt.permutation_summary([0.0, 0.0, 0.0], 0.5, 0.03)
        self.assertEqual(result['pvalue'], 0.25)

    def test_pvalue_counts_ties_as_at_least(self):
        """打平算「不低于」。算成严格大于会让 p 偏小，结论偏乐观。"""
        self.assertEqual(bt.permutation_summary([0.5, 0.0], 0.5, 0.03)['pvalue'],
                         2 / 3)

    def test_significant_flag_at_threshold(self):
        """0.05 那条线的两侧都断言。"""
        rates = [0.0] * 19
        self.assertEqual(bt.permutation_summary(rates, 0.5, 0.03)['pvalue'], 0.05)
        self.assertFalse(bt.permutation_summary(rates, 0.5, 0.03)['significant'])
        self.assertTrue(bt.permutation_summary([0.0] * 20, 0.5, 0.03)['significant'])

    def test_empty_shuffles_is_safe(self):
        result = bt.permutation_summary([], 0.5, 0.03)
        self.assertEqual(result['pvalue'], 1.0)
        self.assertEqual(result['shuffled_mean_rate'], 0.0)

    def test_shuffled_series_preserves_multiset(self):
        """打乱只换顺序，不换内容——少一期或多一期都会让命中率没法比。"""
        series = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        for shuffled in bt.shuffled_series(series, 3, random.Random(0)):
            self.assertEqual(sorted(tuple(x) for x in shuffled), sorted(series))

    def test_shuffled_series_yields_requested_count(self):
        self.assertEqual(
            len(list(bt.shuffled_series([(1, 2, 3)] * 5, 4, random.Random(0)))), 4)


class ObjectiveTests(unittest.TestCase):

    RESULT = {'top3_rate': 0.2, 'top30_rate': 0.4, 'ge2_digit_rate': 0.6}

    def test_composite_weights_each_component(self):
        """三项权重写死在用例里，不引用被测的那个常量——引用的话把权重
        改坏、期望值跟着挪，照样全绿（判据 4）。"""
        self.assertAlmostEqual(bt.objective(self.RESULT, 'composite'),
                               0.55 * 0.2 + 0.30 * 0.4 + 0.15 * 0.6)

    def test_legacy_top_rate_is_an_alias(self):
        self.assertEqual(bt.objective(self.RESULT, 'top_rate'), 0.4)
        self.assertEqual(bt.objective(self.RESULT, 'top30_rate'), 0.4)

    def test_unknown_metric_raises(self):
        """静默回退到默认指标的话，搜出来的参数是在优化别的东西。"""
        with self.assertRaises(ValueError):
            bt.objective(self.RESULT, 'no_such_metric')


class WeightSamplingTests(unittest.TestCase):

    BASE = {'W_A': 2.0, 'SIG': 3.0}
    TUNABLE = ('W_A', 'SIG')
    RANGES = {'W_A': (0.5, 2.0), 'SIG': (2.0, 5.0)}
    ABSOLUTE = frozenset({'SIG'})

    def test_multiplier_key_scales_from_base(self):
        """倍率类：范围是相对基线的倍数，所以落在 [base*lo, base*hi]。"""
        for seed in range(20):
            got = ws.sample_weights(self.BASE, self.TUNABLE, self.RANGES,
                                    random.Random(seed), self.ABSOLUTE)
            self.assertGreaterEqual(got['W_A'], 1.0)
            self.assertLessEqual(got['W_A'], 4.0)

    def test_absolute_key_uses_range_as_value(self):
        """绝对类：范围就是取值本身。两条分支都测——只测一条的话，
        把 absolute_keys 判反了也发现不了。"""
        for seed in range(20):
            got = ws.sample_weights(self.BASE, self.TUNABLE, self.RANGES,
                                    random.Random(seed), self.ABSOLUTE)
            self.assertGreaterEqual(got['SIG'], 2.0)
            self.assertLessEqual(got['SIG'], 5.0)

    def test_sample_covers_every_tunable_key(self):
        got = ws.sample_weights(self.BASE, self.TUNABLE, self.RANGES,
                                random.Random(0), self.ABSOLUTE)
        self.assertEqual(set(got), {'W_A', 'SIG'})

    def test_mutate_changes_exactly_one_key(self):
        """一次只动一个键。动多个的话，分数变好了说不清是哪一项带来的。"""
        for seed in range(20):
            got = ws.mutate_weights(self.BASE, self.TUNABLE, self.RANGES,
                                    random.Random(seed), absolute_keys=self.ABSOLUTE)
            changed = [k for k in self.TUNABLE if got[k] != self.BASE[k]]
            self.assertEqual(len(changed), 1)

    def test_mutate_clamps_absolute_key_into_range(self):
        got = ws.mutate_weights({'W_A': 2.0, 'SIG': 4.99}, ('SIG',), self.RANGES,
                                random.Random(0), scale=10.0, absolute_keys=self.ABSOLUTE)
        self.assertGreaterEqual(got['SIG'], 2.0)
        self.assertLessEqual(got['SIG'], 5.0)

    def test_mutate_floors_multiplier_key(self):
        """倍率类只有下限，**没有上限**——这是迁移前的行为，refine 能把权重
        推到采样阶段根本到不了的地方。改掉会动线上搜索结果，所以留着，
        但用例把它钉住，免得下次有人以为它是被钳住的。"""
        got = ws.mutate_weights({'W_A': 0.01}, ('W_A',), self.RANGES,
                                random.Random(0), scale=10.0)
        self.assertGreaterEqual(got['W_A'], 0.1)

    def test_adapter_signature_dropped_the_unused_base(self):
        """迁移前的签名是 `_mutate_weights(weights, base, rng, scale)`，而
        `base` 在函数体里从头到尾没出现过——调用方却一直在认真地传。
        钉住参数名，免得它哪天又被加回来。"""
        self.assertEqual(
            list(inspect.signature(adapter._mutate_weights).parameters),
            ['weights', 'rng', 'scale'])


class SplitSeriesTests(unittest.TestCase):

    def test_test_segment_comes_last(self):
        """测试段永远在后面。随机切分会把未来漏进搜索里。"""
        series = list(range(10))
        train, test = ws.split_series(series, 0.3)
        self.assertEqual(train, list(range(7)))
        self.assertEqual(test, [7, 8, 9])

    def test_zero_ratio_keeps_everything_for_training(self):
        train, test = ws.split_series(list(range(10)), 0.0)
        self.assertEqual(len(train), 10)
        self.assertEqual(test, [])


class SearchTests(unittest.TestCase):
    """搜索循环本身：用假的评估函数，不跑真回测。"""

    TUNABLE = ('W_A',)
    RANGES = {'W_A': (0.5, 2.0)}
    BASE = {'W_A': 1.0}

    def test_baseline_wins_when_nothing_beats_it(self):
        """采样了几十组还不如什么都不改，这个结论必须留得住。"""
        outcome = ws.search(self.BASE, self.TUNABLE, self.RANGES,
                            lambda w: (0.0 if w != self.BASE else 1.0, {}),
                            random.Random(0), iterations=5, refine_rounds=5)
        self.assertEqual(outcome['best']['weights'], self.BASE)
        self.assertEqual(outcome['improvement'], 0.0)

    def test_ties_do_not_replace_best(self):
        """严格大于才换。相等就换的话，等值平台上报出来的「最优参数」
        是随机的哪一个。"""
        outcome = ws.search(self.BASE, self.TUNABLE, self.RANGES,
                            lambda w: (0.5, {}), random.Random(0),
                            iterations=5, refine_rounds=5)
        self.assertEqual(outcome['best']['weights'], self.BASE)

    def test_better_candidate_replaces_best(self):
        scores = iter([0.1] + [0.9] + [0.2] * 20)

        def evaluate(weights):
            return next(scores), {}

        outcome = ws.search(self.BASE, self.TUNABLE, self.RANGES, evaluate,
                            random.Random(0), iterations=3, refine_rounds=0)
        self.assertNotEqual(outcome['best']['weights'], self.BASE)
        self.assertAlmostEqual(outcome['improvement'], 0.8)

    def test_history_records_both_phases(self):
        outcome = ws.search(self.BASE, self.TUNABLE, self.RANGES,
                            lambda w: (0.0, {}), random.Random(0),
                            iterations=2, refine_rounds=3)
        phases = [entry['phase'] for entry in outcome['history']]
        self.assertEqual(phases, ['random'] * 2 + ['refine'] * 3)

    def test_refine_builds_on_current_best_not_base(self):
        """refine 从当前最优出发，不是每次从基线重来——否则它就只是
        第二轮随机采样。"""
        seen = []

        def evaluate(weights):
            seen.append(weights['W_A'])
            return len(seen) * 0.1, {}

        ws.search(self.BASE, self.TUNABLE, self.RANGES, evaluate,
                  random.Random(0), iterations=1, refine_rounds=2)
        # 每次评估分数都更高，所以每一步的 best 都被换掉；最后一次 refine
        # 必须是在上一次 refine 的结果上扰动，而不是在基线上
        self.assertNotEqual(seen[-1], seen[0])

    def test_on_improve_only_fires_on_improvement(self):
        calls = []
        ws.search(self.BASE, self.TUNABLE, self.RANGES, lambda w: (0.0, {}),
                  random.Random(0), iterations=3, refine_rounds=3,
                  on_improve=lambda *args: calls.append(args))
        self.assertEqual(calls, [])


class DanKillBacktestTests(unittest.TestCase):

    def test_kill_counts_failures_not_hits(self):
        """杀码统计的是**失手**：它出现在开奖号里就是错了。写成命中率
        会让好坏方向反过来，而数字本身看不出来。"""
        acc = cbt.DanKillBacktest()
        acc.observe((1, 2, 3), dan=[1], kill=[3])
        self.assertEqual(acc.summarise()['kill_fail_rate'], 1.0)

    def test_kill_untouched_is_a_success(self):
        acc = cbt.DanKillBacktest()
        acc.observe((1, 2, 3), dan=[1], kill=[9])
        self.assertEqual(acc.summarise()['kill_fail_rate'], 0.0)

    def test_dan_tiers_are_cumulative(self):
        """中两个也算中了至少一个。两档分开数会让 hit2 > hit1 这种
        不可能的组合悄悄出现。"""
        acc = cbt.DanKillBacktest()
        acc.observe((1, 2, 3), dan=[1, 2], kill=[])
        result = acc.summarise()
        self.assertEqual(result['dan_hit1_rate'], 1.0)
        self.assertEqual(result['dan_hit2_rate'], 1.0)

    def test_one_hit_does_not_count_as_two(self):
        acc = cbt.DanKillBacktest()
        acc.observe((1, 2, 3), dan=[1, 9], kill=[])
        result = acc.summarise()
        self.assertEqual(result['dan_hit1_rate'], 1.0)
        self.assertEqual(result['dan_hit2_rate'], 0.0)

    def test_denominator_is_observed_periods(self):
        acc = cbt.DanKillBacktest()
        for _ in range(3):
            acc.observe((1, 2, 3), dan=[1], kill=[])
        self.assertEqual(acc.summarise()['trials'], 3)

    def test_empty_run_is_safe(self):
        self.assertEqual(cbt.DanKillBacktest().summarise()['dan_hit1_rate'], 0.0)


class FormBacktestTests(unittest.TestCase):

    def test_precision_denominator_is_predictions_not_periods(self):
        """精确率的分母是「预测成这个形态的期数」。拿总期数当分母的话，
        一个永远猜组六的模型看起来也很准——组六本来就占七成。"""
        acc = cbt.FormBacktest(('zu6', 'zu3'))
        acc.observe('zu6', 'zu6')
        acc.observe('zu3', 'zu6')
        acc.observe('zu3', 'zu3')
        result = acc.summarise()
        self.assertEqual(result['zu6_precision'], 0.5, '预测两次组六，中一次')
        self.assertEqual(result['zu3_precision'], 1.0, '预测一次组三，中一次')
        self.assertAlmostEqual(result['form_top1_rate'], 2 / 3)

    def test_untracked_prediction_still_counts_toward_top1(self):
        """豹子不在跟踪列表里，但它猜对了也该算进整体命中。"""
        acc = cbt.FormBacktest(('zu6', 'zu3'))
        acc.observe('baozi', 'baozi')
        self.assertEqual(acc.summarise()['form_top1_rate'], 1.0)

    def test_never_predicted_form_has_zero_precision(self):
        acc = cbt.FormBacktest(('zu6', 'zu3'))
        acc.observe('zu6', 'zu6')
        self.assertEqual(acc.summarise()['zu3_precision'], 0.0)


class SumSpanBacktestTests(unittest.TestCase):

    def test_tolerance_tiers_are_nested(self):
        """容差 2 命中的必然也在容差 3、4 里。三档一起报是为了看衰减多快。"""
        acc = cbt.SumSpanBacktest()
        acc.observe((1, 2, 3), sum_center=8, span_center=2)  # 和 6，差 2；跨度 2，差 0
        result = acc.summarise()
        for tol in (2, 3, 4):
            self.assertEqual(result[f'sum_hit_{tol}_rate'], 1.0)
        for tol in (1, 2):
            self.assertEqual(result[f'span_hit_{tol}_rate'], 1.0)

    def test_outside_tightest_tier_only(self):
        """差 3：容差 2 不算命中，容差 3、4 算——边界的两侧都要断言。"""
        acc = cbt.SumSpanBacktest()
        acc.observe((1, 2, 3), sum_center=9, span_center=9)
        result = acc.summarise()
        self.assertEqual(result['sum_hit_2_rate'], 0.0)
        self.assertEqual(result['sum_hit_3_rate'], 1.0)
        self.assertEqual(result['sum_hit_4_rate'], 1.0)
        self.assertEqual(result['span_hit_2_rate'], 0.0)

    def test_span_uses_max_minus_min(self):
        acc = cbt.SumSpanBacktest()
        acc.observe((0, 5, 9), sum_center=14, span_center=9)
        self.assertEqual(acc.summarise()['span_hit_1_rate'], 1.0)


class SlopeBacktestTests(unittest.TestCase):

    def test_denominator_is_signal_count_not_periods(self):
        """分母是**信号条数**。一期可能出好几条，也可能一条都没有；
        拿期数当分母，信号少的那段时间命中率会莫名其妙地低。"""
        acc = cbt.SlopeBacktest()
        acc.observe((1, 2, 3), [
            {'type': 'position_slope', 'position': 0, 'predict_digit': 1},
            {'type': 'position_slope', 'position': 1, 'predict_digit': 9},
        ])
        acc.observe((4, 5, 6), [])
        result = acc.summarise()
        self.assertEqual(result['trials'], 2, '期数照记')
        self.assertEqual(result['position_slope_total'], 2, '分母是信号条数')
        self.assertEqual(result['position_slope_rate'], 0.5)

    def test_two_kinds_counted_separately(self):
        """同位与跨期分开。混在一起会把强的那类摊薄。"""
        acc = cbt.SlopeBacktest()
        acc.observe((1, 2, 3), [
            {'type': 'position_slope', 'position': 0, 'predict_digit': 1},
            {'type': 'cross_period_slope', 'position': 0, 'predict_digit': 9},
        ])
        result = acc.summarise()
        self.assertEqual(result['position_slope_rate'], 1.0)
        self.assertEqual(result['cross_slope_rate'], 0.0)

    def test_signal_without_position_counts_as_miss_not_skip(self):
        """缺 position 的信号预测不到具体分位，记未命中而**不是跳过**——
        跳过等于把它从分母里也拿掉，命中率会虚高。"""
        acc = cbt.SlopeBacktest()
        acc.observe((1, 2, 3), [{'type': 'position_slope', 'predict_digit': 1}])
        result = acc.summarise()
        self.assertEqual(result['position_slope_total'], 1)
        self.assertEqual(result['position_slope_rate'], 0.0)

    def test_unknown_signal_type_is_ignored(self):
        acc = cbt.SlopeBacktest()
        acc.observe((1, 2, 3), [{'type': 'something_else', 'position': 0,
                                 'predict_digit': 1}])
        result = acc.summarise()
        self.assertEqual(result['position_slope_total'], 0)
        self.assertEqual(result['cross_slope_total'], 0)

    def test_baseline_is_one_over_digit_space(self):
        """十个数字里猜中一个。写死 0.10 在用例里，不引用被测的常量。"""
        self.assertEqual(cbt.SlopeBacktest().summarise()['baseline_single_pos'], 0.1)


class RandomBaselineTests(unittest.TestCase):

    def test_same_seed_reproduces(self):
        """对照必须可复现，否则两次回测的差值里混着运气。"""
        series = [(1, 2, 3)] * 30
        first = bt.random_baseline(series, 20, 30, random.Random(42))
        second = bt.random_baseline(series, 20, 30, random.Random(42))
        self.assertEqual(first, second)

    def test_larger_pool_hits_more_often(self):
        """抽 500 注比抽 30 注更容易中——方向反了说明命中判定写错了。"""
        series = [(a % 10, b % 10, (a + b) % 10)
                  for a in range(10) for b in range(10)]
        small = bt.random_baseline(series, 50, 30, random.Random(1))
        large = bt.random_baseline(series, 50, 500, random.Random(1))
        self.assertGreater(large['random_rate'], small['random_rate'])

    def test_trials_reports_evaluated_periods(self):
        result = bt.random_baseline([(1, 2, 3)] * 5, 99, 30, random.Random(0))
        self.assertEqual(result['trials'], 5)


adapter = importlib.import_module('src.lottery3d.backtest')  # noqa: E402
config = importlib.import_module('src.lottery3d.config')  # noqa: E402
features = importlib.import_module('src.lottery3d.features')  # noqa: E402
scoring = importlib.import_module('src.lottery3d.scoring')  # noqa: E402
