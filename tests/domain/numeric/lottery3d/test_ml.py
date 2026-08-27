"""福彩 3D 的机器学习层：54 维特征、正负采样、纯 Python 降级模型、集成加权。

参照物是从迁移前的 `ml.py` 生成的黄金文件
（`tests/fixtures/golden/lottery3d_ml.json.gz`，422 条），**逐条相同**——
端到端的三模型集成预测与 ML 回测也在里面。语料按四种长度的历史
（100 / 30 / 3 / 1 期）× 十二个覆盖各种形态的三元组铺开，另含数据不足、
单期历史这些边界。

黄金比对不到的三处由手写用例守着，它们各自对应一个**迁移前就存在、
但任何测试和任何线上请求都碰不到**的缺陷：

1. **降级路径必然崩。** `train_ensemble` 给模块级的告警标志赋值却没写
   `global`，Python 于是把它当局部变量，读它的那行必抛 `UnboundLocalError`。
   线上和 CI 都装了三个库，所以这条路从来没走过——它被上层的 except
   吞成了「训练失败」。
2. **数据不足的守卫从没拦下过东西。** `build_training_data` 在数据不足时
   返回四个 `None`，而调用方判的是 `if result is None or len(result) < 4`：
   四元组既不是 None，长度也正好是 4。越过守卫之后才在 `len(None)` 上炸。
   实测旧实现 `backtest_ml(train_window=60)` 抛
   `TypeError: object of type 'NoneType' has no len()`。
3. **特征名与特征值曾是两份手工对齐的列表。** 错位不会报错，只会让页面上
   的「特征重要性」张冠李戴。现在两者来自同一次遍历。
"""
import gzip
import importlib
import json
import pathlib
import random
import unittest
from unittest import mock

from src.domain.numeric.lottery3d import ml_features as features
from src.domain.numeric.lottery3d import ml_forest as forest
from src.domain.numeric.lottery3d import ml_training as training
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_ml.json.gz')
NUMBERS = [tuple(r['digits'])
           for r in _load('numeric/lottery3d_history.json.gz')['results']]

SERIES = {'s300': NUMBERS[-300:], 's150': NUMBERS[-150:], 's120': NUMBERS[-120:]}
FEATURE_SERIES = {'s100': NUMBERS[-100:], 's30': NUMBERS[-30:],
                  's3': NUMBERS[-3:], 's1': NUMBERS[-1:]}
TRIPLES = [(0, 0, 0), (9, 9, 9), (1, 2, 3), (3, 2, 1), (0, 5, 9), (4, 4, 7),
           (7, 4, 4), (5, 6, 7), (9, 0, 1), (2, 2, 8), (6, 6, 6), (0, 1, 2)]

# 迁移当时 ml.py 里生效的那组值，写死不 import（判据 12）
SETTINGS = features.FeatureSettings(
    history_window=100, decay=0.96, markov_alpha=1.0, fallback_prob=0.1,
    hot_ratio=1.3, warm_ratio=0.7, trend_window=20,
    default_sum_mean=13.5, default_span_mean=4.5, default_deviation=1.0)
FEATURE_COUNT = 54


def _key(triple):
    return ''.join(map(str, triple))


def golden_entries():
    """按 (键, 值) 逐条产出全部语料，测试与重生成脚本共用。"""
    yield from _basic_entries()
    yield from _feature_entries()
    yield from _training_entries()
    yield from _model_entries()
    yield from _end_to_end_entries()


def _basic_entries():
    from src.domain.numeric.lottery3d import draw, history
    for name, series in FEATURE_SERIES.items():
        for digit in range(10):
            yield f'miss:{name}:{digit}', history.miss_value(series, digit)
            for position in range(3):
                yield (f'miss_pos:{name}:{digit}:{position}',
                       history.miss_value(series, digit, position=position))
        for position in range(3):
            yield (f'markov:{name}:{position}',
                   {str(k): dict(v) for k, v in history.build_markov(series, position).items()})
        yield (f'expw:{name}',
               dict(history.exp_weighted_counts(
                   [d for n in series for d in n], SETTINGS.decay)))
    for digit in range(10):
        yield f'neighbor:{digit}', sorted(draw.neighbor(digit))
        yield f'road:{digit}', draw.road(digit)
    for triple in TRIPLES:
        yield f'oe:{_key(triple)}', draw.odd_even_key(triple)
        yield f'bs:{_key(triple)}', draw.big_small_key(triple)
        yield f'consec:{_key(triple)}', draw.has_consecutive_digits(*triple)
        yield f'form:{_key(triple)}', draw.classify_form(triple)
        for last in ((1, 2, 3), (0, 0, 0), (3, 2, 1)):
            yield (f'posrep:{_key(triple)}:{_key(last)}',
                   sum(1 for i in range(3) if triple[i] == last[i]))
            yield (f'overlap:{_key(triple)}:{_key(last)}',
                   features.distinct_digit_overlap(triple, last))
    for row in ({}, {1: 3, 2: 1}, {5: 10}):
        for alpha in (0.5, 1.0, 2.0):
            yield (f'markovp:{sorted(row.items())}:{alpha}',
                   history.markov_prob_smoothed(row, range(10), alpha))


def _feature_entries():
    for name, series in FEATURE_SERIES.items():
        engineer = features.FeatureEngineer(series, SETTINGS)
        for triple in TRIPLES:
            yield f'features:{name}:{_key(triple)}', engineer.build_features(*triple)
        yield f'feature_names:{name}', engineer.get_feature_names()
        yield f'tier:{name}', engineer.digit_tier
        yield (f'intervals:{name}',
               {d: [s['mean'], s['count']] for d, s in engineer.interval_stats.items()})
        yield (f'sumstats:{name}',
               [engineer.sum_mean, engineer.sum_std, engineer.sum_trend,
                engineer.span_mean, engineer.span_std])
        yield f'consec_rate:{name}', engineer.consec_rate
        yield f'streaks:{name}', engineer.form_streaks


def _training_entries():
    for name in ('s150', 's120'):
        for neg in (5, 20):
            samples = adapter.build_training_data(SERIES[name], neg_samples=neg,
                                                  rng=random.Random(7))
            yield f'train_data:{name}:{neg}', {
                'rows': len(samples.rows), 'pos': sum(samples.labels),
                'weights': sorted(set(samples.weights)),
                'groups': len(set(samples.groups)),
                'first_row': samples.rows[0], 'last_row': samples.rows[-1],
                'weight_seq': samples.weights[:12], 'group_seq': samples.groups[:12],
            }


def _model_entries():
    samples = adapter.build_training_data(SERIES['s120'], neg_samples=5,
                                          rng=random.Random(7))
    rows, labels = samples.rows[:200], samples.labels[:200]
    for depth in (2, 4):
        tree = forest.DecisionTree(max_depth=depth, min_samples_split=10,
                                   rng=random.Random(3)).fit(rows, labels)
        yield f'tree:{depth}', tree.predict(rows[:30])
        yield f'tree_struct:{depth}', tree.tree
    for count in (2, 5):
        trees = forest.RandomForest(n_trees=count, max_depth=3, min_samples_split=10,
                                    rng=random.Random(3)).fit(rows, labels)
        yield f'forest:{count}', trees.predict(rows[:30])

    names = features.FeatureEngineer([], SETTINGS).get_feature_names()
    for ratio in (0.5, 0.85):
        indices, selected = adapter.select_features(rows, labels, names, keep_ratio=ratio)
        # **不固定选中了哪几维**——那是 sklearn 的互信息给的顺序，换个版本就变。
        # 这里固定的是我们自己那部分：保留几维、索引是否映射回了原始维度、
        # 名字与下标是否对得上
        yield f'select:{ratio}', {
            'count': len(indices),
            'unique': len(set(indices)) == len(indices),
            'within_range': all(0 <= index < FEATURE_COUNT for index in indices),
            'names_match_indices': selected == [names[index] for index in indices],
        }
    for probabilities, truth in (([0.9, 0.1], [1, 0]), ([0.5, 0.5], [1, 0]),
                                 ([0.0, 1.0], [1, 0]), ([], [])):
        yield (f'valscore:{probabilities}',
               training.validation_score(truth, probabilities))

    for combo in (((0.2, 1.0),), ((0.2, 1.0), (0.8, 3.0)), ((0.1, 0.0), (0.9, 0.0))):
        models = [(_Stub(value), f'm{i}', score)
                  for i, (value, score) in enumerate(combo)]
        yield f'ensemble:{combo}', adapter.ensemble_predict(models, [[0]] * 3)


def _end_to_end_entries():
    saved = []
    with mock.patch.object(adapter, 'save_ml_backtest_history', saved.append):
        for name in ('s300', 's150'):
            # **模型算出来的数不进黄金**（分数、特征重要性、融合权重）——
            # 它们随 catboost/xgboost/lightgbm 的版本变化，而 CI 装的是
            # `requirements.txt` 里 `>=` 允许的更新版本。把它们钉进黄金，
            # 库一升级 CI 就红，而那不是回归——「失败集合不超出基线」
            # 这条判据会因此失效。模型无关的那条链路由打桩用例覆盖。
            yield f'predict:{name}', _model_free(
                adapter.predict_current(SERIES[name], top_k=5, neg_samples=10))
        yield 'predict:short', adapter.predict_current(NUMBERS[-50:], top_k=5)
        # 同上：命中数与平均排名直接来自模型预测，只固定结构性字段
        yield 'backtest_ml', _model_free(adapter.backtest_ml(
            SERIES['s300'], trials=2, train_window=120,
            base_period='2026228', neg_samples=10))
        yield 'backtest_ml:saved', [_model_free(record) for record in saved]
        yield 'backtest_ml:short', adapter.backtest_ml(NUMBERS[-50:], trials=2,
                                                       train_window=120)


# 这些字段的值由三个梯度提升库算出，随库版本变化
MODEL_DEPENDENT = frozenset({
    'recommendations', 'top3', 'feature_importance', 'model_weights',
    'top3_hit', 'top3_rate', 'top15_hit', 'top15_rate',
    'top30_hit', 'top30_rate', 'top100_hit', 'top100_rate',
    'actual_rank_avg', 'actual_rank_median',
})


def _model_free(result):
    """剥掉随库版本变化的字段，并把推荐替换成「几注 + 格式合法吗」。

    留下的是我们自己那部分：采样出了多少样本、集成了几个模型、
    结构性的基准率、错误分支的措辞。
    """
    stripped = {key: value for key, value in result.items()
                if key not in MODEL_DEPENDENT}
    if 'recommendations' in result:
        picks = result['recommendations']
        stripped['recommendation_count'] = len(picks)
        stripped['nums_well_formed'] = all(
            len(row['num']) == 3 and row['num'].isdigit() for row in picks)
        stripped['scores_descend'] = all(
            picks[i]['model_score'] >= picks[i + 1]['model_score']
            for i in range(len(picks) - 1))
        stripped['shares_sum_to_one'] = round(
            sum(row['topk_score_share'] for row in picks), 2) == 1.0
    return stripped


class _Stub:
    """打桩模型：集成加权不该依赖三个第三方库的输出，否则库一升级，
    「加权算错了」和「库换了个数」在测试里长得一模一样。"""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class FeatureAlignmentTests(unittest.TestCase):
    """特征名与特征值的对齐——迁移前它们是两份手工维护的列表。"""

    def setUp(self):
        self.engineer = features.FeatureEngineer(NUMBERS[-100:], SETTINGS)

    def test_names_and_values_have_the_same_length(self):
        self.assertEqual(len(self.engineer.get_feature_names()), FEATURE_COUNT)
        self.assertEqual(len(self.engineer.build_features(1, 2, 3)), FEATURE_COUNT)

    def test_names_come_from_the_same_pass_as_values(self):
        """`describe` 产出成对的名与值，`build_features` 与
        `get_feature_names` 各取一半——**错位在结构上不可能**。"""
        described = self.engineer.describe((1, 2, 3))
        self.assertEqual([name for name, _ in described],
                         self.engineer.get_feature_names())
        self.assertEqual([value for _, value in described],
                         self.engineer.build_features(1, 2, 3))

    def test_names_are_unique(self):
        """重名会让特征重要性表里两行叫同一个名字，谁也说不清是哪一个。"""
        names = self.engineer.get_feature_names()
        self.assertEqual(len(names), len(set(names)))

    def test_names_do_not_depend_on_the_probe_triple(self):
        """名字与喂进去的那注号码无关。"""
        self.assertEqual([n for n, _ in self.engineer.describe((0, 0, 0))],
                         [n for n, _ in self.engineer.describe((9, 8, 7))])

    def test_empty_history_still_yields_all_names(self):
        """空历史也要能拿到完整名单——`train_ensemble` 就是这么取名字的。"""
        self.assertEqual(len(features.FeatureEngineer([], SETTINGS).get_feature_names()),
                         FEATURE_COUNT)

    def test_accepts_both_call_shapes(self):
        self.assertEqual(self.engineer.build_features(1, 2, 3),
                         self.engineer.build_features((1, 2, 3)))


class OverlapSemanticsTests(unittest.TestCase):
    """两种「重合度」是两个不同的量，不是同一件事写了两遍。"""

    def test_distinct_overlap_ignores_multiplicity(self):
        """ML 特征按集合算：`111` 与 `111` 只共有一个数字。"""
        self.assertEqual(features.distinct_digit_overlap((1, 1, 1), (1, 1, 1)), 1)

    def test_multiset_overlap_counts_multiplicity(self):
        """回测按重数算，同一对输入给 3——两者必须不同，
        否则说明其中一个被替换成了另一个。"""
        from src.domain.numeric.lottery3d import draw
        self.assertEqual(draw.digit_overlap((1, 1, 1), (1, 1, 1)), 3)

    def test_cross_position_reuse_excludes_same_position(self):
        """跨位复用不数「原地不动」的那些。"""
        self.assertEqual(features.cross_position_reuse((1, 2, 3), (1, 3, 2)), 2)

    def test_cross_position_reuse_is_zero_when_all_stay(self):
        self.assertEqual(features.cross_position_reuse((1, 2, 3), (1, 2, 3)), 0)

    def test_size_category_boundaries(self):
        """分界写死在用例里：2 和 5 各自属于低的那一档。"""
        self.assertEqual([features.size_category(d) for d in range(10)],
                         [1, 1, 1, 2, 2, 2, 3, 3, 3, 3])


class TierTests(unittest.TestCase):
    """冷热分档：阈值是相对均值的倍数。"""

    def _tiers(self, series):
        return features.FeatureEngineer(series, SETTINGS).digit_tier

    def test_all_three_tiers_occur_on_real_history(self):
        """真实历史上三档都要出现。**热号与冷号是常态**——若结果只剩一档，
        说明阈值或分母算错了，而那不会报错，只会让这个特征失去区分度。"""
        self.assertEqual(set(self._tiers(NUMBERS[-100:]).values()),
                         {features.COLD, features.WARM, features.HOT})

    def test_only_the_drawn_digits_are_hot(self):
        """频次是**指数加权**的，不是原始计数——所以「出现次数一样多」
        并不意味着分档一样。这里只开一注，三个数字全热、其余全冷。"""
        tiers = self._tiers([(1, 2, 3)] * 30)
        self.assertEqual({digit for digit, tier in tiers.items() if tier == features.HOT},
                         {1, 2, 3})
        self.assertEqual({digit for digit, tier in tiers.items() if tier == features.COLD},
                         {0, 4, 5, 6, 7, 8, 9})

    def test_dominant_digit_is_hot_and_absent_is_cold(self):
        series = [(7, 7, 7)] * 20 + [(1, 2, 3)]
        tiers = self._tiers(series)
        self.assertEqual(tiers[7], features.HOT)
        self.assertEqual(tiers[9], features.COLD, '从没出现过的必然是冷号')

    def test_exactly_at_the_threshold_counts_as_hot(self):
        """**恰好达到阈值算热号**（`>=` 而不是 `>`）。

        构造成不衰减 + 阈值 1.0，十个数字各出现三次、均值也是三次——
        每个数字都恰好压在线上。真实语料里浮点值不会正好落在阈值上，
        所以这条边界只能这样专门造出来。"""
        settings = SETTINGS._replace(decay=1.0, hot_ratio=1.0)
        series = [(digit, digit, digit) for digit in range(10)]
        tiers = features.FeatureEngineer(series, settings).digit_tier
        self.assertEqual(set(tiers.values()), {features.HOT})

    def test_just_below_the_threshold_is_not_hot(self):
        """反方向：差一点点就不算热号。"""
        settings = SETTINGS._replace(decay=1.0, hot_ratio=1.0001)
        series = [(digit, digit, digit) for digit in range(10)]
        tiers = features.FeatureEngineer(series, settings).digit_tier
        self.assertNotIn(features.HOT, tiers.values())

    def test_thresholds_are_ratios_not_counts(self):
        """把窗口整体拉长十倍，分档结果不变——阈值若写成绝对次数就会变。"""
        short = [(7, 7, 7), (1, 2, 3)] * 5
        long_series = [(7, 7, 7), (1, 2, 3)] * 50
        self.assertEqual(self._tiers(short), self._tiers(long_series))


class TrendTests(unittest.TestCase):

    def test_short_history_falls_back_to_defaults(self):
        """历史短于趋势窗口时用兜底值。**标准差不能是 0**——
        偏离度要拿它做分母。"""
        engineer = features.FeatureEngineer([], SETTINGS)
        self.assertEqual(engineer.sum_mean, 13.5)
        self.assertEqual(engineer.sum_std, 1.0)
        self.assertEqual(engineer.sum_trend, 0.0)

    def test_flat_series_has_zero_trend(self):
        engineer = features.FeatureEngineer([(1, 2, 3)] * 30, SETTINGS)
        self.assertEqual(engineer.sum_trend, 0.0)

    def test_rising_series_has_positive_trend(self):
        """和值一路上升时斜率为正。方向反了不会报错，只会让模型学反。"""
        series = [(0, 0, value % 10) for value in range(30)]
        self.assertGreater(features.FeatureEngineer(series, SETTINGS).sum_trend, 0)

    def test_falling_series_has_negative_trend(self):
        series = [(0, 0, (30 - value) % 10) for value in range(30)]
        self.assertLess(features.FeatureEngineer(series, SETTINGS).sum_trend, 0)


class IntervalTests(unittest.TestCase):

    def test_trailing_wait_is_not_counted_as_a_gap(self):
        """末尾那段还没结束的等待不算一个完整间隔——算进去会系统性
        拉高均值，而这个均值正是判断「超期没超期」的分母。"""
        series = [(1, 1, 1), (2, 2, 2), (1, 1, 1), (3, 3, 3), (4, 4, 4)]
        stats = features.FeatureEngineer(series, SETTINGS).interval_stats[1]
        self.assertEqual(stats['count'], 1)
        self.assertEqual(stats['mean'], 2)

    def test_never_seen_digit_falls_back_to_window_length(self):
        series = [(1, 1, 1)] * 6
        stats = features.FeatureEngineer(series, SETTINGS).interval_stats[9]
        self.assertEqual(stats['count'], 0)
        self.assertEqual(stats['mean'], 6)


class StreakTests(unittest.TestCase):

    def test_streaks_are_contiguous_runs(self):
        series = [(1, 2, 3), (4, 5, 6), (1, 1, 2), (7, 7, 8), (7, 7, 7)]
        self.assertEqual(features.FeatureEngineer(series, SETTINGS).form_streaks,
                         [('zu6', 2), ('zu3', 2), ('baozi', 1)])

    def test_empty_history_has_no_streaks(self):
        self.assertEqual(features.FeatureEngineer([], SETTINGS).form_streaks, [])


class TrainingSetTests(unittest.TestCase):

    SETTINGS = training.SamplingSettings(
        neg_samples=5, min_history=3, feature_window=10,
        decay_tiers=((2, 2.0), (4, 1.4)), base_weight=1.0)

    def _build(self, series, rng=None):
        return training.build_training_samples(
            series, lambda history: features.FeatureEngineer(history, SETTINGS),
            self.SETTINGS, rng or random.Random(0))

    def test_short_history_returns_an_empty_set_not_none(self):
        """迁移前返回四个 None，调用方的 `len(result) < 4` 拦不住
        （长度正好是 4），越过之后才在 `len(None)` 上炸。"""
        samples = self._build(NUMBERS[-3:])
        self.assertFalse(samples)
        self.assertEqual(samples.rows, [])
        self.assertEqual(len(samples), 4, '仍是四元组，所以旧守卫确实拦不住')

    def test_exactly_min_history_yields_nothing(self):
        """**恰好** min_history 期时产出空——多一期才开始有样本。
        两侧都断言，只测一侧的话把 `<=` 写成 `<` 也发现不了。"""
        self.assertFalse(self._build(NUMBERS[-3:]))
        self.assertTrue(self._build(NUMBERS[-4:]))

    def test_one_positive_per_period(self):
        samples = self._build(NUMBERS[-8:])
        self.assertEqual(sum(samples.labels), len(set(samples.groups)))

    def test_negatives_per_period_match_the_quota(self):
        samples = self._build(NUMBERS[-8:])
        periods = len(set(samples.groups))
        self.assertEqual(len(samples.labels) - periods, periods * 5)

    def test_actual_draw_is_never_a_negative(self):
        """开出来的那注不能同时当负例——那等于告诉模型「它既开了又没开」。"""
        series = NUMBERS[-8:]
        samples = self._build(series)
        for period in sorted(set(samples.groups)):
            actual = series[period]
            engineer = features.FeatureEngineer(
                series[max(0, period - 10):period], SETTINGS)
            actual_row = engineer.build_features(actual)
            negatives = [row for row, label, group
                         in zip(samples.rows, samples.labels, samples.groups)
                         if group == period and label == 0]
            self.assertNotIn(actual_row, negatives)

    def test_recent_periods_weigh_more(self):
        samples = self._build(NUMBERS[-8:])
        by_period = dict(zip(samples.groups, samples.weights))
        newest, oldest = max(by_period), min(by_period)
        self.assertGreater(by_period[newest], by_period[oldest])

    def test_same_rng_seed_reproduces(self):
        """同一份数据两次构造必须一样，否则回测的差异里混着采样噪声。"""
        self.assertEqual(self._build(NUMBERS[-8:], random.Random(1)).rows,
                         self._build(NUMBERS[-8:], random.Random(1)).rows)

    def test_time_decay_tier_boundaries(self):
        """两档的边界各测两侧。只测一侧的话把 `<=` 写成 `<` 也发现不了。"""
        tiers = ((30, 2.0), (60, 1.4))
        self.assertEqual(training.time_decay_weight(30, tiers, 1.0), 2.0)
        self.assertEqual(training.time_decay_weight(31, tiers, 1.0), 1.4)
        self.assertEqual(training.time_decay_weight(60, tiers, 1.0), 1.4)
        self.assertEqual(training.time_decay_weight(61, tiers, 1.0), 1.0)


class StratifiedQuotaTests(unittest.TestCase):

    def test_every_layer_gets_at_least_one(self):
        """和值 0 那层只有一个组合，按比例分配会得到 0，于是极端和值
        永远不进训练集——模型就学不到「它们也会开」。"""
        sizes = {0: 1, 1: 3, 2: 500}
        quota = training.stratified_quota(sizes, 30, 504, random.Random(0))
        self.assertGreaterEqual(quota[0], 1)

    def test_total_never_exceeds_the_request(self):
        sizes = {index: 50 for index in range(10)}
        quota = training.stratified_quota(sizes, 30, 500, random.Random(0))
        self.assertLessEqual(sum(quota.values()), 30)

    def test_never_asks_a_layer_for_more_than_it_has(self):
        sizes = {0: 2, 1: 2}
        quota = training.stratified_quota(sizes, 100, 4, random.Random(0))
        self.assertEqual(quota, {0: 2, 1: 2})

    def test_spends_the_whole_budget_when_supply_allows(self):
        sizes = {index: 50 for index in range(10)}
        quota = training.stratified_quota(sizes, 30, 500, random.Random(0))
        self.assertEqual(sum(quota.values()), 30)


class SplitTests(unittest.TestCase):

    def test_split_keeps_a_period_whole(self):
        """按期切：同一期的样本不能被劈到两边——它们共享全部历史统计量，
        劈开等于把答案漏给验证集。"""
        groups = [period for period in range(20) for _ in range(6)]
        train, valid = training.split_by_period(groups, len(groups))
        train_periods = {groups[index] for index in train}
        valid_periods = {groups[index] for index in valid}
        self.assertFalse(train_periods & valid_periods)

    def test_validation_comes_from_the_later_periods(self):
        """验证段必须在后面。前后颠倒会让模型拿未来去验过去。"""
        groups = [period for period in range(20) for _ in range(6)]
        train, valid = training.split_by_period(groups, len(groups))
        self.assertLess(max(groups[i] for i in train), min(groups[i] for i in valid))

    def test_too_few_periods_declines_to_split(self):
        """期数不够就不切。

        **六期而不是五期**：五期时 `int(5 * 0.8)` 正好落在最后一期，
        验证段为空，于是真正拦下它的是「验证集最少样本数」那道判断——
        用五期的话，这条用例测的根本不是它自己声称要测的那个常量。"""
        groups = [period for period in range(6) for _ in range(10)]
        self.assertIsNone(training.split_by_period(groups, len(groups)))

    def test_enough_periods_does_split(self):
        """边界另一侧：十期就切得动。两侧都断言，才挡得住把门槛调松。"""
        groups = [period for period in range(10) for _ in range(10)]
        self.assertIsNotNone(training.split_by_period(groups, len(groups)))

    def test_mismatched_lengths_decline_to_split(self):
        self.assertIsNone(training.split_by_period([1, 2, 3], 99))

    def test_fallback_split_is_contiguous(self):
        train, valid = training.fallback_split(100)
        self.assertEqual(train[-1] + 1, valid[0])

    def test_fallback_declines_when_validation_would_be_tiny(self):
        self.assertIsNone(training.fallback_split(11))


class ValidationScoreTests(unittest.TestCase):

    def test_confident_and_correct_beats_confident_and_wrong(self):
        self.assertGreater(training.validation_score([1, 0], [0.9, 0.1]),
                           training.validation_score([1, 0], [0.1, 0.9]))

    def test_uncertain_lands_between_the_two(self):
        confident = training.validation_score([1, 0], [0.9, 0.1])
        uncertain = training.validation_score([1, 0], [0.5, 0.5])
        wrong = training.validation_score([1, 0], [0.1, 0.9])
        self.assertLess(wrong, uncertain)
        self.assertLess(uncertain, confident)

    def test_certainty_at_the_wrong_answer_does_not_explode(self):
        """概率被夹在 (0, 1) 内，log(0) 不会出现——**否则整轮训练前功尽弃**。"""
        self.assertGreater(training.validation_score([1, 0], [0.0, 1.0]), 0)

    def test_empty_returns_neutral(self):
        self.assertEqual(training.validation_score([], []), 0.5)


class BlendTests(unittest.TestCase):

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(training.blend_weights([1.0, 3.0])), 1.0)

    def test_higher_score_gets_more_weight(self):
        weights = training.blend_weights([1.0, 3.0])
        self.assertLess(weights[0], weights[1])

    def test_all_zero_scores_share_evenly(self):
        """权重不能全 0——那样融合结果恒为 0，一千注排名完全随机而毫无迹象。"""
        self.assertEqual(training.blend_weights([0.0, 0.0]), [0.5, 0.5])

    def test_blend_is_a_weighted_average(self):
        self.assertEqual(training.blend_predictions([[0.0, 1.0], [1.0, 0.0]], [0.25, 0.75]),
                         [0.75, 0.25])

    def test_blend_of_nothing_is_empty(self):
        self.assertEqual(training.blend_predictions([], []), [])


class ForestTests(unittest.TestCase):
    """纯 Python 降级模型。线上和 CI 都装了三个库，这条路走不到——
    **正因如此它必须有测试**。"""

    def _separable(self):
        rows = [[value, 0.0] for value in range(20)] + [[value, 1.0] for value in range(20)]
        labels = [0] * 20 + [1] * 20
        return rows, labels

    def test_learns_a_separable_split(self):
        rows, labels = self._separable()
        tree = forest.DecisionTree(max_depth=3, min_samples_split=2,
                                   rng=random.Random(0)).fit(rows, labels)
        predictions = tree.predict(rows)
        self.assertLess(sum(predictions[:20]) / 20, sum(predictions[20:]) / 20)

    def test_pure_labels_become_a_leaf_immediately(self):
        tree = forest.DecisionTree(max_depth=5, min_samples_split=2,
                                   rng=random.Random(0)).fit([[1.0], [2.0]], [1, 1])
        self.assertTrue(tree.tree['leaf'])
        self.assertEqual(tree.tree['value'], 1)

    def test_depth_limit_is_respected(self):
        """**语料必须是分不干净的**。完美可分的数据第一层就分纯了，
        深度限制根本走不到——那样这条用例看起来在测限深，实际什么也没测。"""
        rows = [[float(index)] for index in range(20)]
        labels = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
        tree = forest.DecisionTree(max_depth=1, min_samples_split=2,
                                   rng=random.Random(0)).fit(rows, labels)
        self.assertFalse(tree.tree['leaf'], '这份数据第一层分得动')
        self.assertTrue(tree.tree['left']['leaf'] and tree.tree['right']['leaf'],
                        '深度到顶，两个子节点都必须是叶')

    def test_zero_gain_split_stops(self):
        """增益为 0 就不分——再分下去只是把噪声刻进树里。

        这份数据唯一的候选分裂点两边的正例比例都等于总体，增益精确为 0；
        分裂点靠 `max_splits_per_feature=1` 卡到那一个位置上。"""
        rows = [[float(index)] for index in range(6)]
        labels = [0, 0, 1, 1, 0, 1]
        tree = forest.DecisionTree(max_depth=5, min_samples_split=2,
                                   rng=random.Random(0),
                                   max_splits_per_feature=1).fit(rows, labels)
        self.assertTrue(tree.tree['leaf'])

    def test_single_tree_looks_at_every_feature_by_default(self):
        """单棵树默认看**全部**特征。

        十个特征里只有最后一个能分开，其余是常数。默认比例下每个种子都必然
        抽中它；比例减半的话总有种子抽不到，那棵树会退化成一片叶子。"""
        # 标签**交替**排列，不是前十后十：分段排列时常数特征按原始顺序切
        # 也能切出满增益（见下一条用例），那样这份数据分辨不出抽没抽中
        labels = [index % 2 for index in range(20)]
        rows = [[0.0] * 9 + [float(label)] for label in labels]
        for seed in range(8):
            tree = forest.DecisionTree(max_depth=2, min_samples_split=2,
                                       rng=random.Random(seed)).fit(rows, labels)
            predictions = tree.predict(rows)
            zeros = [p for p, label in zip(predictions, labels) if label == 0]
            ones = [p for p, label in zip(predictions, labels) if label == 1]
            with self.subTest(seed=seed):
                # 断言树**真的区分了两类**，而不是「根不是叶」——常数特征也
                # 会产出一个分裂节点（见下一条用例），根不是叶说明不了什么
                self.assertLess(sum(zeros) / len(zeros), sum(ones) / len(ones))

    def test_constant_feature_still_produces_a_split_node(self):
        """**增益是按排序位置切出来评估的，实际分裂却按阈值切。**

        对取值全同的特征，排序退化成原始顺序，于是「按位置切」能切出一个
        增益很高的假分裂；而真正分裂时所有样本都满足 `<= threshold`，
        全落到左边，右子是个空叶。树因此多一层却什么也没分开。

        这是迁移前就有的行为，原样保留——降级路径线上走不到，改掉会动它的
        输出。这条用例把现状钉住，免得下次有人以为「根不是叶」等于分开了。"""
        rows = [[0.0] for _ in range(20)]
        labels = [0] * 10 + [1] * 10
        tree = forest.DecisionTree(max_depth=2, min_samples_split=2,
                                   rng=random.Random(0)).fit(rows, labels)
        self.assertFalse(tree.tree['leaf'], '产出了一个分裂节点')
        self.assertEqual(len(set(tree.predict(rows))), 1, '但一个样本也没分开')

    def test_same_seed_reproduces_the_forest(self):
        rows, labels = self._separable()
        first = forest.RandomForest(n_trees=3, max_depth=2, min_samples_split=2,
                                    rng=random.Random(5)).fit(rows, labels)
        second = forest.RandomForest(n_trees=3, max_depth=2, min_samples_split=2,
                                     rng=random.Random(5)).fit(rows, labels)
        self.assertEqual(first.predict(rows), second.predict(rows))

    def test_different_seeds_give_different_forests(self):
        """种子不同就该不同——否则说明随机源根本没被用上。"""
        rows, labels = self._separable()
        first = forest.RandomForest(n_trees=5, max_depth=2, min_samples_split=2,
                                    feature_subset_ratio=0.5,
                                    rng=random.Random(1)).fit(rows, labels)
        second = forest.RandomForest(n_trees=5, max_depth=2, min_samples_split=2,
                                     feature_subset_ratio=0.5,
                                     rng=random.Random(2)).fit(rows, labels)
        self.assertNotEqual(first.predict(rows), second.predict(rows))

    def test_gini_is_zero_when_pure_and_maximal_when_even(self):
        self.assertEqual(forest.gini([1, 1, 1]), 0)
        self.assertEqual(forest.gini([0, 0, 0]), 0)
        self.assertEqual(forest.gini([0, 1]), 0.5)

    def test_empty_forest_predicts_neutral(self):
        self.assertEqual(forest.RandomForest(0, 2, 2, rng=random.Random(0))
                         .fit([[1.0]], [1]).predict([[1.0]]), [0.5])


class FallbackPathTests(unittest.TestCase):
    """三个库全不可用时的降级——迁移前这条路必然抛 UnboundLocalError。"""

    def test_predicts_without_any_boosting_library(self):
        with mock.patch.object(adapter, 'HAS_CATBOOST', False), \
             mock.patch.object(adapter, 'HAS_XGBOOST', False), \
             mock.patch.object(adapter, 'HAS_LIGHTGBM', False), \
             mock.patch.object(adapter, '_ML_FALLBACK_WARNED', False):
            result = adapter.predict_current(NUMBERS[-150:], top_k=3, neg_samples=5)
        self.assertNotIn('error', result)
        self.assertEqual(result['model_type'], 'random_forest')
        self.assertEqual(len(result['recommendations']), 3)

    def test_warns_only_once_per_process(self):
        """告警每次预测都刷一遍的话，日志里真正的异常会被淹掉。"""
        with mock.patch.object(adapter, 'HAS_CATBOOST', False), \
             mock.patch.object(adapter, 'HAS_XGBOOST', False), \
             mock.patch.object(adapter, 'HAS_LIGHTGBM', False), \
             mock.patch.object(adapter, '_ML_FALLBACK_WARNED', False), \
             mock.patch.object(adapter.log, 'warning') as warned:
            rows = [[float(i), float(i % 3)] for i in range(40)]
            labels = [i % 2 for i in range(40)]
            adapter._fallback_model(rows, labels)
            adapter._fallback_model(rows, labels)
        self.assertEqual(warned.call_count, 1)


class StubbedPipelineTests(unittest.TestCase):
    """整条预测链路——建特征、给一千注打分、排序、取 TopK、算占比——
    用打桩模型跑通。

    **这是黄金剥掉模型输出之后的补偿。** 黄金不再固定分数与特征重要性
    （它们随库版本变化），那部分逻辑改由这里覆盖：打桩模型与三个库无关，
    库升级动不了它。
    """

    class _Descending:
        """按候选的枚举顺序给递减分数，于是排名完全可预测：
        `_rank_all` 枚举的第一注得分最高。"""

        def predict(self, X):
            return [1.0 - index / len(X) for index in range(len(X))]

    def _stubbed(self, top_k=5):
        stub = [(self._Descending(), 'stub', 1.0)]
        with mock.patch.object(adapter, 'train_ensemble',
                               return_value=(stub, list(range(FEATURE_COUNT)))):
            return adapter.predict_current(NUMBERS[-150:], top_k=top_k, neg_samples=5)

    def test_ranking_follows_model_score(self):
        """候选按 (百, 十, 个) 字典序枚举，打桩模型让第一注分最高——
        推荐必须正好是前五注。排序方向反了这里当场红。"""
        result = self._stubbed()
        self.assertEqual([row['num'] for row in result['recommendations']],
                         ['000', '001', '002', '003', '004'])

    def test_top3_is_the_head_of_recommendations(self):
        result = self._stubbed()
        self.assertEqual(result['top3'], result['recommendations'][:3])

    def test_shares_sum_to_one_within_topk(self):
        """占比的分母是 **TopK 内部**的分数和，不是全部一千注——
        用全部当分母的话每个占比都会小两个数量级。"""
        result = self._stubbed()
        shares = [row['topk_score_share'] for row in result['recommendations']]
        self.assertAlmostEqual(sum(shares), 1.0, places=3)

    def test_shares_are_proportional_to_scores(self):
        result = self._stubbed()
        picks = result['recommendations']
        total = sum(row['model_score'] for row in picks)
        for row in picks:
            self.assertAlmostEqual(row['topk_score_share'],
                                   round(row['model_score'] / total, 4), places=3)

    def test_top_k_controls_how_many_come_back(self):
        self.assertEqual(len(self._stubbed(top_k=12)['recommendations']), 12)

    def test_single_model_reports_its_own_name(self):
        result = self._stubbed()
        self.assertEqual(result['model_type'], 'stub')
        self.assertEqual(result['num_models'], 1)
        self.assertEqual(result['model_weights'], [1.0])

    def test_backtest_counts_hits_against_the_stubbed_ranking(self):
        """回测的命中统计同样与库无关：打桩模型把 '000' 排第一，
        所以只有开出 '000' 的那期才算 Top3 命中。"""
        stub = [(self._Descending(), 'stub', 1.0)]
        series = list(NUMBERS[-190:-2]) + [(0, 0, 0), (9, 9, 9)]
        with mock.patch.object(adapter, 'train_ensemble',
                               return_value=(stub, list(range(FEATURE_COUNT)))), \
             mock.patch.object(adapter, 'save_ml_backtest_history'):
            result = adapter.backtest_ml(series, trials=2, train_window=120,
                                         neg_samples=5)
        self.assertEqual(result['trials'], 2)
        self.assertEqual(result['top3_hit'], 1, "只有 '000' 那期中")
        self.assertEqual(result['top3_rate'], 0.5)


class AdapterGuardTests(unittest.TestCase):

    def test_backtest_at_the_minimum_training_window(self):
        """训练窗口恰好等于历史下限时不能崩。迁移前这里是差一：
        回测判 `< 60` 放行，而构造训练集判 `<= 60` 返回四个 None，
        实测抛 `TypeError: object of type 'NoneType' has no len()`。"""
        result = adapter.backtest_ml(NUMBERS[-190:], trials=2, train_window=60,
                                     neg_samples=5)
        self.assertEqual(result['trials'], 0, '样本不足的期全部跳过，而不是崩')

    def test_prediction_declines_on_short_history(self):
        self.assertEqual(adapter.predict_current(NUMBERS[-50:])['error'], '历史数据不足')

    def test_prediction_declines_when_rows_are_too_few(self):
        """历史够长、但采样出来的样本太少也要拒绝。

        一百期够过第一道守卫，每期只抽一条负例 → 八十行，卡在
        「历史数据不足」与「够训练」之间——这一档此前没有任何用例。"""
        self.assertEqual(
            adapter.predict_current(NUMBERS[-100:], neg_samples=1)['error'],
            '训练数据不足')

    def test_backtest_declines_when_data_cannot_cover_the_window(self):
        result = adapter.backtest_ml(NUMBERS[-50:], trials=2, train_window=120)
        self.assertIn('数据量不足', result['error'])

    def test_native_number_keeps_floats_intact(self):
        """先试 int() 会把 0.7 截成 0，小数部分无声消失。"""
        self.assertEqual(adapter._native_number(0.7), 0.7)
        self.assertIs(adapter._native_number(True), True)
        self.assertEqual(adapter._native_number(3), 3)

    def test_ml_fetch_is_not_the_rule_model_fetch(self):
        """两个 `fetch_data` 用的是不同的缓存键。合并要动缓存语义，
        得单独做——这条用例钉住「它们现在还是两份」。"""
        fetching = importlib.import_module('src.lottery3d.fetching')
        self.assertIsNot(adapter.fetch_data, fetching.fetch_data)


adapter = importlib.import_module('src.lottery3d.ml')  # noqa: E402
