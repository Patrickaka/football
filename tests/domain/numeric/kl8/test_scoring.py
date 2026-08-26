"""kl8 的特征评分与集成排名。

`_calculate_feature_score` 是 180 行、15 个特征的单个函数（其中 `sum`、
`zone` 两个已停用但仍在输出里）。它是选号的核心：13 个活跃特征的加权和决定
了每个号码的排名，也决定了最终选出哪些号。

**任何一个特征的曲线变了，选号就变了，而不会有任何报错。** 所以参照物是
从迁移前的实现生成的黄金文件（`tests/fixtures/golden/kl8_scoring.json.gz`，
11 组参数 × 80 个号码 + 4 组权重的排名），覆盖三种重号方向 × 三种频率模式。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.numeric.kl8.scoring import ensemble_ranking, feature_scores
from tests.domain.golden import as_json

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/kl8_scoring.json.gz')
HISTORY = _load('numeric/kl8_history.json.gz')['results']

COMBOS = [{'repeat_direction': d, 'frequency_mode': m}
          for d in ('neutral', 'avoid', 'follow')
          for m in ('neutral', 'hot', 'mean_reversion')]
COMBOS.append({'repeat_direction': 'avoid', 'frequency_mode': 'hot',
               'repeat_avoid_score': 0.05, 'repeat_non_avoid_score': 0.95})
COMBOS.append({'repeat_direction': 'follow', 'frequency_mode': 'neutral',
               'repeat_follow_score': 0.99, 'repeat_non_follow_score': 0.11})

WEIGHTS = [
    {'frequency': 1.0},
    {'frequency': 1.0, 'gap': 0.5, 'position_residual': 0.5, 'repeat': 0.3},
    {'frequency': 0.3, 'gap': 0.3, 'position_residual': 0.2,
     'position_residual_cross': 0.2, 'road_residual': 0.2, 'repeat': 0.3,
     'odd_even': 0.1, 'big_small': 0.1, 'trend': 0.3, 'adjacent': 0.2,
     'pair_cooccurrence': 0.2, 'next_transition': 0.3, 'seeded_random': 0.05},
    {'frequency': 0.0},
]


def _statistics():
    from src.kl8.analyzer import KL8Analyzer

    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_data = HISTORY
    analyzer.statistics = {}
    analyzer.update_statistics()
    return analyzer.statistics


STATS = _statistics()
ISSUE = HISTORY[0]['issue']


class FeatureScoreGoldenTests(unittest.TestCase):
    def test_every_number_and_parameter_combination(self):
        for i, combo in enumerate(COMBOS):
            for num in range(1, 81):
                with self.subTest(combo=i, num=num):
                    self.assertEqual(
                        as_json(feature_scores(num, STATS, based_on_issue=ISSUE,
                                               **combo)),
                        GOLDEN[f'score:{i}:{num}'])

    def test_disabled_features_are_still_reported(self):
        """`sum` 与 `zone` 已停用但仍在输出里，恒为 0.5。

        悄悄删掉它们会让带权重的旧策略少两项加权和——策略试验里存着 23564 条
        历史记录，它们的权重字典是按当年的特征集写的。
        """
        scores = feature_scores(1, STATS, based_on_issue=ISSUE)
        self.assertEqual(scores['sum'], 0.5)
        self.assertEqual(scores['zone'], 0.5)

    def test_feature_set_is_unchanged(self):
        """特征集少一项、多一项，加权和都会变，而且不会报错。"""
        self.assertEqual(sorted(feature_scores(1, STATS, based_on_issue=ISSUE)),
                         sorted(GOLDEN['score:0:1']))


class RepeatDirectionTests(unittest.TestCase):
    """重号方向：追上期、避上期、还是中立。这是策略之间差别最大的一维。"""

    def _repeat(self, num, **kwargs):
        return feature_scores(num, STATS, based_on_issue=ISSUE, **kwargs)['repeat']

    def test_neutral_is_indifferent(self):
        last = sorted(STATS['last_numbers'])[0]
        self.assertEqual(self._repeat(last, repeat_direction='neutral'), 0.50)

    def test_avoid_penalises_last_draw_numbers(self):
        last = sorted(STATS['last_numbers'])[0]
        other = next(n for n in range(1, 81) if n not in STATS['last_numbers'])
        self.assertLess(self._repeat(last, repeat_direction='avoid'),
                        self._repeat(other, repeat_direction='avoid'))

    def test_follow_rewards_last_draw_numbers(self):
        last = sorted(STATS['last_numbers'])[0]
        other = next(n for n in range(1, 81) if n not in STATS['last_numbers'])
        self.assertGreater(self._repeat(last, repeat_direction='follow'),
                           self._repeat(other, repeat_direction='follow'))

    def test_scores_are_configurable(self):
        last = sorted(STATS['last_numbers'])[0]
        self.assertEqual(self._repeat(last, repeat_direction='avoid',
                                      repeat_avoid_score=0.01), 0.01)


class FrequencyModeTests(unittest.TestCase):
    """频率模式：追热、均值回归、还是不看频率。三种取向互为反面。"""

    def _frequency(self, num, mode):
        return feature_scores(num, STATS, based_on_issue=ISSUE,
                              frequency_mode=mode)['frequency']

    def _hottest_and_coldest(self):
        freq = STATS['frequency']
        ordered = sorted(range(1, 81), key=lambda n: freq.get(n, 0))
        return ordered[-1], ordered[0]

    def test_neutral_is_flat(self):
        hot, cold = self._hottest_and_coldest()
        self.assertEqual(self._frequency(hot, 'neutral'), 0.50)
        self.assertEqual(self._frequency(cold, 'neutral'), 0.50)

    def test_hot_mode_favours_frequent_numbers(self):
        hot, cold = self._hottest_and_coldest()
        self.assertGreater(self._frequency(hot, 'hot'), self._frequency(cold, 'hot'))

    def test_mean_reversion_favours_cold_numbers(self):
        """与追热恰好相反。两种模式接反了不会报错，只会让选号系统性走偏。"""
        hot, cold = self._hottest_and_coldest()
        self.assertGreater(self._frequency(cold, 'mean_reversion'),
                           self._frequency(hot, 'mean_reversion'))


class TransitionClampTests(unittest.TestCase):
    """跨期带出夹在 [0.15, 0.85] 之间。

    这条钳位在真实数据上从不触发——历史概率都落在中间。但它防的正是
    「稀疏关联」：某个号码在极少的样本里恰好每次都跟着开出，条件概率会冲到
    接近 1，不夹住的话这一项会压倒其余十几个特征。所以只能构造。
    """

    def _score(self, probability):
        stats = dict(STATS, next_transition_probability={7: probability})
        return feature_scores(7, stats, based_on_issue=ISSUE)['next_transition']

    def test_extreme_high_is_capped(self):
        self.assertEqual(self._score(1.0), 0.85)

    def test_the_floor_is_only_reached_at_probability_zero(self):
        """下界对合法概率**不会真正触发**：概率最低是 0，对应 lift = -1，
        算出来正好是 0.15。它是防御性的，真正会绑住的是上界。
        写清楚比硬凑一个触发它的用例诚实。"""
        self.assertAlmostEqual(self._score(0.0), 0.15)

    def test_baseline_maps_to_neutral(self):
        self.assertAlmostEqual(self._score(0.25), 0.50)

    def test_inside_the_band_is_untouched(self):
        """带内不动——钳位不该顺手压扁正常范围的信号。"""
        self.assertGreater(self._score(0.30), 0.50)
        self.assertLess(self._score(0.30), 0.85)


class SeededRandomTests(unittest.TestCase):
    """种子随机：同一期内稳定、跨期变化。用来给并列打破僵局。"""

    def _seeded(self, num, issue):
        return feature_scores(num, STATS, based_on_issue=issue)['seeded_random']

    def test_stable_within_the_same_issue(self):
        self.assertEqual(self._seeded(7, '2026225'), self._seeded(7, '2026225'))

    def test_changes_across_issues(self):
        self.assertNotEqual(self._seeded(7, '2026225'), self._seeded(7, '2026226'))

    def test_differs_between_numbers(self):
        self.assertNotEqual(self._seeded(7, '2026225'), self._seeded(8, '2026225'))

    def test_stays_within_the_unit_interval(self):
        values = [self._seeded(n, '2026225') for n in range(1, 81)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values), values[:5])


class AnalyzerWiringTests(unittest.TestCase):
    """分析器接线：期号必须从**最新一期**取并传下去。

    上面那些用例都直接给 `feature_scores` 传期号，绕开了分析器——接错了
    测不出来。而接错的后果是种子随机分不再随期变化（传空）或者一直用最旧
    一期（取错下标），两种都让「同期稳定、跨期变化」失效，且不会报错。
    """

    def _analyzer(self, history=None):
        from src.kl8.analyzer import KL8Analyzer

        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = history if history is not None else HISTORY
        analyzer.statistics = {}
        analyzer.update_statistics()
        return analyzer

    def test_based_on_issue_is_the_newest(self):
        analyzer = self._analyzer()
        self.assertEqual(analyzer._based_on_issue(), HISTORY[0]['issue'])
        self.assertNotEqual(analyzer._based_on_issue(), HISTORY[-1]['issue'])

    def test_empty_history_yields_no_issue(self):
        analyzer = self._analyzer(history=[])
        self.assertEqual(analyzer._based_on_issue(), '')

    def test_ranking_uses_the_newest_issue(self):
        """用只看种子随机的权重，排名就完全由期号决定——接错立刻显形。"""
        analyzer = self._analyzer()
        actual = analyzer.get_ensemble_ranking(
            top_n=10, feature_weights={'seeded_random': 1.0})
        expected = ensemble_ranking(
            analyzer.statistics, {'seeded_random': 1.0}, top_n=10,
            based_on_issue=HISTORY[0]['issue'])
        self.assertEqual([r['num'] for r in actual], [r['num'] for r in expected])

    def test_feature_scores_use_the_newest_issue(self):
        analyzer = self._analyzer()
        self.assertEqual(
            analyzer._calculate_feature_score(7)['seeded_random'],
            feature_scores(7, analyzer.statistics,
                           based_on_issue=HISTORY[0]['issue'])['seeded_random'])


class EnsembleRankingGoldenTests(unittest.TestCase):
    def test_matches_golden(self):
        for wi, weights in enumerate(WEIGHTS):
            for i, combo in enumerate(COMBOS[:3]):
                with self.subTest(weights=wi, combo=i):
                    self.assertEqual(
                        as_json(ensemble_ranking(STATS, weights, top_n=20,
                                                 based_on_issue=ISSUE, **combo)),
                        GOLDEN[f'rank:{wi}:{i}'])

    def test_all_zero_weights_return_nothing(self):
        """没有任何有效权重时返回空，**而不是 1~20**。

        返回前 20 个号码看起来像个结果，实际是「按号码大小排序」——一个
        没有任何信号的输出被当成了预测。
        """
        self.assertEqual(ensemble_ranking(STATS, {'frequency': 0.0}, top_n=20,
                                          based_on_issue=ISSUE), [])
        self.assertEqual(ensemble_ranking(STATS, {}, top_n=20,
                                          based_on_issue=ISSUE), [])

    def test_respects_top_n(self):
        ranking = ensemble_ranking(STATS, {'frequency': 1.0}, top_n=5,
                                   based_on_issue=ISSUE)
        self.assertEqual(len(ranking), 5)

    def test_sorted_by_score_then_number(self):
        """并列时按号码升序——否则同一份输入在不同运行里会给出不同的顺序。"""
        ranking = ensemble_ranking(STATS, {'frequency': 1.0}, top_n=80,
                                   based_on_issue=ISSUE)
        keys = [(-r['ranking_score'], r['num']) for r in ranking]
        self.assertEqual(keys, sorted(keys))

    def test_entry_shape(self):
        entry = ensemble_ranking(STATS, {'frequency': 1.0}, top_n=1,
                                 based_on_issue=ISSUE)[0]
        self.assertEqual(sorted(entry),
                         ['is_probability', 'num', 'ranking_score',
                          'score_type', 'scores'])
        self.assertFalse(entry['is_probability'],
                         '排名分不是概率，标成概率会让下游拿它当胜率用')


if __name__ == '__main__':
    unittest.main()
