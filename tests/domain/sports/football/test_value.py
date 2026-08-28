# -*- coding: utf-8 -*-
"""赔率价值分析、聚类先验融合与比分推荐挑选。

参照物是黄金文件（`tests/fixtures/golden/football_value.json.gz`，360 条），
**逐条相同**。迁移当时另跑过 **185 条**新旧双跑差分，零差异。

**`_pick_recommendations` 的聚类先验是注入的**（判据 16）：取先验要读聚类库，
那是存储。适配层传读库的函数，用例传固定值。

迁移时顺手去掉了两道**恒真**的门（`MARKET_CLUSTERING_AVAILABLE` /
`VALUE_BETTING_AVAILABLE`）——F-1 已证明同包 import 不可能失败，
它们的 `else` 侧从来没执行过。
"""
import ast
import gzip
import json
import pathlib
import unittest

from src.domain.sports.football import scoring, scoring_model as sm, value
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_value.json.gz',
                             'rt', encoding='utf-8'))
PRED = {'H': 0.45, 'D': 0.28, 'A': 0.27}
ODDS = {'H': 2.0, 'D': 3.4, 'A': 3.8}


def golden_entries():
    from scripts.gen_football_value_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, val in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(val))


class ValueAndExpectedValue(unittest.TestCase):

    def test_a_fair_price_has_zero_value(self):
        """概率 0.5、赔率 2.0 正好公平——价值应当是 0。"""
        self.assertAlmostEqual(value.calculate_value(0.5, 2.0), 0.0, places=9)
        self.assertAlmostEqual(value.calculate_ev(0.5, 2.0), 0.0, places=9)

    def test_an_underpriced_outcome_has_positive_value(self):
        self.assertGreater(value.calculate_value(0.6, 2.0), 0.0)
        self.assertGreater(value.calculate_ev(0.6, 2.0), 0.0)

    def test_an_overpriced_outcome_has_negative_value(self):
        """**反方向**：只测正的那一边，把符号弄反也发现不了。"""
        self.assertLess(value.calculate_value(0.4, 2.0), 0.0)
        self.assertLess(value.calculate_ev(0.4, 2.0), 0.0)

    def test_value_scales_with_the_price(self):
        cheap = value.calculate_value(0.5, 1.5)
        rich = value.calculate_value(0.5, 5.0)
        self.assertLess(cheap, rich)


class ValueWeightedAdjustment(unittest.TestCase):

    def test_zero_weight_is_not_a_no_op(self):
        """**★ `value_weight=0` 并不还原原预测 ★**

        实测 {H:0.45, D:0.28, A:0.27} 在权重 0 下变成
        {H:0.4347, D:0.2705, A:0.2948}——名字暗示"不调整"，实际仍然动了
        （归一化那一步与权重无关）。第一版用例断言不变，实测不是。
        钉住现状，见交接文档 §四。
        """
        adjusted = value.adjust_by_value(dict(PRED), ODDS, 0.0)
        self.assertNotEqual(adjusted, PRED)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=9)

    def test_the_default_weight_is_three_tenths(self):
        """判据 29：适配层不传它，默认值没有生产路径覆盖。"""
        self.assertEqual(value.adjust_by_value(dict(PRED), ODDS),
                         value.adjust_by_value(dict(PRED), ODDS, 0.3))
        self.assertNotEqual(value.adjust_by_value(dict(PRED), ODDS),
                            value.adjust_by_value(dict(PRED), ODDS, 0.9))

    def test_a_bigger_weight_moves_it_further(self):
        """**反方向**：权重越大偏离越远，否则权重形同虚设。"""
        small = value.adjust_by_value(dict(PRED), ODDS, 0.1)
        large = value.adjust_by_value(dict(PRED), ODDS, 0.9)
        self.assertGreater(abs(large['H'] - PRED['H']), abs(small['H'] - PRED['H']))

    def test_the_result_stays_normalised(self):
        for weight in (0.0, 0.3, 1.0):
            with self.subTest(weight=weight):
                adjusted = value.adjust_by_value(dict(PRED), ODDS, weight)
                self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=9)

    def test_an_empty_prediction_degrades_instead_of_dividing_by_zero(self):
        for empty in ({}, {'H': 0.0, 'D': 0.0, 'A': 0.0}):
            with self.subTest(empty=empty):
                value.adjust_by_value(empty, ODDS)


class ValueBetIdentification(unittest.TestCase):

    def test_the_threshold_decides_how_many_bets_come_back(self):
        loose = value.identify_value_bets(dict(PRED), ODDS, 0.0)
        strict = value.identify_value_bets(dict(PRED), ODDS, 0.2)
        self.assertGreaterEqual(len(loose), len(strict))

    def test_a_high_threshold_can_return_nothing(self):
        self.assertEqual(value.identify_value_bets(dict(PRED), ODDS, 10.0), [])

    def test_the_default_threshold_is_two_hundredths(self):
        """**语料要真的有价值下注**：`PRED` 那组在任何门槛下都是空的，
        用它测门槛等于什么也没测（判据 23）。这里用一组主胜被低估的赔率。
        """
        underpriced = {'H': 0.60, 'D': 0.22, 'A': 0.18}
        self.assertTrue(value.identify_value_bets(dict(underpriced), ODDS))
        self.assertEqual(value.identify_value_bets(dict(underpriced), ODDS),
                         value.identify_value_bets(dict(underpriced), ODDS, 0.02))
        self.assertEqual(value.identify_value_bets(dict(underpriced), ODDS, 0.5), [])


class ClusterPriorFusion(unittest.TestCase):
    """**先验由调用方传入**——取先验要读聚类库（判据 16）。"""

    def test_no_prior_is_a_no_op_with_a_reason(self):
        """**返回的是 `(probs, meta)` 元组**，不是裸概率（第一版用例比错了）。"""
        probs, meta = value.fuse_with_prior(dict(PRED), 0.5, 2.5, 0.3, None)
        self.assertEqual(probs, PRED)
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'no_prior')

    def test_the_two_paths_return_different_types(self):
        """**★ 判据 17：一半严格一半放任 ★**

        没有先验时早退返回 `(probs, meta)` 元组；有先验时返回**裸字典**。
        同一个函数两条路的返回类型不一样——调用方要么解包炸、要么把
        meta 当概率用。行为原样保留，钉住免得下一个人踩。
        """
        without = value.fuse_with_prior(dict(PRED), 0.5, 2.5, 0.3, None)
        with_prior = value.fuse_with_prior(dict(PRED), 0.5, 2.5, 0.5,
                                           {'H': 0.8, 'D': 0.1, 'A': 0.1})
        self.assertIsInstance(without, tuple)
        self.assertIsInstance(with_prior, dict)

    def test_a_prior_pulls_the_prediction_towards_it(self):
        prior = {'H': 0.8, 'D': 0.1, 'A': 0.1}
        fused = value.fuse_with_prior(dict(PRED), 0.5, 2.5, 0.5, prior)
        self.assertGreater(fused['H'], PRED['H'])
        self.assertLess(fused['H'], prior['H'])

    def test_the_default_prior_weight_is_three_tenths(self):
        prior = {'H': 0.8, 'D': 0.1, 'A': 0.1}
        self.assertEqual(value.fuse_with_prior(dict(PRED), 0.5, 2.5, prior=prior),
                         value.fuse_with_prior(dict(PRED), 0.5, 2.5, 0.3, prior))
        self.assertNotEqual(value.fuse_with_prior(dict(PRED), 0.5, 2.5, prior=prior),
                            value.fuse_with_prior(dict(PRED), 0.5, 2.5, 0.9, prior))

    def test_zero_weight_ignores_the_prior(self):
        """**反方向**：权重为 0 时先验不该起作用。"""
        prior = {'H': 0.8, 'D': 0.1, 'A': 0.1}
        fused = value.fuse_with_prior(dict(PRED), 0.5, 2.5, 0.0, prior)
        for key in PRED:
            with self.subTest(key=key):
                self.assertAlmostEqual(fused[key], PRED[key], places=9)


class RecommendationPickingTakesAnInjectedPrior(unittest.TestCase):

    """用**真实赔率语料**构造入参——手搭的字典总是缺键
    （第一版缺 `expected_goals`，`_alignment_score` 直接 KeyError）。
    判据 23：语料要真的能走到被测分支。
    """

    MATRIX = sm.build_score_matrix(1.5, 1.1, 7, -0.11)
    CANDIDATES = sorted(MATRIX.items(), key=lambda kv: -kv[1])[:14]

    @classmethod
    def setUpClass(cls):
        from scripts.gen_football_modeling_golden import REAL, STRENGTH
        from src.domain.sports.football import risk
        cls.ASIAN, cls.EURO, cls.TOTAL = REAL[0]
        cls.STRENGTH = STRENGTH
        cls.CONF = risk.compute_prediction_confidence(
            cls.ASIAN, cls.EURO, cls.TOTAL, STRENGTH)

    def _pick(self, n=3, **kw):
        return scoring._pick_recommendations(
            self.CANDIDATES, self.ASIAN, self.EURO, self.TOTAL, n, 12,
            self.CONF, None, self.STRENGTH, None, **kw)

    def test_it_returns_picks_and_value_infos(self):
        """**返回的是 `(推荐, 价值信息)` 二元组**，不是推荐列表
        （第一版用例拿 `len()` 当推荐条数，量到的其实恒为 2）。
        """
        picks, value_infos = self._pick(3)
        self.assertIsInstance(picks, list)
        self.assertIsInstance(value_infos, list)
        for pick in picks:
            with self.subTest(pick=pick):
                self.assertEqual(len(pick), 3, '每条推荐是 (主, 客, 概率)')

    def test_without_an_injected_prior_it_still_works(self):
        """默认 `market_prior_fn=None`——领域层不该自己去读聚类库。"""
        self.assertTrue(self._pick(3))

    def test_an_injected_prior_changes_the_ranking(self):
        """**先验的键是比分字符串 `"1-1"`，不是 `H/D/A`**
        （第一版用例喂了赛果键，`prior.get(score_key)` 全取不到，
        注入等于没注入——判据 23）。
        """
        without = self._pick(5)
        # 先验加成是 `1 + prior_prob * 0.3`，**要足够强才翻得动排序**
        # ——给排名靠后的比分一个大先验（判据 28：先验算）
        tail = {f'{h}-{a}': 5.0 for (h, a), _ in self.CANDIDATES[8:]}
        with_prior = self._pick(5, market_prior_fn=lambda h, t: tail)
        self.assertNotEqual(without, with_prior)

    def test_a_prior_with_the_wrong_key_shape_is_silently_ignored(self):
        """**没有别的东西盯着这个落差**（判据 12）：键形状不对时
        `prior.get(score_key, 0.0)` 恒取到 0，先验静默失效、不报错。
        """
        self.assertEqual(
            self._pick(5),
            self._pick(5, market_prior_fn=lambda h, t: {'H': 0.9, 'D': 0.05, 'A': 0.05}))

    def test_similar_market_weighting_changes_the_ranking(self):
        """相似盘口的加成 `1 + w * confidence * 0.5` 也要有样本走到。"""
        similar = {'count': 30, 'confidence': 0.9, 'avg_distance': 0.1,
                   'score_weights': {f'{h}-{a}': 5.0
                                     for (h, a), _ in self.CANDIDATES[8:]}}
        with_similar = scoring._pick_recommendations(
            self.CANDIDATES, self.ASIAN, self.EURO, self.TOTAL, 5, 12,
            self.CONF, None, self.STRENGTH, similar)
        self.assertIsNotNone(with_similar)

    def test_a_prior_function_that_raises_is_swallowed(self):
        """取先验失败不该让整条推荐链崩掉。"""
        def boom(handicap, total):
            raise RuntimeError('聚类库不可用')
        self.assertTrue(self._pick(3, market_prior_fn=boom))


FORBIDDEN_IMPORTS = {'os', 'pathlib', 'requests', 'random', 'time',
                     'src.common.kv_store', 'src.football.config',
                     'src.football.market_clustering', 'src.football.value_betting'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = ('src/domain/sports/football/value.py',
              'src/domain/sports/football/scoring.py')
    ADAPTER = 'src/football/scoring.py'

    def _imports(self, path):
        found = set()
        for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def test_domain_imports_nothing_stateful(self):
        for path in self.DOMAIN:
            with self.subTest(path=path):
                self.assertEqual(self._imports(path) & FORBIDDEN_IMPORTS, set())

    def test_the_two_always_true_gates_are_gone_from_the_domain(self):
        """F-1 已证明同包 import 不可能失败——两个标志恒为 True。"""
        source = pathlib.Path(self.DOMAIN[1]).read_text(encoding='utf-8')
        self.assertNotIn('MARKET_CLUSTERING_AVAILABLE', source)
        self.assertNotIn('VALUE_BETTING_AVAILABLE', source)

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
