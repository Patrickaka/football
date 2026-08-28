# -*- coding: utf-8 -*-
"""足球比分建模：泊松/负二项/Dixon-Coles/λ 拟合/贝叶斯/风险评级。

参照物是从迁移前的 `src/football/modeling.py` 生成的黄金文件
（`tests/fixtures/golden/football_modeling.json.gz`，3759 条），**逐条相同**。
迁移当时另跑过 **3747 条**新旧双跑差分，零差异。

**差分带了每函数覆盖报告**：F-3 吃过一次亏——语料喂错形状，两边抛同样的错，
差分「零差异」其实什么也没测（判据 8）。这一批第一轮就有 **16 个函数一次都
没产出有效值**（全是 TypeError：签名喂错），修正后每个函数都有有效输出。

## 三处迁移期查明、行为原样保留的事

**1. 仓库里三份 Dixon-Coles 是三个不同的模型，不是三份拷贝。**
本模块这份把 τ 修正**扩展到所有比分**（`exp(-(h+a)*0.3)` 衰减）；
`src/football/ml.py` 那份只改四格且用**比值形式**，算出来与标准公式不同；
`domain/sports/beidan/scoring_model.py` 那份是标准四格形式。
`rho = 0` 时三者一致（≤2.8e-17），非零时不一致——**所以没有合并**。

**2. 两个 DC 修正在生产上互相抵消。**
`scoring.py` 用本模块 + `_estimate_dc_rho`（**负** rho，抬高 1-1）；
`pipeline.py:747` 用 `ml` 那份 + `get_dc_rho`（**正** rho，压低 1-1），
两者按 0.50 融合。实测 λ=(1.5, 1.1) 时：modeling 单独 P(平)=0.279、
ml 单独 0.238、五五融合 **0.259**——而完全不做 DC 修正是 **0.258**。
两个各自调过参的修正，净效果约等于没做。

**3. 风险评级有两个因子读的是别的运动的键，永远不会触发。**
`euro['implied_home']` 在**全仓零处写入**；`asian['home_prob']` 与
`euro['expected_total']` 只有 `domain/sports/basketball/elo.py` 写。
所以「欧亚分歧」与「大小球分歧」两个信号对足球恒不生效。
线上 114 场实测只有两个因子出现过：相似盘口样本不足（24）、资金流异常（60）。
"""
import ast
import gzip
import json
import math
import pathlib
import random
import unittest

from src.domain.sports.football import bayes, lambdas, markets, risk, scoring_model
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_modeling.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时 config 的真实取值。**写死不 import**（判据 4）
MAX_GOALS = 7
AVG_LEAGUE_GOAL = 1.35
LAMBDA_WEIGHT_MARKET, LAMBDA_WEIGHT_TEAM, LAMBDA_WEIGHT_ELO = 0.5, 0.3, 0.2
SUP_ASIAN_WEIGHT, SUP_EURO_WEIGHT = 0.48, 0.52
CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD = 0.72, 0.52


def golden_entries():
    from scripts.gen_football_modeling_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class ThreeDixonColesModels(unittest.TestCase):
    """把「三份 DC 不是三份拷贝」钉住，免得下一个人再去合并它们。"""

    LAM = (1.5, 1.1)

    @staticmethod
    def _ml_matrix(lam_home, lam_away, rho, max_goals=6):
        from src.football import ml
        raw = ml.dixon_coles_score_matrix(lam_home, lam_away, max_goals, rho)
        return {(h, a): float(raw[h][a])
                for h in range(max_goals + 1) for a in range(max_goals + 1)}

    def test_all_three_agree_when_rho_is_zero(self):
        lh, la = self.LAM
        mine = scoring_model.build_score_matrix(lh, la, 6, 0.0)
        from src.domain.sports.beidan import scoring_model as beidan
        theirs = beidan.dixon_coles_matrix(lh, la, 0.0, 6)
        ml_m = self._ml_matrix(lh, la, 0.0)
        for cell in mine:
            with self.subTest(cell=cell):
                self.assertAlmostEqual(mine[cell], theirs[cell], places=15)
                self.assertAlmostEqual(mine[cell], ml_m[cell], places=15)

    def test_they_diverge_once_rho_is_not_zero(self):
        """**反方向**：rho≠0 时必须不同，否则「不能合并」这个结论就站不住。"""
        lh, la = self.LAM
        mine = scoring_model.build_score_matrix(lh, la, 6, -0.05)
        from src.domain.sports.beidan import scoring_model as beidan
        theirs = beidan.dixon_coles_matrix(lh, la, -0.05, 6)
        ml_m = self._ml_matrix(lh, la, -0.05)
        self.assertGreater(max(abs(mine[c] - theirs[c]) for c in mine), 1e-4)
        self.assertGreater(max(abs(mine[c] - ml_m[c]) for c in mine), 1e-4)

    def test_our_tau_matches_the_standard_formula_on_the_four_low_cells(self):
        lh, la, rho = 1.5, 1.1, -0.05
        self.assertAlmostEqual(scoring_model._dc_tau(0, 0, lh, la, rho), 1 - lh * la * rho)
        self.assertAlmostEqual(scoring_model._dc_tau(0, 1, lh, la, rho), 1 + lh * rho)
        self.assertAlmostEqual(scoring_model._dc_tau(1, 0, lh, la, rho), 1 + la * rho)
        self.assertAlmostEqual(scoring_model._dc_tau(1, 1, lh, la, rho), 1 - rho)

    def test_our_tau_extends_beyond_the_four_cells_which_standard_dc_does_not(self):
        """标准 DC 只改四格，本模块用指数衰减扩展到了所有格。"""
        lh, la, rho = 1.5, 1.1, -0.05
        for cell in ((2, 0), (2, 2), (3, 1), (0, 3)):
            with self.subTest(cell=cell):
                self.assertNotAlmostEqual(scoring_model._dc_tau(*cell, lh, la, rho), 1.0)
        # rho=0 时连扩展格也退化成 1
        for cell in ((2, 0), (2, 2), (3, 1)):
            self.assertEqual(scoring_model._dc_tau(*cell, lh, la, 0.0), 1.0)

    def test_the_two_live_corrections_very_nearly_cancel(self):
        """生产上两个 DC 五五融合后，几乎回到不做修正的水平。

        钉住的是一个**数量级**结论：融合结果与独立泊松的平局概率差 < 0.002，
        而各自单独用时差 > 0.015。
        """
        lh, la = self.LAM
        draw = lambda m: sum(v for (h, a), v in m.items() if h == a)
        base = scoring_model.build_score_matrix(lh, la, 6, 0.0)
        mine = scoring_model.build_score_matrix(lh, la, 6, -0.11)
        ml_m = self._ml_matrix(lh, la, 0.10)
        blended = {c: 0.5 * mine[c] + 0.5 * ml_m[c] for c in mine}

        self.assertGreater(abs(draw(mine) - draw(base)), 0.015)
        self.assertGreater(abs(draw(ml_m) - draw(base)), 0.015)
        self.assertLess(abs(draw(blended) - draw(base)), 0.002)


class RiskFactorsThatCanNeverFire(unittest.TestCase):
    """两个风险因子读的是**篮球**领域才写的键，对足球恒不生效。"""

    KEYS_NEVER_WRITTEN_FOR_FOOTBALL = ('implied_home', 'home_prob', 'expected_total')

    def test_the_football_market_layer_never_produces_those_keys(self):
        corpus = json.loads((FIXTURES / 'football_markets_corpus.json').read_text(encoding='utf-8'))
        asian = markets.analyze_asian(corpus['asian'][0])
        euro = markets.analyze_euro(corpus['euro'][0])
        total = markets.analyze_total(corpus['total'][0])
        for produced in (asian, euro, total):
            for key in self.KEYS_NEVER_WRITTEN_FOR_FOOTBALL:
                self.assertNotIn(key, produced)

    def test_those_two_factors_stay_silent_on_real_shaped_input(self):
        corpus = json.loads((FIXTURES / 'football_markets_corpus.json').read_text(encoding='utf-8'))
        asian = markets.analyze_asian(corpus['asian'][0])
        euro = markets.analyze_euro(corpus['euro'][0])
        total = markets.analyze_total(corpus['total'][0])
        confidence = risk.compute_prediction_confidence(asian, euro, total)
        result = risk._evaluate_risk_level(asian, euro, total, None, confidence, None)
        for factor in ('欧亚分歧', '大小球分歧'):
            self.assertNotIn(factor, result['risk_factors'])

    def test_but_they_do_fire_if_someone_supplies_the_keys(self):
        """**不是死代码**——键给全了就会触发（判据 9 第三行）。"""
        corpus = json.loads((FIXTURES / 'football_markets_corpus.json').read_text(encoding='utf-8'))
        asian = dict(markets.analyze_asian(corpus['asian'][0]), home_prob=0.20)
        euro = dict(markets.analyze_euro(corpus['euro'][0]),
                    implied_home=0.60, expected_total=1.0)
        total = markets.analyze_total(corpus['total'][0])
        confidence = risk.compute_prediction_confidence(asian, euro, total)
        result = risk._evaluate_risk_level(asian, euro, total, None, confidence, None)
        self.assertIn('欧亚分歧', result['risk_factors'])
        self.assertIn('大小球分歧', result['risk_factors'])


class RiskLevelThresholds(unittest.TestCase):
    """四档风险等级的每一道门槛都测两侧（判据 5）。

    风险分是七个因子的加权和；这里直接按分数造样本，不绕远路。
    """

    @staticmethod
    def _level_for(similar_confidence, steam_confidence, conf_score):
        asian = {'handicap': 0.5, 'open_handicap': 0.5}
        euro = {'kelly': {'spread': 0.0, 'hardest': 'neutral'}}
        total = {'close_line': 2.5}
        steam = ({'summary': {'confidence': steam_confidence}}
                 if steam_confidence is not None else None)
        similar = ({'confidence': similar_confidence}
                   if similar_confidence is not None else None)
        return risk._evaluate_risk_level(asian, euro, total, steam,
                                         {'score': conf_score}, similar)

    def test_no_factor_at_all_is_level_a(self):
        result = self._level_for(None, None, 0.8)
        self.assertEqual(result['level'], 'A')
        self.assertEqual(result['risk_factors'], [])
        self.assertEqual(result['recommend'], '正常推荐')

    def test_one_factor_below_the_first_threshold_is_still_a(self):
        """相似盘口样本不足只加 0.25，不到 0.35。"""
        result = self._level_for(0.2, None, 0.8)
        self.assertEqual(result['risk_factors'], ['相似盘口样本不足'])
        self.assertAlmostEqual(result['risk_score'], 0.25)
        self.assertEqual(result['level'], 'A')

    def test_two_factors_cross_into_b(self):
        result = self._level_for(0.2, 0.9, 0.8)
        self.assertAlmostEqual(result['risk_score'], 0.45)
        self.assertEqual(result['level'], 'B')
        self.assertEqual(result['recommend'], '精简推荐')

    def test_three_factors_cross_into_c(self):
        result = self._level_for(0.2, 0.9, 0.3)
        self.assertAlmostEqual(result['risk_score'], 0.65)
        self.assertEqual(result['level'], 'C')
        self.assertEqual(result['recommend'], '谨慎推荐')

    def test_a_handicap_reversal_pushes_all_the_way_to_d(self):
        asian = {'handicap': -0.5, 'open_handicap': 0.5}   # 方向反转
        euro = {'kelly': {'spread': 0.0, 'hardest': 'neutral'}}
        result = risk._evaluate_risk_level(asian, euro, {'close_line': 2.5},
                                           {'summary': {'confidence': 0.9}},
                                           {'score': 0.3}, {'confidence': 0.2})
        self.assertAlmostEqual(result['risk_score'], 0.95)
        self.assertEqual(result['level'], 'D')
        self.assertEqual(result['recommend_count'], 0)
        self.assertEqual(result['recommend'], '不建议投注比分')

    def test_the_kelly_factor_needs_both_a_wide_spread_and_a_non_neutral_hardest(self):
        """**两个条件是与不是或**——而 F-2 已证明生产上 spread 恒为 0，
        所以这个因子在线上永远不触发（判据 7 + F-2 的发现）。
        """
        base = {'handicap': 0.5, 'open_handicap': 0.5}
        both = risk._evaluate_risk_level(
            base, {'kelly': {'spread': 5.0, 'hardest': 'home'}}, {'close_line': 2.5},
            None, {'score': 0.8}, None)
        self.assertIn('凯利离散度较高，存在明显分化', both['risk_factors'])
        wide_but_neutral = risk._evaluate_risk_level(
            base, {'kelly': {'spread': 5.0, 'hardest': 'neutral'}}, {'close_line': 2.5},
            None, {'score': 0.8}, None)
        self.assertEqual(wide_but_neutral['risk_factors'], [])
        narrow = risk._evaluate_risk_level(
            base, {'kelly': {'spread': 3.9, 'hardest': 'home'}}, {'close_line': 2.5},
            None, {'score': 0.8}, None)
        self.assertEqual(narrow['risk_factors'], [])


class McmcNeedsAnInjectedRandomSource(unittest.TestCase):
    """判据 16：随机源是副作用，不注入的话黄金文件不可复现。"""

    TARGETS = (0.45, 0.28, 0.27)

    class Seeded:
        def __init__(self, seed):
            self._r = random.Random(seed)

        def random(self):
            return self._r.random()

    def test_the_same_seed_gives_the_same_samples(self):
        a = bayes._mcmc_sample_lambdas(self.TARGETS, 2.6, 0.4, None, None,
                                       n_samples=30, burn_in=5, rng=self.Seeded(7))
        b = bayes._mcmc_sample_lambdas(self.TARGETS, 2.6, 0.4, None, None,
                                       n_samples=30, burn_in=5, rng=self.Seeded(7))
        self.assertEqual(a, b)

    def test_a_different_seed_gives_different_samples(self):
        """**反方向**：不同种子必须不同，否则上一条对常量注入也成立。"""
        a = bayes._mcmc_sample_lambdas(self.TARGETS, 2.6, 0.4, None, None,
                                       n_samples=30, burn_in=5, rng=self.Seeded(7))
        b = bayes._mcmc_sample_lambdas(self.TARGETS, 2.6, 0.4, None, None,
                                       n_samples=30, burn_in=5, rng=self.Seeded(8))
        self.assertNotEqual(a, b)

    def test_burn_in_samples_are_dropped(self):
        n_samples, burn_in = 30, 5
        samples = bayes._mcmc_sample_lambdas(self.TARGETS, 2.6, 0.4, None, None,
                                             n_samples=n_samples, burn_in=burn_in,
                                             rng=self.Seeded(7))
        self.assertEqual(len(samples), n_samples - burn_in)

    def test_without_an_injected_source_it_still_runs(self):
        """默认走 `random` 模块本身——与迁移前一致，只是不可复现。"""
        samples = bayes._mcmc_sample_lambdas(self.TARGETS, 2.6, 0.4, None, None,
                                             n_samples=10, burn_in=2)
        self.assertEqual(len(samples), 8)


class ScoreMatrixContract(unittest.TestCase):

    def test_the_matrix_always_sums_to_one(self):
        for lam in ((1.5, 1.1), (0.2, 0.2), (4.0, 3.5)):
            for rho in (0.0, -0.16, 0.12):
                with self.subTest(lam=lam, rho=rho):
                    m = scoring_model.build_score_matrix(*lam, 7, rho)
                    self.assertAlmostEqual(sum(m.values()), 1.0, places=12)

    def test_negative_binomial_is_more_dispersed_than_poisson(self):
        """过离散：负二项把概率从众数往两端挪。"""
        poisson = scoring_model.build_score_matrix(1.5, 1.1, 7, 0.0, 'poisson')
        nb = scoring_model.build_score_matrix(1.5, 1.1, 7, 0.0, 'negative_binomial')
        self.assertNotAlmostEqual(poisson[(1, 1)], nb[(1, 1)], places=4)
        self.assertGreater(nb[(0, 0)], poisson[(0, 0)])

    def test_the_matrix_size_follows_max_goals(self):
        for max_goals in (3, 5, 7):
            with self.subTest(max_goals=max_goals):
                m = scoring_model.build_score_matrix(1.5, 1.1, max_goals, 0.0)
                self.assertEqual(len(m), (max_goals + 1) ** 2)

    def test_estimated_rho_is_always_negative_and_has_three_steps(self):
        """`_estimate_dc_rho` 只会返回三个值之一，且都是负的。"""
        seen = {scoring_model._estimate_dc_rho(1.5, 1.1, p)
                for p in (0.15, 0.20, 0.24, 0.26, 0.28, 0.32, 0.40)}
        self.assertEqual(seen, {-0.16, -0.11, -0.06})
        self.assertTrue(all(v < 0 for v in seen))


FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'requests', 'urllib.request',
                     'urllib.error', 'src.common.kv_store', 'src.foundation.store',
                     'src.football.fetching', 'src.football.config'}
FORBIDDEN_CALLS = {'now', 'today', 'utcnow', 'strftime'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = ('src/domain/sports/football/scoring_model.py',
              'src/domain/sports/football/lambdas.py',
              'src/domain/sports/football/risk.py',
              'src/domain/sports/football/bayes.py')
    ADAPTER = 'src/football/parsing.py'

    def _tree(self, path):
        return ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))

    def _imports(self, path):
        found = set()
        for node in ast.walk(self._tree(path)):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def _clock_calls(self, path):
        return {n.func.attr for n in ast.walk(self._tree(path))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in FORBIDDEN_CALLS}

    def test_domain_imports_nothing_stateful(self):
        for path in self.DOMAIN:
            with self.subTest(path=path):
                self.assertEqual(self._imports(path) & FORBIDDEN_IMPORTS, set())

    def test_domain_never_reads_the_clock(self):
        for path in self.DOMAIN:
            with self.subTest(path=path):
                self.assertEqual(self._clock_calls(path), set())

    def test_only_bayes_touches_randomness_and_it_is_injectable(self):
        """随机源只允许出现在 `bayes`，而且必须能注入。"""
        for path in self.DOMAIN:
            uses_random = 'random' in self._imports(path)
            with self.subTest(path=path):
                if path.endswith('bayes.py'):
                    self.assertTrue(uses_random)
                else:
                    self.assertFalse(uses_random)
        import inspect
        self.assertIn('rng', inspect.signature(bayes._mcmc_sample_lambdas).parameters)
        self.assertIn('rng', inspect.signature(bayes.bayesian_predict_scores).parameters)

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
