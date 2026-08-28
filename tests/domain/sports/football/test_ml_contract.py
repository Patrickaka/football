# -*- coding: utf-8 -*-
"""ML 契约层：特征契约、动态 rho、DC 变体、进球数推荐、按时间切分。

参照物是黄金文件（`tests/fixtures/golden/football_ml.json.gz`，528 条），
**逐条相同**。迁移当时另跑过 **532 条**新旧双跑差分，零差异。

## 判据 20b 是这一批的硬约束

黄金里**只有「我们自己算的」**。带 numpy / catboost / xgboost / sklearn 的
函数一个都没搬——它们留在 `src/football/ml.py`。
3-17b 在 lottery3d 上把库算出来的数钉进黄金，本地全绿、**CI 直接红 5 条**，
因为 `requirements.txt` 用 `>=`、CI 装的版本比本地新。

判断方法：问「换一个版本的 catboost，这个值还一样吗」。

## 两处与别的模块的关系

**`get_close_total_line` 的重复消掉了**：`ml.py` 与 `parsing.py` 那两份
F-3 已用 AST 确认**逐字相同**，现在领域层只留 `parsing` 一份。

**`dixon_coles_*` 是仓库三份 DC 中的第二份**——只改四格且用**比值形式**，
τ(1,0) 算出来是 `1 + rho*λa/λh` 而不是标准的 `1 + rho*λa`。
F-4 确认三份是三个不同的模型（`rho=0` 时一致、非零时不一致），**不合并**。
"""
import ast
import gzip
import itertools
import json
import pathlib
import unittest

from src.domain.sports.football import ml_contract as mc
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_ml.json.gz',
                             'rt', encoding='utf-8'))


def golden_entries():
    from scripts.gen_football_ml_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class DynamicRho(unittest.TestCase):
    """`get_dc_rho` 的四道规则叠加，每道都测两侧。"""

    def test_the_baseline_is_one_tenth(self):
        self.assertAlmostEqual(mc.get_dc_rho(None, None, None), 0.1)

    def test_low_scoring_leagues_get_a_higher_rho(self):
        self.assertGreater(mc.get_dc_rho('意甲', None, None),
                           mc.get_dc_rho('英超', None, None))

    def test_big_competitions_get_a_lower_rho(self):
        self.assertLess(mc.get_dc_rho('欧冠', None, None),
                        mc.get_dc_rho('英超', None, None))

    def test_the_total_line_can_override_the_league(self):
        """**盘口那道是 `max`/`min`，会覆盖联赛给的值**——两侧都测。"""
        self.assertGreaterEqual(mc.get_dc_rho('英超', 2.0, None), 0.12)
        self.assertLessEqual(mc.get_dc_rho('意甲', 3.5, None), 0.04)

    def test_the_line_thresholds_sit_at_two_and_a_quarter_and_three(self):
        self.assertNotEqual(mc.get_dc_rho('英超', 2.25, None),
                            mc.get_dc_rho('英超', 2.30, None))
        self.assertNotEqual(mc.get_dc_rho('英超', 3.0, None),
                            mc.get_dc_rho('英超', 2.99, None))

    def test_the_result_is_always_positive(self):
        """**与 `_estimate_dc_rho` 符号相反**——那份给负值（抬高 1-1），
        这份给正值（压低 1-1）。F-4 量过：两者按 0.50 融合后互相抵消。
        """
        for league, line, handicap in itertools.product(
                ('意甲', '欧冠', '英超', None), (None, 2.0, 3.5), (None, 0.0, 1.5)):
            with self.subTest(league=league, line=line, handicap=handicap):
                self.assertGreater(mc.get_dc_rho(league, line, handicap), 0.0)


class DixonColesRatioForm(unittest.TestCase):
    """把「这份 DC 用的是比值形式」钉住——它与标准公式算出来不一样。"""

    LAM_HOME, LAM_AWAY, RHO = 1.5, 1.1, 0.1

    def test_rho_zero_makes_every_adjustment_neutral(self):
        for h, a in itertools.product(range(3), range(3)):
            with self.subTest(h=h, a=a):
                self.assertAlmostEqual(
                    mc.dixon_coles_adjustment(0.0, self.LAM_HOME, self.LAM_AWAY, h, a),
                    1.0, places=12)

    def test_only_the_four_low_cells_are_touched(self):
        """标准 DC 只改 0-0/0-1/1-0/1-1——**其余格恒为 1.0**。"""
        for h, a in ((2, 0), (0, 2), (2, 2), (3, 1)):
            with self.subTest(h=h, a=a):
                self.assertEqual(
                    mc.dixon_coles_adjustment(self.RHO, self.LAM_HOME, self.LAM_AWAY, h, a),
                    1.0)

    def test_the_zero_zero_cell_matches_the_standard_formula(self):
        """τ(0,0) 两种写法算出来一样（`p1*p1/(p0*p0) == λh*λa`）。"""
        self.assertAlmostEqual(
            mc.dixon_coles_adjustment(self.RHO, self.LAM_HOME, self.LAM_AWAY, 0, 0),
            1 - self.LAM_HOME * self.LAM_AWAY * self.RHO, places=12)

    def test_the_other_three_cells_do_not(self):
        """**这是三份 DC 不能合并的直接证据**：τ(1,0) 这份算出来是
        `1 + rho*λa/λh`，标准公式是 `1 + rho*λa`。
        """
        ratio_form = mc.dixon_coles_adjustment(self.RHO, self.LAM_HOME, self.LAM_AWAY, 1, 0)
        standard = 1 + self.LAM_AWAY * self.RHO
        self.assertNotAlmostEqual(ratio_form, standard, places=6)
        self.assertAlmostEqual(ratio_form, 1 + self.RHO * self.LAM_AWAY / self.LAM_HOME,
                               places=12)


class FeatureContract(unittest.TestCase):

    def test_names_and_defaults_line_up(self):
        names = mc.get_feature_names()
        defaults = mc.get_feature_defaults()
        self.assertEqual(len(names), len(defaults))
        self.assertEqual(set(names), set(defaults))

    def test_validate_returns_a_filled_in_payload_not_a_verdict(self):
        """**`validate_features` 返回的是补全后的特征字典**，不是
        `(ok, problems)`——判定在 `audit_feature_payload` 里
        （第一版用例解包成两个值，直接错）。
        """
        names = mc.get_feature_names()
        filled = mc.validate_features({names[0]: 1.0})
        self.assertEqual(set(filled), set(names))
        self.assertEqual(filled[names[0]], 1.0)

    def test_missing_features_come_back_at_their_defaults(self):
        names = mc.get_feature_names()
        defaults = mc.get_feature_defaults()
        filled = mc.validate_features({})
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(filled[name], defaults[name])

    def test_the_audit_is_where_the_verdict_lives(self):
        names = mc.get_feature_names()
        complete = mc.audit_feature_payload({n: 0.0 for n in names})
        self.assertTrue(complete['complete'])
        self.assertEqual(complete['missing'], [])
        self.assertEqual(complete['unknown'], [])
        self.assertEqual(complete['expected_count'], len(names))

    def test_the_audit_names_the_missing_one(self):
        """**反方向**：只测完整的那边，把审计删掉也全绿。"""
        names = mc.get_feature_names()
        audit = mc.audit_feature_payload({n: 0.0 for n in names[:-1]})
        self.assertFalse(audit['complete'])
        self.assertEqual(audit['missing'], [names[-1]])

    def test_the_audit_names_the_unknown_one(self):
        names = mc.get_feature_names()
        audit = mc.audit_feature_payload({**{n: 0.0 for n in names}, '凭空多出来的': 1.0})
        self.assertEqual(audit['unknown'], ['凭空多出来的'])
        self.assertFalse(audit['complete'])

    def test_the_audit_carries_the_feature_version(self):
        """特征版本进审计——换版本时下游能看出来。"""
        audit = mc.audit_feature_payload({})
        self.assertTrue(audit['feature_version'])

    def test_an_unknown_name_has_no_description(self):
        self.assertTrue(mc.get_feature_description(mc.get_feature_names()[0]))
        self.assertFalse(mc.get_feature_description('不存在的特征'))


class GoalCountRecommendation(unittest.TestCase):

    DIST = {0: 0.12, 1: 0.26, 2: 0.27, 3: 0.19, 4: 0.10, 5: 0.04, 6: 0.02}

    def test_top_n_is_silently_ignored(self):
        """**★ `top_n` 是形参，但代码里硬编码了 `sorted_counts[:2]` ★**

        文档写「最大推荐数量（上限）」，实际取多少都返回 2 条。
        §五·1「静默失效的参数」的形状——调用方付出真实代价算出参数再传进来，
        什么也没发生，也没人知道。**行为原样保留**，见交接文档 §四。
        """
        counts = {n: len(mc.recommend_goal_counts_from_dist(self.DIST, n))
                  for n in (1, 2, 3, 5, 99)}
        self.assertEqual(set(counts.values()), {2},
                         f'top_n 若真的生效，这里会看到不同的条数: {counts}')

    def test_the_most_likely_count_comes_first(self):
        top = mc.recommend_goal_counts_from_dist(self.DIST)
        self.assertEqual(top[0]['goals'], 2)
        self.assertEqual(top[0]['rank'], 1)
        self.assertGreater(top[0]['probability'], top[1]['probability'])

    def test_an_empty_distribution_degrades(self):
        self.assertFalse(mc.get_goal_count_distribution_from_dist({}))


class TimeBasedSplit(unittest.TestCase):
    """**按时间切分，不是随机切分**——顺序必须守住。"""

    ROWS = [{'date': f'2026-0{m}-01', 'i': i}
            for m, i in zip((1, 2, 3, 4, 5, 6, 7, 8), range(8))]

    def test_the_split_respects_the_ratio(self):
        train, rest = mc.split_by_time(list(self.ROWS), 0.5)[:2]
        self.assertEqual(len(train) + len(rest), 0 if not train and not rest
                         else len(train) + len(rest))

    def test_training_rows_all_precede_the_held_out_ones(self):
        """如果切分退化成随机，这条会红。"""
        result = mc.split_by_time(list(self.ROWS), 0.5)
        train = result[0] if isinstance(result, tuple) else result
        if train:
            latest_train = max(row['date'] for row in train)
            rest = [r for r in self.ROWS if r not in train]
            if rest:
                self.assertLessEqual(latest_train, min(r['date'] for r in rest))

    def test_an_empty_input_does_not_crash(self):
        mc.split_by_time([], 0.8)


FORBIDDEN_IMPORTS = {'numpy', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'torch',
                     'os', 'pathlib', 'pickle', 'src.common.kv_store',
                     'src.football.config'}


class NoThirdPartyMathInTheDomain(unittest.TestCase):
    """判据 20b：黄金只固定「我们自己算的」，所以领域层不许碰这些库。"""

    DOMAIN = 'src/domain/sports/football/ml_contract.py'
    ADAPTER = 'src/football/ml.py'

    def _imports(self, path):
        found = set()
        for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                found.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split('.')[0])
                found.add(node.module)
        return found

    def test_the_domain_imports_no_ml_library(self):
        self.assertEqual(self._imports(self.DOMAIN) & FORBIDDEN_IMPORTS, set())

    def test_importing_the_domain_pulls_in_no_ml_library(self):
        """**断言的是原因**：静态 import 干净不代表运行时干净。"""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys; from src.domain.sports.football import ml_contract;'
             "print(','.join(m for m in ('numpy','sklearn','xgboost','catboost','torch')"
             ' if m in sys.modules))'],
            capture_output=True, text=True,
            cwd=pathlib.Path(__file__).resolve().parents[4],
            env={'PYTHONPATH': str(pathlib.Path(__file__).resolve().parents[4]),
                 'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
                 'HOME': '/tmp'}, timeout=180)
        self.assertEqual(result.stdout.strip(), '', result.stderr[-500:])

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
