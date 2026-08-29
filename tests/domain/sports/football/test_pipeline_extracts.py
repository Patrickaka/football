# -*- coding: utf-8 -*-
"""从 `analyze_match` 里抽出来的两段：市场锚定流水线、结果组装。

参照物是黄金文件（`tests/fixtures/golden/football_pipeline.json.gz`，441 条）。
抽取当时另跑过 **441 条**新旧双跑差分——旧实现是从迁移前的 pipeline 里
原样切下来的同一段代码，零差异。

`analyze_match` 原来 1384 行，抽走这两段后是 1262 行。剩下的部分是
**抓取与计算交织的编排**：起线程池抓亚盘/欧赔/大小球、调 ML 模型、
读生产历史、写缓存。那些不是搬不动，是搬过去也还得注入回来，
本轮没有继续拆——尚存两段接口偏宽的纯计算（入 8 出 8、入 19 出 15），
留给下一轮。

## 顺序是有讲究的

市场锚定的五步顺序写在领域模块的说明里：**收盘 1X2 定方向的边际、
大小球盘口定进球均值、亚盘/大小球的公平价再施加软结算约束**。
历史校准夹在第一步和第三步之间，是因为它必须先于市场锚定生效——
否则一段短期热手就能盖过当前这场的盘口。
"""
import ast
import gzip
import inspect
import json
import pathlib
import unittest
from copy import deepcopy

from src.domain.sports.football.analysis_result import build_analysis_result
from src.domain.sports.football.market_anchoring import anchor_candidates_to_market
from tests.domain.golden import as_comparable
from tests.domain.sports.football._pipeline_corpus import BASE_PARTS, CANDIDATES

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_pipeline.json.gz',
                             'rt', encoding='utf-8'))
TOTAL = {'line': 2.5, 'over': 1.9, 'under': 1.9, 'close': {'line': 2.5}}
EURO = {'H': 2.0, 'D': 3.4, 'A': 4.0,
        'close': {'home': 0.522, 'draw': 0.269, 'away': 0.209}}
ASIAN = {'handicap': -0.5, 'home_odds': 0.95, 'away_odds': 0.95}


def golden_entries():
    from scripts.gen_football_pipeline_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class MarketAnchoring(unittest.TestCase):

    def test_it_reports_all_five_steps(self):
        """五个 meta 键报告层按名字取用——少一个就在页面上悄悄空掉（判据 12）。"""
        _, meta = anchor_candidates_to_market(list(CANDIDATES), TOTAL, EURO, ASIAN)
        self.assertEqual(set(meta), {
            'score_goal_anchor', 'production_history_calibration',
            'outcome_market_anchor', 'final_score_goal_anchor', 'joint_market_state'})

    def test_the_distribution_stays_normalised(self):
        adjusted, _ = anchor_candidates_to_market(
            list(CANDIDATES), TOTAL, EURO, ASIAN)
        self.assertAlmostEqual(sum(p for _, p in adjusted), 1.0, places=6)

    def test_the_goal_mean_moves_toward_the_total_line(self):
        """低盘口把进球均值往下压，高盘口往上抬——**两个方向都测**。"""
        def goal_mean(items):
            return sum(sum(score) * p for score, p in items)

        low, _ = anchor_candidates_to_market(
            list(CANDIDATES), dict(TOTAL, line=1.75), EURO, ASIAN)
        high, _ = anchor_candidates_to_market(
            list(CANDIDATES), dict(TOTAL, line=3.75), EURO, ASIAN)
        self.assertLess(goal_mean(low), goal_mean(high))

    def test_the_outcome_marginal_follows_the_closing_market(self):
        """收盘越偏主队，主胜的边际质量越大。

        **`euro['close']` 装的是去水后的概率，不是赔率**——
        `analyze_euro` 出来就是概率（判据 10）。喂赔率进去方向会整个反过来。
        """
        def home_mass(items):
            return sum(p for (h, a), p in items if h > a)

        favourite = {'close': {'home': 0.745, 'draw': 0.176, 'away': 0.079}}
        underdog = {'close': {'home': 0.150, 'draw': 0.214, 'away': 0.636}}
        strong, _ = anchor_candidates_to_market(list(CANDIDATES), TOTAL, favourite, ASIAN)
        weak, _ = anchor_candidates_to_market(list(CANDIDATES), TOTAL, underdog, ASIAN)
        self.assertGreater(home_mass(strong), home_mass(weak))

    def test_a_missing_history_profile_simply_does_not_calibrate(self):
        """**不注入档案就是不校准**——与迁移前"读不到历史"时的行为一致。"""
        for profile in (None, {}):
            with self.subTest(profile=profile):
                _, meta = anchor_candidates_to_market(
                    list(CANDIDATES), TOTAL, EURO, ASIAN, profile)
                self.assertFalse(
                    meta['production_history_calibration'].get('applied'))

    def test_an_active_profile_does_calibrate(self):
        """**反方向**：给了可用的档案就得真的动手，否则注入形同虚设。"""
        profile = {'applied': True, 'goal_beta': 0.05, 'sample_count': 120,
                   'outcome_weights': {'H': 1.05, 'D': 0.95, 'A': 1.0}}
        _, meta = anchor_candidates_to_market(
            list(CANDIDATES), TOTAL, EURO, ASIAN, profile)
        self.assertTrue(meta['production_history_calibration']['applied'])

    def test_one_broken_step_does_not_take_down_the_rest(self):
        """某一步失败要留下 `{'applied': False, 'reason': ...}`，不能让整条塌掉。"""
        _, meta = anchor_candidates_to_market(list(CANDIDATES), TOTAL, EURO, None)
        self.assertFalse(meta['joint_market_state']['applied'])
        self.assertIn('reason', meta['joint_market_state'])
        self.assertTrue(meta['score_goal_anchor'].get('applied'))

    def test_no_candidates_is_not_a_crash(self):
        adjusted, meta = anchor_candidates_to_market([], TOTAL, EURO, ASIAN)
        self.assertEqual(adjusted, [])
        self.assertEqual(len(meta), 5)


class AnalysisResult(unittest.TestCase):

    TOP_LEVEL = ('anomaly', 'asian', 'bookmaker_consensus', 'calibration_effect',
                 'confidence', 'euro', 'league_profile', 'live_context',
                 'live_context_quality', 'lottery', 'market_change', 'match',
                 'model', 'model_status', 'model_weights', 'probability_rank',
                 'professional_evidence', 'recommend_rank', 'risk_level',
                 'settlement', 'similar_market', 'similar_market_detail',
                 'single_odds', 'steam_move', 'team', 'total', 'upset')
    UNDER_MODEL = ('candidates', 'dixon_coles', 'goal_calibration', 'goal_count',
                   'half_full_time', 'lam_away', 'lam_home', 'ml', 'recommend',
                   'risk_level', 'score_goal_anchor', 'top_scores', 'value_bets')

    def test_the_output_contract_holds(self):
        """**字段名就是契约**——报告层与 HTTP 层都按这些名字取（判据 12）。

        入参叫 `risk` 和 `recommend`，出来却是 `risk_level` 和 `model.recommend`：
        改名不会让任何东西红，只会让页面上那一块悄悄空掉。
        """
        result = build_analysis_result(**deepcopy(BASE_PARTS))
        self.assertEqual(sorted(result), sorted(self.TOP_LEVEL))
        self.assertEqual(sorted(result['model']), sorted(self.UNDER_MODEL))

    def test_the_accuracy_gate_hangs_off_the_lottery(self):
        result = build_analysis_result(**deepcopy(BASE_PARTS))
        self.assertIn('accuracy_gate', result['lottery'])

    def test_the_caller_supplies_exactly_the_declared_parameters(self):
        """**调用处传的名字，必须在调用处真的存在。**

        这条是补出来的：抽取时把 `k` 也列成了参数——它其实只是原代码里
        推导式的局部变量（`{k: match.get(k) for k in (...)}`），
        我的数据流分析把推导式的绑定当成了从外部读入的自由名。

        于是 `pipeline.py` 里生成了 `k=k`，而 `analyze_match` 里根本没有 `k`。
        **每次调用必抛 NameError**，被上层 try 吞成一行日志——线上一天报了
        2450 次「时间分层预测异常」，足球分析实际全挂。

        黄金、双跑差分、四十个部件的置空测试**全都没发现**：它们无一例外
        地显式传了 `k=3`，谁也没去看调用处到底有没有这个名字。
        """
        import ast
        import pathlib

        signature = set(inspect.signature(build_analysis_result).parameters)
        source = pathlib.Path('src/football/pipeline.py').read_text(encoding='utf-8')
        caller = next(n for n in ast.walk(ast.parse(source))
                      if isinstance(n, ast.Call)
                      and getattr(n.func, 'id', None) == 'build_analysis_result')
        passed = {kw.arg for kw in caller.keywords}
        self.assertEqual(passed, signature,
                         f'调用处与签名对不上: 多传 {passed - signature}, '
                         f'少传 {signature - passed}')

    def test_no_parameter_is_really_a_comprehension_variable(self):
        """推导式的 `for x in ...` 绑定的是**局部**名字，不是外部入参。

        把它当成参数，调用处就会去传一个根本不存在的变量。
        """
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(
            'src/domain/sports/football/analysis_result.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        params = {a.arg for a in function.args.kwonlyargs + function.args.args}
        bound_by_comprehensions = set()
        for node in ast.walk(function):
            if isinstance(node, (ast.ListComp, ast.SetComp,
                                 ast.DictComp, ast.GeneratorExp)):
                for generator in node.generators:
                    for target in ast.walk(generator.target):
                        if isinstance(target, ast.Name):
                            bound_by_comprehensions.add(target.id)
        self.assertEqual(params & bound_by_comprehensions, set())

    # 逐个部件 × 三种空值实测出来的分界（判据 10：读实测值，别按名字猜）
    BLANK_TOLERANT = frozenset({
        'calibration_effect', 'candidates', 'confidence', 'dixon_coles_result',
        'euro', 'euro_asian_dev', 'goal_dist_after_calibration',
        'goal_dist_before_calibration', 'half_full_time', 'joint_anomaly',
        'lam_away', 'lam_home', 'league_profile', 'live_context',
        'live_context_quality', 'market_change_result', 'ml_result',
        'model_status', 'model_weights', 'probability_rank',
        'production_spf_policy', 'recommend', 'recommend_rank', 'settlement',
        'similar_market_detail', 'similar_market_result', 'single_odds',
        'steam_result', 'team', 'top_scores', 'total', 'upset', 'value_bets'})
    DICT_ONLY = frozenset({'asian', 'match', 'meta', 'lottery'})
    NOT_A_LIST = frozenset({'goal_count_result'})
    MANDATORY = frozenset({'risk'})

    def _blank(self, key, value):
        parts = deepcopy(BASE_PARTS)
        parts[key] = value
        return parts

    def test_thirty_three_parts_survive_any_blank(self):
        """**逐个置空**——上游降级不该让组装塌掉。四十个里三十四个全撑得住。"""
        for key in sorted(self.BLANK_TOLERANT):
            for blank in (None, {}, []):
                with self.subTest(part=key, blank=blank):
                    build_analysis_result(**self._blank(key, blank))

    def test_the_classification_covers_every_part(self):
        """四个分组必须正好覆盖四十个部件——漏一个就等于没测那一个。"""
        classified = (self.BLANK_TOLERANT | self.DICT_ONLY
                      | self.NOT_A_LIST | self.MANDATORY)
        self.assertEqual(classified, set(BASE_PARTS))
        self.assertEqual(len(classified), 39)

    def test_four_parts_only_tolerate_an_empty_dict(self):
        """`asian`/`match`/`meta`/`lottery` 被直接 `.get` 或解包——
        换成 `None` 或列表就抛。
        """
        for key in sorted(self.DICT_ONLY):
            with self.subTest(part=key, blank={}):
                build_analysis_result(**self._blank(key, {}))
            for blank in (None, []):
                with self.subTest(part=key, blank=blank):
                    with self.assertRaises((AttributeError, TypeError, KeyError)):
                        build_analysis_result(**self._blank(key, blank))

    def test_goal_count_result_tolerates_everything_but_a_list(self):
        for blank in (None, {}):
            with self.subTest(blank=blank):
                build_analysis_result(**self._blank('goal_count_result', blank))
        with self.assertRaises(TypeError):
            build_analysis_result(**self._blank('goal_count_result', []))

    def test_risk_is_the_only_wholly_mandatory_part(self):
        """`risk` 三种空值全抛——它被直接下标取 `level` 与 `description`。

        这意味着上游 `_evaluate_risk_level` 一旦降级返回空，整场分析就
        **没有结果**，而不是降级出一份结果。行为照搬未改。
        """
        for blank in (None, {}, []):
            with self.subTest(blank=blank):
                with self.assertRaises((KeyError, TypeError)):
                    build_analysis_result(**self._blank('risk', blank))

    def test_it_takes_keyword_arguments_only(self):
        """四十个位置参数没人能读对，签名强制关键字。"""
        with self.assertRaises(TypeError):
            build_analysis_result({}, {}, {})

    def test_it_writes_the_gate_back_into_the_caller_s_lottery(self):
        """**就地修改**：`accuracy_gate` 是塞进传进来的那个 `lottery` 里的。

        浅拷贝挡不住——黄金第一版就因此不可复现（第二次调用看到的是
        被污染的输入，`information_completeness` 从 1.0 掉到 0.62）。
        行为照搬未改，调用方要么接受，要么自己深拷贝。
        """
        lottery = deepcopy(BASE_PARTS['lottery'])
        parts = deepcopy(BASE_PARTS)
        parts['lottery'] = lottery
        self.assertNotIn('accuracy_gate', lottery)
        build_analysis_result(**parts)
        self.assertIn('accuracy_gate', lottery)

    def test_the_evidence_profile_is_attached(self):
        result = build_analysis_result(**deepcopy(BASE_PARTS))
        self.assertIsNotNone(result['professional_evidence'])


DOMAIN_FILES = {'market_anchoring': 'src/domain/sports/football/market_anchoring.py',
                'analysis_result': 'src/domain/sports/football/analysis_result.py'}
FORBIDDEN_IMPORTS = {'os', 'pathlib', 'requests', 'concurrent.futures',
                     'src.common.kv_store', 'src.football.config',
                     'src.football.result_sync', 'src.football.parsing',
                     'src.football.fetching'}


class NoSideEffectTests(unittest.TestCase):

    ADAPTER = 'src/football/pipeline.py'

    def _imports(self, path):
        found = set()
        for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def test_neither_module_imports_anything_stateful(self):
        for name, path in DOMAIN_FILES.items():
            with self.subTest(module=name):
                self.assertEqual(self._imports(path) & FORBIDDEN_IMPORTS, set())

    def test_every_relative_import_resolves_inside_the_domain_package(self):
        """延迟 import 搬进领域包后解析不到会**静默落到兜底**——F-13 栽过两次。"""
        siblings = {p.stem for p in pathlib.Path('src/domain/sports/football').glob('*.py')}
        for name, path in DOMAIN_FILES.items():
            for node in ast.walk(ast.parse(
                    pathlib.Path(path).read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.level:
                    with self.subTest(module=name, imported=node.module):
                        self.assertIn(node.module, siblings)

    def test_neither_module_fetches_or_caches(self):
        for name, path in DOMAIN_FILES.items():
            source = pathlib.Path(path).read_text(encoding='utf-8')
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Name):
                    with self.subTest(module=name, name=node.id):
                        self.assertNotIn(node.id, {'set_cache', 'get_cache', 'open'})

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
