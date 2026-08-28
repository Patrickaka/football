"""北单的赛果解读与组装。

参照物是从迁移前的 `recommending.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_analysis.json.gz`，150 条），**逐条相同**。

语料的比分分布一律由 `predict_scores_by_poisson` 真实算出来（判据 10），
只有需要专门撞某条分支时才手工构造。**三组输入是为了分离两道判定而挑的**：
`build_decision` 与 `build_score_strategy` 的门槛不同，只喂「都过」或
「都不过」的样本分不出是谁在起作用——实测（判据 28）英超 0.72 那组
首选 0.7019、top1 比分 0.1374 → 单选 ✓ 单比分 ✗；意甲 0.62 那组
首选 0.6204、top1 比分 0.1450 → 单选 ✗ 单比分 ✓。

**两种比分分布形态的输出不会逐字相同**：JSON 列表形态本身是 `round(p, 6)`
存下来的，差异全落在第 6 位小数。它们覆盖的是「两条入口都走得通」，不是相等。
"""
import ast
import gzip
import json
import pathlib
import unittest
from unittest import mock

from src.beidan import recommending as adapter
from src.domain.sports.beidan import analysis
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_analysis.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时生效的那组常量，写死不 import（判据 4、12）
TOTAL_LINE = 2.5
HIGH_SCORE_MIN_OVER = 0.52
CLEAR_EDGE_MARGIN = 0.12
DEFAULT_CONFIDENCE = 0.5
STRONG_MIN_PROBABILITY, STRONG_MIN_LEAD = 0.65, 0.10
MAX_GOALS = 7


def golden_entries():
    from scripts.gen_beidan_analysis_golden import entries
    return entries()


def _matrix(pairs):
    return dict(pairs)


def _spf(score_probs, **fields):
    """最小的 spf 结果：只带这一层真正读的那几个键。"""
    return {'home': '主队', 'away': '客队', 'score_probs': score_probs, **fields}


def _analyse(score_probs, **fields):
    return analysis.build_match_analysis(
        _spf(score_probs, **fields),
        min_single=STRONG_MIN_PROBABILITY, min_margin=STRONG_MIN_LEAD)


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class ScoreMatrixInputTests(unittest.TestCase):
    """比分分布两种形态：元组键字典与 JSON 安全的三元组列表。"""

    MATRIX = {(2, 1): 0.30, (1, 1): 0.25, (0, 1): 0.25, (3, 0): 0.20}

    def test_both_forms_are_accepted(self):
        as_list = [[h, a, p] for (h, a), p in self.MATRIX.items()]
        self.assertEqual(_analyse(self.MATRIX)['wdl'],
                         _analyse(as_list)['wdl'])

    def test_ragged_list_items_are_dropped_not_defaulted(self):
        """长度不为 3 的项直接丢——补一个默认概率会让写坏的记录混进统计。"""
        result = _analyse([[2, 1, 0.6], [1, 1], [0, 1, 0.4]])
        self.assertAlmostEqual(result['wdl']['胜'], 0.6)
        self.assertAlmostEqual(result['wdl']['负'], 0.4)
        self.assertAlmostEqual(result['wdl']['平'], 0.0)

    def test_missing_or_empty_input_returns_none(self):
        for bad in (None, {}, {'error': 'x'}, _spf({}), _spf([])):
            with self.subTest(bad=bad):
                self.assertIsNone(analysis.build_match_analysis(
                    bad, min_single=STRONG_MIN_PROBABILITY,
                    min_margin=STRONG_MIN_LEAD))


class MarginalConsistencyTests(unittest.TestCase):
    """胜平负、进球数、比分**必须来自同一张表**——这是这一层存在的理由。"""

    MATRIX = {(2, 1): 0.30, (1, 1): 0.25, (0, 1): 0.25, (3, 0): 0.20}

    def test_wdl_is_the_marginal_of_the_matrix(self):
        result = _analyse(self.MATRIX)
        self.assertAlmostEqual(result['wdl']['胜'], 0.50)
        self.assertAlmostEqual(result['wdl']['平'], 0.25)
        self.assertAlmostEqual(result['wdl']['负'], 0.25)
        self.assertAlmostEqual(sum(result['wdl'].values()), 1.0)

    def test_goals_are_the_marginal_of_the_same_matrix(self):
        result = _analyse(self.MATRIX)['goals']
        # 3 球：(2,1) 0.30 + (3,0) 0.20 = 0.50；2 球：(1,1) 0.25；1 球：(0,1) 0.25
        by_goals = {item['goals']: item['probability']
                    for item in result['top_goals']}
        self.assertAlmostEqual(by_goals[3], 0.50)
        self.assertAlmostEqual(result['expected'], 3 * 0.5 + 2 * 0.25 + 1 * 0.25)

    def test_over_under_split_at_the_line(self):
        """2.5 那条线两侧分开算，**等于 2.5 的进球数不存在**，所以不会漏。"""
        result = _analyse(self.MATRIX)['goals']
        self.assertEqual(result['line'], TOTAL_LINE)
        self.assertAlmostEqual(result['over_prob'], 0.50)
        self.assertAlmostEqual(result['under_prob'], 0.50)
        # 相等时归大球（`over >= under`），把它改成 `>` 这条就会翻
        self.assertEqual(result['direction'], '大球')

    def test_under_wins_when_strictly_larger(self):
        result = _analyse({(1, 0): 0.60, (2, 1): 0.40})['goals']
        self.assertEqual(result['direction'], '小球')
        self.assertAlmostEqual(result['direction_prob'], 0.60)


class ScorePickTests(unittest.TestCase):

    def test_secondary_must_differ_in_some_dimension(self):
        """`2-1` 之后不能再推 `3-2`……不对，**能**：两注都是主胜但进球数不同。

        真正被排除的是三个维度全同的那种。这里 `3-2` 与 `2-1` 进球数不同，
        所以它是合法次选；而下一条用例里的同分同向同球数才会被跳过。
        """
        picks = _analyse({(2, 1): 0.40, (3, 2): 0.35, (1, 0): 0.25})['score_picks']
        self.assertEqual(picks[0]['score'], '2-1')
        self.assertEqual(picks[1]['score'], '3-2')

    def test_same_direction_and_same_goals_is_skipped(self):
        """`3-0` 与 `2-1` 都是主胜、都是 3 球——次选要跳过它，找下一个。"""
        picks = _analyse({(2, 1): 0.40, (3, 0): 0.35, (1, 1): 0.25})['score_picks']
        self.assertEqual(picks[0]['score'], '2-1')
        self.assertEqual(picks[1]['score'], '1-1')

    def test_falls_back_to_the_runner_up_when_nothing_differs(self):
        picks = _analyse({(2, 1): 0.60, (3, 0): 0.40})['score_picks']
        self.assertEqual([p['score'] for p in picks[:2]], ['2-1', '3-0'])

    def test_single_score_has_no_secondary(self):
        picks = _analyse({(1, 0): 1.0})['score_picks']
        self.assertEqual([p['type'] for p in picks], ['首推'])

    def test_defensive_pick_only_when_upset_alerts(self):
        matrix = {(2, 1): 0.40, (1, 1): 0.35, (0, 1): 0.25}
        upset = {'alert': True, 'candidates': [{'score': '0-1'}]}
        with_alert = _analyse(matrix, upset=upset)['score_picks']
        self.assertIn('防冷', [p['type'] for p in with_alert])
        without = _analyse(matrix, upset=dict(upset, alert=False))['score_picks']
        self.assertNotIn('防冷', [p['type'] for p in without])

    def test_defensive_pick_carries_the_matrix_probability(self):
        picks = _analyse({(2, 1): 0.40, (1, 1): 0.35, (0, 1): 0.25},
                         upset={'alert': True, 'candidates': [{'score': '0-1'}]})
        cover = next(p for p in picks['score_picks'] if p['type'] == '防冷')
        self.assertAlmostEqual(cover['probability'], 0.25)
        self.assertEqual(cover['result'], '负')

    def test_defensive_pick_of_an_absent_score_is_zero_not_missing(self):
        """候选比分不在分布里时给 0，而不是漏掉这一注——**它仍然是一条建议**。"""
        picks = _analyse({(2, 1): 1.0},
                         upset={'alert': True, 'candidates': [{'score': '9-9'}]})
        cover = next(p for p in picks['score_picks'] if p['type'] == '防冷')
        self.assertEqual(cover['probability'], 0.0)

    def test_unparsable_upset_candidate_drops_only_that_pick(self):
        """防冷是附加的一注，它解析不出来不该让整份分析变成 None。"""
        for bad in ({'score': 'x-y'}, {'no_score': 1}):
            with self.subTest(bad=bad):
                result = _analyse({(2, 1): 1.0},
                                  upset={'alert': True, 'candidates': [bad]})
                self.assertIsNotNone(result)
                self.assertNotIn('防冷', [p['type'] for p in result['score_picks']])


class HighScorePickTests(unittest.TestCase):
    """「大比分」这一注的门槛：大球概率要过 0.52。两侧都要撞。

    **第一版语料撞不到这条分支**：挑出来的高分格恰好已经是次选，被去重
    跳掉了，于是「门槛之上」和「门槛之下」都没有大比分，两条用例看起来在
    测门槛，实际测的是去重（判据 23、9c）。下面两组的首推与次选都是低比分，
    高分格排第三，实测 over 分别是 0.48 与 0.56，正好夹住 0.52。
    """

    BELOW = {(1, 0): 0.30, (0, 1): 0.22, (3, 1): 0.18, (2, 2): 0.16, (4, 1): 0.14}
    ABOVE = {(1, 0): 0.24, (0, 1): 0.20, (3, 1): 0.19, (2, 2): 0.19, (4, 1): 0.18}

    def test_added_when_over_probability_clears_the_bar(self):
        result = _analyse(self.ABOVE)
        self.assertGreaterEqual(result['goals']['over_prob'], HIGH_SCORE_MIN_OVER)
        self.assertEqual([p['type'] for p in result['score_picks']],
                         ['首推', '次选', '大比分'])

    def test_not_added_below_the_bar(self):
        result = _analyse(self.BELOW)
        self.assertLess(result['goals']['over_prob'], HIGH_SCORE_MIN_OVER)
        self.assertEqual([p['type'] for p in result['score_picks']],
                         ['首推', '次选'])

    def test_threshold_is_a_parameter_not_a_baked_in_number(self):
        """同一份输入，只把门槛调低，这一注就该出现。"""
        result = analysis.build_match_analysis(
            _spf(self.BELOW), min_single=STRONG_MIN_PROBABILITY,
            min_margin=STRONG_MIN_LEAD, high_score_min_over=0.10)
        self.assertIn('大比分', [p['type'] for p in result['score_picks']])

    def test_carries_the_tail_probability_separately(self):
        """这一注带两个概率：这一格自己的，与「4 球以上」整条尾部的。
        只有后者能解释「为什么值得补这一注」。"""
        pick = next(p for p in _analyse(self.ABOVE)['score_picks']
                    if p['type'] == '大比分')
        self.assertAlmostEqual(pick['probability'], 0.19)
        self.assertGreater(pick['scenario_probability'], pick['probability'])

    def test_not_duplicated_when_already_picked(self):
        """高分格恰好就是首推或次选时不重复添加。"""
        result = _analyse({(3, 1): 0.40, (2, 2): 0.35, (1, 0): 0.25})
        scores = [p['score'] for p in result['score_picks']]
        self.assertEqual(len(scores), len(set(scores)))

    def test_tail_probability_is_zero_without_a_high_score_scenario(self):
        result = _analyse({(1, 0): 0.60, (0, 1): 0.40})
        self.assertEqual(result['goals']['high_score_probability'], 0.0)


class VerdictAndReasonTests(unittest.TestCase):

    def test_clear_edge_wording_switches_at_the_margin(self):
        """措辞门槛的两侧各一条。只测一侧的话，把 0.12 改成 0.30 照样全绿。"""
        wide = _analyse({(3, 0): 0.60, (0, 1): 0.40})
        self.assertGreaterEqual(wide['margin'], CLEAR_EDGE_MARGIN)
        self.assertIn('明显', wide['verdict'])
        narrow = _analyse({(3, 0): 0.52, (0, 1): 0.48})
        self.assertLess(narrow['margin'], CLEAR_EDGE_MARGIN)
        self.assertIn('有限', narrow['verdict'])

    def test_draw_favourite_gets_its_own_sentence(self):
        result = _analyse({(1, 1): 0.50, (2, 1): 0.25, (0, 1): 0.25})
        self.assertEqual(result['favorite'], '平')
        self.assertIn('势均力敌', result['verdict'])

    def test_team_names_come_from_the_input(self):
        result = analysis.build_match_analysis(
            {'home': '曼城', 'away': '狼队', 'score_probs': {(3, 0): 1.0}},
            min_single=STRONG_MIN_PROBABILITY, min_margin=STRONG_MIN_LEAD)
        self.assertIn('曼城', result['verdict'])

    def test_missing_team_names_fall_back_to_generic_labels(self):
        result = analysis.build_match_analysis(
            {'score_probs': {(3, 0): 1.0}}, min_single=STRONG_MIN_PROBABILITY,
            min_margin=STRONG_MIN_LEAD)
        self.assertIn('主队', result['verdict'])

    def test_each_reason_needs_its_own_input(self):
        """四条理由各由一个字段触发——少喂一个就少一条。"""
        bare = _analyse({(2, 1): 1.0})
        self.assertEqual(bare['reasons'], [])
        with_lambdas = _analyse({(2, 1): 1.0}, lambda_home=1.8, lambda_away=0.9)
        self.assertEqual(len(with_lambdas['reasons']), 1)
        self.assertIn('主队进攻占优', with_lambdas['reasons'][0])
        with_odds = _analyse({(2, 1): 1.0}, odds={'胜': 1.8, '平': 3.5, '负': 4.2})
        self.assertEqual(len(with_odds['reasons']), 1)
        with_trend = _analyse({(2, 1): 1.0}, asian_trend={'direction': '升盘'})
        self.assertEqual(len(with_trend['reasons']), 1)

    def test_trend_without_a_direction_adds_no_reason(self):
        result = _analyse({(2, 1): 1.0}, asian_trend={'strength': 9.9})
        self.assertEqual(result['reasons'], [])

    def test_alert_and_confident_are_mutually_exclusive(self):
        """两条理由走的是 if/elif——预警时不会同时出现稳胆那句。"""
        both = _analyse({(2, 1): 1.0},
                        upset={'alert': True, 'confident': True, 'label': 'high',
                               'favorite': '胜', 'gap': 0.05})
        self.assertEqual(len(both['reasons']), 1)
        self.assertIn('爆冷预警', both['reasons'][0])

    def test_lambda_bias_has_three_wordings(self):
        for home, away, wording in ((1.8, 0.9, '主队进攻占优'),
                                    (0.9, 1.8, '客队进攻占优'),
                                    (1.2, 1.2, '双方攻防均衡')):
            with self.subTest(wording=wording):
                result = _analyse({(2, 1): 1.0}, lambda_home=home,
                                  lambda_away=away)
                self.assertIn(wording, result['reasons'][0])


class ConfidenceAndDecisionTests(unittest.TestCase):

    MATRIX = {(2, 1): 0.40, (1, 1): 0.35, (0, 1): 0.25}

    def test_split_is_folded_into_low_for_the_decision(self):
        """`split` 与 `low` 在决策层是同一件事，而在 quality 里不是。"""
        as_split = _analyse(self.MATRIX, quality={'level': 'split'})
        as_low = _analyse(self.MATRIX, quality={'level': 'low'})
        self.assertEqual(as_split['decision'], as_low['decision'])
        self.assertEqual(as_split['risk_level'], 'split')
        self.assertEqual(as_low['risk_level'], 'low')

    def test_risk_level_reports_the_original_grade(self):
        """`risk_level` 报的是原始分档，不是折叠之后的——两者不能混。"""
        self.assertEqual(_analyse(self.MATRIX,
                                  quality={'level': 'medium'})['risk_level'],
                         'medium')

    def test_missing_confidence_falls_back_to_the_default(self):
        """判据 29：默认值本身是一条分支。"""
        self.assertEqual(_analyse(self.MATRIX)['confidence'], DEFAULT_CONFIDENCE)
        self.assertEqual(_analyse(self.MATRIX, confidence=0.77)['confidence'], 0.77)

    def test_zero_confidence_is_treated_as_missing(self):
        """`or` 兜底会把 0 当成缺失——钉住现状。"""
        self.assertEqual(_analyse(self.MATRIX, confidence=0)['confidence'],
                         DEFAULT_CONFIDENCE)

    def test_thresholds_reach_build_decision(self):
        """把门槛调到不可能达到，单选那条路就必须消失——证明参数真的传下去了。"""
        strong = {(3, 0): 0.80, (0, 1): 0.20}
        single = analysis.build_match_analysis(
            _spf(strong, quality={'level': 'strong'}),
            min_single=0.50, min_margin=0.05)
        self.assertEqual(single['decision']['action'], '单选')
        blocked = analysis.build_match_analysis(
            _spf(strong, quality={'level': 'strong'}),
            min_single=0.99, min_margin=0.05)
        self.assertEqual(blocked['decision']['action'], '观望')

    def test_upset_alert_forces_a_double(self):
        strong = {(3, 0): 0.80, (0, 1): 0.20}
        result = analysis.build_match_analysis(
            _spf(strong, quality={'level': 'strong'}, upset={'alert': True}),
            min_single=0.50, min_margin=0.05)
        self.assertEqual(result['decision']['action'], '双选')
        self.assertTrue(result['upset_alert'])


class ZjqGroupTests(unittest.TestCase):

    LIVE = {'0': 0.033, '1': 0.053, '2': 0.143, '3': 0.214,
            '4': 0.247, '5': 0.140, '6': 0.099, '7+': 0.071}

    def test_groups_overlap_on_two_goals_by_design(self):
        """`'2'` 同时在小球组与中位组里，三组之和大于 1——这是有意的。"""
        groups = {g['key']: g for g in analysis.zjq_groups(self.LIVE)['groups']}
        self.assertIn('2', groups['small']['options'])
        self.assertIn('2', groups['middle']['options'])
        self.assertGreater(sum(g['probability'] for g in groups.values()), 1.0)

    def test_primary_is_the_highest_probability_group(self):
        result = analysis.zjq_groups(self.LIVE)
        self.assertEqual(result['primary'], result['groups'][0])
        self.assertEqual(result['primary']['key'], 'big')
        self.assertAlmostEqual(result['primary']['probability'],
                               0.214 + 0.247 + 0.140 + 0.099 + 0.071, places=6)

    def test_groups_are_sorted_by_probability(self):
        probabilities = [g['probability']
                         for g in analysis.zjq_groups(self.LIVE)['groups']]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_missing_and_dirty_options_count_as_zero(self):
        result = analysis.zjq_groups({'0': None, '1': '0.3', '2': 0.2})
        small = next(g for g in result['groups'] if g['key'] == 'small')
        self.assertAlmostEqual(small['probability'], 0.5)

    def test_empty_input_has_no_primary(self):
        self.assertEqual(analysis.zjq_groups({}),
                         {'groups': [], 'primary': None})

    def test_definitions_are_a_parameter(self):
        result = analysis.zjq_groups({'0': 1.0},
                                     definitions=(('all', '全部', ('0', '1')),))
        self.assertEqual(result['primary']['key'], 'all')
        self.assertEqual(result['primary']['advice'], '全部 0/1')


class BqcTests(unittest.TestCase):

    MATCH = {'id': '1', 'num': '1', 'home': 'A', 'away': 'B',
             'league': 'L', 'time': '18:30'}

    def test_marginalises_half_and_full(self):
        result = analysis.bqc(self.MATCH, {'1': {'胜胜': 2.0, '胜负': 4.0,
                                                 '负负': 4.0}})
        self.assertAlmostEqual(result['half_probabilities']['胜'], 0.75)
        self.assertAlmostEqual(result['full_probabilities']['负'], 0.5)
        self.assertEqual(result['prediction'], '胜胜')

    def test_missing_match_reports_an_error(self):
        self.assertEqual(analysis.bqc(self.MATCH, {})['error'], '半全场数据不可用')

    def test_all_dead_odds_leave_no_error_key(self):
        """**没有 probabilities 时不写 `error`**——与胜平负那几个不同，
        调用方靠键是否存在来判断。原样保留自迁移前（判据 17 的形状）。"""
        result = analysis.bqc(self.MATCH, {'1': {'胜胜': 0, '平平': None}})
        self.assertNotIn('error', result)
        self.assertNotIn('probabilities', result)
        self.assertEqual(result['type'], 'bqc')


class TotalGoalsGateInputTests(unittest.TestCase):

    def test_hong_kong_water_is_converted_to_decimal_odds(self):
        """港水报净赢，加 1 才是欧赔。**漏掉这个 1 两边会同时偏高**，
        而比值看起来仍然正常——所以要断言绝对值，不能只断言大小关系。"""
        market, _, _ = analysis.total_goals_gate_inputs({}, (1.0, 1.0))
        self.assertAlmostEqual(market['over'], 0.5)
        market, _, _ = analysis.total_goals_gate_inputs({}, (0.90, 1.00))
        self.assertAlmostEqual(market['over'], (1 / 1.9) / (1 / 1.9 + 1 / 2.0))

    def test_zero_or_dirty_water_yields_no_market_probabilities(self):
        for water in ((0.0, 0.0), ('x', 'y'), (None, 1.0), (-1.0, 1.0)):
            with self.subTest(water=water):
                market, _, _ = analysis.total_goals_gate_inputs({}, water)
                self.assertEqual(market, {})

    def test_model_split_at_three_goals(self):
        section = {'probabilities': {'0': 0.1, '1': 0.1, '2': 0.2,
                                     '3': 0.3, '4': 0.2, '7+': 0.1}}
        _, over, under = analysis.total_goals_gate_inputs(section, (1.0, 1.0))
        self.assertAlmostEqual(over, 0.6)
        self.assertAlmostEqual(under, 0.4)

    def test_split_point_is_a_parameter(self):
        section = {'probabilities': {'2': 0.5, '3': 0.5}}
        _, over, _ = analysis.total_goals_gate_inputs(
            section, (1.0, 1.0), model_over_from=2)
        self.assertAlmostEqual(over, 1.0)

    def test_ceiling_bucket_counts_as_max_goals(self):
        """`'7+'` 换算成 7——它必须落在大球那侧，写死成别的数会静默改口径。"""
        _, over, under = analysis.total_goals_gate_inputs(
            {'probabilities': {'7+': 1.0}}, (1.0, 1.0), max_goals=MAX_GOALS)
        self.assertAlmostEqual(over, 1.0)
        self.assertAlmostEqual(under, 0.0)

    def test_unparsable_buckets_are_skipped_not_defaulted(self):
        section = {'probabilities': {'2': 0.4, 'x': 0.2, '3': 'y', '4': 0.4}}
        _, over, under = analysis.total_goals_gate_inputs(section, (1.0, 1.0))
        self.assertAlmostEqual(over, 0.4)
        self.assertAlmostEqual(under, 0.4)

    def test_missing_section_is_all_zero(self):
        for section in (None, {}, {'probabilities': {}}):
            with self.subTest(section=section):
                _, over, under = analysis.total_goals_gate_inputs(section, (1.0, 1.0))
                self.assertEqual((over, under), (0.0, 0.0))


class ValueBetTests(unittest.TestCase):

    MATCH = {'num': '1', 'home': 'A', 'away': 'B',
             'spf': {'probabilities': {'胜': 0.50, '平': 0.28, '负': 0.22},
                     'odds': {'胜': 2.60, '平': 3.30, '负': 4.10}}}

    def test_edge_is_probability_minus_implied(self):
        picks = analysis.value_bets([self.MATCH], threshold=0.0)
        win = next(p for p in picks if p['option'] == '胜')
        self.assertAlmostEqual(win['implied_probability'], 1 / 2.60)
        self.assertAlmostEqual(win['edge'], 0.50 - 1 / 2.60)

    def test_threshold_is_strict(self):
        """`edge > threshold`，不是 `>=`。把它改成 `>=` 只有这条会红。"""
        edge = 0.50 - 1 / 2.60
        self.assertEqual(analysis.value_bets([self.MATCH], threshold=edge), [])
        self.assertEqual(len(analysis.value_bets([self.MATCH],
                                                 threshold=edge - 1e-9)), 1)

    def test_sorted_by_edge_descending(self):
        picks = analysis.value_bets([self.MATCH], threshold=-1.0)
        self.assertEqual([p['edge'] for p in picks],
                         sorted((p['edge'] for p in picks), reverse=True))

    def test_open_ended_bucket_is_skipped(self):
        """`'7+'` 是开区间，它的赔率长期虚高，算出来的优势是假的。"""
        match = {'num': '1', 'home': 'A', 'away': 'B',
                 'zjq': {'probabilities': {'7+': 0.90, '2': 0.30},
                         'odds': {'7+': 30.0, '2': 4.0}}}
        picks = analysis.value_bets([match], threshold=0.0)
        self.assertEqual([p['option'] for p in picks], ['2'])

    def test_skipped_options_are_a_parameter(self):
        match = {'num': '1', 'home': 'A', 'away': 'B',
                 'zjq': {'probabilities': {'7+': 0.90}, 'odds': {'7+': 30.0}}}
        self.assertEqual(analysis.value_bets([match], threshold=0.0,
                                             skipped_options={}),
                         analysis.value_bets([match], threshold=0.0,
                                             skipped_options={'zjq': ()}))
        self.assertEqual(len(analysis.value_bets(
            [match], threshold=0.0, skipped_options={})), 1)

    def test_dead_odds_are_ignored(self):
        match = {'num': '1', 'home': 'A', 'away': 'B',
                 'spf': {'probabilities': {'胜': 0.6, '平': 0.2, '负': 0.2},
                         'odds': {'胜': 0, '平': None, '负': -2.0}}}
        self.assertEqual(analysis.value_bets([match], threshold=-1.0), [])

    def test_sections_without_probabilities_are_skipped(self):
        match = {'num': '1', 'home': 'A', 'away': 'B', 'spf': {'odds': {'胜': 2.0}}}
        self.assertEqual(analysis.value_bets([match], threshold=-1.0), [])

    def test_lenient_and_strict_odds_lookup_differ(self):
        """**总进球用 `.get` 容错，另外两支直接下标**——`probabilities` 在而
        `odds` 不在时前者跳过、后者 KeyError。原样保留自迁移前，这条把它钉住。
        """
        lenient = {'num': '1', 'home': 'A', 'away': 'B',
                   'zjq': {'probabilities': {'2': 0.5}}}
        self.assertEqual(analysis.value_bets([lenient], threshold=-1.0), [])
        strict = {'num': '1', 'home': 'A', 'away': 'B',
                  'rqspf': {'probabilities': {'让胜': 0.5}}}
        with self.assertRaises(KeyError):
            analysis.value_bets([strict], threshold=-1.0)

    def test_bet_types_and_their_order_are_a_parameter(self):
        match = {'num': '1', 'home': 'A', 'away': 'B',
                 'spf': self.MATCH['spf'],
                 'bifen': {'probabilities': {'1-0': 0.9}, 'odds': {'1-0': 5.0}}}
        default = analysis.value_bets([match], threshold=0.0)
        self.assertEqual({p['type'] for p in default}, {'spf'})
        widened = analysis.value_bets([match], threshold=0.0,
                                      bet_types=('spf', 'bifen'))
        self.assertIn('bifen', {p['type'] for p in widened})


class CandidateDateTests(unittest.TestCase):

    def test_appends_the_following_days(self):
        self.assertEqual(analysis.candidate_dates('2026-08-28'),
                         ['2026-08-28', '2026-08-29', '2026-08-30'])

    def test_default_span_is_two_days(self):
        """判据 29：不传参数的那条路径要有人守。"""
        self.assertEqual(len(analysis.candidate_dates('2026-08-28')), 3)
        self.assertEqual(analysis.candidate_dates('2026-08-28'),
                         analysis.candidate_dates('2026-08-28', True, 2))

    def test_fallback_can_be_switched_off(self):
        self.assertEqual(analysis.candidate_dates('2026-08-28', allow_fallback=False),
                         ['2026-08-28'])

    def test_zero_days_still_keeps_the_original(self):
        self.assertEqual(analysis.candidate_dates('2026-08-28', days=0),
                         ['2026-08-28'])

    def test_crosses_month_and_year_boundaries(self):
        self.assertEqual(analysis.candidate_dates('2026-08-30', days=3),
                         ['2026-08-30', '2026-08-31', '2026-09-01', '2026-09-02'])
        self.assertEqual(analysis.candidate_dates('2026-12-30', days=3),
                         ['2026-12-30', '2026-12-31', '2027-01-01', '2027-01-02'])

    def test_leap_day_is_not_skipped(self):
        self.assertEqual(analysis.candidate_dates('2028-02-27', days=3),
                         ['2028-02-27', '2028-02-28', '2028-02-29', '2028-03-01'])

    def test_unparsable_date_returns_only_itself(self):
        for bad in ('not-a-date', None, '2026/08/28'):
            with self.subTest(bad=bad):
                self.assertEqual(analysis.candidate_dates(bad), [bad])


class AdapterTests(unittest.TestCase):
    """适配层保住旧名字与旧语义，并接住领域层不再吞的那些异常。"""

    def test_adapter_swallows_failures_and_logs(self):
        """迁移前整个函数体裹在 `except Exception` 里。兜异常留在这一层，
        领域层照抛——这条证明适配层确实还在兜。"""
        with mock.patch.object(analysis, 'build_match_analysis',
                               side_effect=RuntimeError('boom')):
            with mock.patch.object(adapter.log, 'warning') as warned:
                self.assertIsNone(adapter.build_beidan_match_analysis({'a': 1}))
        warned.assert_called_once()

    def test_domain_layer_does_not_swallow(self):
        """反面：同样的缺陷在领域层是抛出来的，不会被压成 `None`。"""
        with mock.patch.object(analysis, 'normalize_probabilities',
                               side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                _analyse({(1, 0): 1.0})

    def test_adapter_feeds_the_shipped_thresholds(self):
        with mock.patch.object(analysis, 'build_match_analysis') as built:
            adapter.build_beidan_match_analysis({'a': 1})
        _, kwargs = built.call_args
        self.assertEqual(kwargs['min_single'], STRONG_MIN_PROBABILITY)
        self.assertEqual(kwargs['min_margin'], STRONG_MIN_LEAD)

    def test_gate_adapter_passes_the_line_and_max_goals(self):
        goals = {'history': [{'line': '2.5', 'over_odds': 0.95,
                              'under_odds': 0.90}]}
        section = {'probabilities': {'7+': 1.0}}
        with mock.patch.object(adapter, 'build_total_goals_gate', create=True):
            gate = adapter.build_beidan_total_goals_accuracy_gate(
                section, goals, '英超')
        self.assertEqual(gate['line'], 2.5)
        self.assertEqual(gate['model_direction'], 'over')

    def test_value_bets_adapter_reuses_the_domain_filter(self):
        payload = {'date': '2026-08-28', 'total_matches': 1,
                   'recommendations': [ValueBetTests.MATCH]}
        with mock.patch.object(adapter, 'generate_beidan_recommendations',
                               return_value=payload):
            result = adapter.find_value_bets(threshold=0.0)
        self.assertEqual(result['date'], '2026-08-28')
        self.assertEqual(result['value_bets'],
                         analysis.value_bets([ValueBetTests.MATCH], 0.0))

    def test_value_bets_adapter_passes_errors_through(self):
        with mock.patch.object(adapter, 'generate_beidan_recommendations',
                               return_value={'error': '没有赛程'}):
            self.assertEqual(adapter.find_value_bets(), {'error': '没有赛程'})

    def test_clearing_the_ouzhi_cache_keeps_the_exported_binding(self):
        """**就地清空，不重新绑定**：`__init__.py` 有一行
        `from .recommending import _ouzhi_cache`，重绑定会让那份导出变成
        没人再写的孤儿副本（§五·2）。这条钉住「两边始终是同一个对象」。
        """
        import src.beidan as package
        adapter._ouzhi_cache['probe'] = {'home': 1.0}
        self.assertIs(package._ouzhi_cache, adapter._ouzhi_cache)
        adapter._clear_ouzhi_cache()
        self.assertEqual(adapter._ouzhi_cache, {})
        self.assertIs(package._ouzhi_cache, adapter._ouzhi_cache)
        self.assertEqual(package._ouzhi_cache, {})


# 领域层不许依赖的东西。**按 import 判定，不按文本**（判据 16）。
FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'requests', 'urllib.request',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.foundation.store', 'src.beidan.fetching'}
# `datetime` 本身不禁——这一层要做日期加减。禁的是拿它读当前时刻。
FORBIDDEN_CALLS = {'now', 'today', 'utcnow'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = 'src/domain/sports/beidan/analysis.py'
    ADAPTER = 'src/beidan/recommending.py'

    def _tree(self, path):
        return ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))

    def _imports(self, path):
        found = set()
        for node in ast.walk(self._tree(path)):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def _clock_calls(self, path):
        return {node.func.attr for node in ast.walk(self._tree(path))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in FORBIDDEN_CALLS}

    def test_domain_imports_nothing_stateful(self):
        self.assertEqual(self._imports(self.DOMAIN) & FORBIDDEN_IMPORTS, set())

    def test_domain_never_reads_the_clock(self):
        """`datetime` 进得来，`datetime.now()` 进不来——日期加减是纯计算，
        读当前时刻不是。只按 import 判定的话这两者分不开。"""
        self.assertEqual(self._clock_calls(self.DOMAIN), set())

    def test_the_guards_would_catch_a_real_violation(self):
        """守卫本身要能被证伪：拿适配层来试，两道**都应该**命中。"""
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())
        self.assertNotEqual(self._clock_calls(self.ADAPTER), set())


if __name__ == '__main__':
    unittest.main()
