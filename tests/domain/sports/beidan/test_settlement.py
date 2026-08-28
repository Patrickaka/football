"""北单赛果判定与历史可靠性校准。

参照物是从迁移前的 `settling.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_settlement.json.gz`，287 条）。
**287 条里 277 条逐条相同，10 条是有意的改动**，全部来自同一处：

迁移前 `_actual_rqspf_from_record` 用 `float(record['handicap'])` 解析盘口，
而线上历史记录里盘口存的是页面文本 —— 2026-08-28 读到的 500 条中，143 条有值，
形如 `'(-1)'`（83）、`'(+1)'`（32）、`'(-2)'`（23）、`'(+2)'`（5），其余为 `None`。
`float('(-1)')` 抛 ValueError，被 except 吞成「按平手盘算」，于是**所有分盘
一律退化成不让球**，`让平` 被系统性地误判成让胜或让负。改成与同包另外两处
（`markets.py:63`、`recommending.py:221`）相同的 `handicap.parse`。

变的 10 条：6 条是字符串盘口的直接判定（`让胜`/`让负` → `让平`），
4 条是拿字符串盘口铺出来的校准语料（`让胜 10.0` → `让胜 5.0 + 让平 5.0`）。
数字盘口、`None`、无法解析的脏值三种输入**一条都没变**——
这正是「只有那一处变了」的证据。

**这不改变任何线上行为**：这个函数的两个消费者（历史校准、
`summarize_beidan_history`）都以 `record['settled']` 为前提，而线上 500 条
`settled` 全是 False，`recommending.py:146` 只在「已经是 True」时保留它，
仓库里没有任何一处会把它置 True。整条「预测 → 赛果回填 → 校准」从投产起
就没有接上过（详见交接文档 §五）。
"""
import gzip
import json
import pathlib
import unittest
from unittest import mock

from src.common import kv_store
from src.beidan.settling import (
    _load_beidan_history, _save_beidan_history, apply_beidan_history_calibration,
)
from src.domain.sports.beidan import settlement
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_settlement.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时生效的那组常量，写死不 import（判据 4、12）
FACTOR_MIN, FACTOR_MAX = 0.86, 1.16
FACTOR_PRIOR = 6.0
SAME_LEAGUE_WEIGHT = 1.25
DEFAULT_MIN_SAMPLES = 8
DEFAULT_LIMIT = 200
MAX_GOALS = 7


def golden_entries():
    from scripts.gen_beidan_settlement_golden import entries
    return entries()


def _settled(score, probs=None, league='K2联赛', handicap=None, section='spf'):
    return {
        'league': league, 'handicap': handicap, 'settled': True,
        'actual': {'score': score},
        section: {'probabilities': probs or {'胜': 0.40, '平': 0.30, '负': 0.30}},
    }


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class ImpliedProbabilityTests(unittest.TestCase):
    """去水：分子与分母必须用同一道过滤，否则脏赔率会算出大于 1 的概率。"""

    def test_normalises_to_one(self):
        probs = settlement.implied_probability({'胜': 2.0, '平': 4.0, '负': 4.0})
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs['胜'], 0.5)

    def test_dead_option_is_dropped_from_both_sides(self):
        """0 赔率的那一档既不出现在结果里，也不进分母。"""
        probs = settlement.implied_probability({'胜': 2.0, '平': 0, '负': 2.0})
        self.assertEqual(sorted(probs), ['胜', '负'])
        self.assertAlmostEqual(probs['胜'], 0.5)
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_negative_odds_dropped(self):
        probs = settlement.implied_probability({'胜': 2.0, '平': -3.0})
        self.assertEqual(list(probs), ['胜'])

    def test_all_dead_returns_empty(self):
        self.assertEqual(settlement.implied_probability({'胜': 0, '平': None}), {})
        self.assertEqual(settlement.implied_probability({}), {})

    def test_none_is_not_a_dict(self):
        """入口守卫挡的是 `None`——空 dict 靠后面那道零分母守卫也能兜住，
        `None` 只能靠这一道。变异验证里去掉它，只有这条会红。"""
        self.assertEqual(settlement.implied_probability(None), {})

    def test_infinite_odds_do_not_divide_by_zero(self):
        """`1/inf` 是 0.0，而 `inf > 0` 成立——分母归零但分子这一档还在，
        没有零分母守卫就会 ZeroDivisionError。"""
        self.assertEqual(settlement.implied_probability({'胜': float('inf')}), {})


class ActualOutcomeTests(unittest.TestCase):
    """五个取值候选、两条不同的候选顺序。"""

    def test_spf_from_score(self):
        self.assertEqual(settlement.actual_spf({'actual_score': '2-1'}), '胜')
        self.assertEqual(settlement.actual_spf({'actual_score': '1-1'}), '平')
        self.assertEqual(settlement.actual_spf({'actual_score': '0-2'}), '负')

    def test_spf_direct_field_wins_over_score(self):
        """直取字段优先——比分说主胜，直取字段说平，结果是平。"""
        record = {'actual_spf': '平', 'actual_score': '2-1'}
        self.assertEqual(settlement.actual_spf(record), '平')

    def test_spf_direct_field_must_be_a_valid_outcome(self):
        """直取字段是脏值时退回比分，而不是把脏值原样返回。"""
        record = {'actual_spf': '胜负', 'actual_score': '2-1'}
        self.assertEqual(settlement.actual_spf(record), '胜')

    def test_spf_candidate_order_prefers_top_level(self):
        """两处不一致时以顶层 `actual_score` 为准。"""
        record = {'actual_score': '2-1', 'actual': {'score': '0-3'}}
        self.assertEqual(settlement.actual_spf(record), '胜')

    def test_rqspf_candidate_order_prefers_nested(self):
        """**让球那条链的顺序与上一条相反**：同一份记录得出相反的结论。

        原样保留自迁移前。两条链写在两个函数里，谁也不会同时看到它们
        （判据 11 的形状），这条用例就是让它们同时出现在一处。
        """
        record = {'actual_score': '2-1', 'actual': {'score': '0-3'}}
        self.assertEqual(settlement.actual_rqspf(record, 0.0), '让负')
        self.assertEqual(settlement.actual_spf(record), '胜')

    def test_unparsable_score_returns_none(self):
        for bad in ('21', 'a-b', '', None):
            with self.subTest(score=bad):
                self.assertIsNone(settlement.actual_spf({'actual_score': bad}))
                self.assertIsNone(settlement.actual_bifen({'actual_score': bad}))

    def test_non_dict_containers_are_ignored(self):
        self.assertIsNone(settlement.actual_spf({'actual': '2-1'}))
        self.assertIsNone(settlement.actual_spf({'settlement': ['2-1']}))

    def test_bifen_normalises_score(self):
        self.assertEqual(settlement.actual_bifen({'actual_score': '2-1'}), '2-1')
        self.assertEqual(settlement.actual_bifen({'settlement': {'score': '0-0'}}),
                         '0-0')


class ActualZjqTests(unittest.TestCase):

    def test_buckets_by_total_goals(self):
        self.assertEqual(settlement.actual_zjq({'actual_score': '0-0'}), '0')
        self.assertEqual(settlement.actual_zjq({'actual_score': '2-1'}), '3')
        self.assertEqual(settlement.actual_zjq({'actual_score': '3-3'}), '6')

    def test_ceiling_bucket(self):
        """7 球及以上并成一档，档名跟着 `max_goals` 走。"""
        self.assertEqual(settlement.actual_zjq({'actual_score': '4-3'}), '7+')
        self.assertEqual(settlement.actual_zjq({'actual_score': '5-3'}), '7+')

    def test_default_max_goals_is_seven(self):
        """默认值本身是一条分支（判据 29）：不传参数的那条路径要有人守。"""
        record = {'actual_score': '4-2'}
        self.assertEqual(settlement.actual_zjq(record), '6')
        self.assertEqual(settlement.actual_zjq(record, max_goals=MAX_GOALS), '6')

    def test_max_goals_shifts_both_the_ceiling_and_its_name(self):
        record = {'actual_score': '2-1'}
        self.assertEqual(settlement.actual_zjq(record, max_goals=3), '3+')
        self.assertEqual(settlement.actual_zjq(record, max_goals=4), '3')

    def test_direct_field_outside_range_falls_back_to_score(self):
        record = {'actual_zjq': '9', 'actual_score': '2-1'}
        self.assertEqual(settlement.actual_zjq(record), '3')

    def test_max_goals_itself_is_not_a_bucket_name(self):
        """档位是 `'0'`~`'6'` 加 `'7+'`——**没有 `'7'` 这一档**。

        合法档位取 `range(max_goals)` 而不是 `range(max_goals + 1)`，
        差一位就会让 `'7'` 被当成合法直取值，而它在下游一个也对不上。"""
        record = {'actual_zjq': '7', 'actual_score': '2-1'}
        self.assertEqual(settlement.actual_zjq(record), '3')
        self.assertEqual(settlement.actual_zjq({'actual_zjq': '6'}), '6')

    def test_zero_goal_direct_field_is_skipped_as_if_missing(self):
        """**`0` 是假值，`or` 链把「零球」当成缺失**——钉住现状。

        `actual_zjq=0` 的记录会跳过直取字段去读比分；比分也没有时返回
        `None` 而不是 `'0'`。真实赛果里 0 球并不罕见，启用结算回填前要先修。
        """
        self.assertEqual(settlement.actual_zjq({'actual_zjq': 0,
                                                'actual_score': '2-1'}), '3')
        self.assertIsNone(settlement.actual_zjq({'actual_zjq': 0}))
        # 同一个值写成字符串就正常了，差别只在真假值上
        self.assertEqual(settlement.actual_zjq({'actual_zjq': '0'}), '0')


class ActualRqspfTests(unittest.TestCase):
    """让球值由调用方解析后传入，领域层只做算术。"""

    def test_handicap_shifts_the_margin(self):
        record = {'actual': {'score': '1-0'}}
        self.assertEqual(settlement.actual_rqspf(record, 0.0), '让胜')
        self.assertEqual(settlement.actual_rqspf(record, -1.0), '让平')
        self.assertEqual(settlement.actual_rqspf(record, -2.0), '让负')
        self.assertEqual(settlement.actual_rqspf(record, 1.0), '让胜')

    def test_missing_handicap_is_treated_as_level(self):
        record = {'actual': {'score': '1-0'}}
        self.assertEqual(settlement.actual_rqspf(record, None), '让胜')

    def test_quarter_line_lands_on_a_side_never_on_push(self):
        """半球盘不可能走水——`margin` 永远不等于 0。"""
        for score, expected in (('1-0', '让胜'), ('0-0', '让负')):
            with self.subTest(score=score):
                self.assertEqual(
                    settlement.actual_rqspf({'actual': {'score': score}}, -0.5),
                    expected)


class AdapterHandicapParsingTests(unittest.TestCase):
    """适配层负责把页面文本解析成让球值——**迁移修的就是这一处**。"""

    def test_bracketed_string_handicap_is_parsed(self):
        from src.beidan.settling import _actual_rqspf_from_record
        record = {'handicap': '(-1)', 'actual': {'score': '1-0'}}
        self.assertEqual(_actual_rqspf_from_record(record), '让平')

    def test_fullwidth_brackets_are_parsed(self):
        from src.beidan.settling import _actual_rqspf_from_record
        record = {'handicap': '（-1）', 'actual': {'score': '1-0'}}
        self.assertEqual(_actual_rqspf_from_record(record), '让平')

    def test_all_four_live_handicap_shapes(self):
        """线上真实存在的四种盘口文本，一个都不能落回平手盘。"""
        from src.beidan.settling import _actual_rqspf_from_record
        cases = (('(-1)', '1-0', '让平'), ('(+1)', '0-1', '让平'),
                 ('(-2)', '3-1', '让平'), ('(+2)', '0-2', '让平'))
        for handicap, score, expected in cases:
            with self.subTest(handicap=handicap):
                self.assertEqual(_actual_rqspf_from_record(
                    {'handicap': handicap, 'actual': {'score': score}}), expected)

    def test_unparsable_handicap_falls_back_to_level(self):
        from src.beidan.settling import _actual_rqspf_from_record
        for bad in ('公司', '', None):
            with self.subTest(handicap=bad):
                self.assertEqual(_actual_rqspf_from_record(
                    {'handicap': bad, 'actual': {'score': '1-0'}}), '让胜')


class CalibrationFactorTests(unittest.TestCase):

    def test_no_samples_is_the_identity(self):
        """样本全为零时每个因子恰好 1.0——先验必须加在分子分母两侧。"""
        factors = settlement.calibration_factors(
            {'胜': 0.0, '平': 0.0, '负': 0.0}, {'胜': 0.0, '平': 0.0, '负': 0.0})
        self.assertEqual(set(factors.values()), {1.0})

    def test_more_actual_than_expected_lifts_the_option(self):
        factors = settlement.calibration_factors(
            {'胜': 3.0, '平': 3.0, '负': 3.0}, {'胜': 5.0, '平': 3.0, '负': 1.0})
        self.assertGreater(factors['胜'], 1.0)
        self.assertAlmostEqual(factors['平'], 1.0)
        self.assertLess(factors['负'], 1.0)

    def test_factors_are_clamped_on_both_sides(self):
        """两侧都要撞一次。只撞一侧的话，把另一侧的界改坏是测不出来的。"""
        factors = settlement.calibration_factors(
            {'胜': 0.0, '负': 40.0}, {'胜': 40.0, '负': 0.0})
        self.assertEqual(factors['胜'], FACTOR_MAX)
        self.assertEqual(factors['负'], FACTOR_MIN)

    def test_an_unclamped_factor_sits_strictly_between_the_bounds(self):
        """必须有一条**没被夹到**的样本，否则钳制区间改宽也照样全绿。

        比值是 `(实际+3)/(期望+3)`（两档时先验各摊 3.0），所以实际取 4.0
        时已经是 1.1667、越过上界了——**先把数算出来再写断言**（判据 28）。
        3.5 / 2.5 落在 1.0833 / 0.9167，两侧都没碰到界。
        """
        factors = settlement.calibration_factors(
            {'胜': 3.0, '负': 3.0}, {'胜': 3.5, '负': 2.5})
        self.assertLess(FACTOR_MIN, factors['负'])
        self.assertLess(factors['负'], 1.0)
        self.assertLess(1.0, factors['胜'])
        self.assertLess(factors['胜'], FACTOR_MAX)

    def test_prior_is_split_across_the_options(self):
        """档位越多、先验摊得越薄：同样的悬殊在八档里比在两档里推得更远。"""
        two = settlement.calibration_factors({'a': 0.0, 'b': 2.0},
                                             {'a': 2.0, 'b': 0.0},
                                             factor_min=0.0, factor_max=99.0)
        eight = settlement.calibration_factors(
            {str(i): (0.0 if i == 0 else 2.0 / 7) for i in range(8)},
            {str(i): (2.0 if i == 0 else 0.0) for i in range(8)},
            factor_min=0.0, factor_max=99.0)
        self.assertGreater(eight['0'], two['a'])

    def test_default_prior_shows_up_in_an_unclamped_factor(self):
        """默认值是公开契约的一部分（判据 29），而**要看见它，语料必须
        落在钳制区间之内**——第一版这条用例喂的是 0 对 3 的悬殊样本，
        先验改成 3.0 之后两侧照样被夹到 0.86/1.16，一模一样（判据 23）。

        三档平摊先验 2.0，`(3.5+2)/(3+2) = 1.1`；先验换成 3.0 就是 1.125。"""
        factors = settlement.calibration_factors(
            {'胜': 3.0, '平': 3.0, '负': 3.0}, {'胜': 3.5, '平': 3.0, '负': 2.5})
        self.assertAlmostEqual(factors['胜'], 1.1)
        self.assertAlmostEqual(factors['平'], 1.0)
        self.assertAlmostEqual(factors['负'], 0.9)

    def test_zero_prior_does_not_divide_by_zero(self):
        """先验是可调参数，调成 0 时分母就真的能到 0——护栏为此存在。"""
        self.assertEqual(settlement.calibration_factors(
            {'a': 0.0}, {'a': 0.0}, prior=0.0), {'a': FACTOR_MIN})


class ApplyHistoryCalibrationTests(unittest.TestCase):

    PROBS = {'胜': 0.40, '平': 0.30, '负': 0.30}

    def _apply(self, records, **kwargs):
        return settlement.apply_history_calibration(
            self.PROBS, records, settlement.actual_spf, 'spf', **kwargs)

    def test_four_rejections_are_distinguishable(self):
        """四种「没生效」各有各的 reason——混成一个就查不下去了。"""
        self.assertEqual(
            settlement.apply_history_calibration({}, [], settlement.actual_spf,
                                                 'spf')[1]['reason'],
            'empty_probabilities')
        self.assertEqual(self._apply([])[1]['reason'], 'no_history')
        self.assertEqual(
            self._apply([dict(_settled('2-1'), settled=False)] * 10)[1]['reason'],
            'insufficient_settled_samples')

    def test_rejected_input_is_returned_unchanged(self):
        adjusted, meta = self._apply([])
        self.assertIs(adjusted, self.PROBS)
        self.assertFalse(meta['applied'])

    def test_sample_threshold_is_tested_on_both_sides(self):
        seven = [_settled('2-1') for _ in range(7)]
        self.assertFalse(self._apply(seven)[1]['applied'])
        self.assertTrue(self._apply(seven + [_settled('2-1')])[1]['applied'])

    def test_default_min_samples_matches_the_shipped_value(self):
        records = [_settled('2-1') for _ in range(DEFAULT_MIN_SAMPLES)]
        self.assertTrue(self._apply(records)[1]['applied'])
        self.assertFalse(self._apply(records[:-1])[1]['applied'])

    def test_calibration_actually_moves_the_probabilities(self):
        """判据 27：除了「没报错」，还要断言**结果确实变了**、且方向对。"""
        adjusted, meta = self._apply([_settled('2-1') for _ in range(10)])
        self.assertTrue(meta['applied'])
        self.assertNotEqual(adjusted, self.PROBS)
        self.assertGreater(adjusted['胜'], self.PROBS['胜'])
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_same_league_samples_weigh_more(self):
        records = [_settled('2-1', league='英超') for _ in range(10)]
        plain = self._apply(records)[1]['sample_count']
        weighted = self._apply(records, league='英超')[1]['sample_count']
        self.assertEqual(plain, 10.0)
        self.assertEqual(weighted, 10.0 * SAME_LEAGUE_WEIGHT)

    def test_meta_keeps_enough_precision_to_be_read(self):
        """说明字段的取整位数也是契约：**样本数保到 3 位、因子保到 6 位**。

        整齐的语料看不出来——`1.25 × 10 = 12.5` 与因子恰好撞在钳制界上的
        `0.86`/`1.16`，取 1 位还是 3 位、2 位还是 6 位都一模一样。
        取 9 条同联赛样本（`11.25`，取 1 位会塌成 `11.2`）、
        概率给一组不整齐的值，因子才落到 `0.933063` 这种量级上。
        """
        probs = {'胜': 0.37, '平': 0.31, '负': 0.32}
        history = [_settled(score, probs=probs) for score in
                   ('2-1', '1-1', '0-2', '1-0', '1-1', '0-1', '3-1', '1-1', '0-2')]
        _, meta = settlement.apply_history_calibration(
            probs, history, settlement.actual_spf, 'spf', league='K2联赛')
        self.assertEqual(meta['sample_count'], 11.25)
        self.assertEqual(meta['factors']['胜'], 0.933063)
        self.assertEqual(meta['expected']['平'], 3.488)

    def test_other_league_is_not_weighted(self):
        records = [_settled('2-1', league='K2联赛') for _ in range(10)]
        self.assertEqual(self._apply(records, league='英超')[1]['sample_count'], 10.0)

    def test_limit_truncates_from_the_front(self):
        """窗口只看前 N 条。**不能只断言「截断后样本更少」**——那样把
        `limit` 改大一点照样通过；要断言恰好是那几条。"""
        records = [_settled('2-1') for _ in range(10)]
        self.assertEqual(self._apply(records, limit=3, min_samples=1)[1]['sample_count'], 3.0)
        self.assertEqual(self._apply(records, limit=9, min_samples=1)[1]['sample_count'], 9.0)

    def test_default_limit_matches_the_shipped_value(self):
        records = [_settled('2-1') for _ in range(DEFAULT_LIMIT + 5)]
        self.assertEqual(self._apply(records)[1]['sample_count'], float(DEFAULT_LIMIT))

    def test_unsettled_records_never_count(self):
        records = [dict(_settled('2-1'), settled=False) for _ in range(50)]
        self.assertEqual(self._apply(records, min_samples=1)[1]['sample_count'], 0.0)

    def test_records_without_probabilities_are_skipped(self):
        records = [{'settled': True, 'actual': {'score': '2-1'}, 'spf': {}}
                   for _ in range(10)]
        self.assertEqual(self._apply(records, min_samples=1)[1]['sample_count'], 0.0)

    def test_unresolvable_actual_is_skipped(self):
        records = [_settled('x') for _ in range(10)]
        self.assertEqual(self._apply(records, min_samples=1)[1]['sample_count'], 0.0)

    def test_section_key_selects_the_market(self):
        """分节名错了就一条也攒不上——`section_key` 不是装饰性参数。"""
        records = [_settled('2-1', section='zjq') for _ in range(10)]
        self.assertEqual(self._apply(records, min_samples=1)[1]['sample_count'], 0.0)

    def test_int_keys_collect_samples_but_never_move_the_probabilities(self):
        """**这一条把两处不对称同时钉住**，它们相差一个 `str()`：

        统计侧的档位按 `str(选项)` 建，所以 int 键的概率**攒得到样本**
        （`applied` 为真、因子也算出来了）；而回写侧用的是原始键，
        `factors.get(0)` 一个也对不上，每个选项静默乘以 1.0。
        于是 `applied=True`、因子非平凡、**输出与输入逐位相同**——
        判据 27 说的「不抛异常」与「真的生效」是两回事，就是这个样子。
        """
        history = [{'settled': True, 'actual': {'score': '1-1'},
                    'zjq': {'probabilities': {'0': 0.1, '1': 0.2, '2': 0.3, '3': 0.4}}}
                   for _ in range(12)]
        probs = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        adjusted, meta = settlement.apply_history_calibration(
            probs, history, settlement.actual_zjq, 'zjq')
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['sample_count'], 12.0)
        self.assertEqual(sorted(meta['factors']), ['0', '1', '2', '3'])
        self.assertNotEqual(set(meta['factors'].values()), {1.0})
        self.assertEqual(adjusted, probs)

    def test_all_zero_probabilities_are_rejected_before_normalising(self):
        """全零概率会让归一化的分母也归零——这道守卫是唯一拦它的东西。"""
        records = [_settled('2-1') for _ in range(12)]
        _, meta = self._apply(records)
        self.assertTrue(meta['applied'])
        adjusted, meta = settlement.apply_history_calibration(
            {'胜': 0.0, '平': 0.0, '负': 0.0}, records, settlement.actual_spf, 'spf')
        self.assertEqual(meta['reason'], 'zero_adjusted_total')
        self.assertEqual(adjusted, {'胜': 0.0, '平': 0.0, '负': 0.0})

    def test_non_string_keys_silently_fall_back_to_no_correction(self):
        """**钉住现状**：因子按 `str(选项)` 建，回写时用原始键。

        `recommending.py:852` 的比分那一路传的正是元组键的矩阵，
        `factors.get((1, 0))` 一个也对不上，每个选项静默乘以 1.0。
        校准「成功了」，而输出与输入逐位相同——判据 27 说的正是这种。
        """
        tuple_probs = {(1, 0): 0.5, (1, 1): 0.3, (0, 1): 0.2}
        records = [{'settled': True, 'actual': {'score': s},
                    'bifen': {'probabilities': {'1-0': 0.5, '1-1': 0.3, '0-1': 0.2}}}
                   for s in ['1-0'] * 12]
        adjusted, meta = settlement.apply_history_calibration(
            tuple_probs, records, settlement.actual_bifen, 'bifen')
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'insufficient_settled_samples')
        # 同一份历史，键换成字符串就攒得到样本、也确实改了概率
        str_probs = {'1-0': 0.5, '1-1': 0.3, '0-1': 0.2}
        adjusted, meta = settlement.apply_history_calibration(
            str_probs, records, settlement.actual_bifen, 'bifen')
        self.assertTrue(meta['applied'])
        self.assertNotEqual(adjusted, str_probs)


class RecordKeyTests(unittest.TestCase):

    def test_joins_four_fields(self):
        self.assertEqual(settlement.record_key(
            {'date': '2026-08-28', 'num': '1', 'home': 'A', 'away': 'B'}),
            '2026-08-28|1|A|B')

    def test_missing_fields_become_empty_slots(self):
        """缺字段留空位而不是省略——省略会让两条不同的记录并成一个键。"""
        self.assertEqual(settlement.record_key({'date': '2026-08-28', 'home': 'A'}),
                         '2026-08-28||A|')

    def test_non_string_fields_are_stringified(self):
        self.assertEqual(settlement.record_key(
            {'date': '2026-08-28', 'num': 3, 'home': 'A', 'away': 'B'}),
            '2026-08-28|3|A|B')


class StorageRoundTripTests(unittest.TestCase):
    """**必须真的存一次再读一次**（判据 26）。

    `kv_store` 底层走 JSON，而 JSON 的对象键只能是字符串——`40398d1` 那次
    进球数校准空转几个月就是因为存进去的 `{2: 1.17}` 读回来是 `{"2": 1.17}`。
    北单历史这边**没有 int 键**（2026-08-28 实读 500 条确认：`zjq` 的概率键
    本来就是 `'0'`~`'6'` 与 `'7+'` 的字符串），但「没有」这件事得由一条
    走真实加载路径的用例守着，否则下次有人往里塞一个 int 键就没人拦。

    直接给属性赋值会绕过防腐层，修没修都测不出来，所以这里一律走
    `_load_beidan_history()`。
    """

    HISTORY = [
        _settled('2-1', probs={'胜': 0.40, '平': 0.30, '负': 0.30}),
        _settled('3-0', probs={'胜': 0.45, '平': 0.30, '负': 0.25}),
        _settled('1-0', probs={'胜': 0.38, '平': 0.32, '负': 0.30}),
        _settled('0-2', probs={'胜': 0.40, '平': 0.30, '负': 0.30}),
        _settled('1-1', probs={'胜': 0.35, '平': 0.35, '负': 0.30}),
        _settled('2-0', probs={'胜': 0.50, '平': 0.28, '负': 0.22}),
        _settled('4-1', probs={'胜': 0.55, '平': 0.25, '负': 0.20}),
        _settled('0-1', probs={'胜': 0.30, '平': 0.30, '负': 0.40}),
        _settled('2-2', probs={'胜': 0.40, '平': 0.30, '负': 0.30}),
        _settled('1-3', probs={'胜': 0.33, '平': 0.30, '负': 0.37}),
    ]

    @staticmethod
    def _reloaded(raw):
        """走真实加载路径：JSON 往返之后由适配层读进来。"""
        with mock.patch.object(kv_store, 'load',
                               return_value=json.loads(json.dumps(raw))):
            return _load_beidan_history()

    def test_round_trip_keeps_every_probability_key_a_string(self):
        for record in self._reloaded(self.HISTORY):
            with self.subTest(key=record.get('actual')):
                keys = record['spf']['probabilities'].keys()
                self.assertEqual({type(k).__name__ for k in keys}, {'str'})

    def test_calibration_is_identical_before_and_after_storage(self):
        """最强的那条断言：**存过一轮与没存过，结果必须一致**。"""
        probs = {'胜': 0.40, '平': 0.30, '负': 0.30}
        direct = settlement.apply_history_calibration(
            probs, self.HISTORY, settlement.actual_spf, 'spf')
        through_storage = settlement.apply_history_calibration(
            probs, self._reloaded(self.HISTORY), settlement.actual_spf, 'spf')
        self.assertEqual(direct, through_storage)

    def test_calibration_through_storage_actually_applies(self):
        """判据 27：「两边一致」在两边都失效时也成立，所以还要断言它真的生效了。"""
        with mock.patch.object(kv_store, 'load',
                               return_value=json.loads(json.dumps(self.HISTORY))):
            adjusted, meta = apply_beidan_history_calibration(
                {'胜': 0.40, '平': 0.30, '负': 0.30}, 'spf')
        self.assertTrue(meta['applied'])
        self.assertEqual(meta['sample_count'], 10.0)
        self.assertNotEqual(adjusted, {'胜': 0.40, '平': 0.30, '负': 0.30})

    def test_string_handicap_survives_storage_and_is_parsed(self):
        """盘口存的是文本，往返之后仍然是文本——解析必须发生在读进来之后。"""
        history = [_settled('1-0', handicap='(-1)', section='rqspf',
                            probs={'让胜': 0.30, '让平': 0.18, '让负': 0.52})
                   for _ in range(10)]
        reloaded = self._reloaded(history)
        self.assertEqual({type(r['handicap']).__name__ for r in reloaded}, {'str'})
        with mock.patch.object(kv_store, 'load', return_value=reloaded):
            _, meta = apply_beidan_history_calibration(
                {'让胜': 0.30, '让平': 0.18, '让负': 0.52}, 'rqspf')
        self.assertTrue(meta['applied'])
        # -1 的盘口 + 1-0 的比分 = 走水。**迁移前这里会记成「让胜」**
        self.assertEqual(meta['actuals']['让平'], 10.0)
        self.assertEqual(meta['actuals']['让胜'], 0.0)

    def test_empty_probabilities_never_touches_storage(self):
        """空概率时那道守卫排在 kv 读取之前——顺序也是行为。"""
        with mock.patch.object(kv_store, 'load') as load:
            _, meta = apply_beidan_history_calibration({}, 'spf')
        self.assertEqual(meta['reason'], 'empty_probabilities')
        load.assert_not_called()

    def test_adapter_limit_matches_the_shipped_value(self):
        """适配层自己有一份默认 limit——领域层那份守不住它（变异验证里
        只改适配层这一处，领域层的用例一条都不会红）。"""
        history = [_settled('2-1') for _ in range(DEFAULT_LIMIT + 5)]
        with mock.patch.object(kv_store, 'load',
                               return_value=json.loads(json.dumps(history))):
            _, meta = apply_beidan_history_calibration(
                {'胜': 0.40, '平': 0.30, '负': 0.30}, 'spf')
        self.assertEqual(meta['sample_count'], float(DEFAULT_LIMIT))

    def test_unknown_bet_type_collects_nothing(self):
        with mock.patch.object(kv_store, 'load',
                               return_value=json.loads(json.dumps(self.HISTORY))):
            _, meta = apply_beidan_history_calibration(
                {'胜': 0.40, '平': 0.30, '负': 0.30}, 'dxq')
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['sample_count'], 0.0)

    def test_load_tolerates_a_non_list_payload(self):
        with mock.patch.object(kv_store, 'load', return_value={'a': 1}):
            self.assertEqual(_load_beidan_history(), [])


class SaveHistoryTests(unittest.TestCase):

    HISTORY_LIMIT = 500

    def test_newest_first(self):
        records = [{'created_at': f'2026-08-{day:02d}T00:00:00', 'n': day}
                   for day in range(1, 21)]
        with mock.patch.object(kv_store, 'save') as save:
            _save_beidan_history(records)
        stored = save.call_args[0][1]
        self.assertEqual(stored[0]['n'], 20)
        self.assertEqual(stored[-1]['n'], 1)

    def test_truncated_to_the_limit(self):
        """**上一版这条用例只喂了 20 条**，名字里带 truncated 却从没到过
        500——把截断整句删掉照样全绿（判据 7）。要越过界才测得到界。"""
        records = [{'created_at': f'2026-{1 + n // 300:02d}-01T00:00:{n % 60:02d}',
                    'n': n} for n in range(self.HISTORY_LIMIT + 5)]
        with mock.patch.object(kv_store, 'save') as save:
            _save_beidan_history(records)
        stored = save.call_args[0][1]
        self.assertEqual(len(stored), self.HISTORY_LIMIT)
        # 截掉的是最旧的那 5 条，不是最新的
        self.assertNotIn(0, [r['n'] for r in stored])
        self.assertIn(self.HISTORY_LIMIT + 4, [r['n'] for r in stored])

    def test_records_without_a_timestamp_sort_last(self):
        records = [{'n': 'no_ts'}, {'created_at': '2026-08-01T00:00:00', 'n': 'ts'}]
        with mock.patch.object(kv_store, 'save') as save:
            _save_beidan_history(records)
        stored = save.call_args[0][1]
        self.assertEqual([r['n'] for r in stored], ['ts', 'no_ts'])


if __name__ == '__main__':
    unittest.main()
