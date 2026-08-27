"""北单的概率建模层：泊松、Dixon-Coles、λ 推导、盘口解析、比分锚定。

参照物是从迁移前的 `src/beidan/modeling.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_modeling.json.gz`，273 条），**逐条相同**。
语料按三档赔率 × 四个盘口 × 三条大小球线 × 四个联赛档案铺开，
另含三套 1X2 别名与两条空输入边界。

**有一处签名是有意改的**：`match_lambdas` 删掉了 `split` 参数。它在迁移前的
函数体里从没出现过——那个函数只是转手调 `euro_implied_lambdas`，而后者用的是
模块级的 `SCORE_SPLIT`。三个不同的 split 值算出完全一样的结果（旧黄金里的
18 条 `:0.35`/`:0.55` 与对应的 `:0.45` 逐条相同，那就是证据），而三处调用方
一个都没传过它。领域层的 `lambdas_from_probs` 让这个参数真正生效。
"""
import gzip
import importlib
import json
import pathlib
import unittest

from src.domain.sports.beidan import handicap, scoring_model, totals
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_modeling.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时 config.py / modeling.py 里生效的那组值，写死不 import（判据 12）
MAX_GOALS = 7
SCORE_SPLIT = 0.45
DC_RHO = 0.0
OU_TOTAL_BLEND = 0.6
ANCHOR_STRENGTH = 0.75
LEAGUE_AVG_GOALS = {'英超': 2.8, '德甲': 3.1, '意甲': 2.5, '西甲': 2.7}


def golden_entries():
    """与 `scripts/gen_beidan_modeling_golden.py` 共用同一份语料定义。"""
    from scripts.gen_beidan_modeling_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class PoissonTests(unittest.TestCase):

    def test_probabilities_sum_to_one(self):
        """把 0..30 球加起来应当接近 1——**分布不归一的话，
        下游每个玩法的概率都会同比例偏，而且看起来都很正常。**"""
        total = sum(scoring_model.poisson_pmf(k, 1.4) for k in range(31))
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_zero_mean_means_certain_zero(self):
        """均值为 0 时只有 0 球可能。这是退化情形，不是错误——
        λ 被夹到下限之前可能出现。"""
        self.assertEqual(scoring_model.poisson_pmf(0, 0), 1.0)
        self.assertEqual(scoring_model.poisson_pmf(3, 0), 0.0)

    def test_mode_is_near_the_mean(self):
        """λ 取 1.5 而不是整数：整数 λ 时 P(λ-1) 与 P(λ) 精确相等，
        `index(max)` 只会返回前一个，那样这条断言测的是「取第一个」而不是众数。"""
        probs = [scoring_model.poisson_pmf(k, 1.5) for k in range(8)]
        self.assertEqual(probs.index(max(probs)), 1)


class LambdaSplitTests(unittest.TestCase):
    """`split` 从死参数变成活参数——这一批唯一有意的行为扩展。"""

    PROBS = (0.62, 0.23, 0.15)   # 强主

    def test_split_zero_gives_both_sides_the_same_lambda(self):
        """split=0 表示不看强弱，总进球平分。"""
        home, away = scoring_model.lambdas_from_probs(*self.PROBS, 2.8, 0.0)
        self.assertAlmostEqual(home, away, places=9)

    def test_larger_split_widens_the_gap(self):
        """**split 越大，强队分到的越多。** 迁移前这个参数是死的，
        三个不同的值算出同一个结果；这条用例正是为那个死参数而设。"""
        narrow = scoring_model.lambdas_from_probs(*self.PROBS, 2.8, 0.2)
        wide = scoring_model.lambdas_from_probs(*self.PROBS, 2.8, 0.6)
        self.assertGreater(wide[0] - wide[1], narrow[0] - narrow[1])

    def test_default_split_matches_the_migrated_value(self):
        """适配层传的仍是迁移前那个常量，所以线上行为不变。"""
        adapter = importlib.import_module('src.beidan.modeling')
        self.assertEqual(adapter.match_lambdas(*self.PROBS, 2.8),
                         scoring_model.lambdas_from_probs(*self.PROBS, 2.8, SCORE_SPLIT))

    def test_adapter_no_longer_accepts_split(self):
        """删掉之后再传就是 TypeError——那正是要的效果（§五·1）。"""
        adapter = importlib.import_module('src.beidan.modeling')
        with self.assertRaises(TypeError):
            adapter.match_lambdas(*self.PROBS, 2.8, split=0.6)

    def test_total_is_preserved(self):
        """怎么分都不该改变总量。"""
        for split in (0.0, 0.45, 0.9):
            home, away = scoring_model.lambdas_from_probs(*self.PROBS, 2.8, split)
            self.assertAlmostEqual(home + away, 2.8, places=9)


class DixonColesTests(unittest.TestCase):
    """线上 `DC_RHO = 0.0`，修正当前不生效——**但配置随时会改回来**，
    所以这几条覆盖 rho ≠ 0（判据 9 第二类：配置让它不可达 → 补用例，不删）。"""

    def test_rho_zero_is_proportional_to_independent_poisson(self):
        """rho=0 时每格与裸泊松乘积成**同一个比例**。

        不是逐格相等——矩阵截断在 7 球，归一化会把每格按 1/0.99987 放大。
        断言比例一致比断言相等更严：任何一格被单独改动都会破坏它。"""
        matrix = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)
        ratios = {round(prob / (scoring_model.poisson_pmf(home, 1.4)
                                * scoring_model.poisson_pmf(away, 1.1)), 9)
                  for (home, away), prob in matrix.items()}
        self.assertEqual(len(ratios), 1, '所有格子必须共用同一个归一化系数')

    def test_rho_only_touches_the_four_low_scores(self):
        """DC 只修正 0-0、0-1、1-0、1-1 四格。**改动范围扩大不会报错**，
        只会让别的比分悄悄偏掉。"""
        plain = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)
        tweaked = scoring_model.dixon_coles_matrix(1.4, 1.1, -0.05, MAX_GOALS)
        changed = {key for key in plain
                   if abs(plain[key] - tweaked[key]) > 1e-9}
        self.assertEqual(changed, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_matrix_is_normalised_for_every_rho(self):
        for rho in (-0.1, 0.0, 0.08):
            matrix = scoring_model.dixon_coles_matrix(1.4, 1.1, rho, MAX_GOALS)
            self.assertAlmostEqual(sum(matrix.values()), 1.0, places=9)

    def test_negative_rho_lifts_the_draw_cells(self):
        """负 rho 抬高 0-0 与 1-1。方向反了不会报错，只会让平局推荐反向。"""
        plain = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)
        tweaked = scoring_model.dixon_coles_matrix(1.4, 1.1, -0.05, MAX_GOALS)
        self.assertGreater(tweaked[(0, 0)], plain[(0, 0)])
        self.assertGreater(tweaked[(1, 1)], plain[(1, 1)])

    def test_independent_matrix_drops_the_tail(self):
        """不带 DC 的那条路会丢掉低于阈值的格子——**与 rho=0 不等价**，
        两者不能互相替代。"""
        full = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)
        trimmed = scoring_model.independent_poisson_matrix(1.4, 1.1, MAX_GOALS)
        self.assertEqual(len(full), (MAX_GOALS + 1) ** 2, 'DC 保留全部格子')
        self.assertEqual(len(trimmed), 61, '截断版丢掉了尾部 3 格')
        # places 取 6：归一化的分母加了 1e-9 防除零，差在 1e-10 量级
        self.assertAlmostEqual(sum(trimmed.values()), 1.0, places=6)


class GoalsAggregationTests(unittest.TestCase):

    def test_keys_are_strings_including_the_overflow_bucket(self):
        """**键必须统一是字符串**：`'7+'` 本来就不是数字，混着放 int 和 str，
        下游一 `sorted()` 就炸——进球数校准那条链正是这么空转了几个月。"""
        matrix = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)
        buckets = scoring_model.aggregate_goals(matrix)
        self.assertEqual(sorted({type(k).__name__ for k in buckets}), ['str'])
        self.assertIn('7+', buckets)

    def test_mass_is_preserved(self):
        matrix = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)
        self.assertAlmostEqual(sum(scoring_model.aggregate_goals(matrix).values()),
                               sum(matrix.values()), places=9)

    def test_overflow_collects_everything_at_or_above_the_cap(self):
        dist = {(4, 3): 0.5, (5, 2): 0.3, (1, 1): 0.2}
        buckets = scoring_model.aggregate_goals(dist)
        self.assertAlmostEqual(buckets['7+'], 0.8)
        self.assertAlmostEqual(buckets['2'], 0.2)


class HandicapTests(unittest.TestCase):

    def test_parse_returns_none_not_zero_when_absent(self):
        """**没有盘口 ≠ 平手盘。** 返回 0.0 会把两种情况混成一个值。"""
        self.assertIsNone(handicap.parse(None))
        self.assertIsNone(handicap.parse(''))
        self.assertIsNone(handicap.parse('无'))
        self.assertEqual(handicap.parse('0'), 0.0)

    def test_parse_handles_full_width_brackets(self):
        """页面复制来的盘口带全角括号。"""
        self.assertEqual(handicap.parse('（-1）'), -1.0)
        self.assertEqual(handicap.parse('(-1)'), -1.0)

    def test_split_lines_settle_on_two_lines(self):
        self.assertEqual(handicap.line_parts(2.25), (2.0, 2.5))
        self.assertEqual(handicap.line_parts(2.75), (2.5, 3.0))

    def test_whole_and_half_lines_settle_on_one(self):
        self.assertEqual(handicap.line_parts(2.0), (2.0,))
        self.assertEqual(handicap.line_parts(2.5), (2.5,))

    def test_split_line_can_half_win(self):
        """分盘 2.75 开 3 球：一条线赢、一条走水，**平均收益是赢的一半**。"""
        profit = handicap.over_profit(3, 2.75, 2.0)
        self.assertAlmostEqual(profit, 0.5)

    def test_push_returns_stake_not_profit(self):
        """走水记 0（退本金），不是记赢。"""
        self.assertEqual(handicap.over_profit(3, 3.0, 2.0), 0.0)

    def test_rqspf_needs_a_handicap(self):
        probs, meta = handicap.rqspf_from_scores({(1, 0): 1.0}, None)
        self.assertEqual(probs, {})
        self.assertFalse(meta['available'])

    def test_rqspf_applies_the_handicap_sign(self):
        """让一球时 1-0 变成让平——**盘口是带符号的，这里是加不是减**。"""
        probs, meta = handicap.rqspf_from_scores({(1, 0): 1.0}, '-1')
        self.assertAlmostEqual(probs['让平'], 1.0)
        self.assertTrue(meta['available'])

    def test_rqspf_probabilities_sum_to_one(self):
        matrix = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)
        probs, _ = handicap.rqspf_from_scores(matrix, '-0.5')
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)


class TotalsTests(unittest.TestCase):

    def test_water_is_converted_but_euro_odds_are_not(self):
        """**判错的代价是概率差一倍**：把贴水 0.85 当欧赔，1/0.85 > 1。"""
        self.assertAlmostEqual(totals.to_euro_odds(0.85), 1.85)
        self.assertAlmostEqual(totals.to_euro_odds(1.95), 1.95)

    def test_boundary_between_water_and_euro_odds(self):
        """1.2 这条线的两侧各断言一次。"""
        self.assertAlmostEqual(totals.to_euro_odds(1.2), 2.2)
        self.assertAlmostEqual(totals.to_euro_odds(1.21), 1.21)

    def test_invalid_odds_return_none(self):
        for bad in (None, 0, -1, 'abc'):
            self.assertIsNone(totals.to_euro_odds(bad))

    def test_split_line_text_takes_the_midpoint(self):
        self.assertAlmostEqual(totals.parse_line_value('2.5/3'), 2.75)

    def test_implied_total_rises_with_a_cheaper_over(self):
        """大球越便宜，隐含总进球越高。方向反了整条链都跟着反。"""
        lean_over = totals.implied_total(0.85, 1.05, 2.5)
        lean_under = totals.implied_total(1.08, 0.82, 2.5)
        self.assertGreater(lean_over, lean_under)

    def test_implied_total_needs_both_sides(self):
        self.assertIsNone(totals.implied_total(None, 0.95, 2.5))
        self.assertIsNone(totals.implied_total(0.95, None, 2.5))

    def test_target_total_falls_back_to_league_average(self):
        """没有盘口就用联赛均值——**不是用一个写死的 2.5**。"""
        self.assertAlmostEqual(totals.target_total(3.1), 3.1)

    def test_target_total_blends_toward_the_league_average(self):
        """联赛均值占大头：结果应落在盘口隐含值与均值之间，且更靠近哪边
        由 blend 决定。**盘口主导的话，一条噪声报价就能带偏整场。**"""
        implied = totals.implied_total(0.85, 1.05, 2.5)
        blended = totals.target_total(2.5, 0.85, 1.05, line=2.5, blend=OU_TOTAL_BLEND)
        self.assertGreater(blended, min(implied, 2.5))
        self.assertLess(blended, max(implied, 2.5))

    def test_hard_bounds_survive_extreme_factors(self):
        """两个因子都到上界是 1.32 倍——**只靠软约束仍会推出 4 球以上**，
        所以硬约束那道必须在。"""
        capped = totals.target_total(3.6, asian_factor=9.9, goals_factor=9.9,
                                     total_high=3.6)
        self.assertLessEqual(capped, 3.6)

    def test_unparsable_factors_fall_back_to_neutral(self):
        self.assertAlmostEqual(totals.target_total(2.8, asian_factor='x',
                                                  goals_factor=None), 2.8)


class AnchorTests(unittest.TestCase):

    MATRIX = None

    def setUp(self):
        self.MATRIX = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, MAX_GOALS)

    def test_three_alias_forms_are_equivalent(self):
        """胜/平/负、H/D/A、home/draw/away 三套都得认——**别名解析写错了
        不会报错，只会让锚定悄悄退化成 `applied: False`**。"""
        probs = (0.45, 0.28, 0.27)
        forms = [{'胜': probs[0], '平': probs[1], '负': probs[2]},
                 {'H': probs[0], 'D': probs[1], 'A': probs[2]},
                 {'home': probs[0], 'draw': probs[1], 'away': probs[2]}]
        results = [scoring_model.anchor_outcomes(self.MATRIX, form, 0.5)[0]
                   for form in forms]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_zero_strength_leaves_the_distribution_alone(self):
        adjusted, meta = scoring_model.anchor_outcomes(
            self.MATRIX, {'胜': 0.45, '平': 0.28, '负': 0.27}, 0.0)
        self.assertTrue(meta['applied'])
        for key, value in self.MATRIX.items():
            self.assertAlmostEqual(adjusted[key], value, places=9)

    def test_full_strength_hits_the_target(self):
        target = {'胜': 0.45, '平': 0.28, '负': 0.27}
        adjusted, meta = scoring_model.anchor_outcomes(self.MATRIX, target, 1.0)
        for label, expected in (('胜', 0.45), ('平', 0.28), ('负', 0.27)):
            self.assertAlmostEqual(meta['after'][label], expected, places=6)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=9)

    def test_partial_strength_lands_between(self):
        """0.5 应当落在「不动」与「完全对齐」之间——两端都测了才说明
        中间那一档真的在插值。"""
        target = {'胜': 0.60, '平': 0.20, '负': 0.20}
        before = scoring_model.anchor_outcomes(self.MATRIX, target, 0.0)[1]['before']
        half = scoring_model.anchor_outcomes(self.MATRIX, target, 0.5)[1]['after']
        self.assertGreater(half['胜'], before['胜'])
        self.assertLess(half['胜'], 0.60)

    def test_missing_target_is_reported_not_raised(self):
        """走不通就返回原分布加一个理由，**不抛也不悄悄返回半成品**。"""
        adjusted, meta = scoring_model.anchor_outcomes(self.MATRIX, {}, 0.5)
        self.assertEqual(adjusted, self.MATRIX)
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'missing_distribution_or_target')

    def test_incomplete_outcome_mass_is_reported(self):
        """只有主胜比分时，平和负的质量是 0，比例算不出来——这时候不对齐，
        而不是给一个除零后的巨大因子。"""
        adjusted, meta = scoring_model.anchor_outcomes(
            {(1, 0): 1.0}, {'胜': 0.4, '平': 0.3, '负': 0.3}, 0.5)
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'incomplete_outcome_mass')


class CalibrateDrawTests(unittest.TestCase):

    def test_output_is_normalised(self):
        """places 取 6 而不是 9：归一化的分母加了 1e-9 防除零，
        结果因此差在 1e-10 量级。那是有意的保护，不是误差。"""
        result = scoring_model.calibrate_draw(0.5, 0.2, 0.3, 0)
        self.assertAlmostEqual(sum(result), 1.0, places=6)

    def test_big_handicap_discounts_the_reference_draw_rate(self):
        """盘口越大平局越少——让一球以上时参照率打折，于是同样的输入
        不会被上调。**方向反了会让大盘口的平局推荐变多。**"""
        level = scoring_model.calibrate_draw(0.5, 0.18, 0.32, 0)
        big = scoring_model.calibrate_draw(0.5, 0.18, 0.32, -1.5)
        self.assertGreaterEqual(level[1], big[1])

    def test_unparsable_handicap_does_not_break_the_match(self):
        result = scoring_model.calibrate_draw(0.5, 0.2, 0.3, '看不懂')
        self.assertAlmostEqual(sum(result), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
