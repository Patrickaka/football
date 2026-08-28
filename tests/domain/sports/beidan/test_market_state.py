"""北单的联合市场状态：走势合成与指数倾斜。

参照物是从迁移前的 `markets.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_market_state.json.gz`，259 条），**逐条相同**。

语料覆盖倾斜的**三条出口**：正常收敛（32 次）、目标落在支撑集之外（4 次）、
两侧价格缺失（16 次）。只喂「正常」那条的话，把二分搜索改坏都测不出来。

这一层是整条链里唯一让不同玩法互相自洽的地方——胜平负、让球、比分、
总进球共用同一张被约束过的比分矩阵。各算各的话，同一场比赛会给出互相
矛盾的推荐：让球盘说主队稳，比分推荐却全是平局。
"""
import gzip
import json
import math
import pathlib
import unittest

from src.domain.sports.beidan import handicap as handicap_mod
from src.domain.sports.beidan import market_state, scoring_model
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_market_state.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时生效的那组权重，写死不 import（判据 12）
STRENGTH_DIVISOR, STRENGTH_FLOOR = 0.12, 0.25
DIRECTION_BLEND, TEMPO_BLEND = 0.65, 0.55
CONFLICT_THRESHOLD, CONFLICT_DAMPING = -0.12, 0.40
CONSTRAINT_STRENGTH, CONSTRAINT_PASSES = 0.35, 3


def golden_entries():
    from scripts.gen_beidan_market_state_golden import entries
    return entries()


def _history(handicaps, home_odds, away_odds):
    return [{'handicap': h, 'home_odds': ho, 'away_odds': ao}
            for h, ho, ao in zip(handicaps, home_odds, away_odds)]


def _ou_history(lines, over_odds, under_odds):
    return [{'line': l, 'over_odds': o, 'under_odds': u}
            for l, o, u in zip(lines, over_odds, under_odds)]


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class JointStateTests(unittest.TestCase):

    BACKING = {'direction': 'home_backing', 'strength': 0.15}
    QUIET = {'direction': 'stable', 'strength': 0.0}
    OVER = {'direction': 'over_backing', 'strength': 0.15}
    UNDER = {'direction': 'under_backing', 'strength': 0.15}

    def _state(self, asian_trend, goals_trend, asian=None, goals=None, **kwargs):
        return market_state.joint_state(asian_trend, goals_trend,
                                        asian or [], goals or [], **kwargs)

    def test_home_backing_gives_a_positive_direction(self):
        state = self._state(self.BACKING, self.QUIET)
        self.assertGreater(state['direction_signal'], 0)

    def test_away_side_weighs_less_than_home(self):
        """**主队方向给满、客队方向打折**——北单以主队为基准报价，
        客队那侧的水位变化常常只是跟随。"""
        home = self._state({'direction': 'home_backing', 'strength': 0.2}, self.QUIET)
        away = self._state({'direction': 'away_laying', 'strength': 0.2}, self.QUIET)
        self.assertGreater(abs(home['direction_signal']), abs(away['direction_signal']))

    def test_strength_scales_the_direction(self):
        weak = self._state({'direction': 'home_backing', 'strength': 0.03}, self.QUIET)
        strong = self._state({'direction': 'home_backing', 'strength': 0.30}, self.QUIET)
        self.assertGreater(strong['direction_signal'], weak['direction_signal'])

    def test_strength_has_a_floor(self):
        """强度再小也按 0.25 算——**方向已经定了，只是幅度小**，
        乘成 0 等于把这条信息丢掉。"""
        tiny = self._state({'direction': 'home_backing', 'strength': 0.0001}, self.QUIET)
        floored = self._state({'direction': 'home_backing',
                               'strength': STRENGTH_FLOOR * STRENGTH_DIVISOR}, self.QUIET)
        self.assertAlmostEqual(tiny['direction_signal'], floored['direction_signal'],
                               places=9)

    def test_strength_normalisation_is_exact(self):
        """强度按 `strength_divisor` 归一化，**满格恰好在除数那一点**。

        只断言「强的比弱的大」是不够的——把除数改成 0.5 后大小关系依然成立，
        变异照样逃掉。这里钉住具体的比例：0.06 是半格、0.12 及以上是满格。
        """
        half = self._state({'direction': 'home_backing',
                            'strength': STRENGTH_DIVISOR / 2}, self.QUIET)
        full = self._state({'direction': 'home_backing',
                            'strength': STRENGTH_DIVISOR}, self.QUIET)
        beyond = self._state({'direction': 'home_backing', 'strength': 9.9}, self.QUIET)
        self.assertAlmostEqual(full['direction_signal'], DIRECTION_BLEND, places=9)
        self.assertAlmostEqual(half['direction_signal'], DIRECTION_BLEND / 2, places=9)
        self.assertAlmostEqual(beyond['direction_signal'], DIRECTION_BLEND, places=9)

    def test_two_sources_carry_their_declared_weights(self):
        """水位与线各占 0.65 / 0.35。**只断言「线也有影响」是不够的**——
        把两个权重对调，方向依然会动，变异逃得掉。"""
        water_only = self._state({'direction': 'home_backing',
                                  'strength': STRENGTH_DIVISOR}, self.QUIET)
        line_only = self._state(
            self.QUIET, self.QUIET,
            asian=_history(['0', '-0.5'], [0.95] * 2, [0.95] * 2))
        self.assertAlmostEqual(water_only['direction_signal'], DIRECTION_BLEND, places=9)
        self.assertAlmostEqual(line_only['direction_signal'],
                               -(1 - DIRECTION_BLEND), places=9)

    def test_handicap_move_contributes_independently(self):
        """盘口线本身的移动是第二个来源，与水位方向分开加权。"""
        no_move = self._state(self.QUIET, self.QUIET,
                              asian=_history(['-0.5'] * 3, [0.95] * 3, [0.95] * 3))
        moved = self._state(self.QUIET, self.QUIET,
                            asian=_history(['-0.5', '-0.75', '-1.0'],
                                           [0.95] * 3, [0.95] * 3))
        self.assertEqual(no_move['handicap_signal'], 0.0)
        self.assertLess(moved['handicap_signal'], 0.0)
        self.assertLess(moved['direction_signal'], no_move['direction_signal'])

    def test_signals_use_first_and_last_not_the_average(self):
        """**用首尾而不是逐期平均**：盘口线是阶梯式跳变的，
        中间的往返在这里没有意义。"""
        straight = _history(['-0.5', '-0.75'], [0.95] * 2, [0.95] * 2)
        detoured = _history(['-0.5', '-0.25', '-1.0', '-0.75'],
                            [0.95] * 4, [0.95] * 4)
        self.assertAlmostEqual(self._state(self.QUIET, self.QUIET, asian=straight)
                               ['handicap_signal'],
                               self._state(self.QUIET, self.QUIET, asian=detoured)
                               ['handicap_signal'], places=9)

    def test_conflict_dampens_the_tempo(self):
        """水位压大球、线却往下调 → 两个信号打架，节奏衰减。
        **不是取反也不是忽略**——而是承认「这场看不清」。"""
        aligned = self._state(self.QUIET, self.OVER,
                              goals=_ou_history([2.5, 2.75], [0.95] * 2, [0.95] * 2))
        conflicting = self._state(self.QUIET, self.OVER,
                                  goals=_ou_history([2.75, 2.5], [0.95] * 2, [0.95] * 2))
        self.assertFalse(aligned['conflict'])
        self.assertTrue(conflicting['conflict'])
        self.assertLess(abs(conflicting['tempo_signal']), abs(aligned['tempo_signal']))
        self.assertEqual(conflicting['agreement_factor'], CONFLICT_DAMPING)

    def test_no_conflict_keeps_full_agreement(self):
        state = self._state(self.QUIET, self.OVER)
        self.assertFalse(state['conflict'])
        self.assertEqual(state['agreement_factor'], 1.0)

    def test_signals_are_clamped_to_unit_range(self):
        extreme = self._state({'direction': 'home_backing', 'strength': 99.0},
                              {'direction': 'over_backing', 'strength': 99.0},
                              asian=_history(['0', '-5.0'], [0.95] * 2, [0.95] * 2),
                              goals=_ou_history([2.5, 9.0], [0.95] * 2, [0.95] * 2))
        self.assertLessEqual(abs(extreme['direction_signal']), 1.0)
        self.assertLessEqual(abs(extreme['tempo_signal']), 1.0)

    def test_unparsable_line_yields_no_signal(self):
        state = self._state(self.QUIET, self.QUIET,
                            goals=_ou_history(['abc', 'xyz'], [0.95] * 2, [0.95] * 2))
        self.assertEqual(state['line_signal'], 0.0)

    def test_single_entry_history_yields_no_move_signal(self):
        state = self._state(self.QUIET, self.QUIET,
                            asian=_history(['-0.5'], [0.95], [0.95]))
        self.assertEqual(state['handicap_signal'], 0.0)


class TiltTests(unittest.TestCase):
    """指数倾斜的三条出口。"""

    def setUp(self):
        self.matrix = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, 7)

    def _over_profit(self, line, odds):
        return lambda score: handicap_mod.over_profit(score[0] + score[1], line, odds)

    def test_full_strength_drives_the_expectation_to_zero(self):
        """强度为 1 时期望收益应当收敛到 0——**这是「公平价」的定义**。"""
        _, meta = market_state.tilt_to_fair_price(
            self.matrix, self._over_profit(2.5, 1.95), 1.0)
        self.assertTrue(meta['applied'])
        self.assertAlmostEqual(meta['fair_profit_after'], 0.0, places=4)

    def test_partial_strength_moves_only_part_way(self):
        """**完全对齐等于把模型丢掉，不对齐等于无视已成交的价格。**"""
        _, full = market_state.tilt_to_fair_price(
            self.matrix, self._over_profit(2.5, 1.95), 1.0)
        _, partial = market_state.tilt_to_fair_price(
            self.matrix, self._over_profit(2.5, 1.95), 0.35)
        self.assertLess(abs(partial['theta']), abs(full['theta']))
        self.assertGreater(abs(partial['fair_profit_after']),
                           abs(full['fair_profit_after']))

    def test_zero_strength_leaves_the_distribution_alone(self):
        adjusted, meta = market_state.tilt_to_fair_price(
            self.matrix, self._over_profit(2.5, 1.95), 0.0)
        self.assertEqual(meta['theta'], 0.0)
        for key, value in self.matrix.items():
            self.assertAlmostEqual(adjusted[key], value, places=9)

    def test_output_stays_normalised(self):
        adjusted, _ = market_state.tilt_to_fair_price(
            self.matrix, self._over_profit(2.5, 1.95), 1.0)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=9)

    def test_target_outside_support_is_reported_not_forced(self):
        """所有结果同号时任何 θ 都到不了 0——**强行搜索会把分布推到边界上**。

        这里给一个必然赢的盘口（0 球线、全是正收益），倾斜应当放弃。
        """
        only_wins = {(2, 1): 0.5, (3, 1): 0.3, (2, 2): 0.2}
        adjusted, meta = market_state.tilt_to_fair_price(
            only_wins, self._over_profit(0.5, 1.95), 1.0)
        self.assertFalse(meta['applied'])
        self.assertEqual(meta['reason'], 'target_outside_support')
        self.assertEqual(adjusted, only_wins, '放弃时原样返回')

    def test_tilt_direction_follows_the_price(self):
        """便宜的大球赔率 → 把质量推向高比分。方向反了不会报错，
        只会让所有玩法一起偏。"""
        before = sum(sum(key) * p for key, p in self.matrix.items())
        cheap_over, _ = market_state.tilt_to_fair_price(
            self.matrix, self._over_profit(2.5, 1.40), 1.0)
        after = sum(sum(key) * p for key, p in cheap_over.items())
        self.assertGreater(after, before)

    def test_extreme_theta_does_not_overflow(self):
        """指数要夹紧——**θ·特征值 在极端比分上能到几百**，不夹会溢出。"""
        wide = scoring_model.dixon_coles_matrix(3.5, 0.2, 0.0, 7)
        adjusted, meta = market_state.tilt_to_fair_price(
            wide, self._over_profit(2.5, 1.01), 1.0)
        self.assertTrue(all(math.isfinite(value) for value in adjusted.values()))
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=9)


class ExponentClampTests(unittest.TestCase):
    """指数必须夹紧——**θ·特征值 在极端赔率下能到上万**，不夹会 OverflowError。"""

    # **分布必须不对称**：两个 ±5000 各占一半时期望本来就是 0，
    # θ 会停在 0，根本触发不了夹紧——那样这两条用例测的是「不用调整」。
    SKEWED = {(0, 0): 0.7, (3, 3): 0.3}

    @staticmethod
    def _huge(key):
        return 5000.0 if sum(key) > 3 else -5000.0

    def test_huge_feature_values_do_not_overflow(self):
        """公平赔率可以非常大（一侧报价极低时），此时单位收益也随之变大。
        θ 上限 12，两者相乘轻易越过 `exp` 的定义域。
        """
        adjusted, meta = market_state.tilt_to_fair_price(
            self.SKEWED, self._huge, 1.0)
        self.assertTrue(meta['applied'])
        self.assertNotAlmostEqual(meta['theta'], 0.0, places=6, msg='应当真的需要调整')
        self.assertTrue(all(math.isfinite(value) for value in adjusted.values()))
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=9)

    def test_clamped_exponent_still_shifts_the_mass(self):
        """夹紧不能把调整抹平——**那样约束就等于没做**。

        原分布七三开偏向 0-0，而 0-0 那侧是负收益，所以约束会把质量
        推向 3-3。夹紧只该限制幅度，不该改变方向。
        """
        adjusted, _ = market_state.tilt_to_fair_price(
            self.SKEWED, self._huge, 1.0)
        self.assertGreater(adjusted[(3, 3)], self.SKEWED[(3, 3)])
        self.assertLess(adjusted[(0, 0)], self.SKEWED[(0, 0)])


class AdapterPathTests(unittest.TestCase):
    """适配层里那条「有报价但没有盘口线」的分支。"""

    def test_missing_handicap_is_treated_as_a_pk_market(self):
        """少数备用源只给两侧报价而没有盘口线。**按平手盘处理而不是丢弃**——
        那是这场唯一的方向性价格证据。兜底值必须是 0（平手），
        用别的数会凭空造出一个让球盘。
        """
        import importlib
        markets = importlib.import_module('src.beidan.markets')
        matrix = scoring_model.dixon_coles_matrix(1.4, 1.1, 0.0, 7)
        no_line = {'history': [{'home_odds': 0.95, 'away_odds': 0.95}] * 3}
        with_pk = {'history': [{'handicap': '0',
                                'home_odds': 0.95, 'away_odds': 0.95}] * 3}
        goals = {'history': [{'line': 2.5, 'over_odds': 0.95, 'under_odds': 0.95}] * 3}

        missing_result, missing_meta = markets.apply_beidan_joint_market_state(
            matrix, no_line, goals)
        pk_result, pk_meta = markets.apply_beidan_joint_market_state(
            matrix, with_pk, goals)
        self.assertTrue(missing_meta['asian_constraint']['applied'])
        self.assertEqual(missing_meta['asian_constraint']['theta'],
                         pk_meta['asian_constraint']['theta'],
                         '缺盘口线应当与平手盘算出同一个 θ')
        for key in pk_result:
            self.assertAlmostEqual(missing_result[key], pk_result[key], places=9)


class FairOddsTests(unittest.TestCase):

    def test_removes_the_margin(self):
        """两侧都 1.90 的盘口，公平赔率是 2.0。"""
        self.assertAlmostEqual(market_state.fair_odds(1.90, 1.90), 2.0, places=9)

    def test_favours_the_cheaper_side(self):
        self.assertLess(market_state.fair_odds(1.50, 2.50),
                        market_state.fair_odds(2.50, 1.50))

    def test_missing_side_returns_none(self):
        self.assertIsNone(market_state.fair_odds(None, 1.9))
        self.assertIsNone(market_state.fair_odds(1.9, 0))


class NormaliseMatrixTests(unittest.TestCase):

    def test_merges_duplicate_keys(self):
        merged = market_state.normalise_matrix({(1, 0): 0.3, '10': 0.2, (0, 1): 0.5})
        self.assertAlmostEqual(merged[(1, 0)], 0.5)

    def test_negative_probabilities_are_floored(self):
        result = market_state.normalise_matrix({(1, 0): 0.8, (0, 1): -0.2})
        self.assertAlmostEqual(result[(0, 1)], 0.0)

    def test_unparsable_keys_are_skipped_not_fatal(self):
        """脏值来自上游模型输出，**个别坏键不该让整场预测失败**。"""
        result = market_state.normalise_matrix({'abc': 0.5, (1, 0): 0.5})
        self.assertEqual(set(result), {(1, 0)})

    def test_all_zero_mass_returns_none(self):
        self.assertIsNone(market_state.normalise_matrix({(1, 0): 0.0}))
        self.assertIsNone(market_state.normalise_matrix({}))

    def test_output_is_normalised(self):
        result = market_state.normalise_matrix({(1, 0): 3.0, (0, 1): 1.0})
        self.assertAlmostEqual(sum(result.values()), 1.0, places=9)


class SummariseShiftTests(unittest.TestCase):

    def test_reports_both_observable_quantities(self):
        """**只说「已应用」看不出幅度**，而这两个量正是各玩法真正依赖的。"""
        before = {(1, 0): 0.5, (0, 1): 0.5}
        after = {(1, 0): 0.8, (0, 1): 0.2}
        shift = market_state.summarise_shift(before, after)
        self.assertAlmostEqual(shift['home_win_before'], 0.5)
        self.assertAlmostEqual(shift['home_win_after'], 0.8)
        self.assertAlmostEqual(shift['expected_goals_before'], 1.0)
        self.assertAlmostEqual(shift['expected_goals_after'], 1.0)


if __name__ == '__main__':
    unittest.main()
