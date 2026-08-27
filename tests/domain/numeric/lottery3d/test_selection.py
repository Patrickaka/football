"""福彩 3D 的选号：组六、组三、胆拖杀、形态概率、策略准入。

这一层把数字评分变成一份能照着买的方案，**算错了不会报错，只会让人照着
一份错的方案掏钱**。参照物是从迁移前的实现生成的黄金文件
（`tests/fixtures/golden/lottery3d_selection.json.gz`，2544 条），语料按六种
长度的历史 × 四种评分形状（真实 / 全平 / 递减 / 前后并列）× 三组杀码铺开。

**黄金值里有两处是有意与迁移前不同的**：
1. 组三档位的 `pairs_str` 迁移前渲染成 `(2, 5)0.06338028169014084`——
   `for a, b in top` 把 `((a, b), 概率)` 拆错了。改对了。
2. 四码组六的 payload 迁移前只有 `hit_rate`、没有 `conditional_hit_rate`，
   而同族的主推池与覆盖档位两者都有。统一之后补上（纯增量）。

选出来的**号码**一个都没变。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.numeric.lottery3d import admission, selection
from src.domain.numeric.lottery3d.space import DIGIT_SPACE
from src.lottery3d import scoring as adapter
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_selection.json.gz')
NUMBERS = [tuple(r['digits'])
           for r in _load('numeric/lottery3d_history.json.gz')['results']]

SERIES = {'full': NUMBERS, 'recent200': NUMBERS[-200:], 'recent60': NUMBERS[-60:],
          'recent30': NUMBERS[-30:], 'recent5': NUMBERS[-5:], 'one': NUMBERS[-1:]}
KILLS = {'none': None, 'two': [3, 7], 'many': [0, 1, 2, 3, 4]}
SCORES = {'zu6': None, 'flat': [1.0] * 10,
          'descending': [10.0 - i for i in range(10)],
          'tied': [5.0] * 5 + [1.0] * 5}
# 迁移当时 config.py 里生效的值，写死而不是 import（判据 12）
KILL_PENALTY = 8.0
WINDOW_WEIGHTS = {'none': None, 'default': {30: 0.25, 45: 0.25, 60: 0.25, 90: 0.25},
                  'flat': {30: 1.0}}
ADMISSION_CASES = [
    (0.05, 0.04, 300, None, None),
    (0.10, 0.09, 100, 0.03, {'pvalue': 0.01}),
    (0.02, 0.02, 500, 0.03, {'pvalue': 0.5}),
    (0.045, 0.03, 250, 0.030, {'pvalue': 0.049}),
    (0.0, 0.0, 1000, 0.0, {'pvalue': 1.0}),
    (0.20, 0.20, 50, 0.03, {'pvalue': 0.0001}),
    (0.05, 0.04, 499, 0.03, {}),
    (0.03, 0.03, 500, None, None),
]


def _score_for(name, kind):
    return adapter.zu6_digit_scores(SERIES[name]) if kind == 'zu6' else SCORES[kind]


def golden_entries():
    """按 (键, 值) 逐条产出全部语料，测试与重生成脚本共用。

    走适配层：迁移要保证的是这些公开入口的输出，而「哪个配置常量喂给哪个
    参数」只在适配层发生。领域函数自身的语义由下面的手写用例覆盖。
    """
    yield from _zu3_entries()
    yield from _zu6_entries()
    yield from _dan_entries()
    yield from _form_entries()
    yield from _admission_entries()


def _zu3_entries():
    for name, series in SERIES.items():
        for window in (None, 30, 60, 200):
            yield (f'zu3_presence:{name}:{window}',
                   adapter.zu3_digit_presence(series, window))
        presence = adapter.zu3_digit_presence(series)
        yield f'zu3_pair_scores:{name}', adapter.zu3_pair_scores(presence)
        for limit in (1, 4, 10, 45, 60):
            yield (f'zu3_pairs:{name}:{limit}',
                   adapter.pick_zu3_pairs(series, limit, presence))
        for sizes in ((4,), (1, 4, 10), (45, 60)):
            yield (f'zu3_tiers:{name}:{"-".join(map(str, sizes))}',
                   adapter.zu3_coverage_tiers(series, sizes, presence))
    for pair in ((0, 1), (2, 5), (9, 0), (4, 4), (7, 3)):
        yield f'zu3_combos:{pair[0]}{pair[1]}', adapter.zu3_combos_from_pair(pair)
        yield f'zu3_zu_notes:{pair[0]}{pair[1]}', adapter.zu3_zu_notes_from_pair(pair)


def _zu6_entries():
    for name in SERIES:
        for kind in SCORES:
            score = _score_for(name, kind)
            tag = f'{name}:{kind}'
            for kname, kill in KILLS.items():
                for size in (3, 4, 5, 6, 7, 10):
                    yield (f'zu6_pool:{tag}:{kname}:{size}',
                           adapter.pick_zu6_pool(score, kill, pool_size=size))
                    yield (f'zu6_pool_kill:{tag}:{kname}:{size}',
                           adapter.pick_zu6_pool(score, kill, pool_size=size, use_kill=True))
                yield f'zu6_four:{tag}:{kname}', adapter.pick_zu6_four(score, kill)
                yield (f'zu6_primary:{tag}:{kname}',
                       adapter.build_zu6_primary(score, kill, numbers=SERIES[name]))
                for sizes in ((4, 5, 6, 7), (3,), (8, 9, 10)):
                    yield (f'zu6_tiers:{tag}:{kname}:{"-".join(map(str, sizes))}',
                           adapter.build_zu6_coverage_tiers(score, kill, sizes,
                                                            numbers=SERIES[name]))
                for limit in (1, 4, 8):
                    yield (f'zu6_variants:{tag}:{kname}:{limit}',
                           adapter.build_zu6_four_variants(score, kill, limit,
                                                           numbers=SERIES[name]))
                for digits in ((1, 2, 3, 4), (0, 5, 9), (2, 4, 6, 8, 0)):
                    yield (f'zu6_balance:{tag}:{kname}:{"".join(map(str, digits))}',
                           adapter._zu6_four_balance_score(digits, score, kill))
                for digit in range(DIGIT_SPACE.size):
                    yield (f'effective_score:{tag}:{kname}:{digit}',
                           adapter._effective_digit_score(score, digit, kill))
    for digits in ((1, 2, 3), (0, 1, 2, 3), (5, 6, 7, 8, 9), (0, 9)):
        yield (f'zu6_notes:{"".join(map(str, digits))}',
               adapter.zu6_notes_from_digits(digits))
    for label, digits in (('主推', (1, 2, 3, 4)), ('均衡', (0, 3, 6, 9))):
        yield f'zu6_payload:{label}', adapter._zu6_four_payload(label, digits)
    for name in ('full', 'recent200'):
        for sizes in ((5, 6), (4,), (3, 10)):
            yield (f'zu6_eval:{name}:{"-".join(map(str, sizes))}',
                   adapter.evaluate_zu6_pool_recent(SERIES[name], sizes, trials=20))


def _dan_entries():
    for name in SERIES:
        for kind in SCORES:
            yield (f'dan_tuo_kill:{name}:{kind}',
                   [list(part) if isinstance(part, list) else part
                    for part in adapter.pick_dan_tuo_kill(_score_for(name, kind),
                                                          enable_danma_random=False)])


def _form_entries():
    for name, series in SERIES.items():
        for wname, weights in WINDOW_WEIGHTS.items():
            probability = adapter.analyze_form_probability(series, weights)
            yield f'form_prob:{name}:{wname}', probability
            yield (f'form_bet:{name}:{wname}',
                   adapter.recommend_form_bet(probability, series))


def _admission_entries():
    for index, args in enumerate(ADMISSION_CASES):
        yield f'admission:{index}', adapter.evaluate_strategy_admission(*args)


class GoldenTests(unittest.TestCase):
    """迁移前后逐条比对。任何一条对不上，都意味着推荐的方案变了。"""

    def test_matches_golden(self):
        seen = set()
        for key, value in golden_entries():
            seen.add(key)
            with self.subTest(case=key):
                self.assertEqual(as_comparable(value), GOLDEN[key])
        self.assertEqual(sorted(set(GOLDEN) - seen), [])


class CoverageMathTests(unittest.TestCase):
    """命中率是这一层唯一诚实的卖点，算错了会让人按错的预期掏钱。"""

    def test_zu6_note_count_is_the_combination_count(self):
        for size, expected in ((3, 1), (4, 4), (5, 10), (6, 20), (7, 35), (10, 120)):
            with self.subTest(size=size):
                combos, strings = selection.zu6_notes(range(size))
                self.assertEqual(len(combos), expected)
                self.assertEqual(len(strings), expected)

    def test_conditional_rate_is_notes_over_120(self):
        """给定开奖为组六时：注数 ÷ C(10,3)。"""
        payload = selection.zu6_payload([1, 2, 3, 4])
        self.assertEqual(payload['notes'], 4)
        self.assertAlmostEqual(payload['conditional_hit_rate'], 4 / 120, places=4)

    def test_unconditional_rate_includes_the_form_requirement(self):
        """无条件口径要乘 6（每注对应 6 种排列）再除以 1000。"""
        payload = selection.zu6_payload([1, 2, 3, 4])
        self.assertAlmostEqual(payload['hit_rate'], 4 * 6 / 1000, places=4)

    def test_unconditional_is_lower_than_conditional(self):
        """无条件口径必须更低——它多要求了「开奖得是组六」这一步。"""
        payload = selection.zu6_payload([1, 2, 3, 4, 5, 6])
        self.assertLess(payload['hit_rate'], payload['conditional_hit_rate'])

    def test_full_pool_covers_every_zu6_combination(self):
        self.assertEqual(selection.zu6_payload(range(10))['conditional_hit_rate'], 1.0)

    def test_zu3_tier_rate_is_linear_in_the_pair_count(self):
        presence = {digit: 0.2 for digit in range(10)}
        tiers = selection.zu3_coverage_tiers(presence, (1, 5, 45))
        self.assertEqual([tier['conditional_hit_rate'] for tier in tiers],
                         [round(1 / 45, 4), round(5 / 45, 4), 1.0])

    def test_zu3_tier_size_is_capped_at_the_pair_count(self):
        presence = {digit: 0.2 for digit in range(10)}
        self.assertEqual(selection.zu3_coverage_tiers(presence, (60,))[0]['size'], 45)

    def test_group_notes_cost_a_third_of_the_straight_ones(self):
        """2 注组选三与 6 注单选覆盖完全相同的 6 种排列。"""
        pair = (2, 5)
        group = selection.zu3_group_notes(pair)
        straight = selection.zu3_straight_combos(pair)
        self.assertEqual((len(group), len(straight)), (2, 6))
        expanded = set()
        for note in group:
            digits = [int(c) for c in note]
            expanded.update(''.join(map(str, p)) for p in _permutations(digits))
        self.assertEqual(expanded, set(straight))


def _permutations(digits):
    from itertools import permutations
    return set(permutations(digits))


class Zu3PresenceTests(unittest.TestCase):

    def test_only_zu3_draws_are_counted(self):
        """组六期与豹子期不该影响组三条件下的出现率。"""
        series = [(1, 1, 2)] * 10 + [(7, 8, 9)] * 50
        presence = selection.zu3_presence(series, 60, 5)
        self.assertEqual(presence[1], 1.0)
        self.assertEqual(presence[7], 0.0)

    def test_repeated_digit_counts_once_per_draw(self):
        """`112` 里的两个 1 只算一次——问的是「出现与否」。"""
        presence = selection.zu3_presence([(1, 1, 2)] * 5, 60, 1)
        self.assertEqual(presence[1], 1.0)

    def test_thin_sample_falls_back_to_the_longer_window(self):
        series = [(1, 1, 2)] * 3 + [(7, 8, 9)] * 20
        thin = selection.zu3_presence(series, 5, 10)
        self.assertGreater(thin[1], 0)

    def test_no_zu3_at_all_returns_a_uniform_prior(self):
        """返回均匀先验而不是 0：0 会被下游当成「这个号不会出现」。"""
        presence = selection.zu3_presence([(7, 8, 9)] * 30, 30, 5)
        self.assertEqual(set(presence.values()), {0.2})

    def test_pair_scores_sum_to_one(self):
        presence = {digit: 0.2 for digit in range(10)}
        scores = selection.zu3_pair_scores(presence)
        self.assertEqual(len(scores), 45)
        self.assertAlmostEqual(sum(value for _, value in scores), 1.0, places=9)


class KillTests(unittest.TestCase):
    """杀码降权而不是排除——排除等于断言它开不出来。"""

    def test_killed_digit_is_penalised_not_removed(self):
        score = [5.0] * 10
        self.assertEqual(selection.effective_digit_score(score, 3, [3], 8.0), -3.0)
        self.assertEqual(selection.effective_digit_score(score, 4, [3], 8.0), 5.0)

    def test_a_killed_digit_can_still_enter_a_large_pool(self):
        score = [1.0] * 10
        self.assertIn(3, selection.zu6_pool(score, 10, kill=[3], kill_penalty=8.0))

    def test_kill_pushes_a_digit_out_of_a_small_pool(self):
        score = [5.0] * 10
        pool = selection.zu6_pool(score, 4, kill=[0, 1], kill_penalty=8.0)
        self.assertNotIn(0, pool)
        self.assertNotIn(1, pool)

    def test_a_clear_last_place_is_killed_alone(self):
        """末位与倒数第二分差明显时只杀一个，免得误伤。"""
        score = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -10.0]
        _, _, kill, _ = selection.pick_dan_tuo_kill(score, lambda rank: [0, 1])
        self.assertEqual(kill, [9])

    def test_a_close_last_place_kills_two(self):
        """两个方向都要有样本，只测一边判据接反了也发现不了。"""
        score = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.9]
        _, _, kill, _ = selection.pick_dan_tuo_kill(score, lambda rank: [0, 1])
        self.assertEqual(kill, [8, 9])

    def test_tuoma_is_the_third_to_sixth_ranked(self):
        score = [10.0 - i for i in range(10)]
        _, tuoma, _, _ = selection.pick_dan_tuo_kill(score, lambda rank: [0, 1])
        self.assertEqual(tuoma, [2, 3, 4, 5])


class Zu6VariantTests(unittest.TestCase):

    def setUp(self):
        self.score = [10.0 - i * 0.5 for i in range(10)]

    def test_variants_are_distinct_four_digit_groups(self):
        variants = selection.zu6_four_variants(self.score, 4, kill=[9], kill_penalty=8.0)
        keys = [tuple(v['digits']) for v in variants]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(len(k) == 4 for k in keys))

    def test_the_first_variant_is_the_primary_pool(self):
        variants = selection.zu6_four_variants(self.score, 4, kill=None, kill_penalty=8.0)
        self.assertEqual(variants[0]['label'], '主推')
        self.assertEqual(variants[0]['digits'], selection.zu6_pool(self.score, 4))

    def test_the_avoid_kill_variant_skips_killed_digits(self):
        variants = selection.zu6_four_variants(self.score, 4, kill=[0, 1], kill_penalty=8.0)
        avoid = next((v for v in variants if v['label'] == '避杀'), None)
        self.assertIsNotNone(avoid)
        self.assertNotIn(0, avoid['digits'])
        self.assertNotIn(1, avoid['digits'])

    def test_variants_are_deterministic(self):
        """同一份输入两次给出不同的四码，用户无从判断是模型变了还是掷了骰子。"""
        first = selection.zu6_four_variants(self.score, 4, kill=[9], kill_penalty=8.0)
        second = selection.zu6_four_variants(self.score, 4, kill=[9], kill_penalty=8.0)
        self.assertEqual(first, second)

    def test_limit_bounds_the_variant_count(self):
        self.assertEqual(len(selection.zu6_four_variants(self.score, 2, kill_penalty=8.0)), 2)

    def test_balance_prefers_an_even_odd_split(self):
        flat = [1.0] * 10
        balanced = selection.zu6_balance_score((1, 3, 6, 8), flat, None, 0.0)
        lopsided = selection.zu6_balance_score((1, 3, 5, 7), flat, None, 0.0)
        self.assertGreater(balanced, lopsided)

    def test_balance_penalises_adjacent_runs(self):
        flat = [1.0] * 10
        spread = selection.zu6_balance_score((0, 3, 6, 9), flat, None, 0.0)
        run = selection.zu6_balance_score((0, 1, 2, 3), flat, None, 0.0)
        self.assertGreater(spread, run)


class OutOfSampleTests(unittest.TestCase):
    """样本外检验存在的全部理由，就是把「用当期数据选码」那条泄漏堵死。"""

    def test_each_period_only_sees_earlier_draws(self):
        seen = []

        def score_fn(train):
            seen.append(len(train))
            return [1.0] * 10

        selection.evaluate_zu6_pool(NUMBERS[-40:], (5,), trials=5,
                                    score_fn=score_fn, min_train=30)
        self.assertEqual(seen, sorted(seen))
        self.assertLess(max(seen), 40)

    def test_too_little_history_evaluates_nothing(self):
        result = selection.evaluate_zu6_pool(NUMBERS[-10:], (5,), 5,
                                             lambda t: [1.0] * 10, min_train=30)
        self.assertEqual(result, {'trials': 0, 'zu6_draws': 0, 'tiers': {}})

    def test_out_of_range_sizes_are_dropped(self):
        result = selection.evaluate_zu6_pool(NUMBERS[-60:], (1, 5, 99), 5,
                                             lambda t: [1.0] * 10, min_train=30)
        self.assertEqual(sorted(result['tiers']), ['5'])

    def test_full_hit_is_only_counted_on_zu6_draws(self):
        """非组六期本来就不可能全中，算进去会把命中率压得没有意义。"""
        result = selection.evaluate_zu6_pool(NUMBERS[-60:], (10,), 20,
                                             lambda t: [1.0] * 10, min_train=30)
        tier = result['tiers']['10']
        self.assertEqual(tier['full_hit'], result['zu6_draws'])
        self.assertEqual(tier['conditional_full_rate'], 1.0)


class FormProbabilityTests(unittest.TestCase):

    def test_blend_sums_to_one(self):
        forms = ['zu6'] * 70 + ['zu3'] * 27 + ['baozi'] * 3
        uniform = {'zu6': 1 / 3, 'zu3': 1 / 3, 'baozi': 1 / 3}
        _, blended = selection.form_probability(forms, uniform, uniform)
        self.assertAlmostEqual(sum(blended.values()), 1.0, places=9)

    def test_blend_weights_sum_to_one(self):
        """四个来源的权重写成字面量核对：加起来不是 1 的话融合就带了个常数偏移。"""
        self.assertAlmostEqual(sum(selection.FORM_BLEND.values()), 1.0, places=9)
        self.assertEqual(selection.FORM_BLEND['recent'], 0.40)
        self.assertEqual(selection.FORM_BLEND['markov'], 0.35)

    def test_unnormalised_input_is_still_normalised_out(self):
        """窗口权重不归一时 `recent_p` 也不归一——融合末尾那次归一化就是
        为它准备的。四个输入都规范时它是空转，所以只能这样喂才测得到。
        """
        forms = ['zu6'] * 70 + ['zu3'] * 27 + ['baozi'] * 3
        doubled = {'zu6': 1.44, 'zu3': 0.54, 'baozi': 0.02}   # 和 = 2.0
        theory = {'zu6': 0.72, 'zu3': 0.27, 'baozi': 0.01}
        _, blended = selection.form_probability(forms, doubled, theory)
        self.assertAlmostEqual(sum(doubled.values()), 2.0, places=9)
        self.assertAlmostEqual(sum(blended.values()), 1.0, places=9)

    def test_recent_evidence_moves_the_blend(self):
        forms = ['zu6'] * 100
        neutral = {'zu6': 1 / 3, 'zu3': 1 / 3, 'baozi': 1 / 3}
        zu3_heavy = {'zu6': 0.0, 'zu3': 1.0, 'baozi': 0.0}
        _, base = selection.form_probability(forms, neutral, neutral)
        _, shifted = selection.form_probability(forms, zu3_heavy, neutral)
        self.assertGreater(shifted['zu3'], base['zu3'])

    def test_streak_counts_only_the_tail(self):
        self.assertEqual(selection.form_streak(['zu3'] * 5 + ['zu6'] * 3), 3)
        self.assertEqual(selection.form_streak([]), 0)

    def test_signal_thresholds_are_symmetric(self):
        """抬升与回落用同一个余量。只罚一侧会让信号系统性偏斜。"""
        self.assertEqual(selection.form_signal(0.031), 'elevated')
        self.assertEqual(selection.form_signal(-0.031), 'depressed')
        self.assertEqual(selection.form_signal(0.03), 'normal')
        self.assertEqual(selection.form_signal(-0.03), 'normal')


class AdmissionTests(unittest.TestCase):
    """默认不够格。一票否决不是保守，是因为「看起来不错」多半是过拟合。"""

    def _evaluate(self, **over):
        args = {'served_rate': 0.05, 'raw_rate': 0.05, 'average_rank': 300}
        args.update(over)
        return admission.evaluate(**args)

    def test_all_checks_must_pass(self):
        self.assertTrue(self._evaluate()['eligible'])
        self.assertFalse(self._evaluate(served_rate=0.01)['eligible'])
        self.assertFalse(self._evaluate(raw_rate=0.01)['eligible'])
        self.assertFalse(self._evaluate(average_rank=600)['eligible'])

    def test_baseline_defaults_to_the_theoretical_rate(self):
        """门槛写成字面量，不引用被测常量（判据 12）。"""
        checks = self._evaluate()['checks']
        self.assertEqual(checks['served_top30_last100_above_baseline']['required'], 0.03)

    def test_rank_threshold_is_the_random_expectation(self):
        """1000 注随机排的期望名次是 500，所以这条等于「不比随机差」。"""
        self.assertTrue(self._evaluate(average_rank=499)['eligible'])
        self.assertFalse(self._evaluate(average_rank=500)['eligible'])

    def test_baseline_can_be_overridden(self):
        self.assertFalse(self._evaluate(baseline=0.06)['eligible'])

    def test_significance_is_only_checked_when_supplied(self):
        self.assertNotIn('permutation_significant', self._evaluate()['checks'])
        self.assertIn('permutation_significant',
                      self._evaluate(significance={'pvalue': 0.01})['checks'])

    def test_a_missing_pvalue_counts_as_not_significant(self):
        """没有证据不等于有利证据。"""
        self.assertFalse(self._evaluate(significance={})['eligible'])

    def test_pvalue_threshold_is_one_tenth(self):
        self.assertTrue(self._evaluate(significance={'pvalue': 0.099})['eligible'])
        self.assertFalse(self._evaluate(significance={'pvalue': 0.10})['eligible'])

    def test_every_check_reports_both_actual_and_required(self):
        """过不了的时候，人得看到差多少才知道是接近了还是差得远。"""
        for check in self._evaluate(significance={'pvalue': 0.5})['checks'].values():
            self.assertIn('actual', check)
            self.assertIn('required', check)
            self.assertIn('reason', check)


if __name__ == '__main__':
    unittest.main()
