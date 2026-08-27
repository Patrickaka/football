"""北单的推荐质量分档与爆冷判定。

参照物是从迁移前的 `quality.py` / `upset.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_quality.json.gz`，251 条），**逐条相同**。

语料围绕**每一档阈值的两侧**构造：三档爆冷（high/medium/confident）各有三个
条件，五档质量（strong/medium/split/low/unknown）另有两道门槛，
只喂「明显超过」或「明显不足」的样本，把阈值改一点点是测不出来的（判据 5）。

**第一版语料的名字与它实际走到的分支不符**：三条命名为 `medium_*` 的输入
全都落在 low（热门 0.50 时非热门合计只有 0.50，差 0.02 够不到 0.52），
真正命中 medium 的反而是意外撞上的 `draw_favorite`。名不副实的语料比没有
更糟——读的人会以为那一档测过了。修正后 `medium_hit` 真的命中，
三个 `medium_miss_*` 各在一个条件上差一点。

**规划这一批时我把档数记错了**：计划里写「六个阈值、两档」，实际是八个、三档
——多一档 `confident`（未预警且热门够强 → 热门稳胆）。它与前两档不是同一个
维度，只在没预警的场次里再细分，不改变 `level` 与 `alert`。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.sports.beidan import quality, upset
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_quality.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时生效的那组门槛，写死不 import（判据 12）
STRONG_PROB, STRONG_LEAD = 0.65, 0.10
MEDIUM_PROB, MEDIUM_LEAD = 0.60, 0.10
HIGH_PRECISION_PROB, SPLIT_LEAD = 0.70, 0.035
HIGH_FAV, HIGH_GAP, HIGH_MASS = 0.45, 0.10, 0.58
MED_FAV, MED_GAP, MED_MASS = 0.52, 0.16, 0.52
CONFIDENT_FAV, CONFIDENT_GAP = 0.58, 0.20


def golden_entries():
    from scripts.gen_beidan_quality_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class QualityLevelTests(unittest.TestCase):
    """五档的边界。**每一档都测两侧**——只测「够格」那边，
    把门槛改低了照样全绿。"""

    def test_strong_needs_both_probability_and_lead(self):
        strong = quality.assess({'胜': 0.68, '平': 0.20, '负': 0.12})
        self.assertEqual(strong['level'], 'strong')
        self.assertTrue(strong['single_allowed'])
        self.assertFalse(strong['avoid_single'])

    def test_just_below_strong_probability_is_not_strong(self):
        """0.64 差一点点就不是强推荐。"""
        result = quality.assess({'胜': 0.64, '平': 0.22, '负': 0.14})
        self.assertNotEqual(result['level'], 'strong')

    def test_enough_probability_but_thin_lead_is_not_strong(self):
        """概率够了、领先不够，同样不能单选——**两个条件是与不是或**。"""
        result = quality.assess({'胜': 0.66, '平': 0.60, '负': -0.26})
        self.assertNotEqual(result['level'], 'strong')

    def test_high_precision_needs_a_higher_bar_than_strong(self):
        """0.70 那道额外的门槛：是强推荐，但不一定「高精度」。"""
        precise = quality.assess({'胜': 0.75, '平': 0.15, '负': 0.10})
        merely_strong = quality.assess({'胜': 0.69, '平': 0.19, '负': 0.12})
        self.assertTrue(precise['high_precision'])
        self.assertEqual(merely_strong['level'], 'strong')
        self.assertFalse(merely_strong['high_precision'])

    def test_hairline_lead_reports_split_not_low(self):
        """领先不到 3.5 个百分点报「分歧较大」而不是「低置信」。

        **两者的含义不同**：低置信像是有个答案只是没把握，
        分歧才是实情——谁排第一基本由噪声决定。
        """
        result = quality.assess({'胜': 0.35, '平': 0.34, '负': 0.31})
        self.assertEqual(result['level'], 'split')

    def test_just_above_split_threshold_is_low_not_split(self):
        """边界另一侧：领先 4 个百分点就不再算分歧。"""
        result = quality.assess({'胜': 0.40, '平': 0.36, '负': 0.24})
        self.assertEqual(result['level'], 'low')

    def test_empty_input_is_unknown_and_skipped(self):
        for probabilities in ({}, {'胜': None, '平': None}):
            result = quality.assess(probabilities)
            self.assertEqual(result['level'], 'unknown')
            self.assertEqual(result['advice'], '跳过')
            self.assertTrue(result['avoid_single'])

    def test_thresholds_are_reported_back(self):
        """过不了的时候人要看到差多少——**只报「不够格」等于没说**。"""
        reported = quality.assess({'胜': 0.5, '平': 0.3, '负': 0.2})['thresholds']
        self.assertEqual(reported['strong_probability'], STRONG_PROB)
        self.assertEqual(reported['high_precision_probability'], HIGH_PRECISION_PROB)


class QualityConflictTests(unittest.TestCase):
    """盘口方向与推荐打架时，一律降到 split。"""

    STRONG = {'胜': 0.68, '平': 0.20, '负': 0.12}

    def test_asian_direction_against_the_pick_forces_split(self):
        result = quality.assess({'负': 0.68, '平': 0.20, '胜': 0.12},
                                context={'asian_direction': 'home_backing'})
        self.assertTrue(result['conflict'])
        self.assertEqual(result['level'], 'split')

    def test_asian_direction_with_the_pick_is_not_a_conflict(self):
        """**同向不算冲突**——只测冲突那一侧的话，把判断写成恒真也全绿。"""
        result = quality.assess(self.STRONG,
                                context={'asian_direction': 'home_backing'})
        self.assertFalse(result['conflict'])
        self.assertEqual(result['level'], 'strong')

    def test_score_conflict_alone_forces_split(self):
        result = quality.assess(self.STRONG,
                                context={'score_consistency': {'conflict': True}})
        self.assertTrue(result['conflict'])
        self.assertEqual(result['level'], 'split')

    def test_explicit_prediction_overrides_the_top_pick(self):
        """指定了推荐项就评估那一注——**评的是「这一注」，不是「最高的那注」**。"""
        result = quality.assess({'胜': 0.68, '平': 0.20, '负': 0.12},
                                prediction='平')
        self.assertEqual(result['prediction'], '平')
        self.assertAlmostEqual(result['confidence'], 0.20)


class UpsetLevelTests(unittest.TestCase):
    """三档爆冷判定，每档三个条件全部满足才算。"""

    def test_high_needs_all_three_conditions(self):
        risk = upset.assess_risk({'胜': 0.42, '平': 0.33, '负': 0.25})
        self.assertEqual(risk['level'], 'high')
        self.assertTrue(risk['alert'])

    def test_favourite_too_strong_drops_out_of_high(self):
        """热门刚过 0.45 就不再是高风险。"""
        risk = upset.assess_risk({'胜': 0.46, '平': 0.30, '负': 0.24})
        self.assertNotEqual(risk['level'], 'high')

    def test_gap_too_wide_drops_out_of_high(self):
        """三个条件是与不是或——差距一拉开就不算胶着。"""
        risk = upset.assess_risk({'胜': 0.44, '平': 0.28, '负': 0.28})
        self.assertEqual(risk['gap'], 0.16)
        self.assertNotEqual(risk['level'], 'high')

    def test_medium_is_the_looser_tier(self):
        """三个条件里 `mass >= 0.52` 最容易被忽略——**热门 0.50 时非热门
        合计只有 0.50，差 0.02 就落回 low**。我第一次写这条用例正是栽在这里。"""
        risk = upset.assess_risk({'胜': 0.48, '平': 0.34, '负': 0.18})
        self.assertEqual(risk['level'], 'medium')
        self.assertTrue(risk['alert'])

    def test_medium_falls_back_to_low_when_mass_is_short(self):
        """边界另一侧：非热门合计差 0.02 就不再预警。"""
        risk = upset.assess_risk({'胜': 0.50, '平': 0.35, '负': 0.15})
        self.assertEqual(risk['upset_prob'], 0.5)
        self.assertEqual(risk['level'], 'low')

    def test_confident_only_applies_when_not_alerted(self):
        """`confident` 是另一个维度：**只在未预警的场次里再细分**，
        不改变 level 与 alert。"""
        risk = upset.assess_risk({'胜': 0.62, '平': 0.22, '负': 0.16})
        self.assertEqual(risk['level'], 'low')
        self.assertFalse(risk['alert'])
        self.assertTrue(risk['confident'])
        self.assertEqual(risk['label'], '热门稳胆')

    def test_confident_needs_both_a_strong_favourite_and_a_wide_gap(self):
        weak_favourite = upset.assess_risk({'胜': 0.57, '平': 0.25, '负': 0.18})
        self.assertFalse(weak_favourite['confident'])

    def test_low_without_confident_keeps_the_plain_label(self):
        risk = upset.assess_risk({'胜': 0.55, '平': 0.30, '负': 0.15})
        self.assertFalse(risk['confident'])
        self.assertEqual(risk['label'], '热门稳健')

    def test_empty_input_is_low_risk_not_an_error(self):
        for probabilities in ({}, {'胜': None}):
            risk = upset.assess_risk(probabilities)
            self.assertEqual(risk['level'], 'low')
            self.assertIsNone(risk['favorite'])


class DefensiveSelectionTests(unittest.TestCase):

    def test_defensive_directions_oppose_the_favourite(self):
        """热门主胜就防平与负——**挑同方向的不是防守，是加倍**。"""
        risk = upset.assess_risk({'胜': 0.42, '平': 0.33, '负': 0.25})
        results = {item['result'] for item in risk['defensive_selections']}
        self.assertEqual(results, {'平', '负'})

    def test_draw_favourite_defends_both_sides(self):
        risk = upset.assess_risk({'胜': 0.30, '平': 0.40, '负': 0.30})
        results = {item['result'] for item in risk['defensive_selections']}
        self.assertEqual(results, {'胜', '负'})

    def test_no_defence_when_not_alerted(self):
        """没预警就不给防守方向——**给了反而会让人以为该防**。"""
        risk = upset.assess_risk({'胜': 0.68, '平': 0.20, '负': 0.12})
        self.assertEqual(risk['defensive_selections'], [])
        self.assertIsNone(risk['recommended_cover'])

    def test_defensive_selections_are_sorted_by_probability(self):
        risk = upset.assess_risk({'胜': 0.42, '平': 0.33, '负': 0.25})
        probs = [item['probability'] for item in risk['defensive_selections']]
        self.assertEqual(probs, sorted(probs, reverse=True))


class ScoreParsingTests(unittest.TestCase):
    """三种比分写法都要认——**认错了不会报错，只会挑到反方向的比分**。"""

    def test_all_three_notations_agree(self):
        self.assertEqual(upset.result_from_score((1, 0)), '胜')
        self.assertEqual(upset.result_from_score('1-0'), '胜')
        self.assertEqual(upset.result_from_score('1:0'), '胜')

    def test_draw_and_away_win(self):
        self.assertEqual(upset.result_from_score((1, 1)), '平')
        self.assertEqual(upset.result_from_score((0, 1)), '负')

    def test_unparsable_returns_none_not_a_guess(self):
        for bad in ('abc', None, '', (1,)):
            self.assertIsNone(upset.result_from_score(bad))


class PickScoresTests(unittest.TestCase):

    MATRIX = {(1, 0): 0.12, (1, 1): 0.11, (0, 1): 0.09,
              (2, 1): 0.08, (0, 0): 0.07, (2, 0): 0.06}

    def test_picks_only_the_opposite_direction(self):
        picked = upset.pick_scores(self.MATRIX, '胜', top_n=3)
        self.assertTrue(all(item['result'] in ('平', '负') for item in picked))

    def test_respects_top_n(self):
        self.assertEqual(len(upset.pick_scores(self.MATRIX, '胜', top_n=1)), 1)
        self.assertEqual(len(upset.pick_scores(self.MATRIX, '胜', top_n=2)), 2)

    def test_highest_probability_comes_first(self):
        picked = upset.pick_scores(self.MATRIX, '胜', top_n=3)
        probs = [item['probability'] for item in picked]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_missing_inputs_return_empty(self):
        self.assertEqual(upset.pick_scores({}, '胜'), [])
        self.assertEqual(upset.pick_scores(self.MATRIX, None), [])

    def test_unparsable_keys_are_skipped_not_fatal(self):
        picked = upset.pick_scores({'abc': 0.5, (0, 1): 0.3}, '胜', top_n=2)
        self.assertEqual([item['score'] for item in picked], ['0-1'])


class ScoreConsistencyTests(unittest.TestCase):

    def test_agreement_with_the_prediction_is_not_a_conflict(self):
        scores = [{'score': '2-1', 'probability': 0.12},
                  {'score': '1-0', 'probability': 0.10},
                  {'score': '1-1', 'probability': 0.09}]
        result = upset.score_consistency(scores, '胜')
        self.assertTrue(result['available'])
        self.assertFalse(result['conflict'])

    def test_top_score_disagreeing_alone_is_not_a_conflict(self):
        """**最高比分指向别处很正常**——只有支持推荐的权重也不足才算冲突。

        这条与下一条成对：只测「冲突」那侧的话，把判断写成恒真也全绿。
        """
        scores = [{'score': '1-1', 'probability': 0.12},
                  {'score': '2-1', 'probability': 0.11},
                  {'score': '3-1', 'probability': 0.10}]
        result = upset.score_consistency(scores, '胜')
        self.assertEqual(result['top_score_result'], '平')
        self.assertFalse(result['conflict'], '支持主胜的权重仍然过半')

    def test_conflict_when_support_is_thin(self):
        scores = [{'score': '1-1', 'probability': 0.20},
                  {'score': '0-1', 'probability': 0.18},
                  {'score': '2-1', 'probability': 0.05}]
        result = upset.score_consistency(scores, '胜')
        self.assertTrue(result['conflict'])

    def test_tuple_form_is_accepted(self):
        result = upset.score_consistency([('2-1', 0.12), ('1-1', 0.10)], '胜')
        self.assertTrue(result['available'])

    def test_missing_probabilities_fall_back_to_rank_weights(self):
        """没有概率时按名次退化加权，**下限防止排后面的权重变成 0**。"""
        scores = [{'score': '2-1'}, {'score': '1-0'}, {'score': '1-1'}]
        result = upset.score_consistency(scores, '胜')
        self.assertTrue(result['available'])
        self.assertGreater(result['agreement'], 0.5)

    def test_unavailable_when_nothing_parses(self):
        self.assertFalse(upset.score_consistency([{'score': 'x'}], '胜')['available'])
        self.assertFalse(upset.score_consistency([], '胜')['available'])
        self.assertFalse(upset.score_consistency([{'score': '1-0'}], None)['available'])

    def test_weights_sum_to_one(self):
        scores = [{'score': '2-1', 'probability': 0.12},
                  {'score': '1-1', 'probability': 0.10},
                  {'score': '0-1', 'probability': 0.09}]
        weights = upset.score_consistency(scores, '胜')['result_weights']
        # places 取 5：三个权重各自 round 到 6 位，加起来的舍入误差在 1e-6 量级
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
