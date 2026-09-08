"""One ordinary direction must lead to a compatible handicap scenario."""

import copy
import unittest

from src.domain.sports.football.lottery import (
    apply_lottery_market_availability,
    lottery_market_probabilities,
    lottery_outcomes_compatible,
)


OUTCOMES = ('让胜', '让平', '让负')


class FootballDirectionAnalysisTests(unittest.TestCase):
    def assert_coherent_joint(self, lottery):
        analysis = lottery['direction_analysis']
        self.assertTrue(analysis['available'], analysis['reasons'])
        standard = lottery['standard']['probabilities']
        joint = analysis['joint_distribution']
        self.assertAlmostEqual(sum(item['probability'] for item in joint), 1.0)
        self.assertEqual(analysis['probability_basis'], 'conditional_on_standard_result')
        self.assertEqual(analysis['probability_method'], 'standard_anchored_joint_reweighting')
        for result in ('胜', '平', '负'):
            self.assertAlmostEqual(
                sum(item['probability'] for item in joint if item['standard'] == result),
                standard[result])
        for item in joint:
            self.assertGreaterEqual(item['probability'], 0.0)
            if item['probability'] > 0:
                self.assertTrue(lottery_outcomes_compatible(
                    item['standard'], item['handicap'], analysis['handicap']))
        self.assertEqual(set(analysis['conditional_probabilities']), set(OUTCOMES))
        self.assertEqual(set(analysis['joint_probabilities']), set(OUTCOMES))
        self.assertAlmostEqual(sum(analysis['conditional_probabilities'].values()), 1.0)
        self.assertAlmostEqual(sum(analysis['joint_probabilities'].values()),
                               analysis['standard_probability'])
        for result in OUTCOMES:
            selected_row_mass = sum(
                item['probability'] for item in joint
                if item['standard'] == analysis['standard_prediction']
                and item['handicap'] == result)
            self.assertAlmostEqual(analysis['joint_probabilities'][result], selected_row_mass)
            self.assertAlmostEqual(
                selected_row_mass,
                analysis['standard_probability'] * analysis['conditional_probabilities'][result])
        self.assertEqual(analysis['joint_probability'],
                         analysis['joint_probabilities'][analysis['handicap_prediction']])

    def test_plus_one_loss_direction_excludes_independent_handicap_win(self):
        lottery = lottery_market_probabilities([
            ((1, 0), .276), ((1, 1), .253), ((0, 1), .222), ((0, 2), .249),
        ], 1)
        self.assert_coherent_joint(lottery)
        self.assertEqual(lottery['standard']['prediction'], '负')
        self.assertEqual(lottery['handicap']['prediction'], '让胜')
        direction = lottery['direction_analysis']
        self.assertEqual(direction['standard_prediction'], '负')
        self.assertEqual(direction['handicap_prediction'], '让负')
        self.assertEqual(direction['incompatible_handicap_predictions'], ['让胜'])
        self.assertEqual(direction['conditional_probabilities']['让胜'], 0)
        self.assertAlmostEqual(direction['conditional_probability'], .249 / .471)
        self.assertAlmostEqual(direction['joint_probability'], .249)

    def test_minus_one_win_direction_excludes_independent_handicap_loss(self):
        lottery = lottery_market_probabilities([
            ((1, 0), .238), ((2, 0), .308), ((1, 1), .236), ((0, 1), .218),
        ], -1)
        self.assert_coherent_joint(lottery)
        self.assertEqual(lottery['standard']['prediction'], '胜')
        self.assertEqual(lottery['handicap']['prediction'], '让负')
        direction = lottery['direction_analysis']
        self.assertEqual(direction['handicap_prediction'], '让胜')
        self.assertEqual(direction['incompatible_handicap_predictions'], ['让负'])
        self.assertEqual(direction['conditional_probabilities']['让负'], 0)
        self.assertAlmostEqual(direction['joint_probability'], .308)

    def test_two_goal_lines_allow_small_wins_or_losses_when_settlement_allows_them(self):
        for sign, standard_result, handicap_result in ((-1, '胜', '让负'), (1, '负', '让胜')):
            with self.subTest(line=sign * 2):
                scores = [((1, 0), .30), ((2, 0), .12), ((3, 0), .18),
                          ((1, 1), .20), ((0, 1), .20)]
                if sign > 0:
                    scores = [((away, home), p) for (home, away), p in scores]
                lottery = lottery_market_probabilities(scores, sign * 2)
                self.assert_coherent_joint(lottery)
                direction = lottery['direction_analysis']
                self.assertEqual(direction['standard_prediction'], standard_result)
                self.assertEqual(direction['handicap_prediction'], handicap_result)
                self.assertEqual(direction['compatible_handicap_predictions'], list(OUTCOMES))
                self.assertEqual(direction['incompatible_handicap_predictions'], [])
                self.assertAlmostEqual(direction['joint_probability'], .30)

    def test_zero_line_has_certain_mapping_but_not_certain_full_match_probability(self):
        lottery = lottery_market_probabilities([
            ((1, 0), .50), ((1, 1), .25), ((0, 1), .25),
        ], 0)
        self.assert_coherent_joint(lottery)
        direction = lottery['direction_analysis']
        self.assertEqual(direction['compatible_handicap_predictions'], ['让胜'])
        self.assertEqual(direction['conditional_probability'], 1.0)
        self.assertEqual(direction['joint_probability'], .5)
        self.assertEqual(lottery['handicap']['probabilities']['让胜'], .5)

    def test_draw_condition_is_not_saved_as_a_certain_handicap_match(self):
        scores = [((1, 1), .4), ((1, 0), .2), ((2, 0), .1),
                  ((0, 1), .2), ((0, 2), .1)]
        for line, handicap_pick in ((1, '让胜'), (-1, '让负')):
            with self.subTest(line=line):
                lottery = lottery_market_probabilities(scores, line)
                self.assert_coherent_joint(lottery)
                direction = lottery['direction_analysis']
                self.assertEqual(direction['standard_prediction'], '平')
                self.assertEqual(direction['handicap_prediction'], handicap_pick)
                self.assertAlmostEqual(direction['conditional_probability'], 1.0)
                self.assertAlmostEqual(direction['joint_probability'], .4)
                self.assertTrue(all(0 < p < 1 for p in lottery['handicap']['probabilities'].values()))

    def test_official_market_reweights_the_joint_without_replacing_independent_marginals(self):
        scores = [((2, 0), .30), ((1, 0), .30), ((1, 1), .20), ((0, 1), .20)]
        original = copy.deepcopy(scores)
        spf_odds = {'胜': 2.0, '平': 4.0, '负': 4.0}
        rq_odds = {'让胜': 4.0, '让平': 4.0, '让负': 2.0}
        lottery = lottery_market_probabilities(scores, -1, spf_odds, rq_odds)
        self.assert_coherent_joint(lottery)
        self.assertEqual(scores, original)
        standard = lottery['standard']['probabilities']
        handicap = lottery['handicap']['probabilities']
        self.assertAlmostEqual(standard['胜'], .52)
        self.assertAlmostEqual(handicap['让负'], .48)
        direction = lottery['direction_analysis']
        self.assertAlmostEqual(direction['joint_probability'], .26)
        # In the reconstructed joint, handicap loss equals draw + loss at -1.
        reconstructed_loss = sum(item['probability'] for item in direction['joint_distribution']
                                 if item['handicap'] == '让负')
        self.assertAlmostEqual(reconstructed_loss, standard['平'] + standard['负'])
        self.assertAlmostEqual(direction['joint_probabilities']['让负'], 0)
        self.assertEqual(lottery['primary']['probabilities'], handicap)

    def test_missing_or_non_integer_line_is_explicitly_unavailable(self):
        for line in (None, '', 1.5, 'unknown'):
            with self.subTest(line=line):
                direction = lottery_market_probabilities([
                    ((1, 0), .5), ((1, 1), .3), ((0, 1), .2),
                ], line)['direction_analysis']
                self.assertFalse(direction['available'])
                self.assertEqual(direction['standard_prediction'], '胜')
                self.assertIsNone(direction['handicap_prediction'])
                self.assertIsNone(direction['joint_distribution'])
                self.assertTrue(direction['reasons'])

    def test_no_scores_cannot_fall_back_to_independently_priced_top_results(self):
        direction = lottery_market_probabilities(
            [], -1, {'胜': 2, '平': 4, '负': 4},
            {'让胜': 4, '让平': 4, '让负': 2})['direction_analysis']
        self.assertFalse(direction['available'])
        self.assertEqual(direction['standard_prediction'], '胜')
        self.assertIsNone(direction['handicap_prediction'])
        self.assertIsNone(direction['conditional_probabilities'])
        self.assertEqual(direction['score_probability_mass'], 0)

    def test_empty_predictions_remain_unavailable(self):
        direction = lottery_market_probabilities([], 1)['direction_analysis']
        self.assertFalse(direction['available'])
        self.assertIsNone(direction['standard_prediction'])
        self.assertIsNone(direction['standard_probability'])

    def test_truncated_probability_mass_cannot_be_renormalized_into_certainty(self):
        direction = lottery_market_probabilities([
            ((1, 0), .12), ((2, 0), .10), ((0, 1), .06),
        ], -1)['direction_analysis']
        self.assertFalse(direction['available'])
        self.assertAlmostEqual(direction['score_probability_mass'], .28)
        self.assertIsNone(direction['conditional_probability'])
        self.assertIn('完整的比分', direction['reasons'][0])

    def test_missing_compatible_score_support_does_not_mean_impossible(self):
        direction = lottery_market_probabilities([
            ((1, 0), .6), ((1, 1), .2), ((0, 1), .2),
        ], -1)['direction_analysis']
        self.assertFalse(direction['available'])
        self.assertEqual(direction['compatible_handicap_predictions'], ['让胜', '让平'])
        self.assertEqual(direction['incompatible_handicap_predictions'], ['让负'])
        self.assertIsNone(direction['conditional_probabilities'])
        self.assertIn('让胜', direction['reasons'][0])

    def test_positive_standard_market_row_without_scores_cannot_be_discarded(self):
        direction = lottery_market_probabilities(
            [((1, 0), .6), ((2, 0), .4)], -1,
            {'胜': 1.25, '平': 10, '负': 10})['direction_analysis']
        self.assertFalse(direction['available'])
        self.assertIsNone(direction['joint_distribution'])
        self.assertIn('缺少比分支持', direction['reasons'][0])

    def test_closed_standard_market_clears_direction_but_preserves_handicap(self):
        lottery = lottery_market_probabilities([
            ((2, 0), .4), ((1, 0), .3), ((1, 1), .2), ((0, 1), .1),
        ], -1)
        handicap = copy.deepcopy(lottery['handicap'])
        lottery.update(offer_matched=True, spf_available=False)
        self.assertTrue(lottery['direction_analysis']['available'])
        self.assertFalse(apply_lottery_market_availability(lottery))
        self.assertIsNone(lottery['direction_analysis'])
        self.assertEqual(lottery['handicap'], handicap)

    def test_offered_standard_market_keeps_direction_analysis(self):
        lottery = lottery_market_probabilities([
            ((2, 0), .4), ((1, 0), .3), ((1, 1), .2), ((0, 1), .1),
        ], -1)
        direction = copy.deepcopy(lottery['direction_analysis'])
        lottery.update(offer_matched=True, spf_available=True,
                       spf_odds={'胜': 2, '平': 4, '负': 4})
        self.assertTrue(apply_lottery_market_availability(lottery))
        self.assertEqual(lottery['direction_analysis'], direction)

    def test_all_supported_integer_lines_and_directions_preserve_joint_invariants(self):
        scores = [((home, away), 1 / 64) for home in range(8) for away in range(8)]
        for line in range(-5, 6):
            for standard_pick in ('胜', '平', '负'):
                with self.subTest(line=line, standard=standard_pick):
                    odds = {result: (1.25 if result == standard_pick else 10)
                            for result in ('胜', '平', '负')}
                    lottery = lottery_market_probabilities(
                        scores, line, odds, {'让胜': 2, '让平': 4, '让负': 4})
                    self.assert_coherent_joint(lottery)
                    direction = lottery['direction_analysis']
                    self.assertEqual(direction['standard_prediction'], standard_pick)
                    for result in direction['incompatible_handicap_predictions']:
                        self.assertEqual(direction['conditional_probabilities'][result], 0)


if __name__ == '__main__':
    unittest.main()
