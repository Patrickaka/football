import copy
import unittest

from src.domain.sports.football.accuracy_gate import build_accuracy_gate
from src.domain.sports.football.lottery import lottery_outcomes_compatible


class FootballMarketConsistencyTests(unittest.TestCase):
    @staticmethod
    def markets(standard, handicap, line):
        spf = {key: .84 if key == standard else .08 for key in ('胜', '平', '负')}
        rqspf = {key: .84 if key == handicap else .08 for key in ('让胜', '让平', '让负')}
        return {
            'standard': {'probabilities': spf, 'market_probabilities': dict(spf)},
            'handicap': {'handicap': line, 'probabilities': rqspf,
                         'market_probabilities': dict(rqspf)},
        }

    def test_opposing_one_goal_recommendations_abstain_without_rewriting_probabilities(self):
        for standard, handicap, line in [('胜', '让负', -1), ('负', '让胜', 1)]:
            with self.subTest(line=line):
                lottery = self.markets(standard, handicap, line)
                before = copy.deepcopy(lottery)
                gate = build_accuracy_gate(lottery)
                for market in ('spf', 'rqspf'):
                    self.assertFalse(gate[market]['selected'])
                    self.assertEqual(gate[market]['decision'], '观望')
                    self.assertIn('方向互斥', gate[market]['reasons'][-1])
                self.assertEqual(gate['spf']['candidate'], standard)
                self.assertEqual(gate['rqspf']['candidate'], handicap)
                self.assertEqual(lottery, before)

    def test_two_goal_lines_can_cover_a_one_goal_win_or_loss(self):
        for standard, handicap, line in [('胜', '让负', -2), ('负', '让胜', 2)]:
            with self.subTest(line=line):
                gate = build_accuracy_gate(self.markets(standard, handicap, line))
                self.assertTrue(gate['spf']['selected'])
                self.assertTrue(gate['rqspf']['selected'])

    def test_compatible_one_goal_selections_are_not_downgraded(self):
        cases = [('胜', '让平', -1), ('胜', '让胜', -1),
                 ('负', '让平', 1), ('负', '让负', 1),
                 ('平', '让负', -1), ('平', '让胜', 1)]
        for standard, handicap, line in cases:
            with self.subTest(standard=standard, handicap=handicap, line=line):
                gate = build_accuracy_gate(self.markets(standard, handicap, line))
                self.assertTrue(gate['spf']['selected'])
                self.assertTrue(gate['rqspf']['selected'])

    def test_zero_line_requires_the_same_actual_result(self):
        for standard in ('胜', '平', '负'):
            for handicap in ('让胜', '让平', '让负'):
                with self.subTest(standard=standard, handicap=handicap):
                    gate = build_accuracy_gate(self.markets(standard, handicap, 0))
                    expected = handicap == '让' + standard
                    self.assertEqual(gate['spf']['selected'], expected)
                    self.assertEqual(gate['rqspf']['selected'], expected)

    def test_weak_other_market_does_not_suppress_a_valid_single_recommendation(self):
        lottery = self.markets('胜', '让负', -1)
        lottery['handicap']['probabilities'] = {'让胜': .20, '让平': .25, '让负': .55}
        gate = build_accuracy_gate(lottery)
        self.assertTrue(gate['spf']['selected'])
        self.assertFalse(gate['rqspf']['selected'])
        self.assertNotIn('方向互斥', ';'.join(gate['spf']['reasons']))

    def test_single_handicap_recommendation_does_not_require_a_standard_pick(self):
        lottery = self.markets('胜', '让负', -1)
        lottery['standard'] = None
        gate = build_accuracy_gate(lottery)
        self.assertFalse(gate['spf']['selected'])
        self.assertTrue(gate['rqspf']['selected'])

    def test_unknown_or_asian_line_cannot_certify_a_pair(self):
        for line in (None, '', -0.75, 'not a line'):
            with self.subTest(line=line):
                self.assertIsNone(lottery_outcomes_compatible('胜', '让胜', line))
                gate = build_accuracy_gate(self.markets('胜', '让胜', line))
                self.assertFalse(gate['spf']['selected'])
                self.assertFalse(gate['rqspf']['selected'])
                self.assertIn('无法核验', gate['spf']['reasons'][-1])

    def test_screenshot_marginals_remain_references_without_promoting_conditional_pick(self):
        lottery = {
            'standard': {
                'probabilities': {'胜': .546, '平': .236, '负': .218},
                'market_probabilities': {'胜': .55, '平': .23, '负': .22},
            },
            'handicap': {
                'handicap': -1,
                'probabilities': {'让胜': .316, '让平': .238, '让负': .446},
                'market_probabilities': {'让胜': .32, '让平': .23, '让负': .45},
            },
            'linked_recommendation': {
                'standard_prediction': '胜', 'handicap_prediction': '让平',
                'conditional_probability': 1.0,
            },
        }
        gate = build_accuracy_gate(lottery, confidence={'score': .6})
        self.assertFalse(gate['spf']['selected'])
        self.assertFalse(gate['rqspf']['selected'])
        self.assertEqual(gate['rqspf']['candidate'], '让负')
        self.assertAlmostEqual(gate['rqspf']['probability'], .446)


if __name__ == '__main__':
    unittest.main()
