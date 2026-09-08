import copy
import unittest

from src.domain.sports.football.accuracy_gate import build_accuracy_gate
from src.domain.sports.football.lottery import lottery_outcomes_compatible
from src.domain.sports.football.settlement import _audited_decision_snapshot


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
                    self.assertTrue(gate[market]['independent_selected'])
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

    def test_open_market_pairs_and_settlement_snapshot_share_the_compatibility_rule(self):
        cases = [
            ('胜', '让负', -1, False), ('负', '让胜', 1, False),
            ('胜', '让负', -2, True), ('负', '让胜', 2, True),
            # Two goals are an exception only when the actual score can win both.
            ('胜', '让负', 2, False), ('负', '让胜', -2, False),
        ]
        for standard, handicap, line, expected in cases:
            with self.subTest(standard=standard, handicap=handicap, line=line):
                lottery = self.markets(standard, handicap, line)
                lottery.update(
                    offer_matched=True, spf_available=True, rqspf_available=True,
                    spf_odds={key: 1 / value for key, value
                              in lottery['standard']['probabilities'].items()},
                    rqspf_odds={key: 1 / value for key, value
                                in lottery['handicap']['probabilities'].items()},
                )
                original = copy.deepcopy(lottery)
                gate = build_accuracy_gate(lottery)
                for market in ('spf', 'rqspf'):
                    self.assertTrue(gate[market]['independent_selected'])
                    self.assertEqual(gate[market]['selected'], expected)
                    self.assertEqual(gate[market]['decision'],
                                     gate[market]['candidate'] if expected else '观望')
                snapshot = _audited_decision_snapshot(
                    {'H': .84 if standard == '胜' else .08, 'D': .08,
                     'A': .84 if standard == '负' else .08},
                    {'accuracy_gate': gate},
                )
                self.assertEqual(snapshot['eligible'], expected)
                self.assertEqual(lottery, original)

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

    def test_ordinary_abstention_does_not_allow_opposing_handicap_recommendation(self):
        for standard, handicap, line in [('胜', '让负', -1), ('负', '让胜', 1)]:
            with self.subTest(standard=standard, line=line):
                lottery = self.markets(standard, handicap, line)
                lottery['standard']['probabilities'] = {
                    key: .50 if key == standard else .25 for key in ('胜', '平', '负')}
                before = copy.deepcopy(lottery)
                gate = build_accuracy_gate(lottery)
                self.assertFalse(gate['spf']['independent_selected'])
                self.assertFalse(gate['spf']['selected'])
                self.assertNotIn('互斥', ';'.join(gate['spf']['reasons']))
                self.assertTrue(gate['rqspf']['independent_selected'])
                self.assertFalse(gate['rqspf']['selected'])
                self.assertEqual(gate['rqspf']['decision'], '观望')
                self.assertEqual(gate['rqspf']['candidate'], handicap)
                self.assertEqual(gate['rqspf']['probability'], .84)
                self.assertIn('主方向互斥', gate['rqspf']['reasons'][-1])
                self.assertEqual(lottery, before)

    def test_two_goal_line_preserves_compatible_handicap_when_ordinary_abstains(self):
        for standard, handicap, line in [('胜', '让负', -2), ('负', '让胜', 2)]:
            with self.subTest(line=line):
                lottery = self.markets(standard, handicap, line)
                lottery['standard']['probabilities'] = {
                    key: .50 if key == standard else .25 for key in ('胜', '平', '负')}
                gate = build_accuracy_gate(lottery)
                self.assertFalse(gate['spf']['selected'])
                self.assertTrue(gate['rqspf']['independent_selected'])
                self.assertTrue(gate['rqspf']['selected'])
                self.assertEqual(gate['rqspf']['decision'], handicap)

    def test_unknown_line_blocks_handicap_against_valid_unselected_ordinary_direction(self):
        for line in (None, '', -.75):
            with self.subTest(line=line):
                lottery = self.markets('胜', '让胜', line)
                lottery['standard']['probabilities'] = {'胜': .50, '平': .25, '负': .25}
                gate = build_accuracy_gate(lottery)
                self.assertFalse(gate['spf']['selected'])
                self.assertTrue(gate['rqspf']['independent_selected'])
                self.assertFalse(gate['rqspf']['selected'])
                self.assertEqual(gate['rqspf']['candidate'], '让胜')
                self.assertIn('无法核验', gate['rqspf']['reasons'][-1])

    def test_closed_ordinary_market_cannot_suppress_handicap_only_recommendation(self):
        lottery = self.markets('胜', '让负', -1)
        lottery.update(offer_matched=True, spf_available=False, spf_odds=None,
                       rqspf_available=True)
        before = copy.deepcopy(lottery)
        gate = build_accuracy_gate(lottery)
        self.assertFalse(gate['spf']['selected'])
        self.assertTrue(gate['rqspf']['independent_selected'])
        self.assertTrue(gate['rqspf']['selected'])
        self.assertEqual(gate['rqspf']['decision'], '让负')
        self.assertEqual(lottery, before)

    def test_unavailable_ordinary_distribution_does_not_invent_a_conflicting_direction(self):
        for probabilities in (None, {}, {'胜': 0, '平': 0, '负': 0},
                              {'胜': .4}, {'胜': .4, '平': .2, '负': .1}):
            with self.subTest(probabilities=probabilities):
                lottery = self.markets('胜', '让负', -1)
                lottery['standard']['probabilities'] = probabilities
                gate = build_accuracy_gate(lottery)
                self.assertFalse(gate['spf']['selected'])
                self.assertTrue(gate['rqspf']['selected'])

    def test_tied_ordinary_direction_respects_declared_backend_top_pick(self):
        lottery = self.markets('胜', '让负', -1)
        lottery['standard'].update(
            prediction='平', probabilities={'胜': .4, '平': .4, '负': .2})
        gate = build_accuracy_gate(lottery)
        self.assertFalse(gate['spf']['selected'])
        # Given a draw, -1 settles as 让负; a fresh sort choosing 胜 would block it.
        self.assertTrue(gate['rqspf']['selected'])

    def test_conditional_certainty_does_not_promote_a_rejected_full_match_market(self):
        lottery = self.markets('胜', '让胜', -1)
        lottery['standard']['probabilities'] = {'胜': .5, '平': .3, '负': .2}
        lottery['handicap']['probabilities'] = {'让胜': .55, '让平': .25, '让负': .20}
        lottery['direction_analysis'] = {
            'available': True, 'standard_prediction': '胜', 'handicap_prediction': '让胜',
            'conditional_probability': 1.0,
            'conditional_probabilities': {'让胜': 1.0, '让平': 0.0, '让负': 0.0},
        }
        gate = build_accuracy_gate(lottery)
        self.assertFalse(gate['rqspf']['independent_selected'])
        self.assertFalse(gate['rqspf']['selected'])
        self.assertEqual(gate['rqspf']['probability'], .55)

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
