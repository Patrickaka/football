import unittest

from src.common.local_match_analysis import (
    build_decision, normalize_probabilities, pick_high_score_scenario,
    build_score_strategy,
)
from src.basketball import assess_basketball_upset, build_basketball_analysis
from src.football import _anchor_score_candidates_to_goal_mean


class LocalMatchAnalysisTests(unittest.TestCase):
    def test_normalizes_invalid_probabilities(self):
        probs = normalize_probabilities({'a': 2, 'b': -1, 'c': None})
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(probs['b'], 0.0)

    def test_high_score_scenario_uses_aggregate_tail(self):
        scenario = pick_high_score_scenario([
            ((1, 1), 0.15), ((2, 1), 0.13), ((2, 2), 0.09),
            ((3, 1), 0.08), ((3, 2), 0.06),
        ])
        self.assertEqual(scenario['score'], (2, 2))
        self.assertAlmostEqual(scenario['tail_probability'], 0.23)

    def test_weak_high_score_tail_is_not_forced(self):
        self.assertIsNone(pick_high_score_scenario([
            ((1, 1), 0.30), ((3, 1), 0.08), ((3, 2), 0.04),
        ]))

    def test_weak_exact_score_uses_range(self):
        strategy = build_score_strategy([
            ((1, 1), 0.12), ((1, 0), 0.10), ((2, 1), 0.09),
        ], confidence='medium')
        self.assertEqual(strategy['action'], '比分区间')
        self.assertFalse(strategy['playable'])

    def test_concentrated_exact_score_can_be_marked_cautious(self):
        strategy = build_score_strategy([
            ((2, 0), 0.15), ((2, 1), 0.11), ((1, 0), 0.09),
        ], confidence='high')
        self.assertEqual(strategy['action'], '谨慎单比分')
        self.assertTrue(strategy['playable'])

    def test_score_goal_anchor_preserves_1x2_mass(self):
        candidates = [
            ((0, 0), 0.25), ((1, 1), 0.20),
            ((1, 0), 0.25), ((2, 0), 0.10),
            ((0, 1), 0.15), ((0, 2), 0.05),
        ]
        anchored, meta = _anchor_score_candidates_to_goal_mean(
            candidates, {'implied_total': 2.0}
        )
        before = {'H': 0.35, 'D': 0.45, 'A': 0.20}
        after = {'H': 0.0, 'D': 0.0, 'A': 0.0}
        for (home, away), probability in anchored:
            key = 'H' if home > away else ('D' if home == away else 'A')
            after[key] += probability
        for key in before:
            self.assertAlmostEqual(after[key], before[key], places=8)
        self.assertTrue(meta['applied'])
        self.assertAlmostEqual(meta['expected_after'], meta['target'], places=6)

    def test_high_total_market_restores_large_score_tail(self):
        candidates = [
            ((0, 0), 0.20), ((1, 0), 0.24), ((1, 1), 0.22),
            ((2, 0), 0.14), ((2, 1), 0.10), ((2, 2), 0.05),
            ((3, 1), 0.03), ((3, 2), 0.02),
        ]
        anchored, meta = _anchor_score_candidates_to_goal_mean(
            candidates, {'implied_total': 3.40}
        )
        tail_before = sum(prob for score, prob in candidates if sum(score) >= 4)
        tail_after = sum(prob for score, prob in anchored if sum(score) >= 4)

        self.assertTrue(meta['applied'])
        self.assertEqual(meta['requested_target'], 3.40)
        self.assertGreater(meta['expected_after'], 2.90)
        self.assertGreater(tail_after, tail_before * 3)

    def test_close_match_uses_double_selection(self):
        decision = build_decision({'胜': 0.38, '平': 0.33, '负': 0.29}, confidence='high')
        self.assertEqual(decision['action'], '双选')
        self.assertEqual(decision['secondary'], '平')

    def test_low_confidence_is_not_forced_to_single(self):
        decision = build_decision({'主胜': 0.57, '客胜': 0.43}, confidence='low')
        self.assertEqual(decision['action'], '观望')

    def test_basketball_scores_are_projections_not_fake_probabilities(self):
        result = build_basketball_analysis({
            'match': {'home': 'A', 'away': 'B'},
            'spf': {'home_prob': 0.7, 'away_prob': 0.3, 'confidence': 'high'},
            'rqspf': {'elo_margin': 6},
            'dx': {'total_line': 210, 'elo_total': 208,
                   'over_prob': 0.45, 'under_prob': 0.55,
                   'recommendation': '小分'},
        })
        self.assertEqual(result['decision']['action'], '单选')
        self.assertEqual(len(result['score_picks']), 2)
        self.assertTrue(all(p['projected'] for p in result['score_picks']))
        self.assertTrue(all(p['probability'] is None for p in result['score_picks']))

    def test_basketball_detects_model_market_upset_risk(self):
        upset = assess_basketball_upset({
            'home_prob': 0.62, 'away_prob': 0.38,
            'market_home_prob': 0.46, 'elo_trust': 0.8,
            'books_ml': {'trend': 'away_backing'},
        }, {'recommendation': '让负'})
        self.assertTrue(upset['alert'])
        self.assertEqual(upset['level'], 'high')
        self.assertIn('本地模型与市场胜负方向相反', upset['signals'])
        self.assertIn('赔率资金走势与热门方向相反', upset['signals'])

    def test_basketball_stable_favorite_has_no_alert(self):
        upset = assess_basketball_upset({
            'home_prob': 0.68, 'away_prob': 0.32,
            'market_home_prob': 0.64, 'elo_trust': 0.8,
            'books_ml': {'trend': 'home_backing'},
        }, {'recommendation': '让胜'})
        self.assertFalse(upset['alert'])

    def test_basketball_score_total_follows_over_under_direction(self):
        result = build_basketball_analysis({
            'match': {'home': 'A', 'away': 'B'},
            'spf': {'home_prob': 0.65, 'away_prob': 0.35, 'confidence': 'high',
                    'elo_trust': 0.8},
            'rqspf': {'recommendation': '让胜'},
            # 原始 ELO 总分与校准后的大分结论冲突，分析层应最小校正。
            'dx': {'total_line': 210, 'elo_total': 205,
                   'over_prob': 0.58, 'under_prob': 0.42,
                   'recommendation': '大分'},
        })
        primary = result['score_picks'][0]
        self.assertGreater(primary['home'] + primary['away'], 210)
        self.assertTrue(result['total']['score_consistent'])


if __name__ == '__main__':
    unittest.main()
