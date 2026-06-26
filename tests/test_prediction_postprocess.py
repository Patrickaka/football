import unittest
from unittest.mock import patch

import src.football as football


class PredictionPostprocessTests(unittest.TestCase):
    def test_goal_distribution_anchor_moves_mean_toward_total_line(self):
        dist = {1: 0.50, 2: 0.30, 5: 0.20}
        before = sum(goals * prob for goals, prob in dist.items())

        adjusted, meta = football._anchor_goal_dist_to_total_line(dist, {'close_line': 3.0})
        after = sum(goals * prob for goals, prob in adjusted.items())

        self.assertTrue(meta['applied'])
        self.assertGreater(after, before)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_goal_over_under_uses_actual_line(self):
        result = football._goal_over_under_from_line({2: 0.4, 3: 0.6}, {'close_line': 2.5})

        self.assertAlmostEqual(result['over'], 0.6)
        self.assertAlmostEqual(result['under'], 0.4)
        self.assertEqual(result['line'], 2.5)

    def test_goal_distribution_total_movement_raises_high_goals_on_over_signal(self):
        dist = {1: 0.25, 2: 0.35, 3: 0.25, 4: 0.15}
        before = sum(goals * prob for goals, prob in dist.items())

        adjusted, meta = football._adjust_goal_dist_with_total_movement(dist, {
            'open_line': 2.5,
            'close_line': 3.0,
            'open_prob': {'over': 0.48, 'under': 0.52},
            'close_prob': {'over': 0.57, 'under': 0.43},
        })
        after = sum(goals * prob for goals, prob in adjusted.items())

        self.assertTrue(meta['applied'])
        self.assertEqual(meta['direction'], 'over')
        self.assertGreater(after, before)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_goal_distribution_total_movement_lowers_high_goals_on_under_signal(self):
        dist = {1: 0.20, 2: 0.30, 3: 0.30, 4: 0.20}
        before = sum(goals * prob for goals, prob in dist.items())

        adjusted, meta = football._adjust_goal_dist_with_total_movement(dist, {
            'open_line': 3.0,
            'close_line': 2.5,
            'open_prob': {'over': 0.53, 'under': 0.47},
            'close_prob': {'over': 0.44, 'under': 0.56},
        })
        after = sum(goals * prob for goals, prob in adjusted.items())

        self.assertTrue(meta['applied'])
        self.assertEqual(meta['direction'], 'under')
        self.assertLess(after, before)

    def test_goal_distribution_total_movement_marks_conflict(self):
        adjusted, meta = football._adjust_goal_dist_with_total_movement({2: 0.5, 3: 0.5}, {
            'open_line': 2.5,
            'close_line': 3.0,
            'open_prob': {'over': 0.58, 'under': 0.42},
            'close_prob': {'over': 0.48, 'under': 0.52},
        })

        self.assertTrue(meta['conflict'])
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_score_total_line_factor_penalizes_low_score_on_high_line(self):
        low_score_factor = football._score_total_line_factor(0, 0, 3.25)
        aligned_factor = football._score_total_line_factor(2, 1, 3.25)

        self.assertLess(low_score_factor, aligned_factor)

    def test_common_score_overheat_factor_dampens_hot_common_score(self):
        factor = football._common_score_overheat_factor(1, 1, 0.24, 2.5)

        self.assertLess(factor, 1.0)

    def test_score_total_movement_factor_follows_over_signal(self):
        total = {
            'open_line': 2.5,
            'close_line': 3.0,
            'open_prob': {'over': 0.48, 'under': 0.52},
            'close_prob': {'over': 0.58, 'under': 0.42},
        }

        high_factor = football._score_total_movement_factor(2, 2, total)
        low_factor = football._score_total_movement_factor(0, 0, total)

        self.assertGreater(high_factor, 1.0)
        self.assertLess(low_factor, 1.0)

    def test_score_total_movement_factor_follows_under_signal(self):
        total = {
            'open_line': 3.0,
            'close_line': 2.25,
            'open_prob': {'over': 0.54, 'under': 0.46},
            'close_prob': {'over': 0.44, 'under': 0.56},
        }

        low_factor = football._score_total_movement_factor(1, 0, total)
        high_factor = football._score_total_movement_factor(3, 2, total)

        self.assertGreater(low_factor, 1.0)
        self.assertLess(high_factor, 1.0)

    def test_score_distribution_total_movement_tilts_expected_goals_up(self):
        dist = {'0-0': 0.30, '1-1': 0.35, '2-2': 0.35}
        before = sum(sum(map(int, score.split('-'))) * prob for score, prob in dist.items())

        adjusted, meta = football._adjust_score_probs_with_total_movement(dist, {
            'open_line': 2.5,
            'close_line': 3.25,
            'open_prob': {'over': 0.48, 'under': 0.52},
            'close_prob': {'over': 0.58, 'under': 0.42},
        })
        after = sum(sum(map(int, score.split('-'))) * prob for score, prob in adjusted.items())

        self.assertTrue(meta['applied'])
        self.assertEqual(meta['direction'], 'over')
        self.assertGreater(after, before)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_score_distribution_total_movement_tilts_expected_goals_down(self):
        dist = {'1-0': 0.25, '2-1': 0.35, '3-2': 0.40}
        before = sum(sum(map(int, score.split('-'))) * prob for score, prob in dist.items())

        adjusted, meta = football._adjust_score_probs_with_total_movement(dist, {
            'open_line': 3.0,
            'close_line': 2.0,
            'open_prob': {'over': 0.54, 'under': 0.46},
            'close_prob': {'over': 0.42, 'under': 0.58},
        })
        after = sum(sum(map(int, score.split('-'))) * prob for score, prob in adjusted.items())

        self.assertTrue(meta['applied'])
        self.assertEqual(meta['direction'], 'under')
        self.assertLess(after, before)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_team_poisson_lambdas_apply_xg_without_unbound_recent_data(self):
        strength = {
            'attack_home': 1.4,
            'defense_home': 1.1,
            'attack_away': 1.2,
            'defense_away': 1.3,
            'home_xg_last5': 8.0,
            'away_xg_last5': 5.5,
            'home_xga_last5': 6.0,
            'away_xga_last5': 7.0,
            'home_recent': {'games': 5, 'gf': 4, 'ga': 5, 'form_pts': 8},
            'away_recent': {'games': 5, 'gf': 7, 'ga': 6, 'form_pts': 6},
        }

        lam_home, lam_away = football.team_poisson_lambdas(strength, 2.75)

        self.assertGreater(lam_home, 0)
        self.assertGreater(lam_away, 0)
        self.assertAlmostEqual(lam_home + lam_away, 2.75)

    def test_draw_redistribution_uses_handicap_sensitive_cap(self):
        home, draw, away = football._redistribute_draw_probability(0.55, 0.50, 0.15, 1.5)

        self.assertLessEqual(draw, 0.2700001)
        self.assertAlmostEqual(home + draw + away, 1.0)
        self.assertGreater(home, away)

    def test_draw_calibration_keeps_level_ball_draw_range_wider(self):
        _, level_draw, _ = football._heuristic_draw_calibration(
            0.38, 0.30, 0.32,
            asian_handicap=0.0,
            home_draw_rate=0.30,
            away_draw_rate=0.30,
            league_draw_rate=0.28,
        )

        self.assertGreater(level_draw, 0.30)
        self.assertLessEqual(level_draw, 0.42)

    def test_market_data_quality_reduces_conflicted_market_weight(self):
        quality = football._assess_market_data_quality(
            {'handicap': 0.75, 'implied_supremacy': 0.8, 'open_prob': {'home': 0.5}, 'close_prob': {'home': 0.5}},
            {'close': {'home': 0.30, 'draw': 0.30, 'away': 0.40}, 'implied_supremacy': -0.4},
            {'close_line': 2.5, 'open_prob': {'over': 0.5}, 'close_prob': {'over': 0.5}},
        )

        self.assertLess(quality['weight_factor'], 1.0)
        self.assertIn('asian_euro_direction_conflict', quality['reasons'])

    def test_predict_scores_applies_late_market_weight_bias_meta(self):
        with patch('src.football.prediction_policy.get_prediction_policy') as policy:
            policy.return_value = {
                'static_market_cap': 0.15,
                'change_market_cap': 0.15,
                'late_market_weight_bias': 0.06,
                'draw_bias': 1.0,
                'low_score_bias': 1.0,
                'high_score_bias': 1.0,
            }
            _, _, _, meta = football.predict_scores(
                {
                    'handicap': 0.0,
                    'open_handicap': 0.0,
                    'close_prob': {'home': 0.50, 'away': 0.50},
                    'open_prob': {'home': 0.50, 'away': 0.50},
                },
                {
                    'close': {'home': 0.42, 'draw': 0.30, 'away': 0.28},
                    'open': {'home': 0.42, 'draw': 0.30, 'away': 0.28},
                },
                {
                    'close_line': 2.5,
                    'open_line': 2.5,
                    'close_prob': {'over': 0.50, 'under': 0.50},
                    'open_prob': {'over': 0.50, 'under': 0.50},
                },
                current_time_layer='T-15min',
            )

        adjustment = meta['time_layer_market_adjustment']
        self.assertTrue(adjustment['applied'])
        self.assertEqual(adjustment['layer'], 'T-15min')
        self.assertGreater(adjustment['factor'], 1.0)

    def test_predict_scores_reduces_early_market_weight_with_late_bias(self):
        with patch('src.football.prediction_policy.get_prediction_policy') as policy:
            policy.return_value = {
                'static_market_cap': 0.15,
                'change_market_cap': 0.15,
                'late_market_weight_bias': 0.06,
                'draw_bias': 1.0,
                'low_score_bias': 1.0,
                'high_score_bias': 1.0,
            }
            _, _, _, meta = football.predict_scores(
                {
                    'handicap': 0.0,
                    'open_handicap': 0.0,
                    'close_prob': {'home': 0.50, 'away': 0.50},
                    'open_prob': {'home': 0.50, 'away': 0.50},
                },
                {
                    'close': {'home': 0.42, 'draw': 0.30, 'away': 0.28},
                    'open': {'home': 0.42, 'draw': 0.30, 'away': 0.28},
                },
                {
                    'close_line': 2.5,
                    'open_line': 2.5,
                    'close_prob': {'over': 0.50, 'under': 0.50},
                    'open_prob': {'over': 0.50, 'under': 0.50},
                },
                current_time_layer='T-24h',
            )

        adjustment = meta['time_layer_market_adjustment']
        self.assertTrue(adjustment['applied'])
        self.assertEqual(meta['current_time_layer'], 'T-24h')
        self.assertLess(adjustment['factor'], 1.0)

    def test_prediction_policy_exposes_late_market_weight_bias_default(self):
        from src.football.prediction_policy import get_prediction_policy

        policy = get_prediction_policy(league='Test', total_line=2.5, handicap=0.0)

        self.assertIn('late_market_weight_bias', policy)
        self.assertEqual(policy['late_market_weight_bias'], 0.0)

    def test_prediction_policy_accepts_late_market_weight_alias(self):
        from src.football.prediction_policy import _canonical_params

        params = _canonical_params({'late_market_weight': 0.03})

        self.assertIn('late_market_weight_bias', params)
        self.assertEqual(params['late_market_weight_bias'], 0.03)

    def test_half_full_context_aligns_with_score_candidates(self):
        half_full = {
            'probs': [
                {'code': 'AA', 'raw_prob': 0.45, 'probability': 45.0},
                {'code': 'HH', 'raw_prob': 0.30, 'probability': 30.0},
                {'code': 'DD', 'raw_prob': 0.25, 'probability': 25.0},
            ]
        }
        candidates = [((2, 1), 0.40), ((1, 0), 0.30), ((1, 1), 0.10)]

        adjusted = football._adjust_half_full_with_score_context(half_full, candidates)
        dist = adjusted['distribution']

        self.assertGreater(dist['HH'], dist['AA'])
        self.assertTrue(adjusted['score_context']['applied'])

    def test_half_full_market_context_boosts_tempo_and_favorite_paths(self):
        half_full = {
            'probs': [
                {'code': 'DD', 'raw_prob': 0.34, 'probability': 34.0},
                {'code': 'HH', 'raw_prob': 0.30, 'probability': 30.0},
                {'code': 'DH', 'raw_prob': 0.20, 'probability': 20.0},
                {'code': 'AA', 'raw_prob': 0.16, 'probability': 16.0},
            ]
        }

        adjusted = football._adjust_half_full_with_market_context(
            half_full,
            {'handicap': 1.25, 'favor': 'home'},
            {
                'close_line': 3.25,
                'open_prob': {'over': 0.48, 'under': 0.52},
                'close_prob': {'over': 0.58, 'under': 0.42},
            },
        )
        dist = adjusted['distribution']

        self.assertTrue(adjusted['market_context']['applied'])
        self.assertGreater(dist['HH'], half_full['probs'][1]['raw_prob'])
        self.assertLess(dist['DD'], half_full['probs'][0]['raw_prob'])

    def test_half_full_market_context_protects_slow_half_draw_paths(self):
        half_full = {
            'probs': [
                {'code': 'HH', 'raw_prob': 0.32, 'probability': 32.0},
                {'code': 'DH', 'raw_prob': 0.24, 'probability': 24.0},
                {'code': 'DD', 'raw_prob': 0.22, 'probability': 22.0},
                {'code': 'AA', 'raw_prob': 0.22, 'probability': 22.0},
            ]
        }

        adjusted = football._adjust_half_full_with_market_context(
            half_full,
            {'handicap': 0.0, 'favor': 'even'},
            {
                'open_line': 2.75,
                'close_line': 2.0,
                'open_prob': {'over': 0.53, 'under': 0.47},
                'close_prob': {'over': 0.42, 'under': 0.58},
            },
        )
        dist = adjusted['distribution']

        self.assertGreater(dist['DH'], half_full['probs'][1]['raw_prob'])
        self.assertGreater(dist['DD'], half_full['probs'][2]['raw_prob'])
        self.assertLess(dist['AA'], half_full['probs'][3]['raw_prob'])

    def test_prediction_cache_requires_current_logic_version(self):
        current = {
            'model': {
                'prediction_logic_version': football.FOOTBALL_PREDICTION_LOGIC_VERSION,
            }
        }
        stale = {'model': {'prediction_logic_version': 'old'}}
        missing = {'model': {}}

        self.assertTrue(football._is_prediction_cache_current(current))
        self.assertFalse(football._is_prediction_cache_current(stale))
        self.assertFalse(football._is_prediction_cache_current(missing))

    def test_prediction_cache_version_can_fallback_to_status(self):
        cached = {
            'model_status': {
                'prediction_logic_version': football.FOOTBALL_PREDICTION_LOGIC_VERSION,
            }
        }

        self.assertTrue(football._is_prediction_cache_current(cached))

    def test_diversify_score_recommendations_replaces_third_same_pattern(self):
        picked = [
            (1, 0, 0.20, 'home_win_1', 'home_low', 'core'),
            (2, 0, 0.18, 'home_win_2', 'home_low', 'protection'),
            (2, 0, 0.16, 'home_win_2', 'home_low', 'protection'),
        ]
        scored = [
            ((1, 0), 0.20, 0, '', 'home_win_1', 0.20),
            ((2, 0), 0.18, 0, '', 'home_win_2', 0.18),
            ((3, 0), 0.16, 0, '', 'home_win_3', 0.16),
            ((2, 1), 0.15, 0, '', 'home_win_1', 0.15),
            ((1, 1), 0.14, 0, '', 'draw', 0.14),
        ]

        diversified = football._diversify_score_recommendations(
            picked, scored, n=3, favor='home', upset_count=0, max_upsets=0
        )
        patterns = [item[4] for item in diversified]

        self.assertGreater(len(set(patterns)), 1)

    def test_diversify_score_recommendations_replaces_same_result_cluster(self):
        picked = [
            (1, 0, 0.20, 'home_win_1', 'home_low', 'core'),
            (2, 0, 0.18, 'home_win_2', 'home_mid', 'protection'),
            (2, 1, 0.16, 'home_win_1', 'home_mid', 'protection'),
        ]
        scored = [
            ((1, 0), 0.20, 0, '', 'home_win_1', 0.20),
            ((2, 0), 0.18, 0, '', 'home_win_2', 0.18),
            ((2, 1), 0.16, 0, '', 'home_win_1', 0.16),
            ((1, 1), 0.14, 0, '', 'draw', 0.14),
            ((0, 1), 0.13, 0, '', 'away_win_1', 0.13),
        ]

        diversified = football._diversify_score_recommendations(
            picked, scored, n=3, favor='home', upset_count=0, max_upsets=0
        )
        results = ['H' if h > a else 'A' if h < a else 'D' for h, a, *_ in diversified]

        self.assertGreater(len(set(results)), 1)


if __name__ == '__main__':
    unittest.main()
