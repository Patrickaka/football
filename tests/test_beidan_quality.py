import unittest
from unittest.mock import patch

from src.beidan import (
    apply_beidan_history_calibration,
    analyze_bifen,
    analyze_rqspf,
    analyze_spf,
    analyze_zjq,
    assess_recommendation_quality,
    assess_score_consistency,
    assess_upset_risk,
    build_zjq_group_recommendation,
    apply_beidan_joint_market_state,
    build_beidan_joint_market_state,
    build_water_market_prediction,
    enhance_scores_with_cs,
    generate_beidan_recommendations,
    parse_beidan_handicap,
    pick_upset_scores,
    rqspf_probs_from_score_probs,
    save_beidan_prediction_snapshot,
)


class BeidanQualityTests(unittest.TestCase):
    def test_water_market_prediction_uses_one_matrix_for_all_markets(self):
        result = build_water_market_prediction({
            'score_probs': [[2, 0, 0.40], [1, 0, 0.20], [1, 1, 0.25], [0, 1, 0.15]],
            'raw_probabilities': {'胜': 0.50, '平': 0.25, '负': 0.25},
            'joint_market_state': {'applied': True, 'direction_signal': 0.4},
        }, -1)

        self.assertTrue(result['available'])
        self.assertTrue(result['asian_adjusted'])
        self.assertEqual(result['spf']['prediction'], '胜')
        self.assertEqual(result['rqspf']['prediction'], '让胜')
        self.assertEqual(result['goals']['prediction'], '2')
        self.assertEqual(result['evidence']['euro_prediction'], '胜')
        self.assertEqual(result['evidence']['asian_direction'], '主队增强')
        self.assertFalse(result['evidence']['conflict'])
        self.assertAlmostEqual(sum(result['spf']['probabilities'].values()), 1.0)
        self.assertAlmostEqual(sum(result['rqspf']['probabilities'].values()), 1.0)

    def test_joint_market_state_links_home_backing_and_over_move(self):
        asian = {'history': [
            {'handicap': 0.5, 'home_odds': 0.98, 'away_odds': 0.86},
            {'handicap': 1.0, 'home_odds': 0.82, 'away_odds': 1.02},
        ]}
        goals = {'history': [
            {'line': '2.5', 'over_odds': 1.02, 'under_odds': 0.82},
            {'line': '3.0', 'over_odds': 0.82, 'under_odds': 1.02},
        ]}
        scores = {(1, 0): .25, (0, 1): .25, (1, 1): .25, (3, 1): .25}

        adjusted, state = apply_beidan_joint_market_state(scores, asian, goals)

        self.assertTrue(state['applied'])
        self.assertGreater(state['home_win_after'], state['home_win_before'])
        self.assertGreater(state['expected_goals_after'], state['expected_goals_before'])
        self.assertAlmostEqual(sum(adjusted.values()), 1.0)

    def test_joint_market_state_marks_ou_conflict(self):
        state = build_beidan_joint_market_state(None, {'history': [
            {'line': '2.5', 'over_odds': 1.00, 'under_odds': .82},
            {'line': '3.0', 'over_odds': 1.12, 'under_odds': .76},
        ]})
        self.assertTrue(state['conflict'])
        self.assertEqual(state['agreement_factor'], .40)

    def test_narrow_spf_call_is_marked_as_split(self):
        quality = assess_recommendation_quality({'胜': 0.36, '平': 0.34, '负': 0.30}, '胜')

        self.assertEqual(quality['level'], 'split')
        self.assertTrue(quality['avoid_single'])
        self.assertEqual([x['option'] for x in quality['top2']], ['胜', '平'])

    def test_score_conflict_downgrades_quality(self):
        scores = [
            {'score': '0-1', 'probability': 0.16, 'home_goals': 0, 'away_goals': 1},
            {'score': '1-1', 'probability': 0.12, 'home_goals': 1, 'away_goals': 1},
            {'score': '0-2', 'probability': 0.08, 'home_goals': 0, 'away_goals': 2},
        ]
        consistency = assess_score_consistency(scores, '胜')
        quality = assess_recommendation_quality(
            {'胜': 0.48, '平': 0.28, '负': 0.24},
            '胜',
            {'score_consistency': consistency}
        )

        self.assertTrue(consistency['conflict'])
        self.assertEqual(quality['level'], 'split')
        self.assertTrue(quality['avoid_single'])

    def test_analyze_spf_uses_adjusted_probabilities_for_quality(self):
        match = {
            'id': 'm1',
            'num': '001',
            'home': 'A',
            'away': 'B',
            'league': '',
            'time': '20:00',
            'handicap': 0,
        }
        asian_data = {
            'history': [
                {'home_odds': 0.82, 'away_odds': 0.96},
                {'home_odds': 0.78, 'away_odds': 1.00},
            ]
        }

        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.7, 'draw': 3.0, 'away': 2.6}):
            result = analyze_spf(match, asian_data=asian_data)

        self.assertEqual(result['prediction'], '胜')
        self.assertIn('quality', result)
        self.assertIn(result['quality']['level'], {'medium', 'split', 'low', 'strong'})
        self.assertIn('raw_probabilities', result)
        self.assertIn('score_consistency', result)
        self.assertIn('history_calibration', result)

    def test_spf_market_history_is_applied_only_by_joint_matrix(self):
        match = {
            'id': 'm-single-weight', 'num': '001', 'home': 'A', 'away': 'B',
            'league': '', 'time': '20:00', 'handicap': 0,
        }
        asian_data = {'history': [
            {'handicap': 0, 'home_odds': 0.96, 'away_odds': 0.86},
            {'handicap': 0.25, 'home_odds': 0.80, 'away_odds': 1.02},
        ]}
        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.0, 'draw': 3.2, 'away': 3.6}), \
             patch('src.beidan.adjust_probs_by_asian', side_effect=AssertionError('duplicate adjustment')):
            result = analyze_spf(match, asian_data=asian_data)

        self.assertNotIn('error', result)
        self.assertTrue(result['joint_market_state']['applied'])
        self.assertTrue(result['asian_adjusted'])

    def test_parse_beidan_handicap_accepts_parenthesized_values(self):
        self.assertEqual(parse_beidan_handicap('(-1)'), -1.0)
        self.assertEqual(parse_beidan_handicap('(+2)'), 2.0)
        self.assertIsNone(parse_beidan_handicap(''))

    def test_rqspf_probs_from_score_probs_applies_home_handicap(self):
        probs, meta = rqspf_probs_from_score_probs({
            (1, 0): 0.4,
            (2, 0): 0.3,
            (0, 1): 0.3,
        }, -1)

        self.assertTrue(meta['available'])
        self.assertAlmostEqual(probs['让平'], 0.4)
        self.assertAlmostEqual(probs['让胜'], 0.3)
        self.assertAlmostEqual(probs['让负'], 0.3)
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_analyze_rqspf_returns_real_probabilities(self):
        match = {
            'id': 'm1',
            'num': '001',
            'home': 'A',
            'away': 'B',
            'league': '',
            'time': '20:00',
            'handicap': '(-1)',
        }

        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 1.8, 'draw': 3.4, 'away': 4.2}):
            result = analyze_rqspf(match)

        self.assertNotIn('error', result)
        self.assertEqual(set(result['probabilities']), {'让胜', '让平', '让负'})
        self.assertIn(result['prediction'], {'让胜', '让平', '让负'})
        self.assertIn('quality', result)
        self.assertGreater(len(result['scores']), 0)
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=6)

    def test_history_calibration_lifts_underestimated_settled_outcome(self):
        records = []
        for idx in range(10):
            records.append({
                'settled': True,
                'league': 'L',
                'actual': {'score': '1-0'},
                'spf': {'probabilities': {'胜': 0.30, '平': 0.35, '负': 0.35}},
            })

        with patch('src.beidan._load_beidan_history', return_value=records):
            adjusted, meta = apply_beidan_history_calibration(
                {'胜': 0.34, '平': 0.33, '负': 0.33},
                'spf',
                league='L'
            )

        self.assertTrue(meta['applied'])
        self.assertGreater(adjusted['胜'], 0.34)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=6)

    def test_rqspf_history_calibration_fires_with_settled_records(self):
        records = []
        for _ in range(12):
            records.append({
                'settled': True,
                'league': 'L',
                'handicap': '(-1)',
                'actual': {'score': '2-0'},
                'rqspf': {'probabilities': {'让胜': 0.40, '让平': 0.30, '让负': 0.30}},
            })

        with patch('src.beidan._load_beidan_history', return_value=records):
            adjusted, meta = apply_beidan_history_calibration(
                {'让胜': 0.38, '让平': 0.31, '让负': 0.31},
                'rqspf',
                league='L'
            )

        self.assertTrue(meta['applied'])
        self.assertGreater(adjusted['让胜'], 0.38)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=6)

    def test_upset_risk_flags_weak_favorite(self):
        # 弱热门（三路接近）→ 应判定爆冷预警
        risk = assess_upset_risk({'胜': 0.36, '平': 0.35, '负': 0.29})
        self.assertTrue(risk['alert'])
        self.assertIn(risk['level'], {'medium', 'high'})
        self.assertEqual(risk['favorite'], '胜')
        self.assertAlmostEqual(risk['upset_prob'], 0.64, places=2)

    def test_upset_risk_ignores_strong_favorite(self):
        # 强热门 → 不预警
        risk = assess_upset_risk({'胜': 0.70, '平': 0.18, '负': 0.12})
        self.assertFalse(risk['alert'])
        self.assertEqual(risk['level'], 'low')

    def test_pick_upset_scores_returns_contrarian_scores(self):
        # 热门为主胜时，爆冷比分候选只能是平/负方向
        matrix = {
            (2, 0): 0.14, (1, 0): 0.12, (1, 1): 0.11,
            (0, 0): 0.08, (1, 2): 0.06, (0, 1): 0.05,
        }
        cands = pick_upset_scores(matrix, '胜', top_n=2)
        self.assertEqual(len(cands), 2)
        for c in cands:
            self.assertIn(c['result'], {'平', '负'})
        # 概率应降序
        self.assertGreaterEqual(cands[0]['probability'], cands[1]['probability'])

    def test_analyze_bifen_attaches_upset_block(self):
        match = {
            'id': 'mU', 'num': '001', 'home': 'A', 'away': 'B',
            'league': '英超', 'time': '20:00', 'handicap': 0,
        }
        # 弱热门赔率 → 触发预警且给出爆冷候选
        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.6, 'draw': 3.2, 'away': 2.7}):
            result = analyze_bifen(match, goals_data={'history': [{'over_odds': 1.9, 'under_odds': 1.9}]})
        self.assertIn('upset', result)
        upset = result['upset']
        self.assertTrue(upset['alert'])
        self.assertIn(upset['level'], {'medium', 'high'})
        self.assertTrue(upset['candidates'])

    def test_analyze_spf_attaches_upset_block(self):
        match = {
            'id': 'mS', 'num': '007', 'home': 'A', 'away': 'B',
            'league': '英超', 'time': '20:00', 'handicap': 0,
        }
        # 默认面板走 spf；弱热门赔率应触发爆冷预警并给出反向比分候选
        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.6, 'draw': 3.2, 'away': 2.7}):
            result = analyze_spf(match)
        self.assertIn('upset', result)
        upset = result['upset']
        self.assertTrue(upset['alert'])
        self.assertIn(upset['level'], {'medium', 'high'})
        self.assertTrue(upset['candidates'])
        # 候选比分方向应与热门相反
        fav = upset['favorite']
        for c in upset['candidates']:
            self.assertNotEqual(c['result'], fav)

    def test_upset_watch_falls_back_to_spf(self):
        match = {
            'id': 'mW', 'num': '009', 'date': '2026-07-17', 'home': 'A', 'away': 'B',
            'league': '英超', 'time': '20:00', 'handicap': 0, 'source': 'okooo',
        }
        meta = {'date': '2026-07-17', 'source': 'okooo', 'attempts': []}
        with patch('src.beidan._fetch_beidan_matches_with_fallback', return_value=([match], meta)), \
             patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.6, 'draw': 3.2, 'away': 2.7}), \
             patch('src.beidan.fetch_okooo_asian_history', return_value=None), \
             patch('src.beidan.fetch_okooo_goals_history', return_value=None), \
             patch('src.beidan.fetch_okooo_cs_history', return_value=None):
            result = generate_beidan_recommendations(
                date='2026-07-17', bet_types=['spf'], source='okooo', save_history=False
            )
        self.assertIn('upset_watch', result)
        self.assertTrue(any(w['num'] == '009' for w in result['upset_watch']))

    def test_history_calibration_waits_for_enough_samples(self):
        records = [{
            'settled': True,
            'actual': {'score': '0-1'},
            'spf': {'probabilities': {'胜': 0.40, '平': 0.30, '负': 0.30}},
        }]

        with patch('src.beidan._load_beidan_history', return_value=records):
            adjusted, meta = apply_beidan_history_calibration(
                {'胜': 0.34, '平': 0.33, '负': 0.33},
                'spf'
            )

        self.assertFalse(meta['applied'])
        self.assertEqual(adjusted['胜'], 0.34)

    def test_enhance_scores_with_cs_keeps_dict_shape(self):
        score_prediction = {
            'top3': [
                {'score': '1-0', 'probability': 0.12, 'home_goals': 1, 'away_goals': 0},
                {'score': '1-1', 'probability': 0.10, 'home_goals': 1, 'away_goals': 1},
                {'score': '0-0', 'probability': 0.08, 'home_goals': 0, 'away_goals': 0},
            ]
        }
        cs_history = [
            {'score': '2-1', 'odds': 6.0},
            {'score': '1-0', 'odds': 5.0},
        ]

        enhanced = enhance_scores_with_cs(score_prediction, cs_history)

        self.assertIsInstance(enhanced['top3'][0], dict)
        self.assertIn('score', enhanced['top3'][0])
        self.assertIn('probability', enhanced['top3'][0])

    def test_analyze_zjq_blends_market_goal_odds(self):
        match = {
            'id': 'm1',
            'num': '001',
            'home': 'A',
            'away': 'B',
            'league': '英超',
            'time': '20:00',
        }
        zjq_odds = {'m1': {'0': 15.0, '1': 7.0, '2': 3.0, '3': 3.2, '4': 5.0, '5': 9.0, '6': 18.0, '7+': 28.0}}

        with patch('src.beidan.fetch_ouzhi_odds', return_value={'home': 2.0, 'draw': 3.3, 'away': 3.6}):
            result = analyze_zjq(match, zjq_odds=zjq_odds)

        self.assertTrue(result['market_adjusted'])
        self.assertIn('odds', result)
        self.assertIn('quality', result)
        self.assertIn('goal_groups', result)
        self.assertIn(result['goal_groups']['primary']['key'], {'small', 'middle', 'big'})
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=6)

    def test_zjq_group_recommendation_prefers_best_band(self):
        groups = build_zjq_group_recommendation({
            '0': 0.02, '1': 0.08, '2': 0.16,
            '3': 0.24, '4': 0.22, '5': 0.14, '6': 0.08, '7+': 0.06,
        })

        self.assertEqual(groups['primary']['key'], 'big')
        self.assertEqual(groups['primary']['options'], ['3', '4', '5', '6', '7+'])

    def test_save_prediction_snapshot_upserts_by_match_key(self):
        saved_payloads = []
        result = {
            'source': 'okooo',
            'recommendations': [{
                'date': '2026-07-08',
                'num': '001',
                'time': '20:00',
                'league': 'L',
                'home': 'A',
                'away': 'B',
                'spf': {'prediction': '胜', 'confidence': 0.48, 'quality': {'level': 'strong'}},
                'zjq': {
                    'prediction': '3',
                    'confidence': 0.24,
                    'quality': {'level': 'medium'},
                    'goal_groups': {'primary': {'key': 'big'}},
                },
            }]
        }

        with patch('src.beidan._load_beidan_history', return_value=[]), \
                patch('src.beidan._save_beidan_history', side_effect=lambda rows: saved_payloads.append(rows)):
            summary = save_beidan_prediction_snapshot(result)

        self.assertEqual(summary['saved'], 1)
        self.assertEqual(saved_payloads[0][0]['key'], '2026-07-08|001|A|B')
        self.assertEqual(saved_payloads[0][0]['zjq']['goal_groups']['primary']['key'], 'big')

    def test_generate_recommendations_falls_back_to_okooo_source(self):
        fallback_match = {
            'id': 'm1',
            'num': '001',
            'date': '2026-07-08',
            'time': '20:00',
            'league': 'L',
            'home': 'A',
            'away': 'B',
            'handicap': '(-1)',
        }

        with patch('src.beidan.fetch_beidan_schedule', return_value=[]), \
                patch('src.beidan.fetch_okooo_schedule', return_value=[fallback_match]):
            result = generate_beidan_recommendations(
                date='2026-07-08',
                bet_types=[],
                source='dc',
                save_history=False
            )

        self.assertNotIn('error', result)
        self.assertEqual(result['source'], 'okooo')
        self.assertTrue(result['match_fetch']['source_fallback'])
        self.assertEqual(result['total_matches'], 1)

    def test_generate_recommendations_tries_next_dates_for_default_date(self):
        fallback_match = {
            'id': 'm1',
            'num': '001',
            'date': '2026-07-09',
            'time': '20:00',
            'league': 'L',
            'home': 'A',
            'away': 'B',
            'handicap': '(-1)',
        }

        def fake_schedule(date):
            return [fallback_match] if date == '2026-07-09' else []

        with patch('src.beidan.time.strftime', return_value='2026-07-08'), \
                patch('src.beidan.fetch_okooo_schedule', side_effect=fake_schedule):
            result = generate_beidan_recommendations(
                date=None,
                bet_types=[],
                source='okooo',
                save_history=False
            )

        self.assertNotIn('error', result)
        self.assertEqual(result['date'], '2026-07-09')
        self.assertTrue(result['match_fetch']['date_fallback'])


if __name__ == '__main__':
    unittest.main()
