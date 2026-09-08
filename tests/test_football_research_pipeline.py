import copy
from datetime import datetime, timedelta, timezone
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from src.football.research import (
    apply_qualified_ml, build_research_variants, kickoff_timestamp,
    outcome_probabilities, project_outcome_probabilities, statistical_candidates,
)
from src.football.research_model import intelligence_features, train_residual, artifact_eligibility
from src.football.research_runtime import team_strength_from_history

NOW = datetime(2026, 9, 8, 6, tzinfo=timezone.utc)
TEAM = {'attack_home': 1.6, 'defense_home': 1.0, 'attack_away': 1.2, 'defense_away': 1.4,
        'home_recent': {'games': 10, 'gf': 16, 'ga': 10, 'form_pts': 19},
        'away_recent': {'games': 10, 'gf': 12, 'ga': 14, 'form_pts': 13}}


class ResearchPipelineTests(unittest.TestCase):
    def test_projection_preserves_score_ratios_and_matches_target(self):
        candidates = [((1, 0), .3), ((2, 0), .2), ((0, 0), .2), ((0, 1), .3)]
        target = {'H': .6, 'D': .25, 'A': .15}
        original = copy.deepcopy(candidates)
        adjusted = project_outcome_probabilities(candidates, target)
        for key, probability in outcome_probabilities(adjusted).items():
            self.assertAlmostEqual(probability, target[key])
        self.assertAlmostEqual(dict(adjusted)[(1, 0)]/dict(adjusted)[(2, 0)], 1.5)
        self.assertEqual(candidates, original)

    def test_ml_requires_real_same_version_shadow_metrics(self):
        candidates = statistical_candidates(TEAM, {})
        response = {'available': True, 'H': .6, 'D': .25, 'A': .15, 'model_version': 'v2'}
        metadata = {'test_count': 220, 'training_cutoff_at': '2026-08-01T00:00:00Z'}
        result, trace = apply_qualified_ml(candidates, response, as_of=NOW, metadata=metadata)
        self.assertIs(result, candidates)
        self.assertFalse(trace['applied'])
        stats = {'overall': {'sample_count': 101, 'base_1x2_logloss': 1, 'base_1x2_brier': .6,
                              'fused_5pct_logloss': .99, 'fused_5pct_brier': .59}}
        result, trace = apply_qualified_ml(candidates, response, as_of=NOW, metadata=metadata, stats=stats)
        self.assertTrue(trace['applied'])
        self.assertEqual(trace['weight'], .05)
        expected = {k: .95*trace['base_probabilities'][k]+.05*response[k] for k in 'HDA'}
        for key, value in outcome_probabilities(result).items():
            self.assertAlmostEqual(value, expected[key])
        result, trace = apply_qualified_ml(candidates, response, as_of=NOW, stats=stats,
                                          metadata={**metadata, 'training_cutoff_at': '2027-01-01T00:00:00Z'})
        self.assertFalse(trace['applied'])

    def test_disabled_and_missing_models_leave_exact_original_distribution(self):
        candidates = statistical_candidates(TEAM, {})
        for response in (None, {'available': False, 'reason': 'feature_contract_mismatch'}):
            result, trace = apply_qualified_ml(candidates, response, as_of=NOW)
            self.assertIs(result, candidates)
            self.assertFalse(trace['applied'])
            self.assertEqual(trace['weight'], 0)

    def test_research_has_real_market_free_a_and_fallback_c_equals_b(self):
        euro = {'raw_odds': {'close': {'home': 2, 'draw': 3.4, 'away': 4}},
                'close': {'home': .48, 'draw': .29, 'away': .23}}
        kwargs = dict(team=TEAM, league_profile={}, euro=euro,
                      asian={'source': 'model_proxy'}, total={'source': 'model_proxy'}, as_of=NOW)
        one = build_research_variants(**kwargs)
        two = build_research_variants(**{**kwargs, 'euro': {
            'raw_odds': {'close': {'home': 10, 'draw': 5, 'away': 1.2}},
            'close': {'home': .09, 'draw': .18, 'away': .73}}})
        self.assertEqual(one['statistical'], two['statistical'])
        self.assertNotEqual(one['market_adjusted']['probabilities'], two['market_adjusted']['probabilities'])
        self.assertEqual(one['agent_adjusted']['score_probabilities'], one['market_adjusted']['score_probabilities'])
        self.assertFalse(one['agent_adjusted']['applied'])
        self.assertEqual(len(one['statistical']['score_probabilities']), 64)
        for key in ('statistical', 'market_adjusted', 'agent_adjusted'):
            self.assertAlmostEqual(sum(one[key]['score_probabilities'].values()), 1)

    def test_absent_team_does_not_fabricate_a_from_market(self):
        result = build_research_variants(team=None, league_profile={}, euro={}, asian={}, total={}, as_of=NOW)
        self.assertIsNone(result['statistical']['probabilities'])
        self.assertEqual(result['statistical']['status'], 'unavailable')

    def test_unknown_kickoff_not_frozen_and_china_schedule_timezone(self):
        self.assertIsNone(kickoff_timestamp({'time': ''}, now=NOW))
        self.assertEqual(kickoff_timestamp({'time': '09-08 18:00'}, now=NOW), NOW+timedelta(hours=4))
        dec = datetime(2026, 12, 31, 8, tzinfo=timezone.utc)
        self.assertEqual(kickoff_timestamp({'time': '01-01 18:00'}, now=dec).year, 2027)

    def test_intelligence_ignores_late_unknown_and_llm_probability_fields(self):
        fact = {'verified': True, 'confirmation_status': 'confirmed', 'team': 'home', 'category': 'injuries',
                'data': {'player': 'p1', 'status': 'injured'}, 'source_url': 'https://club.example/news',
                'published_at': '2026-09-08T04:00:00Z', 'collected_at': '2026-09-08T05:00:00Z'}
        context = {'match_id': 'test', 'kickoff': '2026-09-08T18:00:00Z',
                   'captured_at': '2026-09-08T05:01:00Z', 'as_of': '2026-09-08T05:01:00Z'}
        payload = {**context, 'evidence': [fact, copy.deepcopy(fact)], 'probabilities': {'H': 1}}
        features = intelligence_features(payload, as_of=NOW)
        self.assertAlmostEqual(features['features']['home_unavailable_players'], .1)
        self.assertNotIn('probabilities', features['features'])
        for field, value in [('published_at', None), ('collected_at', '2026-09-08T07:00:00Z'),
                             ('confirmation_status', 'reported')]:
            candidate = {**fact, field: value}
            self.assertEqual(intelligence_features({**context, 'evidence': [candidate]}, as_of=NOW)['known_facts'], 0)
        self.assertEqual(intelligence_features({**payload, 'captured_at': '2026-09-09T00:00:00Z'}, as_of=NOW)['known_facts'], 0)

    def test_residual_rejects_unfrozen_and_missing_results_without_throwing(self):
        report = train_residual([{}, {'frozen': True, 'actual': None}])
        self.assertFalse(report['eligible'])
        self.assertEqual(report['accepted_n'], 0)

    def test_residual_test_results_do_not_select_coefficients_or_weight(self):
        from src.football.research import RESEARCH_VERSION
        rows = []
        first = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        for i in range(90):
            kickoff = first+timedelta(days=i)
            cutoff = kickoff-timedelta(hours=2)
            context = {'match_id': str(i), 'as_of': cutoff.isoformat(), 'captured_at': cutoff.isoformat(),
                       'kickoff': kickoff.isoformat(), 'evidence': [{
                           'verified': True, 'confirmation_status': 'confirmed', 'team': 'home', 'category': 'injuries',
                           'data': {'player': 'p1', 'status': 'injured'}, 'source_url': 'https://club.example/news',
                           'published_at': (cutoff-timedelta(hours=1)).isoformat(), 'collected_at': cutoff.isoformat()}]}
            rows.append({'match_id': str(i), 'as_of': cutoff.isoformat(), 'kickoff_at': kickoff.isoformat(),
                         'settled_at': (kickoff+timedelta(hours=3)).isoformat(), 'frozen': True,
                         'intelligence': context, 'actual': 'A', 'base_probabilities': {'H': .5, 'D': .25, 'A': .25},
                         'result_quality': {'grade': 'high'}, 'baseline_version': RESEARCH_VERSION,
                         'production_model_version': 'p1', 'prediction_logic_version': 'l1'})
        one = train_residual(rows, iterations=15)
        changed = copy.deepcopy(rows)
        for row in changed[72:]:
            row['actual'] = 'H'
        two = train_residual(changed, iterations=15)
        self.assertEqual(one['status'], 'trained_candidate')
        self.assertEqual(one['coefficients'], two['coefficients'])
        self.assertEqual(one['weight'], two['weight'])
        self.assertNotEqual(one['validation']['candidate_metrics'], two['validation']['candidate_metrics'])
        self.assertFalse(one['eligible'])
        self.assertFalse(artifact_eligibility(one, as_of=datetime.now(timezone.utc))[0])
        changed[0]['baseline_version'] = 'incompatible'
        self.assertEqual(train_residual(changed)['status'], 'version_filter_required')
        bad_quality = [{**row, 'exclude_from_calibration': True} for row in rows]
        self.assertEqual(train_residual(bad_quality)['accepted_n'], 0)

    def test_broken_residual_preserves_a_b_and_market_baseline(self):
        euro = {'raw_odds': {'close': {'home': 2, 'draw': 3.4, 'away': 4}},
                'close': {'home': .48, 'draw': .29, 'away': .23}}
        with patch('src.football.research_model.apply_intelligence_residual', side_effect=TypeError('bad model')):
            result = build_research_variants(team=TEAM, league_profile={}, euro=euro,
                                             asian={}, total={}, as_of=NOW)
        self.assertEqual(result['statistical']['status'], 'applied')
        self.assertEqual(result['market_only']['status'], 'applied')
        self.assertEqual(result['agent_adjusted']['score_probabilities'], result['market_adjusted']['score_probabilities'])
        self.assertEqual(result['agent_adjusted']['fallback_reason'], 'residual_error:TypeError')

    def test_official_team_history_uses_known_results_and_exact_names(self):
        rows = []
        for day in range(1, 7):
            rows.append({'home': 'Home', 'away': 'Away', 'match_time': f'2026-08-{day:02d}T04:00:00Z',
                         'result_quality': {'grade': 'high'},
                         'settled_at': f'2026-08-{day:02d}T09:00:00Z', 'settled': True, 'actual_score': '2-1'})
        match = {'home': 'Home', 'away': 'Away', 'time': '2026-09-08 18:00'}
        result = team_strength_from_history(rows, match, as_of=NOW)
        self.assertEqual(result['home_recent']['games'], 6)
        self.assertEqual(result['attack_home'], 2)
        self.assertNotIn('home_xg_last5', result)
        self.assertIsNone(team_strength_from_history(rows, {**match, 'home': 'Other'}, as_of=NOW))
        self.assertIsNone(team_strength_from_history([{**r, 'settled_at': None} for r in rows], match, as_of=NOW))

    def test_temporal_similar_queries_exclude_self_unknown_and_future(self):
        from src.football.similar_market import MatchRecord, SimilarMarketDB
        query = MatchRecord({'as_of': NOW, 'exclude_match_id': 'self'})
        self.assertFalse(SimilarMarketDB._available_for_query(query, MatchRecord({'match_id': 'self', 'available_at': '2026-08-01T00:00:00Z'})))
        self.assertFalse(SimilarMarketDB._available_for_query(query, MatchRecord({'date': '2026-01-01'})))
        self.assertFalse(SimilarMarketDB._available_for_query(query, MatchRecord({'available_at': '2026-10-01T00:00:00Z'})))
        self.assertTrue(SimilarMarketDB._available_for_query(query, MatchRecord({'available_at': '2026-08-01T00:00:00Z'})))

    def test_full_pipeline_keeps_final_market_outputs_on_one_fused_matrix(self):
        from src.football import pipeline
        from src.football.prediction_events import append_prediction_event
        actual_now = datetime.now(timezone.utc)
        kickoff = actual_now + timedelta(days=1)
        match = {'match_id': 'research-integration', 'home': 'Home', 'away': 'Away',
                 'league': 'Test', 'time': kickoff.isoformat(), 'kickoff_at': kickoff.isoformat(),
                 'schedule_source': 'sporttery', 'analysis_source_id_available': False,
                 'lottery_spf_odds': {'胜': 1.7, '平': 3.5, '负': 4.5},
                 'lottery_rqspf_odds': {'让胜': 3.1, '让平': 3.2, '让负': 2.1},
                 'lottery_handicap': -1, 'lottery_spf_available': True,
                 'lottery_rqspf_available': True, 'lottery_offer_matched': True,
                 'lottery_primary_market': 'spf', 'lottery_available_markets': ['spf', 'rqspf']}
        history = SimpleNamespace(records=[], get_record=lambda mid: None)
        saved = []
        def save(**kwargs):
            saved.append(kwargs)
            return {'saved': True, 'persistence_backend': 'test'}
        def fusion(candidates, response, **kwargs):
            base = outcome_probabilities(candidates)
            changed = project_outcome_probabilities(candidates, {'H': .70, 'D': .20, 'A': .10})
            return changed, {'base_probabilities': base, 'probabilities': {'H': .8, 'D': .1, 'A': .1},
                             'loaded': True, 'eligible': True, 'executed': True, 'applied': True,
                             'weight': .05, 'reason': 'test_qualified', 'model_version': 'test-ml',
                             'training_cutoff_at': (actual_now-timedelta(days=30)).isoformat()}
        with ExitStack() as stack:
            for name, value in (('CACHE_AVAILABLE', False), ('BAYESIAN_CALIBRATION_AVAILABLE', False),
                                ('SIMILAR_MARKET_AVAILABLE', False), ('STEAM_MOVE_AVAILABLE', False)):
                stack.enter_context(patch.object(pipeline, name, value))
            stack.enter_context(patch('src.football.research_runtime.completed_intelligence', return_value=None))
            stack.enter_context(patch('src.football.research_runtime.ml_runtime_fusion', side_effect=fusion))
            stack.enter_context(patch('src.football.result_sync.get_history', return_value=history))
            stack.enter_context(patch('src.football.result_sync.get_history_stats', return_value={}))
            stack.enter_context(patch('src.football.result_sync.save_prediction', side_effect=save))
            stack.enter_context(patch.object(pipeline, 'resolve_league_profile', return_value={'avg_goal': 1.35}))
            result = pipeline.analyze_match(match, force_refresh=False)
        self.assertEqual(len(saved), 1)
        data = saved[0]
        for key, probability in {'H': .7, 'D': .2, 'A': .1}.items():
            self.assertAlmostEqual(data['predicted_1x2'][key], probability)
        self.assertEqual(len(data['predicted_scores']), 64)
        self.assertTrue(result['model_status']['ml']['applied'])
        self.assertEqual(result['research']['status'], 'shadow')
        record = {'match_id': match['match_id']}
        frozen = append_prediction_event(record, data['prediction_event'])
        self.assertTrue(frozen['appended'], frozen)
        self.assertEqual(record['prediction_events'][0]['variants']['production']['score_probabilities'], data['predicted_scores'])


class TeamHistoryTemporalContractTests(unittest.TestCase):
    def setUp(self):
        self.match = {'home': 'Home', 'away': 'Away', 'league': 'Test', 'time': '2026-09-08 18:00'}
        self.rows = [{'match_id': f'history-{day}', 'home': 'Home', 'away': 'Away', 'league': 'Test',
                      'match_time': f'2026-08-{day:02d}T04:00:00Z', 'actual_score': '2-1',
                      'actual_result': 'H', 'settled': True, 'result_quality': {'grade': 'high'},
                      'settled_at': f'2026-08-{day:02d}T09:00:00Z'} for day in range(1, 7)]

    def test_result_observation_time_takes_priority_over_old_settlement(self):
        rows = [{**row, 'result_observed_at': '2026-09-09T00:00:00Z'} for row in self.rows]
        self.assertIsNone(team_strength_from_history(rows, self.match, as_of=NOW))
        rows = [{**row, 'settled_at': None, 'result_observed_at': row['settled_at']} for row in self.rows]
        result = team_strength_from_history(rows, self.match, as_of=NOW)
        self.assertEqual(result['home_recent']['games'], 6)

    def test_all_ingest_exclusions_and_actual_result_mismatch_are_shared(self):
        for flag in ('exclude_from_calibration', 'exclude_from_stats', 'skip_training_ingest'):
            with self.subTest(flag=flag):
                rows = [{**row, flag: True} for row in self.rows]
                self.assertIsNone(team_strength_from_history(rows, self.match, as_of=NOW))
        self.assertIsNone(team_strength_from_history(
            [{**row, 'actual_result': 'A'} for row in self.rows], self.match, as_of=NOW))

    def test_day_only_dates_share_ml_convention_but_ambiguous_month_day_is_not_guessed(self):
        rows = [{**row, 'match_time': row['match_time'][:10]} for row in self.rows]
        result = team_strength_from_history(rows, self.match, as_of=NOW)
        self.assertEqual(result['home_recent']['games'], 6)
        self.assertEqual(result['home_recent']['gf'], 12)
        ambiguous = [{**row, 'match_time': '12-20 20:00'} for row in rows]
        self.assertIsNone(team_strength_from_history(ambiguous, self.match, as_of=NOW))

    def test_selected_frozen_kickoff_preserves_historical_year_and_checks_integrity(self):
        from src.football.prediction_events import append_prediction_event
        from src.football.research import variant
        rows = []
        for day in range(20, 26):
            played = datetime(2025, 12, day, 12, tzinfo=timezone.utc)
            observed = played - timedelta(hours=2)
            row = {'match_id': f'last-year-{day}', 'home': 'Home', 'away': 'Away', 'league': 'Test',
                   'match_time': f'12-{day} 20:00'}
            payload = {'as_of': observed.isoformat(), 'kickoff_at': played.isoformat(),
                       'model_version': 'test-model', 'prediction_logic_version': 'test-logic',
                       'variants': {'production': variant([((0, 0), .2), ((0, 1), .2),
                                                          ((1, 0), .5), ((1, 1), .1)],
                                                         source='test', captured_at=observed.isoformat())}}
            saved = append_prediction_event(row, payload, now=observed + timedelta(minutes=1))
            self.assertTrue(saved['appended'], saved)
            row.update(settled=True, actual_score='2-1', actual_result='H', result_quality={'grade': 'high'},
                       result_observed_at=(played + timedelta(hours=3)).isoformat())
            rows.append(row)
        result = team_strength_from_history(rows, self.match, as_of=NOW)
        self.assertEqual(result['home_recent']['games'], 6)
        self.assertEqual(result['attack_home'], 2)
        damaged = copy.deepcopy(rows)
        for row in damaged:
            row['prediction_events'][0]['kickoff_at'] = '2026-12-20T12:00:00Z'
        self.assertIsNone(team_strength_from_history(damaged, self.match, as_of=NOW))

    def test_duplicate_fixture_does_not_inflate_sample_count(self):
        repeated = self.rows + [{**row, 'match_id': 'other-' + row['match_id']} for row in self.rows]
        result = team_strength_from_history(repeated, self.match, as_of=NOW)
        self.assertEqual(result['home_recent']['games'], 6)
        self.assertIsNone(team_strength_from_history(self.rows, self.match, as_of='2026-09-08T06:00:00'))


if __name__ == '__main__':
    unittest.main()
