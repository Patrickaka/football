"""Exercise the regular five-source pipeline with research enabled, offline."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.football import pipeline
from src.football.prediction_events import append_prediction_event


def test_regular_sources_freeze_independent_variants_with_missing_ml_features():
    kickoff = datetime.now(timezone.utc) + timedelta(days=1)
    match = {'match_id': 'external-research-test', 'home': 'Home', 'away': 'Away',
             'league': 'Test', 'time': kickoff.isoformat(), 'kickoff_at': kickoff.isoformat(),
             'schedule_source': '500', 'analysis_source_id_available': True,
             'lottery_spf_available': True, 'lottery_rqspf_available': True,
             'lottery_offer_matched': True, 'lottery_primary_market': 'spf',
             'lottery_available_markets': ['spf', 'rqspf'], 'lottery_handicap': -1,
             'lottery_spf_odds': {'胜': 1.8, '平': 3.5, '负': 4.2},
             'lottery_rqspf_odds': {'让胜': 3.1, '让平': 3.2, '让负': 2.1}}
    team = {'attack_home': 1.6, 'defense_home': 1.0, 'attack_away': 1.2, 'defense_away': 1.4,
            'home_recent': {'games': 10, 'gf': 16, 'ga': 10, 'form_pts': 19},
            'away_recent': {'games': 10, 'gf': 12, 'ga': 14, 'form_pts': 13}}
    asian = {'handicap': .5, 'home_odds': 1.9, 'away_odds': 1.9}
    total = {'line': 2.5, 'over_odds': 1.9, 'under_odds': 1.9}
    euro = {'home': 1.8, 'draw': 3.5, 'away': 4.2}
    history = SimpleNamespace(records=[], get_record=lambda mid: None)
    saved = []

    def save(**kwargs):
        saved.append(kwargs)
        return {'saved': True, 'persistence_backend': 'test'}

    with ExitStack() as stack:
        for flag in ('CACHE_AVAILABLE', 'BAYESIAN_CALIBRATION_AVAILABLE',
                     'SIMILAR_MARKET_AVAILABLE', 'STEAM_MOVE_AVAILABLE'):
            stack.enter_context(patch.object(pipeline, flag, False))
        for name, value in (
            ('fetch_yazhi', {'open': dict(asian), 'close': dict(asian), 'odds_format': 'decimal'}),
            ('fetch_ouzhi', {'open': dict(euro), 'close': dict(euro), 'series': []}),
            ('fetch_daxiao', {'open': dict(total), 'close': dict(total), 'odds_format': 'decimal'}),
            ('fetch_team_strength', team), ('fetch_single_company_odds', {}),
        ):
            stack.enter_context(patch.object(pipeline._parsing_mod, name, return_value=value))
        stack.enter_context(patch('src.football.research_runtime.completed_intelligence', return_value=None))
        stack.enter_context(patch('src.football.result_sync.get_history', return_value=history))
        stack.enter_context(patch('src.football.result_sync.get_history_stats', return_value={}))
        stack.enter_context(patch('src.football.result_sync.save_prediction', side_effect=save))
        stack.enter_context(patch.object(pipeline, 'resolve_league_profile', return_value={'avg_goal': 1.35}))
        result = pipeline.analyze_match(match)

    assert len(saved) == 1
    assert result['model_status']['prediction_saved'] is True
    assert result['model_status']['ml']['applied'] is False
    assert result['research']['status'] == 'shadow'
    event = saved[0]['prediction_event']
    variants = event['variants']
    assert variants['statistical']['status'] == 'applied'
    assert variants['market_adjusted']['status'] == 'applied'
    assert variants['agent_adjusted']['status'] == 'fallback'
    assert variants['agent_adjusted']['score_probabilities'] == variants['market_adjusted']['score_probabilities']
    assert variants['statistical']['probabilities'] != variants['market_adjusted']['probabilities']
    assert variants['production']['score_probabilities'] == saved[0]['predicted_scores']
    record = {'match_id': match['match_id']}
    audit = append_prediction_event(record, event)
    assert audit['appended'], audit
    assert record['prediction_events'][0]['frozen'] is True
