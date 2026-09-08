"""Exercise current HKJC quotes through analysis, persisted snapshots and reuse."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.football import pipeline, result_sync


_MASS = {(1, 1): .20, (2, 1): .20, (1, 0): .16, (0, 1): .14,
         (3, 1): .12, (3, 2): .10, (4, 2): .08}
MATRIX = [((home, away), _MASS.get((home, away), 0.0))
          for home in range(pipeline.MAX_GOALS + 1) for away in range(pipeline.MAX_GOALS + 1)]


@pytest.fixture
def scenario(monkeypatch):
    # Both observations precede the real clock as well as kickoff. The aware
    # clock is converted to this host's local timezone for the persistence API.
    clock = SimpleNamespace(now=datetime.now(timezone.utc) - timedelta(hours=2))

    class ObservedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return (clock.now.astimezone(tz) if tz is not None
                    else clock.now.astimezone().replace(tzinfo=None))

    history = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
    history.records = []
    writes = []
    history._save_record = Mock(side_effect=lambda row: writes.append(deepcopy(row)) or 'memory')
    history._hydrate_timeline = lambda row: row
    history._persistable = lambda row: row
    monkeypatch.setattr(result_sync, '_global_history', history)
    monkeypatch.setattr(result_sync, 'get_history', lambda: history)
    monkeypatch.setattr(result_sync, 'get_history_stats', lambda: {})
    monkeypatch.setattr(result_sync.repositories, 'football_prediction_load', lambda: [])
    monkeypatch.setattr(result_sync.repositories, 'prediction_record_load', lambda: [])
    monkeypatch.setattr(pipeline, 'datetime', ObservedTime)
    monkeypatch.setattr(result_sync, 'datetime', ObservedTime)
    for flag in ('CACHE_AVAILABLE', 'BAYESIAN_CALIBRATION_AVAILABLE',
                 'DYNAMIC_WEIGHTS_AVAILABLE', 'SIMILAR_MARKET_AVAILABLE', 'STEAM_MOVE_AVAILABLE'):
        monkeypatch.setattr(pipeline, flag, False)
    monkeypatch.setattr(pipeline, 'clear_fetch_cache', lambda: None)
    predict = Mock(side_effect=lambda *args, **kwargs: (deepcopy(MATRIX), 1.6, 1.3, {}))
    monkeypatch.setattr(pipeline, 'predict_scores', predict)
    monkeypatch.setattr(pipeline, 'apply_market_change_prior',
                        lambda scores, *args, **kwargs: (scores, {'used': False}))
    monkeypatch.setattr(pipeline, 'anchor_candidates_to_market',
                        lambda *args, **kwargs: (deepcopy(MATRIX), {}))
    monkeypatch.setattr(pipeline, 'calculate_half_full_time_probs', lambda *args, **kwargs: {})
    monkeypatch.setattr('src.football.history_calibration.get_runtime_history_profile', lambda **kwargs: None)
    monkeypatch.setattr('src.football.ml.load_trained_ml_model', lambda: False)
    monkeypatch.setattr('src.football.market_db.MarketScoreDB', lambda: SimpleNamespace(sample_counts={}))
    monkeypatch.setattr('src.football.bayes_report.load_professional_validation_summary',
                        lambda: {'available': False, 'prediction_ready': False})
    monkeypatch.setattr('src.football.research_runtime.completed_intelligence', lambda *args, **kwargs: None)
    fetch = Mock(side_effect=AssertionError('network is forbidden in this scenario'))
    monkeypatch.setattr(pipeline._fetching_mod, 'fetch', fetch)
    match = {
        'match_id': 'total-history-integration', 'home': 'Home', 'away': 'Away', 'league': '英超',
        'time': (clock.now + timedelta(days=1)).astimezone(
            timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M'),
        'schedule_source': 'sporttery', 'analysis_source_id_available': False,
        'lottery_handicap': 1, 'lottery_spf_odds': {'胜': 2.0, '平': 3.4, '负': 3.5},
        'hkjc_id': 'HK-integration', 'hkjc_updated_at': (clock.now - timedelta(minutes=5)).isoformat(),
        'total_offer_matched': True,
        'total_current': {'line': 2.5, 'over_odds': 2.1, 'under_odds': 1.75},
    }
    yield SimpleNamespace(clock=clock, history=history, writes=writes, predict=predict, match=match, fetch=fetch)
    fetch.assert_not_called()


def test_two_prematch_predictions_reuse_real_persisted_observation_and_keep_final_tails_consistent(scenario):
    first = pipeline.analyze_match(scenario.match, force_refresh=True)
    assert first['model_status']['prediction_saved'] is True
    assert first['total']['history_available'] is False
    assert first['total']['source_event_id'] == scenario.match['hkjc_id']
    assert len(scenario.writes) == 1
    original_snapshot = deepcopy(scenario.writes[0]['market_timeline'][0])
    observed_at = datetime.fromisoformat(original_snapshot['captured_at'])
    assert observed_at.tzinfo is not None and observed_at <= scenario.clock.now
    assert original_snapshot['is_prematch'] is True
    assert original_snapshot['odds']['total']['close_line'] == 2.5
    assert original_snapshot['odds']['total']['source_event_id'] == scenario.match['hkjc_id']

    scenario.clock.now += timedelta(hours=1)
    current = deepcopy(scenario.match)
    current['total_current'] = {'line': 3.5, 'over_odds': 1.75, 'under_odds': 2.1}
    current['hkjc_updated_at'] = (scenario.clock.now - timedelta(minutes=5)).isoformat()
    second = pipeline.analyze_match(current, force_refresh=True)
    total = second['total']
    assert total['history_available'] is True
    assert total['history_source'] == 'local_observations'
    assert total['open_line'] == 2.5 and total['close_line'] == 3.5
    assert total['line_change'] == 1 and total['implied_change'] > 1
    assert total['open_water'] == {'over': 2.1, 'under': 1.75}
    assert total['close_water'] == {'over': 1.75, 'under': 2.1}
    assert total['history_open_captured_at'] == original_snapshot['captured_at']
    assert scenario.predict.call_args.args[2]['history_available'] is True
    assert scenario.writes[-1]['market_timeline'][0] == original_snapshot
    assert scenario.history.records[0]['market_timeline'][0] == original_snapshot

    model = second['model']
    assessment = model['high_score_assessment']
    assert assessment['available'] is True
    for threshold in (4, 5, 6):
        expected = sum(p for (home, away), p in model['candidates'] if home + away >= threshold)
        assert assessment['tail_probabilities'][f'{threshold}+'] == pytest.approx(expected)
    assert scenario.writes[-1]['goal_count']['high_score_assessment'] == assessment
    ordered = sorted(model['candidates'], key=lambda item: (-item[1], item[0]))
    expected_top3 = [score for score, _ in ordered[:3]]
    assert [(row['home'], row['away']) for row in model['top_scores'][:3]] == expected_top3
    assert [row['prob'] for row in model['top_scores'][:3]] == pytest.approx(
        [probability for _, probability in ordered[:3]])


@pytest.mark.parametrize('history_kind', ['naive', 'future', 'absent', 'store_unavailable'])
def test_legacy_or_unavailable_history_preserves_normal_prediction_without_rewriting_old_timestamps(
        scenario, monkeypatch, history_kind):
    first = pipeline.analyze_match(scenario.match, force_refresh=True)
    scenario.clock.now += timedelta(hours=1)
    old = scenario.history.records[0]['market_timeline'][0]
    if history_kind == 'naive':
        old['captured_at'] = datetime.fromisoformat(old['captured_at']).replace(tzinfo=None).isoformat()
    elif history_kind == 'future':
        old['captured_at'] = (scenario.clock.now + timedelta(minutes=5)).isoformat()
    elif history_kind == 'absent':
        scenario.history.records[0]['market_timeline'] = []
    else:
        monkeypatch.setattr(result_sync, 'get_history', Mock(side_effect=OSError('store unavailable')))
    original = deepcopy(old)
    current = deepcopy(scenario.match)
    current['total_current'] = {'line': 3.5, 'over_odds': 1.75, 'under_odds': 2.1}
    second = pipeline.analyze_match(current, force_refresh=True)
    assert second['model_status']['prediction_saved'] is True
    assert second['total']['history_available'] is False
    assert second['total']['open_line'] == second['total']['close_line'] == 3.5
    assert second['total']['implied_change'] == 0
    assert second['model']['candidates']
    assert first['total']['close_line'] == 2.5
    if history_kind != 'absent':
        assert old == original
