"""联赛识别、结算学习同步与赛前样本冻结的行为回归。全部使用内存库。"""
from copy import deepcopy
from datetime import datetime
from unittest.mock import patch

import pytest

from src.domain.sports.basketball import factory
from src.domain.sports.basketball.analysis import BasketballAnalyzer
from src.domain.sports.basketball.elo import (
    BasketballELORatingSystem, _get_k_factor, _get_league_weight,
    _league_avg_score,
)
from src.domain.sports.basketball.elo_store import EloStore
from src.domain.sports.basketball.parsing import _date_from_time
from src.domain.sports.basketball.records import PredictionRecorder, PredictionRecordStore
from src.domain.sports.basketball.repository import create_all
from src.foundation.cache import Cache, MemoryBackend
from src.foundation.store import Database, make_engine


MATCH = {
    'id': 'learning-a-b', 'home': 'A', 'away': 'B', 'league': 'NBA',
    'date': '2026-08-27', 'time': '08-27 09:00', 'status': 'not_started',
    'spf_home': 1.50, 'spf_away': 2.50,
    'handicap': -3.5, 'rqspf_home': 1.9, 'rqspf_away': 1.9,
    'total_line': 220.5, 'dx_over': 1.9, 'dx_under': 1.9,
}
NOW = datetime(2026, 8, 27, 8, 0)


@pytest.fixture
def db():
    database = Database(make_engine('sqlite+pysqlite:///:memory:'))
    create_all(database)
    return database


def _analysis(match=None):
    match = deepcopy(match or MATCH)
    analyzer = BasketballAnalyzer()
    return {
        'match': match,
        'spf': analyzer.analyze_spf(match),
        'rqspf': analyzer.analyze_rqspf(match),
        'dx': analyzer.analyze_daxiao(match),
    }


@pytest.mark.parametrize('league', ['WNBA', 'WNBA 常规赛', '2026 WNBA', 'wnba'])
def test_womens_league_does_not_match_nba_substring(league):
    assert _league_avg_score(league) == 82
    assert _get_k_factor(league) == 24
    assert _get_league_weight(league) == 0.85
    prediction = BasketballELORatingSystem().predict_total_score('A', 'B', league)
    assert prediction['expected_total'] == 164.0


@pytest.mark.parametrize('alias', ['美职篮', '美职男篮', '2026 NBA 常规赛'])
def test_analyzer_and_elo_use_the_same_league_alias(alias):
    assert _league_avg_score(alias) == 112
    analyzer = BasketballAnalyzer()
    original = analyzer.analyze_daxiao(MATCH)
    alternate = analyzer.analyze_daxiao({**MATCH, 'league': alias})
    assert alternate == original
    assert analyzer.analyze_spf({**MATCH, 'league': alias}) == analyzer.analyze_spf(MATCH)


def test_decorated_wnba_uses_womens_prior_in_analyzer():
    analyzer = BasketballAnalyzer()
    match = {**MATCH, 'league': 'WNBA', 'total_line': 170.5}
    assert analyzer.analyze_daxiao(match) == analyzer.analyze_daxiao(
        {**match, 'league': 'WNBA 常规赛'})


def test_existing_prediction_service_reads_new_settlements_without_restart(db):
    # 与线上装配一致：结算器、预测器各持有一份学习状态。
    recorder = factory.build_recorder(db, now_fn=lambda: NOW)
    service = factory.build_prediction_service(
        db, transport=lambda *a, **k: '', recorder=recorder,
        today_fn=lambda: '2026-08-27')
    service._schedules = {'500': lambda date: [deepcopy(MATCH)]}
    first = service.generate(use_movement=False)['results'][0]['spf']
    assert first['elo_trust'] == 0
    assert service._analyzer._calibrator.stats == {}

    recorder.settle(MATCH['id'], 110, 90)
    second = service.generate(use_movement=False)['results'][0]['spf']
    assert second['elo_trust'] == 0.05
    assert second['home_prob'] > first['home_prob']
    assert service._analyzer._calibrator.stats['spf|NBA|high']['count'] == 1


def test_predicting_unknown_teams_cannot_overwrite_settled_ratings(db):
    stale_analyzer = factory.build_analyzer(db)
    fresh_elo = BasketballELORatingSystem(store=EloStore(db))
    fresh_elo.update_ratings('Settled A', 'Settled B', 110, 90)
    before = EloStore(db).load()

    stale_analyzer.analyze_spf({**MATCH, 'home': 'New A', 'away': 'New B'})
    assert EloStore(db).load() == before


def test_learning_state_is_read_once_per_uncached_batch(db):
    service = factory.build_prediction_service(
        db, cache=Cache(l1=MemoryBackend(), l2=MemoryBackend()),
        transport=lambda *a, **k: '', now_fn=lambda: NOW,
        today_fn=lambda: '2026-08-27')
    service._schedules = {'500': lambda date: [deepcopy(MATCH)]}
    analyzer = service._analyzer
    with patch.object(analyzer, 'refresh_learning_state',
                      wraps=analyzer.refresh_learning_state) as refresh:
        service.generate(use_movement=False)
        service.generate(use_movement=False)
        assert refresh.call_count == 1


def test_model_versions_do_not_reuse_each_others_cached_predictions(db):
    shared_cache = Cache(l1=MemoryBackend(), l2=MemoryBackend())
    service = factory.build_prediction_service(
        db, cache=shared_cache, transport=lambda *a, **k: '',
        now_fn=lambda: NOW, today_fn=lambda: '2026-08-27')
    service._schedules = {'500': lambda date: [deepcopy(MATCH)]}
    service._version = 'before-fix'
    assert service.generate(use_movement=False)['version'] == 'before-fix'
    service._version = 'after-fix'
    assert service.generate(use_movement=False)['version'] == 'after-fix'


def test_settled_prediction_and_original_handicap_are_frozen(db):
    recorder = factory.build_recorder(db, now_fn=lambda: NOW)
    recorder.save('2026-08-27', [_analysis()], 'original')
    recorder.settle(MATCH['id'], 110, 90)
    before = recorder.get()[0]

    changed = _analysis()
    changed['spf'].update(recommendation='客胜', home_prob=0.2, away_prob=0.8)
    changed['rqspf']['handicap'] = -25.5
    recorder.save('2026-08-27', [changed], 'post-result')
    assert recorder.get()[0] == before


@pytest.mark.parametrize('status,time_text', [
    ('in_progress', '08-27 09:00'),
    ('finished', '08-27 09:00'),
    ('not_started', '08-27 07:00'),
    ('not_started', '2026-08-27 07:00'),
    ('not_started', '07:00'),
])
def test_started_but_unsettled_prediction_is_not_replaced(db, status, time_text):
    store = PredictionRecordStore(db)
    recorder = PredictionRecorder(store, now_fn=lambda: NOW)
    recorder.save('2026-08-27', [_analysis()], 'prematch')
    before = store.load()
    incoming = _analysis({**MATCH, 'status': status, 'time': time_text})
    incoming['spf']['recommendation'] = '客胜'
    recorder.save('2026-08-27', [incoming], 'after-kickoff')
    assert store.load() == before
    # 已经开赛后才首次请求，也不能新增一条伪装成赛前的样本。
    incoming['match']['id'] = 'new-after-kickoff'
    recorder.save('2026-08-27', [incoming], 'after-kickoff')
    assert store.load() == before


def test_prematch_predictions_can_still_refresh(db):
    recorder = PredictionRecorder(PredictionRecordStore(db), now_fn=lambda: NOW)
    recorder.save('2026-08-27', [_analysis()], 'opening')
    changed = _analysis({**MATCH, 'handicap': -4.5})
    recorder.save('2026-08-27', [changed], 'latest-prematch')
    updated = recorder.get()[0]
    assert updated['version'] == 'latest-prematch'
    assert updated['rqspf']['handicap'] == -4.5
    assert updated['result'] is None


@pytest.mark.parametrize('anchor,time_text,expected', [
    ('2026-12-31', '01-01 09:00', '2027-01-01'),
    ('2027-01-01', '12-31 09:00', '2026-12-31'),
    ('2026-12-31', '2028-01-01 09:00', '2028-01-01'),
    ('2027-12-31', '02-29 09:00', '2028-02-29'),
])
def test_schedule_month_day_uses_nearest_legal_year(anchor, time_text, expected):
    assert _date_from_time(time_text, anchor) == expected


def test_record_freeze_handles_new_year_without_an_explicit_match_date(db):
    clock = [datetime(2026, 12, 31, 20)]
    recorder = PredictionRecorder(PredictionRecordStore(db), now_fn=lambda: clock[0])
    match = {**MATCH, 'time': '01-01 09:00'}
    match.pop('date')
    recorder.save('2026-12-31', [_analysis(match)], 'before-new-year')
    assert recorder.get()[0]['version'] == 'before-new-year'
    clock[0] = datetime(2027, 1, 1, 10)
    recorder.save('2026-12-31', [_analysis(match)], 'after-kickoff')
    assert recorder.get()[0]['version'] == 'before-new-year'

    previous_year = {**match, 'id': 'previous-year', 'time': '12-31 09:00'}
    recorder.save('2027-01-01', [_analysis(previous_year)], 'too-late')
    assert len(recorder.get()) == 1


def test_prediction_refresh_failure_keeps_last_successful_state(db):
    writer = factory.build_recorder(db)
    writer._elo.update_ratings('A', 'B', 110, 90)
    writer._calibrator.record('spf', 0.6, True, 'NBA', 'high')
    analyzer = factory.build_analyzer(db)
    before = analyzer.analyze_spf(MATCH)
    ratings = deepcopy(analyzer._elo.ratings)
    stats = deepcopy(analyzer._calibrator.stats)
    with patch.object(analyzer._elo._store, 'load', side_effect=OSError('read failed')):
        with patch.object(analyzer._calibrator._store, 'load',
                          side_effect=OSError('read failed')):
            analyzer.refresh_learning_state()
    assert analyzer._elo.ratings == ratings
    assert analyzer._calibrator.stats == stats
    assert analyzer.analyze_spf(MATCH) == before


def test_calibration_read_failure_preserves_elo_marker_and_retries_once(db):
    recorder = factory.build_recorder(db, now_fn=lambda: NOW)
    recorder._calibrator.record('spf', 0.6, True, 'NBA', 'high')
    previous_stats = deepcopy(recorder._calibrator.stats)
    recorder.save('2026-08-27', [_analysis()], 'prematch')
    with patch.object(recorder._calibrator._store, 'load',
                      side_effect=OSError('temporary read failure')):
        failed = recorder.settle(MATCH['id'], 110, 90)
    assert failed['result']['elo_updated'] is True
    assert failed['result']['calibration_fed'] is False
    assert recorder.get()[0]['result']['elo_updated'] is True
    assert recorder._calibrator.stats == previous_stats
    assert recorder._calibrator._store.load() == previous_stats
    assert recorder._elo.games_played('A') == 1

    recovered = recorder.settle(MATCH['id'], 110, 90)
    assert recovered['result']['calibration_fed'] is True
    assert recovered['calibration_samples'] == 1
    assert recorder._elo.games_played('A') == 1
    assert recorder._calibrator.stats['spf|NBA|high']['count'] == 2
    assert recorder.settle(MATCH['id'], 110, 90)['calibration_samples'] == 0
    assert recorder._calibrator.stats['spf|NBA|high']['count'] == 2


def test_elo_read_failure_skips_writing_and_retry_preserves_other_teams(db):
    recorder = factory.build_recorder(db, now_fn=lambda: NOW)
    recorder._elo.update_ratings('Other A', 'Other B', 100, 90)
    previous_ratings = deepcopy(recorder._elo.ratings)
    recorder.save('2026-08-27', [_analysis()], 'prematch')
    with patch.object(recorder._elo._store, 'load',
                      side_effect=OSError('temporary read failure')):
        failed = recorder.settle(MATCH['id'], 110, 90)
    assert failed['result']['elo_updated'] is False
    assert failed['result']['calibration_fed'] is True
    assert recorder._elo.ratings == previous_ratings
    assert EloStore(db).load()['ratings'] == previous_ratings

    recovered = recorder.settle(MATCH['id'], 110, 90)
    assert recovered['result']['elo_updated'] is True
    assert recovered['calibration_samples'] == 0
    assert recorder._elo.games_played('A') == 1
    assert recorder._elo.ratings['Other A'] == previous_ratings['Other A']
    assert recorder._calibrator.stats['spf|NBA|high']['count'] == 1
