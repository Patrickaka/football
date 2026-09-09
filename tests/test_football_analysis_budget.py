"""Bounded waits/late-result guards without contacting any upstream."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
import time
from unittest.mock import patch

import pytest

from src.football import analysis_budget as budget, fetching, parsing, pipeline


def test_nested_limits_never_extend_the_parent_and_reset_afterwards():
    with patch.object(budget.time, 'monotonic', return_value=10) as clock:
        with budget.limit(2):
            with budget.limit(100):
                assert budget.remaining() == 2
                clock.return_value = 12
                with pytest.raises(budget.AnalysisTimeout):
                    budget.check()
        assert budget.remaining(20) == 20


def test_child_contexts_share_deadline_without_sharing_context_objects():
    with ThreadPoolExecutor(max_workers=2) as pool, budget.limit(1):
        tasks = [budget.submit(pool, budget.remaining) for _ in range(2)]
        assert all(0 < task.result() <= 1 for task in tasks)


def test_expired_url_and_semaphore_wait_do_not_start_a_network_request():
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    with patch.object(fetching, '_fetch_semaphore', semaphore), \
         patch.object(fetching, '_fetch_raw') as raw, \
         patch.object(fetching, '_fetch_cache_get', return_value=None):
        started = time.monotonic()
        with budget.limit(.02), pytest.raises(budget.AnalysisTimeout):
            fetching.fetch('http://example.test/match')
        assert time.monotonic() - started < 1
        raw.assert_not_called()
    semaphore.release()


def test_late_fetch_result_is_not_cached():
    with patch.object(fetching, '_fetch_cache_get', return_value=None), \
         patch.object(fetching, '_fetch_raw', side_effect=lambda *_: time.sleep(.03) or 'late'), \
         patch.object(fetching, '_fetch_cache_set') as save:
        with budget.limit(.01), pytest.raises(budget.AnalysisTimeout):
            fetching.fetch('http://example.test/late')
        save.assert_not_called()


def test_cooldown_and_rate_waits_respect_remaining_time():
    for function, attr in ((fetching._await_fetch_throttle, '_fetch_throttle_until'),
                           (fetching._await_rate_slot, '_fetch_next_slot')):
        with patch.object(fetching, attr, time.time() + 5), budget.limit(.02):
            started = time.monotonic()
            with pytest.raises(budget.AnalysisTimeout):
                function()
            assert time.monotonic() - started < 1


def test_socket_timeout_uses_remaining_budget_and_trickle_cannot_extend_it():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return b'content'

    with patch.object(fetching.urllib.request, 'build_opener') as build_opener:
        open_url = build_opener.return_value.open
        open_url.return_value = Response()
        with budget.limit(.1):
            assert fetching._fetch_once('http://example.test', 'utf-8') == 'content'
        assert 0 < open_url.call_args.kwargs['timeout'] <= .1

    class Trickle:
        reads = 0
        def read1(self, size):
            self.reads += 1
            time.sleep(.01)
            return b'x'

    response = Trickle()
    with budget.limit(.025), pytest.raises(budget.AnalysisTimeout):
        fetching._read_response(response)
    assert response.reads < 8


def test_singleflight_waiter_can_timeout_without_cancelling_owner():
    entered, release = threading.Event(), threading.Event()
    match = {'match_id': 'budget-owner', 'home': 'H', 'away': 'A'}
    def compute(*_, **__):
        entered.set()
        release.wait(2)
        return {'match_id': match['match_id']}
    results = []
    with patch.object(pipeline, '_analyze_match_impl', side_effect=compute) as calculate:
        owner = threading.Thread(target=lambda: results.append(pipeline.analyze_match(match)))
        owner.start()
        try:
            assert entered.wait(1)
            with budget.limit(.02), pytest.raises(budget.AnalysisTimeout):
                pipeline.analyze_match(match)
            assert calculate.call_count == 1
        finally:
            release.set()
            owner.join(2)
    assert results == [{'match_id': match['match_id']}]


def test_expired_odds_children_are_drained_without_publishing_a_prediction():
    def slow(*_, **__):
        time.sleep(.06)
        budget.check()
        return {}
    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline, 'resolve_league_profile', return_value={}))
        stack.enter_context(patch.object(pipeline, 'CACHE_AVAILABLE', False))
        save_cache = stack.enter_context(patch.object(pipeline, 'set_cache'))
        for name in ('fetch_yazhi', 'fetch_ouzhi', 'fetch_daxiao',
                     'fetch_team_strength', 'fetch_single_company_odds'):
            stack.enter_context(patch.object(parsing, name, side_effect=slow))
        with budget.limit(.02), pytest.raises(budget.AnalysisTimeout):
            pipeline.analyze_match({'match_id': 'budget-odds', 'home': 'H', 'away': 'A'})
        save_cache.assert_not_called()
    assert not any(thread.is_alive() and thread.name.startswith('FootballOdds')
                   for thread in threading.enumerate())


def test_only_fifth_odds_child_times_out_and_is_still_drained():
    finished = threading.Event()

    def fifth(*_):
        time.sleep(.08)
        finished.set()
        return {}

    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline, 'resolve_league_profile', return_value={}))
        stack.enter_context(patch.object(pipeline, 'CACHE_AVAILABLE', False))
        for name in ('fetch_yazhi', 'fetch_ouzhi', 'fetch_daxiao', 'fetch_team_strength'):
            stack.enter_context(patch.object(parsing, name, return_value={}))
        for name in ('analyze_asian', 'analyze_euro', 'analyze_total'):
            stack.enter_context(patch.object(pipeline, name, return_value={}))
        stack.enter_context(patch.object(parsing, 'fetch_single_company_odds', side_effect=fifth))
        with budget.limit(.03), pytest.raises(budget.AnalysisTimeout):
            pipeline.analyze_match({'match_id': 'budget-fifth', 'home': 'H', 'away': 'A'})
        assert finished.is_set()
    assert not any(thread.is_alive() and thread.name.startswith('FootballOdds')
                   for thread in threading.enumerate())


def test_cache_hit_does_not_republish_after_persistence_uses_up_the_budget():
    from src.football import result_sync
    cached = {'model': {'prediction_logic_version': pipeline.FOOTBALL_PREDICTION_LOGIC_VERSION,
                        'candidates': [((1, 0), 1.0)]},
              'lottery': {'offer_matched': False},
              'model_status': {'prediction_saved': False}}
    with patch.object(budget.time, 'monotonic', return_value=0) as clock:
        def save(**_):
            clock.return_value = 2
            return {'saved': True}
        with patch.object(pipeline, 'resolve_league_profile', return_value={}), \
             patch.object(pipeline, 'CACHE_AVAILABLE', True), \
             patch.object(pipeline, 'get_cache', return_value=cached), \
             patch.object(pipeline, 'set_cache') as publish, \
             patch.object(result_sync, 'save_prediction', side_effect=save):
            with budget.limit(1), pytest.raises(budget.AnalysisTimeout):
                pipeline.analyze_match({'match_id': 'late-save'})
            publish.assert_not_called()
            assert cached['model_status']['prediction_saved'] is False


@pytest.mark.parametrize('change,refresh', [({'lottery_handicap': 2}, False), ({}, True)])
def test_inflight_old_inputs_do_not_override_changed_market_or_force_refresh(change, refresh):
    entered, release = threading.Event(), threading.Event()
    initial = {'match_id': 'different-context', 'lottery_handicap': 1}
    def calculate(match, force_refresh=False):
        if not force_refresh and match == initial:
            entered.set()
            release.wait(2)
        return {'handicap': match['lottery_handicap'], 'refresh': force_refresh}
    with patch.object(pipeline, '_analyze_match_impl', side_effect=calculate) as analyze:
        owner = threading.Thread(target=lambda: pipeline.analyze_match(initial))
        owner.start()
        try:
            assert entered.wait(1)
            result = pipeline.analyze_match({**initial, **change}, force_refresh=refresh)
            assert result == {'handicap': change.get('lottery_handicap', 1), 'refresh': refresh}
            assert analyze.call_count == 2
        finally:
            release.set()
            owner.join(2)
