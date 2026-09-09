"""Offline tests for bounded, incremental football match analysis jobs."""
from collections import Counter
import threading
import time

import pytest

from src.api.runtime import football_analysis_jobs as jobs_module
from src.api.runtime.football_analysis_jobs import (
    FootballAnalysisJobManager, FootballAnalysisManagerClosed, FootballAnalysisQueueFull,
    get_manager, get_football_analysis_job_manager,
)


class Clock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def managers():
    created, releases = [], []
    def make(**kwargs):
        options = {'match_timeout': 2., 'job_timeout': 10., 'monitor_interval': .005}
        options.update(kwargs)
        manager = FootballAnalysisJobManager(**options)
        created.append(manager)
        return manager
    def event():
        release = threading.Event()
        releases.append(release)
        return release
    make.event = event
    yield make
    for release in releases:
        release.set()
    for manager in created:
        manager.close(wait=True)


def matches(*ids):
    return [{'match_id': str(number), 'home': f'Home {number}', 'away': 'Visitor'} for number in ids]


def eventually(read, predicate, timeout=2.):
    deadline = time.monotonic() + timeout
    while True:
        value = read()
        if predicate(value):
            return value
        if time.monotonic() >= deadline:
            pytest.fail(f'timed out waiting for test condition: {value!r}')
        time.sleep(.003)


def finished(manager, job_id):
    return eventually(lambda: manager.get(job_id), lambda job: job['status'] in ('completed', 'failed'))


def test_constructor_and_poll_start_no_threads(managers):
    manager = managers()
    assert manager._workers == []
    assert manager._monitor is None
    assert manager.get('missing') is None
    assert manager._workers == []
    assert manager._monitor is None


def test_short_manager_accessor_is_a_compatible_lazy_alias():
    # Do not construct a process singleton in this unit test.
    assert get_manager is get_football_analysis_job_manager


def test_forty_matches_use_only_two_real_workers_and_isolate_item_errors(managers):
    manager = managers()
    lock = threading.Lock()
    ids, active, maximum = set(), 0, 0
    def analyze(match):
        nonlocal active, maximum
        with lock:
            ids.add(threading.get_ident())
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(.003)
            if match['match_id'] == '7':
                raise RuntimeError('one source failed')
            if match['match_id'] == '8':
                raise TimeoutError('cooperative budget exhausted')
            return {'match_id': match['match_id'], 'probabilities': {'home': .5}}
        finally:
            with lock:
                active -= 1
    submitted = manager.submit(matches(*range(40)), analyze)
    result = finished(manager, submitted['job_id'])
    assert result['status'] == 'completed'
    assert (result['total'], result['completed'], result['succeeded'], result['failed']) == (40, 40, 38, 2)
    assert maximum == len(ids) == 2
    assert len({row['index'] for row in result['results']}) == 40
    entries = {row['match_id']: row for row in result['results']}
    assert entries['7']['status'] == 'failed'
    assert entries['8']['status'] == 'timed_out'
    assert result['created_at'] <= result['started_at'] <= result['finished_at']


def test_fast_matches_publish_while_one_slow_match_is_still_running(managers):
    manager = managers()
    release = managers.event()
    slow_started = threading.Event()
    def analyze(match):
        if match['match_id'] == '0':
            slow_started.set()
            release.wait()
        return {'id': match['match_id']}
    first = manager.submit(matches(*range(40)), analyze)
    assert slow_started.wait(1)
    partial = eventually(lambda: manager.get(first['job_id']), lambda job: job['completed'] == 39)
    assert partial['status'] == 'running'
    assert all(row['match_id'] != '0' for row in partial['results'])
    revision = partial['revision']
    assert manager.get(first['job_id'], after_revision=revision)['results'] == []
    release.set()
    result = finished(manager, first['job_id'])
    delta = manager.get(first['job_id'], after_revision=revision)
    assert delta['completed'] == 40
    assert [row['match_id'] for row in delta['results']] == ['0']
    assert result['revision'] > revision


def test_concurrent_duplicate_batches_and_overlapping_jobs_share_inflight_work(managers):
    manager = managers()
    release = managers.event()
    lock = threading.Lock()
    calls = Counter()
    def analyze(match):
        with lock:
            calls[match['match_id']] += 1
        release.wait()
        return {'id': match['match_id']}
    submitted = []
    barrier = threading.Barrier(7)
    def click():
        barrier.wait()
        submitted.append(manager.submit(matches(0, 1, 2), analyze))
    clickers = [threading.Thread(target=click) for _ in range(6)]
    for thread in clickers:
        thread.start()
    barrier.wait()
    for thread in clickers:
        thread.join(1)
        assert not thread.is_alive()
    assert len({job['job_id'] for job in submitted}) == 1
    overlapping = manager.submit(matches(1, 2, 3), analyze)
    assert overlapping['job_id'] != submitted[0]['job_id']
    release.set()
    left = finished(manager, submitted[0]['job_id'])
    right = finished(manager, overlapping['job_id'])
    assert calls == Counter({'0': 1, '1': 1, '2': 1, '3': 1})
    assert left['succeeded'] == right['succeeded'] == 3


def test_reordered_batch_reuses_original_job_and_returns_complete_current_snapshot(managers):
    manager = managers()
    release = managers.event()
    def analyze(match):
        if match['match_id'] == '1':
            release.wait()
        return {'id': match['match_id']}
    original = manager.submit(matches(0, 1), analyze)
    partial = eventually(lambda: manager.get(original['job_id']), lambda job: job['completed'] == 1)
    reused = manager.submit(matches(1, 0), analyze)
    assert reused['job_id'] == original['job_id']
    assert reused['results'] == partial['results']
    assert reused['results'][0]['index'] == 0
    release.set()
    finished(manager, original['job_id'])


@pytest.mark.parametrize('changed', ['market', 'force_refresh', 'context_token'])
def test_full_market_context_refresh_and_context_token_prevent_wrong_sharing(managers, changed):
    manager = managers()
    release = managers.event()
    calls = []
    def analyze(match):
        calls.append(match)
        release.wait()
        return {'home': match['home']}
    first_match = {**matches(1)[0], 'market': {'line': 2.5, 'over': .93}}
    first = manager.submit([first_match], analyze)
    different = {**first_match, 'market': dict(first_match['market'])}
    options = {}
    if changed == 'market':
        different['market']['over'] = .85
    elif changed == 'force_refresh':
        options['force_refresh'] = True
    else:
        options['context_token'] = 'new-model-version'
    second = manager.submit([different], analyze, **options)
    assert first['job_id'] != second['job_id']
    eventually(lambda: len(calls), lambda total: total == 2)
    release.set()
    assert finished(manager, first['job_id'])['succeeded'] == 1
    assert finished(manager, second['job_id'])['succeeded'] == 1


def test_timeout_cannot_release_busy_workers_or_be_overwritten_by_late_success(managers):
    clock = Clock()
    manager = managers(clock=clock, match_timeout=1, job_timeout=10, max_pending=4)
    release = managers.event()
    started = []
    def analyze(match):
        started.append((match['match_id'], threading.get_ident()))
        release.wait()
        return {'late': True}
    first = manager.submit(matches(0, 1), analyze)
    eventually(lambda: len(started), lambda n: n == 2)
    clock.advance(2)
    timed = manager.get(first['job_id'])
    assert timed['completed'] == timed['failed'] == 2
    assert all(item['status'] == 'timed_out' for item in timed['results'])
    revision = timed['revision']
    retry = manager.submit(matches(0, 1), analyze)
    assert retry['completed'] == 2  # Reuses the still-running terminal work.
    later = manager.submit(matches(2, 3), analyze)
    assert manager.get(later['job_id'])['completed'] == 0
    assert len(started) == 2
    release.set()
    assert finished(manager, later['job_id'])['succeeded'] == 2
    assert manager.get(first['job_id'])['revision'] == revision
    assert manager.get(first['job_id'])['results'] == timed['results']
    assert len({identity for _, identity in started}) == 2


def test_expired_queued_items_are_terminal_and_never_start(managers):
    clock = Clock()
    manager = managers(worker_count=1, clock=clock, match_timeout=1, job_timeout=3)
    release = managers.event()
    started = []
    def analyze(match):
        started.append(match['match_id'])
        release.wait()
        return {'id': match['match_id']}
    job = manager.submit(matches(0, 1, 2, 3), analyze)
    eventually(lambda: started, lambda rows: rows == ['0'])
    clock.advance(4)
    terminal = manager.get(job['job_id'])
    assert terminal['status'] == 'completed'
    assert terminal['failed'] == terminal['completed'] == 4
    assert all(row['status'] == 'timed_out' for row in terminal['results'])
    release.set()
    manager.close(wait=True)
    assert started == ['0']


def test_pending_capacity_rejection_is_atomic_and_overlap_still_joins(managers):
    manager = managers(worker_count=1, max_pending=2)
    release = managers.event()
    started = threading.Event()
    def analyze(match):
        started.set()
        release.wait()
        return {'id': match['match_id']}
    first = manager.submit(matches(0), analyze)
    assert started.wait(1)
    queued = manager.submit(matches(1, 2), analyze)
    with pytest.raises(FootballAnalysisQueueFull):
        manager.submit(matches(3, 4), analyze)
    assert manager.submit(matches(2, 1), analyze)['job_id'] == queued['job_id']
    assert len(manager._jobs) == 2
    assert len(manager._pending) == 2
    release.set()
    assert finished(manager, first['job_id'])['succeeded'] == 1
    assert finished(manager, queued['job_id'])['succeeded'] == 2


def test_completed_ttl_and_job_capacity_do_not_orphan_running_timed_out_work(managers):
    clock = Clock()
    manager = managers(worker_count=1, clock=clock, match_timeout=1,
                       completed_ttl=2, max_jobs=1)
    release = managers.event()
    started = threading.Event()
    def analyze(match):
        started.set()
        release.wait()
        return {'id': match['match_id']}
    job = manager.submit(matches(0), analyze)
    assert started.wait(1)
    clock.advance(2)
    assert manager.get(job['job_id'])['completed'] == 1
    clock.advance(3)
    assert manager.get(job['job_id']) is not None
    with pytest.raises(FootballAnalysisQueueFull):
        manager.submit(matches(1), analyze)
    release.set()
    eventually(lambda: manager.get(job['job_id']), lambda value: value is None)
    new = manager.submit(matches(1), lambda match: {'id': match['match_id']})
    assert finished(manager, new['job_id'])['succeeded'] == 1


def test_finished_jobs_expire_and_capacity_can_evict_old_terminal_jobs(managers):
    clock = Clock()
    manager = managers(clock=clock, completed_ttl=2, max_jobs=1)
    first = manager.submit(matches(0), lambda match: {'ok': True})
    finished(manager, first['job_id'])
    second = manager.submit(matches(1), lambda match: {'ok': True})
    assert manager.get(first['job_id']) is None
    finished(manager, second['job_id'])
    clock.advance(3)
    assert manager.get(second['job_id']) is None


def test_input_callback_result_and_returned_snapshots_are_isolated_copies(managers):
    manager = managers()
    release = managers.event()
    started = threading.Event()
    payload = {'nested': {'score': [1, 2]}}
    original = matches(0)
    original[0]['market'] = {'line': 2.5}
    def analyze(match):
        assert match['market']['line'] == 2.5
        match['market']['line'] = 99
        started.set()
        release.wait()
        return payload
    job = manager.submit(original, analyze)
    assert started.wait(1)
    original[0]['market']['line'] = 3.5
    job['total'] = 999
    release.set()
    result = finished(manager, job['job_id'])
    payload['nested']['score'].append(3)
    result['results'][0]['result']['nested']['score'].append(4)
    again = manager.get(job['job_id'])
    assert again['total'] == 1
    assert again['results'][0]['result']['nested']['score'] == [1, 2]


def test_duplicate_positions_are_one_execution_with_separate_terminal_entries(managers):
    manager = managers()
    calls = []
    job = manager.submit(matches(1, 1, 1), lambda match: calls.append(match) or {'ok': True})
    terminal = finished(manager, job['job_id'])
    assert len(calls) == 1
    assert terminal['completed'] == 3
    assert {entry['index'] for entry in terminal['results']} == {0, 1, 2}


def test_close_rejects_submissions_and_finalizes_pending_without_late_revival(managers):
    manager = managers(worker_count=1)
    release = managers.event()
    started = threading.Event()
    def analyze(match):
        started.set()
        release.wait()
        return {'late': True}
    job = manager.submit(matches(0, 1), analyze)
    assert started.wait(1)
    manager.close(wait=False)
    closed = manager.get(job['job_id'])
    assert closed['status'] == 'failed'
    assert closed['completed'] == closed['failed'] == 2
    with pytest.raises(FootballAnalysisManagerClosed):
        manager.submit(matches(2), analyze)
    release.set()
    manager.close(wait=True)
    assert manager.get(job['job_id']) == closed
    assert all(not worker.is_alive() for worker in manager._workers)
    assert not manager._monitor.is_alive()


@pytest.mark.parametrize('invalid', [[], matches(*range(81)), [None], [{'match_id': 'x', 'odds': float('nan')}]] )
def test_invalid_batches_are_rejected_before_starting_workers(managers, invalid):
    manager = managers()
    with pytest.raises(ValueError):
        manager.submit(invalid, lambda match: {'ok': True})
    assert manager._workers == []
    assert manager._jobs == {}


def test_retained_result_limit_evicts_old_whole_jobs_not_individual_results(managers):
    manager = managers(max_matches=2, max_retained_results=3, max_jobs=10)
    older = manager.submit(matches(0, 1), lambda match: {'id': match['match_id']})
    finished(manager, older['job_id'])
    newer = manager.submit(matches(2, 3), lambda match: {'id': match['match_id']})
    snapshot = finished(manager, newer['job_id'])
    assert len(snapshot['results']) == snapshot['total'] == 2
    assert manager.get(older['job_id']) is None
    assert len(manager._jobs) == 1  # Capacity and TTL have not been reached.


def test_one_largest_legal_batch_fits_the_retained_result_budget(managers):
    manager = managers(max_matches=3, max_retained_results=3)
    job = manager.submit(matches(0, 1, 2), lambda match: {'ok': True})
    snapshot = finished(manager, job['job_id'])
    assert snapshot['completed'] == 3
    assert len(manager.get(job['job_id'])['results']) == 3
    with pytest.raises(ValueError, match='at least max_matches'):
        FootballAnalysisJobManager(max_matches=3, max_retained_results=2)


def test_retention_never_discards_partial_running_jobs(managers):
    manager = managers(max_matches=2, max_retained_results=2)
    release = managers.event()
    started = threading.Event()
    def analyze(match):
        if match['match_id'] == '0':
            started.set()
            release.wait()
        return {'id': match['match_id']}
    running = manager.submit(matches(0, 1), analyze)
    assert started.wait(1)
    eventually(lambda: manager.get(running['job_id']), lambda job: job['completed'] == 1)
    retired = manager.submit(matches(2, 3), analyze)
    finished(manager, retired['job_id'])
    newest = manager.submit(matches(4, 5), analyze)
    finished(manager, newest['job_id'])
    assert manager.get(retired['job_id']) is None
    partial = manager.get(running['job_id'])
    assert partial['status'] == 'running' and partial['completed'] == 1
    release.set()
    assert finished(manager, running['job_id'])['completed'] == 2


def test_timed_out_live_work_is_exempt_until_the_real_callback_exits(managers):
    clock = Clock()
    manager = managers(clock=clock, match_timeout=1, max_matches=2,
                       max_retained_results=2, max_jobs=10)
    release = managers.event()
    started = threading.Event()
    def analyze(match):
        if match['match_id'] == '0':
            started.set()
            release.wait()
        return {'id': match['match_id']}
    protected = manager.submit(matches(0, 1), analyze)
    assert started.wait(1)
    eventually(lambda: manager.get(protected['job_id']), lambda job: job['completed'] == 1)
    clock.advance(2)
    assert manager.get(protected['job_id'])['completed'] == 2
    retired = manager.submit(matches(2, 3), analyze)
    finished(manager, retired['job_id'])
    clock.advance(.1)
    newest = manager.submit(matches(4, 5), analyze)
    finished(manager, newest['job_id'])
    assert manager.get(retired['job_id']) is None
    assert manager.get(protected['job_id']) is not None
    # Once the actual callback exits, its old completed job joins normal
    # eviction candidates; the late success cannot resurrect it.
    release.set()
    eventually(lambda: manager.get(protected['job_id']), lambda job: job is None)
    assert len(manager.get(newest['job_id'])['results']) == 2


def test_incremental_poll_never_deepcopies_previous_result_payloads(managers):
    manager = managers()
    release = managers.event()
    class MustNotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError('an old prediction payload was traversed during an incremental poll')
    def analyze(match):
        if match['match_id'] == '1':
            release.wait()
        return {'id': match['match_id'], 'values': [1, 2]}
    job = manager.submit(matches(0, 1), analyze)
    partial = eventually(lambda: manager.get(job['job_id']), lambda value: value['completed'] == 1)
    revision = partial['revision']
    # Sentinel represents a large already-delivered prediction. The next
    # cursor poll must avoid traversing it, rather than copy then discard it.
    with manager._condition:
        manager._jobs[job['job_id']]['results'][0]['result']['large_payload'] = MustNotCopy()
    assert manager.get(job['job_id'], after_revision=revision)['results'] == []
    release.set()
    delta = eventually(lambda: manager.get(job['job_id'], after_revision=revision),
                       lambda value: value['completed'] == 2)
    assert [entry['match_id'] for entry in delta['results']] == ['1']
    delta['results'][0]['result']['values'].append(3)
    assert manager.get(job['job_id'], after_revision=revision)['results'][0]['result']['values'] == [1, 2]


def test_module_shutdown_does_not_construct_an_unused_singleton(monkeypatch):
    monkeypatch.setattr(jobs_module, '_singleton', None)
    def must_not_construct():
        raise AssertionError('shutdown must not create an analysis manager')
    monkeypatch.setattr(jobs_module, 'FootballAnalysisJobManager', must_not_construct)
    jobs_module.shutdown(wait=True)
    jobs_module.shutdown(wait=False)
    assert jobs_module._singleton is None


def test_nonblocking_shutdown_keeps_old_instance_until_a_blocking_join(managers, monkeypatch):
    manager = managers(worker_count=1)
    monkeypatch.setattr(jobs_module, '_singleton', manager)
    release = managers.event()
    started = threading.Event()
    def analyze(match):
        started.set()
        release.wait()
        return {'ok': True}
    manager.submit(matches(0), analyze)
    assert started.wait(1)
    jobs_module.shutdown(wait=False)
    assert jobs_module.get_manager() is manager
    with pytest.raises(FootballAnalysisManagerClosed):
        jobs_module.submit(matches(1), analyze)
    release.set()
    jobs_module.shutdown(wait=True)
    assert jobs_module._singleton is None
    replacement = jobs_module.get_manager()
    assert replacement is not manager
    assert replacement._workers == []
    jobs_module.shutdown(wait=True)


def test_blocking_shutdown_waits_for_actual_callback_before_allowing_replacement(managers, monkeypatch):
    manager = managers(worker_count=1)
    monkeypatch.setattr(jobs_module, '_singleton', manager)
    release = managers.event()
    started = threading.Event()
    callback_finished = threading.Event()
    errors = []
    def analyze(match):
        started.set()
        release.wait()
        # The accessor is usable during shutdown; joining must not hold its
        # global lock, and it must still return this closed original manager.
        if jobs_module.get_manager() is not manager:
            errors.append('new manager appeared before old callback exited')
        callback_finished.set()
        return {'ok': True}
    manager.submit(matches(0), analyze)
    assert started.wait(1)
    closer = threading.Thread(target=jobs_module.shutdown, kwargs={'wait': True})
    closer.start()
    try:
        eventually(lambda: manager._closed, bool)
        assert closer.is_alive()
        assert jobs_module.get_manager() is manager
        assert not callback_finished.is_set()
    finally:
        release.set()
        closer.join(2)
    assert not closer.is_alive()
    assert callback_finished.is_set()
    assert errors == []
    assert all(not worker.is_alive() for worker in manager._workers)
    assert jobs_module._singleton is None
    replacement = jobs_module.get_manager()
    assert replacement is not manager
    jobs_module.shutdown(wait=True)
