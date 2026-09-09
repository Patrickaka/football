"""Real job manager and ASGI routes; fake match computation, no upstream I/O."""
from collections import Counter
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.routers import football as routes
from src.api.runtime import football_analysis_jobs as jobs
from src.api.services import football as service


@pytest.fixture
def harness(monkeypatch):
    manager = jobs.FootballAnalysisJobManager(match_timeout=.5, monitor_interval=.005)
    monkeypatch.setattr(jobs, 'get_manager', lambda: manager)
    monkeypatch.setattr(service, '_prediction_schedule_snapshot', lambda: [])
    monkeypatch.setattr(service, 'fetch_match_list', lambda: pytest.fail('job API fetched schedule'))
    async def blocked_executor(*_, **__):
        pytest.fail('control route used the blocking analysis executor')
    monkeypatch.setattr(routes, 'run_blocking', blocked_executor)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        try:
            yield client, manager
        finally:
            manager.close()


def poll(client, job_id, predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        response = client.get('/api/predict/batch/status', params={'job_id': job_id})
        assert response.status_code == 200
        result = response.json()['result']
        if predicate(result):
            return result
        time.sleep(.005)
    pytest.fail(f'no expected progress: {result}')


def test_forty_matches_partial_results_timeouts_and_duplicate_start(harness, monkeypatch):
    client, manager = harness
    release = threading.Event()
    entered = threading.Event()
    calls = Counter()
    def analyze(match, **_):
        mid = match['match_id']
        calls[mid] += 1
        if mid == '0':
            entered.set()
            release.wait(2)
        return {'match_id': mid, 'probability': np.float64(.6), 'missing': float('nan')}
    monkeypatch.setattr(service, 'analyze_match', analyze)
    body = {'matches': [{'match_id': str(i)} for i in range(40)]}
    try:
        started = time.monotonic()
        response = client.post('/api/predict/batch/start', json=body)
        elapsed = time.monotonic() - started
        assert response.status_code == 202
        assert elapsed < .4  # Does not wait for the intentionally blocked match.
        job_id = response.json()['result']['job_id']
        assert entered.wait(1)
        duplicate = client.post('/api/predict/batch/start', json=body).json()['result']
        assert duplicate['job_id'] == job_id
        partial = poll(client, job_id, lambda j: j['succeeded'] == 39)
        assert not release.is_set()
        assert {r['match_id'] for r in partial['results'] if r['status'] == 'completed'} == {
            str(i) for i in range(1, 40)}
        assert all(r['result']['missing'] is None for r in partial['results'] if r.get('result'))
        finished = poll(client, job_id, lambda j: j['status'] == 'completed')
        assert (finished['completed'], finished['succeeded'], finished['failed']) == (40, 39, 1)
        assert next(r for r in finished['results'] if r['match_id'] == '0')['status'] == 'timed_out'
        delta = client.get('/api/predict/batch/status', params={
            'job_id': job_id, 'after_revision': finished['revision']}).json()['result']
        assert delta['results'] == []
        release.set()
        manager.close()
        assert manager.get(job_id)['succeeded'] == 39  # Late return cannot overwrite timeout.
        assert len(calls) == 40 and set(calls.values()) == {1}
    finally:
        release.set()


@pytest.mark.parametrize('body', [None, [], {}, {'matches': []}, {'matches': [{}]},
    {'matches': [1]}, {'matches': [{'match_id': 'm'}] * 2},
    {'matches': [{'match_id': str(i)} for i in range(81)]},
    {'matches': [{'match_id': 'm'}], 'force_refresh': 'false'}])
def test_malformed_start_is_rejected_before_work(harness, monkeypatch, body):
    client, _ = harness
    monkeypatch.setattr(service, 'analyze_match', lambda *_: pytest.fail('invalid task ran'))
    response = client.post('/api/predict/batch/start', json=body)
    assert response.status_code == 400
    assert response.json()['code'] == 'invalid_request'


def test_status_errors_and_queue_full_have_retryable_http_contract(harness, monkeypatch):
    client, manager = harness
    assert client.get('/api/predict/batch/status').status_code == 400
    assert client.get('/api/predict/batch/status', params={'job_id': 'a' * 32}).status_code == 404
    assert client.get('/api/predict/batch/status', params={
        'job_id': 'a' * 32, 'after_revision': '-1'}).status_code == 400
    def full(*_, **__):
        raise jobs.FootballAnalysisQueueFull('busy')
    monkeypatch.setattr(manager, 'submit', full)
    assert client.post('/api/predict/batch/start', json={
        'matches': [{'match_id': 'm'}]}).status_code == 429


def test_new_jobs_keep_trusted_market_context_and_force_refresh(harness, monkeypatch):
    from tests.api.test_football_market_context import schedule_match
    client, _ = harness
    known = schedule_match()
    monkeypatch.setattr(service, '_prediction_schedule_snapshot', lambda: [known])
    def analyze(match, force_refresh=False):
        return {'match': match, 'force_refresh': force_refresh}
    monkeypatch.setattr(service, 'analyze_match', analyze)
    response = client.post('/api/predict/batch/start', json={'force_refresh': True, 'matches': [{
        'match_id': known['match_id'], 'asian_current': {'handicap': 99},
        'total_current': {'line': 99}, 'hkjc_id': 'injected'}]})
    assert response.status_code == 202
    finished = poll(client, response.json()['result']['job_id'], lambda j: j['completed'] == 1)
    prediction = finished['results'][0]['result']
    assert prediction['force_refresh'] is True
    assert prediction['match']['asian_current'] == known['asian_current']
    assert prediction['match']['total_current'] == known['total_current']
    assert prediction['match']['hkjc_id'] == known['hkjc_id']


def test_job_routes_use_existing_authentication():
    with TestClient(create_app(auth_settings=AuthSettings(credentials={'user': 'pw'}))) as client:
        assert client.post('/api/predict/batch/start', json={}).status_code == 401
        assert client.get('/api/predict/batch/status', params={'job_id': 'a' * 32}).status_code == 401
