"""Bounded in-process football analysis jobs with per-match incremental results.

Importing this module starts no threads. Only the lazy manager's first submit
starts its fixed workers and one deadline monitor. A timed-out callback keeps
occupying its real worker until it exits; no replacement analysis thread is
created. Callers supply a cooperative-budget ``analyze_fn(match)`` wrapper.
Retired jobs retain at most ``max_retained_results`` result entries in total;
active jobs and callbacks that have timed out but are still executing are
protected separately by the job/match capacity limits. This is an entry budget,
not a byte estimate for potentially differently sized prediction dictionaries.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import threading
import time
import uuid


def _default_timeout():
    try:
        value = float(os.getenv('FOOTBALL_ANALYSIS_MATCH_TIMEOUT', '45'))
        return value if math.isfinite(value) and value > 0 else 45.0
    except (TypeError, ValueError):
        return 45.0


DEFAULT_MATCH_TIMEOUT = _default_timeout()
DEFAULT_JOB_TIMEOUT = 20 * 60.0


class FootballAnalysisQueueFull(RuntimeError):
    """The bounded pending queue or job registry cannot accept this batch."""


class FootballAnalysisManagerClosed(RuntimeError):
    """The application has stopped accepting background analysis work."""


@dataclass
class _Work:
    key: str
    match: dict
    callback: object
    created: float
    state: str = 'queued'
    started: float | None = None
    executing: bool = False
    subscribers: dict = field(default_factory=dict)
    outcome: dict | None = None


class FootballAnalysisJobManager:
    def __init__(self, *, worker_count=2, match_timeout=DEFAULT_MATCH_TIMEOUT,
                 job_timeout=DEFAULT_JOB_TIMEOUT, max_pending=160, max_matches=80,
                 max_jobs=32, completed_ttl=30 * 60.0, max_retained_results=256,
                 monitor_interval=.05,
                 clock=time.monotonic, wall_clock=time.time):
        for name, value in (('worker_count', worker_count), ('max_pending', max_pending),
                            ('max_matches', max_matches), ('max_jobs', max_jobs),
                            ('max_retained_results', max_retained_results)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if max_retained_results < max_matches:
            raise ValueError('max_retained_results must be at least max_matches to retain one complete batch')
        for name, value in (('match_timeout', match_timeout), ('job_timeout', job_timeout),
                            ('completed_ttl', completed_ttl), ('monitor_interval', monitor_interval)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be a finite positive number')
        self.worker_count = worker_count
        self.match_timeout = float(match_timeout)
        self.job_timeout = float(job_timeout)
        self.max_pending = max_pending
        self.max_matches = max_matches
        self.max_jobs = max_jobs
        self.max_retained_results = max_retained_results
        self.completed_ttl = float(completed_ttl)
        self.monitor_interval = float(monitor_interval)
        self._clock, self._wall_clock = clock, wall_clock
        self._condition = threading.Condition(threading.RLock())
        self._jobs = {}
        self._job_works = {}
        self._job_done = {}
        self._job_finished = {}
        self._batch_ids = {}
        self._job_batches = {}
        self._works = {}
        self._pending = deque()
        self._workers = []
        self._monitor = None
        self._closed = False

    def submit(self, matches, analyze_fn, *, force_refresh=False, context_token=''):
        if not isinstance(matches, (list, tuple)) or not 1 <= len(matches) <= self.max_matches:
            raise ValueError(f'matches must contain 1..{self.max_matches} objects')
        if not callable(analyze_fn):
            raise ValueError('analyze_fn must be callable')
        if not isinstance(force_refresh, bool) or not isinstance(context_token, str):
            raise ValueError('force_refresh must be boolean and context_token must be text')
        prepared = []
        for match in matches:
            if not isinstance(match, dict):
                raise ValueError('each match must be an object')
            try:
                canonical = json.dumps(match, ensure_ascii=False, sort_keys=True,
                                       separators=(',', ':'), allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError('match must contain finite JSON data') from exc
            signature = json.dumps([canonical, force_refresh, context_token], ensure_ascii=False,
                                   separators=(',', ':'))
            key = hashlib.sha256(signature.encode('utf-8')).hexdigest()
            prepared.append((key, json.loads(canonical)))
        # Order-independent batch deduplication preserves duplicate positions.
        batch_key = hashlib.sha256(json.dumps(sorted(key for key, _ in prepared)).encode()).hexdigest()
        with self._condition:
            if self._closed:
                raise FootballAnalysisManagerClosed('football analysis manager is closed')
            now = self._clock()
            self._expire_locked(now)
            active_id = self._batch_ids.get(batch_key)
            if active_id in self._jobs:
                return self._copy_locked(active_id)
            new_keys = {key for key, _ in prepared if key not in self._works}
            if len(self._pending) + len(new_keys) > self.max_pending:
                raise FootballAnalysisQueueFull('football analysis pending queue is full')
            self._prune_locked(now, make_room=True)
            if len(self._jobs) >= self.max_jobs:
                raise FootballAnalysisQueueFull('football analysis job capacity is full')
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {
                'job_id': job_id, 'status': 'queued', 'revision': 1,
                'total': len(prepared), 'completed': 0, 'succeeded': 0, 'failed': 0,
                'results': [], 'created_at': self._wall_clock(),
                'started_at': None, 'finished_at': None,
            }
            self._job_done[job_id] = set()
            self._job_works[job_id] = []
            self._job_batches[job_id] = batch_key
            self._batch_ids[batch_key] = job_id
            for index, (key, match) in enumerate(prepared):
                work = self._works.get(key)
                if work is None:
                    work = _Work(key, match, analyze_fn, now)
                    self._works[key] = work
                    self._pending.append(key)
                self._job_works[job_id].append(work)
                work.subscribers.setdefault(job_id, set()).add(index)
            # Attach the entire batch before publishing already-timed-out work.
            for index, work in enumerate(self._job_works[job_id]):
                if work.state == 'running':
                    self._start_job_locked(job_id)
                elif work.outcome is not None:
                    self._publish_locked(job_id, index, work, work.outcome)
            self._start_threads_locked()
            self._condition.notify_all()
            return self._copy_locked(job_id)

    def get(self, job_id, after_revision=0):
        if not isinstance(after_revision, int) or isinstance(after_revision, bool) or after_revision < 0:
            raise ValueError('after_revision must be a nonnegative integer')
        with self._condition:
            self._expire_locked(self._clock())
            return self._copy_locked(job_id, after_revision)

    def close(self, wait=True):
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job['status'] in ('queued', 'running'):
                    # Only manager-level failure sets the whole job to failed.
                    job['status'] = 'failed'
            for work in list(self._works.values()):
                if work.outcome is None:
                    self._finish_work_locked(work, {'status': 'failed', 'error': 'football analysis manager closed'})
            self._pending.clear()
            self._condition.notify_all()
            threads = list(self._workers) + ([self._monitor] if self._monitor else [])
        if wait:
            for thread in threads:
                if thread is not threading.current_thread() and thread.ident is not None:
                    thread.join()

    def _copy_locked(self, job_id, after_revision=0):
        job = self._jobs.get(job_id)
        if job is None:
            return None
        # Filter before copying: a progress poll with no new results must not
        # traverse all prior prediction payloads while holding the manager lock.
        snapshot = {key: value for key, value in job.items() if key != 'results'}
        snapshot['results'] = [entry for entry in job['results'] if entry['revision'] > after_revision]
        return deepcopy(snapshot)

    def _start_threads_locked(self):
        if self._workers:
            return
        self._workers = [threading.Thread(target=self._worker_loop, daemon=True,
                                          name=f'FootballAnalysisWorker-{i + 1}')
                         for i in range(self.worker_count)]
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True,
                                         name='FootballAnalysisDeadlineMonitor')
        try:
            for thread in [*self._workers, self._monitor]:
                thread.start()
        except Exception:
            self.close(wait=False)
            raise

    def _start_job_locked(self, job_id):
        job = self._jobs.get(job_id)
        if job and job['status'] == 'queued':
            job['status'] = 'running'
            job['started_at'] = self._wall_clock()
            job['revision'] += 1

    def _publish_locked(self, job_id, index, work, outcome):
        job = self._jobs.get(job_id)
        done = self._job_done.get(job_id)
        if job is None or done is None or index in done:
            return
        done.add(index)
        job['revision'] += 1
        job['completed'] += 1
        job['succeeded' if outcome['status'] == 'completed' else 'failed'] += 1
        match = work.match or {}
        job['results'].append({'index': index, 'match_id': str(match.get('match_id') or match.get('analysis_id') or match.get('id') or ''),
                               **outcome, 'revision': job['revision']})
        if job['completed'] == job['total']:
            if job['status'] != 'failed':
                job['status'] = 'completed'
            job['finished_at'] = self._wall_clock()
            self._job_finished[job_id] = self._clock()
            batch_key = self._job_batches.get(job_id)
            if self._batch_ids.get(batch_key) == job_id:
                self._batch_ids.pop(batch_key, None)
            for subscribed in self._job_works[job_id]:
                subscribed.subscribers.pop(job_id, None)

    def _finish_work_locked(self, work, outcome):
        if work.outcome is not None:
            return
        work.outcome = outcome
        work.state = 'terminal'
        for job_id, indexes in list(work.subscribers.items()):
            for index in sorted(indexes):
                self._publish_locked(job_id, index, work, outcome)
        work.subscribers.clear()
        if not work.executing and self._works.get(work.key) is work:
            self._works.pop(work.key, None)
        self._condition.notify_all()

    def _expire_locked(self, now):
        for work in list(self._works.values()):
            if work.outcome is not None:
                continue
            error = None
            if now >= work.created + self.job_timeout:
                error = 'football analysis queue deadline exceeded' if work.started is None else 'football analysis job deadline exceeded'
            elif work.started is not None and now >= work.started + self.match_timeout:
                error = 'football match analysis timed out'
            if error:
                self._finish_work_locked(work, {'status': 'timed_out', 'error': error})
        self._pending = deque(key for key in self._pending
                              if key in self._works and self._works[key].state == 'queued')
        self._prune_locked(now)

    def _prune_locked(self, now, make_room=False):
        candidates = [job_id for job_id in sorted(self._job_finished, key=self._job_finished.get)
                      if not any(work.executing for work in self._job_works.get(job_id, []))]
        retained_results = sum(len(self._jobs[job_id]['results']) for job_id in candidates)
        for job_id in candidates:
            # Keep provenance while a real timed-out callback still occupies
            # a worker. Protected jobs never enter this eviction candidate set.
            expired = now - self._job_finished[job_id] >= self.completed_ttl
            over_results = retained_results > self.max_retained_results
            if not expired and not over_results and not (make_room and len(self._jobs) >= self.max_jobs):
                continue
            retained_results -= len(self._jobs[job_id]['results'])
            self._jobs.pop(job_id, None)
            self._job_done.pop(job_id, None)
            self._job_finished.pop(job_id, None)
            self._job_works.pop(job_id, None)
            self._job_batches.pop(job_id, None)

    def _worker_loop(self):
        while True:
            with self._condition:
                self._expire_locked(self._clock())
                while not self._pending and not self._closed:
                    self._condition.wait()
                    self._expire_locked(self._clock())
                if self._closed:
                    return
                key = self._pending.popleft()
                work = self._works.get(key)
                if work is None or work.state != 'queued':
                    continue
                work.state = 'running'
                work.executing = True
                work.started = self._clock()
                for job_id in work.subscribers:
                    self._start_job_locked(job_id)
            try:
                result = work.callback(deepcopy(work.match))
                if not isinstance(result, dict):
                    raise RuntimeError('football analysis returned an invalid result')
                if result.get('error'):
                    raise RuntimeError(str(result['error']))
                outcome = {'status': 'completed', 'result': deepcopy(result)}
            except TimeoutError as exc:
                outcome = {'status': 'timed_out', 'error': str(exc) or 'football match analysis timed out'}
            except BaseException as exc:
                # An item must not destroy a fixed worker, including a callback
                # that raises SystemExit inside this background thread.
                outcome = {'status': 'failed', 'error': str(exc) or type(exc).__name__}
            with self._condition:
                # Check elapsed deadlines before accepting even a late success.
                self._expire_locked(self._clock())
                self._finish_work_locked(work, outcome)
                work.executing = False
                if self._works.get(work.key) is work:
                    self._works.pop(work.key, None)
                work.callback = None
                self._prune_locked(self._clock())
                self._condition.notify_all()

    def _monitor_loop(self):
        with self._condition:
            while not self._closed:
                self._expire_locked(self._clock())
                self._condition.wait(self.monitor_interval)


_singleton = None
_singleton_lock = threading.Lock()


def get_football_analysis_job_manager():
    """Construct lazily; service import and status reads never start workers."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = FootballAnalysisJobManager()
        return _singleton


# Short spelling retained for service adapters; both names share one manager.
get_manager = get_football_analysis_job_manager


def shutdown(wait=True):
    """Stop existing work without constructing a manager during app teardown.

    Keep the closed instance discoverable while its real callbacks are alive.
    A nonblocking shutdown never resets the singleton; a later blocking call
    must finish joining its workers before another manager can replace it.
    """
    global _singleton
    with _singleton_lock:
        manager = _singleton
        if manager is None:
            return
        manager.close(wait=False)
    if wait:
        # Do not hold the singleton lock while waiting: an exiting callback
        # may need to read status through the module accessor.
        manager.close(wait=True)
        with _singleton_lock:
            threads = [*manager._workers, *([manager._monitor] if manager._monitor else [])]
            if _singleton is manager and not any(thread.is_alive() for thread in threads):
                _singleton = None


def submit(matches, analyze_fn, *, force_refresh=False, context_token=''):
    return get_football_analysis_job_manager().submit(matches, analyze_fn,
                                                      force_refresh=force_refresh, context_token=context_token)


def get(job_id, after_revision=0):
    return get_football_analysis_job_manager().get(job_id, after_revision)
