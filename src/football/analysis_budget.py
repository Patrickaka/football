"""Cooperative wall-clock budget shared by one match and its I/O children.

Python cannot safely terminate an arbitrary running thread. Callers retain
their worker slot until it exits; this budget bounds ordinary I/O, lock waits
and retries and prevents an expired analysis from publishing a new result.
"""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import math
import os
import time


def _configured_timeout():
    try:
        value = float(os.getenv('FOOTBALL_ANALYSIS_MATCH_TIMEOUT', '45'))
        return value if math.isfinite(value) and value > 0 else 45.0
    except (TypeError, ValueError):
        return 45.0


MATCH_TIMEOUT_SECONDS = _configured_timeout()
_deadline = ContextVar('football_analysis_deadline', default=None)


class AnalysisTimeout(TimeoutError):
    def __init__(self):
        super().__init__('单场分析超过等待上限，暂缺结果，请稍后重试')


@contextmanager
def limit(seconds=MATCH_TIMEOUT_SECONDS):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('analysis budget must be finite and positive')
    deadline = time.monotonic() + seconds
    parent = _deadline.get()
    token = _deadline.set(min(parent, deadline) if parent is not None else deadline)
    try:
        check()
        yield
    finally:
        _deadline.reset(token)


def remaining(maximum=None):
    deadline = _deadline.get()
    if deadline is None:
        return maximum
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise AnalysisTimeout()
    return min(seconds, maximum) if maximum is not None else seconds


def check():
    remaining()


@contextmanager
def acquire(lock):
    seconds = remaining()
    acquired = lock.acquire() if seconds is None else lock.acquire(timeout=seconds)
    if not acquired:
        raise AnalysisTimeout()
    try:
        check()
        yield
    finally:
        lock.release()


def sleep(seconds):
    time.sleep(remaining(max(0.0, seconds)))
    check()


def wait(event):
    if not event.wait(timeout=remaining()):
        raise AnalysisTimeout()
    check()


def submit(pool, fn, *args):
    # Each child needs its own Context: a Context cannot run concurrently.
    return pool.submit(copy_context().run, fn, *args)


def result(future):
    try:
        value = future.result(timeout=remaining())
    except TimeoutError:
        raise AnalysisTimeout() from None
    check()
    return value
