"""Retention for reproducible intelligence caches, never prediction records.

Defaults: seven days, eight snapshots per match, 256 MiB of managed JSON in a
global sweep. ``max_bytes=0`` disables capacity pruning. Environment overrides:
FOOTBALL_INTELLIGENCE_RETENTION_DAYS / FOOTBALL_INTELLIGENCE_MAX_SNAPSHOTS /
FOOTBALL_INTELLIGENCE_MAX_BYTES. TTL remains an independent freshness rule.

Age and per-match count are applied first, then a global sweep evicts oldest
remaining snapshots until within capacity. Even a match's last cache may be
evicted: frozen business records already contain their own evidence copy. Files
outside our exact naming/schema/directory contract are never deleted. Old tmp
files need both their name timestamp and mtime to be at least one hour old.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time

from ...domain.sports.football.match_context import SCHEMA_VERSION, timestamp

DEFAULT_RETENTION_DAYS = 7
DEFAULT_MAX_SNAPSHOTS = 8
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
TEMP_MIN_AGE_SECONDS = 3600
_MAX_METADATA_READ = 8 * 1024 * 1024
_DIRECTORY = re.compile(r'^[0-9a-f]{64}$')
_FILE = re.compile(r'^(\d{8}T\d{12})-([0-9a-f]{32})\.(json|tmp)$')
_THREAD_LOCK = threading.RLock()


def _root(cache_dir=None):
    value = cache_dir or os.environ.get('FOOTBALL_INTELLIGENCE_CACHE_DIR')
    return Path(os.path.abspath(value if value is not None else Path(__file__).resolve().parents[3] / 'data' / 'intelligence_context'))


def _is_reparse(path):
    info = path.lstat()
    return (stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)))


def validate_cache_path(path, root):
    """Reject traversal, symlinks and Windows junction/reparse ancestors."""
    path, root = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    if not path.is_relative_to(root):
        raise OSError('cache path is outside the configured root')
    # Check lexical ancestors before resolve; resolve alone would conceal a
    # configured-root junction by treating its external target as the root.
    for part in [*reversed(path.parents), path]:
        try:
            if _is_reparse(part):
                raise OSError('symlink or junction in cache path')
        except FileNotFoundError:
            continue
    if not path.resolve(strict=False).is_relative_to(root.resolve(strict=False)):
        raise OSError('resolved cache path is outside the configured root')
    return path


@contextmanager
def intelligence_cache_lock(cache_dir=None, *, create=False, timeout=2.0):
    """Coordinate writers and cleanup across threads and local processes.

    Readers need no lock: snapshots are immutable/atomically published, and the
    existing reader already tolerates a file disappearing before it opens it.
    """
    root = _root(cache_dir)
    if not _THREAD_LOCK.acquire(timeout=max(0.0, timeout)):
        raise TimeoutError('cache maintenance lock is busy')
    handle, locked = None, False
    try:
        validate_cache_path(root, root)
        if create:
            root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise FileNotFoundError('cache directory does not exist')
        lock_path = validate_cache_path(root / '.retention.lock', root)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, 'r+b', buffering=0)
        validate_cache_path(lock_path, root)
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise OSError('invalid cache lock file')
        if info.st_size == 0:
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('cache maintenance lock is busy') from None
                time.sleep(.02)
        yield root
    finally:
        if handle is not None:
            if locked:
                try:
                    handle.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()
        _THREAD_LOCK.release()


def _setting(value, env, default, *, minimum=0, integer=False):
    value = os.environ.get(env, default) if value is None else value
    number = float(value)
    if not math.isfinite(number) or number < minimum or (integer and int(number) != number):
        raise ValueError(f'invalid {env}')
    return int(number) if integer else number


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _scan(root, result, match_id):
    directories = ([root / hashlib.sha256(str(match_id).encode()).hexdigest()]
                   if match_id is not None else list(root.iterdir()))
    entries = []
    for directory in directories:
        if not _DIRECTORY.fullmatch(directory.name):
            continue
        try:
            validate_cache_path(directory, root)
            if not directory.is_dir():
                continue
            paths = list(directory.iterdir())
        except OSError as exc:
            result['errors'].append({'path': directory.name, 'error': str(exc)})
            continue
        for path in paths:
            match = _FILE.fullmatch(path.name)
            if match is None:
                result['skipped_count'] += 1
                continue
            try:
                validate_cache_path(path, root)
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    result['skipped_count'] += 1
                    continue
                captured = datetime.strptime(match[1], '%Y%m%dT%H%M%S%f').replace(tzinfo=timezone.utc)
                kind = match[3]
                if kind == 'json':
                    if info.st_size > _MAX_METADATA_READ:
                        raise ValueError('oversized file cannot be safely verified as an owned snapshot')
                    with path.open('r', encoding='utf-8') as handle:
                        document = json.load(handle)
                    if (not isinstance(document, dict) or document.get('schema_version') != SCHEMA_VERSION
                            or hashlib.sha256(str(document.get('match_id', '')).encode()).hexdigest() != directory.name
                            or timestamp(document.get('captured_at')) != captured):
                        result['skipped_count'] += 1
                        continue
                entries.append({'path': path, 'directory': directory.name, 'captured': captured,
                                'mtime': info.st_mtime, 'bytes': info.st_size, 'kind': kind,
                                'fingerprint': _fingerprint(info)})
            except FileNotFoundError:
                continue
            except (OSError, ValueError, TypeError) as exc:
                result['errors'].append({'path': str(path.relative_to(root)), 'error': str(exc)})
    return entries


def _cleanup(root, result, *, days, count, capacity, now, match_id):
    entries = _scan(root, result, match_id)
    result.update(scanned_count=len(entries), scanned_bytes=sum(entry['bytes'] for entry in entries))
    selected = {}
    cutoff = now - timedelta(days=days)
    for entry in entries:
        if entry['kind'] == 'tmp':
            if min((now - entry['captured']).total_seconds(), now.timestamp() - entry['mtime']) >= TEMP_MIN_AGE_SECONDS:
                selected[entry['path']] = 'orphan_tmp'
        elif entry['captured'] < cutoff:
            selected[entry['path']] = 'age'
    groups = {}
    for entry in entries:
        if entry['kind'] == 'json' and entry['path'] not in selected:
            groups.setdefault(entry['directory'], []).append(entry)
    for group in groups.values():
        group.sort(key=lambda entry: (entry['captured'], entry['path'].name), reverse=True)
        for entry in group[count:]:
            selected[entry['path']] = 'match_count'
    remaining = [entry for entry in entries if entry['kind'] == 'json' and entry['path'] not in selected]
    total = sum(entry['bytes'] for entry in remaining)
    if capacity > 0:
        for entry in sorted(remaining, key=lambda entry: (entry['captured'], entry['path'].name)):
            if total <= capacity:
                break
            selected[entry['path']] = 'capacity'
            total -= entry['bytes']
    result['planned_count'] = len(selected)
    result['planned_bytes'] = sum(entry['bytes'] for entry in entries if entry['path'] in selected)
    removed = set()
    for entry in entries:
        path = entry['path']
        if path not in selected:
            continue
        try:
            validate_cache_path(path, root)
            if _fingerprint(path.lstat()) != entry['fingerprint']:
                raise OSError('cache changed during cleanup; deletion skipped')
            if not result['dry_run']:
                path.unlink()
                removed.add(path)
            result['files'].append({'path': str(path.relative_to(root)), 'bytes': entry['bytes'],
                                    'reason': selected[path], 'deleted': not result['dry_run']})
        except FileNotFoundError:
            continue
        except OSError as exc:
            result['errors'].append({'path': str(path.relative_to(root)), 'error': str(exc)})
    result['removed_count'] = len(removed)
    result['removed_bytes'] = sum(entry['bytes'] for entry in entries if entry['path'] in removed)
    result['remaining_count'] = len(entries) - len(removed)
    result['remaining_bytes'] = result['scanned_bytes'] - result['removed_bytes']
    result['projected_remaining_bytes'] = result['scanned_bytes'] - result['planned_bytes']
    managed_remaining = sum(entry['bytes'] for entry in entries if entry['kind'] == 'json' and entry['path'] not in removed)
    result['over_budget_bytes'] = max(0, managed_remaining - capacity) if capacity else 0
    return result


def cleanup_intelligence_cache(cache_dir=None, *, retention_days=None, max_snapshots_per_match=None,
                               max_bytes=None, dry_run=False, now=None, match_id=None):
    """Clean owned caches only; dry-run reports plans without creating lock files.

    Use no match_id for the maintenance sweep. Writers pass match_id and
    max_bytes=0, applying age/count to one match without scanning every cache.
    removed_* always means actual deletion; planned_* also reports dry runs.
    """
    root = _root(cache_dir)
    result = {'dry_run': bool(dry_run), 'scope': 'match' if match_id is not None else 'global',
              'scanned_count': 0, 'scanned_bytes': 0, 'skipped_count': 0, 'planned_count': 0, 'planned_bytes': 0,
              'removed_count': 0, 'removed_bytes': 0, 'remaining_count': 0, 'remaining_bytes': 0,
              'projected_remaining_bytes': 0, 'over_budget_bytes': 0, 'files': [], 'errors': []}
    try:
        days = _setting(retention_days, 'FOOTBALL_INTELLIGENCE_RETENTION_DAYS', DEFAULT_RETENTION_DAYS)
        count = _setting(max_snapshots_per_match, 'FOOTBALL_INTELLIGENCE_MAX_SNAPSHOTS', DEFAULT_MAX_SNAPSHOTS,
                         minimum=1, integer=True)
        capacity = _setting(max_bytes, 'FOOTBALL_INTELLIGENCE_MAX_BYTES', DEFAULT_MAX_BYTES, integer=True)
        clock = timestamp(now or datetime.now(timezone.utc))
        if clock is None:
            raise ValueError('cleanup time requires an explicit timezone')
        result.update(retention_days=days, max_snapshots_per_match=count, max_bytes=capacity)
        validate_cache_path(root, root)
        if not root.exists():
            return result
        if not root.is_dir():
            raise OSError('configured cache root is not a directory')
        if dry_run:
            # No filesystem mutation even when the cache has no lock file yet.
            with _THREAD_LOCK:
                return _cleanup(root, result, days=days, count=count, capacity=capacity, now=clock, match_id=match_id)
        with intelligence_cache_lock(root):
            return _cleanup(root, result, days=days, count=count, capacity=capacity, now=clock, match_id=match_id)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        result['errors'].append({'path': str(root), 'error': str(exc)})
        return result
