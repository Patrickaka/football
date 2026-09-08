"""Bounded storage maintenance for football, without removing business history."""
from datetime import datetime, timedelta, timezone
import os


def _positive_int(name, default):
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def archive_stale_prediction_records(*, history=None, now=None, dry_run=False,
                                     archive_after_days=None, max_records=None):
    """Archive persisted rows under the writer lock; never persist a thin memory copy.

    Work is bounded and rotates across eligible IDs. The repository independently
    checks the stored row and compares it before updating, so an in-flight result
    correction cannot be replaced by an older document from memory.
    """
    from ..common import repositories
    from ..common.football_storage import _archive_days
    from .research import timestamp
    if history is None:
        from .result_sync import get_history
        history = get_history()
    now = now or datetime.now(timezone.utc)
    now = timestamp(now)
    if now is None:
        raise ValueError('archive maintenance requires an aware time')
    days = _archive_days(archive_after_days)
    limit = (_positive_int('FOOTBALL_ARCHIVE_BATCH_SIZE', 200)
             if max_records is None else max(1, int(max_records)))
    cutoff = now - timedelta(days=days)
    result = {'dry_run': dry_run, 'archive_after_days': days, 'batch_size': limit,
              'eligible_count': 0, 'checked_count': 0, 'archived_count': 0,
              'would_archive_count': 0, 'already_archived_count': 0,
              'bytes_saved_estimate': 0, 'errors': [], 'results': []}
    with history._records_lock:
        ids = sorted({str(row['match_id']) for row in history.records
                      if row.get('match_id') and row.get('settled')
                      and timestamp(row.get('settled_at')) is not None
                      and timestamp(row['settled_at']) <= cutoff})
        cursor = getattr(history, '_storage_archive_cursor', 0) % max(1, len(ids))
    result['eligible_count'] = len(ids)
    candidates = (ids[cursor:] + ids[:cursor])[:limit]
    for match_id in candidates:
        result['checked_count'] += 1
        try:
            with history._records_lock:
                audit = repositories.football_prediction_archive_one(
                    match_id, now=now, archive_after_days=days, dry_run=dry_run)
            result['results'].append({'match_id': match_id, **audit})
            status = audit.get('status')
            count_field = {'archived': 'archived_count', 'would_archive': 'would_archive_count',
                           'already_archived': 'already_archived_count'}.get(status)
            if count_field:
                result[count_field] += 1
            if status in ('archived', 'would_archive'):
                result['bytes_saved_estimate'] += max(0, audit.get('bytes_saved', 0))
            elif status == 'failed':
                result['errors'].append({'match_id': match_id, 'error': audit.get('error', 'archive_failed')})
        except Exception as exc:
            result['errors'].append({'match_id': match_id, 'error': str(exc)})
    if not dry_run:
        with history._records_lock:
            history._storage_archive_cursor = (cursor + len(candidates)) % max(1, len(ids))
    return result


def run_football_storage_maintenance(*, dry_run=False, now=None, history=None):
    """Each stage fails independently, including when an archive is corrupt."""
    result = {'dry_run': dry_run}
    try:
        from .intelligence.retention import cleanup_intelligence_cache
        result['intelligence_cache'] = cleanup_intelligence_cache(dry_run=dry_run, now=now)
    except Exception as exc:
        result['intelligence_cache'] = {'removed_count': 0, 'removed_bytes': 0, 'errors': [str(exc)]}
    try:
        result['prediction_archive'] = archive_stale_prediction_records(
            history=history, dry_run=dry_run, now=now)
    except Exception as exc:
        result['prediction_archive'] = {'archived_count': 0, 'bytes_saved_estimate': 0, 'errors': [str(exc)]}
    return result
