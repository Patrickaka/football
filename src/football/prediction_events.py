"""Seal newly observed prematch predictions inside the existing document store.

The caller holds PredictionHistory's write lock and persists the containing record.
There is no backfill: omission of prediction_event never manufactures frozen data.
"""
from copy import deepcopy
from datetime import datetime, timezone
import os

from ..common.football_storage import FootballStorageError, decode_json_gzip, encode_json_gzip

from ..domain.sports.football.prediction_evaluation import (
    EVENT_SCHEMA, SELECTION_POLICY, canonical_hash, event_hash, validate_event,
)
from ..domain.sports.football.release_evaluation import _timestamp


SEMANTIC_POLICY = 'football-event-model-and-evidence-v1'
ARCHIVE_SCHEMA = 'football-prediction-event-archive-v1'
RETENTION_SCHEMA = 'football-prediction-event-retention-v1'
ACTIVE_LIMIT_ENV = 'FOOTBALL_PREDICTION_EVENT_ACTIVE_LIMIT'
DEFAULT_ACTIVE_LIMIT = 8
_STORAGE_METADATA = ('hash', 'event_id', 'recorded_at', 'frozen', 'selection_policy',
                     'input_hash', 'semantic_hash', 'semantic_policy')


def _active_limit(value=None):
    if value is None:
        try:
            value = int(os.getenv(ACTIVE_LIMIT_ENV, str(DEFAULT_ACTIVE_LIMIT)))
        except (TypeError, ValueError) as exc:
            raise FootballStorageError(f'{ACTIVE_LIMIT_ENV} must be an integer') from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise FootballStorageError('Active event limit must be an integer')
    return max(2, value)


def prediction_event_semantic_hash(event):
    """Ignore only known outer run clocks, never source or evidence timestamps.

    In particular, collected_at/published_at, nested source captured_at, model
    versions, training cutoffs, feature audits, odds and kickoff all participate.
    Existing sealed events are read as-is and never receive new metadata.
    """
    content = deepcopy(event)
    for key in (*_STORAGE_METADATA, 'as_of', 'captured_at'):
        content.pop(key, None)
    variants = content.get('variants')
    if isinstance(variants, dict):
        for variant in variants.values():
            if isinstance(variant, dict):
                variant.pop('captured_at', None)
    return canonical_hash({'policy': SEMANTIC_POLICY, 'content': content})


def _count(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FootballStorageError(f'Invalid event archive {label}')
    return value


def hydrate_prediction_events(record):
    """Return every original event in append order, without mutating the record.

    The selected first event stays hot. Middle events live in one self-contained
    gzip envelope; no external files or growing plaintext hash index is needed.
    Missing/corrupt archives raise instead of exporting a silently partial list.
    Exporters replacing prediction_events with this list must omit the two
    storage-only fields prediction_event_archive and prediction_event_retention.
    """
    if not isinstance(record, dict):
        raise FootballStorageError('Prediction event record must be an object')
    active = record.get('prediction_events')
    if active is None:
        active = []
    if not isinstance(active, list) or any(not isinstance(item, dict) for item in active):
        raise FootballStorageError('Invalid active prediction event list')
    retention = record.get('prediction_event_retention')
    archive = record.get('prediction_event_archive')
    archived = []
    if 'prediction_event_archive' in record:
        if not isinstance(archive, dict) or archive.get('schema_version') != ARCHIVE_SCHEMA:
            raise FootballStorageError('Invalid prediction event archive schema')
        if not isinstance(retention, dict):
            raise FootballStorageError('Prediction event archive retention metadata is missing')
        archived = decode_json_gzip(archive)
        if not isinstance(archived, list) or any(not isinstance(item, dict) for item in archived):
            raise FootballStorageError('Invalid archived prediction event list')
        if _count(archive.get('count'), 'count') != len(archived) or not archived:
            raise FootballStorageError('Prediction event archive count mismatch')
    if retention is not None:
        if not isinstance(retention, dict) or retention.get('schema_version') != RETENTION_SCHEMA:
            raise FootballStorageError('Invalid prediction event retention schema')
        expected_archive = _count(retention.get('archived_events'), 'retained archive count')
        total = _count(retention.get('total_events'), 'total event count')
        active_limit = _count(retention.get('active_limit'), 'active limit')
        if active_limit < 2 or len(active) > active_limit:
            raise FootballStorageError('Prediction event active limit mismatch')
        if expected_archive != len(archived) or total != len(active) + len(archived):
            raise FootballStorageError('Prediction event archive is missing or its count does not match')
    if archived and len(active) < 2:
        raise FootballStorageError('Prediction event archive lost its selected or latest event')
    events = [active[0], *archived, *active[1:]] if active else []
    if events and record.get('selected_prediction_event_id') != events[0].get('event_id'):
        raise FootballStorageError('Selected prediction event does not match the original first event')
    seen, previous_time = set(), None
    for event in events:
        reason = validate_event(event)
        if reason or str(event.get('match_id')) != str(record.get('match_id')):
            raise FootballStorageError(f'Prediction event integrity failure: {reason or "match_id_mismatch"}')
        event_id, observed = event['event_id'], _timestamp(event['recorded_at'])
        if event_id in seen or (previous_time is not None and observed < previous_time):
            raise FootballStorageError('Prediction event identity or append order mismatch')
        seen.add(event_id)
        previous_time = observed
    return deepcopy(events)


def _retention_plan(events, limit):
    if len(events) > limit:
        active = [events[0], *events[-(limit-1):]]
        archived = events[1:-(limit-1)]
    else:
        active, archived = events, []
    archive = ({'schema_version': ARCHIVE_SCHEMA, 'count': len(archived),
                **encode_json_gzip(archived)} if archived else None)
    return {'prediction_events': active, 'prediction_event_archive': archive,
            'prediction_event_retention': {
                'schema_version': RETENTION_SCHEMA, 'total_events': len(events),
                'archived_events': len(archived), 'active_limit': limit}}


def _commit_retention(record, plan):
    changed = (record.get('prediction_events') != plan['prediction_events']
               or record.get('prediction_event_archive') != plan['prediction_event_archive']
               or record.get('prediction_event_retention') != plan['prediction_event_retention'])
    if changed:
        record['prediction_events'] = plan['prediction_events']
        record['prediction_event_retention'] = plan['prediction_event_retention']
        if plan['prediction_event_archive'] is not None:
            record['prediction_event_archive'] = plan['prediction_event_archive']
        else:
            record.pop('prediction_event_archive', None)
    return changed


def compact_prediction_events(record, *, max_active_events=None):
    """Losslessly compact existing observations; this never creates a prediction.

    Maintenance may call this for settled records while holding the history write
    lock. Any error leaves the entire record unchanged, including its first hash.
    """
    try:
        events = hydrate_prediction_events(record)
        if not events:
            return {'status': 'unchanged', 'storage_changed': False, 'archived_count': 0}
        plan = _retention_plan(events, _active_limit(max_active_events))
    except (FootballStorageError, TypeError, ValueError) as exc:
        return {'status': 'rejected', 'storage_changed': False,
                'reason': 'event_retention_failed', 'error': str(exc)}
    changed = _commit_retention(record, plan)
    return {'status': 'compacted' if changed else 'unchanged', 'storage_changed': changed,
            'active_count': len(plan['prediction_events']),
            'archived_count': plan['prediction_event_retention']['archived_events'],
            'total_count': len(events)}


def append_prediction_event(record, payload, *, now=None, max_active_events=None):
    if payload is None:
        return {'status': 'not_provided', 'appended': False}
    if record.get('settled'):
        return {'status': 'rejected', 'reason': 'match_already_settled', 'appended': False}
    if not isinstance(payload, dict):
        return {'status': 'rejected', 'reason': 'invalid_event_payload', 'appended': False}
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('recording time must include a timezone')
    event = deepcopy(payload)
    # Recording metadata belongs to this adapter, never to the caller.
    for key in _STORAGE_METADATA:
        event.pop(key, None)
    event.update(schema_version=EVENT_SCHEMA, match_id=str(record['match_id']))
    try:
        input_hash = canonical_hash(event)
        semantic_hash = prediction_event_semantic_hash(event)
    except (TypeError, ValueError):
        return {'status': 'rejected', 'reason': 'invalid_event_encoding', 'appended': False}
    event.update(recorded_at=now.astimezone(timezone.utc).isoformat(), frozen=True,
                 selection_policy=SELECTION_POLICY, input_hash=input_hash,
                 semantic_hash=semantic_hash, semantic_policy=SEMANTIC_POLICY)
    event['hash'] = event_hash(event)
    event['event_id'] = event['hash']
    reason = validate_event(event)
    if reason:
        return {'status': 'rejected', 'reason': reason, 'appended': False}
    # Validation precedes every duplicate shortcut, including exact replay. A
    # backdated payload submitted after kickoff cannot claim a new frozen save.
    captured = event.get('captured_at')
    if captured is not None and (_timestamp(captured) is None or _timestamp(captured) > _timestamp(event['as_of'])):
        return {'status': 'rejected', 'reason': 'invalid_capture_time', 'appended': False}
    try:
        limit = _active_limit(max_active_events)
    except FootballStorageError as exc:
        return {'status': 'rejected', 'reason': 'invalid_retention_configuration',
                'appended': False, 'storage_changed': False, 'error': str(exc)}
    try:
        existing = hydrate_prediction_events(record)
    except (FootballStorageError, TypeError, ValueError) as exc:
        return {'status': 'rejected', 'reason': 'existing_event_integrity_failure',
                'appended': False, 'storage_changed': False, 'error': str(exc)}
    duplicate = next((previous for previous in existing
                      if prediction_event_semantic_hash(previous) == semantic_hash), None)
    if existing and _timestamp(event['recorded_at']) < _timestamp(existing[-1]['recorded_at']):
        return {'status': 'rejected', 'reason': 'recording_clock_moved_backwards', 'appended': False}
    try:
        plan = _retention_plan(existing if duplicate else [*existing, event], limit)
    except (FootballStorageError, TypeError, ValueError) as exc:
        return {'status': 'rejected', 'reason': 'event_archive_write_failed',
                'appended': False, 'storage_changed': False, 'error': str(exc)}
    changed = _commit_retention(record, plan)
    if duplicate:
        return {'status': 'duplicate', 'event_id': duplicate['event_id'], 'appended': False,
                'duplicate_kind': 'exact' if duplicate.get('input_hash') == input_hash else 'semantic',
                'storage_changed': changed, 'archived_count': plan['prediction_event_retention']['archived_events']}
    if not existing:
        record['selected_prediction_event_id'] = event['event_id']
        record['prediction_event_selection_policy'] = SELECTION_POLICY
    return {'status': 'frozen', 'event_id': event['event_id'], 'appended': True,
            'storage_changed': changed, 'active_count': len(plan['prediction_events']),
            'archived_count': plan['prediction_event_retention']['archived_events'],
            'total_count': len(existing)+1}
