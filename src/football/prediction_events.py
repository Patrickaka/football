"""Seal newly observed prematch predictions inside the existing document store.

The caller holds PredictionHistory's write lock and persists the containing record.
There is no backfill: omission of prediction_event never manufactures frozen data.
"""
from copy import deepcopy
from datetime import datetime, timezone

from ..domain.sports.football.prediction_evaluation import (
    EVENT_SCHEMA, SELECTION_POLICY, canonical_hash, event_hash, validate_event,
)


def append_prediction_event(record, payload, *, now=None):
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
    for key in ('hash', 'event_id', 'recorded_at', 'frozen', 'selection_policy', 'input_hash'):
        event.pop(key, None)
    event.update(schema_version=EVENT_SCHEMA, match_id=str(record['match_id']))
    try:
        input_hash = canonical_hash(event)
    except (TypeError, ValueError):
        return {'status': 'rejected', 'reason': 'invalid_event_encoding', 'appended': False}
    existing = record.get('prediction_events') or []
    if not isinstance(existing, list) or any(not isinstance(item, dict) for item in existing):
        return {'status': 'rejected', 'reason': 'existing_event_integrity_failure', 'appended': False}
    if existing and (validate_event(existing[0]) is not None
                     or record.get('selected_prediction_event_id') != existing[0]['event_id']):
        return {'status': 'rejected', 'reason': 'existing_event_integrity_failure', 'appended': False}
    for previous in existing:
        if previous.get('input_hash') == input_hash:
            if validate_event(previous) is not None:
                return {'status': 'rejected', 'reason': 'existing_event_integrity_failure', 'appended': False}
            return {'status': 'duplicate', 'event_id': previous['event_id'], 'appended': False}
    event.update(recorded_at=now.astimezone(timezone.utc).isoformat(), frozen=True,
                 selection_policy=SELECTION_POLICY, input_hash=input_hash)
    event['hash'] = event_hash(event)
    event['event_id'] = event['hash']
    reason = validate_event(event)
    if reason:
        return {'status': 'rejected', 'reason': reason, 'appended': False}
    record['prediction_events'] = [*existing, event]
    if not existing:
        record['selected_prediction_event_id'] = event['event_id']
        record['prediction_event_selection_policy'] = SELECTION_POLICY
    return {'status': 'frozen', 'event_id': event['event_id'], 'appended': True}
