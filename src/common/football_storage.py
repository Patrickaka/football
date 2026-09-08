"""Lossless, self-contained archives for large settled football documents.

Only repository documents use this storage representation. Business code and
exports receive the original JSON fields, including immutable prediction events.
"""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
import re
import zlib


ARCHIVE_KEY = '_football_storage_archive'
ARCHIVE_MANIFEST_KEY = '_football_storage_archive_manifest'
ARCHIVE_SCHEMA = 'football-storage-archive-v1'
ARCHIVE_FIELDS = frozenset({
    'prediction_events', 'prediction_event_archive', 'market_timeline',
    'professional_snapshot', 'time_layers', 'odds_layers', 'odds_snapshot',
    'last_prematch_odds_snapshot', 'closing_odds_snapshot',
})
MAX_DECOMPRESSED_BYTES = 32 * 1024 * 1024
DEFAULT_ARCHIVE_AFTER_DAYS = 30
MIN_SAVED_BYTES = 1024
MIN_SAVING_RATIO = .10


class FootballStorageError(ValueError):
    """Stored football data could not be verified; callers must not discard it."""


def _json_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise FootballStorageError('Football archive requires valid finite JSON data') from exc


def serialized_size_bytes(value):
    """UTF-8 bytes of compact JSON, for comparable dry-run storage estimates."""
    return len(_json_bytes(value))


def _limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_DECOMPRESSED_BYTES:
        raise FootballStorageError('Invalid football archive decompression limit')
    return value


def encode_json_gzip(value, *, max_uncompressed_bytes=MAX_DECOMPRESSED_BYTES):
    """Encode JSON without pickle or external files; metadata is integrity-checked on read."""
    limit = _limit(max_uncompressed_bytes)
    raw = _json_bytes(value)
    if len(raw) > limit:
        raise FootballStorageError('Football archive exceeds its uncompressed size limit')
    return {
        'encoding': 'gzip+base64',
        'sha256': hashlib.sha256(raw).hexdigest(),
        'uncompressed_bytes': len(raw),
        'data': base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode('ascii'),
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FootballStorageError('Football archive contains duplicate JSON keys')
        result[key] = value
    return result


def _invalid_constant(value):
    raise FootballStorageError('Football archive contains a non-finite JSON number')


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise FootballStorageError('Football archive contains a non-finite JSON number')
    return number


def decode_json_gzip(envelope, *, max_uncompressed_bytes=MAX_DECOMPRESSED_BYTES):
    """Validate size, base64, gzip CRC and SHA-256 before restoring bounded JSON.

    Extra outer schema/count fields are allowed for event-specific envelopes.
    Concatenated gzip members and trailing data are rejected.
    """
    limit = _limit(max_uncompressed_bytes)
    if not isinstance(envelope, dict) or envelope.get('encoding') != 'gzip+base64':
        raise FootballStorageError('Unsupported football archive encoding')
    size, digest, encoded = (envelope.get(k) for k in ('uncompressed_bytes', 'sha256', 'data'))
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= limit:
        raise FootballStorageError('Invalid football archive uncompressed length')
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise FootballStorageError('Invalid football archive SHA-256 metadata')
    # Bound compressed input before decoding it; gzip framing/deflate can add a
    # small overhead but cannot justify arbitrarily large encoded documents.
    encoded_limit = ((limit + 65536 + 2) // 3) * 4
    if not isinstance(encoded, str) or len(encoded) > encoded_limit:
        raise FootballStorageError('Invalid football archive encoded length')
    try:
        packed = base64.b64decode(encoded, validate=True)
        inflater = zlib.decompressobj(wbits=31)
        raw = inflater.decompress(packed, size + 1)
        if (len(raw) != size or not inflater.eof or inflater.unconsumed_tail or inflater.unused_data):
            raise FootballStorageError('Football archive length or gzip framing mismatch')
        if hashlib.sha256(raw).hexdigest() != digest:
            raise FootballStorageError('Football archive checksum mismatch')
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant, parse_float=_finite_float)
    except FootballStorageError:
        raise
    except (binascii.Error, zlib.error, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise FootballStorageError('Corrupt football archive payload') from exc


def is_archived(record):
    """Detect the storage marker; decode_record performs integrity validation."""
    return isinstance(record, dict) and (ARCHIVE_KEY in record or ARCHIVE_MANIFEST_KEY in record)


def decode_record(record):
    """Return a complete record, or None for a missing repository row.

    Invalid archives raise instead of returning a partial or empty record.
    Neither the input document nor any frozen event is modified.
    """
    if record is None:
        return None
    if not isinstance(record, dict):
        raise FootballStorageError('Football repository record must be a JSON object')
    if not is_archived(record):
        return dict(record)
    envelope = record.get(ARCHIVE_KEY)
    if not isinstance(envelope, dict) or envelope.get('schema_version') != ARCHIVE_SCHEMA:
        raise FootballStorageError('Unsupported football storage archive schema')
    manifest = record.get(ARCHIVE_MANIFEST_KEY)
    expected_manifest = {key: envelope.get(key) for key in ('schema_version', 'fields', 'sha256')}
    if not isinstance(manifest, dict) or manifest != expected_manifest:
        raise FootballStorageError('Missing or inconsistent football storage archive manifest')
    fields = envelope.get('fields')
    if (not isinstance(fields, list) or not fields or
            any(not isinstance(field, str) or field not in ARCHIVE_FIELDS for field in fields)
            or len(set(fields)) != len(fields)):
        raise FootballStorageError('Invalid football storage archive field manifest')
    if any(field in record for field in fields):
        raise FootballStorageError('Football storage archive conflicts with live fields')
    restored = decode_json_gzip(envelope)
    if not isinstance(restored, dict) or set(restored) != set(fields):
        raise FootballStorageError('Football storage archive field manifest mismatch')
    return {**{key: value for key, value in record.items()
               if key not in (ARCHIVE_KEY, ARCHIVE_MANIFEST_KEY)}, **restored}


def _aware_time(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _archive_days(value):
    if value is None:
        value = os.getenv('FOOTBALL_ARCHIVE_AFTER_DAYS', str(DEFAULT_ARCHIVE_AFTER_DAYS))
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise FootballStorageError('FOOTBALL_ARCHIVE_AFTER_DAYS must be a nonnegative integer') from exc
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 36500:
        raise FootballStorageError('Football archive age must be an integer from 0 to 36500 days')
    return value


def encode_record(record, *, now=None, archive_after_days=None):
    """Archive old settled payloads only when at least 1 KiB and 10% are saved.

    An already archived input is fully validated and preserved, so maintenance
    can skip unchanged rows. New archives require an explicit settlement time
    with timezone. Missing/future/naive settlement dates stay uncompressed.
    """
    if not isinstance(record, dict):
        raise FootballStorageError('Football repository record must be a JSON object')
    if record.get('_timeline_offloaded'):
        raise FootballStorageError('Hydrate the offloaded football timeline before saving')
    if is_archived(record):
        decode_record(record)
        return dict(record)
    days = _archive_days(archive_after_days)
    current = datetime.now(timezone.utc) if now is None else _aware_time(now)
    if current is None:
        raise FootballStorageError('Football archive current time must include a timezone')
    settled = _aware_time(record.get('settled_at'))
    if (record.get('settled') is not True or settled is None or settled > current
            or current - settled < timedelta(days=days)):
        return dict(record)
    fields = sorted(ARCHIVE_FIELDS.intersection(record))
    if not fields:
        return dict(record)
    payload = {field: record[field] for field in fields}
    size = serialized_size_bytes(payload)
    if size > MAX_DECOMPRESSED_BYTES:
        # Leave oversized but otherwise valid legacy documents readable in the
        # original representation instead of producing an unreadable archive.
        return dict(record)
    archived = {key: value for key, value in record.items() if key not in fields}
    archived[ARCHIVE_KEY] = {'schema_version': ARCHIVE_SCHEMA, 'fields': fields,
                             **encode_json_gzip(payload)}
    archived[ARCHIVE_MANIFEST_KEY] = {key: archived[ARCHIVE_KEY][key]
                                    for key in ('schema_version', 'fields', 'sha256')}
    original_size, stored_size = serialized_size_bytes(record), serialized_size_bytes(archived)
    if original_size - stored_size < max(MIN_SAVED_BYTES, original_size * MIN_SAVING_RATIO):
        return dict(record)
    return archived
