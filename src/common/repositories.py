"""各业务表的仓储函数。

每个 load/save 精确还原对应模块原本的内存结构与 JSON 形态，
使调用方只需替换持久化内部、业务逻辑零改动。
"""
import json
import os
import tempfile

from . import db
from . import doc_store
from .football_storage import (FootballStorageError, decode_record, encode_record,
                               is_archived, serialized_size_bytes)

_J = lambda o: json.dumps(o, ensure_ascii=False)


# ==================== football_prediction（result_sync） ====================

_FOOTBALL_PREDICTION_COLS = [
    'match_id', 'league', 'settled', 'sync_status', 'created_at', 'updated_at', 'doc',
]


def _football_prediction_row(r):
    stored = encode_record(r)
    return (
        r.get('match_id'), r.get('league'), 1 if r.get('settled') else 0,
        r.get('sync_status'), r.get('created_at'), r.get('updated_at'), _J(stored),
    )


def football_prediction_load(transform=None):
    """整表读取；`transform` 对每条解码后的记录逐行执行，用于加载即精简。"""
    transform = transform or (lambda record: record)
    return doc_store.load_all(
        'football_prediction', order_by='created_at, match_id',
        transform=lambda stored: transform(decode_record(stored)))


def football_prediction_save(records):
    rows = [_football_prediction_row(r) for r in records]
    doc_store.replace_all('football_prediction', _FOOTBALL_PREDICTION_COLS, rows)


def football_prediction_get(match_id):
    """按 match_id 读回库里的完整记录，供内存精简副本按需补全。"""
    return decode_record(doc_store.load_one('football_prediction', 'match_id', match_id))


def football_prediction_upsert(record):
    """单行 UPSERT 一条预测记录，避免整表重写（每请求级热点写入）。"""
    return doc_store.upsert_one(
        'football_prediction',
        _FOOTBALL_PREDICTION_COLS,
        _football_prediction_row(record),
        key_cols=['match_id'],
    )


def _football_archive_candidate(record, *, now, archive_after_days, dry_run, backend):
    before = serialized_size_bytes(record)
    report = {'status': 'not_eligible', 'backend': backend, 'before_bytes': before,
              'after_bytes': before, 'bytes_saved': 0}
    if is_archived(record):
        decode_record(record)
        return {**report, 'status': 'already_archived'}, None
    encoded = encode_record(record, now=now, archive_after_days=archive_after_days)
    if not is_archived(encoded):
        return report, None
    after = serialized_size_bytes(encoded)
    return {**report, 'status': 'would_archive' if dry_run else 'archived',
            'after_bytes': after, 'bytes_saved': before - after}, encoded


def _football_archive_fallback(match_id, *, now, archive_after_days, dry_run):
    """Caller holds PredictionHistory._records_lock across this local-file update."""
    path = doc_store._fallback_path('football_prediction')
    if not path.exists():
        return {'status': 'missing', 'backend': 'fallback', 'before_bytes': 0,
                'after_bytes': 0, 'bytes_saved': 0}
    original = path.read_bytes()
    try:
        document = json.loads(original.decode('utf-8-sig'))
    except (ValueError, UnicodeError) as exc:
        raise FootballStorageError('Invalid football fallback JSON; archive was not written') from exc
    records = document.get('records') if isinstance(document, dict) else document
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise FootballStorageError('Invalid football fallback records; archive was not written')
    matches = [i for i, record in enumerate(records) if record.get('match_id') == match_id]
    if len(matches) > 1:
        raise FootballStorageError('Duplicate football fallback match ID; archive was not written')
    if not matches:
        return {'status': 'missing', 'backend': 'fallback', 'before_bytes': 0,
                'after_bytes': 0, 'bytes_saved': 0}
    index = matches[0]
    report, encoded = _football_archive_candidate(records[index], now=now,
                                                  archive_after_days=archive_after_days,
                                                  dry_run=dry_run, backend='fallback')
    if encoded is None or dry_run:
        return report
    records[index] = encoded
    payload = json.dumps(document, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Detect external changes as well as obeying the process-level history
        # lock; never replace a snapshot that changed since it was inspected.
        if path.read_bytes() != original:
            return {**report, 'status': 'changed', 'after_bytes': report['before_bytes'], 'bytes_saved': 0}
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return report


def football_prediction_archive_one(match_id, *, now=None, archive_after_days=None, dry_run=False):
    """Archive one stored document without overwriting concurrent result changes.

    MySQL uses a binary comparison against the exact JSON text read from the
    server. A failed/uncertain UPDATE never falls back to writing a stale copy.
    The maintenance caller must hold PredictionHistory._records_lock, including
    during fallback file operations. No business timestamps or indices change.
    """
    backend = 'mysql'
    empty = {'match_id': match_id, 'before_bytes': 0, 'after_bytes': 0, 'bytes_saved': 0}
    try:
        try:
            row = db.query_one('SELECT CAST(doc AS CHAR CHARACTER SET utf8mb4) AS raw_doc '
                               'FROM football_prediction WHERE match_id=%s', (match_id,))
        except Exception as exc:
            backend = 'fallback'
            doc_store._record_degradation('football_prediction', exc, operation='archive')
            return {**empty, **_football_archive_fallback(match_id, now=now,
                                                          archive_after_days=archive_after_days,
                                                          dry_run=dry_run)}
        if row is None:
            return {**empty, 'status': 'missing', 'backend': backend}
        raw = row['raw_doc']
        record = json.loads(raw)
        if not isinstance(record, dict) or record.get('match_id') != match_id:
            raise FootballStorageError('Football document identity mismatch; archive was not written')
        report, encoded = _football_archive_candidate(record, now=now,
                                                      archive_after_days=archive_after_days,
                                                      dry_run=dry_run, backend=backend)
        if encoded is None or dry_run:
            return {**empty, **report}
        affected = db.execute(
            'UPDATE football_prediction SET doc=%s WHERE match_id=%s '
            'AND CAST(CAST(doc AS CHAR CHARACTER SET utf8mb4) AS BINARY)=CAST(%s AS BINARY)',
            (_J(encoded), match_id, raw),
        )
        if affected != 1:
            report.update(status='changed', after_bytes=report['before_bytes'], bytes_saved=0)
        return {**empty, **report}
    except Exception as exc:
        # Maintenance reports an explicit failure; ordinary repository reads
        # continue to raise FootballStorageError on corrupt archive payloads.
        return {**empty, 'status': 'failed', 'backend': backend,
                'error': str(exc) if isinstance(exc, FootballStorageError) else type(exc).__name__}


# ==================== football_prediction_record（prediction_records） ====================

def prediction_record_load():
    return doc_store.load_all('football_prediction_record', order_by='id')


def prediction_record_save(records):
    cols = ['match_id', 'league', 'model_version', 'created_at', 'doc']
    rows = [
        (r.get('match_id'), r.get('league'), r.get('model_version'), r.get('created_at'), _J(r))
        for r in records
    ]
    doc_store.replace_all('football_prediction_record', cols, rows)


# ==================== elo_rating + elo_history ====================

def elo_load():
    ratings = {r['team']: r['rating'] for r in db.query("SELECT team, rating FROM elo_rating")}
    history = {}
    for r in db.query("SELECT team, rating, date, event FROM elo_history ORDER BY id"):
        history.setdefault(r['team'], []).append(
            {'rating': r['rating'], 'date': r['date'], 'event': r['event']}
        )
    row = db.query_one("SELECT MAX(updated_at) AS u FROM elo_rating")
    return {'ratings': ratings, 'history': history, 'updated_at': row['u'] if row else None}


def _entry_tuple(entry):
    return (entry.get('rating'), entry.get('date'), entry.get('event'))


def _sync_elo_ratings(cur, ratings, updated):
    """评分表按主键 UPSERT，只删真正消失的球队。"""
    if ratings:
        cur.executemany(
            "INSERT INTO elo_rating (team, rating, updated_at) VALUES (%s,%s,%s)"
            " ON DUPLICATE KEY UPDATE rating=VALUES(rating), updated_at=VALUES(updated_at)",
            [(t, v, updated) for t, v in ratings.items()],
        )
    stored = {r['team'] for r in db.query("SELECT team FROM elo_rating")}
    stale = sorted(stored - set(ratings))
    if stale:
        placeholders = ",".join(["%s"] * len(stale))
        cur.execute(f"DELETE FROM elo_rating WHERE team IN ({placeholders})", tuple(stale))


def _sync_elo_history(cur, history):
    """轨迹表按球队比对，只重写内容真的变了的那几支球队。

    一场比赛只动两支球队，整表重写却要把上千行删了再写回去。
    """
    stored = {}
    for row in db.query("SELECT team, rating, date, event FROM elo_history ORDER BY id"):
        stored.setdefault(row['team'], []).append((row['rating'], row['date'], row['event']))

    for team, entries in history.items():
        desired = [_entry_tuple(e) for e in entries]
        if stored.get(team, []) == desired:
            continue
        cur.execute("DELETE FROM elo_history WHERE team=%s", (team,))
        if desired:
            cur.executemany(
                "INSERT INTO elo_history (team, rating, date, event) VALUES (%s,%s,%s,%s)",
                [(team, *entry) for entry in desired],
            )

    for team in sorted(set(stored) - set(history)):
        cur.execute("DELETE FROM elo_history WHERE team=%s", (team,))


def elo_save(data):
    """增量保存 ELO：整表重写会把每场比赛的两行改动放大成上千行 binlog。"""
    conn = db.get_connection()
    try:
        conn.begin()
        with conn.cursor() as cur:
            _sync_elo_ratings(cur, data.get('ratings', {}), data.get('updated_at'))
            _sync_elo_history(cur, data.get('history', {}))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ==================== similar_market（相似盘口库） ====================

SIMILAR_COLS = [
    'asian', 'asian_odds_home', 'asian_odds_away', 'total', 'total_over', 'total_under',
    'euro_home', 'euro_draw', 'euro_away', 'result', 'goals_home', 'goals_away',
    'date', 'league', 'home_team', 'away_team',
]


def similar_market_load():
    rows = db.query(f"SELECT {','.join(SIMILAR_COLS)} FROM similar_market ORDER BY id")
    records = [dict(r) for r in rows]
    return {'records': records, 'version': '1.0', 'count': len(records)}


def similar_market_save(data):
    records = data.get('records', [])
    rows = [tuple(rec.get(c) for c in SIMILAR_COLS) for rec in records]
    doc_store.sync_append_only('similar_market', SIMILAR_COLS, rows)
