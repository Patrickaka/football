"""各业务表的仓储函数。

每个 load/save 精确还原对应模块原本的内存结构与 JSON 形态，
使调用方只需替换持久化内部、业务逻辑零改动。
"""
import json

from . import db
from . import doc_store

_J = lambda o: json.dumps(o, ensure_ascii=False)


# ==================== football_prediction（result_sync） ====================

_FOOTBALL_PREDICTION_COLS = [
    'match_id', 'league', 'settled', 'sync_status', 'created_at', 'updated_at', 'doc',
]


def _football_prediction_row(r):
    return (
        r.get('match_id'), r.get('league'), 1 if r.get('settled') else 0,
        r.get('sync_status'), r.get('created_at'), r.get('updated_at'), _J(r),
    )


def football_prediction_load():
    return doc_store.load_all('football_prediction', order_by='created_at, match_id')


def football_prediction_save(records):
    rows = [_football_prediction_row(r) for r in records]
    doc_store.replace_all('football_prediction', _FOOTBALL_PREDICTION_COLS, rows)


def football_prediction_upsert(record):
    """单行 UPSERT 一条预测记录，避免整表重写（每请求级热点写入）。"""
    return doc_store.upsert_one(
        'football_prediction',
        _FOOTBALL_PREDICTION_COLS,
        _football_prediction_row(record),
        key_cols=['match_id'],
    )


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


# ==================== dlt_history（大乐透） ====================

def dlt_load():
    try:
        rows = db.query("SELECT issue, front, back, draw_date FROM dlt_history ORDER BY seq")
        if rows:
            return [
                {'issue': r['issue'], 'front': json.loads(r['front']),
                 'back': json.loads(r['back']), 'date': r['draw_date']}
                for r in rows
            ]
    except Exception:
        pass

    records = doc_store._fallback_load_all('dlt_history')
    if records:
        return [
            {'issue': r['issue'],
             'front': json.loads(r['front']) if isinstance(r['front'], str) else r['front'],
             'back': json.loads(r['back']) if isinstance(r['back'], str) else r['back'],
             'date': r.get('draw_date') or r.get('date')}
            for r in records
        ]

    # 末级降级：读取遗留 JSON（data/lottery_history.json，结构 {'results': [...]}）。
    # MySQL 未安装且 doc_store 尚未填充时，避免回退到模拟随机数据。
    return _dlt_load_legacy_json()


def _dlt_load_legacy_json():
    from .paths import data_path
    import os
    path = data_path('lottery_history.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return []
    results = payload.get('results') if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        return []
    cleaned = []
    for r in results:
        front = r.get('front'); back = r.get('back')
        if isinstance(front, str):
            front = json.loads(front)
        if isinstance(back, str):
            back = json.loads(back)
        if not (front and back):
            continue
        cleaned.append({
            'issue': str(r.get('issue', '')),
            'front': sorted(int(x) for x in front),
            'back': sorted(int(x) for x in back),
            'date': r.get('date') or r.get('draw_date') or '',
        })
    return cleaned


def dlt_save(results):
    cols = ['issue', 'front', 'back', 'draw_date']
    rows = [(r.get('issue'), _J(r.get('front')), _J(r.get('back')), r.get('date')) for r in results]
    doc_store.replace_all('dlt_history', cols, rows)


# ==================== pailie5_history（排列5） ====================

def pailie5_load():
    try:
        rows = db.query("SELECT issue, numbers, draw_date, ts FROM pailie5_history ORDER BY seq")
        return [
            {'issue': r['issue'], 'numbers': json.loads(r['numbers']),
             'date': r['draw_date'], 'timestamp': r['ts']}
            for r in rows
        ]
    except Exception:
        # MySQL 不可用时从 doc_store 文件降级加载
        records = doc_store._fallback_load_all('pailie5_history')
        return [
            {'issue': r['issue'],
             'numbers': json.loads(r['numbers']) if isinstance(r['numbers'], str) else r['numbers'],
             'date': r.get('draw_date') or r.get('date'),
             'timestamp': r.get('ts') or r.get('timestamp')}
            for r in records
        ]


def pailie5_save(history):
    cols = ['issue', 'numbers', 'draw_date', 'ts']
    rows = [(h.get('issue'), _J(h.get('numbers')), h.get('date'), h.get('timestamp')) for h in history]
    doc_store.replace_all('pailie5_history', cols, rows)


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


def elo_save(data):
    ratings = data.get('ratings', {})
    history = data.get('history', {})
    updated = data.get('updated_at')
    rating_rows = [(t, v, updated) for t, v in ratings.items()]
    hist_rows = [
        (t, e.get('rating'), e.get('date'), e.get('event'))
        for t, entries in history.items() for e in entries
    ]
    conn = db.get_connection()
    try:
        conn.begin()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM elo_rating")
            cur.execute("DELETE FROM elo_history")
            if rating_rows:
                cur.executemany(
                    "INSERT INTO elo_rating (team, rating, updated_at) VALUES (%s,%s,%s)", rating_rows
                )
            if hist_rows:
                cur.executemany(
                    "INSERT INTO elo_history (team, rating, date, event) VALUES (%s,%s,%s,%s)", hist_rows
                )
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
    doc_store.replace_all('similar_market', SIMILAR_COLS, rows)
