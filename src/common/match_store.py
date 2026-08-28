# -*- coding: utf-8 -*-
"""matches 表读写适配器：CSV 行 ↔ matches 行 ↔ 还原 CSV row dict。

读路径将 matches 记录还原成「原始 CSV 列名为键、值为字符串」的 row dict，
供 market_db / similar_market 的现有解析器复用。
"""
import csv
import json
from datetime import datetime

from . import db

LEAGUE_MAP = {
    'E0': 'Premier League',
    'D1': 'Bundesliga',
    'I1': 'Serie A',
    'F1': 'Ligue 1',
    'SP1': 'La Liga',
}

_CORE_COLS = {
    'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee',
}
_STATS_INT_COLS = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
_STATS_INT_SET = frozenset(_STATS_INT_COLS)

MATCHES_COLS = [
    'match_id', 'league', 'league_code', 'match_date', 'match_time',
    'home_team', 'away_team', 'fthg', 'ftag', 'ftr', 'hthg', 'htag', 'htr',
    'odds', 'stats', 'settled', 'created_at', 'updated_at',
]

_MATCHES_UPSERT = (
    "INSERT INTO matches (" + ",".join(MATCHES_COLS) + ") "
    "VALUES (" + ",".join(["%s"] * len(MATCHES_COLS)) + ") "
    "ON DUPLICATE KEY UPDATE "
    "fthg=VALUES(fthg), ftag=VALUES(ftag), ftr=VALUES(ftr), "
    "hthg=VALUES(hthg), htag=VALUES(htag), htr=VALUES(htr), "
    "odds=VALUES(odds), stats=VALUES(stats), settled=VALUES(settled), "
    "updated_at=VALUES(updated_at)"
)


def _parse_date(value):
    text = (value or '').strip()
    for fmt in ('%d/%m/%Y', '%d/%m/%y'):
        try:
            return datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def _parse_time(value):
    text = (value or '').strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, '%H:%M').strftime('%H:%M:%S')
    except ValueError:
        return None


def _to_int(value):
    text = (value or '').strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value):
    text = (value or '').strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _clean_str(value):
    text = (value or '').strip()
    return text or None


def _build_odds(row):
    odds = {}
    for key, value in row.items():
        if not key or key in _CORE_COLS or key in _STATS_INT_SET:
            continue
        parsed = _to_float(value)
        if parsed is not None:
            odds[key] = parsed
    return odds


def _build_stats(row):
    stats = {}
    referee = _clean_str(row.get('Referee'))
    if referee:
        stats['referee'] = referee
    for key in _STATS_INT_COLS:
        parsed = _to_int(row.get(key))
        if parsed is not None:
            stats[key.lower()] = parsed
    return stats


def build_match_row(row, league_code, now):
    """CSV row dict → matches 行元组（按 MATCHES_COLS 列序）。缺日期/队名返回 None。"""
    code = (_clean_str(row.get('Div')) or league_code)
    match_date = _parse_date(row.get('Date'))
    home_team = _clean_str(row.get('HomeTeam'))
    away_team = _clean_str(row.get('AwayTeam'))
    if not match_date or not home_team or not away_team:
        return None

    match_id = f"{code}_{match_date}_{home_team}_{away_team}".replace(' ', '_')
    fthg = _to_int(row.get('FTHG'))
    ftag = _to_int(row.get('FTAG'))
    return (
        match_id,
        LEAGUE_MAP.get(code, code),
        code,
        match_date,
        _parse_time(row.get('Time')),
        home_team,
        away_team,
        fthg,
        ftag,
        _clean_str(row.get('FTR')),
        _to_int(row.get('HTHG')),
        _to_int(row.get('HTAG')),
        _clean_str(row.get('HTR')),
        json.dumps(_build_odds(row), ensure_ascii=False),
        json.dumps(_build_stats(row), ensure_ascii=False),
        1 if fthg is not None and ftag is not None else 0,
        now,
        now,
    )


def _num_str(value):
    return '' if value is None else str(value)


def _loads(value):
    if not value:
        return {}
    return value if isinstance(value, dict) else json.loads(value)


def record_to_csv_row(rec):
    """matches 记录（已格式化 Date/Time 为字符串）→「原始 CSV 列名」row dict，值全为字符串。"""
    row = {
        'Div': rec.get('league_code') or '',
        'Date': rec.get('match_date') or '',
        'Time': rec.get('match_time') or '',
        'HomeTeam': rec.get('home_team') or '',
        'AwayTeam': rec.get('away_team') or '',
        'FTHG': _num_str(rec.get('fthg')),
        'FTAG': _num_str(rec.get('ftag')),
        'FTR': rec.get('ftr') or '',
        'HTHG': _num_str(rec.get('hthg')),
        'HTAG': _num_str(rec.get('htag')),
        'HTR': rec.get('htr') or '',
    }
    for key, value in _loads(rec.get('odds')).items():
        row[key] = _num_str(value)
    for key, value in _loads(rec.get('stats')).items():
        if key == 'referee':
            row['Referee'] = _num_str(value)
        else:
            row[key.upper()] = _num_str(value)
    return row


def season_from_date(date_str):
    """'DD/MM/YYYY' → 赛季标签（8月起算下一赛季），如 15/08/2025 → '2526'。"""
    d = datetime.strptime(date_str, '%d/%m/%Y')
    y = d.year % 100
    if d.month >= 8:
        return f"{y:02d}{(y + 1) % 100:02d}"
    return f"{(y - 1) % 100:02d}{y:02d}"


_SELECT_SQL = (
    "SELECT league_code, "
    "DATE_FORMAT(match_date, '%%d/%%m/%%Y') AS match_date, "
    "TIME_FORMAT(match_time, '%%H:%%i') AS match_time, "
    "home_team, away_team, fthg, ftag, ftr, hthg, htag, htr, odds, stats "
    "FROM matches"
)


def upsert_rows(rows):
    """批量 upsert matches 行（按 match_id 幂等）。返回处理行数。"""
    return db.execute_many(_MATCHES_UPSERT, rows)


def upsert_csv_file(path, league_code):
    """解析一个 CSV 文件并 upsert 进 matches。返回写入行数。"""
    now = datetime.now().isoformat()
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            built = build_match_row(row, league_code, now)
            if built:
                rows.append(built)
    if not rows:
        return 0
    return upsert_rows(rows)


def iter_csv_rows(league_code=None):
    """从 matches 还原「CSV 列名」row dict，逐条 yield。league_code 为 None 遍历全部。"""
    if league_code:
        recs = db.query(_SELECT_SQL + " WHERE league_code=%s ORDER BY matches.match_date, matches.id",
                        (league_code,))
    else:
        recs = db.query(_SELECT_SQL + " ORDER BY matches.match_date, matches.id")
    for rec in recs:
        yield record_to_csv_row(rec)
