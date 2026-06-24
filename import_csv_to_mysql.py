#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 data/ 下的 football-data.co.uk CSV 导入 MySQL。

写入两张表：
- matches         全量比赛数据（比分/半场/赔率 JSON/统计 JSON），按 match_id 幂等 UPSERT。
- similar_market  相似盘口样本库，复用 football.similar_market 解析器全量重建（幂等）。

幂等可重跑。注意 similar_market 采用全量重建（先清空再写入），其内容由 data/ 下的 CSV 决定。

用法：
    export MYSQL_HOST=... MYSQL_PORT=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DB=football
    python import_csv_to_mysql.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import db, repositories as repo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

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
        if not key or key in _CORE_COLS or key in _STATS_INT_COLS:
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


def _build_match_row(row, league_code, now):
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


def _iter_csv():
    for path in sorted(DATA.glob('*.csv')):
        league_code = path.stem.split('_')[0]
        with open(path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                yield league_code, row


def import_matches():
    now = datetime.now().isoformat()
    rows = []
    for league_code, row in _iter_csv():
        built = _build_match_row(row, league_code, now)
        if built:
            rows.append(built)
    db.execute_many(_MATCHES_UPSERT, rows)
    return len(rows)


def import_similar_market():
    from src.football import similar_market as sm

    records = []
    for path in sorted(DATA.glob('*.csv')):
        league_code = path.stem.split('_')[0]
        for record in sm.parse_football_data_csv(str(path), league_code):
            records.append(record.to_dict())
    repo.similar_market_save({'records': records})
    return len(records)


def run():
    db.init_db()
    print("建表完成，开始导入 CSV……")
    n_matches = import_matches()
    print(f"  matches         {n_matches} 行（按 match_id 幂等 UPSERT）")
    n_market = import_similar_market()
    print(f"  similar_market  {n_market} 条（全量重建）")
    print("导入完成。")


if __name__ == '__main__':
    run()
