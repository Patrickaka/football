#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 data/ 下的 football-data.co.uk CSV 导入 MySQL。

写入两张表：
- matches         经 match_store 幂等 UPSERT。
- similar_market  复用 football.similar_market 解析器全量重建（幂等）。

幂等可重跑。similar_market 全量重建（先清空再写），内容由 data/ 下 CSV 决定。

用法：
    export MYSQL_HOST=... MYSQL_PORT=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DB=football
    python import_csv_to_mysql.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import db, match_store, repositories as repo

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def import_matches():
    total = 0
    for path in sorted(DATA.glob('*.csv')):
        league_code = path.stem.split('_')[0]
        total += match_store.upsert_csv_file(str(path), league_code)
    return total


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
