#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 data/ 下的 football-data.co.uk CSV 导入 MySQL。

写入两张表：
- matches         由 data/ 下 CSV 经 match_store 幂等 UPSERT 写入。
- similar_market  从 matches 表全量重建（先清空再写，幂等），不依赖 CSV。

幂等可重跑。matches 是 similar_market 的数据源，故删除历史 CSV 不影响 similar_market 重建。

用法：
    export MYSQL_HOST=... MYSQL_PORT=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DB=football
    python import_csv_to_mysql.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import db, match_store

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'


def import_matches():
    total = 0
    for path in sorted(DATA.glob('*.csv')):
        league_code = path.stem.split('_')[0]
        total += match_store.upsert_csv_file(str(path), league_code)
    return total


def import_similar_market():
    from src.football import similar_market as sm

    sdb = sm.SimilarMarketDB()
    sdb.records.clear()
    sm.build_from_matches(sdb)
    sdb.save()
    return len(sdb.records)


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
