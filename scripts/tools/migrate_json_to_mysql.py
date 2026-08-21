#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性迁移：把现有 JSON 数据导入 MySQL。

特性：
- 幂等可重跑（规范化表全量替换、kv UPSERT）。
- 原 JSON 文件保留作备份，确认无误后再手动清理。

用法：
    export MYSQL_HOST=... MYSQL_PORT=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DB=football
    python migrate_json_to_mysql.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import db, kv_store, repositories as repo

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
FB_DATA = ROOT / 'src' / 'football' / 'data'


def _read(path):
    """读取 JSON 文件，不存在返回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------- 规范化表 ----------

def migrate_football_prediction():
    data = _read(FB_DATA / 'prediction_history.json')
    if data is None:
        return 0
    repo.football_prediction_save(data)
    return len(data)


def migrate_prediction_records():
    data = _read(FB_DATA / 'prediction_records.json')
    if data is None:
        return 0
    repo.prediction_record_save(data)
    return len(data)


def migrate_dlt():
    data = _read(DATA / 'lottery_history.json')
    if data is None:
        return 0
    results = data.get('results', [])
    repo.dlt_save(results)
    return len(results)


def migrate_pailie5():
    data = _read(DATA / 'pailie5_history.json')
    if data is None:
        return 0
    repo.pailie5_save(data)
    return len(data)


def migrate_elo():
    data = _read(DATA / 'elo_ratings.json')
    if data is None:
        return 0
    repo.elo_save(data)
    return len(data.get('ratings', {}))


def migrate_similar_market():
    data = _read(DATA / 'similar_market_db.json')
    if data is None:
        return 0
    repo.similar_market_save(data)
    return len(data.get('records', []))


# ---------- kv_store 普通项 ----------

_KV_ITEMS = {
    'market_score_db': DATA / 'market_score_db.json',
    'market_change_db': DATA / 'market_change_db.json',
    'calibration_db': DATA / 'calibration_db.json',
    'market_cluster_db': DATA / 'market_cluster_db.json',
    'ml_metadata': DATA / 'ml_metadata.json',
    'backtest_config': DATA / 'backtest_config.json',
    'team_alias': FB_DATA.parent / 'team_alias.json',
    'lottery3d_recent_recommend': DATA / 'lottery3d_recent_recommend.json',
    'lottery3d_online_predictions': DATA / 'lottery3d_online_predictions.json',
}


def migrate_kv_items():
    n = 0
    for key, path in _KV_ITEMS.items():
        data = _read(path)
        if data is None:
            continue
        kv_store.save(key, data)
        n += 1
    return n


# ---------- kv_store 缓存项（保留原 date） ----------

_CACHE_FILES = {
    'lottery3d': DATA / 'lottery3d_cache.json',
    'lottery3d_ml': DATA / 'lottery3d_ml_cache.json',
    'pailie5': DATA / 'pailie5_cache.json',
    'lottery': DATA / 'lottery_cache.json',
}

_CACHE_UPSERT = (
    "INSERT INTO kv_store (k, json_value, cache_date, updated_at) VALUES (%s,%s,%s,%s) "
    "ON DUPLICATE KEY UPDATE json_value=VALUES(json_value), "
    "cache_date=VALUES(cache_date), updated_at=VALUES(updated_at)"
)


def migrate_caches():
    n = 0
    for key, path in _CACHE_FILES.items():
        wrapped = _read(path)
        if wrapped is None:
            continue
        db.execute(_CACHE_UPSERT, (
            key,
            json.dumps(wrapped.get('data'), ensure_ascii=False),
            wrapped.get('date'),
            datetime.now().isoformat(),
        ))
        n += 1
    return n


_STEPS = [
    ('football_prediction', migrate_football_prediction),
    ('football_prediction_record', migrate_prediction_records),
    ('dlt_history', migrate_dlt),
    ('pailie5_history', migrate_pailie5),
    ('elo', migrate_elo),
    ('similar_market', migrate_similar_market),
    ('kv_items', migrate_kv_items),
    ('caches', migrate_caches),
]


def run():
    db.init_db()
    print("建表完成，开始迁移……")
    for name, fn in _STEPS:
        count = fn()
        print(f"  {name:30s} {count} 条")
    print("迁移完成。原 JSON 文件已保留，请校验无误后再清理。")


if __name__ == '__main__':
    run()
