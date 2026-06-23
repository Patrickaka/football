"""通用键值存储（kv_store 表）。

承载形态杂、无需 SQL 内部查询的数据：配置、统计 blob、每日缓存等。
- 普通项：load/save，cache_date 为空。
- 缓存项：save_cache/load_cache，带「今天」失效语义，替代原 data_cache 的文件实现。
"""
import json
from datetime import datetime

from . import db

_UPSERT = (
    "INSERT INTO kv_store (k, json_value, cache_date, updated_at) "
    "VALUES (%s, %s, %s, %s) "
    "ON DUPLICATE KEY UPDATE "
    "json_value=VALUES(json_value), cache_date=VALUES(cache_date), updated_at=VALUES(updated_at)"
)


def _now():
    return datetime.now().isoformat()


def _today():
    return datetime.now().strftime('%Y-%m-%d')


def load(key, default=None):
    """读取普通项，反序列化 JSON；不存在返回 default。"""
    row = db.query_one("SELECT json_value FROM kv_store WHERE k=%s", (key,))
    if row is None or row['json_value'] is None:
        return default
    return json.loads(row['json_value'])


def save(key, obj):
    """写入普通项（UPSERT）。"""
    db.execute(_UPSERT, (key, json.dumps(obj, ensure_ascii=False), None, _now()))


def exists(key):
    return db.query_one("SELECT 1 FROM kv_store WHERE k=%s", (key,)) is not None


def delete(key):
    db.execute("DELETE FROM kv_store WHERE k=%s", (key,))


def save_cache(key, data):
    """写入缓存项，标记为今天。"""
    db.execute(_UPSERT, (key, json.dumps(data, ensure_ascii=False), _today(), _now()))


def load_cache(key):
    """读取缓存项；仅当 cache_date 为今天时返回数据，否则 None。"""
    row = db.query_one(
        "SELECT json_value, cache_date FROM kv_store WHERE k=%s", (key,)
    )
    if row is None or row['cache_date'] != _today():
        return None
    return json.loads(row['json_value'])
