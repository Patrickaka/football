"""通用键值存储（kv_store 表）。

承载形态杂、无需 SQL 内部查询的数据：配置、统计 blob、每日缓存等。
- 普通项：load/save，cache_date 为空。
- 缓存项：save_cache/load_cache，带「今天」失效语义，替代原 data_cache 的文件实现。

MySQL 不可用时自动降级到本地 JSON 文件（kv_store_fallback.json）。
"""
import json
import os
from datetime import datetime

from . import db
from .paths import data_path

_UPSERT = (
    "INSERT INTO kv_store (k, json_value, cache_date, updated_at) "
    "VALUES (%s, %s, %s, %s) "
    "ON DUPLICATE KEY UPDATE "
    "json_value=VALUES(json_value), cache_date=VALUES(cache_date), updated_at=VALUES(updated_at)"
)

_FALLBACK_FILE = None


def _fallback_path():
    global _FALLBACK_FILE
    if _FALLBACK_FILE is None:
        _FALLBACK_FILE = data_path('kv_store_fallback.json')
    return _FALLBACK_FILE


def _fallback_load_all():
    """加载整个 fallback 文件"""
    path = _fallback_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _fallback_save(key, value, cache_date=None):
    """写入单条到 fallback 文件"""
    data = _fallback_load_all()
    data[key] = {'json_value': json.dumps(value, ensure_ascii=False), 'cache_date': cache_date}
    path = _fallback_path()
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _fallback_load(key, default=None, check_today=False):
    """从 fallback 文件读取单条"""
    data = _fallback_load_all()
    entry = data.get(key)
    if entry is None:
        return default
    if check_today:
        if entry.get('cache_date') != _today():
            return None
    return json.loads(entry['json_value'])


def _now():
    return datetime.now().isoformat()


def _today():
    return datetime.now().strftime('%Y-%m-%d')


def load(key, default=None):
    """读取普通项，反序列化 JSON；不存在返回 default。"""
    try:
        row = db.query_one("SELECT json_value FROM kv_store WHERE k=%s", (key,))
        if row is None or row['json_value'] is None:
            return default
        return json.loads(row['json_value'])
    except Exception:
        return _fallback_load(key, default)


def save(key, obj):
    """写入普通项（UPSERT）。"""
    try:
        db.execute(_UPSERT, (key, json.dumps(obj, ensure_ascii=False), None, _now()))
    except Exception:
        _fallback_save(key, obj)


def exists(key):
    try:
        return db.query_one("SELECT 1 FROM kv_store WHERE k=%s", (key,)) is not None
    except Exception:
        return key in _fallback_load_all()


def delete(key):
    try:
        db.execute("DELETE FROM kv_store WHERE k=%s", (key,))
    except Exception:
        data = _fallback_load_all()
        data.pop(key, None)
        path = _fallback_path()
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)


def save_cache(key, data):
    """写入缓存项，标记为今天。"""
    try:
        db.execute(_UPSERT, (key, json.dumps(data, ensure_ascii=False), _today(), _now()))
    except Exception:
        _fallback_save(key, data, cache_date=_today())


def load_cache(key):
    """读取缓存项；仅当 cache_date 为今天时返回数据，否则 None。"""
    try:
        row = db.query_one(
            "SELECT json_value, cache_date FROM kv_store WHERE k=%s", (key,)
        )
        if row is None or row['cache_date'] != _today():
            return None
        return json.loads(row['json_value'])
    except Exception:
        return _fallback_load(key, None, check_today=True)
