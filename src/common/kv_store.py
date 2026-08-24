"""通用键值存储（kv_store 表）。

承载形态杂、无需 SQL 内部查询的数据：配置、统计 blob、每日缓存等。
- 普通项：load/save，cache_date 为空。
- 缓存项：save_cache/load_cache，带「今天」失效语义，替代原 data_cache 的文件实现。

MySQL 不可用时自动降级到本地 JSON 文件（kv_store_fallback.json）。
"""
import json
import os
import threading
import time
from datetime import datetime

from . import db
from .logger import setup_logger
from .paths import data_path

log = setup_logger('kv_store')

_UPSERT = (
    "INSERT INTO kv_store (k, json_value, cache_date, updated_at) "
    "VALUES (%s, %s, %s, %s) "
    "ON DUPLICATE KEY UPDATE "
    "json_value=VALUES(json_value), cache_date=VALUES(cache_date), updated_at=VALUES(updated_at)"
)

_FALLBACK_FILE = None
_FALLBACK_CACHE = None
_FALLBACK_CACHE_SIG = None
# 串行化所有 fallback 写操作：所有 key 共用同一个 JSON 文件，并发写会互相覆盖/竞争 .tmp。
_FALLBACK_WRITE_LOCK = threading.Lock()


def _fallback_path():
    global _FALLBACK_FILE
    if _FALLBACK_FILE is None:
        _FALLBACK_FILE = data_path('kv_store_fallback.json')
    return _FALLBACK_FILE


def _fallback_load_all():
    """加载整个 fallback 文件。

    带进程内缓存（按文件 mtime+size 失效）：MySQL 不可用时，load() 会被
    大量调用（每次预测多次），若每次都重新解析整份 JSON（可达数百 KB）会
    造成严重 I/O 开销。缓存后同一份文件只解析一次，写入时失效。
    """
    global _FALLBACK_CACHE, _FALLBACK_CACHE_SIG
    path = _fallback_path()
    if not os.path.exists(path):
        return {}
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_size)
        if _FALLBACK_CACHE is not None and _FALLBACK_CACHE_SIG == sig:
            return _FALLBACK_CACHE
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _FALLBACK_CACHE = data
        _FALLBACK_CACHE_SIG = sig
        return data
    except Exception:
        return {}


def _atomic_replace(path, data_obj):
    """Windows / 线程安全的原子 JSON 写入，带重试与兜底直接写。

    旧实现直接 ``os.replace(tmp, path)``。在 Windows 上，当目标文件被其它
    进程短暂锁定时（例如：未退出的旧 server 进程仍持有句柄、或杀毒软件实时
    扫描），``os.replace`` 会抛 ``PermissionError``（WinError 5，拒绝访问），
    导致写入丢失。对于依赖持久化的功能（如 3D 每日推荐轮换历史
    ``save_recent_3d_recommendations`` / ``save_recent_zu6_four``）而言，写入
    丢失意味着去重历史无法落盘，表现为「每日推荐重复」与「旧进程 PID 异常」。

    策略：
      1. 序列化到 ``.tmp`` 临时文件后，用短重试/退避尝试 ``os.replace``
         （覆盖杀毒软件/锁的瞬时窗口）。
      2. 首次遇到权限错误时，先尝试删除被锁的目标文件再重试
         （删除有时比改名更被允许）。
      3. 作为最后兜底，直接写目标文件，确保数据绝不静默丢失。
    """
    global _FALLBACK_CACHE_SIG
    tmp = path + '.tmp'
    text = json.dumps(data_obj, ensure_ascii=False)
    # 先把内容写到 tmp（重试循环之外），保证即使改名失败也不会在目标文件上
    # 留下半截内容。
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    last_err = None
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            _FALLBACK_CACHE_SIG = None
            return
        except PermissionError as e:
            last_err = e
            if attempt == 0:
                # 目标可能被读者/旧进程锁住：重命名前先尝试删除（删除常比改名更被允许）。
                try:
                    os.remove(path)
                    continue
                except OSError:
                    pass
            time.sleep(0.05 * (attempt + 1))
        except OSError as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
    # 最后兜底：直接写目标文件，避免数据丢失。
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        _FALLBACK_CACHE_SIG = None
    except Exception:
        log.error("kv_store fallback 直接写入失败: %s", last_err)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _fallback_save(key, value, cache_date=None):
    """写入单条到 fallback 文件"""
    global _FALLBACK_CACHE, _FALLBACK_CACHE_SIG
    with _FALLBACK_WRITE_LOCK:
        data = dict(_fallback_load_all())
        data[key] = {'json_value': json.dumps(value, ensure_ascii=False), 'cache_date': cache_date}
        _atomic_replace(_fallback_path(), data)
        _FALLBACK_CACHE = data
        _FALLBACK_CACHE_SIG = None  # 失效缓存，下次 load 重新读取


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
    value, _backend = load_with_backend(key, default)
    return value


def load_with_backend(key, default=None):
    """读取普通项并返回 ``(value, backend)``，便于生产审计数据确认来源。"""
    try:
        row = db.query_one("SELECT json_value FROM kv_store WHERE k=%s", (key,))
        if row is None or row['json_value'] is None:
            return default, 'mysql'
        return json.loads(row['json_value']), 'mysql'
    except Exception:
        return _fallback_load(key, default), 'fallback'


def save(key, obj, require_mysql=False):
    """写入普通项（UPSERT）；关键审计数据可禁止静默本地降级。"""
    try:
        db.execute(_UPSERT, (key, json.dumps(obj, ensure_ascii=False), None, _now()))
        return 'mysql'
    except Exception:
        if require_mysql:
            raise
        _fallback_save(key, obj)
        return 'fallback'


def exists(key):
    try:
        return db.query_one("SELECT 1 FROM kv_store WHERE k=%s", (key,)) is not None
    except Exception:
        return key in _fallback_load_all()


def delete(key):
    try:
        db.execute("DELETE FROM kv_store WHERE k=%s", (key,))
    except Exception:
        with _FALLBACK_WRITE_LOCK:
            data = _fallback_load_all()
            data.pop(key, None)
            _atomic_replace(_fallback_path(), data)
            _FALLBACK_CACHE = data
            _FALLBACK_CACHE_SIG = None


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


def load_cache_stale(key):
    """读取缓存项，忽略「今天」失效语义；返回 (data, cache_date)。

    用于上游抓取失败时的兜底：宁可用上一次缓存的真实历史，也不要硬失败。
    无任何缓存时返回 (None, None)。
    """
    try:
        row = db.query_one(
            "SELECT json_value, cache_date FROM kv_store WHERE k=%s", (key,)
        )
        if row is None or row['json_value'] is None:
            return None, None
        return json.loads(row['json_value']), row['cache_date']
    except Exception:
        entry = _fallback_load_all().get(key)
        if not entry or entry.get('json_value') is None:
            return None, None
        return json.loads(entry['json_value']), entry.get('cache_date')
