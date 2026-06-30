"""
数据缓存工具 - 每天只抓取一次数据

用于所有彩票模块的历史开奖数据缓存。
持久化后端为 MySQL kv_store（带 cache_date 失效语义），公共接口保持不变。
"""
import logging
from datetime import datetime

from . import db
from . import kv_store

log = logging.getLogger(__name__)


def get_today_str():
    """获取今天的日期字符串 YYYY-MM-DD"""
    return datetime.now().strftime('%Y-%m-%d')


def load_cached_data(module_name):
    """加载缓存数据（仅当为今天的数据时返回，否则 None）"""
    return kv_store.load_cache(module_name)


def save_cached_data(module_name, data):
    """保存缓存数据"""
    try:
        kv_store.save_cache(module_name, data)
        return True
    except Exception:
        return False


def cached_fetch(module_name, fetch_func, force_refresh=False):
    """
    带缓存的数据抓取

    参数：
        module_name: 模块名称（用于缓存键）
        fetch_func: 数据抓取函数
        force_refresh: 是否强制刷新

    返回：
        数据（可能是缓存的或新抓取的）
    """
    if not force_refresh:
        cached = load_cached_data(module_name)
        if cached is not None:
            return cached

    try:
        data = fetch_func()
    except Exception:
        # 上游抓取失败：兜底使用上一次缓存的真实历史，而不是硬失败。
        stale, stale_date = kv_store.load_cache_stale(module_name)
        if stale is not None:
            log.warning('%s 抓取失败，回退到 %s 的缓存数据', module_name, stale_date)
            return stale
        raise

    if data is not None:
        save_cached_data(module_name, data)
        return data

    # 抓取返回空：同样尝试回退到旧缓存。
    stale, stale_date = kv_store.load_cache_stale(module_name)
    if stale is not None:
        log.warning('%s 抓取结果为空，回退到 %s 的缓存数据', module_name, stale_date)
        return stale
    return data


def is_cache_valid(module_name):
    """检查缓存是否有效（今天的数据）"""
    return load_cached_data(module_name) is not None


def clear_cache(module_name=None):
    """
    清除缓存

    参数：
        module_name: 模块名称，如果为None则清除所有缓存项
    """
    if module_name:
        kv_store.delete(module_name)
    else:
        db.execute("DELETE FROM kv_store WHERE cache_date IS NOT NULL")
