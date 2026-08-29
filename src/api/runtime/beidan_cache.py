# -*- coding: utf-8 -*-
"""北单结果级缓存：立即返回 + 后台刷新（stale-while-revalidate）

北单一次请求要算完整页，线上实测 160 秒，远超反代的网关超时——所以**任何用户
请求都不能等重算**，否则必然 504。这里的规则是：只要有缓存就立刻返回，过期与否
只决定要不要在后台补一轮刷新；只有「从来没算过」才不得不同步算一次。

TTL 由「最早未开赛场次」推导，但只用来决定何时触发后台刷新，绝不用来阻塞请求。
一份北单结果覆盖三百多场，若像足球那样让最早一场直接判整份缓存过期，
傍晚时 TTL 会缩到 2 分钟，等于每次打开都要重算。
"""

import os
import threading
import time
from datetime import datetime

from src.common.logger import setup_logger
from src.football.cache_manager import get_cache as _memory_get_cache
from src.football.cache_manager import set_cache as _memory_set_cache
from .shared_cache import get_cache as get_shared_cache

log = setup_logger('server')

BEIDAN_CACHE_TYPE = 'beidan_recommendations'
# 两层缓存里的键前缀。L2 是 Redis，与别的业务共用一个库。
BEIDAN_CACHE_PREFIX = 'beidan:pred'
# **一天只是内存上限，不是有效期。** 真正的有效期由 `_cached_at` 与
# `beidan_refresh_after` 决定（见 `read_beidan_cache`）——键里带了日期，
# 隔天自然换键，旧值随 TTL 自行淘汰。
#
# 这个数还决定 Redis 的物理过期（`RedisBackend` 按逻辑 TTL 的十倍存），
# **而跨重启保留靠的正是物理过期够长**：拿刷新档位（120~1800 秒）当 TTL
# 的话物理过期最短只有 20 分钟，一次部署加冷启动就把它耗光了。
BEIDAN_CACHE_TTL_SECONDS = 86400
# 没有临近场次时的默认刷新间隔
BEIDAN_REFRESH_DEFAULT = int(os.getenv('BEIDAN_REFRESH_SECONDS', '1800'))

_refresh_lock = threading.Lock()
_refreshing = set()


def beidan_cache_key(date, source, bet_types):
    return '%s_%s_%s' % (source, date or 'today', '-'.join(sorted(bet_types)))


def beidan_earliest_kickoff(result):
    """取最早的未开赛场次时间，用于推导「多久该刷新一次」。

    已开赛场次要跳过，否则整份结果会被永远钉在最短刷新档上。
    """
    now = datetime.now()
    earliest = None
    for rec in result.get('recommendations') or []:
        stamp = ('%s %s' % (rec.get('date') or '', rec.get('time') or '')).strip()
        try:
            kickoff = datetime.strptime(stamp, '%Y-%m-%d %H:%M')
        except ValueError:
            continue
        if kickoff > now and (earliest is None or kickoff < earliest):
            earliest = kickoff
    return earliest.strftime('%Y-%m-%d %H:%M') if earliest else None


def beidan_refresh_after(result):
    """距下次后台刷新的秒数：越接近开赛刷得越勤，但都不影响请求延迟。"""
    kickoff = beidan_earliest_kickoff(result)
    if kickoff is None:
        return BEIDAN_REFRESH_DEFAULT
    minutes = (datetime.strptime(kickoff, '%Y-%m-%d %H:%M') - datetime.now()).total_seconds() / 60
    if minutes < 15:
        return 120
    if minutes < 60:
        return 300
    if minutes < 360:
        return 900
    return BEIDAN_REFRESH_DEFAULT


def read_beidan_cache(cache_key):
    """返回 (结果, 是否新鲜)。结果为 None 表示从来没算过。

    这里刻意按天读、不传 match_time：分层 TTL 会让缓存管理器直接判过期返回 None，
    那样就拿不到「可以先顶上去的旧数据」了，而旧数据正是不 504 的关键。
    """
    payload = _read(cache_key)
    if payload is None:
        return None, False
    cached_at = payload.get('_cached_at') or 0
    return payload, (time.time() - cached_at) < beidan_refresh_after(payload)


# history_summary.latest 是 30 条完整历史预测记录，单独就占整份响应四成以上，
# 而前端只用 history_summary 里的几个计数器，从不读它。独立的
# /api/beidan/history 端点仍返回完整摘要，所以只在这份大响应里剔除。
_PRUNED_HISTORY_FIELDS = ('latest',)


def prune_beidan_payload(result):
    """剔除前端不消费的重字段，让缓存与响应都只留列表页真正要用的数据。"""
    history = result.get('history_summary')
    if isinstance(history, dict):
        for field in _PRUNED_HISTORY_FIELDS:
            history.pop(field, None)
    return result


def _shared_key(cache_key):
    return '%s:%s' % (BEIDAN_CACHE_PREFIX, cache_key)


def _read(cache_key):
    """从两层缓存里取一份（可能已过期的）结果。

    **降级路径存在但不该被当成常态**：`get_shared_cache()` 只在共享缓存
    连建都建不起来时才返回 `None`（Redis 不可用它自己会退化成纯内存，
    不会返回 None）。真走到这里说明有更严重的问题，此时退回迁移前那个
    进程内缓存——代价正是「不跨重启保留」，也就是这次改动要解决的问题本身。
    """
    cache = get_shared_cache()
    if cache is None:
        log.warning('共享缓存不可用，北单缓存退回进程内存（不跨重启保留）')
        return _memory_get_cache(BEIDAN_CACHE_TYPE, cache_key)
    entry = cache.peek(_shared_key(cache_key))
    return entry.value if entry is not None else None


def write_beidan_cache(cache_key, result):
    prune_beidan_payload(result)
    result['_cached_at'] = time.time()
    cache = get_shared_cache()
    if cache is None:
        _memory_set_cache(BEIDAN_CACHE_TYPE, cache_key, result)
        return
    cache.set(_shared_key(cache_key), result, ttl=BEIDAN_CACHE_TTL_SECONDS)


def try_begin_refresh(cache_key):
    """单飞闸门：同一个键同时只允许一轮后台刷新，避免并发刷新堆叠打爆源站。"""
    with _refresh_lock:
        if cache_key in _refreshing:
            return False
        _refreshing.add(cache_key)
        return True


def end_refresh(cache_key):
    with _refresh_lock:
        _refreshing.discard(cache_key)


def refresh_beidan_async(cache_key, compute):
    """后台重算并回写缓存；已有同键刷新在跑时直接跳过。"""
    if not try_begin_refresh(cache_key):
        return False

    def _run():
        started = time.perf_counter()
        try:
            result = compute()
            if 'error' in result:
                log.warning('北单后台刷新跳过: %s', result['error'])
                return
            write_beidan_cache(cache_key, result)
            log.info('北单后台刷新完成: %s, 耗时 %.1fs', cache_key, time.perf_counter() - started)
        except Exception:
            log.warning('北单后台刷新失败: %s', cache_key, exc_info=True)
        finally:
            end_refresh(cache_key)

    threading.Thread(target=_run, daemon=True, name='BeidanRefresh').start()
    return True
