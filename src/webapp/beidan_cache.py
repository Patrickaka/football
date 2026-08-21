# -*- coding: utf-8 -*-
"""北单结果级缓存：键构造、按开赛时间分层的 TTL 读写

北单原先没有任何结果级缓存，每次打开页面都要重新抓 okooo 全量盘口并重算，
实测 12~15 秒且第二次不会变快。这里复用足球那套按开赛时间分层 TTL 的缓存管理器。
单独成模块是为了让接口层与后台预热线程共用，且不产生循环导入。
"""

from datetime import datetime

from src.football.cache_manager import get_cache, set_cache

BEIDAN_CACHE_TYPE = 'beidan_recommendations'


def beidan_cache_key(date, source, bet_types):
    return '%s_%s_%s' % (source, date or 'today', '-'.join(sorted(bet_types)))


def beidan_earliest_kickoff(result):
    """取最早的未开赛场次时间，让整份结果按最紧迫的那场决定 TTL。

    一份北单结果覆盖几十场、开赛时间各异。取最早一场是保守选择：
    只要有一场临近开赛，整份缓存就跟着短 TTL 走，不会拿过期盘口顶包。
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


def read_beidan_cache(cache_key):
    """读缓存并按「最早开赛时间」二次校验 TTL。

    分两步是因为分层 TTL 需要开赛时间，而开赛时间只能从结果里读出来：
    先按天取一份，再用它自己的最早开赛时间让缓存管理器重判一次是否过期。
    """
    candidate = get_cache(BEIDAN_CACHE_TYPE, cache_key)
    if candidate is None:
        return None
    kickoff = beidan_earliest_kickoff(candidate)
    if kickoff is None:
        return candidate
    return get_cache(BEIDAN_CACHE_TYPE, cache_key, kickoff)


def write_beidan_cache(cache_key, result):
    set_cache(BEIDAN_CACHE_TYPE, cache_key, result, beidan_earliest_kickoff(result))
