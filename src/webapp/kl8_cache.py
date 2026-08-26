"""kl8 预测结果的缓存。

迁移前它挂在 `webapp/caching.py` 的进程内字典上，而 `_PERSIST_KEYS` 只含
3d 系列——**每次部署重启缓存清零**，用户在发版后的第一个请求要等 5.6 秒
（线上实测冷启动 3.5~5.6s，命中 0.01s）。改用 foundation/cache 之后，
L2 是 Redis，跨重启保留。

**失效条件编进 key**，而不是读回来再逐字段校验。旧实现的做法是把结果读出来、
比对 `based_on_issue` 与版本号，不符就丢掉重算——多一条路径就多一处会错的
地方，而且「读到了但要丢掉」这件事本身就说明 key 没有描述清楚它对应的是什么。
新开奖或版本变更自然产生新 key，旧值随 TTL 自行淘汰。
"""
import logging

log = logging.getLogger('webapp.kl8_cache')

# 一期的预测在下一期开奖前不会变，24 小时只是内存上限，不是有效期——
# 真正的有效期由 key 里的期号决定。
TTL_SECONDS = 86400
PREFIX = 'kl8:pred'


def cache_key(latest_issue, version):
    """key 同时区分期号与版本：少任何一个都会让旧结果活过它该活的时候。"""
    return f'{PREFIX}:{latest_issue}:{version}'


def predict(compute_fn, latest_issue, version, cache, ttl=TTL_SECONDS):
    """算或取一份预测。

    `latest_issue` 为空时**绕过缓存**：那说明历史还没加载出来，此时算出的
    结果不对应任何一期，存下来只会污染下一个真正的请求。
    """
    if cache is None or not latest_issue:
        if not latest_issue:
            log.warning('最新期号未知，本次预测不进缓存')
        return compute_fn()
    return cache.get(cache_key(latest_issue, version), compute_fn, ttl=ttl)
