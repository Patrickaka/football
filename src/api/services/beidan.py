# -*- coding: utf-8 -*-
"""北单接口的业务装配。

每个函数收 `parse_qs` 形状的 `params`（`{键: [值]}`）、返回可直接序列化的
dict——**新旧两个入口共用同一份**（判据 11）。

`params` 保持这个别扭形状是为了让新旧双跑差分能喂同一批输入；
切换完成、旧入口删除后再换成具名参数。
"""

import logging

from src.api.runtime.beidan_cache import (
    beidan_cache_key, read_beidan_cache, refresh_beidan_async,
)
from src.api.runtime.jobs import finalize_beidan_recs
from src.api.runtime.lazy_modules import _load_beidan_helpers

log = logging.getLogger('api.services.beidan')


def beidan_payload(params):
    """获取北单推荐预测"""
    try:
        date = params.get('date', [None])[0]
        source = params.get('source', ['zgzcw'])[0]
        bet_types = params.get('types', ['spf,rqspf,zjq'])[0].split(',')
        
        force_refresh = params.get('force_refresh', ['false'])[0].lower() == 'true'

        log.info(f'北单推荐请求: date={date}, source={source}, types={bet_types}')

        cache_key = beidan_cache_key(date, source, bet_types)

        def _compute():
            # 惰性导入放在这里而不是请求线程：北单模块会连带拉起 numpy/sklearn，
            # 首次导入要好几秒，没必要让打开页面的人替它买单。
            generate_beidan_recommendations, _, _ = _load_beidan_helpers()
            computed = generate_beidan_recommendations(
                date=date, bet_types=bet_types, source=source)
            if 'error' not in computed:
                # 落盘与报告生成只在算出新数据时做一次，算完再写缓存，
                # 这样缓存里的 rec 已带报告 URL，读缓存不必重复这些副作用。
                finalize_beidan_recs(computed.get('recommendations'))
            return computed

        cached, fresh = read_beidan_cache(cache_key)
        if cached is None:
            # 从来没算过。整页重算线上要一两分钟，同步算一样会超过网关超时，
            # 所以照样丢后台，先回一个「计算中」，由前端轮询取结果。
            # 触发窗口：服务重启到预热跑起来之间，以及每天零点缓存键翻天之后。
            refresh_beidan_async(cache_key, _compute)
            log.info('北单无缓存，转后台首次计算: %s', cache_key)
            return {'result': {
                'computing': True,
                'refreshing': True,
                'date': date or '',
                'source': source,
                'recommendations': [],
            }}

        # 有缓存就立刻返回，过期与否只决定要不要在后台补一轮刷新。
        result = cached
        if force_refresh or not fresh:
            started = refresh_beidan_async(cache_key, _compute)
            # 无论本次是否真的起了新线程，都有一轮刷新在跑（未起说明已有同键在刷），
            # 都要告诉前端「正在刷新」，否则它不会回来取更新后的数据。
            result = dict(result)
            result['refreshing'] = True
            log.info('北单返回缓存并%s后台刷新: %s',
                           '触发' if started else '复用进行中的', cache_key)
        else:
            log.info('北单推荐命中缓存: %s', cache_key)

        if 'error' in result:
            return result

        return {'result': result}
    except Exception as e:
        log.error('北单推荐失败', exc_info=True)
        return {'error': f'北单推荐失败: {str(e)}'}


def beidan_matches_payload(params):
    """获取北单比赛列表"""
    try:
        date = params.get('date', [None])[0]
        source = params.get('source', ['zgzcw'])[0]
        
        if source == 'zgzcw':
            from src.beidan import fetch_zgzcw_schedule
            matches = fetch_zgzcw_schedule(date=date)
        else:
            from src.beidan import fetch_beidan_schedule
            matches = fetch_beidan_schedule(date=date, source=source)
        
        return {'matches': matches}
    except Exception as e:
        log.error('北单比赛列表获取失败', exc_info=True)
        return {'error': f'获取比赛列表失败: {str(e)}'}


def beidan_value_payload(params):
    """获取北单价值投注推荐"""
    try:
        date = params.get('date', [None])[0]
        source = params.get('source', ['zgzcw'])[0]
        threshold = float(params.get('threshold', [0.05])[0])
        
        _, find_value_bets, _ = _load_beidan_helpers()
        result = find_value_bets(date=date, threshold=threshold, source=source)
        
        if 'error' in result:
            return {'error': result['error']}
        
        return {'result': result}
    except Exception as e:
        log.error('北单价值投注失败', exc_info=True)
        return {'error': f'价值投注分析失败: {str(e)}'}


def beidan_history_payload(params):
    """获取北单预测记录摘要"""
    try:
        limit = int(params.get('limit', ['200'])[0])
        _, _, summarize_beidan_history = _load_beidan_helpers()
        return {'result': summarize_beidan_history(limit=limit)}
    except Exception as e:
        log.error('北单预测记录获取失败', exc_info=True)
        return {'error': f'北单预测记录获取失败: {str(e)}'}
