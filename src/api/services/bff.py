# -*- coding: utf-8 -*-
"""首屏聚合（BFF）。

一次请求返回该屏需要的全部数据，把首屏的多次往返压成一次。

**只读缓存，绝不在请求线程做冷计算**——这是这一层的硬约束，不是优化。
一场比赛的完整分析要十几秒；首屏若代跑，用户就得在白屏上等，而且请求
会穿透到第三方源站。冷数据由后台预热任务算好（`_warm_football_caches`
每隔一段时间把当天未开赛的场次跑一遍），这里只负责把已经算好的取出来。

**缓存键与写入方共用一份**（`pipeline.analysis_cache_key`）：算错不会报错，
只会永远 miss——那样这个接口就变成一个永远返回"计算中"的空壳，
看起来在工作、实际什么也没做。
"""

import logging
from typing import Dict, List

log = logging.getLogger('api.services.bff')


def football_home_payload(params=None) -> Dict:
    """足球首屏：比赛列表 + 职业化状态 + 已算好的预测。

    返回里把预测分成两拨：
    - `ready`：缓存命中的，前端可以立刻渲染。
    - `pending`：还没算好的 match_id，前端自己再走 `/api/predict/batch`。

    **`pending` 不是错误**——它是"后台还没轮到这几场"的如实汇报。
    把它藏起来（比如返回空列表）会让前端以为数据齐了，那几场就永远不显示。
    """
    from src.api.services import football as football_service
    from src.football.config import CACHE_AVAILABLE, get_cache
    from src.football.pipeline import _is_prediction_cache_current, analysis_cache_key

    matches_payload = football_service.matches_payload()
    if matches_payload.get('error'):
        return {'error': matches_payload['error'],
                'source_status': matches_payload.get('source_status')}

    matches = matches_payload.get('matches') or []
    ready: List[Dict] = []
    pending: List[str] = []

    for match in matches:
        cached = None
        if CACHE_AVAILABLE:
            try:
                cached = get_cache('match_analysis', analysis_cache_key(match),
                                   match.get('time', ''))
            except Exception:
                log.warning('读取分析缓存失败 match_id=%s', match.get('match_id'),
                            exc_info=True)
                cached = None
        # 逻辑版本变了的缓存等于没有——照 `analyze_match` 的规矩来，
        # 否则首屏会拿旧口径的结果去渲染，而重算后的数字对不上。
        if cached is not None and _is_prediction_cache_current(cached):
            ready.append({'match_id': match.get('match_id'), 'result': cached})
        else:
            pending.append(match.get('match_id'))

    return {
        'matches': matches,
        'source_status': matches_payload.get('source_status'),
        'professional_status': _professional_status(),
        'predictions': {'ready': ready, 'pending': pending},
        'coverage': {'total': len(matches), 'ready': len(ready),
                     'pending': len(pending)},
    }


def _professional_status() -> Dict:
    """职业化验证状态。**取不到不该让整个首屏失败**——它只是个角标。"""
    from src.api.services import football as football_service

    try:
        return football_service.football_professional_status_payload()
    except Exception as exc:
        log.warning('职业化状态取用失败: %s', exc)
        return {'error': str(exc)}
