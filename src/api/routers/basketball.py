# -*- coding: utf-8 -*-
"""篮球接口。

**同步重活一律经 `run_blocking` 派发到线程池**：这里每个 payload 底下都是
同步的抓取与计算，直接写在 `async def` 里会在事件循环线程上阻塞，
冻住整个进程的所有并发请求——那比旧的 `ThreadingHTTPServer` 还糟。

参数保持 `parse_qs` 的 `{键: [值]}` 形状传给服务层，与旧入口喂的是同一批
输入，双跑差分才有意义。切换完成后再换成具名参数。
"""

from typing import Optional

from fastapi import APIRouter, Query

from src.api.deps import json_result
from src.api.services import basketball as service

router = APIRouter(prefix='/api/basketball', tags=['basketball'])


@router.get('')
async def recommendations(date: Optional[str] = None,
                          types: str = 'spf,rqspf,dx',
                          source: str = 'okooo'):
    """篮球推荐预测。"""
    return await json_result(service.basketball_payload,
                              {'date': [date], 'types': [types], 'source': [source]})


@router.get('/matches')
async def matches(date: Optional[str] = None):
    """比赛列表。"""
    return await json_result(service.basketball_matches_payload, {'date': [date]})


@router.get('/value')
async def value_bets(date: Optional[str] = None, threshold: float = 0.05):
    """价值投注推荐。"""
    return await json_result(service.basketball_value_payload,
                              {'date': [date], 'threshold': [threshold]})


@router.get('/track')
async def track(date: Optional[str] = None):
    """触发一次实时赔率轮询，累积盘路快照。"""
    return await json_result(service.basketball_track_payload, {'date': [date]})


@router.get('/movement')
async def movement(match_id: Optional[str] = Query(default=None)):
    """赔率走势汇总。"""
    return await json_result(service.basketball_movement_payload,
                              {'match_id': [match_id]})
