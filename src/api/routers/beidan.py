# -*- coding: utf-8 -*-
"""北单接口。

同步重活一律经 `run_blocking` 派发到线程池——直接写在 `async def` 里会在
事件循环线程上阻塞，冻住整个进程的所有并发请求。

**`force_refresh` 故意收成字符串**：服务层判的是 `.lower() == 'true'`，
只有字面量 `true` 才算真。改用 FastAPI 的 `bool` 会让 `1` / `yes` / `on`
也变成真——那是**悄悄放宽**，不是等价迁移。
"""

from typing import Optional

from fastapi import APIRouter

from src.api.deps import json_result
from src.api.services import beidan as service

router = APIRouter(prefix='/api/beidan', tags=['beidan'])


@router.get('')
async def recommendations(date: Optional[str] = None,
                          types: str = 'spf,rqspf,zjq',
                          source: str = 'okooo',
                          force_refresh: str = 'false'):
    """北单推荐预测。"""
    return await json_result(service.beidan_payload, {
        'date': [date], 'types': [types], 'source': [source],
        'force_refresh': [force_refresh],
    })


@router.get('/matches')
async def matches(date: Optional[str] = None, source: str = 'okooo'):
    """比赛列表。"""
    return await json_result(service.beidan_matches_payload,
                              {'date': [date], 'source': [source]})


@router.get('/value')
async def value_bets(date: Optional[str] = None, source: str = 'okooo',
                     threshold: float = 0.05):
    """价值投注推荐。"""
    return await json_result(service.beidan_value_payload,
                              {'date': [date], 'source': [source],
                               'threshold': [threshold]})


@router.get('/history')
async def history(limit: int = 200):
    """历史推荐汇总。"""
    return await json_result(service.beidan_history_payload, {'limit': [limit]})
