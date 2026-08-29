# -*- coding: utf-8 -*-
"""首屏聚合端点。"""

from fastapi import APIRouter

from src.api.deps import run_blocking
from src.api.services import bff as service

router = APIRouter(prefix='/api/bff', tags=['bff'])


@router.get('/football/home')
async def football_home():
    """足球首屏的全部数据，一次返回。

    仍走 `run_blocking`：读缓存本身很快，但拿比赛列表要联网。
    """
    return await run_blocking(service.football_home_payload)
