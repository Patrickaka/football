# -*- coding: utf-8 -*-
"""彩票接口（3D / 双色球 / 排列三 / 大乐透）。

路径**逐条照抄旧入口**，包括 `/api/3d` 与 `/api/lottery` 这种不在同一
前缀下的——网页里的地址是写死的，改一个字都是切换当天才发现的故障。

同步重活一律经 `run_blocking` 派发到线程池。
"""

from fastapi import APIRouter

from src.api.deps import run_blocking
from src.api.services import lottery as service

router = APIRouter(tags=['lottery'])


@router.get('/api/3d')
async def lottery_3d():
    return await run_blocking(service.lottery_3d_payload)


@router.get('/api/3d-ml')
async def lottery_3d_ml():
    return await run_blocking(service.lottery_3d_ml_payload)


@router.api_route('/api/3d-refresh', methods=['GET', 'POST'])
async def lottery_3d_refresh(backtest: str = '0'):
    """**`backtest` 收成字符串**：服务层认的是 `1/true/yes/on` 四个字面量。
    换成 FastAPI 的 `bool` 会连 `y`、`t` 一起认，那是悄悄放宽。
    """
    return await run_blocking(service.lottery_3d_refresh_payload,
                              {'backtest': [backtest]})


@router.get('/api/ssq')
async def ssq():
    return await run_blocking(service.ssq_payload)


@router.get('/api/ssq-refresh')
async def ssq_refresh():
    return await run_blocking(service.ssq_refresh_payload)


@router.get('/api/lottery')
async def lottery():
    return await run_blocking(service.lottery_payload)


@router.api_route('/api/lottery-refresh', methods=['GET', 'POST'])
async def lottery_refresh():
    return await run_blocking(service.lottery_refresh_payload)


@router.get('/api/lottery/task-status')
async def task_status():
    return await run_blocking(service.lottery_task_status_payload)


@router.get('/api/lottery/recommend')
async def recommend():
    return await run_blocking(service.lottery_recommend_payload, {})


@router.get('/api/lottery/rank')
async def rank(top_n: int = 10):
    return await run_blocking(service.lottery_rank_payload, {'top_n': [top_n]})


@router.get('/api/lottery/ensemble')
async def ensemble():
    return await run_blocking(service.lottery_ensemble_payload)


@router.get('/api/lottery/cycles')
async def cycles():
    return await run_blocking(service.lottery_cycles_payload)


@router.get('/api/lottery/contribution')
async def contribution():
    return await run_blocking(service.lottery_contribution_payload)


@router.get('/api/lottery/backtest')
async def backtest(method: str = 'balanced', periods: int = 30):
    return await run_blocking(service.lottery_backtest_payload,
                              {'method': [method], 'periods': [periods]})


@router.get('/api/lottery/fetch')
async def fetch():
    return await run_blocking(service.lottery_fetch_payload)


@router.get('/api/lottery/ml')
async def lottery_ml():
    return await run_blocking(service.lottery_ml_payload)


@router.get('/api/lottery/ml-refresh')
async def lottery_ml_refresh():
    return await run_blocking(service.lottery_ml_refresh_payload)
