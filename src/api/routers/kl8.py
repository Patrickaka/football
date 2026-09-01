# -*- coding: utf-8 -*-
"""快乐 8 接口。

**参数原样透传**，不加 FastAPI 的类型标注：这一族的参数全是字符串语义，
服务层自己转类型、自己定默认值（`window_size_str`、`repeat_avoid_score_str`
这些名字就是证据）。标上类型会让非法值从旧入口的「`200` + error 串」变成
422——对 kl8 的十三个参数逐个引入这种差异不值得。

同步重活一律经 `run_blocking` 派发到线程池。
"""

from fastapi import APIRouter, Depends

from src.api.deps import query_params, run_blocking
from src.api.services import kl8 as service

router = APIRouter(tags=['kl8'])


@router.get('/api/kl8')
async def kl8():
    return await run_blocking(service.kl8_payload)


@router.api_route('/api/kl8-refresh', methods=['GET', 'POST'])
async def refresh():
    return await run_blocking(service.kl8_refresh_payload)


@router.post('/api/kl8-refresh/start')
async def refresh_start():
    """启动后台重算，避免网关等待完整预测而返回 504。"""
    # 控制面只登记一个 daemon 任务，不能排在共享阻塞线程池后面；线程池若
    # 正被其他慢分析占满，连“开始任务”本身都可能等到网关超时。
    return service.kl8_refresh_start_payload()


@router.get('/api/kl8-refresh/status')
async def refresh_status(params: dict = Depends(query_params)):
    return service.kl8_refresh_status_payload(params)


@router.get('/api/kl8/fetch')
async def fetch():
    return await run_blocking(service.kl8_fetch_payload)


@router.get('/api/kl8/exclude-recalculate')
async def exclude_recalculate(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_exclude_recalculate_payload, params)


@router.get('/api/kl8/snapshots')
async def snapshots():
    return await run_blocking(service.kl8_snapshots_payload)


@router.get('/api/kl8/records')
async def records(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_records_payload, params)


@router.get('/api/kl8/settle')
async def settle(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_settle_payload, params)


@router.get('/api/kl8/backtest')
async def backtest(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_backtest_payload, params)


@router.get('/api/kl8/parameter-search')
async def parameter_search(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_parameter_search_payload, params)


@router.get('/api/kl8/parameter-search/start')
async def parameter_search_start(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_parameter_search_start_payload, params)


@router.get('/api/kl8/parameter-search/status')
async def parameter_search_status(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_parameter_search_status_payload, params)


@router.get('/api/kl8/integrity')
async def integrity():
    return await run_blocking(service.kl8_integrity_payload)


@router.get('/api/kl8/conflicts')
async def conflicts():
    return await run_blocking(service.kl8_conflicts_payload)


@router.get('/api/kl8/activate')
async def activate(params: dict = Depends(query_params)):
    return await run_blocking(service.kl8_activate_payload, params)
