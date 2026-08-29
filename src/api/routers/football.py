# -*- coding: utf-8 -*-
"""足球接口。

**参数原样透传**（与 kl8 同策略）：服务层自己转类型、自己定默认值，
在路由层标类型会把非法值从旧入口的「`200` + error 串」变成 422。

两条路由不是简单转发：
- `/api/predict/batch` 收 JSON body，交给服务层前**不做形状校验**——
  服务层自己判 `isinstance(body, dict)` 并回一句中文错误，那是它的契约。
- `/reports/{path}` 是静态文件服务，旧入口直接往 `self.wfile` 写流；
  这里用 `FileResponse`，两边各自处理自己那套响应机制。
"""

import pathlib

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.deps import json_result, query_params
from src.api.services import football as service

router = APIRouter(tags=['football'])

REPORTS_DIR = pathlib.Path(__file__).resolve().parents[3] / 'reports'


@router.get('/api/matches')
async def matches():
    return await json_result(service.matches_payload)


@router.get('/api/predict')
async def predict(params: dict = Depends(query_params)):
    return await json_result(service.predict_payload, params)


@router.post('/api/predict/batch')
async def predict_batch(body: dict = Body(default=None)):
    """批量预测。**body 原样交给服务层**——形状校验是它的契约，
    在这里用 pydantic 模型挡一道会把中文错误换成 422 的 detail。
    """
    return await json_result(service.predict_batch_payload, body)


@router.get('/api/football/clear_cache')
async def clear_cache():
    return await json_result(service.football_clear_cache_payload)


@router.get('/api/football/prepare_ml_data')
async def prepare_ml_data():
    return await json_result(service.prepare_ml_history_data_payload)


@router.get('/api/football/diagnostics')
async def diagnostics(params: dict = Depends(query_params)):
    return await json_result(service.football_diagnostics_payload, params)


@router.get('/api/football/review')
async def review(params: dict = Depends(query_params)):
    return await json_result(service.football_review_payload, params)


@router.get('/api/football/professional-status')
async def professional_status():
    return await json_result(service.football_professional_status_payload)


@router.get('/api/calibrate')
async def calibrate(params: dict = Depends(query_params)):
    return await json_result(service.calibrate_payload, params)


@router.get('/api/calibrate/list')
async def calibrate_list():
    return await json_result(service.calibrate_list_payload)


@router.get('/api/calibrate/clear')
async def calibrate_clear():
    return await json_result(service.calibrate_clear_payload)


@router.get('/api/backtest')
async def backtest(params: dict = Depends(query_params)):
    return await json_result(service.backtest_payload, params)


@router.get('/api/backtest/threshold')
async def threshold():
    return await json_result(service.threshold_payload)


@router.get('/api/model/status')
async def model_status():
    return await json_result(service.model_status_payload)


@router.get('/api/model/backtest_stats')
async def backtest_stats(params: dict = Depends(query_params)):
    return await json_result(service.backtest_stats_payload, params)


@router.get('/api/predictions')
async def predictions():
    return await json_result(service.predictions_payload)


@router.get('/api/predictions/export')
async def predictions_export():
    return await json_result(service.predictions_export_payload)


@router.get('/api/sync/status')
async def sync_status():
    return await json_result(service.sync_status_payload)


@router.get('/api/sync/trigger')
async def sync_trigger():
    return await json_result(service.sync_trigger_payload)


@router.get('/api/sync/hide_failed')
async def sync_hide_failed():
    return await json_result(service.sync_hide_failed_payload)


@router.get('/reports/{name:path}')
async def report_file(name: str):
    """报告文件。

    **必须挡住路径穿越**：`name` 来自 URL，`../../etc/passwd` 这种写法
    在拼接后会跑出 reports 目录。`resolve()` 之后比对父目录是唯一可靠的
    判断——只检查字符串里有没有 `..` 挡不住编码变体与符号链接。
    """
    target = (REPORTS_DIR / name).resolve()
    if not target.is_relative_to(REPORTS_DIR.resolve()) or not target.is_file():
        raise HTTPException(status_code=404, detail='报告不存在')
    return FileResponse(target)
