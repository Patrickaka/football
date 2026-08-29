# -*- coding: utf-8 -*-
"""彩票（3D / 双色球 / 排列三等）接口的业务装配。

**新旧两个入口共用同一份**（判据 11）。收 `parse_qs` 形状的 `params`
（`{键: [值]}`）或不收参数，返回可直接序列化的 dict。
"""

import logging
import time

from src.webapp.lazy_modules import (
    clear_ml_cache, get_lottery_analyzer, predict_with_ml,
)
from src.webapp.caching import (
    _CACHE, _compute_3d, _compute_3d_ml, _is_cache_valid,
)
from src.webapp.jobs import (
    LOTTERY_BACKGROUND_JOBS, LOTTERY_BACKGROUND_LOCK,
)
from src.webapp import caching as _caching_mod
from src.webapp import jobs as _jobs_mod
from src.webapp import lazy_modules as _lazy_mod

log = logging.getLogger('api.services.lottery')


def lottery_3d_payload():
    # 单飞 + stale-while-revalidate：命中直接返回；陈旧返回旧值并后台刷新；
    # 冷启动阻塞单飞（并发只算一次），彻底避免惊群导致的生产超时。
    data, err = _caching_mod._serve_cached('3d', _compute_3d)
    if err is not None or data is None:
        return {'error': '3D 预测失败'}
    return {'result': data}


def ssq_payload():
    """双色球：返回近期开奖、预测和历史记录（带单飞缓存）。"""
    try:
        data, err = _caching_mod._serve_cached('ssq', _lazy_mod.ssq_run_prediction)
        if err is not None or data is None:
            log.error('双色球预测失败: %s', err)
            return {'error': '双色球预测失败'}
        if isinstance(data, dict) and data.get('error'):
            return {'error': data['error']}
        return {'result': data}
    except Exception as exc:
        log.error('双色球预测失败: %s', exc, exc_info=True)
        return {'error': '双色球预测失败'}


def ssq_refresh_payload():
    """双色球：清除历史缓存并强制抓取最新开奖。"""
    try:
        _lazy_mod.ssq_clear_cache()
        result = _lazy_mod.ssq_run_prediction(force_refresh=True)
        if result.get('error'):
            return {'error': result['error']}
        _CACHE['ssq']['data'] = result
        _CACHE['ssq']['timestamp'] = time.time()
        return {'result': result}
    except Exception as exc:
        log.error('双色球刷新失败: %s', exc, exc_info=True)
        return {'error': '双色球刷新失败'}


def lottery_3d_refresh_payload(params=None):
    """Start a 3D refresh and return immediately with a pollable task ID."""
    params = params or {}
    enable_backtest = str((params.get('backtest') or ['0'])[0]).lower() in ('1', 'true', 'yes', 'on')
    job = _jobs_mod._start_3d_refresh_job(enable_backtest=enable_backtest)
    return {
        'processing': job.get('status') == 'processing',
        'task_id': job.get('task_id'),
        'message': job.get('message', '福彩3D后台刷新已启动'),
        'backtest_enabled': bool(enable_backtest),
    }


def lottery_3d_ml_payload():
    # 单飞 + stale-while-revalidate：ML 集成训练是唯一的重计算路径，
    # 交给统一缓存层，冷计算全程只发生一次且不阻塞已有用户。
    data, err = _caching_mod._serve_cached('3d_ml', _compute_3d_ml)
    if err is not None or data is None:
        return {'error': 'ML 3D 预测失败'}
    return {'result': data}


def lottery_payload():
    """获取大乐透统计分析（含缓存，调用模块级预测函数）"""
    try:
        now = time.time()
        cache = _CACHE['lottery']

        # 检查 server 级缓存（TTL + 跨天双重校验）
        if cache['data'] is not None and _is_cache_valid(cache, now):
            log.info('大乐透分析使用缓存（server 级）')
            return {'result': cache['data']}

        # 普通页面加载走快速路径：允许模块缓存，禁止请求内回测。
        # 回测和模型重训应由显式后台任务执行，避免反向代理504。
        log.info('大乐透分析快速计算')
        started = time.time()
        result = _lazy_mod.lottery_run_prediction(
            force_refresh=False,
            enable_backtest=False,
            enable_ml=False,
            enable_fusion=False,
            compute_weights=False,
        )
        log.info('大乐透快速计算完成，耗时 %.2f秒', time.time() - started)

        # 处理模块返回的错误
        if 'error' in result:
            return {'error': result['error']}

        # 更新 server 级缓存
        cache['data'] = result
        cache['timestamp'] = now

        return {'result': result}
    except Exception:
        log.error('大乐透分析失败', exc_info=True)
        return {'error': '大乐透分析失败'}


def lottery_refresh_payload(params=None):
    """在后台强制刷新大乐透，HTTP请求立即返回任务ID。"""
    job = _jobs_mod._start_lottery_refresh_job()
    return {
        'processing': job.get('status') == 'processing',
        'task_id': job.get('task_id'),
        'message': job.get('message', '后台刷新已启动'),
    }


def lottery_task_status_payload():
    """Return background status for大乐透 and福彩3D refresh jobs."""
    now = time.time()
    with LOTTERY_BACKGROUND_LOCK:
        # 只保留最近两小时任务，避免常驻服务无限增长。
        expired = [
            job_id for job_id, job in LOTTERY_BACKGROUND_JOBS.items()
            if now - float(job.get('created_at', now)) > 7200
        ]
        for job_id in expired:
            LOTTERY_BACKGROUND_JOBS.pop(job_id, None)
        return {job_id: dict(job) for job_id, job in LOTTERY_BACKGROUND_JOBS.items()}


def lottery_recommend_payload(params):
    """获取大乐透推荐号码 - 返回5组差异化策略组合"""
    try:
        # 推荐展示、主预测缓存和预测记录必须来自同一次计算快照。
        # 旧实现会在此处重新生成一遍组合，导致页面号码与记录不一致。
        prediction = _lazy_mod.lottery_run_prediction(
            force_refresh=False,
            enable_backtest=False,
            enable_ml=False,
            enable_fusion=False,
            compute_weights=False,
        )
        if prediction.get('error'):
            raise RuntimeError(prediction['error'])
        recommendation_map = prediction.get('recommendations') or {}
        recommendations = []
        for strategy, rec in recommendation_map.items():
            item = {
                'strategy': strategy,
                'method': rec.get('label') or rec.get('method') or strategy,
                'front': rec.get('front', []),
                'back': rec.get('back', []),
                'core_front': rec.get('core_front', []),
                'core_back': rec.get('core_back', []),
                'based_on_issue': rec.get('based_on_issue'),
            }
            # 透传精选一注的投票详情等额外字段
            for extra in ('picked_reason', 'selected_from', 'validation_evidence', 'front_vote_detail', 'back_vote_detail', 'cover_reason'):
                if extra in rec:
                    item[extra] = rec[extra]
            recommendations.append(item)

        return {
            'result': {
                'method': 'multi_strategy',
                'recommendations': recommendations,
                'count': len(recommendations),
                'portfolio_policy': prediction.get('portfolio_policy') or {},
                'back_coverage_profile': prediction.get('back_coverage_profile') or {},
                'version': prediction.get('version'),
            }
        }
    except Exception:
        log.error('大乐透推荐失败', exc_info=True)
        return {'error': '大乐透推荐失败'}


def lottery_rank_payload(params):
    """大乐透排名模型 - Top-N排序"""
    try:
        analyzer = get_lottery_analyzer()
        top_n = int(params.get('top_n', [10])[0])
        
        front_ranked, back_ranked = analyzer.rank_model(top_n=top_n)
        
        return {
            'result': {
                'top_n': top_n,
                'front_ranked': [{'number': n, 'score': s, 'features': f} for n, s, f in front_ranked],
                'back_ranked': [{'number': n, 'score': s, 'features': f} for n, s, f in back_ranked],
            }
        }
    except Exception:
        log.error('大乐透排名模型失败', exc_info=True)
        return {'error': '大乐透排名模型失败'}


def lottery_ensemble_payload():
    """大乐透多模型集成投票"""
    try:
        analyzer = get_lottery_analyzer()
        
        result = analyzer.multi_model_voting()
        
        return {'result': result}
    except Exception:
        log.error('大乐透集成预测失败', exc_info=True)
        return {'error': '大乐透集成预测失败'}


def lottery_cycles_payload():
    """大乐透周期与状态识别"""
    try:
        analyzer = get_lottery_analyzer()
        
        cycles = analyzer.identify_cycles()
        
        return {'result': cycles}
    except Exception:
        log.error('大乐透周期识别失败', exc_info=True)
        return {'error': '大乐透周期识别失败'}


def lottery_contribution_payload():
    """大乐透特征贡献度分析"""
    try:
        analyzer = get_lottery_analyzer()
        
        contributions = analyzer.feature_contribution()
        
        return {'result': contributions}
    except Exception:
        log.error('大乐透特征贡献度分析失败', exc_info=True)
        return {'error': '大乐透特征贡献度分析失败'}


def lottery_backtest_payload(params):
    """大乐透历史回测"""
    try:
        analyzer = get_lottery_analyzer()
        method = params.get('method', ['balanced'])[0]
        periods = int(params.get('periods', [30])[0])
        
        result = analyzer.backtest(method=method, test_periods=periods)
        
        return {'result': result}
    except Exception:
        log.error('大乐透回测失败', exc_info=True)
        return {'error': '大乐透回测失败'}


def lottery_fetch_payload():
    """后台增量抓取并重新分析，避免生产代理请求超时。"""
    job = _jobs_mod._start_lottery_refresh_job()
    return {
        'processing': job.get('status') == 'processing',
        'task_id': job.get('task_id'),
        'message': job.get('message', '后台抓取已启动'),
    }


def lottery_ml_payload():
    """大乐透 ML 预测结果"""
    try:
        now = time.time()
        cache = _CACHE['lottery_ml']

        if cache['data'] is not None and _is_cache_valid(cache, now):
            log.info('大乐透ML预测使用缓存')
            return {'result': cache['data']}

        log.info('大乐透ML预测重新计算')
        result = predict_with_ml()

        if 'error' in result:
            return {'error': result['error']}

        cache['data'] = result
        cache['timestamp'] = now
        return {'result': result}
    except Exception:
        log.error('大乐透ML预测失败', exc_info=True)
        return {'error': '大乐透ML预测失败'}


def lottery_ml_refresh_payload():
    """强制刷新大乐透ML预测（重新训练模型）"""
    try:
        clear_ml_cache()
        _CACHE['lottery_ml']['data'] = None
        _CACHE['lottery_ml']['timestamp'] = 0

        log.info('大乐透ML模型重新训练...')
        start = time.time()
        result = predict_with_ml(force_retrain=True)
        elapsed = time.time() - start

        _CACHE['lottery_ml']['data'] = result
        _CACHE['lottery_ml']['timestamp'] = time.time()

        return {
            'success': True,
            'elapsed': round(elapsed, 2),
            'models': {
                'front': list(result.get('front_model_scores', {}).keys()),
                'back': list(result.get('back_model_scores', {}).keys()),
            },
            'version': result.get('version', 'unknown'),
        }
    except Exception as e:
        log.error('大乐透ML重新训练失败: %s', str(e), exc_info=True)
        return {'success': False, 'error': str(e)}
