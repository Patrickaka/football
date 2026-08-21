#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""排列五预测入口：run_prediction 及其内存/磁盘缓存"""

import logging
import time

from .analyzer import get_pailie5_analyzer
from .caching import _is_today_cache, _load_prediction_cache, _save_prediction_cache
from .features import save_recent_recommend

logger = logging.getLogger(__name__)

# 预测结果缓存
_prediction_cache = None
_cache_time = 0


def clear_cache():
    """清除缓存"""
    global _prediction_cache, _cache_time
    _prediction_cache = None
    _cache_time = 0
    logger.info("排列五模块缓存已清除")


def run_prediction(force_refresh=False):
    """运行排列五预测，返回 JSON 可序列化 dict。"""
    global _prediction_cache, _cache_time

    # 内存缓存为空时，尝试从磁盘加载
    if _prediction_cache is None:
        disk_cache, disk_cache_time = _load_prediction_cache()
        if disk_cache is not None:
            _prediction_cache = disk_cache
            _cache_time = disk_cache_time

    if not force_refresh and _prediction_cache is not None:
        if _is_today_cache(_cache_time):
            elapsed = time.time() - _cache_time
            logger.info(f"使用今日缓存数据（缓存时间：{elapsed:.1f}秒前）")
            return _prediction_cache
        else:
            logger.info("缓存已过期，重新计算")

    try:
        analyzer = get_pailie5_analyzer()

        # 抓取最新数据
        analyzer.fetch_history_data(days=1, force_refresh=force_refresh)

        # 获取统计数据
        stats = analyzer.get_statistics()
        recent = analyzer.get_recent_results(10)

        # 集成预测
        ensemble = analyzer.ensemble_predict()

        # 保存推荐历史（用于去重）
        latest_issue = analyzer.history[0]['issue'] if analyzer.history else 'unknown'
        top30 = ensemble.get('top30', [])[:30]
        save_recent_recommend(latest_issue, top30)

        # 多种方法推荐（各取3组）
        recommendations = {}
        for method in ['balanced', 'hot', 'cold']:
            recs = []
            for _ in range(3):
                nums = analyzer.generate_recommendation(method)
                recs.append(nums)
            recommendations[method] = recs

        # 滚动回测（使用缓存，trials 减少到 20）
        backtest = analyzer.rolling_backtest(trials=20)

        result = {
            'statistics': stats,
            'recent_results': recent,
            'ensemble': ensemble,
            'recommendations': recommendations,
            'backtest': backtest,
        }

        _prediction_cache = result
        _cache_time = time.time()
        # 保存到磁盘缓存
        _save_prediction_cache(result, _cache_time)
        logger.info("排列五预测结果已缓存（内存+磁盘）")
        return result

    except Exception:
        logger.error('排列五预测失败', exc_info=True)
        return {'error': '排列五预测失败'}
