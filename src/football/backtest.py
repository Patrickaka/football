#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一回测模块
============

功能：
1. 对历史记录逐场复盘
2. 用赛前盘口生成预测
3. 和真实比分比较
4. 输出命中率、Brier Score、LogLoss

核心指标：
- Top1 比分命中率
- Top3 比分命中率
- Top5 比分命中率
- 胜平负命中率
- 总进球 Top2 命中率
- 让球方向命中率
- Brier Score
- LogLoss
"""

import itertools
import math
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from datetime import datetime


from ..domain.sports.football.backtest import (  # noqa: F401
    BacktestRunner,
    _actual_goals,
    _expand_param_grid,
    _has_real_half_full_sample,
    _is_draw_score,
    _normalize_1x2_probs,
    _normalize_goal_distribution,
    _objective_score,
    _quality_filter,
    _records_with_actual_scores,
    _result_quality_is_usable,
    build_diagnostic_tuning_plan,
    compare_key_parameters,
    compare_parameters,
    get_diagnostic_tuning_suggestions,
    rolling_backtest_report,
    run_backtest,
    run_backtest_report,
)

from ..domain.sports.football import backtest as _bt


def apply_diagnostic_tuning(records, **kwargs):
    """读当前策略、写回调参配置——两者都是存储，注入给领域层（判据 16）"""
    from .prediction_policy import get_prediction_policy, save_tuning_params
    kwargs.setdefault('get_prediction_policy', get_prediction_policy)
    kwargs.setdefault('save_tuning_params', save_tuning_params)
    return _bt.apply_diagnostic_tuning(records, **kwargs)


def optimize_policy_buckets(records, predict_func, **kwargs):
    """逐联赛/逐分桶调参并落盘；落盘那步注入给领域层（判据 16）"""
    from .prediction_policy import save_tuning_params
    kwargs.setdefault('save_tuning_params', save_tuning_params)
    return _bt.optimize_policy_buckets(records, predict_func, **kwargs)


def optimize_prediction_parameters(records, predict_func, **kwargs):
    """`save_best=True` 时要写调参配置，注入给领域层（判据 16）"""
    from .prediction_policy import save_tuning_params
    kwargs.setdefault('save_tuning_params', save_tuning_params)
    return _bt.optimize_prediction_parameters(records, predict_func, **kwargs)


log = logging.getLogger('football')





















def rolling_backtest_from_history(league: str = None,
                                  limit: int = None,
                                  windows: Tuple[int, ...] = (30, 60, 90),
                                  predict_func: Optional[Callable] = None,
                                  **kwargs) -> Dict:
    """Load settled prediction history and run rolling-window diagnostics."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return rolling_backtest_report(
            records,
            windows=windows,
            predict_func=predict_func,
            **kwargs,
        )
    except Exception as e:
        return {'error': str(e)}





def apply_diagnostic_tuning_from_history(league: str = None,
                                         limit: int = None,
                                         **kwargs) -> Dict:
    """Load settled history and plan/apply guarded diagnostic tuning."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return apply_diagnostic_tuning(records, league=league, **kwargs)
    except Exception as e:
        return {'applied': False, 'error': str(e)}


def backtest_from_history(league: str = None, limit: int = None) -> Dict:
    """
    从预测历史中运行回测
    
    参数：
        league: 只回测指定联赛
        limit: 限制回测数量
    
    返回：
        回测汇总结果
    """
    try:
        from .result_sync import _global_history
        
        if league:
            records = [r for r in _global_history.records 
                      if r.get('settled') and r.get('league') == league]
        else:
            records = [r for r in _global_history.records if r.get('settled')]
        
        if limit:
            records = records[-limit:]
        
        runner = BacktestRunner()
        
        for record in records:
            actual_score = record.get('actual_score')
            if not actual_score:
                continue
            
            actual_result = record.get('actual_result', '')
            try:
                parts = actual_score.split('-')
                home_g = int(parts[0])
                away_g = int(parts[1])
                if home_g > away_g:
                    actual_result = 'H'
                elif home_g < away_g:
                    actual_result = 'A'
                else:
                    actual_result = 'D'
            except:
                continue
            
            actual = {'score': actual_score, 'result': actual_result}
            runner.add_result(record, {}, actual)
        
        return runner.get_summary()
        
    except ImportError:
        return {'error': 'result_sync 模块未导入'}


def backtest_report_from_history(league: str = None,
                                 limit: int = None,
                                 quality_filter: bool = True,
                                 min_quality_grade: str = 'medium') -> Dict:
    """Build detailed report from settled prediction history."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return run_backtest_report(
            records,
            verbose=False,
            quality_filter=quality_filter,
            min_quality_grade=min_quality_grade,
        )
    except Exception as e:
        return {'error': str(e)}






def optimize_parameters_from_history(predict_func: Callable,
                                     league: str = None,
                                     limit: int = None,
                                     **kwargs) -> Dict:
    """Load settled prediction history and run parameter optimization."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return optimize_prediction_parameters(records, predict_func, **kwargs)
    except Exception as e:
        return {'error': str(e)}



def optimize_policy_buckets_from_history(predict_func: Callable,
                                         league: str = None,
                                         limit: int = None,
                                         **kwargs) -> Dict:
    """Load settled history and optimize multiple policy groups."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return optimize_policy_buckets(records, predict_func, **kwargs)
    except Exception as e:
        return {'error': str(e)}





# ==================== 测试 ====================

def main():
    print("=== 回测模块测试 ===")
    
    # 测试数据
    test_records = [
        {
            'match_id': 'test_001',
            'league': '英超',
            'home': '曼城',
            'away': '曼联',
            'predicted_scores': {'1-1': 0.25, '2-1': 0.20, '1-0': 0.15, '0-0': 0.10},
            'predicted_1x2': {'home': 0.60, 'draw': 0.25, 'away': 0.15},
            'asian': -1.0,
            'total_line': 2.5,
            'actual_score': '2-1',
            'actual_result': 'H',
        },
        {
            'match_id': 'test_002',
            'league': '英超',
            'home': '阿森纳',
            'away': '切尔西',
            'predicted_scores': {'1-1': 0.30, '0-0': 0.20, '2-1': 0.15, '1-0': 0.12},
            'predicted_1x2': {'home': 0.40, 'draw': 0.35, 'away': 0.25},
            'asian': -0.5,
            'total_line': 2.5,
            'actual_score': '1-1',
            'actual_result': 'D',
        },
    ]
    
    # 运行回测
    summary = run_backtest(test_records, verbose=True)
    
    # 打印结果
    runner = BacktestRunner()
    for record in test_records:
        actual_score = record['actual_score']
        parts = actual_score.split('-')
        home_g, away_g = int(parts[0]), int(parts[1])
        actual_result = 'H' if home_g > away_g else ('A' if home_g < away_g else 'D')
        runner.add_result(record, {}, {'score': actual_score, 'result': actual_result})
    
    runner.print_summary()


if __name__ == '__main__':
    main()
