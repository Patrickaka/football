# -*- coding: utf-8 -*-
"""【适配层】足球概率校准：Platt/isotonic/分层校准与联赛校准缓存"""

import sys
import datetime
import os
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('football')

from ..domain.sports.football import calibration as _cal
from .modeling import (
    _matrix_margins, _sigmoid,
)

# 纯计算全部转发给领域层
fit_platt_scaling = _cal.fit_platt_scaling
calibrate_with_platt = _cal.calibrate_with_platt
isotonic_regression_calibration = _cal.isotonic_regression_calibration
calibrate_probabilities = _cal.calibrate_probabilities
hierarchical_calibration = _cal.hierarchical_calibration
_get_draw_calibration_factor = _cal._get_draw_calibration_factor
_get_goal_calibration_factor = _cal._get_goal_calibration_factor
_get_score_calibration_factor = _cal._get_score_calibration_factor

LEAGUE_CALIBRATION_CACHE = {}


def normalize_team_name(name: str) -> str:
    """标准化球队名称，将所有别名映射到统一名称

    别名表从 `kv_store` 取（IO 留在这里），归一逻辑在领域层。
    """
    try:
        from ..common import kv_store
        alias_map = kv_store.load('team_alias')
    except Exception:
        alias_map = None
    return _cal.resolve_team_alias(name, alias_map)


def fetch_league_historical_data(league_name, limit=10):
    """
    获取指定联赛的历史比赛数据（包含模型预测和实际结果）。
    
    参数:
        league_name: 联赛名称
        limit: 获取最近的比赛数量
        
    返回:
        列表，每个元素包含 {'match_id', 'home', 'away', 'predicted_probs', 'actual_home', 'actual_away'}
    """
    log.info(f"获取联赛 {league_name} 的最近 {limit} 场历史数据")
    
    try:
        from .prediction_records import get_historical_data
        
        # 从真实预测记录中获取数据
        records = get_historical_data(league_name, limit)
        
        historical_data = []
        for record in records:
            # 将比分字符串转换为元组
            predicted_probs = {}
            for score_str, prob in record.get('predicted_scores', {}).items():
                try:
                    h, a = map(int, score_str.split('-'))
                    predicted_probs[(h, a)] = prob
                except ValueError:
                    continue
            
            # 解析实际比分
            actual_score = record.get('actual_score', '')
            actual_home, actual_away = 0, 0
            if actual_score:
                try:
                    actual_home, actual_away = map(int, actual_score.split('-'))
                except ValueError:
                    pass
            
            historical_data.append({
                'match_id': record['match_id'],
                'home': record['home'],
                'away': record['away'],
                'predicted_probs': predicted_probs,
                'actual_home': actual_home,
                'actual_away': actual_away,
            })
        
        # 如果真实数据不足，返回空列表（不使用随机数据）
        return historical_data
        
    except ImportError:
        log.warning("预测记录模块未导入，无法获取真实历史数据")
        return []
    except Exception as e:
        log.error(f"获取历史数据失败: {e}")
        return []


def train_league_platt_params(league_name, recent_matches=10):
    """
    针对特定联赛训练 Platt 缩放参数。
    
    参数:
        league_name: 联赛名称
        recent_matches: 使用最近多少场比赛进行训练
        
    返回:
        (A, B): 训练好的 Platt 参数
    """
    log.info(f"开始训练联赛 {league_name} 的 Platt 参数，使用最近 {recent_matches} 场比赛")
    
    # 获取历史数据
    historical_data = fetch_league_historical_data(league_name, limit=recent_matches)
    
    if len(historical_data) < 5:
        log.warning(f"联赛 {league_name} 历史数据不足（仅 {len(historical_data)} 场），使用默认参数")
        return (1.0, 0.0)
    
    # 准备训练数据：(模型概率, 实际结果) 对；摊平与拟合都在领域层
    A, B = fit_platt_scaling(_cal.platt_pairs_from_history(historical_data))
    
    # 保存到缓存
    LEAGUE_CALIBRATION_CACHE[league_name] = {
        'platt_params': (A, B),
        'trained_on': len(historical_data),
        'last_updated': datetime.datetime.now().isoformat()
    }
    
    log.info(f"联赛 {league_name} Platt 参数训练完成: A={A:.4f}, B={B:.4f}")
    return (A, B)


def get_league_calibration_data(league_name, force_retrain=False):
    """
    获取指定联赛的校准数据。
    
    参数:
        league_name: 联赛名称
        force_retrain: 是否强制重新训练
        
    返回:
        校准数据字典 {'platt_params': (A, B), ...}
    """
    if not force_retrain and league_name in LEAGUE_CALIBRATION_CACHE:
        log.debug(f"使用缓存的联赛 {league_name} 校准参数")
        return LEAGUE_CALIBRATION_CACHE[league_name]
    
    # 训练新参数
    A, B = train_league_platt_params(league_name)
    return {
        'platt_params': (A, B),
        'trained_on': LEAGUE_CALIBRATION_CACHE.get(league_name, {}).get('trained_on', 0),
        'last_updated': datetime.datetime.now().isoformat()
    }


def recalibrate_league(league_name, recent_matches=10):
    """
    手动触发重新校准指定联赛。
    
    参数:
        league_name: 联赛名称
        recent_matches: 使用最近多少场比赛进行重新校准
        
    返回:
        字典，包含校准结果信息
    """
    log.info(f"手动触发联赛 {league_name} 的重新校准，使用最近 {recent_matches} 场比赛")
    
    # 强制重新训练
    A, B = train_league_platt_params(league_name, recent_matches=recent_matches)
    
    # 获取校准数据
    calibration_data = get_league_calibration_data(league_name)
    
    return {
        'league': league_name,
        'platt_params': {'A': A, 'B': B},
        'trained_on': calibration_data.get('trained_on', 0),
        'last_updated': calibration_data.get('last_updated'),
        'status': 'success',
        'message': f"联赛 {league_name} 已使用最近 {recent_matches} 场比赛重新校准"
    }


def clear_calibration_cache():
    """
    清空所有联赛的校准缓存。
    """
    global LEAGUE_CALIBRATION_CACHE
    LEAGUE_CALIBRATION_CACHE = {}
    log.info("已清空所有联赛的校准缓存")
    return {'status': 'success', 'message': '校准缓存已清空'}


def list_calibrated_leagues():
    """
    列出所有已校准的联赛及其参数。
    
    返回:
        列表，每个元素包含联赛校准信息
    """
    result = []
    for league_name, data in LEAGUE_CALIBRATION_CACHE.items():
        result.append({
            'league': league_name,
            'platt_A': data['platt_params'][0],
            'platt_B': data['platt_params'][1],
            'trained_on': data.get('trained_on', 0),
            'last_updated': data.get('last_updated')
        })
    return result

















