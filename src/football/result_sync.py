#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
赛后比分同步模块
================

功能：
1. 保存预测记录（赛前）
2. 定时扫描未结算比赛
3. 自动抓取实际比分
4. 更新校准库、盘口库、ELO、命中率统计

同步状态：
- pending: 等待比赛结束
- ready: 可以同步
- synced: 已回填
- retry: 等待重试
- failed: 多次失败，不再重试
- ignored: 不参与回填

重试策略：
- 失败1次：2小时后再试
- 失败2次：6小时后再试
- 失败3次：24小时后再试
- 失败5次：标记为failed

数据结构：
{
    "match_id": "123456",
    "league": "英超",
    "home": "阿森纳",
    "away": "切尔西",
    "match_time": "2026-06-12 22:00:00",
    "asian": -0.5,
    "total_line": 2.5,
    "predicted_scores": {"1-1": 0.112, "2-1": 0.094},
    "predicted_1x2": {"home": 0.46, "draw": 0.27, "away": 0.27},
    "actual_score": null,
    "actual_result": null,
    "settled": false,
    "sync_status": "pending",      # 新增：同步状态
    "sync_attempts": 0,            # 新增：同步尝试次数
    "last_sync_at": null,          # 新增：上次同步时间
    "last_sync_error": null,       # 新增：上次同步错误
    "next_sync_at": null,          # 新增：下次同步时间
    "created_at": "..."
}
"""

import os
import re
import json
import time
import math
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from threading import Thread

from ..common import repositories


log = logging.getLogger('football')

PRODUCTION_MODEL_VERSION = 'football-v2026.07.20-accuracy-02'
# 3504-match five-league backtest: raising the official-pick gate from 0.50
# to 0.56 improved accuracy from 63.64% to 68.65%, with 34.05% coverage.
ACTIONABLE_MIN_PROBABILITY = 0.56
ACTIONABLE_MIN_MARGIN = 0.15


def _prediction_decision_snapshot(predicted_1x2: Dict[str, float]) -> Dict:
    """Freeze the pre-match rule used to measure selective recommendations."""
    probs = normalize_1x2_probs(predicted_1x2)
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_probability = ranked[0][1] if ranked else 0.0
    margin = top_probability - ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        'policy_version': 'selective-1x2-v2',
        'eligible': top_probability >= ACTIONABLE_MIN_PROBABILITY and margin >= ACTIONABLE_MIN_MARGIN,
        'prediction': ranked[0][0] if ranked else None,
        'top_probability': round(top_probability, 6),
        'margin': round(margin, 6),
        'min_probability': ACTIONABLE_MIN_PROBABILITY,
        'min_margin': ACTIONABLE_MIN_MARGIN,
    }


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HISTORY_FILE = os.path.join(DATA_DIR, 'prediction_history.json')


# ==================== 评估指标计算函数 ====================

def calculate_logloss(probs: Dict[str, float], actual_result: str) -> float:
    """
    计算 LogLoss（对数损失）
    
    参数：
        probs: 预测概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        actual_result: 实际结果 'H', 'D', 或 'A'
    
    返回：
        LogLoss 值，越小越好
    """
    if actual_result not in ['H', 'D', 'A']:
        return float('nan')

    probs = normalize_1x2_probs(probs)
    
    p = probs.get(actual_result, 0.0)
    p = max(min(p, 1 - 1e-15), 1e-15)  # 防止 log(0)
    return -math.log(p)


def normalize_1x2_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize 1X2 probability keys to H/D/A."""
    if not probs:
        return {}

    normalized = {
        'H': probs.get('H', probs.get('home', 0.0)),
        'D': probs.get('D', probs.get('draw', 0.0)),
        'A': probs.get('A', probs.get('away', 0.0)),
    }
    total = sum(normalized.values())
    if total > 0:
        normalized = {key: value / total for key, value in normalized.items()}
    return normalized


def calculate_brier_score(probs: Dict[str, float], actual_result: str) -> float:
    """
    计算 Brier Score（布瑞尔分数）
    
    参数：
        probs: 预测概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        actual_result: 实际结果 'H', 'D', 或 'A'
    
    返回：
        Brier Score 值，越小越好
    """
    if actual_result not in ['H', 'D', 'A']:
        return float('nan')

    probs = normalize_1x2_probs(probs)
    
    # 创建真实标签向量
    true_label = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    true_label[actual_result] = 1.0
    
    # 计算 Brier Score
    score = 0.0
    for key in ['H', 'D', 'A']:
        score += (probs.get(key, 0.0) - true_label[key]) ** 2
    
    return score


def calculate_hit(probs: Dict[str, float], actual_result: str) -> bool:
    """
    判断是否命中（预测概率最高的结果是否等于实际结果）
    
    参数：
        probs: 预测概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        actual_result: 实际结果 'H', 'D', 或 'A'
    
    返回：
        True 如果命中，False 否则
    """
    if actual_result not in ['H', 'D', 'A']:
        return None

    probs = normalize_1x2_probs(probs)
    
    # 找到概率最高的结果
    max_prob = -1
    predicted = None
    for key in ['H', 'D', 'A']:
        p = probs.get(key, 0.0)
        if p > max_prob:
            max_prob = p
            predicted = key
    
    return predicted == actual_result


def _score_to_result(score: str) -> Optional[str]:
    try:
        home_goals, away_goals = map(int, str(score).split('-'))
    except Exception:
        return None
    if home_goals > away_goals:
        return 'H'
    if home_goals < away_goals:
        return 'A'
    return 'D'


def _assess_result_quality(record: Dict,
                           actual_score: str,
                           actual_result: str,
                           source: str = None,
                           actual_half_score: str = None) -> Dict:
    """Assess whether a fetched result is trustworthy enough for calibration."""
    reasons = []
    score = 1.0

    extracted_result = _score_to_result(actual_score)
    if extracted_result is None:
        reasons.append('invalid_score_format')
        score -= 0.60
    elif extracted_result != actual_result:
        reasons.append('result_mismatch')
        score -= 0.45

    try:
        home_goals, away_goals = map(int, str(actual_score).split('-'))
        if home_goals > 12 or away_goals > 12:
            reasons.append('implausible_score')
            score -= 0.50
    except Exception:
        pass

    if not _is_match_settle_due(record.get('match_time'), minutes=180):
        reasons.append('not_settle_due')
        score -= 0.70

    source = source or 'unknown'
    if source == 'live_fid':
        score += 0.05
    elif source == 'live_team':
        score -= 0.05
    elif source == 'shuju':
        score -= 0.12
    else:
        reasons.append('unknown_source')
        score -= 0.15

    if actual_score in {'0-0', '1-1'} and source not in {'live_fid', 'live_team'}:
        reasons.append('low_information_score_without_live_source')
        score -= 0.18

    if actual_half_score:
        if _score_to_result(actual_half_score) is None:
            reasons.append('invalid_half_score_format')
            score -= 0.12

    if not record.get('match_id'):
        reasons.append('missing_match_id')
        score -= 0.10

    score = max(0.0, min(1.0, score))
    if score >= 0.82:
        grade = 'high'
    elif score >= 0.60:
        grade = 'medium'
    elif score >= 0.35:
        grade = 'low'
    else:
        grade = 'reject'

    return {
        'score': round(score, 3),
        'grade': grade,
        'source': source,
        'reasons': reasons,
        'usable_for_calibration': grade in {'high', 'medium'},
    }


def _is_result_quality_usable(record: Dict, min_grade: str = 'medium') -> bool:
    rank = {'reject': 0, 'low': 1, 'medium': 2, 'high': 3}
    quality = record.get('result_quality') or {}
    return rank.get(quality.get('grade'), 0) >= rank.get(min_grade, 2)


def _calibration_sample_weight(record: Dict) -> float:
    if record.get('exclude_from_calibration'):
        return 0.0
    try:
        from .sample_quality import assess_record_quality

        quality = assess_record_quality(record)
        return max(0.0, min(1.0, float(quality.get('calibration_weight', 0.0))))
    except Exception:
        result_quality = record.get('result_quality') or {}
        if result_quality.get('grade') in {'reject', 'low'}:
            return 0.0
        source = result_quality.get('source')
        if source == 'live_fid':
            return 1.0
        if source == 'live_team':
            return 0.85
        if source == 'shuju':
            return 0.60
        return 0.70


def fuse_probabilities(base_probs: Dict[str, float], ml_probs: Dict[str, float], 
                      ml_weight: float = 0.05) -> Dict[str, float]:
    """
    融合基础模型和 ML 模型的概率
    
    参数：
        base_probs: 基础模型概率 {'H': 0.48, 'D': 0.27, 'A': 0.25}
        ml_probs: ML 模型概率 {'H': 0.45, 'D': 0.29, 'A': 0.26}
        ml_weight: ML 模型权重（默认0.05）
    
    返回：
        融合后的概率（已归一化）
    """
    fused = {}
    total = 0.0
    
    for key in ['H', 'D', 'A']:
        fused[key] = (1 - ml_weight) * base_probs.get(key, 0.0) + ml_weight * ml_probs.get(key, 0.0)
        total += fused[key]
    
    # 归一化
    if total > 0:
        for key in ['H', 'D', 'A']:
            fused[key] /= total
    
    return fused


def evaluate_ml_prediction(record: Dict) -> Dict:
    """
    评估 ML 模型预测结果
    
    参数：
        record: 预测记录
    
    返回：
        评估结果字典
    """
    evaluation = {}
    
    actual_result = record.get('actual_result')
    if actual_result not in ['H', 'D', 'A']:
        return evaluation
    
    # 基础模型评估
    base_1x2 = record.get('base_1x2')
    if base_1x2:
        evaluation['base_1x2_logloss'] = calculate_logloss(base_1x2, actual_result)
        evaluation['base_1x2_brier'] = calculate_brier_score(base_1x2, actual_result)
        evaluation['base_1x2_hit'] = calculate_hit(base_1x2, actual_result)
    
    # ML 模型评估
    ml_1x2 = record.get('ml_1x2')
    if ml_1x2 and record.get('ml_available', False):
        evaluation['ml_1x2_logloss'] = calculate_logloss(ml_1x2, actual_result)
        evaluation['ml_1x2_brier'] = calculate_brier_score(ml_1x2, actual_result)
        evaluation['ml_1x2_hit'] = calculate_hit(ml_1x2, actual_result)
        
        # 模拟融合评估（5% ML权重）
        if base_1x2:
            fused_5pct = fuse_probabilities(base_1x2, ml_1x2, ml_weight=0.05)
            evaluation['fused_5pct_logloss'] = calculate_logloss(fused_5pct, actual_result)
            evaluation['fused_5pct_brier'] = calculate_brier_score(fused_5pct, actual_result)
            
            # 模拟融合评估（10% ML权重）
            fused_10pct = fuse_probabilities(base_1x2, ml_1x2, ml_weight=0.10)
            evaluation['fused_10pct_logloss'] = calculate_logloss(fused_10pct, actual_result)
            evaluation['fused_10pct_brier'] = calculate_brier_score(fused_10pct, actual_result)
    
    return evaluation


def infer_time_layer(match_time_str: str) -> str:
    """
    根据比赛时间推断当前预测应该记录到哪个时间层
    
    参数：
        match_time_str: 比赛时间字符串（格式："06-14 09:00"）
    
    返回：
        时间层标识: 'T-24h', 'T-6h', 'T-1h', 'T-15min', 'final'
    """
    try:
        now = datetime.now()
        match_time = _parse_match_datetime(match_time_str)
        if not match_time:
            return 'final'
        
        diff_minutes = (match_time - now).total_seconds() / 60
        
        if diff_minutes >= 24 * 60:
            return 'T-24h'
        if diff_minutes >= 6 * 60:
            return 'T-6h'
        if diff_minutes >= 60:
            return 'T-1h'
        if diff_minutes >= 15:
            return 'T-15min'
        return 'final'
    except Exception as e:
        log.debug(f"推断时间层失败: {e}")
        return 'final'


def time_layer_weight(time_layer: str) -> float:
    """Information weight for prediction snapshots at different pre-match layers."""
    weights = {
        'T-24h': 0.35,
        'T-6h': 0.55,
        'T-1h': 0.75,
        'T-15min': 0.90,
        'final': 1.00,
    }
    return weights.get(time_layer, 0.50)


def _prediction_content_sig(predicted_scores, predicted_1x2, asian, total_line,
                            odds_data, predicted_half_full, model_version):
    """预测的「有意义内容」签名，用于跳过无变化的重复写入。

    只覆盖影响预测结果的字段，刻意排除 updated_at 等时间戳——否则缓存命中时
    每次内容相同却因时间戳不同而反复写库（整表重写风暴的根源之一）。
    """
    try:
        payload = json.dumps(
            [predicted_scores, predicted_1x2, asian, total_line,
             odds_data, predicted_half_full, model_version],
            ensure_ascii=False, sort_keys=True, default=str,
        )
    except Exception:
        # 任意不可序列化内容都视作「已变化」，从而照常写入，绝不吞掉真实更新。
        return None
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


class PredictionHistory:
    """预测历史记录管理器"""
    
    def __init__(self):
        self.records: List[Dict] = []
        self._load()
    
    def _load(self):
        """从 MySQL 加载记录"""
        try:
            self.records = repositories.football_prediction_load()
            log.info(f"已加载 {len(self.records)} 条预测历史记录")
        except Exception as e:
            log.error(f"加载预测历史失败: {e}")
            self.records = []

    def _save(self):
        """保存记录到 MySQL（整表重写）。仅用于批量操作（audit/repair 等）。

        每请求级的单条变更请用 _save_record，避免整表 DELETE+INSERT 把
        binlog/磁盘写爆。
        """
        try:
            repositories.football_prediction_save(self.records)
        except Exception as e:
            log.error(f"保存预测历史失败: {e}")

    def _save_record(self, record):
        """仅 UPSERT 单条记录，把每请求写入量从 O(表行数) 降到 O(1)。"""
        try:
            repositories.football_prediction_upsert(record)
        except Exception as e:
            log.error(f"保存预测记录失败: {e}")
    
    def add_prediction(self, match_id: str, league: str, home: str, away: str,
                       match_time: str, predicted_scores: Dict[str, float],
                       predicted_1x2: Dict[str, float], asian: float = None,
                       total_line: float = None, odds_data: Dict = None,
                       predicted_half_full: Dict[str, float] = None,
                       # 影子预测相关字段
                       base_1x2: Dict[str, float] = None,
                       ml_1x2: Dict[str, float] = None,
                       ml_model_version: str = None,
                       ml_available: bool = False,
                       ml_feature_snapshot: Dict = None,
                       lottery_handicap: int = None,
                       predicted_rqspf: Dict[str, float] = None,
                       goal_count: Dict = None,
                       model_version: str = PRODUCTION_MODEL_VERSION):
        """
        添加预测记录
        
        参数：
            match_id: 比赛ID
            league: 联赛名称
            home: 主队名称
            away: 客队名称
            match_time: 比赛时间
            predicted_scores: 预测比分概率 {"1-1": 0.108, ...}
            predicted_1x2: 预测胜平负 {"home": 0.46, "draw": 0.27, "away": 0.27}
            asian: 亚盘让球
            total_line: 大小球盘口
            odds_data: 原始赔率数据（可选）
            predicted_half_full: 预测半全场概率 {"HH": 0.24, "DH": 0.19, ...}（可选）
            base_1x2: 基础模型胜平负预测 {"H": 0.48, "D": 0.27, "A": 0.25}
            ml_1x2: ML模型胜平负预测 {"H": 0.45, "D": 0.29, "A": 0.26}
            ml_model_version: ML模型版本
            ml_available: ML模型是否可用
            ml_feature_snapshot: ML特征快照
        """
        # 检查是否已存在
        for record in self.records:
            if record.get('match_id') == match_id:
                # 跳过无变化的重复写入：缓存命中时同一场比赛会被反复「预测」，
                # 但内容与时间层其实一字未变。此时直接返回，不写库、不更新时间戳，
                # 消灭每请求整表重写的写入风暴。
                layer = infer_time_layer(match_time)
                new_sig = _prediction_content_sig(
                    predicted_scores, predicted_1x2, asian, total_line,
                    odds_data, predicted_half_full, model_version,
                )
                existing_layers = record.get('time_layers') or {}
                if (
                    new_sig is not None
                    and record.get('_pred_sig') == new_sig
                    and existing_layers.get(layer) is not None
                    and not record.get('settled')
                ):
                    return

                # 更新现有记录
                update_data = {
                    'predicted_scores': predicted_scores,
                    'predicted_1x2': predicted_1x2,
                    'asian': asian,
                    'total_line': total_line,
                    'updated_at': datetime.now().isoformat(),
                    'odds_snapshot': odds_data,
                    'model_version': model_version,
                    'decision_snapshot': _prediction_decision_snapshot(predicted_1x2),
                    '_pred_sig': new_sig,
                }
                if predicted_half_full:
                    update_data['predicted_half_full'] = predicted_half_full
                # 添加影子预测字段
                if base_1x2:
                    update_data['base_1x2'] = base_1x2
                if ml_1x2:
                    update_data['ml_1x2'] = ml_1x2
                if ml_model_version:
                    update_data['ml_model_version'] = ml_model_version
                update_data['ml_available'] = ml_available
                if ml_feature_snapshot:
                    update_data['ml_feature_snapshot'] = ml_feature_snapshot
                update_data['lottery_handicap'] = lottery_handicap
                update_data['predicted_rqspf'] = predicted_rqspf
                if goal_count:
                    update_data['goal_count'] = goal_count
                record.update(update_data)

                # 更新对应时间层的预测
                if 'time_layers' not in record:
                    record['time_layers'] = {}
                record['time_layers']['final'] = predicted_scores  # 始终更新最终预测
                # 只在该层为None时才更新（保留更早时间点的预测）
                if record['time_layers'].get(layer) is None:
                    record['time_layers'][layer] = predicted_scores

                # 更新赔率分层记录
                if 'odds_layers' not in record:
                    record['odds_layers'] = {}
                record['odds_layers'][layer] = odds_data
                record['odds_layers']['final'] = odds_data

                self._save_record(record)
                return
        
        # 新增记录
        # 时间分层预测记录
        time_layers = {
            'T-24h': None,  # 赛前24小时预测
            'T-6h': None,   # 赛前6小时预测
            'T-1h': None,   # 赛前1小时预测
            'T-15min': None, # 赛前15分钟预测
            'final': predicted_scores,  # 最终预测
        }
        
        # 当前时间层也记录预测（与下方 odds_layers 对齐），
        # 否则按值去重时同一层会被重复预测一次
        layer = infer_time_layer(match_time)
        if layer in time_layers:
            time_layers[layer] = predicted_scores

        # 赔率分层记录
        odds_layers = {
            'T-24h': None,
            'T-6h': None,
            'T-1h': None,
            'T-15min': None,
            'final': odds_data,
        }
        odds_layers[layer] = odds_data
        
        record_data = {
            'match_id': match_id,
            'league': league,
            'home': home,
            'away': away,
            'match_time': match_time,
            'asian': asian,
            'total_line': total_line,
            'predicted_scores': predicted_scores,
            'predicted_1x2': predicted_1x2,
            'model_version': model_version,
            'decision_snapshot': _prediction_decision_snapshot(predicted_1x2),
            'lottery_handicap': lottery_handicap,
            'predicted_rqspf': predicted_rqspf,
            'goal_count': goal_count,
            'predicted_half_full': predicted_half_full,  # 新增：半全场预测
            'time_layers': time_layers,  # 新增：时间分层预测记录
            'odds_layers': odds_layers,  # 新增：赔率分层记录
            # 影子预测字段
            'base_1x2': base_1x2,
            'ml_1x2': ml_1x2,
            'ml_model_version': ml_model_version,
            'ml_available': ml_available,
            'ml_feature_snapshot': ml_feature_snapshot,
            # 赛后评估字段（结算时填充）
            'evaluation': None,
            'actual_score': None,
            'actual_result': None,
            'actual_half_score': None,   # 新增：实际半场比分
            'actual_half_result': None,  # 新增：实际半场结果
            'actual_half_full': None,    # 新增：实际半全场结果
            'settled': False,
            # 同步状态字段
            'sync_status': 'pending',
            'sync_attempts': 0,
            'last_sync_at': None,
            'last_sync_error': None,
            'next_sync_at': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'odds_snapshot': odds_data,
            '_pred_sig': _prediction_content_sig(
                predicted_scores, predicted_1x2, asian, total_line,
                odds_data, predicted_half_full, model_version,
            ),
        }
        self.records.append(record_data)
        self._save_record(record_data)
        log.info(f"添加预测记录: {home} vs {away} (match_id={match_id})")
    
    def get_record(self, match_id: str) -> Optional[Dict]:
        """按比赛ID获取单条记录，无则返回 None"""
        return next((r for r in self.records if r.get('match_id') == match_id), None)

    def get_unsettled(self) -> List[Dict]:
        """获取未结算的记录"""
        return [r for r in self.records if not r.get('settled', False)]
    
    def get_settled(self, limit: int = None) -> List[Dict]:
        """获取已结算的记录"""
        records = [r for r in self.records if r.get('settled', False)]
        if limit:
            records = records[-limit:]
        return records
    
    def get_ready_to_settle(self, minutes: int = 180) -> List[Dict]:
        """
        获取可以结算的记录（比赛时间已过）
        
        参数：
            minutes: 比赛开始后等待分钟数（默认180分钟=3小时）
        """
        ready = []
        now = datetime.now()
        
        for record in self.records:
            if record.get('settled', False):
                continue
            
            match_time_str = record.get('match_time')
            if not match_time_str:
                continue
            
            try:
                if _is_match_settle_due(match_time_str, minutes=minutes, now=now):
                    ready.append(record)
                    
            except Exception:
                continue
        
        return ready
    
    def update_time_layer(self, match_id: str, time_layer: str, predicted_scores: Dict[str, float]):
        """
        更新时间分层预测记录
        
        参数：
            match_id: 比赛ID
            time_layer: 时间层标识 ('T-24h', 'T-6h', 'T-1h', 'T-15min', 'final')
            predicted_scores: 该时间点的预测比分概率
        """
        for record in self.records:
            if record.get('match_id') == match_id:
                if 'time_layers' not in record:
                    record['time_layers'] = {}
                record['time_layers'][time_layer] = predicted_scores
                record['updated_at'] = datetime.now().isoformat()
                self._save_record(record)
                log.info(f"更新时间分层预测: {match_id} -> {time_layer}")
                return True
        return False
    
    def _calculate_hit_flags(self, record: Dict) -> Dict:
        """计算命中标志和失败原因"""
        actual_score = record.get('actual_score')
        actual_result = record.get('actual_result')
        predicted_scores = record.get('predicted_scores', {})
        predicted_1x2 = normalize_1x2_probs(record.get('predicted_1x2', {}))
        
        # 半全场相关
        actual_half_full = record.get('actual_half_full')
        predicted_half_full = record.get('predicted_half_full', {})
        
        sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
        top1 = sorted_scores[0][0] if sorted_scores else None
        top3 = [s for s, _ in sorted_scores[:3]]
        top5 = [s for s, _ in sorted_scores[:5]]
        top10 = [s for s, _ in sorted_scores[:10]]
        top20 = [s for s, _ in sorted_scores[:20]]
        top30 = [s for s, _ in sorted_scores[:30]]
        
        # 计算真实比分的排名和概率
        actual_score_rank = None
        actual_score_prob = predicted_scores.get(actual_score, 0)
        
        for i, (score, prob) in enumerate(sorted_scores):
            if score == actual_score:
                actual_score_rank = i + 1  # 排名从1开始
                break
        
        pred_result = max(predicted_1x2.items(), key=lambda x: x[1])[0] if predicted_1x2 else None
        
        hit_top1 = actual_score == top1
        hit_top3 = actual_score in top3
        hit_top5 = actual_score in top5
        hit_top10 = actual_score in top10
        hit_top20 = actual_score in top20
        hit_top30 = actual_score in top30
        hit_1x2 = pred_result == actual_result

        actual_rqspf = None
        hit_rqspf = None
        predicted_rqspf = record.get('predicted_rqspf') or {}
        lottery_handicap = record.get('lottery_handicap')
        if predicted_rqspf and lottery_handicap is not None and actual_score:
            try:
                actual_home, actual_away = (int(value) for value in actual_score.split('-'))
                adjusted_margin = actual_home + int(lottery_handicap) - actual_away
                actual_rqspf = '让胜' if adjusted_margin > 0 else '让负' if adjusted_margin < 0 else '让平'
                predicted_rqspf_result = max(predicted_rqspf, key=predicted_rqspf.get)
                hit_rqspf = predicted_rqspf_result == actual_rqspf
            except (TypeError, ValueError):
                pass
        
        # 半全场命中计算
        hit_half_full_top1 = False
        hit_half_full_top3 = False
        if predicted_half_full and actual_half_full:
            sorted_htf = sorted(predicted_half_full.items(), key=lambda x: -x[1])
            htf_top1 = sorted_htf[0][0] if sorted_htf else None
            htf_top3 = [s[0] for s in sorted_htf[:3]]
            hit_half_full_top1 = actual_half_full == htf_top1
            hit_half_full_top3 = actual_half_full in htf_top3
        
        # 半场胜平负命中
        actual_half_result = record.get('actual_half_result')
        hit_half_1x2 = False
        if actual_half_result and predicted_half_full:
            # 从半全场预测中推断半场结果概率
            half_probs = {}
            for key, prob in predicted_half_full.items():
                half_res = key[0]  # 取第一个字符作为半场结果
                half_probs[half_res] = half_probs.get(half_res, 0) + prob
            if half_probs:
                pred_half_result = max(half_probs.items(), key=lambda x: x[1])[0]
                hit_half_1x2 = pred_half_result == actual_half_result
        
        fail_reasons = []
        if not hit_top3:
            fail_reasons = self._analyze_fail_reasons(record, sorted_scores, predicted_1x2)
        
        return {
            'hit_top1': hit_top1,
            'hit_top3': hit_top3,
            'hit_top5': hit_top5,
            'hit_top10': hit_top10,
            'hit_top20': hit_top20,
            'hit_top30': hit_top30,
            'hit_1x2': hit_1x2,
            'hit_rqspf': hit_rqspf,
            'actual_rqspf': actual_rqspf,
            # 半全场命中指标
            'hit_half_full_top1': hit_half_full_top1,
            'hit_half_full_top3': hit_half_full_top3,
            'hit_half_1x2': hit_half_1x2,
            'actual_score_rank': actual_score_rank,
            'actual_score_prob': actual_score_prob,
            'fail_reasons': fail_reasons,
        }
    
    def _analyze_fail_reasons(self, record: Dict, sorted_scores: List[Tuple[str, float]], 
                             predicted_1x2: Dict[str, float]) -> List[str]:
        """分析失败原因"""
        reasons = []
        actual_score = record.get('actual_score', '')
        actual_result = record.get('actual_result', '')
        
        if not actual_score or not actual_result:
            return reasons
        
        try:
            parts = actual_score.split('-')
            home_goals = int(parts[0])
            away_goals = int(parts[1])
            actual_goals = home_goals + away_goals
        except:
            return reasons
        
        pred_total = 0.0
        for score, prob in sorted_scores:
            try:
                h, a = map(int, score.split('-'))
                pred_total += (h + a) * prob
            except:
                pass
        
        pred_max = max(predicted_1x2.items(), key=lambda x: x[1])[0] if predicted_1x2 else None
        
        if pred_total < 2.5 and actual_goals >= 3:
            reasons.append('lambda_error_high')
        elif pred_total >= 2.5 and actual_goals <= 1:
            reasons.append('lambda_error_low')
        
        if pred_max == 'H' and actual_result == 'A':
            reasons.append('supremacy_error')
        elif pred_max == 'A' and actual_result == 'H':
            reasons.append('supremacy_error')
        
        draw_prob = predicted_1x2.get('D', 0)
        if actual_result == 'D' and draw_prob < 0.25:
            reasons.append('draw_underestimated')
        
        away_prob = predicted_1x2.get('A', 0)
        if actual_result == 'A' and away_prob < 0.2:
            reasons.append('away_underestimated')
        
        if actual_goals >= 4:
            has_high_score = False
            for score, prob in sorted_scores[:3]:
                try:
                    h, a = map(int, score.split('-'))
                    if h + a >= 4:
                        has_high_score = True
                        break
                except:
                    pass
            if not has_high_score:
                reasons.append('high_score_missed')
        
        market_weight = record.get('model_params', {}).get('market_weight', 0)
        if market_weight > 0.3 and actual_score not in [s for s, _ in sorted_scores[:5]]:
            reasons.append('market_prior_error')
        
        top3_list = [s for s, _ in sorted_scores[:3]]
        if sorted_scores and sorted_scores[0][1] > 0.4 and actual_score not in top3_list:
            reasons.append('bayes_overcorrect')
        
        steam_signals = record.get('steam_signals', [])
        if steam_signals:
            steam_bias = sum(s.get('bias', 0) for s in steam_signals)
            if steam_bias > 0.5 and actual_result == 'A':
                reasons.append('steam_misread')
            elif steam_bias < -0.5 and actual_result == 'H':
                reasons.append('steam_misread')
        
        return reasons
    
    def get_fail_reason_statistics(self, limit: int = 100) -> Dict[str, int]:
        """获取失败原因统计"""
        statistics = {}
        recent_records = self.get_settled(limit)
        
        for record in recent_records:
            fail_reasons = record.get('fail_reasons', [])
            for reason in fail_reasons:
                statistics[reason] = statistics.get(reason, 0) + 1
        
        return dict(sorted(statistics.items(), key=lambda x: -x[1]))
    
    def print_fail_reason_report(self, limit: int = 100) -> Dict[str, int]:
        """打印失败原因报告"""
        statistics = self.get_fail_reason_statistics(limit)
        
        print(f"\n{'='*60}")
        print(f"最近 {limit} 场失败原因统计")
        print(f"{'='*60}")
        
        if not statistics:
            print("  暂无失败记录")
            return statistics
        
        total_failures = sum(statistics.values())
        print(f"  总失败场次: {total_failures}")
        print(f"{'原因':<25} | {'场次':^8} | {'占比':^10}")
        print(f"{'-'*60}")
        
        reason_descriptions = {
            'lambda_error_high': '总进球高估',
            'lambda_error_low': '总进球低估',
            'supremacy_error': '强弱方向错',
            'draw_underestimated': '平局低估',
            'away_underestimated': '客队低估',
            'high_score_missed': '高比分漏掉',
            'market_prior_error': '盘口先验拉偏',
            'bayes_overcorrect': '贝叶斯校准过度',
            'steam_misread': '资金流误判',
        }
        
        for reason, count in statistics.items():
            desc = reason_descriptions.get(reason, reason)
            percentage = (count / total_failures) * 100
            print(f"  {desc:<25} | {count:^8} | {percentage:^10.1f}%")
        
        print(f"{'='*60}\n")
        
        return statistics
    
    def update_result(self, match_id: str, actual_score: str, actual_result: str,
                      actual_half_score: str = None, error: str = None,
                      source: str = None):
        """
        更新比赛结果
        
        参数：
            match_id: 比赛ID
            actual_score: 实际比分 "2-1"
            actual_result: 实际结果 "H"/"D"/"A"
            actual_half_score: 实际半场比分 "1-0"（可选）
            error: 同步错误信息（可选）
        """
        for record in self.records:
            if record.get('match_id') == match_id:
                if actual_score and actual_result:
                    if not _is_match_settle_due(record.get('match_time'), minutes=180):
                        record['sync_status'] = 'pending'
                        record['last_sync_error'] = '比赛尚未到结算时间，拒绝提前回填'
                        record['last_sync_at'] = datetime.now().isoformat()
                        log.warning(
                            f"拒绝提前回填: {record.get('home')} vs {record.get('away')} "
                            f"match_time={record.get('match_time')} score={actual_score}"
                        )
                        self._save_record(record)
                        return False

                    result_quality = _assess_result_quality(
                        record,
                        actual_score,
                        actual_result,
                        source=source,
                        actual_half_score=actual_half_score,
                    )
                    if result_quality['grade'] == 'reject':
                        record['sync_status'] = 'retry'
                        record['last_sync_error'] = f"赛果可信度过低，拒绝回填: {result_quality['reasons']}"
                        record['last_sync_at'] = datetime.now().isoformat()
                        record['next_sync_at'] = (datetime.now() + timedelta(hours=6)).isoformat()
                        self._save_record(record)
                        return False

                    # 成功结算
                    record['actual_score'] = actual_score
                    record['actual_result'] = actual_result
                    record['result_quality'] = result_quality
                    record['settled'] = True
                    settled_at = datetime.now().isoformat()
                    record['settled_at'] = settled_at
                    record['last_sync_at'] = settled_at
                    record['sync_status'] = 'synced'
                    
                    # 处理半场比分
                    if actual_half_score:
                        record['actual_half_score'] = actual_half_score
                        # 计算半场结果
                        try:
                            half_h, half_a = map(int, actual_half_score.split('-'))
                            half_res = 'H' if half_h > half_a else 'A' if half_h < half_a else 'D'
                            record['actual_half_result'] = half_res
                            # 计算半全场结果
                            record['actual_half_full'] = f"{half_res}{actual_result}"
                            # 标记数据质量
                            record['half_time_data_quality'] = 'real'
                        except:
                            record['half_time_data_quality'] = 'invalid'
                    else:
                        record['half_time_data_quality'] = 'missing'
                    
                    # 计算命中结果（包含半全场）
                    record.update(self._calculate_hit_flags(record))
                    
                    # 计算 ML 评估指标
                    record['evaluation'] = evaluate_ml_prediction(record)
                    
                    # 更新各模块
                    if _is_result_quality_usable(record):
                        self._update_calibrator(record)
                        self._update_market_db(record)
                        self._update_score_frequency_db(record)
                        self._update_elo_ratings(record)
                        self._update_half_time_stats(record)  # 新增：更新半场统计
                        self._update_goal_count_stats(record)  # 新增：更新总进球校准闭环

                        # 最后写入盘口变化库
                        self._update_market_change_db(record)
                    else:
                        log.warning(
                            f"赛果质量不足，仅保存结果不更新训练库: "
                            f"{record.get('home')} vs {record.get('away')} {record.get('result_quality')}"
                        )
                    
                    log.info(f"结算比赛: {record['home']} vs {record['away']} -> {actual_score} ({actual_result})")
                else:
                    # 同步失败
                    self._handle_sync_failure(record, error or '无法获取赛果')

                self._save_record(record)
                return True
        return False
    
    def _handle_sync_failure(self, record: Dict, error: str):
        """处理同步失败"""
        attempts = record.get('sync_attempts', 0) + 1
        record['sync_attempts'] = attempts
        record['last_sync_at'] = datetime.now().isoformat()
        record['last_sync_error'] = error
        
        # 计算下次重试时间
        retry_intervals = {
            1: 2,    # 2小时
            2: 6,    # 6小时
            3: 24,   # 24小时
            4: 48,   # 48小时
        }
        
        if attempts >= 5:
            record['sync_status'] = 'failed'
            record['next_sync_at'] = None
            log.warning(f"同步失败超过5次，标记为失败: {record.get('home')} vs {record.get('away')}")
        else:
            hours = retry_intervals.get(attempts, 24)
            record['sync_status'] = 'retry'
            record['next_sync_at'] = (datetime.now() + timedelta(hours=hours)).isoformat()
            log.debug(f"同步失败，等待 {hours} 小时后重试: {record.get('home')} vs {record.get('away')}")
    
    def get_ready_to_sync(self, minutes: int = 180) -> List[Dict]:
        """
        获取可以同步的记录（比赛结束且未结算）
        
        参数：
            minutes: 比赛开始后等待分钟数（默认180分钟=3小时）
        """
        ready = []
        now = datetime.now()
        
        for record in self.records:
            if record.get('settled', False):
                continue
            
            sync_status = record.get('sync_status', 'pending')
            if sync_status in ('synced', 'failed', 'ignored'):
                continue
            
            # 检查是否在重试等待中
            if sync_status == 'retry':
                next_sync = record.get('next_sync_at')
                if next_sync:
                    try:
                        next_time = datetime.fromisoformat(next_sync)
                        if now < next_time:
                            continue
                    except:
                        pass
            
            # 检查比赛是否已结束
            match_time_str = record.get('match_time')
            if not match_time_str:
                continue
            
            try:
                if _is_match_settle_due(match_time_str, minutes=minutes, now=now):
                    record['sync_status'] = 'ready'
                    ready.append(record)
                elif record.get('sync_status') == 'ready':
                    record['sync_status'] = 'pending'
                    
            except Exception:
                continue
        
        return ready
    
    def get_sync_status_summary(self) -> Dict:
        """获取同步状态汇总"""
        pending = 0
        ready = 0
        synced = 0
        retry = 0
        failed = 0
        ignored = 0
        
        last_sync = None
        last_settled = None
        
        for record in self.records:
            status = record.get('sync_status', 'pending')
            
            if status == 'pending':
                pending += 1
            elif status == 'ready':
                ready += 1
            elif status == 'synced':
                synced += 1
            elif status == 'retry':
                retry += 1
            elif status == 'failed':
                failed += 1
            elif status == 'ignored':
                ignored += 1
            
            last_sync_at = record.get('last_sync_at')
            if last_sync_at:
                try:
                    sync_time = datetime.fromisoformat(last_sync_at)
                    if last_sync is None or sync_time > last_sync:
                        last_sync = sync_time
                except:
                    pass

            if status == 'synced':
                settled_at = record.get('settled_at') or record.get('last_sync_at')
                if settled_at:
                    try:
                        settled_time = datetime.fromisoformat(settled_at)
                        if last_settled is None or settled_time > last_settled:
                            last_settled = settled_time
                    except:
                        pass
        
        return {
            'total': len(self.records),
            'settled': synced,
            'pending_sync': pending + ready,
            'retry': retry,
            'failed': failed,
            'ignored': ignored,
            'last_sync_at': last_sync.isoformat() if last_sync else None,
            'last_settled_at': last_settled.isoformat() if last_settled else None,
        }

    def repair_future_settlements(self, minutes: int = 180) -> Dict:
        """Reset records that were settled before kickoff plus wait window."""
        repaired = []
        now = datetime.now()
        fields_to_clear = [
            'actual_score', 'actual_result', 'actual_half_score', 'actual_half_result',
            'actual_half_full', 'settled_at', 'evaluation', 'hit_top1', 'hit_top3',
            'hit_top5', 'hit_top10', 'hit_top20', 'hit_top30', 'hit_1x2',
            'actual_score_rank', 'actual_score_prob',
        ]

        for record in self.records:
            if not record.get('settled') and record.get('sync_status') != 'synced':
                continue
            match_time = record.get('match_time')
            if not match_time:
                continue
            if _is_match_settle_due(match_time, minutes=minutes, now=now):
                continue

            for field in fields_to_clear:
                if field in record:
                    record[field] = None
            record['settled'] = False
            record['sync_status'] = 'pending'
            record['last_sync_error'] = '已撤销提前回填，等待比赛结束后重新同步'
            record['updated_at'] = now.isoformat()
            repaired.append({
                'match_id': record.get('match_id'),
                'home': record.get('home'),
                'away': record.get('away'),
                'match_time': match_time,
            })

        if repaired:
            self._save()
        return {'repaired': len(repaired), 'records': repaired}

    def audit_prediction_history(self, repair: bool = False, minutes: int = 180) -> Dict:
        """Audit historical records for unsafe calibration/backtest samples."""
        issues = []
        repaired = []
        now = datetime.now()

        def add_issue(record, code, severity='warning', detail=None):
            item = {
                'match_id': record.get('match_id'),
                'home': record.get('home'),
                'away': record.get('away'),
                'match_time': record.get('match_time'),
                'code': code,
                'severity': severity,
            }
            if detail is not None:
                item['detail'] = detail
            issues.append(item)

        for record in self.records:
            actual_score = record.get('actual_score')
            settled = bool(record.get('settled'))
            sync_status = record.get('sync_status')
            match_time = record.get('match_time')

            is_future_settled = False
            if (settled or sync_status == 'synced') and match_time:
                try:
                    is_future_settled = not _is_match_settle_due(match_time, minutes=minutes, now=now)
                except Exception:
                    is_future_settled = False

            if is_future_settled:
                add_issue(record, 'future_settlement', 'error')
                if repair:
                    for field in (
                        'actual_score', 'actual_result', 'actual_half_score', 'actual_half_result',
                        'actual_half_full', 'settled_at', 'evaluation', 'hit_top1', 'hit_top3',
                        'hit_top5', 'hit_top10', 'hit_top20', 'hit_top30', 'hit_1x2',
                        'actual_score_rank', 'actual_score_prob',
                    ):
                        if field in record:
                            record[field] = None
                    record['settled'] = False
                    record['sync_status'] = 'pending'
                    record['audit_repaired_at'] = now.isoformat()
                    repaired.append({'match_id': record.get('match_id'), 'action': 'reset_future_settlement'})
                continue

            if settled and actual_score:
                try:
                    home_goals, away_goals = map(int, str(actual_score).split('-'))
                    if home_goals < 0 or away_goals < 0 or home_goals > 15 or away_goals > 15:
                        add_issue(record, 'implausible_actual_score', 'error', actual_score)
                except Exception:
                    add_issue(record, 'invalid_actual_score', 'error', actual_score)

            result_quality = record.get('result_quality') or {}
            grade = result_quality.get('grade')
            if settled and not result_quality:
                add_issue(record, 'missing_result_quality', 'warning')
            elif grade in {'reject', 'low'}:
                add_issue(record, f'result_quality_{grade}', 'error' if grade == 'reject' else 'warning')
                if repair:
                    record['exclude_from_calibration'] = True
                    record['audit_repaired_at'] = now.isoformat()
                    repaired.append({'match_id': record.get('match_id'), 'action': 'exclude_from_calibration'})

            if record.get('half_time_data_quality') == 'invalid':
                add_issue(record, 'invalid_half_time_data', 'warning')
                if repair:
                    record['actual_half_score'] = None
                    record['actual_half_result'] = None
                    record['actual_half_full'] = None
                    record['half_time_data_quality'] = 'missing'
                    record['audit_repaired_at'] = now.isoformat()
                    repaired.append({'match_id': record.get('match_id'), 'action': 'clear_invalid_half_time'})

            if settled and _calibration_sample_weight(record) <= 0:
                add_issue(record, 'zero_calibration_weight', 'warning')
                if repair:
                    record['exclude_from_calibration'] = True
                    record['audit_repaired_at'] = now.isoformat()

        issue_counts = {}
        for issue in issues:
            issue_counts[issue['code']] = issue_counts.get(issue['code'], 0) + 1

        if repair and repaired:
            self._save()

        return {
            'checked': len(self.records),
            'issue_count': len(issues),
            'issue_counts': dict(sorted(issue_counts.items())),
            'issues': issues[:50],
            'repaired_count': len(repaired),
            'repaired': repaired[:50],
            'repair': repair,
        }
    
    def _update_calibrator(self, record: Dict):
        """更新贝叶斯校准库"""
        try:
            from .bayesian_calibration import get_calibrator
            
            calibrator = get_calibrator()
            predicted_scores = record.get('predicted_scores', {})
            actual_score = record.get('actual_score', '')
            league = record.get('league', '')
            total_line = record.get('total_line')
            asian = record.get('asian')
            sample_weight = _calibration_sample_weight(record)
            if sample_weight <= 0:
                return
            
            for score, prob in predicted_scores.items():
                is_correct = (score == actual_score)
                # 添加市场环境信息
                calibrator.add_record(score, prob, is_correct, league, total_line or 2.5, asian or 0.0, sample_weight)
            
            calibrator.save()
            log.debug(f"已更新贝叶斯校准库")
        except Exception as e:
            log.debug(f"更新贝叶斯校准库失败: {e}")
    
    def _update_market_db(self, record: Dict):
        """更新盘口聚类库"""
        try:
            from .market_clustering import get_cluster
            
            cluster = get_cluster()
            
            asian = record.get('asian')
            total_line = record.get('total_line')
            actual_score = record.get('actual_score', '')
            
            if asian is not None and total_line is not None and actual_score:
                cluster.add_match(asian, total_line, actual_score)
                cluster.save()
                log.debug(f"已更新盘口聚类库")
        except Exception as e:
            log.debug(f"更新盘口聚类库失败: {e}")
    
    def _update_score_frequency_db(self, record: Dict):
        """更新盘口比分频率库"""
        try:
            from .market_db import MarketScoreDB
            
            db = MarketScoreDB()
            
            asian = record.get('asian')
            total_line = record.get('total_line')
            actual_score = record.get('actual_score', '')
            
            if asian is not None and total_line is not None and actual_score:
                db.add_match_result(asian, total_line, actual_score)
                db.save()
                log.debug(f"已更新盘口比分频率库")
        except Exception as e:
            log.debug(f"更新盘口比分频率库失败: {e}")
    
    def _update_elo_ratings(self, record: Dict):
        """更新ELO评分"""
        try:
            from .elo import get_elo_system
            
            elo = get_elo_system()
            actual_score = record.get('actual_score', '')
            
            if actual_score:
                parts = actual_score.split('-')
                if len(parts) == 2:
                    home_goals, away_goals = map(int, parts)
                    elo.update_ratings(
                        home_team=record['home'],
                        away_team=record['away'],
                        home_score=home_goals,
                        away_score=away_goals,
                        league_type=record.get('league', '联赛')
                    )
                    log.debug(f"已更新ELO评分: {record['home']} vs {record['away']}")
        except Exception as e:
            log.debug(f"更新ELO评分失败: {e}")

    def _update_half_time_stats(self, record: Dict):
        """更新半场比分统计数据库"""
        try:
            from .half_time_stats import record_half_time_result

            if record.get('half_time_data_quality') != 'real' or not _is_result_quality_usable(record):
                return
            
            league = record.get('league', '')
            total_line = record.get('total_line')
            handicap = record.get('asian')
            actual_score = record.get('actual_score', '')
            actual_half_score = record.get('actual_half_score', '')
            
            # 只有当有真实半场比分时才记录
            if actual_half_score and actual_score and total_line is not None:
                try:
                    half_h, half_a = map(int, actual_half_score.split('-'))
                    full_h, full_a = map(int, actual_score.split('-'))
                    
                    # 判断比赛类型
                    match_type = 'league'
                    if league:
                        league_lower = league.lower()
                        if '杯' in league or 'cup' in league_lower or 'tournament' in league_lower:
                            match_type = 'cup'
                        elif '友谊' in league or 'friendly' in league_lower:
                            match_type = 'friendly'
                    
                    record_half_time_result(
                        league=league,
                        total_line=total_line,
                        handicap=handicap or 0.0,
                        match_type=match_type,
                        half_home=half_h,
                        half_away=half_a,
                        full_home=full_h,
                        full_away=full_a,
                        sample_weight=_calibration_sample_weight(record)
                    )
                    log.debug(f"已更新半场统计数据库")
                except Exception as e:
                    log.debug(f"解析半场比分失败: {e}")
        except Exception as e:
            log.debug(f"更新半场统计数据库失败: {e}")

    def _update_goal_count_stats(self, record: Dict):
        """更新总进球数校准数据库（赛后回填闭环）。

        此前 GoalCountCalibrator 的写入函数在生产代码从未被调用，导致校准表恒空、
        校准恒等返回。这里由预测比分分布边缘化出「预测进球数分布」与「期望总进球」，
        与真实总进球一起回填，逐步积累后校准器才能真正生效。
        """
        try:
            from .goal_count_calibrator import record_goal_count_result

            predicted_scores = record.get('predicted_scores') or {}
            actual_score = record.get('actual_score', '')
            total_line = record.get('total_line')
            if not predicted_scores or not actual_score or total_line is None:
                return

            goal_dist = {}
            expected_total = 0.0
            prob_sum = 0.0
            for score, prob in predicted_scores.items():
                try:
                    h, a = map(int, str(score).split('-'))
                    p = float(prob)
                except (ValueError, TypeError):
                    continue
                if p <= 0:
                    continue
                g = h + a
                goal_dist[g] = goal_dist.get(g, 0.0) + p
                expected_total += g * p
                prob_sum += p
            if prob_sum <= 0:
                return
            goal_dist = {g: p / prob_sum for g, p in goal_dist.items()}
            expected_total /= prob_sum

            try:
                full_h, full_a = map(int, actual_score.split('-'))
            except (ValueError, TypeError):
                return
            actual_total = full_h + full_a

            record_goal_count_result(
                league=record.get('league', ''),
                total_line=total_line,
                predicted_goal_dist=goal_dist,
                actual_total_goals=actual_total,
                expected_total_goals=expected_total,
                asian=record.get('asian') or 0.0,
                sample_weight=_calibration_sample_weight(record),
            )
            log.debug("已更新总进球校准数据库")
        except Exception as e:
            log.debug(f"更新总进球校准数据库失败: {e}")

    def _first_not_none(self, *values):
        for v in values:
            if v is not None:
                return v
        return None

    def _update_market_change_db(self, record: Dict):
        """赛后回填成功后，写入盘口变化数据库"""
        try:
            # 防止同一场重复结算导致重复写入
            if record.get('market_change_updated'):
                return

            from .market_db import MarketChangeDB, normalize_asian, normalize_ou

            odds = record.get('odds_snapshot') or {}
            asian_data = odds.get('asian') or {}
            total_data = odds.get('total') or {}

            # 兼容 analyze_asian 后结构
            asian_from = self._first_not_none(
                asian_data.get('open_handicap'),
                asian_data.get('open', {}).get('handicap')
            )

            asian_to = self._first_not_none(
                asian_data.get('handicap'),
                asian_data.get('close', {}).get('handicap'),
                record.get('asian')
            )

            # 兼容 analyze_total 后结构
            ou_from = self._first_not_none(
                total_data.get('open_line'),
                total_data.get('open', {}).get('line')
            )

            ou_to = self._first_not_none(
                total_data.get('close_line'),
                total_data.get('line'),
                total_data.get('close', {}).get('line'),
                record.get('total_line')
            )

            actual_score = record.get('actual_score')

            if asian_from is None or asian_to is None:
                log.debug(f"盘口变化库跳过：缺少亚盘开终盘 match_id={record.get('match_id')}")
                return

            if ou_from is None or ou_to is None:
                log.debug(f"盘口变化库跳过：缺少大小球开终盘 match_id={record.get('match_id')}")
                return

            if not actual_score:
                return

            asian_from_n = normalize_asian(asian_from)
            asian_to_n = normalize_asian(asian_to)
            ou_from_n = normalize_ou(ou_from)
            ou_to_n = normalize_ou(ou_to)

            if asian_from_n is None or asian_to_n is None or ou_from_n is None or ou_to_n is None:
                return

            db = MarketChangeDB()
            db.add_record(
                asian_from_n,
                asian_to_n,
                ou_from_n,
                ou_to_n,
                actual_score
            )
            db.save()

            record['market_change_updated'] = True
            record['market_change_updated_at'] = datetime.now().isoformat()
            record['market_change_key'] = f"{asian_from_n:.2f}→{asian_to_n:.2f}_{ou_from_n:.2f}→{ou_to_n:.2f}"

            log.info(
                f"盘口变化库已更新: "
                f"{record.get('home')} vs {record.get('away')} | "
                f"亚盘 {asian_from_n}->{asian_to_n}, "
                f"大小球 {ou_from_n}->{ou_to_n}, "
                f"比分 {actual_score}"
            )

        except Exception as e:
            log.debug(f"更新盘口变化数据库失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息（包含时间分层统计）"""
        total = len(self.records)
        settled = len([r for r in self.records if r.get('settled', False)])
        unsettled = total - settled
        
        # 时间分层统计
        time_layers = ['T-24h', 'T-6h', 'T-1h', 'T-15min', 'final']
        layer_stats = {
            layer: {
                'correct_top1': 0,
                'correct_top3': 0,
                'correct_top5': 0,
                'total': 0,
                'weighted_correct_top1': 0.0,
                'weighted_correct_top3': 0.0,
                'weighted_correct_top5': 0.0,
                'weighted_total': 0.0,
            }
            for layer in time_layers
        }
        
        # 计算命中率
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        correct_1x2 = 0
        valid_score_predictions = 0
        valid_1x2_predictions = 0
        actionable_total = 0
        actionable_correct = 0
        version_1x2 = {}
        
        for record in self.records:
            if not record.get('settled', False):
                continue
            
            actual_score = record.get('actual_score', '')
            predicted_scores = record.get('predicted_scores', {})
            actual_result = record.get('actual_result', '')
            predicted_1x2 = normalize_1x2_probs(record.get('predicted_1x2', {}))
            time_layers_data = record.get('time_layers', {})
            
            if not predicted_scores or not actual_score:
                continue
            valid_score_predictions += 1
            
            # 统计各时间层命中率
            for layer in time_layers:
                layer_pred = time_layers_data.get(layer)
                if layer == 'final' and not layer_pred:
                    layer_pred = predicted_scores
                if not layer_pred:
                    continue
                layer_weight = time_layer_weight(layer)
                
                sorted_scores = sorted(layer_pred.items(), key=lambda x: -x[1])
                layer_stats[layer]['total'] += 1
                layer_stats[layer]['weighted_total'] += layer_weight
                
                if sorted_scores and sorted_scores[0][0] == actual_score:
                    layer_stats[layer]['correct_top1'] += 1
                    layer_stats[layer]['weighted_correct_top1'] += layer_weight
                
                top3_scores = [s[0] for s in sorted_scores[:3]]
                if actual_score in top3_scores:
                    layer_stats[layer]['correct_top3'] += 1
                    layer_stats[layer]['weighted_correct_top3'] += layer_weight
                
                top5_scores = [s[0] for s in sorted_scores[:5]]
                if actual_score in top5_scores:
                    layer_stats[layer]['correct_top5'] += 1
                    layer_stats[layer]['weighted_correct_top5'] += layer_weight
            
            # 最终预测统计
            sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
            
            if sorted_scores and sorted_scores[0][0] == actual_score:
                correct_top1 += 1
            
            top3_scores = [s[0] for s in sorted_scores[:3]]
            if actual_score in top3_scores:
                correct_top3 += 1
            
            top5_scores = [s[0] for s in sorted_scores[:5]]
            if actual_score in top5_scores:
                correct_top5 += 1
            
            # 胜平负
            if actual_result and actual_result in predicted_1x2:
                valid_1x2_predictions += 1
                pred_result = max(predicted_1x2.items(), key=lambda x: x[1])[0]
                if pred_result == actual_result:
                    correct_1x2 += 1
                version = record.get('model_version') or 'legacy-unversioned'
                version_stats = version_1x2.setdefault(version, {'total': 0, 'correct': 0})
                version_stats['total'] += 1
                if pred_result == actual_result:
                    version_stats['correct'] += 1
                decision = record.get('decision_snapshot') or _prediction_decision_snapshot(predicted_1x2)
                if decision.get('eligible'):
                    actionable_total += 1
                    if pred_result == actual_result:
                        actionable_correct += 1

        hit_rate_top1 = correct_top1 / valid_score_predictions if valid_score_predictions > 0 else 0
        hit_rate_top3 = correct_top3 / valid_score_predictions if valid_score_predictions > 0 else 0
        hit_rate_top5 = correct_top5 / valid_score_predictions if valid_score_predictions > 0 else 0
        hit_rate_1x2 = correct_1x2 / valid_1x2_predictions if valid_1x2_predictions > 0 else 0

        # 计算各时间层命中率
        layer_hit_rates = {}
        for layer in time_layers:
            total_layer = layer_stats[layer]['total']
            if total_layer > 0:
                layer_hit_rates[layer] = {
                    'hit_rate_top1': layer_stats[layer]['correct_top1'] / total_layer,
                    'hit_rate_top3': layer_stats[layer]['correct_top3'] / total_layer,
                    'hit_rate_top5': layer_stats[layer]['correct_top5'] / total_layer,
                    'correct_top1': layer_stats[layer]['correct_top1'],
                    'correct_top3': layer_stats[layer]['correct_top3'],
                    'correct_top5': layer_stats[layer]['correct_top5'],
                    'total': total_layer,
                    'weight': time_layer_weight(layer),
                    'weighted_hit_rate_top1': (
                        layer_stats[layer]['weighted_correct_top1'] / layer_stats[layer]['weighted_total']
                        if layer_stats[layer]['weighted_total'] > 0 else 0.0
                    ),
                    'weighted_hit_rate_top3': (
                        layer_stats[layer]['weighted_correct_top3'] / layer_stats[layer]['weighted_total']
                        if layer_stats[layer]['weighted_total'] > 0 else 0.0
                    ),
                    'weighted_hit_rate_top5': (
                        layer_stats[layer]['weighted_correct_top5'] / layer_stats[layer]['weighted_total']
                        if layer_stats[layer]['weighted_total'] > 0 else 0.0
                    ),
                    'weighted_total': round(layer_stats[layer]['weighted_total'], 3),
                }
            else:
                layer_hit_rates[layer] = {
                    'hit_rate_top1': 0.0,
                    'hit_rate_top3': 0.0,
                    'hit_rate_top5': 0.0,
                    'correct_top1': 0,
                    'correct_top3': 0,
                    'correct_top5': 0,
                    'total': 0,
                    'weight': time_layer_weight(layer),
                    'weighted_hit_rate_top1': 0.0,
                    'weighted_hit_rate_top3': 0.0,
                    'weighted_hit_rate_top5': 0.0,
                    'weighted_total': 0.0,
                }
        
        return {
            'total_predictions': total,
            'settled': settled,
            'unsettled': unsettled,
            'hit_rate_top1': hit_rate_top1,
            'hit_rate_top3': hit_rate_top3,
            'hit_rate_top5': hit_rate_top5,
            'hit_rate_1x2': hit_rate_1x2,
            'by_time_layer': layer_hit_rates,
            'correct_top1': correct_top1,
            'correct_top3': correct_top3,
            'correct_top5': correct_top5,
            'correct_1x2': correct_1x2,
            'valid_score_predictions': valid_score_predictions,
            'valid_1x2_predictions': valid_1x2_predictions,
            'actionable_1x2': {
                'policy_version': 'selective-1x2-v2',
                'total': actionable_total,
                'correct': actionable_correct,
                'hit_rate': actionable_correct / actionable_total if actionable_total else 0.0,
                'coverage': actionable_total / valid_1x2_predictions if valid_1x2_predictions else 0.0,
                'min_probability': ACTIONABLE_MIN_PROBABILITY,
                'min_margin': ACTIONABLE_MIN_MARGIN,
            },
            'by_model_version': {
                version: {
                    **values,
                    'hit_rate_1x2': values['correct'] / values['total'] if values['total'] else 0.0,
                }
                for version, values in version_1x2.items()
            },
        }
    
    def get_ml_evaluation_stats(self, min_samples: int = 45) -> Dict:
        """
        获取 ML 模型评估统计（按维度）
        
        参数：
            min_samples: 最小样本数阈值
        
        返回：
            按维度统计的评估结果
        """
        # 五大联赛列表
        top_leagues = ['英超', '西甲', '德甲', '意甲', '法甲']
        
        # 初始化统计结构
        stats = {
            'overall': {
                'sample_count': 0,
                'base_1x2_logloss': [],
                'base_1x2_brier': [],
                'base_1x2_hit': [],
                'ml_1x2_logloss': [],
                'ml_1x2_brier': [],
                'ml_1x2_hit': [],
                'fused_5pct_logloss': [],
                'fused_5pct_brier': [],
                'fused_10pct_logloss': [],
                'fused_10pct_brier': [],
            },
            'by_league': {},
            'by_handicap_type': {
                'strong_favorite': {},  # 让球 >= 1.0
                'balanced': {},         # -0.5 < 让球 < 0.5
                'weak_favorite': {},    # 让球 <= -1.0
            },
            'by_total_line': {
                'low': {},              # <= 2.25
                'medium': {},           # 2.25 < x < 3.0
                'high': {},             # >= 3.0
            },
            'by_result': {
                'H': {},
                'D': {},
                'A': {},
            },
        }
        
        # 初始化联赛统计
        for league in top_leagues:
            stats['by_league'][league] = {
                'sample_count': 0,
                'base_1x2_logloss': [],
                'base_1x2_brier': [],
                'base_1x2_hit': [],
                'ml_1x2_logloss': [],
                'ml_1x2_brier': [],
                'ml_1x2_hit': [],
                'fused_5pct_logloss': [],
                'fused_5pct_brier': [],
                'fused_10pct_logloss': [],
                'fused_10pct_brier': [],
            }
        
        # 初始化其他维度统计
        for dim in ['by_handicap_type', 'by_total_line', 'by_result']:
            for key in stats[dim]:
                stats[dim][key] = {
                    'sample_count': 0,
                    'base_1x2_logloss': [],
                    'base_1x2_brier': [],
                    'base_1x2_hit': [],
                    'ml_1x2_logloss': [],
                    'ml_1x2_brier': [],
                    'ml_1x2_hit': [],
                    'fused_5pct_logloss': [],
                    'fused_5pct_brier': [],
                    'fused_10pct_logloss': [],
                    'fused_10pct_brier': [],
                }
        
        # 遍历记录收集数据
        for record in self.records:
            if not record.get('settled', False):
                continue
            
            actual_result = record.get('actual_result')
            if actual_result not in ['H', 'D', 'A']:
                continue
            
            evaluation = record.get('evaluation', {})
            league = record.get('league', '')
            handicap = record.get('asian', 0.0)
            total_line = record.get('total_line', 2.5)
            
            # 确定维度分类
            if league in top_leagues:
                league_key = league
            else:
                league_key = None
            
            # 让球盘类型
            if handicap >= 1.0:
                handicap_key = 'strong_favorite'
            elif handicap <= -1.0:
                handicap_key = 'weak_favorite'
            else:
                handicap_key = 'balanced'
            
            # 大小球类型
            if total_line <= 2.25:
                total_key = 'low'
            elif total_line >= 3.0:
                total_key = 'high'
            else:
                total_key = 'medium'
            
            # 结果类型
            result_key = actual_result
            
            # 收集评估数据到各维度
            dimensions = [('overall', None), 
                          ('by_handicap_type', handicap_key),
                          ('by_total_line', total_key),
                          ('by_result', result_key)]
            
            # 只有当联赛在五大联赛列表中时才添加联赛维度
            if league_key is not None:
                dimensions.insert(1, ('by_league', league_key))
            
            for dim_key, key in dimensions:
                if key is None:
                    target = stats[dim_key]
                elif key in stats[dim_key]:
                    target = stats[dim_key][key]
                else:
                    continue
                
                target['sample_count'] += 1
                
                # 添加评估指标
                for metric in ['base_1x2_logloss', 'base_1x2_brier', 'base_1x2_hit',
                               'ml_1x2_logloss', 'ml_1x2_brier', 'ml_1x2_hit',
                               'fused_5pct_logloss', 'fused_5pct_brier',
                               'fused_10pct_logloss', 'fused_10pct_brier']:
                    value = evaluation.get(metric)
                    if value is not None and not math.isnan(value):
                        target[metric].append(value)
        
        # 计算汇总统计
        def summarize_dimension(dim_stats):
            result = {}
            for key, data in dim_stats.items():
                if data['sample_count'] == 0:
                    result[key] = {
                        'sample_count': 0,
                        'base_1x2_logloss': None,
                        'base_1x2_brier': None,
                        'base_1x2_hit_rate': None,
                        'ml_1x2_logloss': None,
                        'ml_1x2_brier': None,
                        'ml_1x2_hit_rate': None,
                        'fused_5pct_logloss': None,
                        'fused_5pct_brier': None,
                        'fused_10pct_logloss': None,
                        'fused_10pct_brier': None,
                        'qualified': False,
                    }
                    continue
                
                # 计算均值
                result[key] = {
                    'sample_count': data['sample_count'],
                    'base_1x2_logloss': sum(data['base_1x2_logloss']) / len(data['base_1x2_logloss']) if data['base_1x2_logloss'] else None,
                    'base_1x2_brier': sum(data['base_1x2_brier']) / len(data['base_1x2_brier']) if data['base_1x2_brier'] else None,
                    'base_1x2_hit_rate': sum(data['base_1x2_hit']) / len(data['base_1x2_hit']) if data['base_1x2_hit'] else None,
                    'ml_1x2_logloss': sum(data['ml_1x2_logloss']) / len(data['ml_1x2_logloss']) if data['ml_1x2_logloss'] else None,
                    'ml_1x2_brier': sum(data['ml_1x2_brier']) / len(data['ml_1x2_brier']) if data['ml_1x2_brier'] else None,
                    'ml_1x2_hit_rate': sum(data['ml_1x2_hit']) / len(data['ml_1x2_hit']) if data['ml_1x2_hit'] else None,
                    'fused_5pct_logloss': sum(data['fused_5pct_logloss']) / len(data['fused_5pct_logloss']) if data['fused_5pct_logloss'] else None,
                    'fused_5pct_brier': sum(data['fused_5pct_brier']) / len(data['fused_5pct_brier']) if data['fused_5pct_brier'] else None,
                    'fused_10pct_logloss': sum(data['fused_10pct_logloss']) / len(data['fused_10pct_logloss']) if data['fused_10pct_logloss'] else None,
                    'fused_10pct_brier': sum(data['fused_10pct_brier']) / len(data['fused_10pct_brier']) if data['fused_10pct_brier'] else None,
                    'qualified': data['sample_count'] >= min_samples,
                }
            return result
        
        return {
            'overall': summarize_dimension({'overall': stats['overall']})['overall'],
            'by_league': summarize_dimension(stats['by_league']),
            'by_handicap_type': summarize_dimension(stats['by_handicap_type']),
            'by_total_line': summarize_dimension(stats['by_total_line']),
            'by_result': summarize_dimension(stats['by_result']),
            'min_samples_required': min_samples,
        }


# 全局实例
_global_history = PredictionHistory()


# ==================== ML 融合门槛判断 ====================

def check_ml_fusion_eligibility(ml_stats: Dict, test_set_samples: int = 0) -> Dict:
    """
    检查 ML 模型是否满足参与正式融合的门槛
    
    参数：
        ml_stats: ML 评估统计（来自 get_ml_evaluation_stats）
        test_set_samples: 测试集样本数
    
    返回：
        包含是否合格及原因的字典
    """
    overall = ml_stats.get('overall', {})
    shadow_samples = overall.get('sample_count', 0)
    
    conditions = {
        'test_set_samples': {
            'passed': test_set_samples >= 45,
            'actual': test_set_samples,
            'required': 45,
            'reason': '测试集样本 >= 45 场',
            'required_for_fusion': True,
        },
        'shadow_samples': {
            'passed': shadow_samples >= 45,
            'actual': shadow_samples,
            'required': 45,
            'reason': '影子实盘样本 >= 45 场',
            'required_for_fusion': True,
        },
        'ml_logloss_better': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': 'ML LogLoss < 基础模型 LogLoss',
            'required_for_fusion': False,
        },
        'ml_brier_not_worse': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': 'ML Brier Score <= 基础模型 Brier Score',
            'required_for_fusion': False,
        },
        'fused_5pct_logloss_better': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': '5% ML 融合后的 LogLoss < 基础模型 LogLoss',
            'required_for_fusion': False,
        },
        'fused_5pct_brier_not_worse': {
            'passed': False,
            'actual': None,
            'required': None,
            'reason': '5% ML 融合后的 Brier Score 不变差',
            'required_for_fusion': False,
        },
    }
    
    # 检查 LogLoss 和 Brier 条件（仅供参考，不阻断融合）
    base_logloss = overall.get('base_1x2_logloss')
    base_brier = overall.get('base_1x2_brier')
    ml_logloss = overall.get('ml_1x2_logloss')
    ml_brier = overall.get('ml_1x2_brier')
    fused_5pct_logloss = overall.get('fused_5pct_logloss')
    fused_5pct_brier = overall.get('fused_5pct_brier')
    
    if base_logloss is not None and ml_logloss is not None:
        conditions['ml_logloss_better']['passed'] = ml_logloss < base_logloss
        conditions['ml_logloss_better']['actual'] = f"{ml_logloss:.4f} vs {base_logloss:.4f}"
    
    if base_brier is not None and ml_brier is not None:
        conditions['ml_brier_not_worse']['passed'] = ml_brier <= base_brier
        conditions['ml_brier_not_worse']['actual'] = f"{ml_brier:.4f} vs {base_brier:.4f}"
    
    if base_logloss is not None and fused_5pct_logloss is not None:
        conditions['fused_5pct_logloss_better']['passed'] = fused_5pct_logloss < base_logloss
        conditions['fused_5pct_logloss_better']['actual'] = f"{fused_5pct_logloss:.4f} vs {base_logloss:.4f}"
    
    if base_brier is not None and fused_5pct_brier is not None:
        conditions['fused_5pct_brier_not_worse']['passed'] = fused_5pct_brier <= base_brier
        conditions['fused_5pct_brier_not_worse']['actual'] = f"{fused_5pct_brier:.4f} vs {base_brier:.4f}"
    
    # 仅样本数达标即可参与融合；指标条件仅作参考
    eligible = all(
        cond['passed']
        for cond in conditions.values()
        if cond.get('required_for_fusion')
    )
    metrics_passed = all(
        cond['passed']
        for cond in conditions.values()
        if not cond.get('required_for_fusion')
    )
    
    return {
        'eligible': eligible,
        'metrics_passed': metrics_passed,
        'conditions': conditions,
        'test_set_samples': test_set_samples,
        'shadow_samples': shadow_samples,
        'stats': overall,
    }


def get_ml_fusion_weight(eligible: bool, shadow_samples: int, 
                        current_weight: float = 0.0) -> float:
    """
    根据资格和样本数确定 ML 融合权重
    
    参数：
        eligible: 是否满足融合门槛
        shadow_samples: 影子实盘样本数
        current_weight: 当前权重
    
    返回：
        建议的 ML 融合权重
    """
    if not eligible:
        return 0.0
    
    # 根据样本数逐步提升权重
    max_weight = 0.15
    
    if shadow_samples >= 500:
        # 500+ 场可以考虑更高权重，但不超过 0.15
        if current_weight < 0.10:
            return min(0.10, max_weight)
        elif current_weight < 0.15:
            return min(0.15, max_weight)
        return current_weight
    elif shadow_samples >= 300:
        # 300-500 场，最高 0.10
        return min(0.10, max_weight)
    elif shadow_samples >= 45:
        # 45-300 场，初始权重 0.05
        return min(0.05, max_weight)
    else:
        return 0.0


# ==================== 便捷函数 ====================

def save_prediction(match_id: str, league: str, home: str, away: str,
                   match_time: str, predicted_scores: Dict[str, float],
                   predicted_1x2: Dict[str, float], asian: float = None,
                   total_line: float = None, odds_data: Dict = None,
                   predicted_half_full: Dict[str, float] = None,
                   # 影子预测相关字段
                   base_1x2: Dict[str, float] = None,
                   ml_1x2: Dict[str, float] = None,
                   ml_model_version: str = None,
                   ml_available: bool = False,
                   ml_feature_snapshot: Dict = None,
                   lottery_handicap: int = None,
                   predicted_rqspf: Dict[str, float] = None,
                   goal_count: Dict = None,
                   model_version: str = PRODUCTION_MODEL_VERSION):
    """保存预测记录"""
    return _global_history.add_prediction(
        match_id, league, home, away, match_time,
        predicted_scores, predicted_1x2, asian, total_line, odds_data,
        predicted_half_full=predicted_half_full,
        base_1x2=base_1x2,
        ml_1x2=ml_1x2,
        ml_model_version=ml_model_version,
        ml_available=ml_available,
        ml_feature_snapshot=ml_feature_snapshot,
        lottery_handicap=lottery_handicap,
        predicted_rqspf=predicted_rqspf,
        goal_count=goal_count,
        model_version=model_version,
    )


def sync_results():
    """同步比赛结果"""
    return auto_sync_results()


def auto_sync_results():
    """
    自动同步比赛结果（三层兜底）
    1. 第一优先：match_id 对应赛果页面
    2. 第二优先：主队 + 客队 + 比赛日期模糊匹配
    3. 第三优先：放弃自动同步，标记 failed
    """
    ready = _global_history.get_ready_to_sync()
    
    if not ready:
        return {'synced': 0, 'failed': 0, 'message': '没有需要同步的比赛'}
    
    synced = 0
    failed = 0
    
    for record in ready:
        match_id = record['match_id']
        home = record['home']
        away = record['away']
        match_time = record.get('match_time', '')
        league = record.get('league', '')

        if not _is_valid_match_id(match_id):
            log.debug(f"跳过非数字 match_id 的同步: {home} vs {away} ({match_id})")
            continue
        
        try:
            # 三层兜底抓取赛果
            result = fetch_result_by_match_id(match_id, match_time)
            if not result:
                result = fetch_result_by_team_and_date(home, away, match_time)
            
            if result:
                if _global_history.update_result(
                    match_id,
                    result['score'],
                    result['result'],
                    source=result.get('source'),
                ):
                    synced += 1
                    log.info(f"同步成功: {home} vs {away} -> {result['score']}")
                else:
                    failed += 1
            else:
                _global_history.update_result(match_id, None, None, error='未找到赛果')
                failed += 1
                log.warning(f"无法获取比赛结果: {home} vs {away}")
                
        except Exception as e:
            _global_history.update_result(match_id, None, None, error=str(e))
            failed += 1
            log.error(f"同步比赛结果异常: {home} vs {away} - {e}")
    
    return {
        'synced': synced,
        'failed': failed,
        'total': len(ready),
        'message': f'结算了 {synced}/{len(ready)} 场比赛，失败 {failed} 场'
    }


def _is_valid_match_id(match_id: str) -> bool:
    """仅对 500.com 数字型 fid 尝试抓取赛果"""
    return bool(match_id) and str(match_id).isdigit()


def _extract_score_text(raw: str) -> Optional[str]:
    """将页面比分文本规范为 home-away 格式，未开赛返回 None"""
    text = (raw or '').strip().replace('：', ':')
    if not text or text.upper() == 'VS':
        return None

    text = re.sub(r'\s+', '', text)
    if ':' in text:
        home_goals, away_goals = text.split(':', 1)
    elif '-' in text:
        home_goals, away_goals = text.split('-', 1)
    else:
        return None

    if not (home_goals.isdigit() and away_goals.isdigit()):
        return None

    home_goals = int(home_goals)
    away_goals = int(away_goals)
    if home_goals > 15 or away_goals > 15:
        return None

    return f"{home_goals}-{away_goals}"


def _fetch_match_html(match_id: str) -> str:
    """复用足球模块抓取逻辑，保持与预测数据同源"""
    from . import fetch as fetch_html
    return fetch_html(f'https://odds.500.com/fenxi/shuju-{match_id}.shtml')


def _parse_shuju_score(html: str, match_id: str) -> Optional[str]:
    """从 odds.500.com 赛事数据页解析终场比分"""
    patterns = [
        rf'shuju-{re.escape(match_id)}\.shtml[^>]*>.*?<em class="l">[^<]*</em><span class="gray">([^<]+)</span><em class="r">[^<]*</em>',
        rf'<em class="l">[^<]*</em><span class="gray">([^<]+)</span><em class="r">[^<]*</em>',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.DOTALL)
        if m:
            score = _extract_score_text(m.group(1))
            if score:
                return score
    return None


def fetch_result_by_match_id(match_id: str, match_time: str = '') -> Optional[Dict]:
    """
    通过 match_id 抓取赛果：
    1. live.500.com 按 fid + 动态日期（竞彩官方赛果页）
    2. odds.500.com 赛事数据页（兜底）
    """
    if not _is_valid_match_id(match_id):
        return None

    if match_time:
        score = _fetch_live_score_by_fid(match_id, match_time)
        if score:
            result = _parse_score_string(score)
            if result:
                result['source'] = 'live_fid'
            return result

    try:
        html = _fetch_match_html(match_id)
        score = _parse_shuju_score(html, match_id)
        if score:
            log.info(f"通过 shuju 页面抓取赛果: match_id={match_id} -> {score}")
            result = _parse_score_string(score)
            if result:
                result['source'] = 'shuju'
            return result
    except Exception as e:
        log.debug(f"shuju 页面抓取失败: {e}")

    return None


def _parse_match_datetime(match_time: str) -> Optional[datetime]:
    """解析比赛时间，兼容 MM-DD HH:MM 与完整日期格式"""
    if not match_time:
        return None

    now = datetime.now()
    text = str(match_time).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M'):
        try:
            match_dt = datetime.strptime(text, fmt)
            return match_dt
        except ValueError:
            continue
    try:
        match_dt = datetime.strptime(f"{now.year}-{text}", '%Y-%m-%d %H:%M')
        if now.month == 12 and match_dt.month == 1:
            match_dt = match_dt.replace(year=now.year + 1)
        elif now.month == 1 and match_dt.month == 12:
            match_dt = match_dt.replace(year=now.year - 1)
        return match_dt
    except ValueError:
        pass
    return None


def _is_match_settle_due(match_time: str, minutes: int = 180, now: datetime = None) -> bool:
    """Return True only after kickoff plus the settlement wait window."""
    match_dt = _parse_match_datetime(match_time)
    if not match_dt:
        return False
    now = now or datetime.now()
    return now >= match_dt + timedelta(minutes=minutes)


def _live_query_dates(match_time: str) -> List[str]:
    """
    动态计算 live.500.com 的 ?e= 查询日期。

    竞彩赛果页规则（见 https://live.500.com/?e=YYYY-MM-DD ）：
    - e=某日 的页面展示该「开售日」对应场次的赛果
    - 开球时间为 06-13 03:00 的比赛，出现在 e=2026-06-12 页面（开球日前一天）
    - 因此优先查 kickoff_date - 1，再查当日及邻近日，最后兜底今天/昨天
    """
    today = datetime.now().date()
    candidates = []

    match_dt = _parse_match_datetime(match_time)
    if match_dt:
        kickoff_date = match_dt.date()
        candidates.extend([
            kickoff_date - timedelta(days=1),
            kickoff_date,
            kickoff_date - timedelta(days=2),
            kickoff_date + timedelta(days=1),
        ])

    candidates.extend([today, today - timedelta(days=1), today - timedelta(days=2)])

    seen = set()
    dates = []
    for day in candidates:
        key = day.strftime('%Y-%m-%d')
        if key not in seen:
            seen.add(key)
            dates.append(key)
    return dates


def _fetch_live_html(search_date: str) -> str:
    from . import fetch as fetch_html
    return fetch_html(f'https://live.500.com/?e={search_date}')


def _parse_live_row_final_score(row: str) -> Optional[str]:
    """
    从 live 表格行解析全场比分。

    live.500.com 列结构：
    - <div class="pk"> 中 clt1 / clt3 = 全场比分（如 1-1）
    - 其后 class="red" 的 td = 半场比分（如 0-1），不能当作终场
    """
    pk_m = re.search(
        r'<div class="pk">.*?class="clt1"[^>]*>\s*(\d+)\s*</a>.*?class="clt3"[^>]*>\s*(\d+)\s*</a>',
        row,
        re.DOTALL,
    )
    if pk_m:
        score = _extract_score_text(f"{pk_m.group(1)}-{pk_m.group(2)}")
        if score:
            return score
    return None


def _fetch_live_score_by_fid(match_id: str, match_time: str) -> Optional[str]:
    """在 live.500.com 按 fid 查找赛果，日期动态推算"""
    for search_date in _live_query_dates(match_time):
        try:
            html = _fetch_live_html(search_date)
        except Exception as e:
            log.debug(f"live 页面抓取失败 e={search_date}: {e}")
            continue

        row_m = re.search(
            rf'<tr[^>]*\bfid="{re.escape(match_id)}"[^>]*>.*?</tr>',
            html,
            re.DOTALL,
        )
        if not row_m:
            continue

        score = _parse_live_row_final_score(row_m.group(0))
        if score:
            log.info(f"通过 live 页面(fid)抓取赛果: match_id={match_id}, e={search_date} -> {score}")
            return score

    return None


def _parse_live_row_score(row: str, home: str, away: str) -> Optional[str]:
    """从 live.500.com 单行比赛记录提取终场比分"""
    if home not in row or away not in row:
        return None

    score = _parse_live_row_final_score(row)
    if score:
        return score

    fid_m = re.search(r'fid="(\d+)"', row)
    if fid_m:
        # 行内已有 fid，直接解析比分列，避免重复请求
        return None

    home_idx = row.find(home)
    away_idx = row.find(away)
    if home_idx < 0 or away_idx < 0:
        return None

    start = min(home_idx, away_idx)
    end = max(home_idx, away_idx) + max(len(home), len(away))
    segment = row[start:end]

    for pat in (
        r'>(\d{1,2})\s*[-:：]\s*(\d{1,2})<',
        r'(\d{1,2})\s*[-:：]\s*(\d{1,2})',
    ):
        m = re.search(pat, segment)
        if m:
            score = _extract_score_text(f"{m.group(1)}-{m.group(2)}")
            if score:
                return score
    return None


def fetch_result_by_team_and_date(home: str, away: str, match_time: str) -> Optional[Dict]:
    """
    第二优先：通过球队名和比赛时间在 live.500.com 模糊匹配抓取赛果
    """
    try:
        for search_date in _live_query_dates(match_time):
            try:
                html = _fetch_live_html(search_date)
            except Exception as e:
                log.debug(f"live 页面抓取失败 e={search_date}: {e}")
                continue

            for row in re.finditer(r'<tr[^>]*>.*?</tr>', html, re.DOTALL):
                score = _parse_live_row_score(row.group(0), home, away)
                if score:
                    log.info(f"通过 live 页面(球队)抓取赛果: {home} vs {away}, e={search_date} -> {score}")
                    result = _parse_score_string(score)
                    if result:
                        result['source'] = 'live_team'
                    return result

    except Exception as e:
        log.debug(f"通过球队名+日期抓取失败: {e}")

    return None


def _parse_score_result(score_match) -> Optional[Dict]:
    """解析比分匹配结果"""
    home_goals = int(score_match.group(1))
    away_goals = int(score_match.group(2))
    return _parse_score_string(f"{home_goals}-{away_goals}")


def _parse_score_string(score_str: str) -> Optional[Dict]:
    """解析比分字符串"""
    try:
        parts = score_str.split('-')
        if len(parts) != 2:
            return None
        
        home_goals = int(parts[0])
        away_goals = int(parts[1])
        
        if home_goals > away_goals:
            result = 'H'
        elif home_goals < away_goals:
            result = 'A'
        else:
            result = 'D'
        
        return {'score': score_str, 'result': result}
    except:
        return None


def get_history_stats() -> Dict:
    """获取历史统计"""
    return _global_history.get_stats()


def get_sync_status_summary() -> Dict:
    """获取同步状态汇总"""
    return _global_history.get_sync_status_summary()


def repair_future_settlements(minutes: int = 180) -> Dict:
    """撤销尚未到结算时间却已经回填的记录。"""
    return _global_history.repair_future_settlements(minutes=minutes)


def audit_prediction_history(repair: bool = False, minutes: int = 180) -> Dict:
    return _global_history.audit_prediction_history(repair=repair, minutes=minutes)


def get_prediction_records(include_hidden: bool = False) -> List[Dict]:
    """
    获取预测记录列表
    
    参数：
        include_hidden: 是否包含已失败的记录
    """
    records = []
    for record in _global_history.records:
        if not include_hidden:
            if record.get('sync_status') == 'failed':
                continue
        
        is_future_settled = (
            (record.get('settled') or record.get('sync_status') == 'synced')
            and record.get('match_time')
            and not _is_match_settle_due(record.get('match_time'), minutes=180)
        )

        records.append({
            'match_id': record.get('match_id'),
            'league': record.get('league'),
            'home': record.get('home'),
            'away': record.get('away'),
            'match_time': record.get('match_time'),
            'settled': False if is_future_settled else record.get('settled', False),
            'actual_score': None if is_future_settled else record.get('actual_score'),
            'sync_status': 'pending' if is_future_settled else record.get('sync_status', 'pending'),
            'sync_attempts': record.get('sync_attempts', 0),
            'last_sync_error': (
                '比赛尚未到结算时间，已隐藏提前回填结果'
                if is_future_settled else record.get('last_sync_error')
            ),
            'next_sync_at': record.get('next_sync_at'),
            'hit_top1': None if is_future_settled else record.get('hit_top1'),
            'hit_top3': None if is_future_settled else record.get('hit_top3'),
            # 预测记录页以两个竞彩赛果市场为主。精确比分仍保留在存储和
            # 完整导出中，列表只在赛后输出 actual_score。
            'predicted_1x2': record.get('predicted_1x2'),
            'predicted_rqspf': record.get('predicted_rqspf'),
            'lottery_handicap': record.get('lottery_handicap'),
            'actual_result': None if is_future_settled else record.get('actual_result'),
            'actual_rqspf': None if is_future_settled else record.get('actual_rqspf'),
            'hit_1x2': None if is_future_settled else record.get('hit_1x2'),
            'hit_rqspf': None if is_future_settled else record.get('hit_rqspf'),
        })
    
    # 按比赛时间倒序排列
    records.sort(key=lambda x: x.get('match_time', ''), reverse=True)
    return records


def get_prediction_export() -> Dict:
    """返回可用于离线回测/校准的完整预测记录（不包含数据库配置）。"""
    export_fields = (
        'match_id', 'league', 'home', 'away', 'match_time',
        'created_at', 'updated_at', 'settled_at', 'model_version',
        'prediction_logic_version', 'asian', 'total_line',
        'predicted_scores', 'predicted_1x2', 'predicted_rqspf', 'goal_count',
        'lottery_handicap', 'predicted_half_full',
        'time_layers', 'odds_layers', 'odds_snapshot',
        'base_1x2', 'ml_1x2', 'ml_model_version', 'ml_available',
        'ml_feature_snapshot', 'actual_score', 'actual_result',
        'actual_half_score', 'actual_half_result', 'actual_half_full',
        'settled', 'sync_status', 'evaluation', 'hit_top1', 'hit_top3',
        'hit_top5', 'hit_1x2', 'hit_rqspf', 'actual_rqspf',
        'actual_score_rank', 'actual_score_prob',
    )
    records = [
        {key: record.get(key) for key in export_fields if key in record}
        for record in _global_history.records
    ]
    records.sort(key=lambda item: item.get('match_time', ''))
    return {
        'schema_version': 'football-prediction-export-v1',
        'exported_at': datetime.now().astimezone().isoformat(),
        'record_count': len(records),
        'settled_count': sum(
            1 for record in records
            if record.get('settled') or record.get('actual_score')
        ),
        'stats': _global_history.get_stats(),
        'records': records,
    }


def hide_failed_records():
    """隐藏所有失败记录（标记为 ignored）"""
    for record in _global_history.records:
        if record.get('sync_status') == 'failed':
            record['sync_status'] = 'ignored'
    _global_history._save()
    log.info("已隐藏所有失败记录")


def predict_at_time_layer(match: Dict, time_layer: str) -> bool:
    """
    在指定时间层对比赛进行预测并落库

    参数：
        match: fetch_match_list 的比赛字典（含 match_id/home/away/league/time）
        time_layer: 时间层标识

    返回：
        是否成功
    """
    match_id = match.get('match_id') or match.get('mid')
    try:
        from . import analyze_match

        log.info(f"正在进行时间分层预测: match_id={match_id}, time_layer={time_layer}")

        # analyze_match 内部会保存预测记录；传完整字段以便记录含队名/联赛
        analyze_match({
            'match_id': match_id,
            'home': match.get('home', ''),
            'away': match.get('away', ''),
            'league': match.get('league', ''),
            'time': match.get('time', match.get('match_time', '')),
        }, force_refresh=True)

        log.info(f"时间分层预测成功: match_id={match_id}, time_layer={time_layer}")
        return True

    except Exception as e:
        log.error(f"时间分层预测异常: match_id={match_id}, time_layer={time_layer}, error={e}")
        return False


def scan_and_predict_time_layers() -> Dict[str, int]:
    """
    扫描未来比赛并在时间分层点进行预测
    
    返回：
        统计结果 {'T-24h': 数量, 'T-6h': 数量, 'T-1h': 数量, 'T-15min': 数量}
    """
    result = {'T-24h': 0, 'T-6h': 0, 'T-1h': 0, 'T-15min': 0}
    
    try:
        from .data_loader import fetch_future_matches
        
        matches = fetch_future_matches()
        
        for match in matches:
            match_id = match.get('mid', match.get('match_id'))
            match_time_str = match.get('time', match.get('match_time', ''))
            
            if not match_id or not match_time_str:
                continue
            
            # 推断当前应该属于哪个时间层
            time_layer = infer_time_layer(match_time_str)
            
            # 检查是否需要在这个时间层进行预测
            if time_layer in result:
                # 检查是否已经在这个时间层预测过（避免重复）
                history = PredictionHistory()
                existing = history.get_record(match_id)
                
                if existing:
                    time_layers = existing.get('time_layers', {})
                    if time_layers.get(time_layer) is not None:
                        log.debug(f"已在 {time_layer} 层预测过: {match_id}")
                        continue
                
                # 执行预测
                if predict_at_time_layer(match, time_layer):
                    result[time_layer] += 1
        
        log.info(f"时间分层扫描完成: T-24h={result['T-24h']}, T-6h={result['T-6h']}, T-1h={result['T-1h']}, T-15min={result['T-15min']}")
        
    except Exception as e:
        log.error(f"时间分层扫描异常: {e}")
    
    return result


def get_history() -> PredictionHistory:
    """获取全局预测历史管理器实例"""
    return _global_history


def start_background_sync(interval_seconds: int = 7200):
    """
    启动后台定时同步线程（使用 APScheduler）
    
    参数：
        interval_seconds: 同步间隔（秒），默认2小时
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.schedulers.blocking import BlockingScheduler
        
        scheduler = BlockingScheduler(timezone="Asia/Shanghai")
        
        # 每2小时同步一次（赛后回填）
        scheduler.add_job(
            auto_sync_results,
            'interval',
            seconds=interval_seconds,
            id='football_result_sync',
            replace_existing=True
        )
        
        # 每10分钟扫描时间分层预测
        scheduler.add_job(
            scan_and_predict_time_layers,
            'interval',
            seconds=600,  # 10分钟
            id='football_time_layer_scan',
            replace_existing=True
        )
        
        scheduler.start()
        log.info(f"已启动后台同步调度器，同步间隔 {interval_seconds} 秒，时间分层扫描间隔 600 秒")
        return scheduler
        
    except ImportError:
        # 如果没有 APScheduler，使用简单线程
        log.warning("APScheduler 未安装，使用简单线程调度")
        
        def sync_loop():
            while True:
                try:
                    # 赛后回填
                    result = auto_sync_results()
                    if result['synced'] > 0 or result['failed'] > 0:
                        log.info(f"后台同步: {result['message']}")
                    
                    # 时间分层扫描（每10分钟）
                    layer_result = scan_and_predict_time_layers()
                    if sum(layer_result.values()) > 0:
                        log.info(f"时间分层预测: {layer_result}")
                        
                except Exception as e:
                    log.error(f"后台同步异常: {e}")
                
                time.sleep(interval_seconds)
        
        thread = Thread(target=sync_loop, daemon=True)
        thread.start()
        log.info(f"已启动后台同步线程，间隔 {interval_seconds} 秒")
        return thread


# ==================== 测试 ====================

def main():
    print("=== 预测历史模块测试 ===")
    
    # 查看统计
    stats = get_history_stats()
    print(f"统计信息: {stats}")
    
    # 手动同步
    result = sync_results()
    print(f"同步结果: {result}")


if __name__ == '__main__':
    main()
