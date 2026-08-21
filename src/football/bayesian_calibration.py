#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯校准层 - 职业模型的核心模块
==================================

功能：
1. 收集模型预测与实际结果的历史数据
2. 使用贝叶斯方法校准预测概率
3. 自动修正系统偏差

例如：
模型预测 1:1 = 13%
历史发现实际只打出 9%
自动修正为 9%
"""

import os
import json
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..common import kv_store
from ..common.logger import setup_logger

log = setup_logger('football.bayesian_calibration')

# ==================== 常量配置 ====================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
CALIBRATION_DB_FILE = os.path.join(DATA_DIR, 'calibration_db.json')

class BayesianCalibrator:
    """
    贝叶斯校准器 - 市场环境校准版
    
    使用层级回退机制进行概率校准：
    - 层级1：比分 + 联赛 + 大小球 + 让球（样本 >= 50）
    - 层级2：比分 + 大小球 + 让球（样本 >= 100）
    - 层级3：仅比分全局（样本 >= 200）
    - 不足：不校准
    
    避免桶过细导致过拟合
    """
    
    def __init__(self):
        self.history = {}  # {bucket_key: {'count': 0, 'success': 0, 'predicted_sum': 0.0}}
        self._load()
    
    def _load(self):
        """加载校准数据库"""
        try:
            self.history = kv_store.load('calibration_db') or {}
            log.debug("已加载贝叶斯校准数据库: %d 个分桶", len(self.history))
        except Exception as e:
            log.warning("加载贝叶斯校准数据库失败: %s", e)
            self.history = {}

    def save(self):
        """保存校准数据库"""
        kv_store.save('calibration_db', self.history)
    
    def _get_bucket_key(self, score: str, league: str, total_line: float, asian: float, level: int) -> str:
        """
        生成分桶键
        
        参数：
            score: 比分
            league: 联赛名称
            total_line: 大小球盘口
            asian: 让球盘口
            level: 层级（1=最细，3=最粗）
        
        返回：
            分桶键字符串
        """
        if level == 1:
            # 层级1：比分 + 联赛 + 大小球 + 让球
            bucketed_line = round(total_line * 4) / 4
            bucketed_asian = round(asian * 2) / 2
            return f"{score}_{league}_{bucketed_line:.2f}_{bucketed_asian:+.2f}"
        elif level == 2:
            # 层级2：比分 + 大小球 + 让球
            bucketed_line = round(total_line * 4) / 4
            bucketed_asian = round(asian * 2) / 2
            return f"{score}_all_{bucketed_line:.2f}_{bucketed_asian:+.2f}"
        else:
            # 层级3：仅比分
            return f"{score}_all_all_all"
    
    def add_record(self, score: str, predicted_prob: float, actual_outcome: bool,
                   league: str = '', total_line: float = 2.5, asian: float = 0.0,
                   sample_weight: float = 1.0):
        """
        添加一条校准记录
        
        参数：
            score: 比分（如 "1-1"）
            predicted_prob: 模型预测概率
            actual_outcome: 实际是否发生
            league: 联赛名称
            total_line: 大小球盘口
            asian: 让球盘口
        """
        # 为所有层级添加记录
        try:
            sample_weight = max(0.0, min(1.0, float(sample_weight)))
        except (TypeError, ValueError):
            sample_weight = 1.0
        if sample_weight <= 0:
            return

        for level in [1, 2, 3]:
            bucket_key = self._get_bucket_key(score, league, total_line, asian, level)
            if bucket_key not in self.history:
                self.history[bucket_key] = {
                    'count': 0,
                    'weighted_count': 0.0,
                    'success': 0,
                    'weighted_success': 0.0,
                    'predicted_sum': 0.0,
                    'weighted_predicted_sum': 0.0,
                }
            
            self.history[bucket_key]['count'] += 1
            self.history[bucket_key]['predicted_sum'] += predicted_prob
            self.history[bucket_key]['weighted_count'] = (
                self.history[bucket_key].get('weighted_count', self.history[bucket_key]['count'] - 1)
                + sample_weight
            )
            self.history[bucket_key]['weighted_predicted_sum'] = (
                self.history[bucket_key].get('weighted_predicted_sum', self.history[bucket_key]['predicted_sum'] - predicted_prob)
                + predicted_prob * sample_weight
            )
            if actual_outcome:
                self.history[bucket_key]['success'] += 1
                self.history[bucket_key]['weighted_success'] = (
                    self.history[bucket_key].get('weighted_success', self.history[bucket_key]['success'] - 1)
                    + sample_weight
                )
    
    def calibrate(self, score: str, predicted_prob: float,
                  league: str = '', total_line: float = 2.5, asian: float = 0.0) -> float:
        """
        校准预测概率（使用层级回退）
        
        参数：
            score: 比分
            predicted_prob: 原始预测概率
            league: 联赛名称
            total_line: 大小球盘口
            asian: 让球盘口
        
        返回：
            校准后的概率
        """
        # 层级回退策略
        level_requirements = [
            (1, 50),  # 层级1需要50个样本
            (2, 100), # 层级2需要100个样本
            (3, 200)  # 层级3需要200个样本
        ]
        
        for level, min_samples in level_requirements:
            bucket_key = self._get_bucket_key(score, league, total_line, asian, level)
            if bucket_key in self.history and self.history[bucket_key].get('weighted_count', self.history[bucket_key]['count']) >= min_samples:
                record = self.history[bucket_key]
                total = record.get('weighted_count', record['count'])
                success = record.get('weighted_success', record['success'])
                predicted_sum = record.get('weighted_predicted_sum', record['predicted_sum'])
                avg_predicted = predicted_sum / total
                
                if avg_predicted < 0.001:
                    continue
                
                # 计算校准因子：实际命中率 / 平均预测概率
                actual_rate = success / total
                correction_factor = actual_rate / avg_predicted
                
                # 校准概率（限制在合理范围内）
                calibrated = predicted_prob * correction_factor
                calibrated = max(0.001, min(0.999, calibrated))
                
                # 加权融合：历史数据越多，权重越大
                weight = min(total / 1000, 1.0)
                final = (1 - weight) * predicted_prob + weight * calibrated
                
                return final
        
        # 所有层级都不满足要求，返回原始概率
        return predicted_prob
    
    def calibrate_all(self, predictions: Dict[str, float],
                   league: str = '', total_line: float = 2.5, asian: float = 0.0) -> Dict[str, float]:
        """
        校准所有预测概率
        
        参数：
            predictions: {比分: 概率}
            league: 联赛名称
            total_line: 大小球盘口
            asian: 让球盘口
        
        返回：
            校准后的概率字典
        """
        calibrated = {}
        total_prob = 0.0
        
        for score, prob in predictions.items():
            calibrated[score] = self.calibrate(score, prob, league, total_line, asian)
            total_prob += calibrated[score]
        
        # 归一化
        if total_prob > 0:
            return {k: v / total_prob for k, v in calibrated.items()}
        return predictions

# ==================== 全局实例 ====================
_calibrator = None

def get_calibrator() -> BayesianCalibrator:
    """获取全局校准器实例"""
    global _calibrator
    if _calibrator is None:
        _calibrator = BayesianCalibrator()
    return _calibrator

def calibrate_predictions(predictions: Dict[str, float],
                           league: str = '', total_line: float = 2.5, asian: float = 0.0) -> Dict[str, float]:
    """校准预测概率的便捷接口"""
    return get_calibrator().calibrate_all(predictions, league, total_line, asian)
