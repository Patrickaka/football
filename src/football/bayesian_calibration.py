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

# ==================== 常量配置 ====================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
CALIBRATION_DB_FILE = os.path.join(DATA_DIR, 'calibration_db.json')

class BayesianCalibrator:
    """
    贝叶斯校准器
    
    使用Beta分布进行概率校准
    Beta(alpha, beta) 其中：
    - alpha = 成功次数 + 1（伪计数）
    - beta = 失败次数 + 1（伪计数）
    """
    
    def __init__(self):
        self.history = {}  # {score: {'predicted': [...], 'actual': [...]}
        self._load()
    
    def _load(self):
        """加载校准数据库"""
        if os.path.exists(CALIBRATION_DB_FILE):
            try:
                with open(CALIBRATION_DB_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"已加载贝叶斯校准数据库，{len(self.history)} 种比分")
            except Exception as e:
                print(f"加载校准数据库失败: {e}")
                self.history = {}
    
    def save(self):
        """保存校准数据库"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CALIBRATION_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def add_record(self, score: str, predicted_prob: float, actual_outcome: bool):
        """
        添加一条校准记录
        
        参数：
            score: 比分（如 "1-1"）
            predicted_prob: 模型预测概率
            actual_outcome: 实际是否发生
        """
        if score not in self.history:
            self.history[score] = {'count': 0, 'success': 0, 'predicted_sum': 0.0}
        
        self.history[score]['count'] += 1
        self.history[score]['predicted_sum'] += predicted_prob
        if actual_outcome:
            self.history[score]['success'] += 1
    
    def calibrate(self, score: str, predicted_prob: float) -> float:
        """
        校准预测概率
        
        参数：
            score: 比分
            predicted_prob: 原始预测概率
        
        返回：
            校准后的概率
        """
        if score not in self.history or self.history[score]['count'] < 20:
            # 数据不足时使用原始概率
            return predicted_prob
        
        record = self.history[score]
        total = record['count']
        success = record['success']
        avg_predicted = record['predicted_sum'] / total
        
        if avg_predicted < 0.001:
            return predicted_prob
        
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
    
    def calibrate_all(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        校准所有预测概率
        
        参数：
            predictions: {比分: 概率}
        
        返回：
            校准后的概率字典
        """
        calibrated = {}
        total_prob = 0.0
        
        for score, prob in predictions.items():
            calibrated[score] = self.calibrate(score, prob)
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

def calibrate_predictions(predictions: Dict[str, float]) -> Dict[str, float]:
    """校准预测概率的便捷接口"""
    return get_calibrator().calibrate_all(predictions)
