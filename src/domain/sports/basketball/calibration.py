#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
篮球预测贝叶斯校准器
====================
对三类预测（SPF/RQSPF/DX）分别追踪历史准确率，进行贝叶斯校准。

设计思路：
1. 分桶策略：
   - 层级1: 联赛 + bet类型 + 置信度区间 (≥50样本)
   - 层级2: bet类型 + 置信度区间 (≥100样本)
   - 层级3: 全局 (≥200样本)
2. 校准公式: calibrated = predicted × (actual_rate / predicted_rate)
3. 加权融合: 历史样本越多，校准权重越大

与 football/bayesian_calibration.py 同架构，适配篮球预测场景。
"""

import logging
from typing import Dict, Optional
from collections import defaultdict

from .calibration_store import CalibrationStore

logger = logging.getLogger(__name__)

# 数据落在 foundation/store 的 bb_calibration 表；kv_store 中的同名 key 从未有过数据。
BB_CALIBRATION_KEY = 'basketball_calibration_db'

# 置信度分桶（用于校准）
CONFIDENCE_BUCKETS = ['high', 'medium', 'low']


def _confidence_bucket(confidence: str) -> str:
    """标准化置信度分桶"""
    if confidence in CONFIDENCE_BUCKETS:
        return confidence
    return 'medium'


class BasketballCalibrator:
    """
    篮球预测贝叶斯校准器

    维护三类预测各自的历史命中率统计：
    - spf (胜负)
    - rqspf (让分胜负)
    - dx (大小分)
    """

    def __init__(self, store=None):
        """store 可注入，便于测试用 SQLite 内存库；生产传 CalibrationStore(db)。"""
        self.stats: Dict[str, dict] = {}  # {bucket_key: {count, success, predicted_sum}}
        self._store = store
        self._load()

    def _load(self):
        """从 foundation/store 加载校准数据"""
        try:
            if self._store is None:
                self.stats = {}
                return
            data = self._store.load()
            if isinstance(data, dict):
                self.stats = data
            logger.info(f"篮球校准器已加载 {len(self.stats)} 个分桶")
        except Exception as e:
            logger.error(f"加载篮球校准数据失败: {e}")
            self.stats = {}

    def save(self):
        """保存校准数据"""
        if self._store is None:
            return
        try:
            self._store.save(self.stats)
        except Exception as e:
            logger.error(f"保存篮球校准数据失败: {e}")

    def _bucket_key(self, bet_type: str, league: str, confidence: str, level: int) -> str:
        """
        生成分桶键

        level 1: bet_type + league + confidence (最细)
        level 2: bet_type + confidence
        level 3: bet_type only (最粗)
        """
        if level == 1:
            return f"{bet_type}|{league}|{confidence}"
        elif level == 2:
            return f"{bet_type}|*|{confidence}"
        else:
            return f"{bet_type}|*|*"

    def record(self, bet_type: str, predicted_prob: float, actual_hit: bool,
               league: str = '', confidence: str = 'medium', weight: float = 1.0):
        """
        记录一条预测结果

        参数:
            bet_type: 'spf' | 'rqspf' | 'dx'
            predicted_prob: 预测概率 (0~1)
            actual_hit: 实际是否命中
            league: 联赛
            confidence: 置信度
            weight: 样本权重（默认1.0）
        """
        if bet_type not in ('spf', 'rqspf', 'dx'):
            return

        try:
            weight = max(0.0, min(1.0, float(weight)))
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0:
            return

        confidence = _confidence_bucket(confidence)

        for level in [1, 2, 3]:
            key = self._bucket_key(bet_type, league, confidence, level)
            if key not in self.stats:
                self.stats[key] = {
                    'count': 0,
                    'weighted_count': 0.0,
                    'success': 0,
                    'weighted_success': 0.0,
                    'predicted_sum': 0.0,
                    'weighted_predicted_sum': 0.0,
                }

            s = self.stats[key]
            s['count'] += 1
            s['predicted_sum'] += predicted_prob
            s['weighted_count'] = s.get('weighted_count', s['count'] - 1) + weight
            s['weighted_predicted_sum'] = s.get('weighted_predicted_sum',
                                                 s['predicted_sum'] - predicted_prob) + predicted_prob * weight
            if actual_hit:
                s['success'] += 1
                s['weighted_success'] = s.get('weighted_success',
                                              s['success'] - 1) + weight

        self.save()

    def calibrate(self, bet_type: str, predicted_prob: float,
                  league: str = '', confidence: str = 'medium') -> float:
        """
        校准预测概率（层级回退）

        参数:
            bet_type: 'spf' | 'rqspf' | 'dx'
            predicted_prob: 原始预测概率
            league: 联赛
            confidence: 置信度

        返回:
            校准后的概率
        """
        confidence = _confidence_bucket(confidence)

        level_requirements = [
            (1, 50),   # 层级1需要50个样本
            (2, 100),  # 层级2需要100个样本
            (3, 200),  # 层级3需要200个样本
        ]

        for level, min_samples in level_requirements:
            key = self._bucket_key(bet_type, league, confidence, level)
            if key in self.stats:
                s = self.stats[key]
                total = s.get('weighted_count', s['count'])
                if total >= min_samples:
                    success = s.get('weighted_success', s['success'])
                    predicted_sum = s.get('weighted_predicted_sum', s['predicted_sum'])
                    avg_predicted = predicted_sum / total

                    if avg_predicted < 0.001:
                        continue

                    actual_rate = success / total
                    correction_factor = actual_rate / avg_predicted

                    calibrated = predicted_prob * correction_factor
                    calibrated = max(0.001, min(0.999, calibrated))

                    # 加权融合：样本越多，校准权重越大
                    weight = min(total / 1000, 1.0)
                    final = (1 - weight) * predicted_prob + weight * calibrated

                    return round(final, 4)

        # 没有足够的样本，返回原始概率
        return predicted_prob

    def get_stats(self, bet_type: str = None, league: str = None) -> Dict:
        """
        获取校准统计信息

        参数:
            bet_type: None 返回所有，否则指定类型
            league: None 返回所有，否则指定联赛
        """
        result = {}
        for key, s in self.stats.items():
            parts = key.split('|')
            bt = parts[0] if len(parts) > 0 else ''
            lg = parts[1] if len(parts) > 1 else ''
            cf = parts[2] if len(parts) > 2 else ''

            if bet_type and bt != bet_type:
                continue
            if league and lg != league and lg != '*':
                continue

            total = s.get('weighted_count', s['count'])
            if total > 0:
                success = s.get('weighted_success', s['success'])
                predicted_sum = s.get('weighted_predicted_sum', s['predicted_sum'])
                result[key] = {
                    'total_samples': int(total),
                    'hit_rate': round(success / total, 4),
                    'avg_predicted': round(predicted_sum / total, 4),
                    'bias': round(success / total - predicted_sum / total, 4),
                }
        return result

    def clear_stats(self):
        """清空所有校准数据"""
        self.stats = {}
        self.save()
        logger.info("篮球校准数据已清空")


# ==================== 全局实例 ====================

_calibrator: Optional[BasketballCalibrator] = None


def get_calibrator() -> BasketballCalibrator:
    """获取全局校准器实例"""
    global _calibrator
    if _calibrator is None:
        _calibrator = BasketballCalibrator()
    return _calibrator


def reload_calibrator():
    """重新加载校准器"""
    global _calibrator
    _calibrator = BasketballCalibrator()
    return _calibrator
