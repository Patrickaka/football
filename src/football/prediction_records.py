#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测记录存储模块（支持模型版本冻结）
====================================

功能：
1. 保存每场比赛的预测结果和实际结果
2. 支持按联赛、球队、模型版本查询历史记录
3. 提供校准所需的历史数据
4. 支持模型版本冻结，便于回测对比

数据格式：
{
    "match_id": "...",
    "league": "英超",
    "home": "...",
    "away": "...",
    "predicted_scores": {"1-1": 0.108, "2-1": 0.091},
    "predicted_1x2": {"home": 0.46, "draw": 0.27, "away": 0.27},
    "actual_score": "2-1",
    "actual_result": "H",
    "asian": -0.5,
    "total_line": 2.5,
    "model_version": "v2026.06.22-03",
    "params_snapshot": {
        "market_weight": 0.6,
        "rho": 0.3,
        "prior_weight": 0.2,
        "half_time_ratio": 0.45,
        "goal_calibration_enabled": true
    },
    "created_at": "..."
}
"""

import os
import json
import math
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from ..common import repositories

# **与 result_sync / tune_params 那两份逐字相同**（F-1 已 AST 确认）
from ..domain.sports.football.settlement import normalize_1x2_probs




DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RECORD_FILE = os.path.join(DATA_DIR, 'prediction_records.json')


class PredictionRecords:
    """预测记录管理器"""
    
    def __init__(self):
        self.records: List[Dict] = []
        self._load()
    
    def _load(self):
        """从 MySQL 加载记录"""
        try:
            self.records = repositories.prediction_record_load()
        except Exception as e:
            print(f"加载预测记录失败: {e}")
            self.records = []

    def _save(self):
        """保存记录到 MySQL"""
        repositories.prediction_record_save(self.records)
    
    def add_record(self, match_id: str, league: str, home: str, away: str,
                   predicted_scores: Dict[str, float], predicted_1x2: Dict[str, float],
                   asian: float = None, total_line: float = None,
                   model_version: str = None, params_snapshot: Dict = None):
        """
        添加预测记录（比赛前保存预测）
        
        参数：
            match_id: 比赛ID
            league: 联赛名称
            home: 主队名称
            away: 客队名称
            predicted_scores: 预测的比分概率 {"1-1": 0.108, ...}
            predicted_1x2: 预测的胜平负概率 {"home": 0.46, "draw": 0.27, "away": 0.27}
            asian: 亚盘让球
            total_line: 大小球盘口
            model_version: 模型版本号（如 "v2026.06.22-03"）
            params_snapshot: 参数快照字典
        """
        # 检查是否已存在相同比赛的未完成记录
        for record in self.records:
            if record['match_id'] == match_id and 'actual_score' not in record:
                record.update({
                    'predicted_scores': predicted_scores,
                    'predicted_1x2': predicted_1x2,
                    'asian': asian,
                    'total_line': total_line,
                    'model_version': model_version,
                    'params_snapshot': params_snapshot,
                    'updated_at': datetime.now().isoformat()
                })
                self._save()
                return
        
        # 添加新记录
        self.records.append({
            'match_id': match_id,
            'league': league,
            'home': home,
            'away': away,
            'predicted_scores': predicted_scores,
            'predicted_1x2': predicted_1x2,
            'asian': asian,
            'total_line': total_line,
            'model_version': model_version,
            'params_snapshot': params_snapshot,
            'created_at': datetime.now().isoformat()
        })
        self._save()
    
    def update_result(self, match_id: str, actual_score: str, actual_result: str):
        """
        更新比赛结果（比赛后添加实际结果）
        
        参数：
            match_id: 比赛ID
            actual_score: 实际比分，如 "2-1"
            actual_result: 实际结果，"H"（主胜）、"D"（平局）、"A"（客胜）
        """
        for record in self.records:
            if record['match_id'] == match_id:
                record['actual_score'] = actual_score
                record['actual_result'] = actual_result
                record['result_updated_at'] = datetime.now().isoformat()
                self._save()
                return
    
    def get_records_by_league(self, league: str, limit: int = None) -> List[Dict]:
        """获取指定联赛的记录"""
        records = [r for r in self.records 
                   if r.get('league') == league and 'actual_score' in r]
        if limit:
            records = records[-limit:]
        return records
    
    def get_all_completed_records(self, limit: int = None) -> List[Dict]:
        """获取所有已完成的记录（有实际结果的）"""
        records = [r for r in self.records if 'actual_score' in r]
        if limit:
            records = records[-limit:]
        return records
    
    def get_record_by_match_id(self, match_id: str) -> Optional[Dict]:
        """根据比赛ID获取记录"""
        for record in self.records:
            if record['match_id'] == match_id:
                return record
        return None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = len(self.records)
        completed = len([r for r in self.records if 'actual_score' in r])
        by_league = defaultdict(int)
        by_version = defaultdict(int)
        for r in self.records:
            if 'actual_score' in r:
                by_league[r.get('league', '未知')] += 1
                by_version[r.get('model_version', '未知')] += 1
        
        return {
            'total_records': total,
            'completed_records': completed,
            'by_league': dict(by_league),
            'by_version': dict(by_version)
        }
    
    def get_records_by_version(self, model_version: str) -> List[Dict]:
        """获取指定模型版本的记录"""
        return [r for r in self.records 
                if r.get('model_version') == model_version and 'actual_score' in r]
    
    def get_all_versions(self) -> List[str]:
        """获取所有已记录的模型版本"""
        versions = set()
        for r in self.records:
            version = r.get('model_version')
            if version:
                versions.add(version)
        return sorted(list(versions))
    
    def calculate_version_metrics(self, model_version: str = None) -> Dict:
        """
        计算指定模型版本的预测指标
        
        参数：
            model_version: 模型版本号（None表示全部版本）
        
        返回：
            指标字典包含：
            - score_top3_accuracy: 比分Top3命中率
            - goals_top2_accuracy: 总进球Top2命中率
            - htf_top3_accuracy: 半全场Top3命中率
            - result_accuracy: 胜平负命中率
            - total_samples: 样本数
        """
        if model_version:
            records = self.get_records_by_version(model_version)
        else:
            records = self.get_all_completed_records()
        
        if not records:
            return {
                'model_version': model_version or 'all',
                'total_samples': 0,
                'score_top3_accuracy': 0.0,
                'goals_top2_accuracy': 0.0,
                'result_accuracy': 0.0,
                'avg_brier': 0.0,
                'avg_logloss': 0.0
            }
        
        score_top3_hit = 0
        goals_top2_hit = 0
        result_hit = 0
        total_brier = 0.0
        total_logloss = 0.0
        
        for record in records:
            # 比分Top3命中
            predicted_scores = record.get('predicted_scores', {})
            if predicted_scores:
                top3_scores = [k for k, v in sorted(predicted_scores.items(), 
                                                    key=lambda x: -x[1])[:3]]
                actual_score = record.get('actual_score', '')
                if actual_score in top3_scores:
                    score_top3_hit += 1
            
            # 总进球Top2命中
            if predicted_scores:
                goals_dist = defaultdict(float)
                for score, prob in predicted_scores.items():
                    goals = sum(int(g) for g in score.split('-'))
                    goals_dist[goals] += prob
                top2_goals = [k for k, v in sorted(goals_dist.items(), 
                                                   key=lambda x: -x[1])[:2]]
                actual_goals = sum(int(g) for g in record.get('actual_score', '0-0').split('-'))
                if actual_goals in top2_goals:
                    goals_top2_hit += 1
            
            # 胜平负命中
            predicted_1x2 = normalize_1x2_probs(record.get('predicted_1x2', {}))
            if predicted_1x2:
                pred_result = max(predicted_1x2.keys(), key=lambda k: predicted_1x2[k])
                actual_result = record.get('actual_result', '')
                if pred_result == actual_result:
                    result_hit += 1
            
            # Brier分数
            if predicted_1x2 and record.get('actual_result'):
                actual = record['actual_result']
                brier = 0.0
                for key in ['H', 'D', 'A']:
                    pred = predicted_1x2.get(key, 0.0)
                    actual_val = 1.0 if key == actual else 0.0
                    brier += (pred - actual_val) ** 2
                total_brier += brier
                
                # LogLoss
                eps = 1e-15
                pred_prob = max(eps, min(1-eps, predicted_1x2.get(actual, eps)))
                total_logloss += -math.log(pred_prob)
        
        n = len(records)
        return {
            'model_version': model_version or 'all',
            'total_samples': n,
            'score_top3_accuracy': round(score_top3_hit / n, 4),
            'goals_top2_accuracy': round(goals_top2_hit / n, 4),
            'result_accuracy': round(result_hit / n, 4),
            'avg_brier': round(total_brier / n, 4),
            'avg_logloss': round(total_logloss / n, 4)
        }
    
    def compare_versions(self, versions: List[str] = None) -> List[Dict]:
        """
        对比多个模型版本的指标
        
        参数：
            versions: 版本号列表（None表示对比所有版本）
        
        返回：
            各版本指标对比列表
        """
        if versions is None:
            versions = self.get_all_versions()
        
        results = []
        for version in versions:
            metrics = self.calculate_version_metrics(version)
            results.append(metrics)
        
        # 按版本号排序
        results.sort(key=lambda x: x['model_version'])
        return results
    
    def clear_old_records(self, days: int = 90):
        """清除指定天数之前的记录"""
        cutoff = datetime.now() - datetime.timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        self.records = [r for r in self.records 
                        if r.get('created_at', '') > cutoff_str]
        self._save()


# 全局实例
_global_records = PredictionRecords()


# ==================== 便捷函数 ====================

def add_prediction(match_id: str, league: str, home: str, away: str,
                   predicted_scores: Dict[str, float], predicted_1x2: Dict[str, float],
                   asian: float = None, total_line: float = None):
    """添加预测记录"""
    return _global_records.add_record(match_id, league, home, away,
                                      predicted_scores, predicted_1x2,
                                      asian, total_line)


def update_prediction_result(match_id: str, actual_score: str, actual_result: str):
    """更新预测结果"""
    return _global_records.update_result(match_id, actual_score, actual_result)


def get_historical_data(league: str = None, limit: int = 100) -> List[Dict]:
    """
    获取历史预测数据（用于校准）
    
    参数：
        league: 联赛名称，如果为None则获取所有联赛
        limit: 返回的记录数量
    
    返回：
        包含预测和实际结果的记录列表
    """
    if league:
        return _global_records.get_records_by_league(league, limit)
    return _global_records.get_all_completed_records(limit)


def get_prediction_stats() -> Dict:
    """获取预测统计信息"""
    return _global_records.get_stats()


def record_prediction_result(predictions: Dict[str, float], actual_score: str):
    """
    记录预测结果（简化接口，用于贝叶斯校准）
    
    参数：
        predictions: 预测的比分概率 {"1-1": 0.108, ...}
        actual_score: 实际比分 "2-1"
    """
    from .bayesian_calibration import get_calibrator
    
    calibrator = get_calibrator()
    for score, prob in predictions.items():
        calibrator.add_record(score, prob, score == actual_score)
    calibrator.save()


# ==================== 测试 ====================

def main():
    print("=== 预测记录模块测试 ===")
    
    # 添加测试记录
    add_prediction(
        match_id='test_001',
        league='英超',
        home='曼联',
        away='利物浦',
        predicted_scores={'1-1': 0.25, '2-1': 0.20, '1-0': 0.15},
        predicted_1x2={'home': 0.40, 'draw': 0.30, 'away': 0.30},
        asian=-0.5,
        total_line=2.5
    )
    
    # 更新结果
    update_prediction_result('test_001', '2-1', 'H')
    
    # 获取统计
    stats = get_prediction_stats()
    print(f"统计信息: {stats}")
    
    # 获取历史数据
    data = get_historical_data('英超', limit=10)
    print(f"英超历史记录数: {len(data)}")


if __name__ == '__main__':
    main()