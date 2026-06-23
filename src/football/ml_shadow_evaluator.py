#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML影子预测评估器
==================

功能：
1. 记录ML模型的影子预测
2. 评估影子预测与实际结果的对比
3. 生成评估报告
4. 判断是否具备融合条件
"""

import os
import json
import csv
from typing import Dict, List, Any
from datetime import datetime

from ..common import kv_store


# ==================== 常量配置 ====================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
SHADOW_RECORDS_FILE = os.path.join(DATA_DIR, 'ml_shadow_records.json')


# ==================== 评估指标计算 ====================

def calculate_hit_rate(predictions: List[Dict]) -> float:
    """
    计算命中率
    
    参数：
        predictions: 预测记录列表
    
    返回：
        命中率
    """
    if not predictions:
        return 0.0
    
    correct = sum(1 for p in predictions if p.get('hit', False))
    return correct / len(predictions)


def calculate_logloss(predictions: List[Dict]) -> float:
    """
    计算LogLoss
    
    参数：
        predictions: 预测记录列表
    
    返回：
        LogLoss值
    """
    if not predictions:
        return float('inf')
    
    eps = 1e-15
    logloss_sum = 0.0
    count = 0
    
    for pred in predictions:
        actual = pred.get('actual_result')
        probs = pred.get('prediction', {})
        
        if actual and actual in probs:
            prob = max(eps, min(1 - eps, probs[actual]))
            logloss_sum += -float(prob).log()
            count += 1
    
    return logloss_sum / count if count > 0 else float('inf')


def calculate_brier(predictions: List[Dict]) -> float:
    """
    计算Brier分数
    
    参数：
        predictions: 预测记录列表
    
    返回：
        Brier分数
    """
    if not predictions:
        return float('inf')
    
    brier_sum = 0.0
    count = 0
    
    for pred in predictions:
        actual = pred.get('actual_result')
        probs = pred.get('prediction', {})
        
        if actual:
            # 计算Brier分数：(p_H - actual_H)^2 + (p_D - actual_D)^2 + (p_A - actual_A)^2
            actual_one_hot = {'H': 0, 'D': 0, 'A': 0}
            actual_one_hot[actual] = 1
            
            brier = 0.0
            for label in ['H', 'D', 'A']:
                p = probs.get(label, 0.0)
                brier += (p - actual_one_hot[label]) ** 2
            
            brier_sum += brier
            count += 1
    
    return brier_sum / count if count > 0 else float('inf')


# ==================== 影子评估器 ====================

class ShadowEvaluator:
    """ML影子预测评估器"""
    
    def __init__(self):
        self.records = self._load_records()
    
    def _load_records(self) -> List[Dict]:
        """从 MySQL 加载影子预测记录"""
        try:
            return kv_store.load('ml_shadow_records') or []
        except Exception as e:
            print(f"加载影子记录失败: {e}")
            return []

    def _save_records(self):
        """保存影子预测记录到 MySQL"""
        kv_store.save('ml_shadow_records', self.records)
    
    def record_prediction(self, match_id: str, match_date: str, league: str,
                         prediction: Dict, actual_result: str = None):
        """
        记录影子预测
        
        参数：
            match_id: 比赛ID
            match_date: 比赛日期
            league: 联赛
            prediction: ML预测结果
            actual_result: 实际结果（赛后补充）
        """
        # 检查是否已存在
        existing_idx = None
        for i, record in enumerate(self.records):
            if record['match_id'] == match_id:
                existing_idx = i
                break
        
        record = {
            'match_id': match_id,
            'match_date': match_date,
            'league': league,
            'prediction': prediction,
            'predicted_at': datetime.now().isoformat(),
            'actual_result': actual_result,
            'hit': None,
            'evaluated_at': None,
        }
        
        if actual_result:
            record['hit'] = (prediction.get('predicted_label') == actual_result)
            record['evaluated_at'] = datetime.now().isoformat()
        
        if existing_idx is not None:
            self.records[existing_idx] = record
        else:
            self.records.append(record)
        
        self._save_records()
    
    def update_result(self, match_id: str, actual_result: str):
        """
        更新比赛结果
        
        参数：
            match_id: 比赛ID
            actual_result: 实际结果
        """
        for record in self.records:
            if record['match_id'] == match_id and not record['actual_result']:
                record['actual_result'] = actual_result
                record['hit'] = (record['prediction'].get('predicted_label') == actual_result)
                record['evaluated_at'] = datetime.now().isoformat()
        
        self._save_records()
    
    def get_recent_records(self, days: int = 30) -> List[Dict]:
        """
        获取最近N天的记录
        
        参数：
            days: 天数
        
        返回：
            最近N天的记录列表
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return [r for r in self.records if r['match_date'] >= cutoff_date]
    
    def evaluate(self, records: List[Dict] = None) -> Dict:
        """
        评估影子预测性能
        
        参数：
            records: 要评估的记录列表，默认为所有已评估的记录
        
        返回：
            评估指标字典
        """
        if records is None:
            records = [r for r in self.records if r['actual_result']]
        
        if not records:
            return {
                'sample_count': 0,
                'hit_rate': 0.0,
                'logloss': float('inf'),
                'brier': float('inf'),
                'breakdown': {}
            }
        
        # 计算整体指标
        hit_rate = calculate_hit_rate(records)
        logloss = calculate_logloss(records)
        brier = calculate_brier(records)
        
        # 按联赛拆分
        league_records = {}
        for record in records:
            league = record['league']
            if league not in league_records:
                league_records[league] = []
            league_records[league].append(record)
        
        breakdown = {}
        for league, league_data in league_records.items():
            breakdown[league] = {
                'sample_count': len(league_data),
                'hit_rate': calculate_hit_rate(league_data),
                'logloss': calculate_logloss(league_data),
                'brier': calculate_brier(league_data),
            }
        
        return {
            'sample_count': len(records),
            'hit_rate': hit_rate,
            'logloss': logloss,
            'brier': brier,
            'breakdown': breakdown,
            'evaluated_at': datetime.now().isoformat()
        }
    
    def should_integrate(self, min_samples: int = 100, min_hit_rate: float = 0.52) -> bool:
        """
        判断是否具备融合条件
        
        参数：
            min_samples: 最小样本数
            min_hit_rate: 最低命中率
        
        返回：
            是否具备融合条件
        """
        evaluated = [r for r in self.records if r['actual_result']]
        
        if len(evaluated) < min_samples:
            print(f"样本数不足: {len(evaluated)}/{min_samples}")
            return False
        
        hit_rate = calculate_hit_rate(evaluated)
        if hit_rate < min_hit_rate:
            print(f"命中率不足: {hit_rate:.4f}/{min_hit_rate}")
            return False
        
        print(f"具备融合条件: 样本数={len(evaluated)}, 命中率={hit_rate:.4f}")
        return True
    
    def generate_report(self) -> str:
        """
        生成评估报告
        
        返回：
            报告文本
        """
        report = []
        report.append("=" * 60)
        report.append("ML影子预测评估报告")
        report.append("=" * 60)
        
        # 基本统计
        total = len(self.records)
        evaluated = len([r for r in self.records if r['actual_result']])
        pending = total - evaluated
        
        report.append(f"\n总预测数: {total}")
        report.append(f"已评估: {evaluated}")
        report.append(f"待评估: {pending}")
        
        # 评估指标
        metrics = self.evaluate()
        report.append("\n评估指标:")
        report.append(f"  样本数: {metrics['sample_count']}")
        report.append(f"  命中率: {metrics['hit_rate']:.4f}")
        report.append(f"  LogLoss: {metrics['logloss']:.4f}")
        report.append(f"  Brier: {metrics['brier']:.4f}")
        
        # 联赛细分
        report.append("\n联赛细分:")
        for league, data in metrics['breakdown'].items():
            report.append(f"  {league}:")
            report.append(f"    样本数: {data['sample_count']}")
            report.append(f"    命中率: {data['hit_rate']:.4f}")
            report.append(f"    LogLoss: {data['logloss']:.4f}")
            report.append(f"    Brier: {data['brier']:.4f}")
        
        # 融合建议
        report.append("\n融合建议:")
        if self.should_integrate():
            report.append("  ✓ 建议将ML模型以低权重参与融合")
        else:
            report.append("  ✗ 暂不建议融合，继续积累影子数据")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def export_to_csv(self, filepath: str = None):
        """
        导出记录到CSV
        
        参数：
            filepath: 输出文件路径
        """
        if filepath is None:
            filepath = os.path.join(DATA_DIR, 'ml_shadow_records.csv')
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # 表头
            writer.writerow([
                'match_id', 'match_date', 'league',
                'pred_H', 'pred_D', 'pred_A', 'predicted_label',
                'actual_result', 'hit', 'predicted_at', 'evaluated_at'
            ])
            
            # 数据
            for record in self.records:
                pred = record.get('prediction', {})
                writer.writerow([
                    record['match_id'],
                    record['match_date'],
                    record['league'],
                    pred.get('H', ''),
                    pred.get('D', ''),
                    pred.get('A', ''),
                    pred.get('predicted_label', ''),
                    record.get('actual_result', ''),
                    record.get('hit', ''),
                    record.get('predicted_at', ''),
                    record.get('evaluated_at', ''),
                ])
        
        print(f"影子记录已导出到: {filepath}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    evaluator = ShadowEvaluator()
    report = evaluator.generate_report()
    print(report)
    
    # 导出CSV
    evaluator.export_to_csv()


if __name__ == '__main__':
    from datetime import timedelta
    main()