#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
总球数校准器
=============

功能：
1. 按联赛、大小球盘口、让球区间分桶统计进球数偏差
2. 赛后记录预测分布与实际结果
3. 提供校准后的进球数分布

分桶维度：
- 联赛 (league)
- 大小球盘口区间 (total_line bucketed by 0.25)
- 让球区间 (asian bucketed by 0.5)
- 模型预测总进球区间 (expected_total bucketed by 0.5)
"""

import os
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict


# ==================== 常量配置 ====================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
CALIBRATION_DB_FILE = os.path.join(DATA_DIR, 'goal_count_calibration.json')


class GoalCountCalibrator:
    """
    总球数校准器
    
    用于按分桶统计模型预测与实际结果的偏差，以便后续校正预测分布。
    """
    
    def __init__(self):
        self.db: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self):
        """从文件加载校准数据库"""
        if os.path.exists(CALIBRATION_DB_FILE):
            try:
                with open(CALIBRATION_DB_FILE, 'r', encoding='utf-8') as f:
                    self.db = json.load(f)
            except Exception as e:
                print(f"加载总球数校准数据库失败: {e}")
                self.db = {}
    
    def save(self):
        """保存校准数据库到文件"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CALIBRATION_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, ensure_ascii=False, indent=2)
        print(f"总球数校准数据库已保存，{len(self.db)} 个分桶")
    
    def _get_bucket_key(self, league: str, total_line: float, 
                        asian: float = 0.0, expected_total: float = 2.5) -> str:
        """
        生成分桶键
        
        分桶维度：
        - 联赛名
        - 大小球盘口（四舍五入到0.25）
        - 让球盘口（四舍五入到0.5，带符号）
        - 预测总进球（四舍五入到0.5）
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            asian: 亚盘让球
            expected_total: 模型预测总进球数
        
        返回：
            分桶键字符串
        """
        # 大小球盘口按0.25分桶
        bucketed_line = round(total_line * 4) / 4
        # 让球盘口按0.5分桶，带符号（+/-）
        bucketed_asian = round(asian * 2) / 2
        # 预测总进球按0.5分桶
        bucketed_expected = round(expected_total * 2) / 2
        
        return f"{league}_{bucketed_line:.2f}_{bucketed_asian:+.2f}_{bucketed_expected:.2f}"
    
    def record_result(self, league: str, total_line: float, 
                      predicted_goal_dist: Dict[int, float],
                      actual_total_goals: int,
                      expected_total_goals: float,
                      asian: float = 0.0):
        """
        记录赛后结果
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            predicted_goal_dist: 预测的进球数分布
            actual_total_goals: 实际总进球数
            expected_total_goals: 模型期望总进球数
            asian: 亚盘让球
        """
        bucket_key = self._get_bucket_key(league, total_line, asian, expected_total_goals)
        
        if bucket_key not in self.db:
            self.db[bucket_key] = {
                'league': league,
                'total_line': round(total_line * 4) / 4,
                'asian_bucket': round(asian * 2) / 2,
                'expected_total_bucket': round(expected_total_goals * 2) / 2,
                'sample_count': 0,
                'predicted_distributions': [],  # 存储历史预测分布用于分析
                'actual_goals': [],             # 存储实际进球数
                'calibration_factors': {},      # 预计算的校准因子
            }
        
        # 添加记录
        self.db[bucket_key]['sample_count'] += 1
        self.db[bucket_key]['predicted_distributions'].append(predicted_goal_dist)
        self.db[bucket_key]['actual_goals'].append(actual_total_goals)
        
        # 更新校准因子
        self._update_calibration_factors(bucket_key)
    
    def _update_calibration_factors(self, bucket_key: str):
        """
        更新分桶的校准因子
        
        计算每个进球数的实际频率与预测频率的比值，作为校准因子。
        """
        bucket = self.db[bucket_key]
        sample_count = bucket['sample_count']
        
        if sample_count < 10:
            # 样本不足，不计算校准因子
            bucket['calibration_factors'] = {}
            return
        
        # 计算实际频率分布
        actual_dist = defaultdict(int)
        for goals in bucket['actual_goals']:
            actual_dist[goals] += 1
        
        # 归一化
        actual_total = sum(actual_dist.values())
        actual_dist = {k: v / actual_total for k, v in actual_dist.items()}
        
        # 计算平均预测分布
        pred_dist = defaultdict(float)
        for dist in bucket['predicted_distributions']:
            for goals, prob in dist.items():
                pred_dist[goals] += prob
        
        # 归一化
        pred_total = sum(pred_dist.values())
        if pred_total > 0:
            pred_dist = {k: v / pred_total for k, v in pred_dist.items()}
        else:
            bucket['calibration_factors'] = {}
            return
        
        # 计算校准因子：实际频率 / 预测频率
        # 添加平滑处理，避免除零和极端值
        calibration_factors = {}
        all_goals = set(actual_dist.keys()) | set(pred_dist.keys())
        
        for goals in all_goals:
            actual_prob = actual_dist.get(goals, 0.01)  # 平滑
            pred_prob = pred_dist.get(goals, 0.01)      # 平滑
            
            # 计算校准因子，限制在合理范围
            factor = actual_prob / pred_prob
            factor = max(0.5, min(2.0, factor))  # 限制在0.5~2.0之间
            
            calibration_factors[goals] = factor
        
        bucket['calibration_factors'] = calibration_factors
    
    def get_calibration_factors(self, league: str, total_line: float,
                                expected_total: float, asian: float = 0.0) -> Dict[int, float]:
        """
        获取分桶的校准因子
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            expected_total: 模型预测总进球数
            asian: 亚盘让球
        
        返回：
            校准因子字典 {进球数: 校准因子}
        """
        bucket_key = self._get_bucket_key(league, total_line, asian, expected_total)
        
        if bucket_key in self.db:
            factors = self.db[bucket_key].get('calibration_factors', {})
            if factors and self.db[bucket_key].get('sample_count', 0) >= 10:
                return factors
        
        # 如果没有匹配的分桶或样本不足，返回空字典（不校准）
        return {}
    
    def calibrate_goal_dist(self, league: str, total_line: float,
                            goal_dist: Dict[int, float],
                            expected_total: float,
                            asian: float = 0.0,
                            min_samples: int = 10) -> Dict[int, float]:
        """
        校准进球数分布
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            goal_dist: 原始进球数分布
            expected_total: 模型预测总进球数
            asian: 亚盘让球
            min_samples: 最小样本数阈值
        
        返回：
            校准后的进球数分布
        """
        factors = self.get_calibration_factors(league, total_line, expected_total, asian)
        
        if not factors:
            # 没有校准因子，返回原始分布
            return goal_dist
        
        # 应用校准因子
        calibrated = {}
        all_goals = set(goal_dist.keys()) | set(factors.keys())
        
        for goals in all_goals:
            original_prob = goal_dist.get(goals, 0.0)
            factor = factors.get(goals, 1.0)
            calibrated[goals] = original_prob * factor
        
        # 归一化
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {k: v / total for k, v in sorted(calibrated.items())}
        
        return calibrated
    
    def get_bucket_stats(self, league: str = None) -> List[Dict]:
        """
        获取分桶统计信息
        
        参数：
            league: 可选，按联赛过滤
        
        返回：
            分桶统计列表
        """
        stats = []
        for bucket_key, bucket in self.db.items():
            if league and bucket.get('league') != league:
                continue
            
            stats.append({
                'bucket_key': bucket_key,
                'league': bucket.get('league'),
                'total_line': bucket.get('total_line'),
                'expected_total_bucket': bucket.get('expected_total_bucket'),
                'sample_count': bucket.get('sample_count', 0),
                'has_calibration': len(bucket.get('calibration_factors', {})) > 0,
            })
        
        # 按样本数排序
        stats.sort(key=lambda x: -x['sample_count'])
        return stats
    
    def analyze_bias(self, league: str = None) -> Dict[str, Any]:
        """
        分析校准偏差
        
        参数：
            league: 可选，按联赛过滤
        
        返回：
            偏差分析结果
        """
        results = []
        
        for bucket_key, bucket in self.db.items():
            if league and bucket.get('league') != league:
                continue
            
            sample_count = bucket.get('sample_count', 0)
            if sample_count < 10:
                continue
            
            # 计算实际平均进球数
            actual_avg = sum(bucket['actual_goals']) / sample_count
            
            # 计算预测平均进球数
            pred_avgs = []
            for dist in bucket['predicted_distributions']:
                pred_avg = sum(goals * prob for goals, prob in dist.items())
                pred_avgs.append(pred_avg)
            pred_avg = sum(pred_avgs) / len(pred_avgs) if pred_avgs else 0
            
            # 计算偏差
            bias = actual_avg - pred_avg
            
            results.append({
                'bucket_key': bucket_key,
                'league': bucket.get('league'),
                'total_line': bucket.get('total_line'),
                'expected_total_bucket': bucket.get('expected_total_bucket'),
                'sample_count': sample_count,
                'actual_avg': round(actual_avg, 2),
                'predicted_avg': round(pred_avg, 2),
                'bias': round(bias, 2),
                'calibration_factors': bucket.get('calibration_factors', {}),
            })
        
        # 按偏差绝对值排序
        results.sort(key=lambda x: abs(x['bias']), reverse=True)
        
        return {
            'total_buckets': len(results),
            'total_samples': sum(r['sample_count'] for r in results),
            'average_bias': sum(r['bias'] for r in results) / len(results) if results else 0,
            'buckets': results[:20],  # 返回前20个偏差最大的分桶
        }
    
    def clear(self):
        """清空数据库"""
        self.db = {}


# ==================== 便捷函数 ====================

def calibrate_goal_count_distribution(league: str, total_line: float,
                                      goal_dist: Dict[int, float],
                                      expected_total: float,
                                      asian: float = 0.0) -> Dict[int, float]:
    """
    便捷函数：校准进球数分布
    
    参数：
        league: 联赛名称
        total_line: 大小球盘口线
        goal_dist: 原始进球数分布
        expected_total: 模型预测总进球数
        asian: 亚盘让球
    
    返回：
        校准后的进球数分布
    """
    calibrator = GoalCountCalibrator()
    return calibrator.calibrate_goal_dist(league, total_line, goal_dist, 
                                          expected_total, asian)


def record_goal_count_result(league: str, total_line: float,
                             predicted_goal_dist: Dict[int, float],
                             actual_total_goals: int,
                             expected_total_goals: float,
                             asian: float = 0.0):
    """
    便捷函数：记录赛后结果
    
    参数：
        league: 联赛名称
        total_line: 大小球盘口线
        predicted_goal_dist: 预测的进球数分布
        actual_total_goals: 实际总进球数
        expected_total_goals: 模型期望总进球数
        asian: 亚盘让球
    """
    calibrator = GoalCountCalibrator()
    calibrator.record_result(league, total_line, predicted_goal_dist,
                            actual_total_goals, expected_total_goals, asian)
    calibrator.save()


# ==================== 命令行工具 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='总球数校准器')
    parser.add_argument('--stats', action='store_true', help='查看分桶统计')
    parser.add_argument('--analyze', action='store_true', help='分析偏差')
    parser.add_argument('--clear', action='store_true', help='清空数据库')
    
    args = parser.parse_args()
    
    calibrator = GoalCountCalibrator()
    
    if args.stats:
        stats = calibrator.get_bucket_stats()
        print("="*60)
        print("总球数校准分桶统计")
        print("="*60)
        for stat in stats[:20]:
            print(f"{stat['bucket_key']}: {stat['sample_count']}场 "
                  f"(校准: {'有' if stat['has_calibration'] else '无'})")
        
    elif args.analyze:
        analysis = calibrator.analyze_bias()
        print("="*60)
        print("总球数校准偏差分析")
        print("="*60)
        print(f"总分桶数: {analysis['total_buckets']}")
        print(f"总样本数: {analysis['total_samples']}")
        print(f"平均偏差: {analysis['average_bias']:.2f}球")
        print("\n偏差最大的分桶:")
        for bucket in analysis['buckets'][:10]:
            print(f"\n{bucket['bucket_key']}:")
            print(f"  样本数: {bucket['sample_count']}")
            print(f"  实际平均: {bucket['actual_avg']:.2f}球")
            print(f"  预测平均: {bucket['predicted_avg']:.2f}球")
            print(f"  偏差: {bucket['bias']:+.2f}球")
    
    elif args.clear:
        confirm = input("确定要清空校准数据库吗? (y/n): ")
        if confirm.lower() == 'y':
            calibrator.clear()
            calibrator.save()
            print("数据库已清空")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()