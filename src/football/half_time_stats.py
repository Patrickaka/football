#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
半场比分统计数据库
====================

功能：
1. 按分桶统计半场比分数据
2. 记录真实半场比分（非倒推）
3. 提供半场进球比例、平局率等统计指标

分桶维度：
- 联赛
- 大小球盘口区间
- 让球深度区间
- 比赛类型（联赛/杯赛/友谊赛）
"""

import os
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict


# ==================== 常量配置 ====================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
HALF_TIME_DB_FILE = os.path.join(DATA_DIR, 'half_time_stats.json')


class HalfTimeStatsDB:
    """
    半场比分统计数据库
    
    用于按分桶统计半场比分特征，支持后续优化半场模型。
    """
    
    def __init__(self):
        self.db: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self):
        """从文件加载数据库"""
        if os.path.exists(HALF_TIME_DB_FILE):
            try:
                with open(HALF_TIME_DB_FILE, 'r', encoding='utf-8') as f:
                    self.db = json.load(f)
            except Exception as e:
                print(f"加载半场统计数据库失败: {e}")
                self.db = {}
    
    def save(self):
        """保存数据库到文件"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HALF_TIME_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, ensure_ascii=False, indent=2)
        print(f"半场统计数据库已保存，{len(self.db)} 个分桶")
    
    def _get_bucket_key(self, league: str, total_line: float, 
                        handicap: float = 0.0, match_type: str = 'league') -> str:
        """
        生成分桶键
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            handicap: 亚盘让球
            match_type: 比赛类型 (league/cup/friendly)
        
        返回：
            分桶键字符串
        """
        # 大小球盘口按0.25分桶
        bucketed_line = round(total_line * 4) / 4
        # 让球按0.5分桶
        bucketed_handicap = round(handicap * 2) / 2
        
        return f"{league}_{match_type}_{bucketed_line:.2f}_{bucketed_handicap:.2f}"
    
    def record_match(self, league: str, total_line: float, handicap: float,
                     match_type: str, half_home: int, half_away: int,
                     full_home: int, full_away: int):
        """
        记录一场比赛的半场和全场比分
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            handicap: 亚盘让球
            match_type: 比赛类型 (league/cup/friendly)
            half_home: 半场主队进球
            half_away: 半场客队进球
            full_home: 全场主队进球
            full_away: 全场客队进球
        """
        bucket_key = self._get_bucket_key(league, total_line, handicap, match_type)
        
        if bucket_key not in self.db:
            self.db[bucket_key] = {
                'league': league,
                'match_type': match_type,
                'total_line': round(total_line * 4) / 4,
                'handicap': round(handicap * 2) / 2,
                'sample_count': 0,
                'half_goals': [],        # 半场总进球数列表
                'half_home_goals': [],   # 半场主队进球数列表
                'half_away_goals': [],   # 半场客队进球数列表
                'full_goals': [],        # 全场总进球数列表
                'half_results': [],      # 半场结果: H/D/A
                'full_results': [],      # 全场结果: H/D/A
                'half_full_results': [], # 半全场结果: HH/HD/HA/DH/DD/DA/AH/AD/AA
                # 预计算统计
                'stats': {},
            }
        
        bucket = self.db[bucket_key]
        bucket['sample_count'] += 1
        
        # 记录进球数
        bucket['half_goals'].append(half_home + half_away)
        bucket['half_home_goals'].append(half_home)
        bucket['half_away_goals'].append(half_away)
        bucket['full_goals'].append(full_home + full_away)
        
        # 记录结果
        half_res = 'H' if half_home > half_away else 'A' if half_home < half_away else 'D'
        full_res = 'H' if full_home > full_away else 'A' if full_home < full_away else 'D'
        
        bucket['half_results'].append(half_res)
        bucket['full_results'].append(full_res)
        bucket['half_full_results'].append(f"{half_res}{full_res}")
        
        # 更新预计算统计
        self._update_stats(bucket_key)
    
    def _update_stats(self, bucket_key: str):
        """更新分桶的统计数据"""
        bucket = self.db[bucket_key]
        sample_count = bucket['sample_count']
        
        if sample_count < 1:
            bucket['stats'] = {}
            return
        
        stats = {}
        
        # 半场进球统计
        half_goals = bucket['half_goals']
        stats['first_half_goals_avg'] = round(sum(half_goals) / sample_count, 3)
        stats['first_half_goals_std'] = round(
            (sum((g - stats['first_half_goals_avg'])**2 for g in half_goals) / sample_count) ** 0.5,
            3
        )
        
        # 全场进球统计
        full_goals = bucket['full_goals']
        stats['full_goals_avg'] = round(sum(full_goals) / sample_count, 3)
        
        # 半场比例
        if stats['full_goals_avg'] > 0:
            stats['half_time_ratio_avg'] = round(stats['first_half_goals_avg'] / stats['full_goals_avg'], 3)
        else:
            stats['half_time_ratio_avg'] = 0.42
        
        # 半场结果分布
        half_results = bucket['half_results']
        stats['first_half_draw_rate'] = round(half_results.count('D') / sample_count, 3)
        stats['home_lead_at_half_rate'] = round(half_results.count('H') / sample_count, 3)
        stats['away_lead_at_half_rate'] = round(half_results.count('A') / sample_count, 3)
        
        # 半全场结果分布
        htf_results = bucket['half_full_results']
        htf_dist = {}
        for res in ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']:
            htf_dist[res] = round(htf_results.count(res) / sample_count, 3)
        stats['half_full_distribution'] = htf_dist
        
        # 半场主客进球比例
        half_home_goals = bucket['half_home_goals']
        half_away_goals = bucket['half_away_goals']
        total_half_goals = sum(half_home_goals) + sum(half_away_goals)
        if total_half_goals > 0:
            stats['half_home_goal_ratio'] = round(sum(half_home_goals) / total_half_goals, 3)
        else:
            stats['half_home_goal_ratio'] = 0.5
        
        bucket['stats'] = stats
    
    def get_stats(self, league: str, total_line: float, handicap: float = 0.0, 
                  match_type: str = 'league', min_samples: int = 20) -> Optional[Dict]:
        """
        获取分桶的统计数据
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            handicap: 亚盘让球
            match_type: 比赛类型
            min_samples: 最小样本数阈值
        
        返回：
            统计数据字典，如果样本不足返回None
        """
        bucket_key = self._get_bucket_key(league, total_line, handicap, match_type)
        
        if bucket_key in self.db:
            bucket = self.db[bucket_key]
            if bucket.get('sample_count', 0) >= min_samples:
                return bucket.get('stats', {})
        
        return None
    
    def get_bucket_list(self, league: str = None) -> List[Dict]:
        """
        获取所有分桶列表
        
        参数：
            league: 可选，按联赛过滤
        
        返回：
            分桶列表
        """
        buckets = []
        for bucket_key, bucket in self.db.items():
            if league and bucket.get('league') != league:
                continue
            
            buckets.append({
                'bucket_key': bucket_key,
                'league': bucket.get('league'),
                'match_type': bucket.get('match_type'),
                'total_line': bucket.get('total_line'),
                'handicap': bucket.get('handicap'),
                'sample_count': bucket.get('sample_count', 0),
                'has_stats': len(bucket.get('stats', {})) > 0,
            })
        
        buckets.sort(key=lambda x: -x['sample_count'])
        return buckets
    
    def get_half_time_ratio(self, league: str, total_line: float, 
                            handicap: float = 0.0, match_type: str = 'league') -> float:
        """
        获取半场进球比例
        
        参数：
            league: 联赛名称
            total_line: 大小球盘口线
            handicap: 亚盘让球
            match_type: 比赛类型
        
        返回：
            半场进球比例（默认0.42）
        """
        stats = self.get_stats(league, total_line, handicap, match_type, min_samples=20)
        
        if stats and 'half_time_ratio_avg' in stats:
            return stats['half_time_ratio_avg']
        
        # 默认比例，根据盘口调整
        ratio = 0.42
        
        if total_line <= 2.0:
            ratio -= 0.03
        elif total_line <= 2.25:
            ratio -= 0.015
        elif total_line >= 3.0:
            ratio += 0.025
        elif total_line >= 2.75:
            ratio += 0.015
        
        if abs(handicap) >= 1.0:
            ratio += 0.015
        
        return min(0.49, max(0.36, ratio))
    
    def clear(self):
        """清空数据库"""
        self.db = {}


# ==================== 便捷函数 ====================

def get_half_time_statistics(league: str, total_line: float, 
                             handicap: float = 0.0, match_type: str = 'league') -> Optional[Dict]:
    """
    获取半场统计数据
    
    参数：
        league: 联赛名称
        total_line: 大小球盘口线
        handicap: 亚盘让球
        match_type: 比赛类型
    
    返回：
        统计数据字典
    """
    db = HalfTimeStatsDB()
    return db.get_stats(league, total_line, handicap, match_type)


def record_half_time_result(league: str, total_line: float, handicap: float,
                            match_type: str, half_home: int, half_away: int,
                            full_home: int, full_away: int):
    """
    记录半场比分结果
    
    参数：
        league: 联赛名称
        total_line: 大小球盘口线
        handicap: 亚盘让球
        match_type: 比赛类型
        half_home: 半场主队进球
        half_away: 半场客队进球
        full_home: 全场主队进球
        full_away: 全场客队进球
    """
    db = HalfTimeStatsDB()
    db.record_match(league, total_line, handicap, match_type, 
                    half_home, half_away, full_home, full_away)
    db.save()


# ==================== 命令行工具 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='半场比分统计数据库')
    parser.add_argument('--stats', action='store_true', help='查看分桶统计')
    parser.add_argument('--analyze', action='store_true', help='分析半场数据')
    parser.add_argument('--clear', action='store_true', help='清空数据库')
    
    args = parser.parse_args()
    
    db = HalfTimeStatsDB()
    
    if args.stats:
        buckets = db.get_bucket_list()
        print("="*70)
        print("半场比分统计分桶列表")
        print("="*70)
        for bucket in buckets[:20]:
            print(f"{bucket['bucket_key']}:")
            print(f"  样本数: {bucket['sample_count']} | 类型: {bucket['match_type']}")
            print(f"  盘口: {bucket['total_line']} | 让球: {bucket['handicap']}")
            print(f"  有统计: {'是' if bucket['has_stats'] else '否'}")
            print()
        
    elif args.analyze:
        buckets = db.get_bucket_list()
        print("="*70)
        print("半场比分统计分析")
        print("="*70)
        
        total_samples = sum(b['sample_count'] for b in buckets)
        avg_ratio = []
        draw_rates = []
        
        for bucket in buckets:
            if bucket['has_stats'] and bucket['sample_count'] >= 20:
                stats = db.get_stats(bucket['league'], bucket['total_line'], 
                                     bucket['handicap'], bucket['match_type'])
                if stats:
                    avg_ratio.append(stats.get('half_time_ratio_avg', 0))
                    draw_rates.append(stats.get('first_half_draw_rate', 0))
        
        print(f"总分桶数: {len(buckets)}")
        print(f"总样本数: {total_samples}")
        print(f"平均半场比例: {sum(avg_ratio)/len(avg_ratio):.3f}" if avg_ratio else "无足够样本")
        print(f"平均半场平局率: {sum(draw_rates)/len(draw_rates):.3f}" if draw_rates else "无足够样本")
        print()
        
        # 显示有统计的分桶详情
        print("有统计的分桶:")
        for bucket in buckets[:10]:
            if bucket['has_stats'] and bucket['sample_count'] >= 20:
                stats = db.get_stats(bucket['league'], bucket['total_line'], 
                                     bucket['handicap'], bucket['match_type'])
                print(f"\n{bucket['bucket_key']}:")
                print(f"  半场进球均值: {stats.get('first_half_goals_avg', 0):.2f}")
                print(f"  半场比例: {stats.get('half_time_ratio_avg', 0):.3f}")
                print(f"  半场平局率: {stats.get('first_half_draw_rate', 0):.3f}")
                print(f"  主队半场领先率: {stats.get('home_lead_at_half_rate', 0):.3f}")
        
    elif args.clear:
        confirm = input("确定要清空半场统计数据库吗? (y/n): ")
        if confirm.lower() == 'y':
            db.clear()
            db.save()
            print("数据库已清空")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()