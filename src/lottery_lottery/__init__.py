#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透分析器 - 融合排名模型、特征贡献度、动态权重调整、周期识别、多模型集成投票
============================================================================

大乐透规则：
- 前区：从01-35中选择5个号码
- 后区：从01-12中选择2个号码

功能：
1. 排名模型（Top-N排序）- 基于多特征加权综合评分
2. 特征贡献度分析 - 频率、遗漏、位置、012路等特征贡献度
3. 动态权重调整 - 根据回测结果自动优化权重
4. 周期与状态识别 - 识别冷热状态和趋势
5. 多模型集成投票 - 综合多个模型结果
"""

import json
import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# 导入公共日志模块
try:
    from ..logger import setup_logger
except ImportError:
    import sys
    sys.path.insert(0, '../')
    from logger import setup_logger

log = setup_logger('lottery')

# ===================== 常量配置 =====================
FRONT_NUMBERS = list(range(1, 36))  # 前区号码 01-35
BACK_NUMBERS = list(range(1, 13))   # 后区号码 01-12
FEATURE_WEIGHTS = {
    'frequency': 0.25,      # 频率特征权重
    'gap': 0.25,            # 遗漏特征权重
    'position': 0.20,       # 位置特征权重
    'road': 0.15,           # 012路特征权重
    'sum': 0.15,            # 和值特征权重
}

class LotteryAnalyzer:
    """大乐透分析器"""
    
    def __init__(self, history_file: str = 'lottery_history.json'):
        self.history_file = history_file
        self.history_data = self._load_history()
        self.statistics = {}
        self.update_statistics()
    
    def _load_history(self) -> List[Dict]:
        """加载历史数据"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('results', [])
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果没有历史文件，生成一些模拟数据
            return self._generate_simulated_data()
    
    def _generate_simulated_data(self) -> List[Dict]:
        """生成模拟历史数据"""
        results = []
        for i in range(100):
            front = sorted(random.sample(FRONT_NUMBERS, 5))
            back = sorted(random.sample(BACK_NUMBERS, 2))
            results.append({
                'issue': f'2025{str(i+1).zfill(3)}',
                'front': front,
                'back': back,
                'date': f'2025-{str((i//30)+1).zfill(2)}-{str((i%30)+1).zfill(2)}'
            })
        return results
    
    def save_history(self):
        """保存历史数据"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump({'results': self.history_data}, f, ensure_ascii=False, indent=2)
    
    def add_result(self, issue: str, front: List[int], back: List[int], date: str):
        """添加新的开奖结果"""
        self.history_data.insert(0, {
            'issue': issue,
            'front': front,
            'back': back,
            'date': date
        })
        self.update_statistics()
    
    def update_statistics(self):
        """更新统计数据"""
        self.statistics = self._calculate_statistics()
    
    def _calculate_statistics(self) -> Dict:
        """计算各项统计数据"""
        if not self.history_data:
            return {}
        
        # 前区频率统计
        front_freq = defaultdict(int)
        # 后区频率统计
        back_freq = defaultdict(int)
        # 位置频率统计（前区5个位置）
        position_freq = [defaultdict(int) for _ in range(5)]
        # 遗漏统计
        front_gaps = defaultdict(int)
        back_gaps = defaultdict(int)
        # 最近出现位置记录
        last_positions = {}
        
        # 初始化遗漏为当前期数
        for num in FRONT_NUMBERS:
            front_gaps[num] = len(self.history_data)
        for num in BACK_NUMBERS:
            back_gaps[num] = len(self.history_data)
        
        # 遍历历史数据计算频率和遗漏
        for idx, result in enumerate(self.history_data):
            front = result['front']
            back = result['back']
            
            for i, num in enumerate(front):
                front_freq[num] += 1
                position_freq[i][num] += 1
                front_gaps[num] = idx  # 更新遗漏
                last_positions[num] = i
            
            for num in back:
                back_freq[num] += 1
                back_gaps[num] = idx
        
        # 计算平均遗漏
        front_avg_gap = sum(front_gaps.values()) / len(FRONT_NUMBERS)
        back_avg_gap = sum(back_gaps.values()) / len(BACK_NUMBERS)
        
        # 计算和值统计
        sum_counts = defaultdict(int)
        for result in self.history_data:
            s = sum(result['front'])
            sum_counts[s] += 1
        
        # 012路统计
        road_counts = {'0': defaultdict(int), '1': defaultdict(int), '2': defaultdict(int)}
        for num in FRONT_NUMBERS:
            road = str(num % 3)
            road_counts[road][num] = front_freq[num]
        
        # 奇偶分布统计
        odd_even_dist = defaultdict(int)
        for result in self.history_data:
            front = result['front']
            odd = sum(1 for n in front if n % 2 == 1)
            even = 5 - odd
            key = f'{odd}:{even}'
            odd_even_dist[key] += 1
        
        # 大小分布统计（前区1-17为小，18-35为大）
        size_dist = defaultdict(int)
        for result in self.history_data:
            front = result['front']
            small = sum(1 for n in front if n <= 17)
            large = 5 - small
            key = f'{small}:{large}'
            size_dist[key] += 1
        
        # 012路分布统计
        road_dist = defaultdict(int)
        road_total = [0, 0, 0]  # 0路、1路、2路的总数
        for result in self.history_data:
            front = result['front']
            rc = [0, 0, 0]
            for n in front:
                road = n % 3
                rc[road] += 1
                road_total[road] += 1
            key = f'{rc[0]}:{rc[1]}:{rc[2]}'
            road_dist[key] += 1
        
        return {
            'total_issues': len(self.history_data),
            'front_frequency': dict(front_freq),
            'back_frequency': dict(back_freq),
            'position_frequency': [dict(pf) for pf in position_freq],
            'front_current_gaps': dict(front_gaps),
            'back_current_gaps': dict(back_gaps),
            'front_avg_gap': front_avg_gap,
            'back_avg_gap': back_avg_gap,
            'sum_analysis': {
                'min': min(sum_counts.keys(), default=0),
                'max': max(sum_counts.keys(), default=0),
                'avg': sum(k * v for k, v in sum_counts.items()) / sum(sum_counts.values()) if sum_counts else 0,
                'most_common': sorted(sum_counts.items(), key=lambda x: -x[1])[:5]
            },
            'road_analysis': {
                '0': dict(road_counts['0']),
                '1': dict(road_counts['1']),
                '2': dict(road_counts['2']),
                'distribution': dict(road_dist),
                'total': {
                    0: road_total[0],
                    1: road_total[1],
                    2: road_total[2]
                }
            },
            'odd_even_analysis': {
                'distribution': dict(odd_even_dist),
                'description': '奇偶比: 奇数个数:偶数个数'
            },
            'size_analysis': {
                'distribution': dict(size_dist),
                'description': '大小比: 小数个数(1-17):大数个数(18-35)'
            },
            'hot_front': sorted(front_freq.items(), key=lambda x: -x[1])[:10],
            'cold_front': sorted(front_freq.items(), key=lambda x: x[1])[:10],
            'hot_back': sorted(back_freq.items(), key=lambda x: -x[1])[:5],
            'cold_back': sorted(back_freq.items(), key=lambda x: x[1])[:5],
        }
    
    def get_statistics(self) -> Dict:
        """获取统计数据"""
        return self.statistics
    
    def get_recent_results(self, count: int = 10) -> List[Dict]:
        """获取最近开奖结果"""
        return self.history_data[:count]
    
    def get_ensemble_ranking(self, is_front: bool = True, top_n: int = 10) -> List[Dict]:
        """
        获取综合排名
        is_front: True=前区, False=后区
        top_n: 返回前N个号码
        """
        numbers = FRONT_NUMBERS if is_front else BACK_NUMBERS
        scores = []
        
        for num in numbers:
            feature_scores = self._calculate_feature_score(num, is_front)
            if not feature_scores:
                continue
            
            # 综合评分（可以根据需要调整权重）
            total_score = (
                0.3 * feature_scores.get('frequency', 0) +
                0.4 * feature_scores.get('gap', 0) +
                0.2 * feature_scores.get('position', 0) +
                0.1 * feature_scores.get('road', 0)
            )
            
            scores.append({
                'number': num,
                'score': round(total_score, 4),
                'features': feature_scores
            })
        
        # 按综合评分排序
        scores.sort(key=lambda x: -x['score'])
        return scores[:top_n]
    
    def rolling_backtest(self, trials: int = 50) -> Dict:
        """
        滚动回测 - 逐步添加训练数据，每期用之前的数据预测前区和后区
        返回与3D类似的回测统计
        """
        if len(self.history_data) < trials + 10:
            trials = max(20, len(self.history_data) - 10)
        
        start = len(self.history_data) - trials
        
        # 前区命中统计
        front_hit_ge2 = front_hit_ge3 = front_hit_ge4 = 0
        # 后区命中统计
        back_hit_ge1 = back_hit_ge2 = 0
        
        for i in range(start, len(self.history_data)):
            # 重建分析器用于回测
            bt_analyzer = LotteryAnalyzer(self.history_file)
            bt_analyzer.history_data = []
            bt_analyzer.statistics = {}
            
            # 使用之前的数据进行训练
            for j in range(i):
                bt_analyzer.history_data.append(self.history_data[j])
            
            bt_analyzer.update_statistics()
            
            actual = self.history_data[i]
            actual_front = actual['front']
            actual_back = actual['back']
            
            # 获取排名和推荐
            front_ranking = bt_analyzer.get_ensemble_ranking(is_front=True)
            back_ranking = bt_analyzer.get_ensemble_ranking(is_front=False)
            
            front_top5 = [r['number'] for r in front_ranking[:5]]
            back_top3 = [r['number'] for r in back_ranking[:3]]
            
            # 前区命中检查
            front_common = set(actual_front) & set(front_top5)
            if len(front_common) >= 2:
                front_hit_ge2 += 1
            if len(front_common) >= 3:
                front_hit_ge3 += 1
            if len(front_common) >= 4:
                front_hit_ge4 += 1
            
            # 后区命中检查
            back_common = set(actual_back) & set(back_top3)
            if len(back_common) >= 1:
                back_hit_ge1 += 1
            if len(back_common) >= 2:
                back_hit_ge2 += 1
        
        n = trials
        return {
            'trials': n,
            'front_ge2_rate': front_hit_ge2 / n,
            'front_ge3_rate': front_hit_ge3 / n,
            'front_ge4_rate': front_hit_ge4 / n,
            'back_ge1_rate': back_hit_ge1 / n,
            'back_ge2_rate': back_hit_ge2 / n,
        }
    
    # ==================== 排名模型 ====================
    
    def _calculate_feature_score(self, num: int, is_front: bool = True) -> Dict[str, float]:
        """计算单个号码的各特征得分"""
        stats = self.statistics
        if not stats:
            return {}
        
        scores = {}
        freq = stats['front_frequency'] if is_front else stats['back_frequency']
        gaps = stats['front_current_gaps'] if is_front else stats['back_current_gaps']
        avg_gap = stats['front_avg_gap'] if is_front else stats['back_avg_gap']
        
        # 频率得分（越高越热门）
        max_freq = max(freq.values()) if freq else 1
        scores['frequency'] = freq.get(num, 0) / max_freq
        
        # 遗漏得分（越小越可能出现，取倒数）
        gap = gaps.get(num, avg_gap)
        scores['gap'] = 1.0 / (gap + 1)
        
        # 位置得分（在前区各位置的分布）
        if is_front:
            pos_scores = []
            for i in range(5):
                pos_freq = stats['position_frequency'][i]
                pos_max = max(pos_freq.values()) if pos_freq else 1
                pos_scores.append(pos_freq.get(num, 0) / pos_max)
            scores['position'] = sum(pos_scores) / 5
        
        # 012路得分
        road = num % 3
        road_data = stats['road_analysis'].get(str(road), {})
        road_max = max(road_data.values()) if road_data else 1
        scores['road'] = road_data.get(num, 0) / road_max
        
        # 和值相关性得分
        if is_front:
            sum_range = (stats['sum_analysis']['min'], stats['sum_analysis']['max'])
            sum_avg = stats['sum_analysis']['avg']
            # 号码值接近平均和值的1/5时得分较高
            ideal_value = sum_avg / 5
            scores['sum'] = 1.0 - abs(num - ideal_value) / (max(FRONT_NUMBERS))
        
        return scores
    
    def rank_model(self, top_n: int = 10, weights: Dict = None) -> Tuple[List, List]:
        """
        排名模型 - Top-N排序
        返回：(前区排名, 后区排名)
        """
        if weights is None:
            weights = FEATURE_WEIGHTS
        
        # 计算前区排名
        front_scores = []
        for num in FRONT_NUMBERS:
            features = self._calculate_feature_score(num, is_front=True)
            total = sum(features.get(k, 0) * weights.get(k, 0) for k in weights)
            front_scores.append((num, total, features))
        
        # 计算后区排名
        back_scores = []
        for num in BACK_NUMBERS:
            features = self._calculate_feature_score(num, is_front=False)
            # 后区简化计算，主要考虑频率和遗漏
            total = features.get('frequency', 0) * 0.5 + features.get('gap', 0) * 0.5
            back_scores.append((num, total, features))
        
        # 排序并取Top-N
        front_ranked = sorted(front_scores, key=lambda x: -x[1])[:top_n]
        back_ranked = sorted(back_scores, key=lambda x: -x[1])[:min(top_n, 6)]
        
        return front_ranked, back_ranked
    
    # ==================== 特征贡献度分析 ====================
    
    def feature_contribution(self) -> Dict[str, Any]:
        """计算各特征对每个号码的贡献度"""
        # 返回特征权重配置（前端期望的格式）
        return {
            'weights': FEATURE_WEIGHTS,
            'description': {
                'frequency': '频率特征 - 号码出现次数的归一化值',
                'gap': '遗漏特征 - 号码未出现期数的倒数',
                'position': '位置特征 - 号码在前区各位置的分布',
                'road': '012路特征 - 号码按3取模的分布',
                'sum': '和值特征 - 与平均和值的相关性'
            }
        }
    
    # ==================== 动态权重调整 ====================
    
    def dynamic_weight_adjustment(self, backtest_results: List[Dict]) -> Dict[str, float]:
        """根据回测结果动态调整特征权重"""
        if not backtest_results:
            return FEATURE_WEIGHTS.copy()
        
        # 分析每个特征的预测准确性
        feature_scores = defaultdict(float)
        total_count = 0
        
        for result in backtest_results:
            predicted = result.get('predicted', [])
            actual = result.get('actual', [])
            features = result.get('features', {})
            
            for num in actual:
                if num in predicted:
                    # 预测命中，增加相关特征权重
                    for feature, score in features.get(str(num), {}).items():
                        feature_scores[feature] += score
                total_count += 1
        
        # 归一化权重
        total = sum(feature_scores.values()) if feature_scores else 1
        new_weights = {k: v / total for k, v in feature_scores.items()}
        
        # 确保所有特征都有权重
        for feature in FEATURE_WEIGHTS:
            new_weights.setdefault(feature, FEATURE_WEIGHTS[feature] * 0.1)
        
        return new_weights
    
    # ==================== 周期与状态识别 ====================
    
    def identify_cycles(self) -> Dict[str, Dict]:
        """周期与状态识别"""
        stats = self.statistics
        if not stats:
            return {}
        
        front_status = {}
        back_status = {}
        
        # 前区状态识别
        front_avg_freq = sum(stats['front_frequency'].values()) / len(FRONT_NUMBERS)
        for num in FRONT_NUMBERS:
            freq = stats['front_frequency'].get(num, 0)
            gap = stats['front_current_gaps'].get(num, stats['front_avg_gap'])
            avg_gap = stats['front_avg_gap']
            
            status = '稳定'
            trend = '平稳'
            
            # 判断冷热状态
            if freq > front_avg_freq * 1.2:
                status = '热门'
            elif freq < front_avg_freq * 0.8:
                status = '冷门'
            
            # 判断趋势（基于遗漏偏离度）
            # 前区号码遗漏范围 97-109，平均遗漏 104.69，需要更宽松的阈值
            if avg_gap > 0 and gap < avg_gap * 0.94:  # 当前遗漏小于平均遗漏的 94%
                trend = '升温'
            elif avg_gap > 0 and gap > avg_gap * 1.05:  # 当前遗漏大于平均遗漏的 105%
                trend = '降温'
            
            front_status[num] = {
                'status': status,
                'trend': trend,
                'frequency': freq,
                'gap': gap
            }
        
        # 后区状态识别
        back_avg_freq = sum(stats['back_frequency'].values()) / len(BACK_NUMBERS)
        for num in BACK_NUMBERS:
            freq = stats['back_frequency'].get(num, 0)
            gap = stats['back_current_gaps'].get(num, stats['back_avg_gap'])
            avg_gap = stats['back_avg_gap']
            
            status = '稳定'
            trend = '平稳'
            
            if freq > back_avg_freq * 1.15:  # 放宽到 15%
                status = '热门'
            elif freq < back_avg_freq * 0.85:  # 放宽到 85%
                status = '冷门'
            
            # 判断趋势（基于遗漏偏离度）
            # 后区号码遗漏范围 101-109，平均遗漏 105.42，需要更宽松的阈值
            if avg_gap > 0 and gap < avg_gap * 0.96:  # 当前遗漏小于平均遗漏的 96%
                trend = '升温'
            elif avg_gap > 0 and gap > avg_gap * 1.03:  # 当前遗漏大于平均遗漏的 103%
                trend = '降温'
            
            back_status[num] = {
                'status': status,
                'trend': trend,
                'frequency': freq,
                'gap': gap
            }
        
        return {
            'front': front_status,
            'back': back_status,
            'hot_front': [k for k, v in front_status.items() if v['status'] == '热门'],
            'cold_front': [k for k, v in front_status.items() if v['status'] == '冷门'],
            'rising_front': [k for k, v in front_status.items() if v['trend'] == '升温'],
            'falling_front': [k for k, v in front_status.items() if v['trend'] == '降温'],
            'hot_back': [k for k, v in back_status.items() if v['status'] == '热门'],
            'cold_back': [k for k, v in back_status.items() if v['status'] == '冷门'],
            'rising_back': [k for k, v in back_status.items() if v['trend'] == '升温'],
            'falling_back': [k for k, v in back_status.items() if v['trend'] == '降温'],
        }
    
    # ==================== 多模型集成投票 ====================
    
    def _model_bayesian(self, top_n: int = 8) -> List[int]:
        """贝叶斯模型"""
        stats = self.statistics
        if not stats:
            return []
        
        front_freq = stats['front_frequency']
        total = sum(front_freq.values()) if front_freq else 1
        
        scores = {}
        for num in FRONT_NUMBERS:
            freq = front_freq.get(num, 0)
            # 贝叶斯平滑
            scores[num] = (freq + 1) / (total + len(FRONT_NUMBERS))
        
        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
    
    def _model_hot(self, top_n: int = 8) -> List[int]:
        """热号模型"""
        stats = self.statistics
        if not stats:
            return []
        return [num for num, _ in stats['hot_front'][:top_n]]
    
    def _model_cold(self, top_n: int = 8) -> List[int]:
        """冷号模型"""
        stats = self.statistics
        if not stats:
            return []
        return [num for num, _ in stats['cold_front'][:top_n]]
    
    def _model_rank(self, top_n: int = 8) -> List[int]:
        """排名模型"""
        front_ranked, _ = self.rank_model(top_n=top_n)
        return [num for num, _, _ in front_ranked]
    
    def _model_markov(self, top_n: int = 8) -> List[int]:
        """马尔可夫链模型（简化版）"""
        # 基于最近开奖结果的号码关联性
        if len(self.history_data) < 3:
            return []
        
        # 统计相邻期数之间的号码转移频率
        transition = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.history_data) - 1):
            current = set(self.history_data[i]['front'])
            prev = set(self.history_data[i+1]['front'])
            for num in current:
                for next_num in prev:
                    transition[num][next_num] += 1
        
        # 基于最近一期预测
        recent = set(self.history_data[0]['front'])
        scores = defaultdict(int)
        for num in recent:
            for next_num, count in transition[num].items():
                scores[next_num] += count
        
        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
    
    def multi_model_voting(self, front_n: int = 5, back_n: int = 2, n_votes: int = 3) -> Dict:
        """
        多模型集成投票
        - 综合5个模型：贝叶斯、热号、冷号、排名、马尔可夫
        - 获得至少n_votes票的号码进入最终推荐
        """
        # 前区投票
        models = [
            self._model_bayesian(top_n=12),
            self._model_hot(top_n=12),
            self._model_cold(top_n=12),
            self._model_rank(top_n=12),
            self._model_markov(top_n=12),
        ]
        
        # 统计票数
        votes = defaultdict(int)
        for model_idx, model_result in enumerate(models):
            for rank, num in enumerate(model_result):
                # 排名越靠前权重越高
                weight = 1.0 - (rank / len(model_result))
                votes[num] += weight
        
        # 排序并选择
        front_candidates = sorted(votes.items(), key=lambda x: -x[1])[:front_n * 2]
        front_selected = [num for num, _ in front_candidates[:front_n]]
        
        # 后区投票（简化）
        back_votes = defaultdict(int)
        back_freq = self.statistics.get('back_frequency', {})
        back_gaps = self.statistics.get('back_current_gaps', {})
        
        for num in BACK_NUMBERS:
            back_votes[num] = back_freq.get(num, 0) * 0.6 + (1.0 / (back_gaps.get(num, 1) + 1)) * 0.4
        
        back_selected = [num for num, _ in sorted(back_votes.items(), key=lambda x: -x[1])[:back_n]]
        
        # 获取周期状态信息
        cycles = self.identify_cycles()
        
        return {
            'front': front_selected,
            'back': back_selected,
            'front_candidates': [{'number': num, 'score': score} for num, score in front_candidates],
            'back_candidates': [{'number': num, 'score': round(back_votes[num], 3)} for num in BACK_NUMBERS],
            'front_votes': {num: round(v, 3) for num, v in votes.items()},
            'cycle_info': cycles,
            'hot_front': cycles.get('hot_front', []),
            'cold_front': cycles.get('cold_front', []),
            'hot_back': cycles.get('hot_back', []),
            'cold_back': cycles.get('cold_back', []),
        }
    
    # ==================== 生成推荐号码 ====================
    
    def generate_recommendation(self, method: str = 'balanced') -> Dict:
        """生成推荐号码"""
        if method == 'hot':
            # 全热号（添加随机选择）
            hot_front = [num for num, _ in self.statistics.get('hot_front', [])[:10]]
            hot_back = [num for num, _ in self.statistics.get('hot_back', [])[:5]]
            front = sorted(random.sample(hot_front, min(5, len(hot_front))))
            back = sorted(random.sample(hot_back, min(2, len(hot_back))))
        elif method == 'cold':
            # 全冷号（添加随机选择）
            cold_front = [num for num, _ in self.statistics.get('cold_front', [])[:10]]
            cold_back = [num for num, _ in self.statistics.get('cold_back', [])[:5]]
            front = sorted(random.sample(cold_front, min(5, len(cold_front))))
            back = sorted(random.sample(cold_back, min(2, len(cold_back))))
        elif method == 'rank':
            # 排名模型（添加随机选择）
            front_ranked, back_ranked = self.rank_model(top_n=12)
            front_candidates = [num for num, _, _ in front_ranked[:10]]
            back_candidates = [num for num, _, _ in back_ranked[:6]]
            front = sorted(random.sample(front_candidates, min(5, len(front_candidates))))
            back = sorted(random.sample(back_candidates, min(2, len(back_candidates))))
        else:
            # 平衡模式（集成投票，添加随机选择）
            result = self.multi_model_voting(front_n=10, back_n=5)
            front_candidates = result['front'][:8]
            back_candidates = result['back'][:4]
            front = sorted(random.sample(front_candidates, min(5, len(front_candidates))))
            back = sorted(random.sample(back_candidates, min(2, len(back_candidates))))
        
        return {
            'front': front,
            'back': back,
            'method': method
        }
    
    # ==================== 回测功能 ====================
    
    def backtest(self, method: str = 'balanced', test_periods: int = 30) -> Dict:
        """历史回测"""
        if len(self.history_data) < test_periods:
            return {'error': '历史数据不足'}
        
        results = []
        total_matched = 0
        total_front_matched = 0
        total_back_matched = 0
        
        for i in range(test_periods):
            # 使用第i+1期及之前的数据进行预测
            test_data = self.history_data[i+1:]
            original_data = self.history_data
            self.history_data = test_data
            
            # 生成预测
            pred = self.generate_recommendation(method)
            front_pred = set(pred['front'])
            back_pred = set(pred['back'])
            
            # 实际结果
            actual = self.history_data[0]
            front_actual = set(actual['front'])
            back_actual = set(actual['back'])
            
            # 计算命中
            front_match = len(front_pred & front_actual)
            back_match = len(back_pred & back_actual)
            
            results.append({
                'issue': actual['issue'],
                'predicted_front': sorted(list(front_pred)),
                'actual_front': actual['front'],
                'front_matched': front_match,
                'predicted_back': sorted(list(back_pred)),
                'actual_back': actual['back'],
                'back_matched': back_match,
                'full_match': front_match == 5 and back_match == 2
            })
            
            total_front_matched += front_match
            total_back_matched += back_match
            if front_match == 5 and back_match == 2:
                total_matched += 1
            
            # 恢复原始数据
            self.history_data = original_data
        
        return {
            'method': method,
            'test_periods': test_periods,
            'total_matched': total_matched,
            'front_accuracy': total_front_matched / (test_periods * 5),
            'back_accuracy': total_back_matched / (test_periods * 2),
            'detailed_results': results
        }

    # ==================== 动态抓取开奖号码 ====================
    
    def fetch_latest_results(self, count: int = 10) -> Dict:
        """动态抓取最新开奖号码
        
        尝试从多个数据源抓取最新的大乐透开奖结果，如果网络不可用则返回模拟数据。
        
        Args:
            count: 要抓取的期数（默认10期）
        
        Returns:
            包含抓取结果和状态信息的字典
        """
        try:
            # 尝试从网络抓取
            results = self._fetch_from_web(count)
            
            if results:
                # 更新本地数据
                self._update_with_fetched(results)
                
                return {
                    'success': True,
                    'source': 'web',
                    'count': len(results),
                    'message': f'成功抓取 {len(results)} 期数据',
                    'latest_issue': results[0]['issue'] if results else None,
                    'results': results[:count]
                }
            else:
                # 抓取失败，返回本地数据
                return {
                    'success': False,
                    'source': 'local',
                    'count': min(count, len(self.history_data)),
                    'message': '网络抓取失败，使用本地缓存数据',
                    'latest_issue': self.history_data[0]['issue'] if self.history_data else None,
                    'results': self.get_recent_results(count)
                }
        except Exception as e:
            log.error(f'抓取开奖号码失败: {e}')
            return {
                'success': False,
                'source': 'local',
                'count': min(count, len(self.history_data)),
                'message': f'抓取失败: {str(e)}，使用本地缓存数据',
                'latest_issue': self.history_data[0]['issue'] if self.history_data else None,
                'results': self.get_recent_results(count)
            }
    
    def _fetch_from_web(self, count: int) -> List[Dict]:
        """从网络抓取开奖数据"""
        try:
            import urllib.request
            import urllib.error
            
            # 使用更可靠的数据源
            sources = [
                # ip138 大乐透历史开奖
                ('https://cp.ip138.com/daletou/', 'ip138'),
                # 彩票之家 API
                ('https://www.cailele.com/static/new lottery/info/dlt_kaijiang.json', 'cailele'),
            ]
            
            for url, source in sources:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'zh-CN,zh;q=0.9',
                    }
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as response:
                        data = response.read().decode('utf-8', errors='ignore')
                        results = self._parse_web_data(data, source)
                        if results:
                            return results[:count]
                except Exception as e:
                    log.debug(f'数据源 {source} 抓取失败: {e}')
                    continue
            
            # 如果都失败，返回空列表
            return []
        except ImportError:
            return []
    
    def _parse_web_data(self, data: str, source: str) -> List[Dict]:
        """解析网络抓取的数据"""
        results = []
        
        try:
            import re
            
            if source == 'ip138':
                # ip138 新格式：使用span标签
                # 最新一期: <span class="period">2026061</span>
                # 历史数据表格格式:
                # <tr><td><span>2026061</span></td><td><span>06-03</span></td><td class="award"><span class="icon-redball" data-value="10">10</span>...</td></tr>
                
                # 查找所有表格
                table_pattern = r'<table[^>]*>(.*?)</table>'
                table_matches = re.findall(table_pattern, data, re.DOTALL)
                
                for table_content in table_matches:
                    # 检查是否包含历史开奖数据
                    if 'icon-redball' in table_content and 'icon-blueball' in table_content:
                        # 提取表格中的所有行（排除表头）
                        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL)
                        
                        for row in rows[1:]:  # 跳过表头行
                            # 提取期号
                            issue_match = re.search(r'<td[^>]*><span>(\d{7})</span></td>', row)
                            # 提取日期
                            date_match = re.search(r'<td[^>]*>\s*<span>(\d{2}-\d{2})</span>\s*</td>', row)
                            # 提取红球号码
                            red_balls = re.findall(r'icon-redball[^>]*>(\d+)</span>', row)
                            # 提取蓝球号码
                            blue_balls = re.findall(r'icon-blueball[^>]*>(\d+)</span>', row)
                            
                            if issue_match and len(red_balls) >= 5 and len(blue_balls) >= 2:
                                issue = issue_match.group(1)
                                date_str = date_match.group(1) if date_match else ''
                                front = sorted([int(n) for n in red_balls[:5]])
                                back = sorted([int(n) for n in blue_balls[:2]])
                                
                                if all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back):
                                    # 避免重复添加
                                    if not any(r['issue'] == issue for r in results):
                                        # 添加年份（假设是当前年份2026）
                                        full_date = f'2026-{date_str}' if date_str else ''
                                        results.append({
                                            'issue': issue,
                                            'front': front,
                                            'back': back,
                                            'date': full_date
                                        })
                
                # 如果没有从表格中获取到数据，尝试提取最新一期（旧格式）
                if not results:
                    period_match = re.search(r'<span class="period">(\d{7})</span>', data)
                    if period_match:
                        latest_issue = period_match.group(1)
                        all_balls = re.findall(r'alt="(\d+)"', data)
                        
                        if len(all_balls) >= 7:
                            front = sorted([int(all_balls[i]) for i in range(5)])
                            back = sorted([int(all_balls[5]), int(all_balls[6])])
                            
                            if all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back):
                                results.append({
                                    'issue': latest_issue,
                                    'front': front,
                                    'back': back,
                                    'date': ''
                                })
                            
            elif source == 'cailele':
                # 尝试解析JSON格式
                import json
                try:
                    json_data = json.loads(data)
                    if isinstance(json_data, list):
                        for item in json_data[:20]:
                            if isinstance(item, dict):
                                # 尝试多种可能的字段名
                                issue = item.get('issue') or item.get('period') or item.get('qihao') or ''
                                front_str = item.get('front') or item.get('red') or item.get('red_ball') or ''
                                back_str = item.get('back') or item.get('blue') or item.get('blue_ball') or ''
                                
                                # 解析前区号码
                                front_nums = re.findall(r'\d+', front_str)
                                back_nums = re.findall(r'\d+', back_str)
                                
                                if len(front_nums) >= 5 and len(back_nums) >= 2:
                                    front = sorted([int(n) for n in front_nums[:5]])
                                    back = sorted([int(n) for n in back_nums[:2]])
                                    
                                    if all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back):
                                        results.append({
                                            'issue': str(issue),
                                            'front': front,
                                            'back': back,
                                            'date': item.get('date', '')
                                        })
                except json.JSONDecodeError:
                    pass
            
            # 如果以上都不行，尝试通用正则
            if not results:
                # 通用格式: 2026061期：10 12 26 31 35 + 02 12
                patterns = [
                    r'(\d{7})[^：:\d]*(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s*[+＋\-]\s*(\d{2})\s+(\d{2})',
                    r'(\d{7})\D*(\d{2})\D+(\d{2})\D+(\d{2})\D+(\d{2})\D+(\d{2})\D*[+－-]\D*(\d{2})\D+(\d{2})',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, data)
                    for match in matches:
                        if len(match) >= 8:
                            issue = match[0]
                            front = sorted([int(match[i]) for i in range(1, 6)])
                            back = sorted([int(match[6]), int(match[7])])
                            
                            if all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back):
                                results.append({
                                    'issue': issue,
                                    'front': front,
                                    'back': back,
                                    'date': ''
                                })
                                
        except Exception as e:
            log.debug(f'解析数据失败: {e}')
        
        return results
    
    def _update_with_fetched(self, fetched_results: List[Dict]):
        """使用抓取的数据更新本地历史"""
        if not fetched_results:
            return
        
        # 按抓取到的数据重建历史
        # 创建一个期号到数据的映射
        fetched_map = {r['issue']: r for r in fetched_results}
        
        # 获取抓取到的最新期号和最早期号
        fetched_issues = sorted(fetched_map.keys(), reverse=True)
        latest_fetched = fetched_issues[0]
        earliest_fetched = fetched_issues[-1]
        
        # 保留本地数据中比抓取数据更早的部分
        preserved_local = []
        for result in self.history_data:
            if result['issue'] < earliest_fetched:
                preserved_local.append(result)
        
        # 合并抓取数据和保留的本地数据
        # 先添加抓取的数据（按降序），然后添加保留的本地数据
        merged_results = sorted(fetched_results, key=lambda x: x['issue'], reverse=True) + preserved_local
        
        # 更新历史数据
        self.history_data = merged_results
        self.update_statistics()
        self.save_history()
        log.info(f'成功更新历史数据，共 {len(fetched_results)} 期来自网络')


# 全局分析器实例
_lottery_analyzer = None

def get_lottery_analyzer() -> LotteryAnalyzer:
    """获取大乐透分析器实例"""
    global _lottery_analyzer
    if _lottery_analyzer is None:
        _lottery_analyzer = LotteryAnalyzer()
    return _lottery_analyzer


if __name__ == '__main__':
    analyzer = get_lottery_analyzer()
    
    print("=== 大乐透分析器 ===")
    stats = analyzer.get_statistics()
    print(f"总期数: {stats.get('total_issues', 0)}")
    
    # 测试排名模型
    front_ranked, back_ranked = analyzer.rank_model(top_n=10)
    print("\n前区排名 Top-10:")
    for num, score, features in front_ranked[:10]:
        print(f"  {num:02d}: {score:.4f}")
    
    print("\n后区排名 Top-6:")
    for num, score, features in back_ranked[:6]:
        print(f"  {num:02d}: {score:.4f}")
    
    # 测试集成投票
    print("\n=== 多模型集成投票推荐 ===")
    result = analyzer.multi_model_voting()
    print(f"前区推荐: {[f'{n:02d}' for n in result['front']]}")
    print(f"后区推荐: {[f'{n:02d}' for n in result['back']]}")
    
    # 测试周期识别
    print("\n=== 周期状态识别 ===")
    cycles = analyzer.identify_cycles()
    print(f"热门前区: {[f'{n:02d}' for n in cycles.get('hot_front', [])[:5]]}")
    print(f"冷门前区: {[f'{n:02d}' for n in cycles.get('cold_front', [])[:5]]}")
    print(f"升温前区: {[f'{n:02d}' for n in cycles.get('rising_front', [])[:5]]}")
    print(f"降温前区: {[f'{n:02d}' for n in cycles.get('falling_front', [])[:5]]}")
