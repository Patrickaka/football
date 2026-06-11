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
6. 趋势分析 - AC值、连号、重号、区间分布、和值趋势、升温降温轨迹
"""

import json
import random
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# 导入公共日志模块
from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common.data_cache import cached_fetch, is_cache_valid, save_cached_data

log = setup_logger('lottery')

# ===================== 常量配置 =====================
FRONT_NUMBERS = list(range(1, 36))  # 前区号码 01-35
BACK_NUMBERS = list(range(1, 13))   # 后区号码 01-12

# 统一特征权重 (所有排名/评分函数共用同一套)
FEATURE_WEIGHTS = {
    'frequency': 0.22,      # 频率特征 (指数衰减加权)
    'gap': 0.22,            # 遗漏特征 (含标准差维度)
    'position': 0.16,       # 位置特征
    'road': 0.13,           # 012路特征
    'sum': 0.12,            # 和值特征
    'trend': 0.10,          # 升温降温趋势
    'zone': 0.05,           # 区间分布特征
}

# 时间衰减因子 (最近一期权重1.0，20期前≈0.19，50期前≈0.016)
TIME_DECAY_FACTOR = 0.92

# 二阶马尔可夫权重
MARKOV2_WEIGHT = 0.35

# 推荐约束参数
MAX_CONSECUTIVE_IN_RECOMMEND = 1   # 推荐号码中最多允许1对连号
ODD_PARITY_TOLERANCE = 1           # 奇偶比容忍偏差
ZONE_COVERAGE_MIN = 2              # 至少覆盖2个区间(1-12/13-24/25-35)
SIZE_BALANCE_RANGE = (1, 4)        # 大小比范围: 最小1个小号, 最多4个小号

# 遗漏评分: 基于热度衰减估算"距期望出现剩余期数"
GAP_TIGHTEN_FACTOR = 0.88          # 接近平均遗漏时的紧化因子

# 预测结果缓存配置
_prediction_cache = None
_cache_time = 0

def _is_today_cache(cache_timestamp):
    """检查缓存是否是今天的（按自然天判断）"""
    if cache_timestamp is None or cache_timestamp == 0:
        return False
    
    import datetime
    cache_date = datetime.date.fromtimestamp(cache_timestamp)
    today = datetime.date.today()
    return cache_date == today

def clear_cache():
    """清除缓存"""
    global _prediction_cache, _cache_time
    _prediction_cache = None
    _cache_time = 0
    log.info("大乐透模块缓存已清除")


class LotteryAnalyzer:
    """大乐透分析器"""

    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or data_path('lottery_history.json')
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
        """计算各项统计数据 (v2: 时间衰减+更多维度)"""
        if not self.history_data:
            return {}

        total = len(self.history_data)

        # ---- 频率统计 (原始+衰减) ----
        front_freq_raw = defaultdict(int)
        back_freq_raw = defaultdict(int)
        front_freq_decayed = defaultdict(float)
        back_freq_decayed = defaultdict(float)
        position_freq = [defaultdict(float) for _ in range(5)]

        # ---- 遗漏统计 (真实遗漏期数) ----
        last_seen_front = {}   # num -> index (0=most recent)
        last_seen_back = {}
        front_gap_history = defaultdict(list)  # num -> [gap1, gap2, ...]
        back_gap_history = defaultdict(list)

        # ---- 其他统计 ----
        sum_counts = defaultdict(int)
        sum_list = []  # 按时间顺序的和值 (用于趋势分析)
        odd_even_dist = defaultdict(int)
        size_dist = defaultdict(int)
        road_dist = defaultdict(int)
        road_total = [0, 0, 0]
        zone_dist = defaultdict(int)   # 区间分布: key="z1,z2,z3"

        # AC值统计
        ac_counts = defaultdict(int)

        # 连号统计
        consecutive_counts = defaultdict(int)  # pair count per draw

        # 重号统计
        duplicate_counts = []  # 每期与前一期重叠个数

        # 遍历历史数据 (idx=0 是最新一期)
        for idx, result in enumerate(self.history_data):
            front = result['front']
            back = result['back']
            decay = TIME_DECAY_FACTOR ** idx

            # 时间衰减频率
            for i, num in enumerate(front):
                front_freq_raw[num] += 1
                front_freq_decayed[num] += decay
                position_freq[i][num] += decay
                last_seen_front[num] = idx

            for num in back:
                back_freq_raw[num] += 1
                back_freq_decayed[num] += decay
                last_seen_back[num] = idx

            # 和值
            s = sum(front)
            sum_counts[s] += 1
            sum_list.append(s)

            # 奇偶分布
            odd = sum(1 for n in front if n % 2 == 1)
            odd_even_dist[f'{odd}:{5-odd}'] += 1

            # 大小分布 (小:1-17, 大:18-35)
            small = sum(1 for n in front if n <= 17)
            size_dist[f'{small}:{5-small}'] += 1

            # 012路
            rc = [0, 0, 0]
            for n in front:
                r = n % 3
                rc[r] += 1
                road_total[r] += int(decay)
            road_dist[f'{rc[0]}:{rc[1]}:{rc[2]}'] += 1

            # 区间分布 (zone 1:1-12, zone 2:13-24, zone 3:25-35)
            z = [0, 0, 0]
            for n in front:
                if n <= 12:
                    z[0] += 1
                elif n <= 24:
                    z[1] += 1
                else:
                    z[2] += 1
            zone_dist[f'{z[0]}:{z[1]}:{z[2]}'] += 1

            # AC值
            ac = self._calc_ac_value(front)
            ac_counts[ac] += 1

            # 连号对数
            cp = sum(1 for j in range(len(front)-1) if front[j+1] - front[j] == 1)
            consecutive_counts[cp] += 1

            # 重号 (与前一期比较)
            if idx < total - 1:
                prev_front = self.history_data[idx + 1]['front']
                dup = len(set(front) & set(prev_front))
                duplicate_counts.append(dup)

        # ---- 遗漏期数计算 ----
        front_gaps = {}
        back_gaps = {}
        for num in FRONT_NUMBERS:
            if num in last_seen_front:
                front_gaps[num] = last_seen_front[num]
            else:
                front_gaps[num] = total
        for num in BACK_NUMBERS:
            if num in last_seen_back:
                back_gaps[num] = last_seen_back[num]
            else:
                back_gaps[num] = total

        # 遗漏历史 (每个号码的历次遗漏间隔)
        for num in FRONT_NUMBERS:
            if num in last_seen_front:
                prev = -1
                for idx, result in enumerate(reversed(self.history_data)):
                    if num in result['front']:
                        if prev >= 0:
                            front_gap_history[num].append(idx - prev)
                        prev = idx
                if not front_gap_history[num]:
                    front_gap_history[num] = [total]
            else:
                front_gap_history[num] = [total]

        for num in BACK_NUMBERS:
            if num in last_seen_back:
                prev = -1
                for idx, result in enumerate(reversed(self.history_data)):
                    if num in result['back']:
                        if prev >= 0:
                            back_gap_history[num].append(idx - prev)
                        prev = idx
                if not back_gap_history[num]:
                    back_gap_history[num] = [total]
            else:
                back_gap_history[num] = [total]

        # 遗漏标准差
        front_gap_std = {}
        back_gap_std = {}
        for num in FRONT_NUMBERS:
            gaps = front_gap_history[num]
            if len(gaps) >= 2:
                mean = sum(gaps) / len(gaps)
                front_gap_std[num] = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
            else:
                front_gap_std[num] = 5.0  # 默认值

        for num in BACK_NUMBERS:
            gaps = back_gap_history[num]
            if len(gaps) >= 2:
                mean = sum(gaps) / len(gaps)
                back_gap_std[num] = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
            else:
                back_gap_std[num] = 3.0

        # 平均遗漏
        front_avg_gap = sum(front_gaps.values()) / len(FRONT_NUMBERS)
        back_avg_gap = sum(back_gaps.values()) / len(BACK_NUMBERS)

        # ---- 和值趋势分析 ----
        sum_trend = self._analyze_sum_trend(sum_list)

        # ---- 升温降温轨迹 ----
        temp_trajectory = self._calc_temperature_trajectory()

        # ---- 重号率 ----
        dup_rate = sum(duplicate_counts) / len(duplicate_counts) if duplicate_counts else 0

        return {
            'total_issues': total,
            # 频率 (兼容前端: 保留原始计数作为主key，decayed版本供评分使用)
            'front_frequency': dict(front_freq_raw),
            'back_frequency': dict(back_freq_raw),
            'front_frequency_decayed': dict(front_freq_decayed),
            'back_frequency_decayed': dict(back_freq_decayed),
            'position_frequency': [dict(pf) for pf in position_freq],
            # 遗漏
            'front_current_gaps': dict(front_gaps),
            'back_current_gaps': dict(back_gaps),
            'front_gap_std': dict(front_gap_std),
            'back_gap_std': dict(back_gap_std),
            'front_avg_gap': front_avg_gap,
            'back_avg_gap': back_avg_gap,
            # 和值
            'sum_analysis': {
                'min': min(sum_counts.keys(), default=0),
                'max': max(sum_counts.keys(), default=0),
                'avg': sum(k * v for k, v in sum_counts.items()) / sum(sum_counts.values()) if sum_counts else 0,
                'most_common': sorted(sum_counts.items(), key=lambda x: -x[1])[:5],
                'trend': sum_trend,
            },
            # 012路
            'road_analysis': {
                '0': {}, '1': {}, '2': {},
                'distribution': dict(road_dist),
                'total': {0: road_total[0], 1: road_total[1], 2: road_total[2]}
            },
            # 奇偶/大小
            'odd_even_analysis': {
                'distribution': dict(odd_even_dist),
                'description': '奇偶比: 奇数个数:偶数个数'
            },
            'size_analysis': {
                'distribution': dict(size_dist),
                'description': '大小比: 小数个数(1-17):大数个数(18-35)'
            },
            # 热冷号
            'hot_front': sorted(front_freq_raw.items(), key=lambda x: -x[1])[:10],
            'cold_front': sorted(front_freq_raw.items(), key=lambda x: x[1])[:10],
            'hot_back': sorted(back_freq_raw.items(), key=lambda x: -x[1])[:5],
            'cold_back': sorted(back_freq_raw.items(), key=lambda x: x[1])[:5],
            # 新增: AC值分析
            'ac_analysis': {
                'distribution': dict(ac_counts),
                'most_common_ac': sorted(ac_counts.items(), key=lambda x: -x[1])[:3],
                'avg_ac': sum(k * v for k, v in ac_counts.items()) / sum(ac_counts.values()) if ac_counts else 0,
                'description': 'AC值: 号码离散度，范围4-10，越高越分散'
            },
            # 新增: 连号分析
            'consecutive_analysis': {
                'distribution': dict(consecutive_counts),
                'pct_with_consecutive': sum(v for k, v in consecutive_counts.items() if k > 0) / total if total else 0,
                'description': '每期连号对数统计'
            },
            # 新增: 重号分析
            'duplicate_analysis': {
                'avg_duplicates': round(dup_rate, 3),
                'pct_has_duplicate': sum(1 for d in duplicate_counts if d > 0) / len(duplicate_counts) if duplicate_counts else 0,
                'description': '与上期号码的重叠情况'
            },
            # 新增: 区间分布
            'zone_analysis': {
                'distribution': dict(zone_dist),
                'description': '区间分布: zone1(1-12):zone2(13-24):zone3(25-35)'
            },
            # 新增: 升温降温轨迹
            'temperature_trajectory': temp_trajectory,
        }

    # ==================== 新增分析功能 ====================

    @staticmethod
    def _calc_ac_value(numbers: List[int]) -> int:
        """计算AC值（算术复杂度）
        AC = 两两差值的去重个数 - (n-1), 范围 4-10"""
        n = len(numbers)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(abs(numbers[i] - numbers[j]))
        return len(diffs) - (n - 1)

    @staticmethod
    def _analyze_sum_trend(sum_list: List[int]) -> Dict:
        """分析和值趋势"""
        if len(sum_list) < 10:
            return {'direction': 'data_insufficient'}

        # 最近10期 vs 之前10期
        recent_10 = sum(sum_list[:10]) / 10 if len(sum_list) >= 10 else 0
        prior_10 = sum(sum_list[10:20]) / 10 if len(sum_list) >= 20 else recent_10
        all_avg = sum(sum_list) / len(sum_list)

        # 最近5期移动平均趋势
        if len(sum_list) >= 5:
            ma5_recent = sum(sum_list[:5]) / 5
            ma5_prev = sum(sum_list[5:10]) / 5 if len(sum_list) >= 10 else ma5_recent
            ma5_slope = ma5_recent - ma5_prev
        else:
            ma5_slope = 0

        diff = recent_10 - prior_10
        if diff > 5:
            direction = 'up'
        elif diff < -5:
            direction = 'down'
        else:
            direction = 'stable'

        return {
            'direction': direction,
            'recent_avg': round(recent_10, 1),
            'prior_avg': round(prior_10, 1),
            'all_time_avg': round(all_avg, 1),
            'ma5_slope': round(ma5_slope, 1),
            'current_sum': sum_list[0] if sum_list else 0,
        }

    def _calc_temperature_trajectory(self) -> Dict:
        """计算升温降温轨迹 (最近N期的热度变化)"""
        if len(self.history_data) < 5:
            return {}

        recent_span = min(10, len(self.history_data))
        trajectory = {}

        for num in FRONT_NUMBERS:
            # 分两个窗口: 最近5期 vs 之前5期
            m = min(5, recent_span // 2)
            recent_hits = sum(1 for r in self.history_data[:m] if num in r['front'])
            prior_hits = sum(1 for r in self.history_data[m:m*2] if num in r['front'])

            if recent_hits > prior_hits:
                direction = 'rising'
            elif recent_hits < prior_hits:
                direction = 'falling'
            else:
                direction = 'stable'

            trajectory[num] = {
                'direction': direction,
                'recent_hits': recent_hits,
                'prior_hits': prior_hits,
                'current_gap': self._get_current_gap(num, is_front=True),
            }

        return trajectory

    def _get_current_gap(self, num: int, is_front: bool = True) -> int:
        """获取当前遗漏期数"""
        for idx, result in enumerate(self.history_data):
            numbers = result['front'] if is_front else result['back']
            if num in numbers:
                return idx
        return len(self.history_data)

    # ==================== 排名模型 ====================

    def _calculate_feature_score(self, num: int, is_front: bool = True) -> Dict[str, float]:
        """计算单个号码的各特征得分 (v2: 使用衰减频率+遗漏标准差)"""
        stats = self.statistics
        if not stats:
            return {}

        scores = {}
        total = stats['total_issues']

        # 频率得分（使用时间衰减版本）
        freq = stats.get('front_frequency_decayed' if is_front else 'back_frequency_decayed',
                         stats['front_frequency' if is_front else 'back_frequency'])
        max_freq = max(freq.values()) if freq else 1
        scores['frequency'] = freq.get(num, 0) / max_freq

        # 遗漏得分 (v2: 考虑标准差，接近均值时更高)
        gaps = stats['front_current_gaps' if is_front else 'back_current_gaps']
        gap_std = stats.get('front_gap_std' if is_front else 'back_gap_std', {})
        avg_gap = stats['front_avg_gap' if is_front else 'back_avg_gap']

        gap = gaps.get(num, avg_gap)
        std = gap_std.get(num, avg_gap / 2)

        # 使用改良的遗漏评分: 距离平均遗漏越近，得分越高
        # + 冷号到一定程度后也会加分（回补预期）
        gap_ratio = gap / max(avg_gap, 1)
        if gap_ratio < 0.7:
            # 近期出现过的热号: 回归均值加分
            scores['gap'] = 1.0 / (gap_ratio + 0.2)
        elif gap_ratio < 1.3:
            # 接近平均遗漏: 最高分
            scores['gap'] = 0.85
        else:
            # 超过平均遗漏: 遗漏越大回补预期越强，但需打折
            scores['gap'] = 0.85 * (1.0 - math.exp(-(gap_ratio - 1.3) * 2.0))

        # 位置得分
        if is_front:
            pos_scores = []
            for i in range(5):
                pos_freq = stats['position_frequency'][i]
                pos_max = max(pos_freq.values()) if pos_freq else 1
                pos_scores.append(pos_freq.get(num, 0) / pos_max)
            scores['position'] = sum(pos_scores) / 5

        # 012路得分
        road = num % 3
        road_data = stats['road_analysis'].get('total', {})
        road_total_val = road_data.get(road, 0)
        road_all_max = max(road_data.values()) if road_data else 1
        scores['road'] = road_total_val / road_all_max if road_all_max > 0 else 0

        # 和值相关性得分
        if is_front:
            sum_avg = stats['sum_analysis']['avg']
            ideal_value = sum_avg / 5
            scores['sum'] = 1.0 - abs(num - ideal_value) / max(FRONT_NUMBERS)

        # 升温降温趋势得分
        trajectory = stats.get('temperature_trajectory', {})
        traj = trajectory.get(num, {})
        if traj.get('direction') == 'rising':
            scores['trend'] = 0.85
        elif traj.get('direction') == 'falling':
            scores['trend'] = 0.30
        else:
            scores['trend'] = 0.55

        # 区间平衡得分 (基于当前号码在哪个区间)
        if is_front:
            zone_dist = stats.get('zone_analysis', {}).get('distribution', {})
            if num <= 12:
                zone_key_prefix = '1'
            elif num <= 24:
                zone_key_prefix = '2'
            else:
                zone_key_prefix = '3'
            # 查找最常见的区间分布模式
            most_common_zone = max(zone_dist.items(), key=lambda x: x[1]) if zone_dist else ('', 0)
            zone_parts = most_common_zone[0].split(':')
            # 该号码所在区间在热门模式中的占比
            try:
                zone_idx = int(zone_key_prefix) - 1
                zone_count = int(zone_parts[zone_idx]) if zone_idx < len(zone_parts) else 0
                scores['zone'] = min(1.0, zone_count / 3.0)
            except (ValueError, IndexError):
                scores['zone'] = 0.5
        else:
            scores['zone'] = 0.5

        return scores

    def get_ensemble_ranking(self, is_front: bool = True, top_n: int = 10) -> List[Dict]:
        """获取综合排名 (v2: 使用统一FEATURE_WEIGHTS)"""
        numbers = FRONT_NUMBERS if is_front else BACK_NUMBERS
        scores = []

        weights = {k: v for k, v in FEATURE_WEIGHTS.items()}

        for num in numbers:
            feature_scores = self._calculate_feature_score(num, is_front)
            if not feature_scores:
                continue

            total_score = sum(
                feature_scores.get(k, 0) * weights.get(k, 0)
                for k in weights
            )

            scores.append({
                'number': num,
                'score': round(total_score, 4),
                'features': feature_scores
            })

        scores.sort(key=lambda x: -x['score'])
        return scores[:top_n]

    def rolling_backtest(self, trials: int = 50) -> Dict:
        """滚动回测 (v2: 高效版，不复建分析器)"""
        if len(self.history_data) < trials + 10:
            trials = max(20, len(self.history_data) - 10)

        start = len(self.history_data) - trials

        front_hit_ge2 = front_hit_ge3 = front_hit_ge4 = 0
        back_hit_ge1 = back_hit_ge2 = 0

        # 保存当前状态
        saved_data = list(self.history_data)
        saved_stats = dict(self.statistics) if self.statistics else {}

        for i in range(start, len(self.history_data)):
            # 只用前i期数据
            self.history_data = list(saved_data[i:])  # 从第i期倒序
            self.update_statistics()

            actual = saved_data[i]
            actual_front = actual['front']
            actual_back = actual['back']

            front_ranking = self.get_ensemble_ranking(is_front=True)
            back_ranking = self.get_ensemble_ranking(is_front=False)

            front_top5 = [r['number'] for r in front_ranking[:5]]
            back_top3 = [r['number'] for r in back_ranking[:3]]

            front_common = set(actual_front) & set(front_top5)
            if len(front_common) >= 2:
                front_hit_ge2 += 1
            if len(front_common) >= 3:
                front_hit_ge3 += 1
            if len(front_common) >= 4:
                front_hit_ge4 += 1

            back_common = set(actual_back) & set(back_top3)
            if len(back_common) >= 1:
                back_hit_ge1 += 1
            if len(back_common) >= 2:
                back_hit_ge2 += 1

        # 恢复原始状态
        self.history_data = saved_data
        self.statistics = saved_stats

        n = trials
        return {
            'trials': n,
            'front_ge2_rate': front_hit_ge2 / n,
            'front_ge3_rate': front_hit_ge3 / n,
            'front_ge4_rate': front_hit_ge4 / n,
            'back_ge1_rate': back_hit_ge1 / n,
            'back_ge2_rate': back_hit_ge2 / n,
        }

    def rank_model(self, top_n: int = 10, weights: Dict = None) -> Tuple[List, List]:
        """排名模型 - Top-N排序 (v2: 统一权重)"""
        if weights is None:
            weights = FEATURE_WEIGHTS

        # 前区排名
        front_scores = []
        for num in FRONT_NUMBERS:
            features = self._calculate_feature_score(num, is_front=True)
            total = sum(features.get(k, 0) * weights.get(k, 0) for k in weights)
            front_scores.append((num, total, features))

        # 后区排名 (v2: 使用统一权重子集)
        back_weights = {
            'frequency': weights.get('frequency', 0.22),
            'gap': weights.get('gap', 0.22),
            'trend': weights.get('trend', 0.10),
            'zone': weights.get('zone', 0.05),
        }
        back_weights_norm = sum(back_weights.values()) or 1
        back_scores = []
        for num in BACK_NUMBERS:
            features = self._calculate_feature_score(num, is_front=False)
            total = sum(features.get(k, 0) * back_weights.get(k, 0) for k in back_weights) / back_weights_norm
            back_scores.append((num, total, features))

        front_ranked = sorted(front_scores, key=lambda x: -x[1])[:top_n]
        back_ranked = sorted(back_scores, key=lambda x: -x[1])[:min(top_n, 6)]

        return front_ranked, back_ranked

    # ==================== 特征贡献度分析 ====================

    def feature_contribution(self) -> Dict[str, Any]:
        """计算各特征对每个号码的贡献度"""
        return {
            'weights': FEATURE_WEIGHTS,
            'description': {
                'frequency': '频率特征 - 时间衰减加权的号码出现频率',
                'gap': '遗漏特征 - 基于遗漏期数和标准差的回补概率',
                'position': '位置特征 - 号码在前区各位置的分布',
                'road': '012路特征 - 号码按3取模的分布',
                'sum': '和值特征 - 与平均和值的相关性',
                'trend': '趋势特征 - 近期的升温降温方向',
                'zone': '区间特征 - 号码在三个区间的分布平衡度',
            }
        }

    # ==================== 动态权重调整 ====================

    def dynamic_weight_adjustment(self, backtest_results: List[Dict]) -> Dict[str, float]:
        """根据回测结果动态调整特征权重"""
        if not backtest_results:
            return FEATURE_WEIGHTS.copy()

        feature_scores = defaultdict(float)

        for result in backtest_results:
            predicted = result.get('predicted', [])
            actual = result.get('actual', [])
            features = result.get('features', {})

            for num in actual:
                if num in predicted:
                    for feature, score in features.get(str(num), {}).items():
                        feature_scores[feature] += score

        total = sum(feature_scores.values()) if feature_scores else 1
        new_weights = {k: v / total for k, v in feature_scores.items()}

        for feature in FEATURE_WEIGHTS:
            new_weights.setdefault(feature, FEATURE_WEIGHTS[feature] * 0.1)

        return new_weights

    # ==================== 周期与状态识别 ====================

    def identify_cycles(self) -> Dict[str, Dict]:
        """周期与状态识别 (v2: 使用衰减频率)"""
        stats = self.statistics
        if not stats:
            return {}

        front_status = {}
        back_status = {}

        decayed_front = stats.get('front_frequency_decayed', stats['front_frequency'])
        decayed_back = stats.get('back_frequency_decayed', stats['back_frequency'])

        front_avg_freq = sum(decayed_front.values()) / len(FRONT_NUMBERS)
        back_avg_freq = sum(decayed_back.values()) / len(BACK_NUMBERS)

        for num in FRONT_NUMBERS:
            freq = decayed_front.get(num, 0)
            gap = stats['front_current_gaps'].get(num, stats['front_avg_gap'])
            avg_gap = stats['front_avg_gap']

            if freq > front_avg_freq * 1.2:
                status = '热门'
            elif freq < front_avg_freq * 0.8:
                status = '冷门'
            else:
                status = '稳定'

            if avg_gap > 0 and gap < avg_gap * 0.94:
                trend = '升温'
            elif avg_gap > 0 and gap > avg_gap * 1.05:
                trend = '降温'
            else:
                trend = '平稳'

            front_status[num] = {
                'status': status,
                'trend': trend,
                'frequency': freq,
                'gap': gap
            }

        for num in BACK_NUMBERS:
            freq = decayed_back.get(num, 0)
            gap = stats['back_current_gaps'].get(num, stats['back_avg_gap'])
            avg_gap = stats['back_avg_gap']

            if freq > back_avg_freq * 1.15:
                status = '热门'
            elif freq < back_avg_freq * 0.85:
                status = '冷门'
            else:
                status = '稳定'

            if avg_gap > 0 and gap < avg_gap * 0.96:
                trend = '升温'
            elif avg_gap > 0 and gap > avg_gap * 1.03:
                trend = '降温'
            else:
                trend = '平稳'

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

    # ==================== 多模型集成投票 (v2: 增加二阶马尔可夫) ====================

    def _model_bayesian(self, top_n: int = 8) -> List[int]:
        """贝叶斯模型 (使用衰减频率做先验)"""
        stats = self.statistics
        if not stats:
            return []

        freq = stats.get('front_frequency_decayed', stats['front_frequency'])
        total_weight = sum(freq.values()) if freq else 1

        scores = {}
        for num in FRONT_NUMBERS:
            f = freq.get(num, 0)
            scores[num] = (f + 0.5) / (total_weight + len(FRONT_NUMBERS) * 0.5)

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
        """马尔可夫链模型 (v2: 分位转移)"""
        if len(self.history_data) < 3:
            return []

        # 构建位置感知转移矩阵 (每个位置独立)
        pos_transition = [defaultdict(lambda: defaultdict(int)) for _ in range(5)]

        for i in range(len(self.history_data) - 1):
            curr = self.history_data[i]['front']
            prev = self.history_data[i + 1]['front']
            for pos in range(5):
                pos_transition[pos][prev[pos]][curr[pos]] += 1

        # 基于最近一期预测
        recent = self.history_data[0]['front']
        scores = defaultdict(float)

        for pos in range(5):
            transitions = pos_transition[pos].get(recent[pos], {})
            total_t = sum(transitions.values()) or 1
            for next_num, count in transitions.items():
                scores[next_num] += count / total_t * 2.0  # 位置权重2x

        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

    def _model_markov2(self, top_n: int = 8) -> List[int]:
        """二阶马尔可夫链模型 (基于前2期)"""
        if len(self.history_data) < 4:
            return []

        # 二阶转移: (last2[0], last2[1]) -> next
        pos_transition2 = [defaultdict(lambda: defaultdict(int)) for _ in range(5)]

        for i in range(len(self.history_data) - 2):
            curr = self.history_data[i]['front']
            last1 = self.history_data[i + 1]['front']
            last2 = self.history_data[i + 2]['front']
            for pos in range(5):
                key = (last2[pos], last1[pos])
                pos_transition2[pos][key][curr[pos]] += 1

        recent = self.history_data[0]['front']
        recent_prev = self.history_data[1]['front'] if len(self.history_data) > 1 else recent
        scores = defaultdict(float)

        for pos in range(5):
            key = (recent_prev[pos], recent[pos])
            transitions = pos_transition2[pos].get(key, {})
            total_t = sum(transitions.values()) or 1
            for next_num, count in transitions.items():
                scores[next_num] += count / total_t * 2.5  # 二阶位置权重2.5x

        # 如果没有匹配的转移，降级到一阶
        if not scores:
            return self._model_markov(top_n)

        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

    def multi_model_voting(self, front_n: int = 5, back_n: int = 2, n_votes: int = 3) -> Dict:
        """多模型集成投票 (v2: 6个模型 + 二阶马尔可夫)"""
        # 前区投票 (6个模型)
        models = [
            self._model_bayesian(top_n=12),
            self._model_hot(top_n=12),
            self._model_cold(top_n=12),
            self._model_rank(top_n=12),
            self._model_markov(top_n=12),
            self._model_markov2(top_n=12),
        ]
        model_weights = [1.0, 0.8, 0.7, 1.2, 1.0, 1.0]  # 排名模型权重最高

        votes = defaultdict(float)
        for model_idx, model_result in enumerate(models):
            mw = model_weights[model_idx]
            for rank, num in enumerate(model_result):
                weight = (1.0 - (rank / max(len(model_result), 1))) * mw
                votes[num] += weight

        front_candidates = sorted(votes.items(), key=lambda x: -x[1])[:front_n * 2]
        front_selected = [num for num, _ in front_candidates[:front_n]]

        # 后区投票 (使用统一权重子集)
        back_votes = defaultdict(float)
        back_freq_decayed = self.statistics.get('back_frequency_decayed', self.statistics.get('back_frequency', {}))
        back_gaps = self.statistics.get('back_current_gaps', {})

        for num in BACK_NUMBERS:
            back_votes[num] = back_freq_decayed.get(num, 0) * 0.6 + (
                1.0 / (back_gaps.get(num, 1) + 1)
            ) * 0.4

        back_selected = [num for num, _ in sorted(back_votes.items(), key=lambda x: -x[1])[:back_n]]

        cycles = self.identify_cycles()

        return {
            'front': front_selected,
            'back': back_selected,
            'front_candidates': [{'number': num, 'score': round(score, 3)} for num, score in front_candidates],
            'back_candidates': [{'number': num, 'score': round(back_votes[num], 3)} for num in BACK_NUMBERS],
            'front_votes': {num: round(v, 3) for num, v in votes.items()},
            'cycle_info': cycles,
            'hot_front': cycles.get('hot_front', []),
            'cold_front': cycles.get('cold_front', []),
            'hot_back': cycles.get('hot_back', []),
            'cold_back': cycles.get('cold_back', []),
        }

    # ==================== 约束推荐生成器 ====================

    def _score_based_select(self, candidates: List[int], count: int,
                            is_front: bool = True) -> List[int]:
        """基于评分的约束选择 (替代 random.sample)"""
        stats = self.statistics
        if not stats or len(candidates) < count:
            return sorted(candidates[:count])

        # 获取每个候选的评分
        scored = []
        for num in candidates:
            features = self._calculate_feature_score(num, is_front)
            score = sum(
                features.get(k, 0) * FEATURE_WEIGHTS.get(k, 0)
                for k in FEATURE_WEIGHTS
            )
            scored.append((num, score))

        scored.sort(key=lambda x: -x[1])

        # 贪心选择: 每次选最高分+约束检查
        selected = []
        for num, score in scored:
            if len(selected) >= count:
                break
            selected.append(num)

        selected.sort()

        # 约束检查与修复
        if is_front and len(selected) == 5:
            selected = self._apply_front_constraints(selected, scored)

        return selected

    def _apply_front_constraints(self, selected: List[int],
                                  all_scored: List[Tuple[int, float]]) -> List[int]:
        """对前区推荐应用约束 (奇偶平衡+区间覆盖+连号控制)"""
        # 1. 连号检查: 最多允许1对连号
        cp = sum(1 for i in range(len(selected)-1) if selected[i+1] - selected[i] == 1)
        if cp > MAX_CONSECUTIVE_IN_RECOMMEND:
            # 尝试替换: 找非连号的次高分候选
            used = set(selected)
            backup = [(n, s) for n, s in all_scored if n not in used][:3]
            for i in range(len(selected)-1):
                if selected[i+1] - selected[i] == 1 and backup:
                    selected[i+1] = backup[0][0]
                    backup = backup[1:]
                    selected.sort()
                    break

        # 2. 区间覆盖: 至少覆盖2个区间
        zones_present = set()
        for n in selected:
            if n <= 12:
                zones_present.add(1)
            elif n <= 24:
                zones_present.add(2)
            else:
                zones_present.add(3)

        if len(zones_present) < ZONE_COVERAGE_MIN:
            missing_zones = {1, 2, 3} - zones_present
            used = set(selected)
            for z in missing_zones:
                z_nums = [(n, s) for n, s in all_scored
                          if n not in used and (
                              (z == 1 and n <= 12) or
                              (z == 2 and 13 <= n <= 24) or
                              (z == 3 and n >= 25)
                          )]
                if z_nums:
                    # 替换最低分的一个号码
                    selected.sort()
                    # 找最低分且可以被替换的
                    min_score_idx = None
                    min_score = float('inf')
                    for idx, n in enumerate(selected):
                        fs = all_scored[0][1]  # default
                        for ns, ss in all_scored:
                            if ns == n:
                                fs = ss
                                break
                        # 检查当前区是否有多余
                        if n <= 12:
                            z_curr = 1
                        elif n <= 24:
                            z_curr = 2
                        else:
                            z_curr = 3
                        # 如果当前区还有其他号码，可以替换
                        same_zone_count = sum(
                            1 for x in selected
                            if (x <= 12 and z_curr == 1) or
                            (13 <= x <= 24 and z_curr == 2) or
                            (x >= 25 and z_curr == 3)
                        )
                        if same_zone_count > 1 and fs < min_score:
                            min_score = fs
                            min_score_idx = idx
                    if min_score_idx is not None:
                        selected[min_score_idx] = z_nums[0][0]
                        zones_present.add(z)
            selected.sort()

        # 3. 奇偶平衡
        odd = sum(1 for n in selected if n % 2 == 1)
        even = 5 - odd
        if abs(odd - even) > ODD_PARITY_TOLERANCE + 1:  # 允许3:2或2:3
            used = set(selected)
            needed = 'even' if odd > 3 else 'odd'
            backup = [(n, s) for n, s in all_scored
                      if n not in used and (
                          (needed == 'even' and n % 2 == 0) or
                          (needed == 'odd' and n % 2 == 1)
                      )]
            if backup:
                # 替换一个多余的奇数/偶数
                target_parity = 1 if needed == 'odd' else 0
                for idx, n in enumerate(selected):
                    if n % 2 == target_parity ^ 1:  # 多余的
                        selected[idx] = backup[0][0]
                        selected.sort()
                        break

        # 4. 大小平衡
        small = sum(1 for n in selected if n <= 17)
        if small < SIZE_BALANCE_RANGE[0]:
            used = set(selected)
            backup = [(n, s) for n, s in all_scored if n not in used and n <= 17]
            if backup:
                # 找一个大的替换
                for idx, n in enumerate(selected):
                    if n > 17:
                        selected[idx] = backup[0][0]
                        backup = backup[1:]
                        selected.sort()
                        break
        elif small > SIZE_BALANCE_RANGE[1]:
            used = set(selected)
            backup = [(n, s) for n, s in all_scored if n not in used and n > 17]
            if backup:
                for idx, n in enumerate(selected):
                    if n <= 17:
                        selected[idx] = backup[0][0]
                        backup = backup[1:]
                        selected.sort()
                        break

        return selected

    def generate_recommendation(self, method: str = 'balanced') -> Dict:
        """生成推荐号码 (v2: 基于评分+约束选择)"""
        if method == 'hot':
            hot_front = [num for num, _ in self.statistics.get('hot_front', [])[:12]]
            hot_back = [num for num, _ in self.statistics.get('hot_back', [])[:6]]
            front = self._score_based_select(hot_front, 5, is_front=True)
            back = self._score_based_select(hot_back, 2, is_front=False)
        elif method == 'cold':
            cold_front = [num for num, _ in self.statistics.get('cold_front', [])[:12]]
            cold_back = [num for num, _ in self.statistics.get('cold_back', [])[:6]]
            front = self._score_based_select(cold_front, 5, is_front=True)
            back = self._score_based_select(cold_back, 2, is_front=False)
        elif method == 'rank':
            front_ranked, back_ranked = self.rank_model(top_n=15)
            front_candidates = [num for num, _, _ in front_ranked[:12]]
            back_candidates = [num for num, _, _ in back_ranked[:6]]
            front = self._score_based_select(front_candidates, 5, is_front=True)
            back = self._score_based_select(back_candidates, 2, is_front=False)
        else:
            # 平衡模式 (集成投票)
            result = self.multi_model_voting(front_n=12, back_n=6)
            front_candidates = result['front'][:10]
            back_candidates = result['back'][:5]
            front = self._score_based_select(front_candidates, 5, is_front=True)
            back = self._score_based_select(back_candidates, 2, is_front=False)

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
        total_front_matched = 0
        total_back_matched = 0

        saved_data = list(self.history_data)
        saved_stats = dict(self.statistics) if self.statistics else {}

        for i in range(test_periods):
            test_data = list(saved_data[i + 1:])
            self.history_data = test_data
            self.update_statistics()

            pred = self.generate_recommendation(method)
            front_pred = set(pred['front'])
            back_pred = set(pred['back'])

            actual = saved_data[0]  # saved_data[0] wasn't in test_data
            # Actually, we need the actual result of period i
            actual = saved_data[i]
            front_actual = set(actual['front'])
            back_actual = set(actual['back'])

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

        # 恢复
        self.history_data = saved_data
        self.statistics = saved_stats

        return {
            'method': method,
            'test_periods': test_periods,
            'total_matched': 0,  # full match保持兼容
            'front_accuracy': total_front_matched / (test_periods * 5),
            'back_accuracy': total_back_matched / (test_periods * 2),
            'detailed_results': results
        }

    # ==================== 动态抓取开奖号码 ====================

    def fetch_latest_results(self, count: int = 10, force_refresh: bool = False) -> Dict:
        """动态抓取最新开奖号码（带缓存，每天只抓取一次）

        尝试从多个数据源抓取最新的大乐透开奖结果，如果网络不可用则返回模拟数据。

        Args:
            count: 要抓取的期数（默认10期）
            force_refresh: 是否强制刷新缓存

        Returns:
            包含抓取结果和状态信息的字典
        """
        try:
            if not force_refresh and is_cache_valid('lottery'):
                log.info("大乐透使用缓存数据")
                return {
                    'success': True,
                    'source': 'cache',
                    'count': min(count, len(self.history_data)),
                    'message': '使用缓存数据',
                    'latest_issue': self.history_data[0]['issue'] if self.history_data else None,
                    'results': self.get_recent_results(count)
                }

            results = self._fetch_from_web(count)

            if results:
                self._update_with_fetched(results)
                save_cached_data('lottery', self.history_data)

                return {
                    'success': True,
                    'source': 'web',
                    'count': len(results),
                    'message': f'成功抓取 {len(results)} 期数据',
                    'latest_issue': results[0]['issue'] if results else None,
                    'results': results[:count]
                }
            else:
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

            sources = [
                ('https://cp.ip138.com/daletou/', 'ip138'),
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

            return []
        except ImportError:
            return []

    def _parse_web_data(self, data: str, source: str) -> List[Dict]:
        """解析网络抓取的数据"""
        results = []

        try:
            import re

            if source == 'ip138':
                table_pattern = r'<table[^>]*>(.*?)</table>'
                table_matches = re.findall(table_pattern, data, re.DOTALL)

                for table_content in table_matches:
                    if 'icon-redball' in table_content and 'icon-blueball' in table_content:
                        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL)

                        for row in rows[1:]:
                            issue_match = re.search(r'<td[^>]*><span>(\d{7})</span></td>', row)
                            date_match = re.search(r'<td[^>]*>\s*<span>(\d{2}-\d{2})</span>\s*</td>', row)
                            red_balls = re.findall(r'icon-redball[^>]*>(\d+)</span>', row)
                            blue_balls = re.findall(r'icon-blueball[^>]*>(\d+)</span>', row)

                            if issue_match and len(red_balls) >= 5 and len(blue_balls) >= 2:
                                issue = issue_match.group(1)
                                date_str = date_match.group(1) if date_match else ''
                                front = sorted([int(n) for n in red_balls[:5]])
                                back = sorted([int(n) for n in blue_balls[:2]])

                                if all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back):
                                    if not any(r['issue'] == issue for r in results):
                                        if date_str:
                                            year = issue[:4]
                                            full_date = f'{year}-{date_str}'
                                        else:
                                            full_date = ''
                                        results.append({
                                            'issue': issue,
                                            'front': front,
                                            'back': back,
                                            'date': full_date
                                        })

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
                try:
                    json_data = json.loads(data)
                    if isinstance(json_data, list):
                        for item in json_data[:20]:
                            if isinstance(item, dict):
                                issue = item.get('issue') or item.get('period') or item.get('qihao') or ''
                                front_str = item.get('front') or item.get('red') or item.get('red_ball') or ''
                                back_str = item.get('back') or item.get('blue') or item.get('blue_ball') or ''

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

            if not results:
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

        fetched_map = {r['issue']: r for r in fetched_results}
        fetched_issues = sorted(fetched_map.keys(), reverse=True)
        earliest_fetched = fetched_issues[-1]

        preserved_local = []
        for result in self.history_data:
            if result['issue'] < earliest_fetched:
                preserved_local.append(result)

        merged_results = sorted(fetched_results, key=lambda x: x['issue'], reverse=True) + preserved_local

        self.history_data = merged_results
        self.update_statistics()
        self.save_history()
        log.info(f'成功更新历史数据，共 {len(fetched_results)} 期来自网络')

    def get_statistics(self) -> Dict:
        """获取统计数据"""
        return self.statistics

    def get_recent_results(self, count: int = 10) -> List[Dict]:
        """获取最近开奖结果"""
        return self.history_data[:count]


# 全局分析器实例
_lottery_analyzer = None

def get_lottery_analyzer() -> LotteryAnalyzer:
    """获取大乐透分析器实例"""
    global _lottery_analyzer
    if _lottery_analyzer is None:
        _lottery_analyzer = LotteryAnalyzer()
    return _lottery_analyzer


def run_prediction(force_refresh=False):
    """运行大乐透预测，返回 JSON 可序列化 dict。

    Args:
        force_refresh: 是否强制刷新缓存（默认 False，使用缓存）
    """
    global _prediction_cache, _cache_time

    # 检查模块级内存缓存（按自然天判断）
    if not force_refresh and _prediction_cache is not None:
        if _is_today_cache(_cache_time):
            elapsed = time.time() - _cache_time
            log.info(f"使用今日缓存数据（缓存时间：{elapsed:.1f}秒前）")
            return _prediction_cache
        else:
            log.info("缓存已过期（非今日数据），重新计算")

    try:
        analyzer = get_lottery_analyzer()

        # 抓取最新开奖数据
        analyzer.fetch_latest_results(count=20, force_refresh=force_refresh)

        # 获取统计数据
        stats = analyzer.get_statistics()
        recent = analyzer.get_recent_results(10)

        # 滚动回测
        backtest = analyzer.rolling_backtest(trials=50)

        # 多模型集成投票
        voting = analyzer.multi_model_voting()

        # 多种方法推荐
        recommendations = {}
        for method in ['balanced', 'hot', 'cold', 'rank']:
            rec = analyzer.generate_recommendation(method)
            recommendations[method] = rec

        result = {
            'statistics': stats,
            'recent_results': recent,
            'backtest': backtest,
            'voting': voting,
            'recommendations': recommendations,
        }

        # 保存到模块级内存缓存
        _prediction_cache = result
        _cache_time = time.time()
        log.info("大乐透预测结果已缓存")

        return result
    except Exception:
        log.error('大乐透预测失败', exc_info=True)
        return {'error': '大乐透预测失败'}


if __name__ == '__main__':
    analyzer = get_lottery_analyzer()

    print("=== 大乐透分析器 (v2) ===")
    stats = analyzer.get_statistics()
    print(f"总期数: {stats.get('total_issues', 0)}")

    # 新增分析维度
    print("\n【AC值分析】")
    ac = stats.get('ac_analysis', {})
    print(f"  平均AC值: {ac.get('avg_ac', 0):.2f}")
    print(f"  常见AC值: {ac.get('most_common_ac', [])}")

    print("\n【连号分析】")
    ca = stats.get('consecutive_analysis', {})
    print(f"  含连号比例: {ca.get('pct_with_consecutive', 0):.1%}")

    print("\n【重号分析】")
    da = stats.get('duplicate_analysis', {})
    print(f"  平均重号数: {da.get('avg_duplicates', 0):.2f}")
    print(f"  有重号比例: {da.get('pct_has_duplicate', 0):.1%}")

    print("\n【和值趋势】")
    st = stats['sum_analysis'].get('trend', {})
    print(f"  方向: {st.get('direction', 'N/A')}")
    print(f"  5期MA斜率: {st.get('ma5_slope', 0)}")

    print("\n【升温降温轨迹 (Top5上升)】")
    traj = stats.get('temperature_trajectory', {})
    rising = sorted(
        [(k, v) for k, v in traj.items() if v.get('direction') == 'rising'],
        key=lambda x: x[1]['recent_hits'], reverse=True
    )[:5]
    for num, info in rising:
        print(f"  {num:02d}: {info['direction']} (近期{info['recent_hits']}次 vs 前期{info['prior_hits']}次)")

    print("\n【降温轨迹 (Top5下降)】")
    falling = sorted(
        [(k, v) for k, v in traj.items() if v.get('direction') == 'falling'],
        key=lambda x: x[1]['prior_hits'], reverse=True
    )[:5]
    for num, info in falling:
        print(f"  {num:02d}: {info['direction']} (近期{info['recent_hits']}次 vs 前期{info['prior_hits']}次)")

    # 排名模型
    front_ranked, back_ranked = analyzer.rank_model(top_n=10)
    print("\n前区排名 Top-10:")
    for num, score, features in front_ranked[:10]:
        print(f"  {num:02d}: {score:.4f}")

    print("\n后区排名 Top-6:")
    for num, score, features in back_ranked[:6]:
        print(f"  {num:02d}: {score:.4f}")

    # 集成投票
    print("\n=== 多模型集成投票推荐 (含二阶马尔可夫) ===")
    result = analyzer.multi_model_voting()
    print(f"前区推荐: {[f'{n:02d}' for n in result['front']]}")
    print(f"后区推荐: {[f'{n:02d}' for n in result['back']]}")

    # 约束推荐
    print("\n=== 约束推荐 ===")
    for method in ['balanced', 'hot', 'cold', 'rank']:
        rec = analyzer.generate_recommendation(method)
        print(f"  {method}: 前区{[f'{n:02d}' for n in rec['front']]} + 后区{[f'{n:02d}' for n in rec['back']]}")
