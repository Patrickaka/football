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
import threading
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# 导入公共日志模块
from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common.data_cache import cached_fetch, is_cache_valid, save_cached_data
from ..common import repositories
from ..common import kv_store

log = setup_logger('lottery')

# ===================== 常量配置 =====================
FRONT_NUMBERS = list(range(1, 36))  # 前区号码 01-35
BACK_NUMBERS = list(range(1, 13))   # 后区号码 01-12

# 统一特征权重 (v3.2: 连续平滑评分+均值回归贝叶斯)
FEATURE_WEIGHTS = {
    'frequency': 0.10,      # 频率偏离度 — 超热号反而降分
    'gap': 0.14,            # 遗漏偏离度 — 当前遗漏偏离期望越大越值得选
    'position': 0.14,       # 位置特征
    'road': 0.10,           # 012路特征
    'sum': 0.12,            # 和值特征
    'trend': 0.08,          # 升温降温趋势
    'zone': 0.08,           # 区间分布特征
    'repeat': 0.0,          # 重号: 真实重号率0.73≈随机期望0.71, 无预测信号; 权重0消除"照搬上期"假象
    'adjacent': 0.08,       # 邻号: 真实邻号率80%、前区均1.31个, 轻微真实信号, 保留校准
}

# 时间衰减因子 (最近一期权重1.0，20期前≈0.19，50期前≈0.016)
TIME_DECAY_FACTOR = 0.92
MIN_REAL_HISTORY_FOR_RANKING = 80  # 降低阈值: 120期真实数据足够支撑统计模型
LOTTERY_PREDICTOR_VERSION = "dlt-v4.0-primary-rotation"
FULL_HISTORY_FETCH_COUNT = 100
MIN_FULL_HISTORY_ISSUES = 500
ROLLING_BACKTEST_TRIALS = 30
FEATURE_BACKTEST_TRIALS = 40
ML_BACKTEST_TRIALS = 10

# v3.6: 去锚定校准 — 消融+600期回测确认
# repeat 权重 0.27→0.0: 真实重号率0.73≈随机期望0.71, 重号本身无可预测性;
#   0.27 仅造成"前区含3.99个上期码"的过度锚定(真实0.73), 去重号后锚定降至0.71且准确率不变(仍在随机噪声带)
# adjacent 权重 0.05→0.08: 邻号是真实信号(前区均1.31个/覆盖80%), 适度保留让推荐形态自然
FEATURE_WEIGHTS.update({
    'frequency': 0.05,     # 消融: 关掉降2% → 微弱贡献, 保持v3.2
    'gap': 0.08,           # 消融: 关掉零影响 → 纯噪声确认, 保持v3.2
    'position': 0.05,      # 消融: 关掉降2% → 微弱贡献, 保持
    'road': 0.12,          # 消融: 关掉降8% → 重要, 但提升反而降≥3, 保持v3.2
    'sum': 0.15,           # 消融: 关掉降8% → 第二重要, 保持
    'trend': 0.03,         # 消融: 关掉降2% → 微弱贡献, 保持
    'zone': 0.20,          # 消融: 关掉降10% → 最关键, 保持(已足够高)
    'repeat': 0.0,         # v3.6: 0.27→0.0, 重号=纯噪声, 去锚定
    'adjacent': 0.08,      # v3.6: 0.05→0.08, 保留真实邻号信号
})

# ===================== 公平摇奖理论基准 (诚实标注) =====================
# 大乐透为均匀摇奖: 前区5/35、后区2/12。模型对「当期实际号码」的预测无法
# 系统性超过下列随机基准——任何回测中明显高于基准的结果都是小样本噪声/过拟合。
# 推导(超几何分布):
#   前区命中(Top5对实际5): P(>=2)=1-C(30,5)/C(35,5)-5C(30,4)/C(35,5)=0.1389
#                          P(>=3)=[C(5,3)C(30,2)+C(5,4)·30+1]/C(35,5)=0.0139
#   后区命中(Top3对实际2): P(>=1)=1-C(9,2)/C(12,2)=0.4545
#                          P(>=2)=C(3,2)/C(12,2)=0.0455
RANDOM_BASELINE = {
    'front_ge2_rate': 0.1389,
    'front_ge3_rate': 0.0139,
    'front_ge4_rate': 0.00046,
    'back_ge1_rate': 0.4545,
    'back_ge2_rate': 0.0455,
}

# 后区专用权重（v3.4: 下调 adjacent，避免后区推荐被上期临号锁死）
# 后区仅 12 码，临号天然覆盖面大；adjacent 给 0.30 会主导 Top2，多策略后区趋同。
# v3.6: repeat 0.10→0.0 (后区重号率0.33≈随机, 无信号, 去锚定), adjacent 0.10→0.08
BACK_FEATURE_WEIGHTS = {
    'frequency': 0.10,
    'gap': 0.15,
    'trend': 0.10,
    'road': 0.15,
    'repeat': 0.0,         # v3.6: 去重号锚定
    'adjacent': 0.08,      # v3.6: 适度保留真实邻号信号
    'position': 0.15,
    'sum': 0.15,
}

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


def _needs_full_history_bootstrap(data_quality: Dict[str, Any]) -> bool:
    """生产环境未提交运行时 JSON 时，自动引导全量历史。"""
    warnings = set(data_quality.get('warnings') or [])
    return (
        data_quality.get('using_simulated_data')
        or int(data_quality.get('issues') or 0) < MIN_FULL_HISTORY_ISSUES
        or 'issue_gaps' in warnings
        or 'date_anomalies' in warnings
        or not data_quality.get('ranking_allowed', False)
    )


class LotteryAnalyzer:
    """大乐透分析器"""

    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or data_path('lottery_history.json')
        self.using_simulated_data = False
        self.history_data = self._load_history()
        self.statistics = {}
        self.update_statistics()

    def _load_history(self) -> List[Dict]:
        """加载历史数据"""
        try:
            results = repositories.dlt_load()
            if results:
                self.using_simulated_data = False
                return results
            self.using_simulated_data = True
            return self._generate_simulated_data()
        except Exception:
            self.using_simulated_data = True
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
        repositories.dlt_save(self.history_data)

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

    def assess_data_quality(self) -> Dict[str, Any]:
        """Assess whether the current history is suitable for ranking/backtest use.
        v3: 大乐透期号不连续是正常的(每周一三五开奖)，改用日期连续性判断。
        """
        issues = [str(x.get('issue', '')) for x in self.history_data if x.get('issue')]
        dates = [str(x.get('date', '')) for x in self.history_data if x.get('date')]
        duplicate_issues = len(issues) - len(set(issues))
        issue_gaps = 0

        ordered = list(reversed(issues))
        for prev, curr in zip(ordered, ordered[1:]):
            try:
                prev_year, prev_seq = int(prev[:4]), int(prev[4:])
                curr_year, curr_seq = int(curr[:4]), int(curr[4:])
            except Exception:
                continue
            if curr_year == prev_year and curr_seq - prev_seq != 1:
                issue_gaps += 1
            elif curr_year == prev_year + 1 and curr_seq != 1:
                issue_gaps += 1

        # 日期连续性检查(大乐透每周3期，通常间隔2-4天)
        date_gaps = 0
        date_anomalies = 0
        if len(dates) >= 2:
            import datetime
            date_objs = []
            for d in reversed(dates):
                try:
                    date_objs.append(datetime.datetime.strptime(d, '%Y-%m-%d').date())
                except Exception:
                    continue
            for prev_d, curr_d in zip(date_objs, date_objs[1:]):
                diff = (curr_d - prev_d).days
                if diff > 7:
                    date_gaps += 1
                elif diff < 2:
                    date_anomalies += 1

        date_gap_tolerance = max(3, int(len(dates) * 0.02))
        date_anomaly_tolerance = max(3, int(len(dates) * 0.02))

        warnings = []
        if self.using_simulated_data:
            warnings.append('simulated_data')
        if len(issues) < MIN_REAL_HISTORY_FOR_RANKING:
            warnings.append('history_too_short')
        if duplicate_issues:
            warnings.append('duplicate_issues')
        if issue_gaps > 3:
            warnings.append('issue_gaps')
        if date_gaps > date_gap_tolerance:
            warnings.append('date_gaps')
        if date_anomalies > date_anomaly_tolerance:
            warnings.append('date_anomalies')

        return {
            'issues': len(issues),
            'latest_issue': issues[0] if issues else None,
            'oldest_issue': issues[-1] if issues else None,
            'latest_date': dates[0] if dates else None,
            'oldest_date': dates[-1] if dates else None,
            'duplicate_issues': duplicate_issues,
            'issue_gaps': issue_gaps,
            'date_gaps': date_gaps,
            'date_anomalies': date_anomalies,
            'date_gap_tolerance': date_gap_tolerance,
            'date_anomaly_tolerance': date_anomaly_tolerance,
            'using_simulated_data': self.using_simulated_data,
            'ranking_allowed': (
                not self.using_simulated_data
                and len(issues) >= MIN_REAL_HISTORY_FOR_RANKING
                and duplicate_issues == 0
                and issue_gaps <= 3
                and date_gaps <= date_gap_tolerance
                and date_anomalies <= date_anomaly_tolerance
            ),
            'warnings': warnings,
        }

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
        back_position_freq = [defaultdict(float) for _ in range(2)]

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
        back_road_dist = defaultdict(int)
        back_road_total = [0.0, 0.0, 0.0]
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

            for i, num in enumerate(back):
                back_freq_raw[num] += 1
                back_freq_decayed[num] += decay
                back_position_freq[i][num] += decay
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
                road_total[r] += decay
            road_dist[f'{rc[0]}:{rc[1]}:{rc[2]}'] += 1

            back_rc = [0, 0, 0]
            for n in back:
                r = n % 3
                back_rc[r] += 1
                back_road_total[r] += decay
            back_road_dist[f'{back_rc[0]}:{back_rc[1]}:{back_rc[2]}'] += 1

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
        temp_trajectory = self._calc_temperature_trajectory(is_front=True)
        back_temp_trajectory = self._calc_temperature_trajectory(is_front=False)

        # ---- 重号率 ----
        dup_rate = sum(duplicate_counts) / len(duplicate_counts) if duplicate_counts else 0

        return {
            'total_issues': total,
            # 频率 (兼容前端: 保留原始计数作为主key，decayed版本供评分使用)
            'front_frequency': dict(front_freq_raw),
            'back_frequency': dict(back_freq_raw),
            'front_frequency_decayed': dict(front_freq_decayed),
            'back_frequency_decayed': dict(back_freq_decayed),
            # 衰减权重和 W=Σ(0.92^idx)。前/后区相同(每期5前2后号，故Σ衰减频率=5W=2W')。
            # 频率偏离度评分须以 W 归一化(而非期数 total)，否则 deviation_ratio 被严重压缩→特征退化。
            'decay_weight_sum': (sum(front_freq_decayed.values()) / 5.0) if front_freq_decayed else 1.0,
            'position_frequency': [dict(pf) for pf in position_freq],
            'back_position_frequency': [dict(pf) for pf in back_position_freq],
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
            'back_road_analysis': {
                '0': {}, '1': {}, '2': {},
                'distribution': dict(back_road_dist),
                'total': {0: back_road_total[0], 1: back_road_total[1], 2: back_road_total[2]}
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
            'back_temperature_trajectory': back_temp_trajectory,
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

    def _calc_temperature_trajectory(self, is_front: bool = True) -> Dict:
        """计算升温降温轨迹 (最近N期的热度变化)"""
        if len(self.history_data) < 5:
            return {}

        recent_span = min(10, len(self.history_data))
        trajectory = {}
        numbers = FRONT_NUMBERS if is_front else BACK_NUMBERS
        key = 'front' if is_front else 'back'

        for num in numbers:
            # 分两个窗口: 最近5期 vs 之前5期
            m = min(5, recent_span // 2)
            recent_hits = sum(1 for r in self.history_data[:m] if num in r[key])
            prior_hits = sum(1 for r in self.history_data[m:m*2] if num in r[key])

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
                'current_gap': self._get_current_gap(num, is_front=is_front),
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
        """计算单个号码的各特征得分 (v3.2: 连续平滑函数+均值回归贝叶斯)

        关键改进(vs v3.1硬阈值):
        - 频率偏离度: 反sigmoid连续函数, 消除阶梯跳变
        - 遗漏偏离度: 非对称平滑曲线, 左侧快升右侧缓降
        - 贝叶斯模型: 均值回归修正(不再给热号加分)
        """
        stats = self.statistics
        if not stats:
            return {}

        scores = {}
        total = stats['total_issues']

        # ---- 频率偏离度得分 (v3.2: 连续平滑函数替代硬阈值) ----
        # 期望频率: 前区每号5/35≈14.3%, 后区每号2/12≈16.7%
        expected_rate = 5.0 / 35.0 if is_front else 2.0 / 12.0
        freq_raw = stats['front_frequency' if is_front else 'back_frequency']
        freq_decayed = stats.get('front_frequency_decayed' if is_front else 'back_frequency_decayed', freq_raw)

        # 实际出现率(衰减版): 以衰减权重和 W 归一化，使其与每期期望率 expected_rate 同尺度。
        # (历史 bug: 误用 total 期数归一化，actual_rate 被压缩 ~total/W 倍，所有号码恒判为冷号→特征退化)
        decay_w = stats.get('decay_weight_sum', max(total, 1))
        actual_rate = freq_decayed.get(num, 0) / max(decay_w, 1e-9)
        # 偏离度: 实际率 / 期望率
        deviation_ratio = actual_rate / max(expected_rate, 0.01)

        # v3.2: 分段连续评分 — 冷号线性加分, 热号指数降分
        # 核心原则: deviation=1.0(期望频率) → 中性分0.55
        # deviation<1.0(冷号) → 从0.55线性升到0.70(轻微回补预期)
        # deviation>1.0(热号) → 从0.55指数降到0.15(回归均值预期)
        # 与v3硬阈值对比: 0.5→0.625(vs0.70), 1.0→0.55(vs0.55), 1.5→0.22(vs0.20)
        if deviation_ratio <= 1.0:
            # 冷号到中性: 线性加分
            scores['frequency'] = 0.55 + 0.15 * (1.0 - deviation_ratio)
        else:
            # 热号: 指数衰减(越热越低)
            scores['frequency'] = max(0.15, 0.55 * math.exp(-1.8 * (deviation_ratio - 1.0)))

        # ---- 遗漏偏离度得分 (v3.2: 连续平滑函数替代硬阈值) ----
        gaps = stats['front_current_gaps' if is_front else 'back_current_gaps']
        gap_std = stats.get('front_gap_std' if is_front else 'back_gap_std', {})

        # 期望遗漏: 前区1/p=7期, 后区1/p=6期
        expected_gap = 35.0 / 5.0 if is_front else 12.0 / 2.0
        gap = gaps.get(num, expected_gap)
        std = gap_std.get(num, expected_gap / 2)

        gap_ratio = gap / expected_gap  # 偏离期望遗漏的倍数

        # v3.2: 非对称平滑评分
        # 左侧(刚出现): gap_ratio<1 → 快速从低分升到高分
        # 右侧(超期): gap_ratio>1 → 缓慢从高分降(回补预期仍存在)
        if gap_ratio <= 1.0:
            # 接近期望遗漏: 从0.25线性升到0.85
            scores['gap'] = 0.25 + 0.60 * (gap_ratio ** 0.7)
        else:
            # 超过期望遗漏: 缓慢降分(回补预期衰减但仍有)
            scores['gap'] = 0.85 - 0.45 * (1.0 - math.exp(-(gap_ratio - 1.0) * 0.8))

        # 标准差修正: 遗漏波动越大的号码，回补时机更难预测
        if std > expected_gap * 0.8:
            scores['gap'] *= 0.85

        # ---- 位置得分 ----
        if is_front:
            pos_scores = []
            for i in range(5):
                pos_freq = stats['position_frequency'][i]
                pos_max = max(pos_freq.values()) if pos_freq else 1
                pos_scores.append(pos_freq.get(num, 0) / pos_max)
            scores['position'] = sum(pos_scores) / 5

        # ---- 012路得分 ----
        road = num % 3
        road_data = stats['road_analysis'].get('total', {})
        road_total_val = road_data.get(road, 0)
        road_all_max = max(road_data.values()) if road_data else 1
        scores['road'] = road_total_val / road_all_max if road_all_max > 0 else 0

        # ---- 和值相关性得分 ----
        if is_front:
            sum_avg = stats['sum_analysis']['avg']
            ideal_value = sum_avg / 5
            scores['sum'] = 1.0 - abs(num - ideal_value) / max(FRONT_NUMBERS)

        # ---- 升温降温趋势得分 ----
        trajectory = stats.get('temperature_trajectory', {})
        traj = trajectory.get(num, {})
        if traj.get('direction') == 'rising':
            scores['trend'] = 0.75  # v3: 降低rising权重(被证实区分度弱)
        elif traj.get('direction') == 'falling':
            scores['trend'] = 0.40
        else:
            scores['trend'] = 0.55

        # ---- 区间平衡得分 ----
        if is_front:
            zone_dist = stats.get('zone_analysis', {}).get('distribution', {})
            if num <= 12:
                zone_key_prefix = '1'
            elif num <= 24:
                zone_key_prefix = '2'
            else:
                zone_key_prefix = '3'
            most_common_zone = max(zone_dist.items(), key=lambda x: x[1]) if zone_dist else ('', 0)
            zone_parts = most_common_zone[0].split(':')
            try:
                zone_idx = int(zone_key_prefix) - 1
                zone_count = int(zone_parts[zone_idx]) if zone_idx < len(zone_parts) else 0
                scores['zone'] = min(1.0, zone_count / 3.0)
            except (ValueError, IndexError):
                scores['zone'] = 0.5
        else:
            scores['zone'] = 0.5

        # ---- 重号概率得分 (v3新增, v3.6去权重) ----
        # 注意: 真实重号率仅0.73个/期(≈随机期望0.71), "重号率约60%"指56%的期至少含1个重号。
        # 重号本身无预测性, FEATURE_WEIGHTS['repeat'] 已置0, 故此特征当前不贡献分数。
        if len(self.history_data) >= 1:
            last_front = self.history_data[0].get('front', []) if is_front else self.history_data[0].get('back', [])
            if num in last_front:
                # 是上期号码 — 重号概率高
                # 但需要考虑: 最近几期已重过的号码，再重概率下降
                recent_repeat_count = 0
                for r in self.history_data[:3]:
                    last_r = r.get('front', []) if is_front else r.get('back', [])
                    if num in last_r:
                        recent_repeat_count += 1
                if recent_repeat_count <= 1:
                    scores['repeat'] = 0.80  # 上期号码，重号概率高
                elif recent_repeat_count <= 2:
                    scores['repeat'] = 0.45  # 连续2期出现，再重概率下降
                else:
                    scores['repeat'] = 0.20  # 连续3期+，极不可能继续
            else:
                scores['repeat'] = 0.35  # 不是上期号码，基础分
        else:
            scores['repeat'] = 0.40

        # ---- 邻号概率得分 (v3新增) ----
        # 大乐透邻号率约80%，即下期至少1个号码与上期号码±1
        if len(self.history_data) >= 1:
            last_nums = self.history_data[0].get('front', []) if is_front else self.history_data[0].get('back', [])
            neighbors = set()
            for n in last_nums:
                if n - 1 >= 1:
                    neighbors.add(n - 1)
                if n + 1 <= (35 if is_front else 12):
                    neighbors.add(n + 1)

            if num in neighbors:
                scores['adjacent'] = 0.75  # 是上期号码的邻号
            else:
                scores['adjacent'] = 0.30  # 不是邻号
        else:
            scores['adjacent'] = 0.40

        return scores

    def get_ensemble_ranking(self, is_front: bool = True, top_n: int = 10) -> List[Dict]:
        """获取综合排名 (v3: 前区/后区使用不同权重体系)"""
        numbers = FRONT_NUMBERS if is_front else BACK_NUMBERS
        scores = []

        weights = FEATURE_WEIGHTS if is_front else BACK_FEATURE_WEIGHTS

        for num in numbers:
            feature_scores = self._calculate_feature_score(num, is_front) if is_front else self._calculate_back_feature_score(num)
            if not feature_scores:
                continue

            total_score = sum(
                feature_scores.get(k, 0) * weights.get(k, 0)
                for k in weights
                if k in feature_scores
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
            trials = max(1, len(self.history_data) - 10)

        start = 0

        front_hit_ge2 = front_hit_ge3 = front_hit_ge4 = 0
        back_hit_ge1 = back_hit_ge2 = 0
        evaluated = 0

        # 保存当前状态
        saved_data = list(self.history_data)
        saved_stats = dict(self.statistics) if self.statistics else {}

        for i in range(start, trials):
            # Newest-first history: predict saved_data[i] using only older draws.
            self.history_data = list(saved_data[i + 1:])
            if len(self.history_data) < 10:
                continue
            self.update_statistics()

            actual = saved_data[i]
            actual_front = actual['front']
            actual_back = actual['back']
            evaluated += 1

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

        n = evaluated or 1
        rates = {
            'front_ge2_rate': front_hit_ge2 / n,
            'front_ge3_rate': front_hit_ge3 / n,
            'front_ge4_rate': front_hit_ge4 / n,
            'back_ge1_rate': back_hit_ge1 / n,
            'back_ge2_rate': back_hit_ge2 / n,
        }

        # 诚实对照: 附带公平摇奖随机基准 + 提升量 + 噪声带(±1.96·SE)。
        # |lift| 落在噪声带内即与随机无显著差异——这是公平博弈的预期结果，不应解读为模型有效。
        baseline_cmp = {}
        for key, observed in rates.items():
            base = RANDOM_BASELINE.get(key, 0.0)
            se = math.sqrt(max(base * (1 - base), 1e-9) / n)
            baseline_cmp[key] = {
                'observed': round(observed, 4),
                'random_baseline': round(base, 4),
                'lift': round(observed - base, 4),
                'noise_band': round(1.96 * se, 4),
                'significant': abs(observed - base) > 1.96 * se,
            }

        return {
            'trials': n,
            **rates,
            'random_baseline': RANDOM_BASELINE,
            'baseline_comparison': baseline_cmp,
            'note': '大乐透为公平摇奖，命中率受随机基准约束；lift 落在 noise_band 内属正常，无法据此判定模型优于随机。',
        }

    def rank_model(self, top_n: int = 10, weights: Dict = None) -> Tuple[List, List]:
        """排名模型 - Top-N排序 (v3: 前区/后区使用不同权重)"""
        if weights is None:
            weights = FEATURE_WEIGHTS

        # 前区排名
        front_scores = []
        for num in FRONT_NUMBERS:
            features = self._calculate_feature_score(num, is_front=True)
            total = sum(features.get(k, 0) * weights.get(k, 0) for k in weights if k in features)
            front_scores.append((num, total, features))

        # 后区排名 (v3: 使用专用BACK_FEATURE_WEIGHTS)
        back_scores = []
        for num in BACK_NUMBERS:
            features = self._calculate_back_feature_score(num)
            total = sum(features.get(k, 0) * BACK_FEATURE_WEIGHTS.get(k, 0) for k in BACK_FEATURE_WEIGHTS if k in features)
            back_scores.append((num, total, features))

        front_ranked = sorted(front_scores, key=lambda x: -x[1])[:top_n]
        back_ranked = sorted(back_scores, key=lambda x: -x[1])[:min(top_n, 6)]

        return front_ranked, back_ranked

    # ==================== 特征贡献度分析 ====================

    def feature_contribution(self) -> Dict[str, Any]:
        """计算各特征对每个号码的贡献度 (v3: 新增repeat/adjacent)"""
        return {
            'weights': FEATURE_WEIGHTS,
            'back_weights': BACK_FEATURE_WEIGHTS,
            'description': {
                'frequency': '频率偏离度 — 超热号降分(回归均值), 低频号轻微加分',
                'gap': '遗漏偏离度 — 当前遗漏偏离期望遗漏的评估',
                'position': '位置特征 — 号码在前区各位置的分布',
                'road': '012路特征 — 号码按3取模的分布',
                'sum': '和值特征 — 与平均和值的相关性',
                'trend': '趋势特征 — 近期的升温降温方向(区分度弱,权重低)',
                'zone': '区间特征 — 号码在三个区间的分布平衡度',
            'repeat': '重号概率 — 与上期号码重叠(v3.6: 真实重号0.73≈随机, 权重置0, 无贡献)',
            'adjacent': '邻号概率 — 与上期号码±1的覆盖(真实邻号率80%, 轻微真实信号)',
            }
        }

    # ==================== 动态权重调整 ====================

    def dynamic_weight_adjustment(self, backtest_results: List[Dict] = None) -> Dict[str, float]:
        """根据回测结果动态调整特征权重 (v2.2: 回测驱动优化)

        如果未提供 backtest_results，则自动执行滚动回测来获取数据。
        """
        # 自动执行回测获取数据
        if backtest_results is None:
            backtest_results = self._run_feature_backtest()

        if not backtest_results:
            return FEATURE_WEIGHTS.copy()

        # 统计每个特征在命中/未命中号码中的得分(含平方和，用于估计方差→显著性门控)
        feature_hit_scores = defaultdict(float)
        feature_hit_sq = defaultdict(float)
        feature_miss_scores = defaultdict(float)
        feature_miss_sq = defaultdict(float)
        hit_count = 0
        miss_count = 0

        for result in backtest_results:
            actual_nums = result.get('actual_front', result.get('actual', []))
            predicted_nums = result.get('predicted_front', result.get('predicted', []))
            features_map = result.get('features', {})

            for num in actual_nums:
                num_key = str(num)
                num_features = features_map.get(num_key, {})
                if not num_features:
                    continue
                if num in predicted_nums:
                    # 命中的号码：该号码各特征得分加权累积
                    for feature, score in num_features.items():
                        feature_hit_scores[feature] += score
                        feature_hit_sq[feature] += score * score
                    hit_count += 1
                else:
                    # 未命中的号码：各特征得分作为"噪音"
                    for feature, score in num_features.items():
                        feature_miss_scores[feature] += score
                        feature_miss_sq[feature] += score * score
                    miss_count += 1

        # 过拟合护栏: 命中样本过少时，特征间得分差异不可信(公平摇奖下本就是噪声)，
        # 直接返回基础权重，避免把回测噪声固化进权重(见记忆 lottery3d-fair-game-ceiling)。
        if hit_count < 60 or miss_count < 60:
            return FEATURE_WEIGHTS.copy()

        # 命中/未命中比率作为权重修正因子
        new_weights = {}
        for feature in FEATURE_WEIGHTS:
            base_weight = FEATURE_WEIGHTS[feature]
            hit_avg = feature_hit_scores.get(feature, 0) / hit_count
            miss_avg = feature_miss_scores.get(feature, 0) / miss_count

            # 显著性门控: 均分之差落在 ±1.96·SE 噪声带内 → 无可信信号 → 保持基础权重。
            hit_var = max(feature_hit_sq.get(feature, 0) / hit_count - hit_avg ** 2, 0.0)
            miss_var = max(feature_miss_sq.get(feature, 0) / miss_count - miss_avg ** 2, 0.0)
            se = math.sqrt(hit_var / hit_count + miss_var / miss_count) or 1e-9
            if abs(hit_avg - miss_avg) <= 1.96 * se:
                new_weights[feature] = base_weight
                continue

            # 信号强度 = 命中得分 / (命中得分 + 未命中得分)
            signal_strength = hit_avg / max(hit_avg + miss_avg, 0.001)

            # 新权重 = 基础权重 * (1 + 信号修正)
            adjustment = 0.3  # 修正幅度限制在30%
            adjusted_weight = base_weight * (1 + adjustment * (signal_strength - 0.5))
            new_weights[feature] = max(0.05, adjusted_weight)

        # 归一化
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights

    def _run_feature_backtest(self, trials: int = FEATURE_BACKTEST_TRIALS) -> List[Dict]:
        """执行特征级回测，收集每个号码的特征得分（默认40期，兼顾速度与护栏样本量）"""
        if len(self.history_data) < trials + 10:
            return []

        saved_data = list(self.history_data)
        saved_stats = dict(self.statistics) if self.statistics else {}

        results = []
        for i in range(trials):
            self.history_data = list(saved_data[i + 1:])
            if len(self.history_data) < 10:
                continue
            self.update_statistics()

            actual = saved_data[i]
            actual_front = set(actual['front'])

            # 获取排名模型的特征得分
            front_ranked, _ = self.rank_model(top_n=15)
            predicted_front = [num for num, _, _ in front_ranked[:8]]

            # 收集每个号码的特征得分
            features_map = {}
            for num, score, features in front_ranked:
                features_map[str(num)] = features

            results.append({
                'actual_front': list(actual_front),
                'predicted_front': predicted_front,
                'features': features_map,
            })

        # 恢复
        self.history_data = saved_data
        self.statistics = saved_stats

        return results

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

    def _model_bayesian(self, top_n: int = 18) -> List[int]:
        """贝叶斯模型 (v3.2: 均值回归修正 — 超热号降分，不再与排名模型矛盾)

        基础贝叶斯概率 * 回归因子:
        - deviation_ratio>1(热号): 回归因子<1, 降低概率
        - deviation_ratio<1(冷号): 回归因子>1, 轻微提升概率
        - deviation_ratio≈1(正常): 回归因子≈1, 保持原概率
        """
        stats = self.statistics
        if not stats:
            return []

        freq = stats.get('front_frequency_decayed', stats['front_frequency'])
        total = stats['total_issues']
        total_weight = sum(freq.values()) if freq else 1
        expected_rate = 5.0 / 35.0  # 前区期望频率

        scores = {}
        for num in FRONT_NUMBERS:
            f = freq.get(num, 0)
            # 基础贝叶斯概率(拉普拉斯平滑)
            base_prob = (f + 0.5) / (total_weight + len(FRONT_NUMBERS) * 0.5)
            # 均值回归修正: 热号降分, 冷号轻微加分
            actual_rate = f / max(total, 1)
            deviation_ratio = actual_rate / max(expected_rate, 0.01)
            # 回归因子 = 1 / (1 + 0.6*(deviation-1))
            # deviation=1 → factor=1.0(不变)
            # deviation=1.5 → factor=0.71(热号降29%)
            # deviation=0.5 → factor=1.43(冷号提43%)
            reversion_factor = 1.0 / (1.0 + 0.6 * max(0, deviation_ratio - 1.0))
            if deviation_ratio < 1.0:
                # 冷号: 轻微提升(上限1.5x, 防止极端冷号过度加分)
                reversion_factor = min(1.5, 1.0 + 0.5 * (1.0 - deviation_ratio))
            scores[num] = base_prob * reversion_factor

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

    # ==================== 后区专用模型 (v2.2新增) ====================

    def _model_bayesian_back(self, top_n: int = 8) -> List[int]:
        """后区贝叶斯模型 (v3.2: 均值回归修正)"""
        stats = self.statistics
        if not stats:
            return []

        freq = stats.get('back_frequency_decayed', stats.get('back_frequency', {}))
        total = stats['total_issues']
        total_weight = sum(freq.values()) if freq else 1
        expected_rate = 2.0 / 12.0  # 后区期望频率

        scores = {}
        for num in BACK_NUMBERS:
            f = freq.get(num, 0)
            base_prob = (f + 0.5) / (total_weight + len(BACK_NUMBERS) * 0.5)
            actual_rate = f / max(total, 1)
            deviation_ratio = actual_rate / max(expected_rate, 0.01)
            # 均值回归修正(后区更强的回归预期)
            reversion_factor = 1.0 / (1.0 + 0.8 * max(0, deviation_ratio - 1.0))
            if deviation_ratio < 1.0:
                reversion_factor = min(1.5, 1.0 + 0.6 * (1.0 - deviation_ratio))
            scores[num] = base_prob * reversion_factor

        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

    def _model_repeat_back(self, top_n: int = 8) -> List[int]:
        """后区重号模型 — 与上期后区号码重复的概率更高（2/12≈16.7%）"""
        if len(self.history_data) < 2:
            return []

        last_back = set(self.history_data[0]['back'])
        # 统计历史上后区重号率
        repeat_counts = defaultdict(float)
        for idx, r in enumerate(self.history_data[1:]):
            decay = TIME_DECAY_FACTOR ** idx
            for num in r['back']:
                if num in last_back:
                    repeat_counts[num] += decay

        # 重号加分 + 历史频率基线
        freq = self.statistics.get('back_frequency_decayed', {})
        scores = {}
        for num in BACK_NUMBERS:
            base = freq.get(num, 0)
            repeat_boost = repeat_counts.get(num, 0) * 3  # 重号加权
            scores[num] = base + repeat_boost

        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

    def _model_adjacent_back(self, top_n: int = 8) -> List[int]:
        """后区邻号模型 — 上期后区号码±1出现的概率更高"""
        if len(self.history_data) < 2:
            return []

        last_back = self.history_data[0]['back']
        adjacent_nums = set()
        for num in last_back:
            if num - 1 >= 1:
                adjacent_nums.add(num - 1)
            if num + 1 <= 12:
                adjacent_nums.add(num + 1)

        freq = self.statistics.get('back_frequency_decayed', {})
        scores = {}
        for num in BACK_NUMBERS:
            base = freq.get(num, 0)
            if num in adjacent_nums:
                base *= 1.12  # 邻号微弱加权（原 1.4 过强）
            scores[num] = base

        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

    def _model_markov_back(self, top_n: int = 8) -> List[int]:
        """后区马尔可夫模型（一阶+二阶）"""
        if len(self.history_data) < 3:
            return []

        # 一阶转移（后区2个位置独立）
        pos_transition = [defaultdict(lambda: defaultdict(int)) for _ in range(2)]
        for i in range(len(self.history_data) - 1):
            curr = self.history_data[i]['back']
            prev = self.history_data[i + 1]['back']
            for pos in range(2):
                pos_transition[pos][prev[pos]][curr[pos]] += 1

        recent = self.history_data[0]['back']
        scores = defaultdict(float)

        for pos in range(2):
            transitions = pos_transition[pos].get(recent[pos], {})
            total_t = sum(transitions.values()) or 1
            for next_num, count in transitions.items():
                scores[next_num] += count / total_t * 2.0

        # 二阶转移
        if len(self.history_data) >= 4:
            pos_transition2 = [defaultdict(lambda: defaultdict(int)) for _ in range(2)]
            for i in range(len(self.history_data) - 2):
                curr = self.history_data[i]['back']
                last1 = self.history_data[i + 1]['back']
                last2 = self.history_data[i + 2]['back']
                for pos in range(2):
                    key = (last2[pos], last1[pos])
                    pos_transition2[pos][key][curr[pos]] += 1

            recent_prev = self.history_data[1]['back'] if len(self.history_data) > 1 else recent
            for pos in range(2):
                key = (recent_prev[pos], recent[pos])
                transitions = pos_transition2[pos].get(key, {})
                total_t = sum(transitions.values()) or 1
                for next_num, count in transitions.items():
                    scores[next_num] += count / total_t * 2.5

        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

    def _model_rank_back(self, top_n: int = 8) -> List[int]:
        """后区排名模型（使用BACK_FEATURE_WEIGHTS）"""
        stats = self.statistics
        if not stats:
            return []

        scores = {}
        for num in BACK_NUMBERS:
            feature_scores = self._calculate_back_feature_score(num)
            total = sum(
                feature_scores.get(k, 0) * BACK_FEATURE_WEIGHTS.get(k, 0)
                for k in BACK_FEATURE_WEIGHTS
            )
            scores[num] = total

        return [num for num, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

    def _calculate_back_feature_score(self, num: int) -> Dict[str, float]:
        """后区号码的特征得分 (v3: 频率偏离度反转 + 期望遗漏)"""
        stats = self.statistics
        if not stats:
            return {}

        scores = {}
        total = stats['total_issues']

        # ---- 频率偏离度得分 (v3.2: 分段连续 — 冷号线性/热号指数) ----
        expected_rate = 2.0 / 12.0  # 后区期望频率
        freq_raw = stats['back_frequency']
        freq_decayed = stats.get('back_frequency_decayed', freq_raw)
        decay_w = stats.get('decay_weight_sum', max(total, 1))
        actual_rate = freq_decayed.get(num, 0) / max(decay_w, 1e-9)
        deviation_ratio = actual_rate / max(expected_rate, 0.01)

        if deviation_ratio <= 1.0:
            scores['frequency'] = 0.55 + 0.15 * (1.0 - deviation_ratio)
        else:
            scores['frequency'] = max(0.15, 0.55 * math.exp(-1.8 * (deviation_ratio - 1.0)))

        # ---- 遗漏偏离度得分 (v3.2: 连续平滑函数) ----
        expected_gap = 12.0 / 2.0  # 后区期望遗漏
        gaps = stats['back_current_gaps']
        gap_std = stats.get('back_gap_std', {})
        gap = gaps.get(num, expected_gap)
        std = gap_std.get(num, expected_gap / 2)

        gap_ratio = gap / expected_gap
        if gap_ratio <= 1.0:
            scores['gap'] = 0.25 + 0.60 * (gap_ratio ** 0.7)
        else:
            scores['gap'] = 0.85 - 0.45 * (1.0 - math.exp(-(gap_ratio - 1.0) * 0.8))

        if std > expected_gap * 0.8:
            scores['gap'] *= 0.85

        # ---- 趋势得分 ----
        trajectory = stats.get('back_temperature_trajectory', {})
        traj = trajectory.get(num, {})
        if traj.get('direction') == 'rising':
            scores['trend'] = 0.75
        elif traj.get('direction') == 'falling':
            scores['trend'] = 0.40
        else:
            scores['trend'] = 0.55

        # ---- 012路得分 ----
        road = num % 3
        road_data = stats.get('back_road_analysis', {}).get('total', {})
        road_all_max = max(road_data.values()) if road_data else 1
        scores['road'] = road_data.get(road, 0) / road_all_max if road_all_max > 0 else 0

        # ---- 重号概率得分 (v3: 改进) ----
        if len(self.history_data) >= 1:
            last_back = set(self.history_data[0]['back'])
            if num in last_back:
                # 检查连续出现次数
                recent_count = 0
                for r in self.history_data[:3]:
                    if num in r.get('back', []):
                        recent_count += 1
                if recent_count <= 1:
                    scores['repeat'] = 0.85  # 上期号码，重号概率高
                elif recent_count <= 2:
                    scores['repeat'] = 0.45
                else:
                    scores['repeat'] = 0.15
            else:
                scores['repeat'] = 0.30
        else:
            scores['repeat'] = 0.30

        # ---- 邻号概率得分 ----
        # 上期开奖号自身的 adjacent 不加分（已由 repeat 覆盖，避免双重计数）
        if len(self.history_data) >= 1:
            last_back = set(self.history_data[0]['back'])
            last_neighbors = set()
            for n in last_back:
                if n - 1 >= 1:
                    last_neighbors.add(n - 1)
                if n + 1 <= 12:
                    last_neighbors.add(n + 1)
            # 排除上期开奖号自身，只对真正的"邻号"给高分
            true_neighbors = last_neighbors - last_back
            if num in true_neighbors:
                scores['adjacent'] = 0.58  # 微弱加成，不再用 0.75 碾压
            elif num not in last_back:
                scores['adjacent'] = 0.42
            else:
                scores['adjacent'] = 0.42  # 上期开奖号不给adjacent加成
        else:
            scores['adjacent'] = 0.45

        # ---- 位置得分 (后区2位) ----
        # 后区号码的位置分布: 第一位vs第二位
        position_freq = stats.get('back_position_frequency', [{}, {}])
        if len(position_freq) >= 2:
            pos_scores = []
            for i in range(2):
                pf = position_freq[i] if i < len(position_freq) else {}
                pos_max = max(pf.values()) if pf else 1
                pos_scores.append(pf.get(num, 0) / pos_max)
            scores['position'] = sum(pos_scores) / 2
        else:
            scores['position'] = 0.5

        # ---- 和值得分 ----
        # 后区期望值约(1+12)/2 = 6.5
        ideal_back = 6.5
        scores['sum'] = 1.0 - abs(num - ideal_back) / 12.0

        return scores

    # ==================== ML模型接口 (v2.2新增) ====================

    def _model_ml_front(self, top_n: int = 12) -> Optional[List[int]]:
        """ML模型前区预测"""
        try:
            from .ml import predict_with_ml
            ml_result = predict_with_ml(self.history_data)
            front_probs = ml_result.get('front_probs', {})
            front_top = sorted(front_probs.keys(), key=lambda n: -front_probs[n])[:top_n]
            return front_top
        except Exception as e:
            log.debug(f"ML前区模型调用失败: {e}")
            return None

    def _model_ml_back(self, top_n: int = 8) -> Optional[List[int]]:
        """ML模型后区预测"""
        try:
            from .ml import predict_with_ml
            ml_result = predict_with_ml(self.history_data)
            back_probs = ml_result.get('back_probs', {})
            back_top = sorted(back_probs.keys(), key=lambda n: -back_probs[n])[:top_n]
            return back_top
        except Exception as e:
            log.debug(f"ML后区模型调用失败: {e}")
            return None

    def _get_ml_front_auc(self) -> float:
        """获取ML前区模型的AUC(用于动态权重)"""
        try:
            from .ml import _ml_cache
            if _ml_cache and _ml_cache.get('predictor'):
                predictor = _ml_cache['predictor']
                front_scores = predictor.front_scores if hasattr(predictor, 'front_scores') else {}
                if front_scores:
                    return max(front_scores.values())
            return 0.50  # 默认: 接近随机
        except Exception:
            return 0.50

    def _get_ml_back_auc(self) -> float:
        """获取ML后区模型的AUC(用于动态权重)"""
        try:
            from .ml import _ml_cache
            if _ml_cache and _ml_cache.get('predictor'):
                predictor = _ml_cache['predictor']
                back_scores = predictor.back_scores if hasattr(predictor, 'back_scores') else {}
                if back_scores:
                    return max(back_scores.values())
            return 0.50
        except Exception:
            return 0.50

    def multi_model_voting(self, front_n: int = 5, back_n: int = 2, n_votes: int = 3, skip_ml: bool = False) -> Dict:
        """多模型集成投票 (v3.3: 前区+后区排名绝对主导)

        v3.3改进:
        - 后区排名权重从2.0→4.0(绝对主导), 修复后区投票低于随机基线
        - ML权重上限进一步降低(回测中可skip_ml提升速度)
        - 以ensemble_ranking为核心(回测+17.6%超随机)
        - Markov/Markov2保留(结构性信号)
        - ML权重动态调整
        - Bayes保留(概率信号)
        """
        # 前区投票 (4个有效模型，去除hot/cold)
        models = [
            self._model_bayesian(top_n=18),      # 贝叶斯概率
            self._model_rank(top_n=18),           # 排名模型(v3特征评分，最有效)
            self._model_markov(top_n=18),         # 马尔可夫转移
            self._model_markov2(top_n=18),        # 二阶马尔可夫
        ]
        # v3.3: 排名绝对主导(前区回测证实纯排名≥3=+19.4%超随机,投票无改善)
        # 贝叶斯0.1(信号0.500=噪声), Markov0.2(回测证实有害), 排名4.0(最强信号)
        model_weights = [0.1, 4.0, 0.2, 0.2]

        votes = defaultdict(float)
        for model_idx, model_result in enumerate(models):
            mw = model_weights[model_idx]
            for rank, num in enumerate(model_result):
                weight = (1.0 - (rank / max(len(model_result), 1))) * mw
                votes[num] += weight

        # ML模型集成 — v3.3: 回测中skip_ml跳过(避免每期重新训练), 实盘保持微弱权重
        if not skip_ml:
            ml_result = self._model_ml_front(top_n=18)
            ml_front_auc = self._get_ml_front_auc()
            if ml_result:
                # AUC=0.50 → weight=0.1(几乎零), AUC=0.60 → weight=0.3
                ml_weight = max(0.1, min(0.5, (ml_front_auc - 0.48) * 2.0))
                for rank, num in enumerate(ml_result):
                    weight = (1.0 - (rank / max(len(ml_result), 1))) * ml_weight
                    votes[num] += weight

        # v3: 扩展候选集到18
        front_candidates = sorted(votes.items(), key=lambda x: -x[1])[:18]
        front_selected = [num for num, _ in front_candidates[:front_n]]

        # 后区投票 (5个模型 + ML)
        back_votes = defaultdict(float)
        back_models = [
            self._model_bayesian_back(top_n=8),
            self._model_repeat_back(top_n=8),
            self._model_adjacent_back(top_n=8),
            self._model_markov_back(top_n=8),
            self._model_rank_back(top_n=8),
        ]
        # v3.4: 后区排名仍主导；相邻模型权重下调，避免投票后区扎堆临号
        # bayesian=0.1, repeat=0.1, adjacent=0.2(原0.5), markov=0.1, rank=4.0
        back_model_weights = [0.1, 0.1, 0.2, 0.1, 4.0]

        for model_idx, model_result in enumerate(back_models):
            mw = back_model_weights[model_idx]
            for rank, num in enumerate(model_result):
                weight = (1.0 - (rank / max(len(model_result), 1))) * mw
                back_votes[num] += weight

        # ML后区集成 — v3.3: 回测中skip_ml跳过, 实盘保持极低权重
        if not skip_ml:
            ml_back_result = self._model_ml_back(top_n=8)
            ml_back_auc = self._get_ml_back_auc()
            if ml_back_result:
                # AUC=0.50 → weight=0.05(几乎零), AUC=0.60 → weight=0.15
                ml_back_weight = max(0.05, min(0.2, (ml_back_auc - 0.48) * 1.0))
                for rank, num in enumerate(ml_back_result):
                    weight = (1.0 - (rank / max(len(ml_back_result), 1))) * ml_back_weight
                    back_votes[num] += weight

        back_candidates_sorted = sorted(back_votes.items(), key=lambda x: -x[1])
        back_selected = [num for num, _ in back_candidates_sorted[:back_n]]

        cycles = self.identify_cycles()

        return {
            'front': front_selected,
            'back': back_selected,
            'front_candidates': [{'number': num, 'score': round(score, 3)} for num, score in front_candidates],
            'back_candidates': [{'number': num, 'score': round(score, 3)} for num, score in back_candidates_sorted],
            'front_votes': {num: round(v, 3) for num, v in votes.items()},
            'back_votes': {num: round(v, 3) for num, v in back_votes.items()},
            'cycle_info': cycles,
            'hot_front': cycles.get('hot_front', []),
            'cold_front': cycles.get('cold_front', []),
            'hot_back': cycles.get('hot_back', []),
            'cold_back': cycles.get('cold_back', []),
        }

    # ==================== 约束推荐生成器 ====================

    def _score_based_select(self, candidates: List[int], count: int,
                            is_front: bool = True,
                            fallback_pool: List[int] = None,
                            weights_override: Dict[str, float] = None,
                            exclude: set = None) -> List[int]:
        """基于评分的约束选择 (替代 random.sample)。

        当候选数量不足时，若提供 fallback_pool，则从 fallback_pool 中按评分补充。
        weights_override: 策略自定义权重，为 None 时使用默认 FEATURE_WEIGHTS/BACK_FEATURE_WEIGHTS。
        exclude: 组间去重已用号码，选号与 fallback 均避开。
        """
        stats = self.statistics
        exclude = set(exclude or [])
        candidates = [n for n in candidates if n not in exclude]

        # 如果候选不足，用 fallback_pool 补充（仍避开 exclude）
        if fallback_pool and len(candidates) < count:
            pool = [n for n in fallback_pool if n not in exclude and n not in candidates]
            candidates = list(dict.fromkeys(list(candidates) + pool))

        # 池子仍不足时放宽：只保证取够 count（极少发生）
        if len(candidates) < count and fallback_pool:
            pool = [n for n in fallback_pool if n not in candidates]
            candidates = list(dict.fromkeys(list(candidates) + pool))

        if not stats or len(candidates) < count:
            return sorted(candidates[:count])

        # 获取每个候选的评分
        scored = []
        weights = weights_override if weights_override is not None else (
            FEATURE_WEIGHTS if is_front else BACK_FEATURE_WEIGHTS)
        for num in candidates:
            features = self._calculate_feature_score(num, is_front=True) if is_front else self._calculate_back_feature_score(num)
            score = sum(
                features.get(k, 0) * weights.get(k, 0)
                for k in weights
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

    def generate_recommendation(self, method: str = 'balanced',
                                 exclude_front: List[int] = None,
                                 exclude_back: List[int] = None,
                                 voting_result: Dict = None) -> Dict:
        """生成推荐号码 (v2: 基于评分+约束选择)。

        Args:
            method: 推荐策略 balanced/hot/cold/rank。
            exclude_front: 已选前区号码，生成新组时避免重复。
            exclude_back: 已选后区号码，生成新组时避免重复。
            voting_result: 预计算的 multi_model_voting 结果（避免 balanced 重复投票）。
        """
        exclude_front = set(exclude_front or [])
        exclude_back = set(exclude_back or [])

        def _filter(candidates, exclude):
            return [n for n in candidates if n not in exclude]

        def _expand_front_candidates(result):
            # 从投票结果中取前 15 个作为候选池，避免排除后不足
            return [c['number'] for c in result.get('front_candidates', [])][:15]

        def _expand_back_candidates(result):
            return [c['number'] for c in result.get('back_candidates', [])][:10]

        if method == 'hot':
            hot_front = [num for num, _ in self.statistics.get('hot_front', [])[:20]]
            hot_back = [num for num, _ in self.statistics.get('hot_back', [])[:10]]
            front = self._score_based_select(
                _filter(hot_front, exclude_front), 5, is_front=True,
                fallback_pool=FRONT_NUMBERS,
                exclude=exclude_front,
                weights_override={
                    'frequency': 0.25, 'gap': 0.06, 'position': 0.14,
                    'road': 0.10, 'sum': 0.12, 'trend': 0.15,
                    'zone': 0.08, 'repeat': 0.10, 'adjacent': 0.05,
                })
            back = self._score_based_select(
                _filter(hot_back, exclude_back), 2, is_front=False,
                fallback_pool=BACK_NUMBERS,
                exclude=exclude_back,
                weights_override={
                    'frequency': 0.22, 'gap': 0.08, 'trend': 0.12,
                    'road': 0.15, 'repeat': 0.08, 'adjacent': 0.08,
                    'position': 0.15, 'sum': 0.12,
                })
        elif method == 'cold':
            cold_front = [num for num, _ in self.statistics.get('cold_front', [])[:20]]
            cold_back = [num for num, _ in self.statistics.get('cold_back', [])[:10]]
            front = self._score_based_select(
                _filter(cold_front, exclude_front), 5, is_front=True,
                fallback_pool=FRONT_NUMBERS,
                exclude=exclude_front,
                weights_override={
                    'frequency': 0.05, 'gap': 0.25, 'position': 0.14,
                    'road': 0.10, 'sum': 0.12, 'trend': 0.04,
                    'zone': 0.10, 'repeat': 0.10, 'adjacent': 0.05,
                })
            back = self._score_based_select(
                _filter(cold_back, exclude_back), 2, is_front=False,
                fallback_pool=BACK_NUMBERS,
                exclude=exclude_back,
                weights_override={
                    'frequency': 0.05, 'gap': 0.32, 'trend': 0.05,
                    'road': 0.15, 'repeat': 0.08, 'adjacent': 0.08,
                    'position': 0.15, 'sum': 0.12,
                })
        elif method == 'rank':
            front_ranked, back_ranked = self.rank_model(top_n=20)
            front_candidates = [num for num, _, _ in front_ranked[:20]]
            back_candidates = [num for num, _, _ in back_ranked[:10]]
            front = self._score_based_select(
                _filter(front_candidates, exclude_front), 5, is_front=True,
                fallback_pool=FRONT_NUMBERS, exclude=exclude_front)
            back = self._score_based_select(
                _filter(back_candidates, exclude_back), 2, is_front=False,
                fallback_pool=BACK_NUMBERS, exclude=exclude_back)
        else:
            # 平衡模式：复用外部投票结果，避免二次 multi_model_voting
            result = voting_result
            if not result or not result.get('front_candidates'):
                result = self.multi_model_voting(front_n=20, back_n=10)
            front_candidates = _filter(_expand_front_candidates(result), exclude_front)
            back_candidates = _filter(_expand_back_candidates(result), exclude_back)
            # 候选不足时回退到排名模型候选
            if len(front_candidates) < 5 or len(back_candidates) < 2:
                fr, br = self.rank_model(top_n=20)
                if len(front_candidates) < 5:
                    front_candidates = _filter([n for n, _, _ in fr], exclude_front)
                if len(back_candidates) < 2:
                    back_candidates = _filter([n for n, _, _ in br], exclude_back)
            front = self._score_based_select(
                front_candidates, 5, is_front=True,
                fallback_pool=FRONT_NUMBERS, exclude=exclude_front)
            back = self._score_based_select(
                back_candidates, 2, is_front=False,
                fallback_pool=BACK_NUMBERS, exclude=exclude_back)

        return {
            'front': front,
            'back': back,
            'method': method
        }

    def generate_multi_strategy_recommendations(self, voting_result: Dict = None) -> Dict:
        """生成多策略推荐，组间互斥避免「多组完全重号」。

        主推取排名 Top5/Top2；后续策略依次避开已用号码，保证每组前后区组合不同。
        """
        voting = voting_result or self.multi_model_voting(front_n=20, back_n=10)
        used_front = set()
        used_back = set()
        recommendations = []

        front_ranked, back_ranked = self.rank_model(top_n=20)
        ranked_front_numbers = [n for n, _, _ in front_ranked]
        ranked_back_numbers = [n for n, _, _ in back_ranked]
        latest_issue = str((self.history_data[0] if self.history_data else {}).get('issue') or '0')
        try:
            issue_seed = int(latest_issue)
        except (TypeError, ValueError):
            issue_seed = sum(ord(ch) for ch in latest_issue)

        # Keep the two strongest front numbers and strongest back number, while
        # rotating the remaining positions inside high-ranked pools.  This
        # prevents a stable rank table from pinning the first ticket forever.
        primary_front_core = ranked_front_numbers[:2]
        front_rotation_pool = ranked_front_numbers[2:10]
        front_support = []
        if front_rotation_pool:
            offset = issue_seed % len(front_rotation_pool)
            step = 3 if len(front_rotation_pool) >= 8 else 1
            cursor = offset
            while len(front_support) < 3 and len(front_support) < len(front_rotation_pool):
                number = front_rotation_pool[cursor % len(front_rotation_pool)]
                if number not in front_support:
                    front_support.append(number)
                cursor += step
        if len(primary_front_core) + len(front_support) < 5:
            front_support.extend(
                n for n in ranked_front_numbers
                if n not in primary_front_core and n not in front_support
            )

        primary_back_core = ranked_back_numbers[:1]
        back_rotation_pool = ranked_back_numbers[1:6]
        back_support = []
        if back_rotation_pool:
            back_support = [back_rotation_pool[issue_seed % len(back_rotation_pool)]]
        if len(primary_back_core) + len(back_support) < 2:
            back_support.extend(
                n for n in ranked_back_numbers
                if n not in primary_back_core and n not in back_support
            )

        primary = {
            'front': sorted((primary_front_core + front_support)[:5]),
            'back': sorted((primary_back_core + back_support)[:2]),
            'method': '主推（核心+高分轮换）',
            'strategy': 'primary_rank',
            'core_front': sorted(primary_front_core),
            'core_back': sorted(primary_back_core),
            'based_on_issue': latest_issue,
        }
        recommendations.append(primary)
        used_front.update(primary['front'])
        used_back.update(primary['back'])

        # A portfolio should diversify combinations, not blindly discard every
        # high-ranked number after the first ticket.  Keep two front anchors and
        # one back anchor in play; diversify the remaining positions.  The old
        # full-exclusion policy made later tickets progressively lower quality.
        front_anchors = set(primary_front_core)
        # 后区只有12个号码。五组共10个后区位置时，重复锚点会直接浪费
        # 组合覆盖；保留前区锚点，但后区优先使用尚未覆盖的号码。
        back_anchors = set()

        strategies = [
            ('balanced', '均衡策略'),
            ('rank', '排名策略'),
            ('hot', '热号策略'),
            ('cold', '冷号策略'),
        ]
        seen_tickets = {
            (tuple(primary['front']), tuple(primary['back']))
        }

        for key, name in strategies:
            # 后区盘口小：若已用超过 8 个，只排除「曾整组出现过的后区组合」不够，
            # 优先排除 used_back；不够 2 个可用时再放宽。
            exclude_f = list(used_front - front_anchors)
            exclude_b = list(used_back)
            if len(BACK_NUMBERS) - len(used_back) < 2:
                exclude_b = []

            rec = self.generate_recommendation(
                key,
                exclude_front=exclude_f,
                exclude_back=exclude_b,
                voting_result=voting if key == 'balanced' else None,
            )
            ticket = (tuple(rec['front']), tuple(rec['back']))
            if ticket in seen_tickets:
                # 强制再避一次：前区排除已用，后区排除已用（必要时放宽前区保留后区差异）
                rec = self.generate_recommendation(
                    key,
                    exclude_front=list(used_front - front_anchors),
                    exclude_back=list(used_back) if len(BACK_NUMBERS) - len(used_back) >= 2 else [],
                    voting_result=voting if key == 'balanced' else None,
                )
                ticket = (tuple(rec['front']), tuple(rec['back']))
                # 若前区仍撞车，至少打散后区
                if ticket in seen_tickets and len(BACK_NUMBERS) - len(used_back) >= 2:
                    alt_back = self._score_based_select(
                        [n for n in BACK_NUMBERS if n not in used_back],
                        2, is_front=False,
                        fallback_pool=BACK_NUMBERS,
                        exclude=used_back,
                    )
                    rec = {**rec, 'front': rec['front'], 'back': alt_back}
                    ticket = (tuple(rec['front']), tuple(rec['back']))

            item = {
                'front': rec['front'],
                'back': rec['back'],
                'method': name,
                'strategy': key,
            }
            recommendations.append(item)
            seen_tickets.add(ticket)
            used_front.update(rec['front'])
            used_back.update(rec['back'])

        back_pairs = {
            tuple(sorted(item.get('back', [])))
            for item in recommendations
            if len(item.get('back', [])) == 2
        }
        covered_back = set().union(*(set(pair) for pair in back_pairs)) if back_pairs else set()
        total_back_outcomes = math.comb(len(BACK_NUMBERS), 2)
        uncovered_count = len(BACK_NUMBERS) - len(covered_back)
        miss_all = (
            math.comb(uncovered_count, 2) / total_back_outcomes
            if uncovered_count >= 2 else 0.0
        )
        back_coverage_profile = {
            'unique_numbers': sorted(covered_back),
            'unique_number_count': len(covered_back),
            'unique_pair_count': len(back_pairs),
            'at_least_one_group_ge1_probability': round(1.0 - miss_all, 6),
            'at_least_one_group_ge2_probability': round(len(back_pairs) / total_back_outcomes, 6),
            'method': 'exact_combinatorial_portfolio',
        }

        return {
            'recommendations': recommendations,
            'portfolio_policy': {
                'name': 'rank_core_rotating_primary_back_coverage',
                'front_anchors': sorted(front_anchors),
                'back_anchors': sorted(back_anchors),
                'primary_based_on_issue': latest_issue,
                'primary_front_pool': ranked_front_numbers[:10],
                'primary_back_pool': ranked_back_numbers[:6],
                'note': '首注保留排名核心并按最新期号轮换高分候选；后区五组优先覆盖10个不同号码',
            },
            'back_coverage_profile': back_coverage_profile,
            'voting_front': [c['number'] for c in (voting.get('front_candidates') or [])[:12]],
            'voting_back': [c['number'] for c in (voting.get('back_candidates') or [])[:6]],
            'voting': voting,
        }

    # ==================== 回测功能 ====================

    def backtest(self, method: str = 'balanced', test_periods: int = 30) -> Dict:
        """历史回测"""
        if len(self.history_data) < test_periods:
            return {'error': '历史数据不足'}

        results = []
        total_front_matched = 0
        total_back_matched = 0
        front_match_distribution = {i: 0 for i in range(6)}
        back_match_distribution = {i: 0 for i in range(3)}

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
            front_match_distribution[front_match] += 1
            back_match_distribution[back_match] += 1

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

        n = test_periods or 1

        def _hypergeom_distribution(population, winners, picks):
            dist = {}
            denom = math.comb(population, picks)
            for k in range(0, min(winners, picks) + 1):
                if picks - k <= population - winners:
                    dist[k] = math.comb(winners, k) * math.comb(population - winners, picks - k) / denom
            return dist

        front_random = _hypergeom_distribution(35, 5, 5)
        back_random = _hypergeom_distribution(12, 2, 2)
        front_ge2 = sum(v for k, v in front_match_distribution.items() if k >= 2) / n
        front_ge3 = sum(v for k, v in front_match_distribution.items() if k >= 3) / n
        back_ge1 = sum(v for k, v in back_match_distribution.items() if k >= 1) / n
        back_ge2 = back_match_distribution.get(2, 0) / n

        return {
            'method': method,
            'test_periods': test_periods,
            'total_matched': 0,  # full match保持兼容
            'front_accuracy': total_front_matched / (test_periods * 5),
            'back_accuracy': total_back_matched / (test_periods * 2),
            'front_match_distribution': front_match_distribution,
            'back_match_distribution': back_match_distribution,
            'rates': {
                'front_ge2_rate': front_ge2,
                'front_ge3_rate': front_ge3,
                'back_ge1_rate': back_ge1,
                'back_ge2_rate': back_ge2,
            },
            'random_baseline': {
                'front_match_distribution': {k: round(v, 6) for k, v in front_random.items()},
                'back_match_distribution': {k: round(v, 6) for k, v in back_random.items()},
                'front_ge2_rate': round(sum(v for k, v in front_random.items() if k >= 2), 6),
                'front_ge3_rate': round(sum(v for k, v in front_random.items() if k >= 3), 6),
                'back_ge1_rate': round(sum(v for k, v in back_random.items() if k >= 1), 6),
                'back_ge2_rate': round(back_random.get(2, 0.0), 6),
            },
            'detailed_results': results
        }

    # ==================== 动态抓取开奖号码 ====================

    def fetch_latest_results(self, count: int = 10, force_refresh: bool = False,
                             network_timeout: Optional[float] = None) -> Dict:
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

            if network_timeout is None:
                results = self._fetch_from_web(count)
            else:
                # urlopen 的 timeout 是“每个数据源”的超时，主源和备源会累加。
                # 生产刷新需要一个覆盖全部数据源的总预算；超时线程只负责读取
                # 原始网页，不会修改分析器状态，因此可安全降级到本地历史。
                fetch_state = {'results': [], 'error': None}
                fetch_done = threading.Event()

                def _bounded_fetch():
                    try:
                        fetch_state['results'] = self._fetch_from_web(count)
                    except Exception as exc:
                        fetch_state['error'] = exc
                    finally:
                        fetch_done.set()

                threading.Thread(
                    target=_bounded_fetch,
                    daemon=True,
                    name='DLTNetworkFetch',
                ).start()
                fetch_done.wait(max(0.1, float(network_timeout)))
                if not fetch_done.is_set():
                    log.warning('大乐透网络抓取超过 %.1f 秒，使用本地历史快速重算', network_timeout)
                    results = []
                elif fetch_state['error'] is not None:
                    raise fetch_state['error']
                else:
                    results = fetch_state['results']

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
        """从网络抓取开奖数据

        主源 500.com（datachart）：覆盖 2007 年至今全量历史，含开奖日期，最稳定。
        备源 ip138：仅近 30 期，用于主源失效时兜底。
        count 控制抓取期数（limit）；传入较大值即可一次性引导全量历史。
        """
        c500 = self._fetch_500_history(count)
        if c500:
            return c500[:count]

        try:
            import urllib.request
            import urllib.error

            sources = [
                ('https://cp.ip138.com/daletou/', 'ip138'),
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

    @staticmethod
    def _normalize_issue_500(raw: str) -> str:
        """500.com 期号为 5 位(yyNNN，如 26071)，统一为本项目的 7 位(2026071)。"""
        raw = raw.strip()
        if len(raw) == 5 and raw.isdigit():
            return f"20{raw[:2]}{raw[2:]}"
        return raw

    def _fetch_500_history(self, count: int) -> List[Dict]:
        """从 500.com 抓取大乐透历史（按期号倒序返回）。"""
        try:
            import urllib.request
            import ssl
            import re

            limit = max(count, 30)
            url = (
                'https://datachart.500.com/dlt/history/newinc/history.php'
                f'?start=07001&end=99999&limit={limit}'
            )
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://datachart.500.com/dlt/',
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25, context=ctx) as resp:
                data = resp.read().decode('gbk', errors='ignore')

            results = []
            rows = re.findall(r'<tr class="t_tr1">(.*?)</tr>', data, re.DOTALL)
            for row in rows:
                tds = [re.sub(r'<[^>]+>', '', x).strip()
                       for x in re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)]
                # 500.com 列布局随 limit 变化（小 limit 多一个前导序号列），
                # 因此动态定位 5 位期号列，再取其后 5+2 个号码，避免硬编码列下标。
                idx = next((i for i, v in enumerate(tds) if re.fullmatch(r'\d{5}', v)), None)
                if idx is None or idx + 7 >= len(tds):
                    continue
                try:
                    nums = [int(x) for x in tds[idx + 1:idx + 8]]
                except ValueError:
                    continue
                front = sorted(nums[:5])
                back = sorted(nums[5:7])
                if not (all(1 <= n <= 35 for n in front) and all(1 <= n <= 12 for n in back)):
                    continue
                date = tds[-1] if re.fullmatch(r'\d{4}-\d{2}-\d{2}', tds[-1]) else ''
                results.append({
                    'issue': self._normalize_issue_500(tds[idx]),
                    'front': front,
                    'back': back,
                    'date': date,
                })
            results.sort(key=lambda x: x['issue'], reverse=True)
            return results
        except Exception as e:
            log.debug(f'500.com 抓取失败: {e}')
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

        # 仅保留早于抓取范围的「真实」本地数据。若当前是模拟数据，则全部丢弃，
        # 避免 100 期随机模拟号码污染统计基底（历史 bug：fetch 后模拟数据仍被保留）。
        preserved_local = []
        if not self.using_simulated_data:
            for result in self.history_data:
                if result['issue'] < earliest_fetched and result['issue'] not in fetched_map:
                    preserved_local.append(result)

        merged_results = sorted(fetched_results, key=lambda x: x['issue'], reverse=True) + preserved_local

        self.history_data = merged_results
        self.using_simulated_data = False
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


# ==================== 规则+ML融合预测 ====================

def fuse_rule_ml(rule_front_ranked: List[Tuple], rule_back_ranked: List[Tuple],
                 ml_prediction: Dict, rule_weight: float = 0.55,
                 ml_weight: float = 0.45) -> Dict:
    """融合规则排名模型和ML模型的预测结果

    与 lottery3d 的融合逻辑一致: 基于回测表现的动态权重融合。

    Args:
        rule_front_ranked: 规则模型前区排名 [(num, score, features), ...]
        rule_back_ranked: 规则模型后区排名 [(num, score, features), ...]
        ml_prediction: ML模型预测结果 {front_probs, back_probs, ...}
        rule_weight: 规则模型权重
        ml_weight: ML模型权重

    Returns:
        {
            'front_ranked': 融合后前区排名 [(num, fused_score, tag), ...],
            'back_ranked': 融合后后区排名 [(num, fused_score, tag), ...],
            'front_top12': 前区Top12推荐,
            'back_top6': 后区Top6推荐,
        }
    """
    total_w = rule_weight + ml_weight
    rule_w = rule_weight / total_w
    ml_w = ml_weight / total_w

    # 归一化规则得分到 0-1
    rule_front_scores = {}
    if rule_front_ranked:
        max_s = max(s for _, s, _ in rule_front_ranked)
        min_s = min(s for _, s, _ in rule_front_ranked)
        s_range = max_s - min_s if max_s > min_s else 1.0
        for num, score, _ in rule_front_ranked:
            rule_front_scores[num] = (score - min_s) / s_range
    else:
        rule_front_scores = {}

    rule_back_scores = {}
    if rule_back_ranked:
        max_s = max(s for _, s, _ in rule_back_ranked)
        min_s = min(s for _, s, _ in rule_back_ranked)
        s_range = max_s - min_s if max_s > min_s else 1.0
        for num, score, _ in rule_back_ranked:
            rule_back_scores[num] = (score - min_s) / s_range
    else:
        rule_back_scores = {}

    # ML概率
    ml_front = ml_prediction.get('front_probs', {}) if ml_prediction else {}
    ml_back = ml_prediction.get('back_probs', {}) if ml_prediction else {}

    # 前区融合
    all_front = set(rule_front_scores.keys()) | set(ml_front.keys())
    front_fused = []
    for num in sorted(all_front):
        r_score = rule_front_scores.get(num, 0.0)
        m_score = ml_front.get(num, 0.0)
        fused = r_score * rule_w + m_score * ml_w
        in_rule = num in rule_front_scores
        in_ml = num in ml_front
        if in_rule and in_ml:
            tag = 'high_confidence'
            fused += 0.1  # 双方一致加分
        elif in_rule:
            tag = 'rule_preferred'
        elif in_ml:
            tag = 'ml_preferred'
        else:
            tag = 'other'
        front_fused.append((num, round(fused, 4), tag, in_rule, in_ml))

    front_fused.sort(key=lambda x: -x[1])

    # 后区融合
    all_back = set(rule_back_scores.keys()) | set(ml_back.keys())
    back_fused = []
    for num in sorted(all_back):
        r_score = rule_back_scores.get(num, 0.0)
        m_score = ml_back.get(num, 0.0)
        fused = r_score * rule_w + m_score * ml_w
        in_rule = num in rule_back_scores
        in_ml = num in ml_back
        if in_rule and in_ml:
            tag = 'high_confidence'
            fused += 0.1
        elif in_rule:
            tag = 'rule_preferred'
        elif in_ml:
            tag = 'ml_preferred'
        else:
            tag = 'other'
        back_fused.append((num, round(fused, 4), tag, in_rule, in_ml))

    back_fused.sort(key=lambda x: -x[1])

    return {
        'front_ranked': front_fused,
        'back_ranked': back_fused,
        'front_top12': [num for num, _, _, _, _ in front_fused[:12]],
        'back_top6': [num for num, _, _, _, _ in back_fused[:6]],
        'rule_weight': rule_w,
        'ml_weight': ml_w,
    }


def compute_fusion_weights(rule_backtest: Dict, ml_backtest: Dict) -> Tuple[float, float]:
    """基于回测表现计算融合权重

    Args:
        rule_backtest: 规则模型回测结果
        ml_backtest: ML模型回测结果

    Returns:
        (rule_weight, ml_weight)
    """
    if not ml_backtest or ml_backtest.get('error'):
        return (1.0, 0.0)

    # 使用前区≥2命中率作为核心指标（兼容顶层字段与 rates 嵌套）
    rule_front_ge2 = (
        rule_backtest.get('front_ge2_rate')
        or (rule_backtest.get('rates') or {}).get('front_ge2_rate')
        or 0
    )
    ml_front_ge2 = ml_backtest.get('front_ge2_rate') or 0
    baseline = RANDOM_BASELINE.get('front_ge2_rate', 0.1389)

    if ml_front_ge2 <= 0 and rule_front_ge2 <= 0:
        return (0.70, 0.30)

    # 相对随机基准的 lift；ML 未超过基准时压低权重，避免噪声模型拖累
    rule_lift = max(rule_front_ge2 - baseline, 0.0)
    ml_lift = max(ml_front_ge2 - baseline, 0.0)
    if ml_lift <= 0 and rule_lift <= 0:
        return (0.75, 0.25)
    if ml_lift <= 0:
        return (0.85, 0.15)

    total = rule_lift + ml_lift
    rule_w = max(0.55, rule_lift / total)  # 规则底权重至少 55%
    ml_w = 1.0 - rule_w
    return (round(rule_w, 2), round(ml_w, 2))


# ==================== 线上预测记录系统 ====================

DALETOU_PREDICTIONS_KEY = 'lottery_dlt_online_predictions'


def load_online_predictions() -> List[Dict]:
    """加载线上预测记录"""
    try:
        return kv_store.load(DALETOU_PREDICTIONS_KEY, [])
    except Exception as e:
        log.error(f"加载大乐透预测记录失败: {e}")
        return []


def save_online_prediction(period: str, recommendations: Dict,
                           fusion_result: Dict = None,
                           based_on_issue: str = None) -> None:
    """保存线上预测记录

    Args:
        period: 目标期号
        recommendations: 各策略推荐 {method: {front, back}}
        fusion_result: 融合结果 (可选)
    """
    try:
        records = load_online_predictions()

        record = {
            'version': LOTTERY_PREDICTOR_VERSION,
            'period': period,
            'based_on_issue': str(based_on_issue or ''),
            'integrity_status': 'pending',
            'recommendations': {
                method: {
                    'front': rec.get('front', []),
                    'back': rec.get('back', []),
                }
                for method, rec in recommendations.items()
            },
            'actual': None,
            'settled': False,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        if fusion_result:
            record['fusion'] = {
                'front_top12': fusion_result.get('front_top12', []),
                'back_top6': fusion_result.get('back_top6', []),
                'weights': {
                    'rule': fusion_result.get('rule_weight', 0.55),
                    'ml': fusion_result.get('ml_weight', 0.45),
                },
            }

        # 按期号去重
        existing_idx = None
        for i, r in enumerate(records):
            if r.get('period') == period:
                existing_idx = i
                break

        if existing_idx is not None:
            if records[existing_idx].get('settled'):
                log.info(f"预测记录已结算，跳过更新: {period}")
                return
            record['created_at'] = records[existing_idx].get('created_at', record['created_at'])
            records[existing_idx] = record
        else:
            records.append(record)

        # 保留最近200期
        records = records[-200:]
        kv_store.save(DALETOU_PREDICTIONS_KEY, records)
        log.info(f"大乐透预测记录已保存: {period}")
    except Exception as e:
        log.error(f"保存大乐透预测记录失败: {e}")


def settle_predictions(history_data: List[Dict]) -> int:
    """结算未回填的预测记录

    Args:
        history_data: 历史开奖数据 (idx=0最新)

    Returns:
        结算的记录数
    """
    records = load_online_predictions()
    if not records:
        return 0

    changed = False
    settled_count = 0

    period_index = {h['issue']: i for i, h in enumerate(history_data)}

    for record in records:
        if record.get('settled'):
            continue

        period = record.get('period')
        if period not in period_index:
            continue

        idx = period_index[period]
        actual_data = history_data[idx]
        actual_front = set(actual_data['front'])
        actual_back = set(actual_data['back'])

        # A real forward prediction must prove which already-drawn issue it was
        # based on. Legacy rows without this field cannot be distinguished from
        # hindsight-generated predictions and must not be reported as hits.
        based_on = str(record.get('based_on_issue') or '')
        try:
            forward_valid = bool(based_on) and int(based_on) < int(period)
        except (TypeError, ValueError):
            forward_valid = False
        if not forward_valid:
            record['actual'] = {
                'front': sorted(actual_front),
                'back': sorted(actual_back),
            }
            record['settled'] = True
            record['settled_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            record['integrity_status'] = 'legacy_unverified'
            record['integrity_note'] = '缺少开奖前生成凭证，不计入实盘命中'
            changed = True
            settled_count += 1
            continue

        record['integrity_status'] = 'verified_forward'

        record['actual'] = {
            'front': list(actual_front),
            'back': list(actual_back),
        }
        record['settled'] = True
        record['settled_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # 结算各策略命中
        for method, rec in record.get('recommendations', {}).items():
            pred_front = set(rec.get('front', []))
            pred_back = set(rec.get('back', []))
            record[f'{method}_front_hit'] = len(pred_front & actual_front)
            record[f'{method}_back_hit'] = len(pred_back & actual_back)

        # 结算融合命中
        if record.get('fusion'):
            fusion_front = set(record['fusion'].get('front_top12', []))
            fusion_back = set(record['fusion'].get('back_top6', []))
            record['fusion_front_hit'] = len(fusion_front & actual_front)
            record['fusion_back_hit'] = len(fusion_back & actual_back)

        changed = True
        settled_count += 1

    if changed:
        kv_store.save(DALETOU_PREDICTIONS_KEY, records)
        log.info(f"大乐透预测已结算 {settled_count} 条")

    return settled_count


def calculate_online_stats() -> Dict:
    """计算线上实盘命中率统计

    Returns:
        {
            'total_records': int,
            'settled_count': int,
            'by_method': {method: {front_ge1_rate, front_ge2_rate, back_ge1_rate, back_ge2_rate}},
            'fusion': {front_ge1_rate, front_ge2_rate, back_ge1_rate, back_ge2_rate},
            'baseline': 随机基准,
        }
    """
    records = load_online_predictions()
    settled = [
        r for r in records
        if r.get('settled') and r.get('integrity_status') == 'verified_forward'
    ]
    n = len(settled)

    if n == 0:
        return {
            'total_records': len(records),
            'settled_count': 0,
            'by_method': {},
            'fusion': {},
        }

    methods = set()
    for r in settled:
        methods.update(r.get('recommendations', {}).keys())

    by_method = {}
    for method in sorted(methods):
        # Strategies were added over time.  A method that did not exist in an
        # older version is not a miss and must not dilute its hit rate.
        method_records = [r for r in settled if method in (r.get('recommendations') or {})]
        method_n = len(method_records)
        front_hits = [r.get(f'{method}_front_hit', 0) for r in method_records]
        back_hits = [r.get(f'{method}_back_hit', 0) for r in method_records]
        by_method[method] = {
            'count': method_n,
            'front_ge1_rate': round(sum(1 for h in front_hits if h >= 1) / method_n, 4),
            'front_ge2_rate': round(sum(1 for h in front_hits if h >= 2) / method_n, 4),
            'front_ge3_rate': round(sum(1 for h in front_hits if h >= 3) / method_n, 4),
            'front_avg': round(sum(front_hits) / method_n, 2),
            'back_ge1_rate': round(sum(1 for h in back_hits if h >= 1) / method_n, 4),
            'back_ge2_rate': round(sum(1 for h in back_hits if h >= 2) / method_n, 4),
            'back_avg': round(sum(back_hits) / method_n, 2),
        }

    def _portfolio_stats(rows):
        if not rows:
            return {}
        best_front = []
        best_back = []
        same_ticket_3p1 = []
        for record in rows:
            names = (record.get('recommendations') or {}).keys()
            pairs = [
                (record.get(f'{name}_front_hit', 0), record.get(f'{name}_back_hit', 0))
                for name in names
            ]
            best_front.append(max((front for front, _ in pairs), default=0))
            best_back.append(max((back for _, back in pairs), default=0))
            same_ticket_3p1.append(any(front >= 3 and back >= 1 for front, back in pairs))
        count = len(rows)
        return {
            'count': count,
            'front_any_ge2_rate': round(sum(hit >= 2 for hit in best_front) / count, 4),
            'front_any_ge3_rate': round(sum(hit >= 3 for hit in best_front) / count, 4),
            'back_any_ge1_rate': round(sum(hit >= 1 for hit in best_back) / count, 4),
            'back_any_ge2_rate': round(sum(hit >= 2 for hit in best_back) / count, 4),
            'same_ticket_front3_back1_rate': round(sum(same_ticket_3p1) / count, 4),
            'avg_ticket_count': round(sum(len(r.get('recommendations') or {}) for r in rows) / count, 2),
        }

    versions = sorted({r.get('version') or 'legacy-unversioned' for r in settled})
    by_version = {
        version: _portfolio_stats([
            r for r in settled if (r.get('version') or 'legacy-unversioned') == version
        ])
        for version in versions
    }

    # 融合统计
    fusion_records = [r for r in settled if r.get('fusion')]
    fusion_stats = {}
    if fusion_records:
        fk = len(fusion_records)
        f_front_hits = [r.get('fusion_front_hit', 0) for r in fusion_records]
        f_back_hits = [r.get('fusion_back_hit', 0) for r in fusion_records]
        fusion_stats = {
            'count': fk,
            'front_ge1_rate': round(sum(1 for h in f_front_hits if h >= 1) / fk, 4),
            'front_ge2_rate': round(sum(1 for h in f_front_hits if h >= 2) / fk, 4),
            'front_ge3_rate': round(sum(1 for h in f_front_hits if h >= 3) / fk, 4),
            'front_avg': round(sum(f_front_hits) / fk, 2),
            'back_ge1_rate': round(sum(1 for h in f_back_hits if h >= 1) / fk, 4),
            'back_ge2_rate': round(sum(1 for h in f_back_hits if h >= 2) / fk, 4),
            'back_avg': round(sum(f_back_hits) / fk, 2),
        }

    return {
        'total_records': len(records),
        'settled_count': n,
        'unsettled_count': len(records) - n,
        'by_method': by_method,
        'portfolio': _portfolio_stats(settled),
        'by_version': by_version,
        'fusion': fusion_stats,
        'baseline': {
            'front_ge1': round(1 - math.comb(30, 5) / math.comb(35, 5), 4),
            'front_ge2': 0.1389,
            'front_ge3': 0.0139,
            'back_ge1': 0.4545,
            'back_ge2': 0.0455,
        },
    }


def run_prediction(force_refresh=False, enable_backtest=True,
                   enable_ml=True, enable_fusion=True,
                   compute_weights=False, network_fetch_timeout=None):
    """运行大乐透预测，返回 JSON 可序列化 dict。

    Args:
        force_refresh: 是否强制刷新缓存（默认 False，使用缓存）
        enable_backtest: 是否启用滚动回测（默认 True）
        enable_ml: 是否启用 ML 模型预测（默认 True）
        enable_fusion: 是否启用规则+ML 融合推荐（默认 True）
        compute_weights: 是否计算动态权重（默认 False；特征回测较慢，仅排障时开启）
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

        # 仅在历史不足/脏数据时全量引导；日常 force_refresh 只增量抓近20期
        initial_quality = analyzer.assess_data_quality()
        full_bootstrap = _needs_full_history_bootstrap(initial_quality)
        fetch_count = FULL_HISTORY_FETCH_COUNT if full_bootstrap else 20
        fetch_result = analyzer.fetch_latest_results(
            count=fetch_count,
            force_refresh=True if force_refresh else full_bootstrap,
            network_timeout=network_fetch_timeout,
        )
        if full_bootstrap:
            log.info(
                "大乐透全量历史引导完成: source=%s count=%s latest=%s",
                fetch_result.get('source'),
                fetch_result.get('count'),
                fetch_result.get('latest_issue'),
            )
        elif force_refresh:
            log.info(
                "大乐透增量抓取完成: source=%s count=%s latest=%s",
                fetch_result.get('source'),
                fetch_result.get('count'),
                fetch_result.get('latest_issue'),
            )

        # 获取统计数据
        stats = analyzer.get_statistics()
        recent = analyzer.get_recent_results(10)
        data_quality = analyzer.assess_data_quality()

        # 滚动回测（默认30期，兼顾显著性与耗时）
        if enable_backtest:
            backtest = analyzer.rolling_backtest(trials=ROLLING_BACKTEST_TRIALS)
        else:
            backtest = {'trials': 0, 'note': 'backtest disabled', 'baseline_comparison': {}}

        # 动态权重：默认关闭重型特征回测；开启时用缩短的 FEATURE_BACKTEST_TRIALS
        if compute_weights and enable_backtest:
            optimized_weights = analyzer.dynamic_weight_adjustment()
            weight_diff = {
                k: round(optimized_weights.get(k, 0) - FEATURE_WEIGHTS.get(k, 0), 4)
                for k in FEATURE_WEIGHTS
            }
        else:
            optimized_weights = dict(FEATURE_WEIGHTS)
            weight_diff = {k: 0.0 for k in FEATURE_WEIGHTS}

        # ML 先跑（投票内若启用 ML 可复用今日缓存）
        ml_prediction = None
        ml_backtest_result = None
        fusion_result = None
        if enable_ml:
            try:
                from .ml import (
                    predict_with_ml, backtest_ml, TRAINING_WINDOW as _ML_TW,
                )
                ml_prediction = predict_with_ml(
                    analyzer.history_data, force_retrain=False
                )
                if enable_backtest:
                    ml_trials = min(
                        ML_BACKTEST_TRIALS,
                        max(3, len(analyzer.history_data) - _ML_TW),
                    )
                    ml_backtest_result = backtest_ml(
                        analyzer.history_data, trials=ml_trials
                    )
            except Exception as e:
                log.warning(f"ML模型预测失败（不影响整体功能）: {e}")

        # 多模型投票一次；多策略推荐组间互斥，避免主推/均衡/排名三组重号
        # 快速页面/刷新路径必须真正跳过 ML。此前即使 enable_ml=False，
        # multi_model_voting 仍会冷启动 CatBoost，生产机器可能额外耗时数十秒。
        voting = analyzer.multi_model_voting(
            front_n=20,
            back_n=10,
            skip_ml=not enable_ml,
        )
        multi = analyzer.generate_multi_strategy_recommendations(voting_result=voting)
        recommendations = {}
        for item in multi.get('recommendations') or []:
            key = item.get('strategy') or item.get('method') or 'unknown'
            recommendations[key] = {
                'front': item.get('front', []),
                'back': item.get('back', []),
                'method': key,
                'label': item.get('method'),
                'core_front': item.get('core_front', []),
                'core_back': item.get('core_back', []),
                'based_on_issue': item.get('based_on_issue'),
            }

        # ML推荐 (v2.2新增策略)
        if ml_prediction and ml_prediction.get('front_top'):
            front_top5 = ml_prediction['front_top'][:5]
            back_top2 = ml_prediction['back_top'][:2]
            recommendations['ml'] = {
                'front': front_top5,
                'back': back_top2,
                'method': 'ml',
                'front_probs': ml_prediction.get('front_probs', {}),
                'back_probs': ml_prediction.get('back_probs', {}),
                'front_model_scores': ml_prediction.get('front_model_scores', {}),
                'back_model_scores': ml_prediction.get('back_model_scores', {}),
            }

        # v3.3: 规则+ML 融合推荐
        if enable_fusion and enable_ml and ml_prediction and ml_prediction.get('front_top'):
            try:
                front_ranked, back_ranked = analyzer.rank_model(top_n=35)
                rule_w, ml_w = compute_fusion_weights(backtest, ml_backtest_result or {})
                fusion_result = fuse_rule_ml(
                    front_ranked, back_ranked, ml_prediction,
                    rule_weight=rule_w, ml_weight=ml_w
                )
                recommendations['fusion'] = {
                    'front': fusion_result['front_top12'][:5],
                    'back': fusion_result['back_top6'][:2],
                    'method': 'fusion',
                    'front_top12': fusion_result['front_top12'],
                    'back_top6': fusion_result['back_top6'],
                    'front_fused': fusion_result['front_ranked'][:20],
                    'back_fused': fusion_result['back_ranked'][:10],
                    'fusion_weights': {
                        'rule': fusion_result['rule_weight'],
                        'ml': fusion_result['ml_weight'],
                    },
                }
            except Exception as e:
                log.warning(f"规则+ML融合失败: {e}")

        # 保存线上预测记录（对下一期的预测，不是已开奖期号）
        latest_issue = data_quality.get('latest_issue', '')
        if latest_issue and not analyzer.using_simulated_data:
            try:
                next_issue = str(int(latest_issue) + 1).zfill(len(latest_issue))
                save_online_prediction(
                    next_issue,
                    recommendations,
                    fusion_result,
                    based_on_issue=latest_issue,
                )
            except Exception as e:
                log.warning(f"保存预测记录失败: {e}")

        # 结算待回填的预测
        try:
            settled = settle_predictions(analyzer.history_data)
            if settled > 0:
                log.info(f"已结算 {settled} 条大乐透预测")
        except Exception as e:
            log.warning(f"结算预测失败: {e}")

        # 线上统计
        online_stats = calculate_online_stats()

        algorithm_summary = {
            'version': LOTTERY_PREDICTOR_VERSION,
            'history_source': '500.com 全量历史 + 本地 doc_store 缓存',
            'history_issues': data_quality.get('issues'),
            'latest_issue': data_quality.get('latest_issue'),
            'latest_date': data_quality.get('latest_date'),
            'ranking_allowed': data_quality.get('ranking_allowed'),
            'scoring': [
                '前区使用衰减频率、遗漏、位置、012路、和值、趋势、区间、重号、邻号综合排名。',
                '后区使用独立的衰减频率、遗漏、位置、012路、趋势、重号、邻号与和值评分。',
                'v3.3新增规则+ML动态权重融合，基于回测表现自动分配权重。',
                'v3.3新增ML滚动回测和线上预测记录，支持闭环学习。',
            ],
            'front_weights': FEATURE_WEIGHTS,
            'back_weights': BACK_FEATURE_WEIGHTS,
            'portfolio_policy': multi.get('portfolio_policy'),
            'rolling_backtest': {
                'trials': backtest.get('trials'),
                'baseline_comparison': backtest.get('baseline_comparison'),
                'note': backtest.get('note'),
            },
            'ml_backtest': ml_backtest_result,
            'fusion_weights': {
                'rule': fusion_result['rule_weight'] if fusion_result else 0.55,
                'ml': fusion_result['ml_weight'] if fusion_result else 0.45,
            } if fusion_result else None,
        }

        result = {
            'statistics': stats,
            'recent_results': recent,
            'backtest': backtest,
            'voting': voting,
            'recommendations': recommendations,
            'portfolio_policy': multi.get('portfolio_policy'),
            'back_coverage_profile': multi.get('back_coverage_profile'),
            'data_quality': data_quality,
            'algorithm_summary': algorithm_summary,
            'optimized_weights': optimized_weights,
            'weight_adjustment': weight_diff,
            'ml_prediction': ml_prediction,
            'ml_backtest': ml_backtest_result,
            'fusion': fusion_result,
            'online_stats': online_stats,
            'prediction_records': list(reversed(load_online_predictions()[-20:])),
            'version': LOTTERY_PREDICTOR_VERSION,
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
