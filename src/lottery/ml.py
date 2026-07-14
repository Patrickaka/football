#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透 ML 预测器 V1.0 - 多模型集成版（CatBoost/XGBoost/LightGBM + 特征选择 + 动态权重）
================================================================================

大乐透规则：
- 前区：从01-35中选择5个号码
- 后区：从01-12中选择2个号码

核心思路：对每个号码做二分类预测（是否会在下期出现），再按概率排序取 Top-N。

特征工程（约30+特征）：
  1. 全局热度得分（指数衰减加权频率）
  2. 各位置热度得分（前区5个位置）
  3. 马尔可夫一阶转移概率
  4. 马尔可夫二阶转移概率
  5. 遗漏值（全局）
  6. 遗漏值 / 平均遗漏比值
  7. 遗漏标准差
  8. 近5期命中率
  9. 近10期命中率
  10. 和值兼容性
  11. 012路兼容性
  12. 区间兼容性
  13. AC值影响
  14. 升温降温趋势得分
  15. 与上期重号概率
  16. 邻号（±1）在上期出现概率
  17. 奇偶属性匹配
  18. 大小属性匹配
  19. 尾数特征
  20. 期号周期性（周几开奖模式）

训练策略：
  - 正例：历史每一期中实际出现的号码（前区5个/后区2个）
  - 负例：同一期中未出现的号码
  - 时间衰减加权：最近期数权重更高
  - 时序验证：前80%训练，后20%验证
  - 滚动训练窗口：最近120期

预测：
  - 对所有前区号码(1-35)和后区号码(1-12)分别预测出现概率
  - 使用动态权重集成多模型预测
  - 取 Top-N 作为推荐候选
"""

import math
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('lottery_ml')

# 尝试导入机器学习库
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception as e:
    HAS_LIGHTGBM = False
    lgb = None
    log_import_lightgbm_error = e

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception as e:
    HAS_CATBOOST = False
    log_import_catboost_error = e

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False
    log_import_xgboost_error = e

try:
    from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except Exception as e:
    HAS_SKLEARN = False
    log_import_sklearn_error = e

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ===================== 常量配置 =====================
FRONT_NUMBERS = list(range(1, 36))
BACK_NUMBERS = list(range(1, 13))

# 训练参数
TRAIN_RATIO = 0.8
NEGATIVE_SAMPLES_RATIO = 1.0  # 负例:正例 比例 (前区: 30:5=6:1, 后区: 10:2=5:1)
TOP_K_FRONT = 12  # 前区推荐候选数
TOP_K_BACK = 6    # 后区推荐候选数
FEATURE_SUBSET_RATIO = 0.8
MIN_VARIANCE = 0.001
TRAINING_WINDOW = 120  # 滚动训练窗口

# 时间衰减
TIME_DECAY_FACTOR = 0.92
TIME_DECAY_RECENT = 30
TIME_DECAY_RECENT_WEIGHT = 1.5
TIME_DECAY_MID = 60
TIME_DECAY_MID_WEIGHT = 1.2
TIME_DECAY_OLD_WEIGHT = 1.0

LOTTERY_ML_VERSION = "dlt-ml-v3.4-recent-window"


def _native_number(x):
    """numpy 标量 → Python int/float"""
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, float):
        return float(x)
    return x


# ===================== 特征工程 =====================

class DaletouFeatureEngineer:
    """大乐透特征工程"""

    # 质数集合（1-35内的质数）
    PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

    def __init__(self, history_data: List[Dict], window: int = 90):
        """
        Args:
            history_data: 历史开奖数据，格式 [{issue, front, back, date}, ...]
                          idx=0 是最新一期
            window: 特征计算的历史窗口（期数）
        """
        self.history = history_data
        self.window = min(window, len(history_data))
        self.recent = history_data[:self.window] if len(history_data) >= self.window else list(history_data)
        self._precompute()

    def _precompute(self):
        """预计算统计量"""
        n = len(self.recent)

        # 全局衰减频率（前区）
        self.front_freq_decayed = defaultdict(float)
        self.front_freq_raw = defaultdict(int)
        for idx, r in enumerate(self.recent):
            decay = TIME_DECAY_FACTOR ** idx
            for num in r['front']:
                self.front_freq_decayed[num] += decay
                self.front_freq_raw[num] += 1
        self.front_total_decayed = sum(self.front_freq_decayed.values()) or 1.0

        # 全局衰减频率（后区）
        self.back_freq_decayed = defaultdict(float)
        self.back_freq_raw = defaultdict(int)
        for idx, r in enumerate(self.recent):
            decay = TIME_DECAY_FACTOR ** idx
            for num in r['back']:
                self.back_freq_decayed[num] += decay
                self.back_freq_raw[num] += 1
        self.back_total_decayed = sum(self.back_freq_decayed.values()) or 1.0

        # 分位衰减频率（前区5个位置）
        self.pos_freq_decayed = [defaultdict(float) for _ in range(5)]
        self.pos_total_decayed = [0.0] * 5
        for idx, r in enumerate(self.recent):
            decay = TIME_DECAY_FACTOR ** idx
            for pos in range(5):
                num = r['front'][pos]
                self.pos_freq_decayed[pos][num] += decay
                self.pos_total_decayed[pos] += decay
        for pos in range(5):
            self.pos_total_decayed[pos] = self.pos_total_decayed[pos] or 1.0

        # 分位衰减频率（后区2个位置）
        self.back_pos_freq_decayed = [defaultdict(float) for _ in range(2)]
        self.back_pos_total_decayed = [0.0] * 2
        for idx, r in enumerate(self.recent):
            decay = TIME_DECAY_FACTOR ** idx
            for pos in range(2):
                num = r['back'][pos]
                self.back_pos_freq_decayed[pos][num] += decay
                self.back_pos_total_decayed[pos] += decay
        for pos in range(2):
            self.back_pos_total_decayed[pos] = self.back_pos_total_decayed[pos] or 1.0

        # 遗漏值
        self.front_gap = {}
        self.back_gap = {}
        for num in FRONT_NUMBERS:
            self.front_gap[num] = n  # 默认=总期数
            for idx, r in enumerate(self.recent):
                if num in r['front']:
                    self.front_gap[num] = idx
                    break
        for num in BACK_NUMBERS:
            self.back_gap[num] = n
            for idx, r in enumerate(self.recent):
                if num in r['back']:
                    self.back_gap[num] = idx
                    break

        # 遗漏历史（每号码的历次遗漏间隔）
        self.front_gap_history = defaultdict(list)
        self.back_gap_history = defaultdict(list)
        for num in FRONT_NUMBERS:
            prev = -1
            for idx in range(n - 1, -1, -1):  # 从最旧到最新
                if num in self.recent[idx]['front']:
                    if prev >= 0:
                        self.front_gap_history[num].append(idx - prev)
                    prev = idx
            if not self.front_gap_history[num]:
                self.front_gap_history[num] = [n]
        for num in BACK_NUMBERS:
            prev = -1
            for idx in range(n - 1, -1, -1):
                if num in self.recent[idx]['back']:
                    if prev >= 0:
                        self.back_gap_history[num].append(idx - prev)
                    prev = idx
            if not self.back_gap_history[num]:
                self.back_gap_history[num] = [n]

        # 平均遗漏
        self.front_avg_gap = sum(self.front_gap.values()) / len(FRONT_NUMBERS)
        self.back_avg_gap = sum(self.back_gap.values()) / len(BACK_NUMBERS)

        # 遗漏标准差
        self.front_gap_std = {}
        self.back_gap_std = {}
        for num in FRONT_NUMBERS:
            gaps = self.front_gap_history[num]
            if len(gaps) >= 2:
                mean = sum(gaps) / len(gaps)
                self.front_gap_std[num] = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
            else:
                self.front_gap_std[num] = 5.0
        for num in BACK_NUMBERS:
            gaps = self.back_gap_history[num]
            if len(gaps) >= 2:
                mean = sum(gaps) / len(gaps)
                self.back_gap_std[num] = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
            else:
                self.back_gap_std[num] = 3.0

        # 马尔可夫一阶转移矩阵（前区：分位）
        self.front_markov1 = [defaultdict(Counter) for _ in range(5)]
        for i in range(len(self.history) - 1):
            curr = self.history[i]['front']
            prev = self.history[i + 1]['front']
            for pos in range(5):
                self.front_markov1[pos][prev[pos]][curr[pos]] += 1

        # 马尔可夫一阶转移矩阵（后区：分位）
        self.back_markov1 = [defaultdict(Counter) for _ in range(2)]
        for i in range(len(self.history) - 1):
            curr = self.history[i]['back']
            prev = self.history[i + 1]['back']
            for pos in range(2):
                self.back_markov1[pos][prev[pos]][curr[pos]] += 1

        # 马尔可夫二阶转移矩阵（前区：分位）
        self.front_markov2 = [defaultdict(Counter) for _ in range(5)]
        for i in range(len(self.history) - 2):
            curr = self.history[i]['front']
            prev1 = self.history[i + 1]['front']
            prev2 = self.history[i + 2]['front']
            for pos in range(5):
                key = (prev2[pos], prev1[pos])
                self.front_markov2[pos][key][curr[pos]] += 1

        # 马尔可夫二阶转移矩阵（后区：分位）
        self.back_markov2 = [defaultdict(Counter) for _ in range(2)]
        for i in range(len(self.history) - 2):
            curr = self.history[i]['back']
            prev1 = self.history[i + 1]['back']
            prev2 = self.history[i + 2]['back']
            for pos in range(2):
                key = (prev2[pos], prev1[pos])
                self.back_markov2[pos][key][curr[pos]] += 1

        # 和值统计
        self.front_sums = [sum(r['front']) for r in self.recent]
        self.front_sum_avg = sum(self.front_sums) / len(self.front_sums) if self.front_sums else 90

        # 和值频次
        self.front_sum_freq = Counter(self.front_sums)

        # 近5期/近10期命中率（前区）
        self.front_hit_rate_5 = {}
        self.front_hit_rate_10 = {}
        for num in FRONT_NUMBERS:
            hits_5 = sum(1 for r in self.recent[:5] if num in r['front']) / min(5, n)
            hits_10 = sum(1 for r in self.recent[:10] if num in r['front']) / min(10, n)
            self.front_hit_rate_5[num] = hits_5
            self.front_hit_rate_10[num] = hits_10

        # 近5期/近10期命中率（后区）
        self.back_hit_rate_5 = {}
        self.back_hit_rate_10 = {}
        for num in BACK_NUMBERS:
            hits_5 = sum(1 for r in self.recent[:5] if num in r['back']) / min(5, n)
            hits_10 = sum(1 for r in self.recent[:10] if num in r['back']) / min(10, n)
            self.back_hit_rate_5[num] = hits_5
            self.back_hit_rate_10[num] = hits_10

        # 升温降温轨迹
        self.front_temperature = {}
        for num in FRONT_NUMBERS:
            m = min(5, n // 2) if n >= 10 else min(3, n)
            recent_hits = sum(1 for r in self.recent[:m] if num in r['front'])
            prior_hits = sum(1 for r in self.recent[m:m * 2] if num in r['front'])
            if recent_hits > prior_hits:
                self.front_temperature[num] = 1  # rising
            elif recent_hits < prior_hits:
                self.front_temperature[num] = -1  # falling
            else:
                self.front_temperature[num] = 0  # stable

        self.back_temperature = {}
        for num in BACK_NUMBERS:
            m = min(5, n // 2) if n >= 10 else min(3, n)
            recent_hits = sum(1 for r in self.recent[:m] if num in r['back'])
            prior_hits = sum(1 for r in self.recent[m:m * 2] if num in r['back'])
            if recent_hits > prior_hits:
                self.back_temperature[num] = 1
            elif recent_hits < prior_hits:
                self.back_temperature[num] = -1
            else:
                self.back_temperature[num] = 0

        # 012路统计
        self.front_road_total = [0, 0, 0]
        for r in self.recent:
            for num in r['front']:
                self.front_road_total[num % 3] += 1

        self.back_road_total = [0, 0, 0]
        for r in self.recent:
            for num in r['back']:
                self.back_road_total[num % 3] += 1

        # 区间统计
        self.front_zone_total = [0, 0, 0]  # 1-12, 13-24, 25-35
        for r in self.recent:
            for num in r['front']:
                if num <= 12:
                    self.front_zone_total[0] += 1
                elif num <= 24:
                    self.front_zone_total[1] += 1
                else:
                    self.front_zone_total[2] += 1

        # 奇偶统计
        self.front_odd_total = 0
        self.front_even_total = 0
        for r in self.recent:
            for num in r['front']:
                if num % 2 == 1:
                    self.front_odd_total += 1
                else:
                    self.front_even_total += 1

        # 大小统计（小:1-17, 大:18-35）
        self.front_small_total = 0
        self.front_big_total = 0
        for r in self.recent:
            for num in r['front']:
                if num <= 17:
                    self.front_small_total += 1
                else:
                    self.front_big_total += 1

        # AC值统计
        self.front_ac_values = []
        for r in self.recent:
            ac = self._calc_ac(r['front'])
            self.front_ac_values.append(ac)
        self.front_ac_avg = sum(self.front_ac_values) / len(self.front_ac_values) if self.front_ac_values else 7

        # 连号统计
        self.front_consecutive_rate = 0
        for r in self.recent:
            cp = sum(1 for j in range(4) if r['front'][j + 1] - r['front'][j] == 1)
            if cp > 0:
                self.front_consecutive_rate += 1
        self.front_consecutive_rate /= n if n else 1

        # 重号统计
        self.front_duplicate_rate = 0
        for i in range(len(self.recent) - 1):
            dup = len(set(self.recent[i]['front']) & set(self.recent[i + 1]['front']))
            self.front_duplicate_rate += dup
        self.front_duplicate_rate /= (n - 1) if n > 1 else 1

        # 期号星期模式（大乐透每周一三六开奖）
        self.weekday_pattern = defaultdict(float)
        for idx, r in enumerate(self.recent):
            date_str = r.get('date', '')
            if date_str:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                    weekday = dt.weekday()  # 0=Mon, 1=Tue, ...
                    decay = TIME_DECAY_FACTOR ** idx
                    self.weekday_pattern[weekday] += decay
                except Exception:
                    pass

    @staticmethod
    def _calc_ac(numbers):
        """计算AC值"""
        diffs = set()
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                diffs.add(abs(numbers[i] - numbers[j]))
        return len(diffs) - (len(numbers) - 1)

    def _markov_prob(self, trans_dict, prev_state, all_states, alpha=1.0):
        """拉普拉斯平滑的马尔可夫转移概率"""
        row = trans_dict.get(prev_state, Counter())
        total = sum(row.values())
        denom = total + alpha * len(all_states)
        return {s: (row.get(s, 0) + alpha) / denom for s in all_states}

    def build_front_features(self, num: int) -> List[float]:
        """为前区号码构建特征向量"""
        features = []
        n = len(self.recent)
        last_draw = self.history[0]['front'] if self.history else None
        last2_draw = self.history[1]['front'] if len(self.history) > 1 else None

        # 1. 全局热度得分 (1个特征)
        features.append(self.front_freq_decayed.get(num, 0) / self.front_total_decayed)

        # 2. 分位热度得分 (5个特征)
        for pos in range(5):
            features.append(self.pos_freq_decayed[pos].get(num, 0) / self.pos_total_decayed[pos])

        # 3. 马尔可夫一阶转移概率 (5个特征)
        for pos in range(5):
            if last_draw:
                prob = self._markov_prob(
                    self.front_markov1[pos], last_draw[pos], FRONT_NUMBERS
                ).get(num, 0.1)
                features.append(prob)
            else:
                features.append(0.1)

        # 4. 马尔可夫二阶转移概率 (5个特征)
        for pos in range(5):
            if last_draw and last2_draw:
                key = (last2_draw[pos], last_draw[pos])
                prob = self._markov_prob(
                    self.front_markov2[pos], key, FRONT_NUMBERS
                ).get(num, 0.05)
                features.append(prob)
            else:
                features.append(0.05)

        # 5. 遗漏值 (1个特征)
        gap = self.front_gap.get(num, n)
        features.append(gap)

        # 6. 遗漏/平均遗漏比值 (1个特征)
        features.append(gap / max(self.front_avg_gap, 1))

        # 7. 遗漏标准差 (1个特征)
        features.append(self.front_gap_std.get(num, 5.0))

        # 8. 近5期命中率 (1个特征)
        features.append(self.front_hit_rate_5.get(num, 0))

        # 9. 近10期命中率 (1个特征)
        features.append(self.front_hit_rate_10.get(num, 0))

        # 10. 和值兼容性 (1个特征)
        # 如果包含该号码，和值偏离平均值的程度
        if last_draw:
            projected_sum = sum(last_draw) - min(last_draw) + num  # 替换最小值加入该号码
        else:
            projected_sum = self.front_sum_avg
        features.append(1.0 - abs(projected_sum - self.front_sum_avg) / 100)

        # 11. 和值频次 (1个特征)
        features.append(self.front_sum_freq.get(sum(last_draw) if last_draw else int(self.front_sum_avg), 0) / n)

        # 12. 012路兼容性 (1个特征)
        road = num % 3
        road_max = max(self.front_road_total) if self.front_road_total else 1
        features.append(self.front_road_total[road] / road_max)

        # 13. 区间兼容性 (1个特征)
        if num <= 12:
            zone_idx = 0
        elif num <= 24:
            zone_idx = 1
        else:
            zone_idx = 2
        zone_max = max(self.front_zone_total) if self.front_zone_total else 1
        features.append(self.front_zone_total[zone_idx] / zone_max)

        # 14. AC值影响 (1个特征)
        # 该号码加入后对AC值的贡献（离散度）
        if last_draw:
            test_set = list(set(last_draw) | {num})
            ac_with = self._calc_ac(sorted(test_set)[:5]) if len(test_set) >= 5 else 0
        else:
            ac_with = 0
        features.append(ac_with / max(self.front_ac_avg, 1))

        # 15. 升温降温趋势 (1个特征)
        features.append(self.front_temperature.get(num, 0))

        # 16. 与上期重号概率 (1个特征)
        if last_draw:
            features.append(1.0 if num in last_draw else 0.0)
        else:
            features.append(0.0)

        # 17. 邻号在上期出现 (1个特征)
        if last_draw:
            neighbors = {num - 1, num + 1}
            neighbors = {n for n in neighbors if 1 <= n <= 35}
            features.append(len(neighbors & set(last_draw)) / max(len(neighbors), 1))
        else:
            features.append(0.0)

        # 18. 奇偶属性匹配 (1个特征)
        odd_ratio = self.front_odd_total / max(self.front_odd_total + self.front_even_total, 1)
        if num % 2 == 1:
            features.append(odd_ratio)
        else:
            features.append(1 - odd_ratio)

        # 19. 大小属性匹配 (1个特征)
        small_ratio = self.front_small_total / max(self.front_small_total + self.front_big_total, 1)
        if num <= 17:
            features.append(small_ratio)
        else:
            features.append(1 - small_ratio)

        # 20. 尾数特征 (1个特征)
        tail = num % 10
        tail_freq = defaultdict(float)
        for idx, r in enumerate(self.recent):
            decay = TIME_DECAY_FACTOR ** idx
            for n2 in r['front']:
                tail_freq[n2 % 10] += decay
        tail_total = sum(tail_freq.values()) or 1.0
        features.append(tail_freq.get(tail, 0) / tail_total)

        # 21. 期号周期性 (1个特征)
        # 大乐透周一三六开奖，编码为 0,2,5
        features.append(sum(self.weekday_pattern.get(wd, 0) for wd in [0, 2, 5]) /
                        max(sum(self.weekday_pattern.values()), 1))

        # 22. 遗漏回补指数 (1个特征)
        # 综合遗漏偏离度和标准差的回补评分
        gap_ratio = gap / max(self.front_avg_gap, 1)
        if gap_ratio < 0.7:
            rebound = 0.45
        elif gap_ratio < 1.3:
            rebound = 0.85
        else:
            rebound = min(0.70, 0.35 + 0.35 * (1.0 - math.exp(-(gap_ratio - 1.3) * 1.2)))
        features.append(rebound)

        # 23. 号码跨期出现间隔模式 (1个特征)
        # 最近3次遗漏的平均值 vs 历史平均遗漏
        gap_hist = self.front_gap_history.get(num, [n])
        if len(gap_hist) >= 3:
            recent_3_avg = sum(gap_hist[-3:]) / 3
            features.append(recent_3_avg / max(self.front_avg_gap, 1))
        else:
            features.append(1.0)

        # 24. 3期移动平均热度 (1个特征)
        ma3 = sum(1 for r in self.recent[:3] if num in r['front']) / 3
        features.append(ma3)

        # 25. 质数属性 (1个特征)
        features.append(1.0 if num in self.PRIMES else 0.0)

        # 26. 号码极值距离 (1个特征)
        # 该号码距离前区号码中心(18)的距离
        features.append(1.0 - abs(num - 18) / 17)

        # 27. 二阶马尔可夫综合得分 (1个特征)
        # 各位置二阶马尔可夫概率之和（该号码在所有位置的概率总和）
        markov2_sum = 0
        if last_draw and last2_draw:
            for pos in range(5):
                key = (last2_draw[pos], last_draw[pos])
                prob = self._markov_prob(
                    self.front_markov2[pos], key, FRONT_NUMBERS
                ).get(num, 0.05)
                markov2_sum += prob
        features.append(markov2_sum)

        # 28. 连号邻接概率 (1个特征)
        # 与上期号码形成连号的可能性
        if last_draw:
            adj_count = sum(1 for ld in last_draw if abs(num - ld) == 1)
            features.append(adj_count / 5)
        else:
            features.append(0.0)

        return features

    def build_back_features(self, num: int) -> List[float]:
        """为后区号码构建特征向量"""
        features = []
        n = len(self.recent)
        last_draw = self.history[0]['back'] if self.history else None
        last2_draw = self.history[1]['back'] if len(self.history) > 1 else None

        # 1. 全局热度得分 (1个特征)
        features.append(self.back_freq_decayed.get(num, 0) / self.back_total_decayed)

        # 2. 分位热度得分 (2个特征)
        for pos in range(2):
            features.append(self.back_pos_freq_decayed[pos].get(num, 0) / self.back_pos_total_decayed[pos])

        # 3. 马尔可夫一阶转移概率 (2个特征)
        for pos in range(2):
            if last_draw:
                prob = self._markov_prob(
                    self.back_markov1[pos], last_draw[pos], BACK_NUMBERS
                ).get(num, 0.1)
                features.append(prob)
            else:
                features.append(0.1)

        # 4. 马尔可夫二阶转移概率 (2个特征)
        for pos in range(2):
            if last_draw and last2_draw:
                key = (last2_draw[pos], last_draw[pos])
                prob = self._markov_prob(
                    self.back_markov2[pos], key, BACK_NUMBERS
                ).get(num, 0.05)
                features.append(prob)
            else:
                features.append(0.05)

        # 5. 遗漏值 (1个特征)
        gap = self.back_gap.get(num, n)
        features.append(gap)

        # 6. 遗漏/平均遗漏比值 (1个特征)
        features.append(gap / max(self.back_avg_gap, 1))

        # 7. 遗漏标准差 (1个特征)
        features.append(self.back_gap_std.get(num, 3.0))

        # 8. 近5期命中率 (1个特征)
        features.append(self.back_hit_rate_5.get(num, 0))

        # 9. 近10期命中率 (1个特征)
        features.append(self.back_hit_rate_10.get(num, 0))

        # 10. 012路兼容性 (1个特征)
        road = num % 3
        road_max = max(self.back_road_total) if self.back_road_total else 1
        features.append(self.back_road_total[road] / road_max)

        # 11. 升温降温趋势 (1个特征)
        features.append(self.back_temperature.get(num, 0))

        # 12. 与上期重号概率 (1个特征)
        if last_draw:
            features.append(1.0 if num in last_draw else 0.0)
        else:
            features.append(0.0)

        # 13. 邻号在上期出现 (1个特征)
        if last_draw:
            neighbors = {num - 1, num + 1}
            neighbors = {n for n in neighbors if 1 <= n <= 12}
            features.append(len(neighbors & set(last_draw)) / max(len(neighbors), 1))
        else:
            features.append(0.0)

        # 14. 奇偶属性 (1个特征)
        features.append(1.0 if num % 2 == 1 else 0.0)

        # 15. 大小属性 (1个特征) - 后区: 小1-6, 大7-12
        features.append(1.0 if num <= 6 else 0.0)

        # 16. 尾数特征 (1个特征)
        tail = num % 10
        tail_freq = defaultdict(float)
        for idx, r in enumerate(self.recent):
            decay = TIME_DECAY_FACTOR ** idx
            for n2 in r['back']:
                tail_freq[n2 % 10] += decay
        tail_total = sum(tail_freq.values()) or 1.0
        features.append(tail_freq.get(tail, 0) / tail_total)

        # 17. 遗漏回补指数 (1个特征)
        gap_ratio = gap / max(self.back_avg_gap, 1)
        if gap_ratio < 0.7:
            rebound = 0.45
        elif gap_ratio < 1.3:
            rebound = 0.85
        else:
            rebound = min(0.70, 0.35 + 0.35 * (1.0 - math.exp(-(gap_ratio - 1.3) * 1.2)))
        features.append(rebound)

        # 18. 跨期间隔模式 (1个特征)
        gap_hist = self.back_gap_history.get(num, [n])
        if len(gap_hist) >= 3:
            recent_3_avg = sum(gap_hist[-3:]) / 3
            features.append(recent_3_avg / max(self.back_avg_gap, 1))
        else:
            features.append(1.0)

        # 19. 3期移动平均热度 (1个特征)
        ma3 = sum(1 for r in self.recent[:3] if num in r['back']) / 3
        features.append(ma3)

        # 20. 号码中心距离 (1个特征)
        features.append(1.0 - abs(num - 6.5) / 5.5)

        # 21. 二阶马尔可夫综合得分 (1个特征)
        markov2_sum = 0
        if last_draw and last2_draw:
            for pos in range(2):
                key = (last2_draw[pos], last_draw[pos])
                prob = self._markov_prob(
                    self.back_markov2[pos], key, BACK_NUMBERS
                ).get(num, 0.05)
                markov2_sum += prob
        features.append(markov2_sum)

        # 22. 连号邻接概率 (1个特征)
        if last_draw:
            adj_count = sum(1 for ld in last_draw if abs(num - ld) == 1)
            features.append(adj_count / 2)
        else:
            features.append(0.0)

        # 23. 前区关联特征 (1个特征)
        # 该后区号码与上期前区号码的尾数关联度
        if last_draw and self.history:
            last_front = self.history[0]['front']
            tail_match = sum(1 for fn in last_front if fn % 10 == num % 10)
            features.append(tail_match / 5)
        else:
            features.append(0.0)

        # 24. 质数属性 (1个特征)
        primes_back = {2, 3, 5, 7, 11}
        features.append(1.0 if num in primes_back else 0.0)

        # 25. 期号周期性 (1个特征)
        features.append(sum(self.weekday_pattern.get(wd, 0) for wd in [0, 2, 5]) /
                        max(sum(self.weekday_pattern.values()), 1))

        return features

    def get_front_feature_names(self):
        """返回前区特征名称列表"""
        names = [
            "global_freq",                # 1
            "pos_0_freq", "pos_1_freq", "pos_2_freq", "pos_3_freq", "pos_4_freq",  # 2-6
            "pos_0_markov1", "pos_1_markov1", "pos_2_markov1", "pos_3_markov1", "pos_4_markov1",  # 7-11
            "pos_0_markov2", "pos_1_markov2", "pos_2_markov2", "pos_3_markov2", "pos_4_markov2",  # 12-16
            "gap", "gap_ratio", "gap_std",  # 17-19
            "hit_rate_5", "hit_rate_10",   # 20-21
            "sum_compat", "sum_freq",      # 22-23
            "road_compat", "zone_compat",  # 24-25
            "ac_impact",                   # 26
            "trend",                       # 27
            "is_repeat", "neighbor_overlap",  # 28-29
            "parity_match", "size_match",  # 30-31
            "tail_freq",                   # 32
            "weekday_pattern",             # 33
            "rebound_index",               # 34
            "gap_pattern",                 # 35
            "ma3_hit_rate",                # 36
            "is_prime",                    # 37
            "center_distance",             # 38
            "markov2_sum",                 # 39
            "adjacent_prob",               # 40
        ]
        return names

    def get_back_feature_names(self):
        """返回后区特征名称列表"""
        names = [
            "global_freq",                # 1
            "pos_0_freq", "pos_1_freq",   # 2-3
            "pos_0_markov1", "pos_1_markov1",  # 4-5
            "pos_0_markov2", "pos_1_markov2",  # 6-7
            "gap", "gap_ratio", "gap_std",  # 8-10
            "hit_rate_5", "hit_rate_10",   # 11-12
            "road_compat",                 # 13
            "trend",                       # 14
            "is_repeat", "neighbor_overlap",  # 15-16
            "is_odd", "is_small",          # 17-18
            "tail_freq",                   # 19
            "rebound_index",               # 20
            "gap_pattern",                 # 21
            "ma3_hit_rate",                # 22
            "center_distance",             # 23
            "markov2_sum",                 # 24
            "adjacent_prob",               # 25
            "front_tail_match",            # 26
            "is_prime",                    # 27
            "weekday_pattern",             # 28
        ]
        return names


# ===================== 纯 Python 随机森林 =====================

class SimpleDecisionTree:
    """简易决策树"""

    def __init__(self, max_depth=4, min_samples_split=10, feature_subset_ratio=1.0, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.rng = rng if rng else random.Random(42)
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p1 = sum(y) / len(y)
        return 1 - p1 ** 2 - (1 - p1) ** 2

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature = None
        best_threshold = None
        current_gini = self._gini(y)
        n = len(y)

        for fi in feature_indices:
            values = sorted(set(row[fi] for row in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left_y = [y[j] for j in range(n) if X[j][fi] <= threshold]
                right_y = [y[j] for j in range(n) if X[j][fi] > threshold]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain = current_gini - (len(left_y) / n * self._gini(left_y) +
                                        len(right_y) / n * self._gini(right_y))
                if gain > best_gain:
                    best_gain = gain
                    best_feature = fi
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {"leaf": True, "value": sum(y) / len(y) if y else 0.5}
        if len(set(y)) == 1:
            return {"leaf": True, "value": y[0]}
        n_features = len(X[0])
        n_select = max(1, int(n_features * self.feature_subset_ratio))
        feature_indices = self.rng.sample(range(n_features), n_select)
        best_f, best_t, best_g = self._best_split(X, y, feature_indices)
        if best_f is None or best_g <= 0:
            return {"leaf": True, "value": sum(y) / len(y) if y else 0.5}
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(y)):
            if X[i][best_f] <= best_t:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return {
            "leaf": False, "feature": best_f, "threshold": best_t,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, row, node):
        if node["leaf"]:
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        else:
            return self._predict_one(row, node["right"])

    def predict_proba(self, X):
        return [self._predict_one(row, self.tree) for row in X]


class SimpleRandomForest:
    """纯 Python 随机森林"""

    def __init__(self, n_trees=30, max_depth=4, feature_subset_ratio=0.7, rng=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.feature_subset_ratio = feature_subset_ratio
        self.rng = rng if rng else random.Random(42)
        self.trees = []

    def _weighted_choice(self, probs):
        """按概率加权随机选择索引"""
        r = self.rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probs) - 1

    def fit(self, X, y, sample_weights=None):
        n = len(y)
        for _ in range(self.n_trees):
            # Bootstrap sample (按 sample_weights 加权采样)
            if sample_weights:
                total_w = sum(sample_weights)
                probs = [w / total_w for w in sample_weights]
                indices = [self._weighted_choice(probs) for _ in range(n)]
            else:
                indices = [self.rng.randint(0, n - 1) for _ in range(n)]
            bX = [X[i] for i in indices]
            by = [y[i] for i in indices]
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=10,
                feature_subset_ratio=self.feature_subset_ratio,
                rng=random.Random(self.rng.randint(0, 1000000))
            )
            tree.fit(bX, by)
            self.trees.append(tree)

    def predict_proba(self, X):
        probs = []
        for row in X:
            p = sum(t._predict_one(row, t.tree) for t in self.trees) / len(self.trees)
            probs.append(p)
        return probs


# ===================== 特征选择 =====================

def select_features(X, y, feature_names, subset_ratio=FEATURE_SUBSET_RATIO,
                    min_variance=MIN_VARIANCE):
    """特征选择：方差过滤 + 互信息"""
    n_features = len(feature_names)

    # 1. 方差过滤
    variances = []
    for i in range(n_features):
        vals = [row[i] for row in X]
        if len(vals) < 2:
            variances.append(0)
        else:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            variances.append(var)

    kept_by_var = [i for i in range(n_features) if variances[i] >= min_variance]
    if not kept_by_var:
        kept_by_var = list(range(n_features))

    # 2. 互信息选择 (如果 sklearn 可用)
    if HAS_SKLEARN and len(X) >= 20:
        try:
            import numpy as np
            X_np = np.array(X)
            y_np = np.array(y)
            mi = mutual_info_classif(X_np[:, kept_by_var], y_np, random_state=42)
            n_keep = max(5, int(len(kept_by_var) * subset_ratio))
            top_indices = sorted(range(len(mi)), key=lambda i: -mi[i])[:n_keep]
            kept = [kept_by_var[i] for i in top_indices]
        except Exception:
            kept = kept_by_var[:max(5, int(len(kept_by_var) * subset_ratio))]
    else:
        kept = kept_by_var[:max(5, int(len(kept_by_var) * subset_ratio))]

    return kept, [feature_names[i] for i in kept]


# ===================== 多模型集成训练与预测 =====================

class DaletouMLPredictor:
    """大乐透多模型集成预测器"""

    def __init__(self, history_data: List[Dict]):
        self.history = history_data
        self.front_models = {}    # model_name -> trained model
        self.back_models = {}
        self.front_feature_indices = None
        self.back_feature_indices = None
        self.front_feature_names_selected = None
        self.back_feature_names_selected = None
        self.front_scores = {}    # model_name -> validation score
        self.back_scores = {}
        self.feature_engineer = None
        self.trained = False

    def _build_training_data(self, is_front: bool = True):
        """构建训练数据 (v3.2: 高效版 — 用单个FeatureEngineer替代每期重建)

        核心优化: 对于彩票预测，频率/遗漏/马尔可夫等特征在包含或排除
        最近一期时几乎没有差异(1期/200期≈0.5%变化)，因此使用全量历史
        计算特征即可，避免了为每期重建FeatureEngineer的巨大开销。

        对于依赖"上期号码"的特征(重号、邻号、马尔可夫转移)，我们使用
        该期之前的那一期数据作为"上期"，保证时序正确性。
        """
        numbers = FRONT_NUMBERS if is_front else BACK_NUMBERS
        n_history = len(self.history)

        # history[0]=最新。训练用最近 TRAINING_WINDOW 期（旧版错误取了最旧端）
        window_data = self.history[:TRAINING_WINDOW]
        if len(window_data) < 6:
            return [], [], []

        # 跳过最新一期作标签；对其余期用「更旧」历史建特征，保证无泄漏
        # window: [newest, ..., older]；训练样本为 window[0..-2]，特征用 window[i+1:]
        n_periods = len(window_data) - 1
        if n_periods < 5:
            return [], [], []

        fe_points = []
        if n_periods >= 50:
            fe_indices = [0, n_periods // 4, n_periods // 2, 3 * n_periods // 4, n_periods - 1]
        elif n_periods >= 30:
            fe_indices = [0, n_periods // 3, n_periods // 2, 2 * n_periods // 3, n_periods - 1]
        elif n_periods >= 15:
            fe_indices = [0, n_periods // 2, n_periods - 1]
        else:
            fe_indices = [0, n_periods - 1]

        for fe_idx in fe_indices:
            # history[0]最新：fe_idx 之后的切片是更旧历史
            fe_history = window_data[fe_idx + 1:]
            if len(fe_history) >= 10:
                fe = DaletouFeatureEngineer(fe_history, window=min(90, len(fe_history)))
                fe_points.append((fe_idx, fe))

        if not fe_points:
            return [], [], []

        X_all = []
        y_all = []
        w_all = []

        # 从旧到新写入，便于后续 TRAIN_RATIO 时序切分（后段=较近期）
        for period_idx in range(n_periods - 1, -1, -1):
            period = window_data[period_idx]
            actual = set(period['front'] if is_front else period['back'])
            # period_idx=0 仍是最新期，权重最高
            period_weight = self._time_weight(period_idx, n_periods)

            best_fe = None
            best_dist = float('inf')
            for fe_idx, fe in fe_points:
                dist = abs(period_idx - fe_idx)
                if dist < best_dist:
                    best_dist = dist
                    best_fe = fe

            if best_fe is None:
                continue

            prev_period_idx = period_idx + 1
            original_history = best_fe.history
            if prev_period_idx < len(window_data):
                temp_history = [window_data[prev_period_idx]] + list(best_fe.recent)
                best_fe.history = temp_history
                best_fe.recent = temp_history[:best_fe.window] if len(temp_history) >= best_fe.window else temp_history

            for num in numbers:
                if is_front:
                    features = best_fe.build_front_features(num)
                else:
                    features = best_fe.build_back_features(num)

                X_all.append(features)
                y_all.append(1 if num in actual else 0)
                w_all.append(period_weight * (2.0 if num in actual else 1.0))

            best_fe.history = original_history
            best_fe.recent = original_history[:best_fe.window] if len(original_history) >= best_fe.window else list(original_history)

        return X_all, y_all, w_all

    def _time_weight(self, idx, total):
        """时间衰减权重。history 为 newest-first：idx 越小越近。"""
        if idx <= TIME_DECAY_RECENT:
            return TIME_DECAY_RECENT_WEIGHT
        elif idx <= TIME_DECAY_MID:
            return TIME_DECAY_MID_WEIGHT
        else:
            return TIME_DECAY_OLD_WEIGHT

    def train(self):
        """训练所有可用模型"""
        log.info("大乐透 ML: 开始训练...")

        # 前区训练
        X_front, y_front, w_front = self._build_training_data(is_front=True)
        if len(X_front) < 50:
            log.warning("大乐透 ML: 前区训练数据不足，跳过ML训练")
            return False

        # 特征选择
        fe_sample = DaletouFeatureEngineer(self.history, window=90)
        front_names = fe_sample.get_front_feature_names()
        kept_front, selected_front_names = select_features(
            X_front, y_front, front_names
        )
        self.front_feature_indices = kept_front
        self.front_feature_names_selected = selected_front_names

        X_front_selected = [[row[i] for i in kept_front] for row in X_front]

        # 时序划分
        split = int(len(X_front_selected) * TRAIN_RATIO)
        X_train_f = X_front_selected[:split]
        y_train_f = y_front[:split]
        w_train_f = w_front[:split]
        X_val_f = X_front_selected[split:]
        y_val_f = y_front[split:]

        # 训练前区模型
        self.front_models, self.front_scores = self._train_models(
            X_train_f, y_train_f, w_train_f, X_val_f, y_val_f, "前区"
        )

        # 后区训练
        X_back, y_back, w_back = self._build_training_data(is_front=False)
        if len(X_back) < 50:
            log.warning("大乐透 ML: 后区训练数据不足，跳过ML训练")
            return False

        back_names = fe_sample.get_back_feature_names()
        kept_back, selected_back_names = select_features(
            X_back, y_back, back_names
        )
        self.back_feature_indices = kept_back
        self.back_feature_names_selected = selected_back_names

        X_back_selected = [[row[i] for i in kept_back] for row in X_back]

        split = int(len(X_back_selected) * TRAIN_RATIO)
        X_train_b = X_back_selected[:split]
        y_train_b = y_back[:split]
        w_train_b = w_back[:split]
        X_val_b = X_back_selected[split:]
        y_val_b = y_back[split:]

        self.back_models, self.back_scores = self._train_models(
            X_train_b, y_train_b, w_train_b, X_val_b, y_val_b, "后区"
        )

        self.trained = True
        log.info(f"大乐透 ML: 训练完成 - 前区模型: {list(self.front_models.keys())}, "
                 f"后区模型: {list(self.back_models.keys())}")
        return True

    def _train_models(self, X_train, y_train, w_train, X_val, y_val, area_name):
        """训练多个模型"""
        models = {}
        scores = {}

        # CatBoost
        if HAS_CATBOOST:
            try:
                model = CatBoostClassifier(
                    iterations=50,
                    depth=3,
                    learning_rate=0.05,
                    l2_leaf_reg=20,
                    random_strength=2,
                    verbose=0,
                    auto_class_weights='Balanced',
                    random_seed=42,
                )
                model.fit(X_train, y_train, sample_weight=w_train)
                val_pred = model.predict_proba(X_val)[:, 1]
                score = self._calc_auc(y_val, val_pred)
                models['catboost'] = model
                scores['catboost'] = score
                log.info(f"大乐透 ML {area_name}: CatBoost AUC={score:.4f}")
            except Exception as e:
                log.warning(f"大乐透 ML {area_name}: CatBoost训练失败: {e}")

        # XGBoost
        if HAS_XGBOOST:
            try:
                model = XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    reg_lambda=20,
                    reg_alpha=1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    scale_pos_weight=6.0 if area_name == "前区" else 5.0,
                )
                model.fit(X_train, y_train, sample_weight=w_train)
                val_pred = model.predict_proba(X_val)[:, 1]
                score = self._calc_auc(y_val, val_pred)
                models['xgboost'] = model
                scores['xgboost'] = score
                log.info(f"大乐透 ML {area_name}: XGBoost AUC={score:.4f}")
            except Exception as e:
                log.warning(f"大乐透 ML {area_name}: XGBoost训练失败: {e}")

        # LightGBM
        if HAS_LIGHTGBM:
            try:
                model = lgb.LGBMClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    reg_lambda=20,
                    num_leaves=8,
                    min_child_samples=20,
                    random_state=42,
                    is_unbalance=True,
                    verbose=-1,
                )
                model.fit(X_train, y_train, sample_weight=w_train)
                val_pred = model.predict_proba(X_val)[:, 1]
                score = self._calc_auc(y_val, val_pred)
                models['lightgbm'] = model
                scores['lightgbm'] = score
                log.info(f"大乐透 ML {area_name}: LightGBM AUC={score:.4f}")
            except Exception as e:
                log.warning(f"大乐透 ML {area_name}: LightGBM训练失败: {e}")

        # 纯Python随机森林：仅当外部 GBDT 全部不可用时兜底
        if not models:
            try:
                n_samples = min(800, len(X_train))
                if len(X_train) > n_samples:
                    rng_sub = random.Random(42)
                    indices = rng_sub.sample(range(len(X_train)), n_samples)
                    X_sub = [X_train[i] for i in indices]
                    y_sub = [y_train[i] for i in indices]
                    w_sub = [w_train[i] for i in indices] if w_train else None
                else:
                    X_sub, y_sub, w_sub = X_train, y_train, w_train

                model = SimpleRandomForest(n_trees=15, max_depth=3, feature_subset_ratio=0.6)
                model.fit(X_sub, y_sub, sample_weights=w_sub)
                val_pred = model.predict_proba(X_val)
                score = self._calc_auc(y_val, val_pred)
                models['random_forest'] = model
                scores['random_forest'] = score
                log.info(f"大乐透 ML {area_name}: RandomForest AUC={score:.4f}")
            except Exception as e:
                log.warning(f"大乐透 ML {area_name}: RandomForest训练失败: {e}")

        return models, scores

    def _calc_auc(self, y_true, y_pred):
        """计算AUC（简化版）"""
        try:
            # 排序计算AUC
            pairs = list(zip(y_pred, y_true))
            pairs.sort(key=lambda x: -x[0])

            pos = sum(y_true)
            neg = len(y_true) - pos
            if pos == 0 or neg == 0:
                return 0.5

            tp = 0
            fp = 0
            auc = 0.0
            prev_fpr = 0.0
            prev_tpr = 0.0

            for pred, true in pairs:
                if true == 1:
                    tp += 1
                else:
                    fp += 1
                tpr = tp / pos
                fpr = fp / neg
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
                prev_fpr = fpr
                prev_tpr = tpr

            return max(0.0, min(1.0, auc))
        except Exception:
            return 0.5

    def predict(self) -> Dict:
        """预测下期号码概率

        Returns:
            {
                'front_probs': {num: probability},
                'back_probs': {num: probability},
                'front_top': [top numbers],
                'back_top': [top numbers],
                'front_model_scores': {model: auc},
                'back_model_scores': {model: auc},
                'version': str,
            }
        """
        if not self.trained:
            log.warning("大乐透 ML: 模型未训练，使用默认预测")
            return self._default_predict()

        # 构建当前特征
        fe = DaletouFeatureEngineer(self.history, window=90)
        self.feature_engineer = fe

        # 前区预测
        front_probs = {}
        for num in FRONT_NUMBERS:
            raw_features = fe.build_front_features(num)
            selected = [raw_features[i] for i in self.front_feature_indices]
            prob = self._ensemble_predict(selected, self.front_models, self.front_scores)
            front_probs[num] = _native_number(prob)

        # 后区预测
        back_probs = {}
        for num in BACK_NUMBERS:
            raw_features = fe.build_back_features(num)
            selected = [raw_features[i] for i in self.back_feature_indices]
            prob = self._ensemble_predict(selected, self.back_models, self.back_scores)
            back_probs[num] = _native_number(prob)

        front_top = sorted(front_probs.keys(), key=lambda n: -front_probs[n])[:TOP_K_FRONT]
        back_top = sorted(back_probs.keys(), key=lambda n: -back_probs[n])[:TOP_K_BACK]

        return {
            'front_probs': front_probs,
            'back_probs': back_probs,
            'front_top': front_top,
            'back_top': back_top,
            'front_model_scores': {k: _native_number(v) for k, v in self.front_scores.items()},
            'back_model_scores': {k: _native_number(v) for k, v in self.back_scores.items()},
            'front_selected_features': self.front_feature_names_selected,
            'back_selected_features': self.back_feature_names_selected,
            'version': LOTTERY_ML_VERSION,
        }

    def _ensemble_predict(self, features, models, scores):
        """动态权重集成预测"""
        if not models:
            return 0.5

        total_score = sum(scores.values()) or 1.0
        weights = {k: v / total_score for k, v in scores.items()}

        ensemble_prob = 0.0
        for name, model in models.items():
            w = weights.get(name, 0.25)
            try:
                if name in ('catboost', 'xgboost', 'lightgbm'):
                    prob = model.predict_proba([features])[0][1]
                elif name == 'random_forest':
                    prob = model.predict_proba([features])[0]
                else:
                    prob = 0.5
                ensemble_prob += w * prob
            except Exception:
                ensemble_prob += w * 0.5

        return ensemble_prob

    def _default_predict(self):
        """默认预测（基于简单统计）"""
        fe = DaletouFeatureEngineer(self.history, window=90)

        front_probs = {}
        for num in FRONT_NUMBERS:
            front_probs[num] = fe.front_freq_decayed.get(num, 0) / fe.front_total_decayed

        back_probs = {}
        for num in BACK_NUMBERS:
            back_probs[num] = fe.back_freq_decayed.get(num, 0) / fe.back_total_decayed

        front_top = sorted(front_probs.keys(), key=lambda n: -front_probs[n])[:TOP_K_FRONT]
        back_top = sorted(back_probs.keys(), key=lambda n: -back_probs[n])[:TOP_K_BACK]

        return {
            'front_probs': front_probs,
            'back_probs': back_probs,
            'front_top': front_top,
            'back_top': back_top,
            'front_model_scores': {},
            'back_model_scores': {},
            'version': f"{LOTTERY_ML_VERSION}-fallback",
        }


# ===================== ML 回测系统 =====================

def backtest_ml(history_data: List[Dict], trials: int = 40,
                train_window: int = TRAINING_WINDOW) -> Dict:
    """快速滚动回测：训一次，再对最近 trials 期逐期 predict。

    history[0]=最新。模型在 history[trials:] 的最近 train_window 期上训练，
    保证训练数据不包含回测窗口内的开奖；逐期预测时用 history[i+1:] 建特征。
    """
    n_history = len(history_data)
    if n_history < train_window + trials:
        effective_trials = max(3, n_history - train_window)
        if effective_trials < 3:
            return {'error': '历史数据不足以回测'}
        trials = effective_trials

    total_front_matched = 0
    total_back_matched = 0
    front_match_dist = {i: 0 for i in range(6)}
    back_match_dist = {i: 0 for i in range(3)}
    front_rank_errors = []

    # 一次训练：回测窗口之后（更旧）的最近 train_window 期
    train_data = history_data[trials:trials + train_window]
    if len(train_data) < 40:
        return {'error': '训练数据不足'}

    predictor = DaletouMLPredictor(train_data)
    try:
        success = predictor.train()
    except Exception as e:
        log.warning(f"ML回测训练失败: {e}")
        return {'error': f'训练失败: {e}'}

    if not success or not predictor.trained:
        return {'error': 'ML回测模型未训练成功'}

    original_history = predictor.history
    for i in range(trials):
        actual = history_data[i]
        actual_front = set(actual['front'])
        actual_back = set(actual['back'])

        # 特征用「该期之前」的历史，避免偷看当期
        predictor.history = history_data[i + 1:]
        if len(predictor.history) < 10:
            continue
        try:
            result = predictor.predict()
        except Exception as e:
            log.debug(f"ML回测第{i}期预测失败: {e}")
            continue

        front_top = result.get('front_top', [])[:TOP_K_FRONT]
        back_top = result.get('back_top', [])[:TOP_K_BACK]

        front_matched = len(set(front_top) & actual_front)
        back_matched = len(set(back_top) & actual_back)

        total_front_matched += front_matched
        total_back_matched += back_matched
        front_match_dist[front_matched] += 1
        back_match_dist[back_matched] += 1

        front_probs = result.get('front_probs', {})
        if front_probs:
            sorted_nums = sorted(front_probs.keys(), key=lambda n: -front_probs[n])
            for actual_num in actual_front:
                try:
                    rank = sorted_nums.index(actual_num) + 1
                except ValueError:
                    rank = 36
                front_rank_errors.append(rank)

    predictor.history = original_history

    n_valid = sum(front_match_dist.values())
    if n_valid == 0:
        return {'error': '回测未产生有效预测'}

    log.info(
        f"ML回测完成(单次训练): {n_valid}期, "
        f"前区≥2={sum(front_match_dist[k] for k in range(2, 6))}/{n_valid}"
    )

    def _hypergeom(pop, winners, picks):
        dist = {}
        for k in range(min(winners, picks) + 1):
            if picks - k <= pop - winners:
                dist[k] = (
                    math.comb(winners, k) * math.comb(pop - winners, picks - k)
                    / math.comb(pop, picks)
                )
        return dist

    front_baseline = _hypergeom(35, 5, TOP_K_FRONT)
    back_baseline = _hypergeom(12, 2, TOP_K_BACK)
    front_ranks = front_rank_errors if front_rank_errors else [0]

    return {
        'trials': n_valid,
        'train_window': train_window,
        'model_version': LOTTERY_ML_VERSION,
        'mode': 'train_once_roll_predict',
        'front_ge1_rate': round((n_valid - front_match_dist[0]) / n_valid, 4),
        'front_ge2_rate': round(sum(front_match_dist[k] for k in range(2, 6)) / n_valid, 4),
        'front_ge3_rate': round(sum(front_match_dist[k] for k in range(3, 6)) / n_valid, 4),
        'front_avg_matched': round(total_front_matched / n_valid, 2),
        'front_match_distribution': front_match_dist,
        'back_ge1_rate': round((n_valid - back_match_dist[0]) / n_valid, 4),
        'back_ge2_rate': round(back_match_dist.get(2, 0) / n_valid, 4),
        'back_avg_matched': round(total_back_matched / n_valid, 2),
        'back_match_distribution': back_match_dist,
        'front_rank_avg': round(sum(front_ranks) / len(front_ranks), 1),
        'front_rank_median': sorted(front_ranks)[len(front_ranks) // 2] if front_ranks else 0,
        'front_top_pool_size': TOP_K_FRONT,
        'back_top_pool_size': TOP_K_BACK,
        'baseline': {
            'front_ge1_rate': round(1 - front_baseline.get(0, 0), 4),
            'front_ge2_rate': round(sum(v for k, v in front_baseline.items() if k >= 2), 4),
            'front_ge3_rate': round(sum(v for k, v in front_baseline.items() if k >= 3), 4),
            'back_ge1_rate': round(1 - back_baseline.get(0, 0), 4),
            'back_ge2_rate': round(back_baseline.get(2, 0), 4),
        },
    }


# ===================== 对外接口 =====================

_ml_predictor = None
_ml_cache = None
_ml_cache_time = 0


def get_ml_predictor(history_data: List[Dict] = None) -> DaletouMLPredictor:
    """获取/创建 ML 预测器"""
    global _ml_predictor
    if history_data is None:
        from ..common import repositories
        history_data = repositories.dlt_load() or []
    if _ml_predictor is None or len(history_data) != len(_ml_predictor.history):
        _ml_predictor = DaletouMLPredictor(history_data)
    return _ml_predictor


def predict_with_ml(history_data: List[Dict] = None, force_retrain: bool = False) -> Dict:
    """使用 ML 模型预测大乐透

    Args:
        history_data: 历史开奖数据（可选，默认从仓储加载）
        force_retrain: 是否强制重新训练模型

    Returns:
        预测结果字典
    """
    global _ml_cache, _ml_cache_time

    # 检查缓存
    if not force_retrain and _ml_cache is not None:
        import datetime
        if _ml_cache_time > 0:
            cache_date = datetime.date.fromtimestamp(_ml_cache_time)
            if cache_date == datetime.date.today():
                log.info("大乐透 ML: 使用今日缓存")
                return _ml_cache

    predictor = get_ml_predictor(history_data)

    if not predictor.trained or force_retrain:
        success = predictor.train()
        if not success:
            log.warning("大乐透 ML: 训练失败，使用统计兜底")
            result = predictor._default_predict()
            _ml_cache = result
            _ml_cache_time = time.time()
            return result

    result = predictor.predict()
    _ml_cache = result
    _ml_cache_time = time.time()
    return result


def clear_ml_cache():
    """清除 ML 缓存"""
    global _ml_predictor, _ml_cache, _ml_cache_time
    _ml_predictor = None
    _ml_cache = None
    _ml_cache_time = 0
    log.info("大乐透 ML: 缓存已清除")
