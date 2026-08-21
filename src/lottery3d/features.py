# -*- coding: utf-8 -*-
"""福彩3D基础特征：形态/遗漏/马尔可夫/热冷/和值/对频/斜率"""

import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import combinations, product

from ..common.logger import setup_logger
from ..common.data_cache import cached_fetch
from ..common import kv_store

log = setup_logger('lottery3d')

from .config import (
    EXP_DECAY, FORM_SWITCH_WEIGHT, HOT_WINDOW, MARKOV_LAPLACE_ALPHA, MISS_CYCLE_WINDOW, MISS_OVER_BONUS, MISS_OVER_RATIO_THRESHOLD, PAIR_BONUS, PAIR_FREQ_WINDOWS, PAIR_HIGH_FREQ_THRESHOLD, POS_NAMES_3D, REBOUND_BONUS, REBOUND_THRESHOLD, RECENT_RECOMMEND_CONSECUTIVE_PENALTY, RECENT_RECOMMEND_PENALTY, RECENT_RECOMMEND_WINDOW, RECENT_WINDOW_REBOUND, SLOPE_MAX_CHAIN, SLOPE_MIN_CHAIN, SUM_EXTREME_PENALTY, SUM_INTERVAL_BONUS, SUM_INTERVAL_WIDTH, SUM_INTERVAL_WINDOW, SUM_TREND_ADJUST, SUM_TREND_WINDOW, W_SLOPE_MATCH, ZU3_STREAK_THRESHOLD, ZU6_STREAK_THRESHOLD,
)

def calc_span(n):
    return max(n) - min(n)


def miss_value(numbers, digit, position=None):
    for i in range(len(numbers) - 1, -1, -1):
        n = numbers[i]
        if position is None:
            if digit in n:
                return len(numbers) - 1 - i
        elif n[position] == digit:
            return len(numbers) - 1 - i
    return len(numbers)


def neighbor(d):
    return {(d - 1) % 10, (d + 1) % 10}


def road(d):
    return d % 3


def exp_weighted_counts(series, decay=EXP_DECAY):
    cnt = Counter()
    w = 1.0
    for item in reversed(series):
        cnt[item] += w
        w *= decay
    return cnt


def build_markov(numbers, position):
    trans = defaultdict(Counter)
    for i in range(len(numbers) - 1):
        a, b = numbers[i][position], numbers[i + 1][position]
        trans[a][b] += 1
    return trans


def build_markov2(numbers, position):
    """二阶马尔可夫转移矩阵：P(next | prev2, prev1) → Counter[(prev2, prev1)][next]"""
    trans2 = defaultdict(Counter)
    for i in range(len(numbers) - 2):
        p2, p1 = numbers[i][position], numbers[i + 1][position]
        nx = numbers[i + 2][position]
        trans2[(p2, p1)][nx] += 1
    return trans2


def markov_prob_smoothed(row, states, alpha=MARKOV_LAPLACE_ALPHA):
    """转移概率 P(next|prev)，拉普拉斯平滑：(count + α) / (total + α·|S|)"""
    states = list(states)
    row_total = sum(row.values())
    denom = row_total + alpha * len(states)
    return {s: (row.get(s, 0) + alpha) / denom for s in states}


def gaussian_score(value, center, sigma):
    if sigma <= 0:
        return 0.0
    z = (value - center) / sigma
    return math.exp(-0.5 * z * z)


def _recent_slice(series, window):
    return series[-window:] if len(series) > window else list(series)


def odd_even_key(triple):
    """奇偶比 (奇数个数, 偶数个数)"""
    odds = sum(1 for d in triple if d % 2 == 1)
    return odds, 3 - odds


def big_small_key(triple):
    """大小比 (大数个数, 小数个数)，0-4 小、5-9 大"""
    big = sum(1 for d in triple if d >= 5)
    return big, 3 - big


def ratio_label(key, kind="oe"):
    a, b = key
    if kind == "oe":
        return f"{a}奇{b}偶"
    return f"{a}大{b}小"


def has_consecutive_digits(a, b, c):
    """是否存在相邻连号（差值为 1，不含 9-0）"""
    digits = (a, b, c)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(digits[i] - digits[j]) == 1:
                return True
    return False


def has_consecutive_digits(a, b, c):
    """是否存在相邻连号（差值为 1，不含 9-0）"""
    digits = (a, b, c)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(digits[i] - digits[j]) == 1:
                return True
    return False


def _slope_step(prev_digit, cur_digit):
    """斜连步长：仅认可 ±1（不含 9↔0 绕回）。"""
    diff = cur_digit - prev_digit
    return diff if diff in (-1, 1) else None


def _detect_position_slope_chain(digits_at_pos, min_len=SLOPE_MIN_CHAIN, max_len=SLOPE_MAX_CHAIN):
    """同一位上最近若干期是否形成等差斜连，返回最长有效链。"""
    if len(digits_at_pos) < min_len:
        return None
    best = None
    upper = min(max_len, len(digits_at_pos))
    for length in range(upper, min_len - 1, -1):
        seq = digits_at_pos[-length:]
        step = None
        valid = True
        for i in range(1, len(seq)):
            s = _slope_step(seq[i - 1], seq[i])
            if s is None:
                valid = False
                break
            if step is None:
                step = s
            elif s != step:
                valid = False
                break
        if not valid or step is None:
            continue
        nxt = seq[-1] + step
        if 0 <= nxt <= 9:
            best = {
                "chain": seq,
                "step": step,
                "predict_digit": nxt,
                "length": length,
            }
            break
    return best


def _cross_period_slope_signals(numbers):
    """跨期斜连：近三期在百→十→个（及轮换）上形成对角走势。"""
    if len(numbers) < 3:
        return []
    signals = []
    draws = numbers[-3:]
    # 三种位次轮换的对角：起始位 offset ∈ {0,1,2}
    for offset in range(3):
        vals = []
        for k in range(3):
            pos = (offset + k) % 3
            vals.append(draws[k][pos])
        step = _slope_step(vals[0], vals[1])
        if step is None or _slope_step(vals[1], vals[2]) != step:
            continue
        predict_pos = offset
        predict_digit = vals[-1] + step
        if not (0 <= predict_digit <= 9):
            continue
        route = "→".join(
            POS_NAMES_3D[(offset + k) % 3] for k in range(3)
        )
        signals.append({
            "type": "cross_period_slope",
            "position": predict_pos,
            "position_name": POS_NAMES_3D[predict_pos],
            "chain": vals,
            "route": route,
            "step": step,
            "predict_digit": predict_digit,
            "length": 3,
            "label": (
                f"跨期斜连 {route} {'+' if step > 0 else ''}{step} "
                f"({'→'.join(map(str, vals))}) → 下期{POS_NAMES_3D[predict_pos]}位关注 {predict_digit}"
            ),
            "strength": 1.0,
        })
    return signals


def analyze_slope_patterns(numbers, min_len=SLOPE_MIN_CHAIN):
    """识别斜连走势并给出下期分位关注码（辅助参考，非主预测）。"""
    signals = []
    position_hints = {i: [] for i in range(3)}

    for pos in range(3):
        hist = [n[pos] for n in numbers]
        det = _detect_position_slope_chain(hist, min_len=min_len)
        if not det:
            continue
        chain_s = "→".join(map(str, det["chain"]))
        step = det["step"]
        pred = det["predict_digit"]
        strength = 1.0 + (det["length"] - min_len) * 0.25
        sig = {
            "type": "position_slope",
            "position": pos,
            "position_name": POS_NAMES_3D[pos],
            "chain": det["chain"],
            "step": step,
            "predict_digit": pred,
            "length": det["length"],
            "label": (
                f"同位斜连 {POS_NAMES_3D[pos]}位 {'+' if step > 0 else ''}{step} "
                f"({chain_s}) → 关注 {pred}"
            ),
            "strength": round(strength, 2),
        }
        signals.append(sig)
        position_hints[pos].append({
            "digit": pred,
            "strength": strength,
            "type": "position_slope",
        })

    for sig in _cross_period_slope_signals(numbers):
        signals.append(sig)
        pos = sig["position"]
        position_hints[pos].append({
            "digit": sig["predict_digit"],
            "strength": sig["strength"],
            "type": "cross_period_slope",
        })

    # 上期三位本身呈斜连（百→十→个等差），下期同向延伸作弱提示
    if len(numbers) >= 1:
        last = numbers[-1]
        s01 = _slope_step(last[0], last[1])
        s12 = _slope_step(last[1], last[2])
        if s01 is not None and s01 == s12:
            for pos in range(3):
                nxt = last[pos] + s01
                if 0 <= nxt <= 9:
                    position_hints[pos].append({
                        "digit": nxt,
                        "strength": 0.6,
                        "type": "in_draw_slope",
                    })
            signals.append({
                "type": "in_draw_slope",
                "chain": list(last),
                "step": s01,
                "label": (
                    f"上期位内斜连 {'+' if s01 > 0 else ''}{s01} "
                    f"({last[0]}→{last[1]}→{last[2]})，下期各位可顺势延伸"
                ),
                "position_hints": [
                    {"position_name": POS_NAMES_3D[i], "digit": last[i] + s01}
                    for i in range(3)
                    if 0 <= last[i] + s01 <= 9
                ],
            })

    return {
        "active": len(signals) > 0,
        "signal_count": len(signals),
        "signals": signals,
        "position_hints": {
            POS_NAMES_3D[i]: position_hints[i] for i in range(3)
        },
        "note": "斜连为走势辅助信号；历史回测命中率接近随机，请与和值/共现等一并参考。",
    }


def slope_triplet_bonus(a, b, c, meta):
    """直选组合与斜连关注码吻合时的加分。"""
    slope = meta.get("slope") or {}
    hints = slope.get("position_hints") or {}
    bonus = 0.0
    digits = (a, b, c)
    for pos, name in enumerate(POS_NAMES_3D):
        for hint in hints.get(name, []):
            if hint.get("digit") == digits[pos]:
                bonus += W_SLOPE_MATCH * float(hint.get("strength", 1.0))
    return bonus


def backtest_slope_patterns(numbers, trials=100):
    """斜连信号独立回测（分位预测是否命中）。"""
    pos_hit = pos_total = 0
    cross_hit = cross_total = 0
    start = max(SLOPE_MIN_CHAIN + 1, len(numbers) - trials)

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        slope = analyze_slope_patterns(train)
        for sig in slope.get("signals", []):
            if sig["type"] == "position_slope":
                pos_total += 1
                if actual[sig["position"]] == sig["predict_digit"]:
                    pos_hit += 1
            elif sig["type"] == "cross_period_slope":
                cross_total += 1
                if actual[sig["position"]] == sig["predict_digit"]:
                    cross_hit += 1

    return {
        "trials": trials,
        "position_slope_hit": pos_hit,
        "position_slope_total": pos_total,
        "position_slope_rate": round(pos_hit / pos_total, 4) if pos_total else 0.0,
        "cross_slope_hit": cross_hit,
        "cross_slope_total": cross_total,
        "cross_slope_rate": round(cross_hit / cross_total, 4) if cross_total else 0.0,
        "baseline_single_pos": 0.10,
    }


def entropy_model(numbers, min_appear_window=30):
    """熵值模型：统计数字熵、和值熵、跨度熵，计算长期未出现号码的奖励
    
    参数：
        numbers: 历史开奖号码列表
        min_appear_window: 最小统计窗口（期数）
    
    返回：
        熵值奖励字典 {digit: entropy_bonus}
    
    注意：实盘版本已关闭熵值奖励，所谓"长期未出现"并不会提高下一期出现概率，容易形成追冷号。
    """
    # 实盘版本：关闭熵值奖励，避免追冷
    return {d: 0.0 for d in range(10)}


def rebound_model(numbers, window=RECENT_WINDOW_REBOUND):
    """近期回补模型：统计最近 N 期数字出现次数，严重欠账的号码额外加分
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口（期数）
    
    返回：
        回补奖励字典 {digit: rebound_bonus}
    """
    if len(numbers) < window:
        return {d: 0.0 for d in range(10)}
    
    # 统计最近 window 期数字出现次数
    digit_counts = Counter()
    for n in numbers[-window:]:
        for d in n:
            digit_counts[d] += 1
    
    # 计算理论值：每期待 3 个数字，window 期共 3*window 个数字，10 个数字平均分配
    theoretical = (3 * window) / 10.0  # 理论出现次数
    
    # 计算回补奖励
    rebound_bonus = {}
    for d in range(10):
        actual = digit_counts.get(d, 0)
        ratio = actual / theoretical if theoretical > 0 else 0
        
        # 严重欠账：实际值/理论值 < 阈值
        if ratio < REBOUND_THRESHOLD:
            rebound_bonus[d] = REBOUND_BONUS
        else:
            rebound_bonus[d] = 0.0
    
    return rebound_bonus


def classify_digits_by_hot(numbers, window=HOT_WINDOW):
    """将数字分为热号、温号、冷号三类
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口
    
    返回：
        (hot_digits, warm_digits, cold_digits)
    """
    if len(numbers) < window:
        return list(range(10)), [], []
    
    # 统计最近 window 期数字出现次数
    digit_counts = Counter()
    for n in numbers[-window:]:
        for d in n:
            digit_counts[d] += 1
    
    # 计算理论值
    theoretical = (3 * window) / 10.0
    
    hot_digits = []
    warm_digits = []
    cold_digits = []
    
    for d in range(10):
        actual = digit_counts.get(d, 0)
        ratio = actual / theoretical if theoretical > 0 else 0
        
        if ratio >= 1.2:  # 超过理论值 20% 为热号
            hot_digits.append(d)
        elif ratio >= 0.8:  # 理论值 80%-120% 为温号
            warm_digits.append(d)
        else:  # 低于理论值 80% 为冷号
            cold_digits.append(d)
    
    return hot_digits, warm_digits, cold_digits


def sum_trend_model(numbers, window=SUM_TREND_WINDOW):
    """和值趋势模型：统计最近 N 期和值趋势，动态调整和值中心
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口
    
    返回：
        adjusted_sum_center: 调整后的和值中心
        trend_direction: 趋势方向 ('up', 'down', 'oscillate')
    """
    if len(numbers) < window:
        return 13.5, 'oscillate'  # 默认和值中心（0-27 的中间值）
    
    # 计算最近 window 期的和值
    recent_sums = [sum(n) for n in numbers[-window:]]
    
    # 计算前一半和后一半的平均和值
    half = window // 2
    first_half_avg = sum(recent_sums[:half]) / half if half > 0 else 0
    second_half_avg = sum(recent_sums[half:]) / (window - half) if (window - half) > 0 else 0
    
    # 计算整体平均和值
    overall_avg = sum(recent_sums) / window
    
    # 判断趋势
    if second_half_avg > first_half_avg + 1.5:
        trend_direction = 'up'
        adjusted_sum_center = overall_avg + SUM_TREND_ADJUST
    elif second_half_avg < first_half_avg - 1.5:
        trend_direction = 'down'
        adjusted_sum_center = overall_avg - SUM_TREND_ADJUST
    else:
        trend_direction = 'oscillate'
        adjusted_sum_center = overall_avg
    
    # 限制和值中心在合理范围内（0-27）
    adjusted_sum_center = max(0, min(27, adjusted_sum_center))
    
    return adjusted_sum_center, trend_direction


def average_miss_cycle(numbers, digit, window=MISS_CYCLE_WINDOW):
    """计算单个数字的平均遗漏周期
    
    参数：
        numbers: 历史开奖号码列表
        digit: 目标数字
        window: 统计窗口大小
    
    返回：
        avg_cycle: 平均遗漏周期（期数），如果数据不足返回默认值 7
    """
    if len(numbers) < 10:
        return 7.0  # 默认平均遗漏周期
    
    # 使用最近 window 期数据
    recent_numbers = numbers[-window:] if len(numbers) > window else numbers
    
    miss_periods = []
    current_miss = 0
    
    for n in recent_numbers:
        if digit in n:
            miss_periods.append(current_miss)
            current_miss = 0
        else:
            current_miss += 1
    
    # 如果最后还有未结束的遗漏，不计入
    if miss_periods:
        return sum(miss_periods) / len(miss_periods)
    else:
        return 7.0  # 默认值


def miss_cycle_bonus(numbers):
    """遗漏周期模型：计算超期遗漏奖励
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        bonus: 各数字的超期奖励 {digit: bonus}
    """
    bonus = {}
    
    for d in range(10):
        current_miss = miss_value(numbers, d)
        avg_miss = average_miss_cycle(numbers, d)
        
        if avg_miss > 0:
            ratio = current_miss / avg_miss
            if ratio > MISS_OVER_RATIO_THRESHOLD:
                # 超期倍率越高，奖励越多
                bonus[d] = MISS_OVER_BONUS * (ratio - MISS_OVER_RATIO_THRESHOLD + 1)
            else:
                bonus[d] = 0.0
        else:
            bonus[d] = 0.0
    
    return bonus


def pair_frequency(numbers, window=50):
    """统计数字对出现频率
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口大小
    
    返回：
        pair_freq: 数字对频率字典 {(a, b): freq}，a <= b
    """
    recent_numbers = numbers[-window:] if len(numbers) > window else numbers
    total_draws = len(recent_numbers)
    
    if total_draws == 0:
        return {}
    
    pair_counts = Counter()
    
    for n in recent_numbers:
        # 生成所有不重复的数字对（不考虑顺序）
        digits = sorted(set(n))  # 去重并排序
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                pair_counts[(digits[i], digits[j])] += 1
    
    # 计算频率
    pair_freq = {}
    for pair, count in pair_counts.items():
        pair_freq[pair] = count / total_draws
    
    return pair_freq


def high_freq_pairs(numbers):
    """获取高频数字对
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        high_pairs: 高频数字对集合 {(a, b), ...}
    """
    high_pairs = set()
    
    for window in PAIR_FREQ_WINDOWS:
        pair_freq = pair_frequency(numbers, window)
        for pair, freq in pair_freq.items():
            if freq > PAIR_HIGH_FREQ_THRESHOLD:
                high_pairs.add(pair)
    
    return high_pairs


def pair_bonus(triple, high_pairs):
    """计算号码组合中的数字配对奖励（使用预计算的高频对子）"""
    bonus = 0.0
    digits = sorted(set(triple))
    for i in range(len(digits)):
        for j in range(i + 1, len(digits)):
            if (digits[i], digits[j]) in high_pairs:
                bonus += PAIR_BONUS
    return bonus


def form_switch_bonus(numbers):
    """组三组六切换模型：根据连续出现次数计算切换奖励
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        bonus: {"zu3": 组三奖励, "zu6": 组六奖励}
    """
    if len(numbers) < 5:
        return {"zu3": 0.0, "zu6": 0.0}
    
    # 统计最近的形式序列
    forms = [classify_form(n) for n in numbers]
    last_form = forms[-1]
    
    # 计算连续出现次数
    streak = 1
    for i in range(len(forms) - 2, -1, -1):
        if forms[i] == last_form:
            streak += 1
        else:
            break
    
    bonus = {"zu3": 0.0, "zu6": 0.0}
    
    # 如果组六连续出现过多，增加组三权重
    if last_form == "zu6" and streak >= ZU6_STREAK_THRESHOLD:
        # 连续次数越多，切换奖励越大
        bonus["zu3"] = FORM_SWITCH_WEIGHT * (streak - ZU6_STREAK_THRESHOLD + 1)
    
    # 如果组三连续出现过多，增加组六权重
    elif last_form == "zu3" and streak >= ZU3_STREAK_THRESHOLD:
        bonus["zu6"] = FORM_SWITCH_WEIGHT * (streak - ZU3_STREAK_THRESHOLD + 1)
    
    return bonus


def sum_interval_bonus(numbers):
    """和值区间回归模型：计算和值区间奖励
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        interval_info: {"center": 和值中心, "low": 区间下限, "high": 区间上限}
    """
    if len(numbers) < SUM_INTERVAL_WINDOW:
        return {"center": 13.5, "low": 10, "high": 17, "bonus": {}}
    
    # 计算最近 SUM_INTERVAL_WINDOW 期的和值中心
    recent_numbers = numbers[-SUM_INTERVAL_WINDOW:]
    recent_sums = [sum(n) for n in recent_numbers]
    sum_center = sum(recent_sums) / len(recent_sums)
    
    # 定义区间
    interval_low = max(0, int(sum_center - SUM_INTERVAL_WIDTH))
    interval_high = min(27, int(sum_center + SUM_INTERVAL_WIDTH))
    
    # 构建奖励字典
    bonus = {}
    for s in range(28):
        if interval_low <= s <= interval_high:
            bonus[s] = SUM_INTERVAL_BONUS
        elif s <= 5 or s >= 25:
            bonus[s] = -SUM_EXTREME_PENALTY
        else:
            bonus[s] = 0.0
    
    return {"center": sum_center, "low": interval_low, "high": interval_high, "bonus": bonus}


def recent_recommend_penalty(pool, recent_recommendations):
    """最近5期排除机制：对重复推荐进行惩罚
    
    参数：
        pool: 当前推荐池 [(权重, 号码字符串), ...]
        recent_recommendations: 最近推荐历史列表（新格式：[{"period": ..., "recommendations": [...]}, ...]）
    
    返回：
        penalized_pool: 应用惩罚后的推荐池
    """
    if not recent_recommendations:
        return pool
    
    # 扁平化最近推荐历史
    recent_set = set()
    consecutive_count = {}
    
    for entry in recent_recommendations[-RECENT_RECOMMEND_WINDOW:]:
        # 兼容新格式（字典）和旧格式（列表）
        if isinstance(entry, dict):
            rec_list = entry.get("recommendations", [])
        else:
            rec_list = entry
        for num_str in rec_list:
            recent_set.add(num_str)
            consecutive_count[num_str] = consecutive_count.get(num_str, 0) + 1
    
    # 应用惩罚
    penalized_pool = []
    for w, num_str in pool:
        penalty = 0.0
        
        # 如果最近推荐过
        if num_str in recent_set:
            penalty -= RECENT_RECOMMEND_PENALTY
        
        # 如果连续推荐过（出现多次）
        if consecutive_count.get(num_str, 0) >= 2:
            penalty -= RECENT_RECOMMEND_CONSECUTIVE_PENALTY
        
        penalized_pool.append((w + penalty, num_str))
    
    return penalized_pool


def max_digit_overlap(actual_s, candidates):
    """候选号码中与开奖号的最大数字重合数（ multiset 计数）"""
    actual_counter = Counter(actual_s)
    if not candidates:
        return 0
    return max(
        sum((actual_counter & Counter(num)).values())
        for num in candidates
    )


def classify_form(triple):
    """形态：组六 / 组三 / 豹子"""
    n = len(set(triple))
    if n == 3:
        return "zu6"
    if n == 2:
        return "zu3"
    return "baozi"


FORM_LABELS = {"zu6": "组六", "zu3": "组三", "baozi": "豹子"}


THEORY_FORM_P = {"zu6": 0.72, "zu3": 0.27, "baozi": 0.01}


def form_miss(forms, target):
    """距上次出现 target 形态的期数"""
    for i in range(len(forms) - 1, -1, -1):
        if forms[i] == target:
            return len(forms) - 1 - i
    return len(forms)


def _form_recent_p(forms, window):
    recent = _recent_slice(forms, window)
    w_cnt = exp_weighted_counts(recent)
    w_total = sum(w_cnt.values()) or 1.0
    return {k: w_cnt.get(k, 0) / w_total for k in THEORY_FORM_P}


