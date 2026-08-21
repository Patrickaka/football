# -*- coding: utf-8 -*-
"""足球市场分析：亚盘/欧赔/大小球/凯利/离散度/联合异常"""

import sys
import os
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('football')
from . import fetching as _fetching_mod

from .config import (
    BASE, EURO_PROB_TREND_EPS, HANDICAP_TREND_EPS, KELLY_BIAS_EPS, OUZHI_JSON_URL, TOTAL_LEAN_THRESHOLD, WATER_TREND_EPS,
)


def remove_vig(o1, o2, o3=None):
    """去水率，返回真实概率"""
    if o3 is None:
        p1, p2 = 1 / o1, 1 / o2
        total = p1 + p2
        return p1 / total, p2 / total
    else:
        p1, p2, p3 = 1 / o1, 1 / o2, 1 / o3
        total = p1 + p2 + p3
        return p1 / total, p2 / total, p3 / total


def _analyze_handicap_trend(open_hcap, close_hcap):
    """分析让球走势"""
    dh = close_hcap - open_hcap
    if dh > HANDICAP_TREND_EPS:
        return f"让球升高 {open_hcap:+.2f} → {close_hcap:+.2f}（主队被看好）"
    elif dh < -HANDICAP_TREND_EPS:
        return f"让球降低 {open_hcap:+.2f} → {close_hcap:+.2f}（客队被看好）"
    else:
        return f"让球不变 {close_hcap:+.2f}（盘口稳定）"


def calculate_implied_total(line, over_odds, under_odds):
    """根据大小球盘口和水位计算隐含总进球数"""
    # 简化版：直接使用盘口线作为隐含总进球数的基础
    # 水位可以微调：低水方更被看好
    if over_odds < under_odds:
        # 大球低水，略微上调
        return line + 0.1
    elif under_odds < over_odds:
        # 小球低水，略微下调
        return line - 0.1
    return line


def analyze_asian(data):
    """解析亚盘，返回让球走势、水位走势、真实概率与强弱判断"""
    if not isinstance(data, dict):
        raise ValueError(f"亚盘数据格式错误，期望字典但得到: {type(data)}")
    
    if 'open' not in data:
        raise ValueError(f"亚盘数据缺少 'open' 键，可用键: {list(data.keys())}")
    
    if 'close' not in data:
        raise ValueError(f"亚盘数据缺少 'close' 键，可用键: {list(data.keys())}")
    
    op, cl = data['open'], data['close']
    hcap = cl['handicap']
    open_hcap = op['handicap']

    dh = hcap - open_hcap
    if dh > HANDICAP_TREND_EPS:
        handicap_trend = f"让球升高 {open_hcap:+.2f} → {hcap:+.2f}（主队被看好）"
        trend_direction = 'up'
        trend_strength = min(dh / 0.5, 1.0)  # 归一化强度
    elif dh < -HANDICAP_TREND_EPS:
        handicap_trend = f"让球降低 {open_hcap:+.2f} → {hcap:+.2f}（客队被看好）"
        trend_direction = 'down'
        trend_strength = min(-dh / 0.5, 1.0)
    else:
        handicap_trend = f"让球不变 {hcap:+.2f}（盘口稳定）"
        trend_direction = 'stable'
        trend_strength = 0.0

    # 水位变化分析
    dw = cl['home_odds'] - op['home_odds']
    if dw > WATER_TREND_EPS:
        water_trend = "主队水位上升 → 资金偏向客队"
        water_direction = 'up'
    elif dw < -WATER_TREND_EPS:
        water_trend = "主队水位下降 → 资金偏向主队"
        water_direction = 'down'
    else:
        water_trend = "水位基本稳定"
        water_direction = 'stable'

    hp_o, ap_o = remove_vig(op['home_odds'], op['away_odds'])
    hp_c, ap_c = remove_vig(cl['home_odds'], cl['away_odds'])

    # 概率变化分析
    prob_change_home = hp_c - hp_o
    prob_change_away = ap_c - ap_o
    
    # 让球变化评分（用于后续λ调整）
    # 让球升高 → 主队λ +, 客队λ -
    # 让球降低 → 主队λ -, 客队λ +
    lambda_adjust_home = dh * 0.15  # 让球每变化0.25球，λ调整0.0375
    lambda_adjust_away = -dh * 0.05  # 客队调整幅度较小

    if abs(hcap) <= 0.25:
        diff_range, diff_desc = [0, 0.5], "势均力敌"
    elif abs(hcap) <= 0.75:
        diff_range, diff_desc = [0.5, 1.5], "预期1球差"
    elif abs(hcap) <= 1.25:
        diff_range, diff_desc = [1, 2], "预期1-2球差"
    elif abs(hcap) <= 1.75:
        diff_range, diff_desc = [1.5, 2.5], "预期2球差"
    else:
        diff_range = [abs(hcap) - 0.25, abs(hcap) + 0.25]
        diff_desc = f"预期{abs(hcap):.1f}球差以上"

    if hcap > 0:
        # 主队让球：主队是让球方，客队是受让方
        favor, favor_desc = 'home', f"主队让 {hcap} 球（主强客弱）"
        open_prob_label = {'home_give': hp_o, 'away_recv': ap_o}
        close_prob_label = {'home_give': hp_c, 'away_recv': ap_c}
    elif hcap < 0:
        # 客队让球：客队是让球方，主队是受让方
        favor, favor_desc = 'away', f"客队让 {abs(hcap)} 球（客强主弱）"
        open_prob_label = {'home_recv': hp_o, 'away_give': ap_o}
        close_prob_label = {'home_recv': hp_c, 'away_give': ap_c}
    else:
        # 平手盘
        favor, favor_desc = 'even', "平手盘（势均力敌）"
        open_prob_label = {'home': hp_o, 'away': ap_o}
        close_prob_label = {'home': hp_c, 'away': ap_c}

    # 综合信号强度
    signal_strength = 'weak'
    if abs(dh) >= 0.5:
        signal_strength = 'strong'
    elif abs(dh) >= 0.25:
        signal_strength = 'medium'

    return {
        'handicap': hcap,
        'open_handicap': open_hcap,
        'handicap_change': dh,
        'favor': favor, 'favor_desc': favor_desc, 'diff_desc': diff_desc,
        'diff_range': diff_range,
        'handicap_trend': handicap_trend, 'water_trend': water_trend,
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'signal_strength': signal_strength,
        'open_prob': open_prob_label,
        'close_prob': close_prob_label,
        'prob_change': {'home': prob_change_home, 'away': prob_change_away},
        'open_water': {'home': op['home_odds'], 'away': op['away_odds']},
        'close_water': {'home': cl['home_odds'], 'away': cl['away_odds']},
        'lambda_adjust': {'home': lambda_adjust_home, 'away': lambda_adjust_away},
    }


def _return_rate_from_odds(home, draw, away):
    """由欧赔估算理论返还率（%），JSON 无返还率字段时兜底"""
    total = 1.0 / home + 1.0 / draw + 1.0 / away
    return 100.0 / total if total > 0 else 92.0


def kelly_index_triple(home_odds, draw_odds, away_odds, p_home, p_draw, p_away):
    """三项凯利指数（%）= 赔率 × 去水概率 × 100，与 500.com 口径一致"""
    return {
        'home': home_odds * p_home * 100,
        'draw': draw_odds * p_draw * 100,
        'away': away_odds * p_away * 100,
    }


def _kelly_outcome_label(key):
    return {'home': '主胜', 'draw': '平局', 'away': '客胜'}[key]


def _linear_regression_slope(x_vals, y_vals):
    """计算线性回归斜率"""
    n = len(x_vals)
    if n < 2:
        return 0.0
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    denominator = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def analyze_kelly(ouzhi_data, probs_open, probs_close):
    """
    欧赔凯利指数分析：初/终盘凯利、返还率对比、离散度与打出难度提示。
    probs 通常取同一组欧赔去水概率（与计算凯利的赔率对应）。
    """
    op, cl = ouzhi_data['open'], ouzhi_data['close']
    ph_o, pd_o, pa_o = probs_open
    ph_c, pd_c, pa_c = probs_close

    rr_o = op.get('return_rate') or _return_rate_from_odds(op['home'], op['draw'], op['away'])
    rr_c = cl.get('return_rate') or _return_rate_from_odds(cl['home'], cl['draw'], cl['away'])

    k_open = kelly_index_triple(op['home'], op['draw'], op['away'], ph_o, pd_o, pa_o)
    k_close = kelly_index_triple(cl['home'], cl['draw'], cl['away'], ph_c, pd_c, pa_c)
    delta = {k: k_close[k] - k_open[k] for k in k_close}

    labels = ('home', 'draw', 'away')
    spread = max(k_close.values()) - min(k_close.values())
    
    KELLY_NEUTRAL_SPREAD = 1.0
    
    # 根据离散度判断是否有明显最难项
    if spread < KELLY_NEUTRAL_SPREAD:
        hardest = 'neutral'
        favored = 'neutral'
    else:
        hardest = max(labels, key=lambda k: k_close[k] - rr_c)
        favored = min(labels, key=lambda k: k_close[k] - rr_c)

    risks, favors, kelly_changes = [], [], []
    for k in labels:
        name = _kelly_outcome_label(k)
        diff = k_close[k] - rr_c
        if diff > KELLY_BIAS_EPS:
            risks.append(f"{name}凯利{k_close[k]:.1f}高于返还率{rr_c:.1f}（+{diff:.1f}）→ 打出偏难")
        elif diff < -KELLY_BIAS_EPS:
            favors.append(f"{name}凯利{k_close[k]:.1f}低于返还率（{diff:.1f}）→ 相对看好")
        if abs(delta[k]) >= 1.0:
            arrow = '↑' if delta[k] > 0 else '↓'
            kelly_changes.append(f"{name}凯利{arrow}{abs(delta[k]):.1f}")

    # 构建摘要
    summary_parts = []
    if spread >= 4.0:
        bias_desc = f"凯利离散度{spread:.1f}，庄家态度分化明显"
        summary_parts.append(bias_desc)
    elif spread < KELLY_NEUTRAL_SPREAD:
        bias_desc = f"凯利离散度{spread:.1f}，三项较为均衡"
        summary_parts.append(bias_desc)
        summary_parts.append("暂无明显最难项")
    else:
        bias_desc = f"凯利离散度{spread:.1f}，三项较为均衡"
        summary_parts.append(bias_desc)
        summary_parts.append(f"最难项倾向{_kelly_outcome_label(hardest)}")
    
    if favors:
        summary_parts.append(favors[0])
    summary = '；'.join(summary_parts)

    rr_delta = rr_c - rr_o
    
    return {
        'return_rate': {'open': rr_o, 'close': rr_c, 'delta': rr_delta},
        'open': k_open,
        'close': k_close,
        'delta': delta,
        'spread': spread,
        'hardest': hardest,
        'favored': favored,
        'risks': risks,
        'favors': favors,
        'kelly_changes': kelly_changes,
        'summary': summary,
    }


def analyze_kelly_trend(series, recent_n=5):
    """
    凯利指数时序分析：
    1. 最近 N 条凯利值的斜率
    2. 超过返还率最大项的变化趋势（诱盘检测）
    """
    if not series or len(series) < 2:
        return {
            'slopes': {},
            'crossing_events': [],
            'summary': '数据不足',
        }
    
    chrono = list(reversed(series))
    window = min(recent_n, len(chrono))
    recent = chrono[:window]
    
    # 计算每条记录的凯利值
    kelly_history = []
    rr_history = []
    for rec in recent:
        if len(rec) >= 3:
            p_home, p_draw, p_away = remove_vig(rec[0], rec[1], rec[2])
            rr = rec[3] if len(rec) > 3 else _return_rate_from_odds(rec[0], rec[1], rec[2])
            k = kelly_index_triple(rec[0], rec[1], rec[2], p_home, p_draw, p_away)
            kelly_history.append(k)
            rr_history.append(rr)
    
    if len(kelly_history) < 2:
        return {
            'slopes': {},
            'crossing_events': [],
            'summary': '数据不足',
        }
    
    # 计算斜率
    x_vals = list(range(len(kelly_history)))
    slopes = {}
    for label in ['home', 'draw', 'away']:
        y_vals = [kh[label] for kh in kelly_history]
        slopes[label] = round(_linear_regression_slope(x_vals, y_vals), 4)
    
    # 检测超过返还率的穿越事件
    crossing_events = []
    labels = ['home', 'draw', 'away']
    for i in range(1, len(kelly_history)):
        prev_k = kelly_history[i-1]
        curr_k = kelly_history[i]
        prev_rr = rr_history[i-1]
        curr_rr = rr_history[i]
        
        for label in labels:
            prev_above = prev_k[label] > prev_rr + KELLY_BIAS_EPS
            curr_above = curr_k[label] > curr_rr + KELLY_BIAS_EPS
            
            if prev_above and not curr_above:
                crossing_events.append({
                    'type': 'cross_down',
                    'label': label,
                    'desc': f"{_kelly_outcome_label(label)}凯利从高于返还率降至正常区间",
                })
            elif not prev_above and curr_above:
                crossing_events.append({
                    'type': 'cross_up', 
                    'label': label,
                    'desc': f"{_kelly_outcome_label(label)}凯利从正常区间升至高于返还率（可能诱盘）",
                })
    
    # 构建摘要
    summary_parts = []
    for label in labels:
        slope = slopes[label]
        if abs(slope) > 0.2:
            direction = '↑' if slope > 0 else '↓'
            summary_parts.append(f"{_kelly_outcome_label(label)}凯利{direction}{abs(slope):.2f}/步")
    
    for event in crossing_events:
        summary_parts.append(event['desc'])
    
    return {
        'slopes': slopes,
        'crossing_events': crossing_events,
        'summary': '；'.join(summary_parts) if summary_parts else '凯利走势平稳',
    }


def analyze_euro_momentum(series):
    """由欧赔时间序列提取主/客胜概率走势，用于修正净胜球"""
    if not series or len(series) < 2:
        return {'shift_supremacy': 0.0, 'summary': '欧赔走势数据不足'}

    chrono = list(reversed(series))
    first = remove_vig(chrono[0][0], chrono[0][1], chrono[0][2])
    last = remove_vig(chrono[-1][0], chrono[-1][1], chrono[-1][2])
    d_home = last[0] - first[0]
    d_away = last[2] - first[2]
    shift = max(-0.45, min(0.45, (d_home - d_away) * 1.8))

    parts = []
    if d_home > EURO_PROB_TREND_EPS:
        parts.append(f"主胜概率累积↑{d_home * 100:.1f}%")
    elif d_home < -EURO_PROB_TREND_EPS:
        parts.append(f"主胜概率累积↓{-d_home * 100:.1f}%")
    if d_away > EURO_PROB_TREND_EPS:
        parts.append(f"客胜概率累积↑{d_away * 100:.1f}%")
    elif d_away < -EURO_PROB_TREND_EPS:
        parts.append(f"客胜概率累积↓{-d_away * 100:.1f}%")

    return {
        'shift_supremacy': shift,
        'delta_home': d_home,
        'delta_away': d_away,
        'summary': '，'.join(parts) if parts else '欧赔走势平稳',
    }


def fetch_ouzhi_company(match_id, cid=1):
    """抓取指定公司的欧赔时间序列（cid=1 为威廉希尔等）"""
    url = f'{OUZHI_JSON_URL}?fid={match_id}&cid={cid}&type=europe&r=1'
    referer = f'{BASE}/fenxi/ouzhi-{match_id}.shtml'
    try:
        series = _fetching_mod.fetch_json(url, referer=referer)
        if isinstance(series, list) and len(series) >= 2:
            return series
    except Exception:
        pass
    return None


def compute_dispersion(series):
    """计算离散度：同一公司初盘与终盘的赔率差异的方差（多家公司）"""
    if not series or len(series) < 2:
        return 0.0
    
    close, open_ = series[0], series[-1]
    diffs = []
    
    for i in range(3):  # 主胜、平局、客胜
        if len(open_) > i and len(close) > i:
            diffs.append(abs(close[i] - open_[i]))
    
    if len(diffs) == 0:
        return 0.0
    
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return variance


def compute_joint_anomaly(asian_data, total_data):
    """
    计算联合异常特征：
    1. 让球盘水位变化 × 大小球水位变化
    2. 亚盘与欧赔转换偏差（由欧赔转换出的理论让球值与实际亚盘让球值的差值）
    """
    # 让球盘水位变化
    asian_op, asian_cl = asian_data['open'], asian_data['close']
    asian_water_change = asian_cl['home_odds'] - asian_op['home_odds']  # 主队水位变化
    
    # 大小球水位变化
    total_op, total_cl = total_data['open'], total_data['close']
    total_water_change = total_cl['over_odds'] - total_op['over_odds']  # 大球水位变化
    
    # 联合特征：水位变化乘积
    joint_water_feature = asian_water_change * total_water_change
    
    # 判断是否暗示主队大胜
    hint_big_win = False
    if asian_water_change < -WATER_TREND_EPS and total_water_change < -WATER_TREND_EPS:
        hint_big_win = True  # 主队水位下降 + 大球水位下降
    
    return {
        'asian_water_change': round(asian_water_change, 4),
        'total_water_change': round(total_water_change, 4),
        'joint_water_feature': round(joint_water_feature, 6),
        'hint_big_win': hint_big_win,
        'hint_desc': '主队水位下降+大球水位下降，暗示主队可能大胜' if hint_big_win else None,
    }


def euro_to_handicap_implied(p_home, p_away, k=1.8):
    """
    由欧赔转换出理论让球值：(p_home - p_away) * 常数
    k 为转换系数，通常在 1.5-2.0 之间
    """
    return (p_home - p_away) * k


def compute_euro_asian_deviation(euro_probs, asian_handicap, k=1.8):
    """
    计算亚盘与欧赔转换偏差：
    理论让球值（由欧赔转换）与实际亚盘让球值的差值
    """
    p_home = euro_probs.get('home', 0.5)
    p_away = euro_probs.get('away', 0.5)
    implied_handicap = euro_to_handicap_implied(p_home, p_away, k)
    deviation = implied_handicap - asian_handicap
    return {
        'implied_handicap': round(implied_handicap, 4),
        'actual_handicap': asian_handicap,
        'deviation': round(deviation, 4),
        'abs_deviation': round(abs(deviation), 4),
    }


def analyze_euro(data):
    """解析欧赔，返回初终盘 1X2 真实概率、凯利、走势与变化趋势"""
    try:
        op = data.get('open')
        cl = data.get('close')
        
        if not op or not cl:
            raise ValueError("欧赔数据缺少 open 或 close 字段")
        
        # 验证必需字段
        required_fields = ['home', 'draw', 'away']
        for field in required_fields:
            if field not in op:
                raise ValueError(f"初盘数据缺少 {field} 字段")
            if field not in cl:
                raise ValueError(f"终盘数据缺少 {field} 字段")
        
        # 验证赔率值是否为有效正数
        for field in required_fields:
            if not isinstance(op[field], (int, float)) or op[field] <= 0:
                raise ValueError(f"初盘{field}赔率无效: {op[field]}")
            if not isinstance(cl[field], (int, float)) or cl[field] <= 0:
                raise ValueError(f"终盘{field}赔率无效: {cl[field]}")

        ph_o, pd_o, pa_o = remove_vig(op['home'], op['draw'], op['away'])
        ph_c, pd_c, pa_c = remove_vig(cl['home'], cl['draw'], cl['away'])

        # 验证概率值
        for p, name in [(ph_o, '主胜初盘概率'), (pd_o, '平局初盘概率'), (pa_o, '客胜初盘概率'),
                        (ph_c, '主胜终盘概率'), (pd_c, '平局终盘概率'), (pa_c, '客胜终盘概率')]:
            if not (0 <= p <= 1):
                raise ValueError(f"{name}超出范围: {p}")

        changes = []
        if ph_c - ph_o > EURO_PROB_TREND_EPS: 
            changes.append(f"主胜概率↑{(ph_c-ph_o)*100:.1f}%")
        elif ph_c - ph_o < -EURO_PROB_TREND_EPS: 
            changes.append(f"主胜概率↓{(ph_o-ph_c)*100:.1f}%")
        if pa_c - pa_o > EURO_PROB_TREND_EPS: 
            changes.append(f"客胜概率↑{(pa_c-pa_o)*100:.1f}%")
        elif pa_c - pa_o < -EURO_PROB_TREND_EPS: 
            changes.append(f"客胜概率↓{(pa_o-pa_c)*100:.1f}%")
        if pd_c - pd_o > EURO_PROB_TREND_EPS: 
            changes.append(f"平局概率↑{(pd_c-pd_o)*100:.1f}%")
        elif pd_c - pd_o < -EURO_PROB_TREND_EPS: 
            changes.append(f"平局概率↓{(pd_o-pd_c)*100:.1f}%")

        kelly = analyze_kelly(data, (ph_o, pd_o, pa_o), (ph_c, pd_c, pa_c))
        momentum = analyze_euro_momentum(data.get('series', []))

        return {
            'open': {'home': ph_o, 'draw': pd_o, 'away': pa_o},
            'close': {'home': ph_c, 'draw': pd_c, 'away': pa_c},
            'raw_odds': {'open': dict(op), 'close': dict(cl)},
            'kelly': kelly,
            'momentum': momentum,
            'changes': changes,
        }
    
    except Exception as e:
        raise ValueError(f"欧赔分析失败: {e}")


def analyze_total(data):
    """解析大小球，返回盘口线、大小球真实概率、倾向与期望进球区间"""
    op, cl = data['open'], data['close']
    line = cl['line']
    open_line = op['line']

    po_o, pu_o = remove_vig(op['over_odds'], op['under_odds'])
    po_c, pu_c = remove_vig(cl['over_odds'], cl['under_odds'])

    # 大小球盘口变化分析
    dl = line - open_line
    if dl > 0.125:
        line_trend = f"盘口升高 {open_line:.2f} → {line:.2f}（大球被看好）"
        trend_direction = 'up'
        trend_strength = min(dl / 0.5, 1.0)
    elif dl < -0.125:
        line_trend = f"盘口降低 {open_line:.2f} → {line:.2f}（小球被看好）"
        trend_direction = 'down'
        trend_strength = min(-dl / 0.5, 1.0)
    else:
        line_trend = f"盘口稳定 {line:.2f}"
        trend_direction = 'stable'
        trend_strength = 0.0

    # 概率变化分析
    prob_change_over = po_c - po_o
    prob_change_under = pu_c - pu_o

    # 大小球变化对λ的调整
    # 盘口升高 → 总进球λ +
    # 盘口降低 → 总进球λ -
    lambda_adjust_total = dl * 0.6  # 盘口每变化0.25球，λ调整0.15

    if po_c >= TOTAL_LEAN_THRESHOLD:
        lean, lean_desc = 'over', f"大球倾向（大球概率{po_c*100:.1f}%）"
    elif pu_c >= TOTAL_LEAN_THRESHOLD:
        lean, lean_desc = 'under', f"小球倾向（小球概率{pu_c*100:.1f}%）"
    else:
        lean, lean_desc = None, f"大小球均衡（线{line}，各约50%）"

    over_lean = lean == 'over'
    if line <= 1.0:
        expected_goals = [1, 3]
    elif line <= 2.0:
        expected_goals = [1, 4]
    elif line <= 2.5:
        expected_goals = [2, 4] if over_lean else [1, 3]
    elif line <= 3.0:
        expected_goals = [2, 5] if over_lean else [1, 3]
    elif line <= 3.5:
        expected_goals = [3, 6] if over_lean else [2, 4]
    else:
        lo = max(0, int(line))
        expected_goals = [lo, lo + 2]

    implied_total = implied_total_goals(line, po_c)
    open_implied = implied_total_goals(op['line'], po_o)

    # 综合信号强度
    signal_strength = 'weak'
    if abs(dl) >= 0.5:
        signal_strength = 'strong'
    elif abs(dl) >= 0.25:
        signal_strength = 'medium'

    return {
        'open_line': open_line, 'close_line': line,
        'line_change': dl,
        'line_trend': line_trend,
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'signal_strength': signal_strength,
        'implied_total': implied_total,
        'open_implied_total': open_implied,
        'implied_change': implied_total - open_implied,
        'lean': lean, 'lean_desc': lean_desc,
        'open_prob': {'over': po_o, 'under': pu_o},
        'close_prob': {'over': po_c, 'under': pu_c},
        'prob_change': {'over': prob_change_over, 'under': prob_change_under},
        'open_water': {'over': op['over_odds'], 'under': op['under_odds']},
        'close_water': {'over': cl['over_odds'], 'under': cl['under_odds']},
        'expected_goals': expected_goals,
        'lambda_adjust': {'total': lambda_adjust_total},
    }


def _poisson_pmf(k, lam):
    """泊松概率质量函数 P(X=k)"""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _poisson_tail_over(lam_total, line):
    """泊松总进球模型下 P(总进球 > line)；四分盘按相邻半球盘各半权重"""
    frac = round((line * 4) % 4)
    if frac in (1, 3):
        low, high = line - 0.25, line + 0.25
        return 0.5 * _poisson_tail_over(lam_total, low) + 0.5 * _poisson_tail_over(lam_total, high)
    k_min = math.floor(line + 0.501)
    prob = 0.0
    for k in range(k_min, 30):
        prob += _poisson_pmf(k, lam_total)
    return min(1.0, prob)


def implied_total_goals(line, p_over, tol=1e-4):
    """由大小球盘口线与去水大球概率反推期望总进球 λ_total"""
    p_over = max(0.02, min(0.98, p_over))
    lo, hi = 0.3, 6.5
    for _ in range(48):
        mid = (lo + hi) / 2
        if _poisson_tail_over(mid, line) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


