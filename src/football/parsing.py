# -*- coding: utf-8 -*-
"""足球赔率页解析：盘口文本、公司赔率、联赛画像、球队实力"""

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
    BASE, CLOSE_BLEND_WEIGHT, ELO_AVAILABLE, LEAGUE_PROFILES, LOTTERY_OFFICIAL_ODDS_WEIGHT, MIN_AVG_NUMBERS, ODDS_PAGES, OUZHI_JSON_URL, elo_to_goals_expected, elo_to_strength_factor, get_elo_system,
)


def get_close_total_line(total: dict, default: float = 2.5) -> float:
    """
    统一获取大小球终盘线
    
    支持多种数据结构：
    - total.get('close_line') - 直接存储的终盘线
    - total.get('line') - 直接存储的线
    - total.get('close', {}).get('line') - 通过 close 字典获取（如 fetch_daxiao 返回的结构）
    
    参数：
        total: 大小球数据字典
        default: 默认值（当所有来源都取不到时使用）
    
    返回：
        大小球终盘线
    """
    return (
        total.get('close_line')
        or total.get('line')
        or total.get('close', {}).get('line')
        or default
    )


def parse_handicap(text):
    """将让球文本转换为数值（正=主让，负=客让）"""
    t = text.strip()
    sign = -1 if '受' in t else 1
    t = t.replace('受', '')

    mapping = {
        '平手': 0, '半球': 0.5, '一球': 1.0, '球半': 1.5,
        '两球': 2.0, '两球半': 2.5, '三球': 3.0, '三球半': 3.5,
        '平手/半球': 0.25, '半球/一球': 0.75,
        '一球/球半': 1.25, '球半/两球': 1.75,
        '两球/两球半': 2.25, '两球半/三球': 2.75, '三球/三球半': 3.25,
    }
    if t in mapping:
        return sign * mapping[t]
    try:
        return sign * float(t)
    except ValueError:
        return 0


def parse_total_line(text):
    """解析大小球盘口线"""
    t = text.strip()
    mapping = {
        '0.5/1': 0.75, '1/1.5': 1.25, '1.5/2': 1.75,
        '2/2.5': 2.25, '2.5/3': 2.75, '3/3.5': 3.25, '3.5/4': 3.75,
    }
    if t in mapping:
        return mapping[t]
    try:
        return float(t)
    except ValueError:
        return 2.5


def parse_lottery_handicap(value):
    """Parse the integer home-team handicap used by China Sports Lottery.

    This is deliberately separate from the Asian handicap.  For example, a
    lottery handicap of ``-1`` means the settlement score is home goals - 1
    versus away goals; quarter-ball Asian lines are never accepted here.
    """
    if value is None or value == '':
        return None
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(value).replace('（', '(').replace('）', ')'))
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    if not number.is_integer() or abs(number) > 5:
        return None
    return int(number)


def _lottery_odds_probabilities(odds, keys):
    """Return normalized, overround-removed probabilities for one lottery market."""
    implied = {}
    for key in keys:
        try:
            value = float((odds or {}).get(key))
        except (TypeError, ValueError):
            return None
        if value <= 1.0:
            return None
        implied[key] = 1.0 / value
    total = sum(implied.values())
    return {key: value / total for key, value in implied.items()} if total > 0 else None


def _blend_lottery_probabilities(model_probs, market_probs, market_weight=LOTTERY_OFFICIAL_ODDS_WEIGHT):
    if not market_probs:
        return dict(model_probs)
    weight = max(0.0, min(1.0, float(market_weight)))
    blended = {
        key: (1.0 - weight) * float(model_probs.get(key, 0.0))
        + weight * float(market_probs.get(key, 0.0))
        for key in model_probs
    }
    total = sum(blended.values())
    return {key: value / total for key, value in blended.items()} if total > 0 else blended


def _spf_selection_profile(probabilities):
    """Keep top-1 honest while exposing a close draw as explicit cover."""
    probs = {key: max(0.0, float((probabilities or {}).get(key, 0.0))) for key in ('胜', '平', '负')}
    ranked = sorted(probs, key=lambda key: (-probs[key], key))
    primary = ranked[0] if ranked and sum(probs.values()) > 0 else None
    if not primary:
        return {'primary': None, 'selections': [], 'mode': 'unavailable'}

    runner_up = ranked[1]
    gap = probs[primary] - probs[runner_up]
    selections = [primary]
    reason = 'top_probability'
    mode = 'single'

    if primary != '平':
        draw_gap = probs[primary] - probs['平']
        if probs['平'] >= 0.24 and draw_gap <= 0.12:
            selections.append('平')
            mode = 'draw_cover'
            reason = 'draw_probability_close_to_primary'
    elif probs[runner_up] >= 0.24 and gap <= 0.08:
        selections.append(runner_up)
        mode = 'draw_primary_cover'
        reason = 'draw_primary_but_margin_is_small'

    return {
        'primary': primary,
        'selections': selections,
        'mode': mode,
        'reason': reason,
        'primary_probability': probs[primary],
        'draw_probability': probs['平'],
        'top_gap': gap,
        'is_single': len(selections) == 1,
    }


def lottery_market_probabilities(candidates, lottery_handicap=None, spf_odds=None, rqspf_odds=None):
    """Build JCZQ probabilities from scores and independently priced official markets."""
    spf = {'胜': 0.0, '平': 0.0, '负': 0.0}
    handicap = parse_lottery_handicap(lottery_handicap)
    rqspf = {'让胜': 0.0, '让平': 0.0, '让负': 0.0} if handicap is not None else None
    joint_probs = {}

    for item in candidates or []:
        try:
            (home_goals, away_goals), probability = item
            home_goals, away_goals = int(home_goals), int(away_goals)
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        margin = home_goals - away_goals
        standard_label = '胜' if margin > 0 else '负' if margin < 0 else '平'
        spf[standard_label] += probability
        if rqspf is not None:
            adjusted_margin = margin + handicap
            label = '让胜' if adjusted_margin > 0 else '让负' if adjusted_margin < 0 else '让平'
            rqspf[label] += probability
            joint_key = (standard_label, label)
            joint_probs[joint_key] = joint_probs.get(joint_key, 0.0) + probability

    def normalize(values):
        total = sum(values.values())
        return {key: value / total for key, value in values.items()} if total > 0 else values

    spf = normalize(spf)
    if rqspf is not None:
        rqspf = normalize(rqspf)
        joint_total = sum(joint_probs.values())
        if joint_total > 0:
            joint_probs = {key: value / joint_total for key, value in joint_probs.items()}
    joint_ranked = sorted(joint_probs.items(), key=lambda item: -item[1])
    joint_recommendation = None
    if joint_ranked:
        (standard_pick, handicap_pick), joint_probability = joint_ranked[0]
        joint_recommendation = {
            'standard_prediction': standard_pick,
            'handicap_prediction': handicap_pick,
            'probability': joint_probability,
            'label': f'{standard_pick} + {handicap_pick}',
            'distribution': [
                {
                    'standard': standard_result,
                    'handicap': handicap_result,
                    'probability': probability,
                }
                for (standard_result, handicap_result), probability in joint_ranked
            ],
        }
    model_spf = dict(spf)
    model_rqspf = dict(rqspf) if rqspf is not None else None
    market_spf = _lottery_odds_probabilities(spf_odds, ('胜', '平', '负'))
    market_rqspf = _lottery_odds_probabilities(rqspf_odds, ('让胜', '让平', '让负')) if rqspf is not None else None
    spf = _blend_lottery_probabilities(model_spf, market_spf)
    spf_selection = _spf_selection_profile(spf)
    if rqspf is not None:
        rqspf = _blend_lottery_probabilities(model_rqspf, market_rqspf)
    linked_recommendation = None
    if rqspf is not None and joint_probs:
        standard_pick = max(spf, key=spf.get)
        compatible = {
            rq_result: probability
            for (standard_result, rq_result), probability in joint_probs.items()
            if standard_result == standard_pick and probability > 0
        }
        adjusted = {}
        for rq_result, probability in compatible.items():
            model_value = float((model_rqspf or {}).get(rq_result, 0.0))
            fused_value = float(rqspf.get(rq_result, 0.0))
            market_factor = fused_value / model_value if model_value > 0 else 1.0
            adjusted[rq_result] = probability * market_factor
        adjusted_total = sum(adjusted.values())
        conditional = ({key: value / adjusted_total for key, value in adjusted.items()}
                       if adjusted_total > 0 else {})
        if conditional:
            handicap_pick = max(conditional, key=conditional.get)
            linked_recommendation = {
                'standard_prediction': standard_pick,
                'handicap_prediction': handicap_pick,
                'compatible_handicap_predictions': list(conditional),
                'handicap_conditional_probabilities': conditional,
                'conditional_probability': conditional[handicap_pick],
                'label': f'{standard_pick} ⇒ {handicap_pick}',
                'rule': '先取胜平负最高概率，再在同一赛果兼容的让球结果中分析',
            }
    primary_type = 'rqspf' if handicap not in (None, 0) else 'spf'
    primary_probs = rqspf if primary_type == 'rqspf' else spf
    return {
        'standard': {
            'type': 'spf', 'name': '胜平负', 'probabilities': spf,
            'model_probabilities': model_spf,
            'market_probabilities': market_spf,
            'market_weight': LOTTERY_OFFICIAL_ODDS_WEIGHT if market_spf else 0.0,
            'prediction': max(spf, key=spf.get) if sum(spf.values()) > 0 else None,
            'selections': spf_selection['selections'],
            'selection_profile': spf_selection,
        },
        'handicap': ({
            'type': 'rqspf', 'name': '让球胜平负', 'handicap': handicap,
            'probabilities': rqspf,
            'model_probabilities': model_rqspf,
            'market_probabilities': market_rqspf,
            'market_weight': LOTTERY_OFFICIAL_ODDS_WEIGHT if market_rqspf else 0.0,
            'prediction': max(rqspf, key=rqspf.get) if rqspf and sum(rqspf.values()) > 0 else None,
        } if rqspf is not None else None),
        'primary_market': primary_type,
        'primary': {
            'type': primary_type,
            'probabilities': primary_probs,
            'prediction': max(primary_probs, key=primary_probs.get) if primary_probs and sum(primary_probs.values()) > 0 else None,
        },
        'joint_recommendation': joint_recommendation,
        'linked_recommendation': linked_recommendation,
        'settlement_rule': '中国体彩：主队进球 + 让球数，与客队进球比较',
    }


def _apply_lottery_market_availability(lottery):
    """对已匹配的竞彩场次，关闭官方未开售的胜平负输出。"""
    spf_prediction_enabled = (
        not lottery.get('offer_matched') or bool(lottery.get('spf_available'))
    )
    if not spf_prediction_enabled:
        # 模型内部仍可用比分分布分析让球玩法，对外不产生 SPF 推荐。
        lottery['standard'] = None
        lottery['joint_recommendation'] = None
        lottery['linked_recommendation'] = None
    return spf_prediction_enabled


def _html_to_text(html):
    """去除标签与转义空白，压缩为单行纯文本"""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _extract_avg(html, keyword='平均值'):
    """从HTML中提取包含keyword行后续的全部数字"""
    text = _html_to_text(html)
    idx = text.find(keyword)
    if idx < 0:
        raise ValueError(f"未找到'{keyword}'行")
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text[idx:])
    return [float(n) for n in numbers]


def _handicap_text_to_num(handicap_text):
    """将让球文字转换为数字
    
    主队让球 → 正数（如：半球 → 0.5）
    主队受让球 → 负数（如：受让半球 → -0.5）
    """
    if not handicap_text:
        return 0
    
    # 定义让球类型映射（不含受让前缀）
    # 按字符串长度排序，确保长字符串优先匹配
    handicap_map = [
        ('平手/半球', 0.25),
        ('半球/一球', 0.75),
        ('一球/球半', 1.25),
        ('球半/两球', 1.75),
        ('两球/两球半', 2.25),
        ('平半', 0.25),
        ('半球', 0.5),
        ('半一', 0.75),
        ('一球', 1.0),
        ('球半', 1.5),
        ('两球', 2.0),
        ('两球半', 2.5),
        ('平手', 0),
    ]
    
    # 判断是否为受让球
    is_receive = False
    text_to_check = handicap_text
    
    # 移除受让前缀
    if handicap_text.startswith('受让'):
        is_receive = True
        text_to_check = handicap_text[2:]  # 移除"受让"
    elif handicap_text.startswith('受'):
        is_receive = True
        text_to_check = handicap_text[1:]  # 移除"受"
    
    # 尝试精确匹配
    for key, value in handicap_map:
        if key == text_to_check:
            # 受让球返回负数，让球返回正数
            return -value if is_receive else value
    
    # 如果精确匹配失败，尝试包含匹配（长字符串优先）
    for key, value in handicap_map:
        if key in text_to_check:
            return -value if is_receive else value
    
    return 0


def _extract_company_odds(html, company_name, is_total=False):
    """从HTML中提取指定博彩公司的赔率数据
    
    Args:
        html: HTML页面内容
        company_name: 公司名称
        is_total: 是否为大小球页面（True表示大小球，False表示亚盘）
    
    Returns:
        列表：亚盘格式为 [初主队水位, 初让球, 初客队水位, 终主队水位, 终让球, 终客队水位]
             大小球格式为 [初大球水位, 初大小球盘, 初小球水位, 终大球水位, 终大小球盘, 终小球水位]
    """
    text = _html_to_text(html)
    
    # 找到公司名的位置（支持原始名称和被*替换的名称）
    idx = text.find(company_name)
    
    # 如果找不到原始名称，尝试查找被替换的名称
    if idx < 0:
        # 定义常见的被替换模式
        aliases = {
            'Bet365': ['**t3*5', 'B*t365', 'B**365'],
            'Pinnacle': ['Pi****le', 'Pin***le', 'Pinnacle平*'],
        }
        if company_name in aliases:
            for alias in aliases[company_name]:
                idx = text.find(alias)
                if idx >= 0:
                    log.debug(f"找到 {company_name} 的别名: {alias}")
                    break
    
    if idx < 0:
        return None
    
    # 提取该公司行的数据（截取更长的片段）
    segment = text[idx:idx + 500]
    
    # 找到第二个公司名出现的位置（因为格式是公司名重复两次）
    second_idx = segment.find('**', 10)
    if second_idx > 0:
        # 从第二个公司名之后开始提取数据，跳过第二个公司名
        data_part = segment[second_idx:]
        # 跳过公司名部分（找到第一个空格后的内容）
        first_space = data_part.find(' ')
        if first_space > 0:
            data_part = data_part[first_space+1:].strip()
    else:
        data_part = segment[len(company_name):]
    
    # 提取所有数字（水位和盘口）
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', data_part)
    
    # 过滤：只保留水位范围(0.3-2.5)和盘口范围(-5到+5或1.5-5.0)的数字
    filtered = []
    for n in numbers[:25]:
        num = float(n)
        # 水位通常在 0.3-2.5 之间
        if 0.3 <= num <= 2.5:
            filtered.append(num)
        # 让球通常在 -5 到 +5 之间（亚盘）
        elif -5 <= num <= 5 and abs(num) != 0:
            filtered.append(num)
    
    # 提取时间记录
    time_pattern = r'(\d{2}-\d{2}\s+\d{2}:\d{2})'
    times = re.findall(time_pattern, data_part)
    
    log.debug(f"{company_name} 原始数字: {numbers[:15]}, 过滤后: {filtered}, 时间: {times}")
    
    if is_total:
        # 大小球页面格式：公司名 公司名 终大球水位 大小球盘 终小球水位 时间 初大球水位 大小球盘 初小球水位 时间
        # 例如：**t3*5 **t3*5 0.950 2.5 0.900 06-09 16:17 0.900 2.5 0.900 06-08 09:36
        # 返回：[初大球水位, 初大小球盘, 初小球水位, 终大球水位, 终大小球盘, 终小球水位]
        
        # 提取大小球盘口关键词
        total_pattern = r'(\d+\.?\d*球)'
        total_matches = re.findall(total_pattern, data_part)
        
        # 解析盘口值
        final_line = 0
        initial_line = 0
        for match in total_matches[:2]:
            # 解析 "2.5球" -> 2.5
            line_str = match.replace('球', '')
            try:
                line_val = float(line_str)
                if 1.5 <= line_val <= 5.0:
                    if final_line == 0:
                        final_line = line_val
                    elif initial_line == 0:
                        initial_line = line_val
            except ValueError:
                pass
        
        # 如果文本中找不到，从数字中找
        if final_line == 0:
            for n in numbers[:20]:
                num = float(n)
                if 1.5 <= num <= 5.0:
                    final_line = num
                    break
        
        # 确保有足够的水位数据
        if len(filtered) >= 4:
            # 终盘在前，开盘在后
            final_over = filtered[0]
            final_under = filtered[1]
            initial_over = filtered[2] if len(filtered) > 2 else filtered[0]
            initial_under = filtered[3] if len(filtered) > 3 else filtered[1]
            
            # 如果还有更多数据，说明有初盘数据
            if len(filtered) >= 6:
                initial_over = filtered[3]
                initial_under = filtered[4]
            
            # 提取时间
            final_time = times[0] if len(times) > 0 else None
            initial_time = times[1] if len(times) > 1 else None
                
            result = [
                initial_over,
                initial_line if initial_line != 0 else final_line,
                initial_under,
                final_over,
                final_line,
                final_under,
                final_time,
                initial_time,
            ]
            log.debug(f"{company_name} 大小球提取结果: {result}")
            return result
        return None
    else:
        # 亚盘格式：公司名 公司名 终主队水位 让球类型 终客队水位 时间 初主队水位 让球类型 初客队水位 时间
        # 提取让球类型文本（包含受让球情况）
        handicap_pattern = r'(受让平手|受让平半|受让半球|受让半一|受让一球|受让球半|受让两球|受让两球半|平手|平半|半球|半一|一球|球半|两球|两球半|平手/半球|半球/一球|一球/球半|球半/两球|两球/两球半|受平手|受平半|受半球|受半一|受一球|受球半|受两球|受两球半)'
        handicap_matches = re.findall(handicap_pattern, data_part)
        
        # 解析让球值
        final_handicap = 0
        initial_handicap = 0
        if len(handicap_matches) >= 1:
            final_handicap = _handicap_text_to_num(handicap_matches[0])
        if len(handicap_matches) >= 2:
            initial_handicap = _handicap_text_to_num(handicap_matches[1])
        
        # 返回8个数据（初主队水位、初让球、初客队水位、终主队水位、终让球、终客队水位、终盘时间、初盘时间）
        # 需要确保数据足够且合理
        if len(filtered) >= 4:
            # 过滤后的数字应该是：终主队水位, 终客队水位, 初主队水位, 初客队水位
            # 检查前两个是否是合理的水位（0.3-2.5）
            if 0.3 <= filtered[0] <= 2.5 and 0.3 <= filtered[1] <= 2.5:
                # 提取时间
                final_time = times[0] if len(times) > 0 else None
                initial_time = times[1] if len(times) > 1 else None
                
                result = [
                    filtered[2] if len(filtered) > 2 and 0.3 <= filtered[2] <= 2.5 else filtered[0],
                    initial_handicap if initial_handicap != 0 else final_handicap,
                    filtered[3] if len(filtered) > 3 and 0.3 <= filtered[3] <= 2.5 else filtered[1],
                    filtered[0],
                    final_handicap,
                    filtered[1],
                    final_time,
                    initial_time,
                ]
                log.debug(f"{company_name} 亚盘提取结果: {result}")
                return result
    return None


def _fetch_avg_page(match_id, page):
    """抓取指定赔率页，返回 (html, 平均值行数字列表)，并校验数据量"""
    label = ODDS_PAGES[page]
    html = _fetching_mod.fetch(f'{BASE}/fenxi/{page}-{match_id}.shtml')
    nums = _extract_avg(html)
    if len(nums) < MIN_AVG_NUMBERS:
        raise ValueError(f"{label}平均值数据不足 (match_id={match_id}), 获取到: {nums}")
    return html, nums


def calculate_bookmaker_consensus(bet365_data, pinnacle_data, avg_handicap):
    """
    计算博彩公司分歧指数

    参数：
        bet365_data: Bet365 的亚盘数据 {'asian': {'close': {'handicap': ...}}}
        pinnacle_data: Pinnacle 的亚盘数据
        avg_handicap: 平均盘口数据

    返回：
        dict: 包含分歧指数和调整值
    """
    result = {
        'available': False,
        'bet365_handicap': None,
        'pinnacle_handicap': None,
        'avg_handicap': avg_handicap,
        'pinnacle_diff': 0.0,
        'sharp_bias': 'neutral',  # 'home', 'away', 'neutral'
        'adjustment': 0.0,
        'confidence': 0.0
    }

    if not bet365_data or not pinnacle_data:
        return result

    try:
        bet365_handicap = bet365_data.get('asian', {}).get('close', {}).get('handicap')
        pinnacle_handicap = pinnacle_data.get('asian', {}).get('close', {}).get('handicap')

        if bet365_handicap is None or pinnacle_handicap is None:
            return result

        result['bet365_handicap'] = bet365_handicap
        result['pinnacle_handicap'] = pinnacle_handicap
        result['available'] = True

        # Pinnacle 与平均盘的差异（Pinnacle 更接近真实概率）
        result['pinnacle_diff'] = pinnacle_handicap - avg_handicap

        # 判断 Sharp Money 方向
        # Pinnacle 让球更激进（数值更大）= 更看好主队
        # Pinnacle 让球更保守（数值更小）= 更看好客队
        if result['pinnacle_diff'] > 0.125:
            result['sharp_bias'] = 'home'
            result['confidence'] = min(result['pinnacle_diff'] / 0.5, 1.0)
            # 调整 lam_home：Pinnacle 更看好主队，增加主队期望进球
            result['adjustment'] = result['confidence'] * 0.15
        elif result['pinnacle_diff'] < -0.125:
            result['sharp_bias'] = 'away'
            result['confidence'] = min(abs(result['pinnacle_diff']) / 0.5, 1.0)
            # 调整 lam_home：Pinnacle 更看好客队，减少主队期望进球
            result['adjustment'] = -result['confidence'] * 0.15
        else:
            result['sharp_bias'] = 'neutral'
            result['confidence'] = 0.0
            result['adjustment'] = 0.0

        log.info(f"博彩公司分歧指数: Pinnacle={pinnacle_handicap}, 平均={avg_handicap}, 差异={result['pinnacle_diff']:.3f}, 方向={result['sharp_bias']}, 调整={result['adjustment']:.3f}")

    except Exception as e:
        log.warning(f"计算博彩公司分歧指数失败: {e}")

    return result


def fetch_single_company_odds(match_id):
    """
    抓取 Bet365 和 Pinnacle 的独赔数据
    
    返回：
        dict: 包含各公司的亚盘和大小球数据
    """
    log.info(f"========== 开始抓取独赔数据 match_id={match_id} ==========")
    result = {
        'bet365': None,
        'pinnacle': None,
    }
    
    try:
        # 抓取亚盘页面
        yazhi_html = _fetching_mod.fetch(f'{BASE}/fenxi/yazhi-{match_id}.shtml')
        
        # 抓取大小球页面
        daxiao_html = _fetching_mod.fetch(f'{BASE}/fenxi/daxiao-{match_id}.shtml')
        
        # 提取 Bet365 数据
        bet365_yazhi = _extract_company_odds(yazhi_html, 'Bet365', is_total=False)
        bet365_daxiao = _extract_company_odds(daxiao_html, 'Bet365', is_total=True)
        
        log.info(f"Bet365 亚盘原始数据: {bet365_yazhi}")
        
        # 只要有亚盘数据就返回（大小球可选）
        if bet365_yazhi:
            bet365_data = {
                'asian': {
                    'open': {
                        'handicap': bet365_yazhi[1],  # 初让球
                        'home_odds': bet365_yazhi[0],  # 初主队水位
                        'away_odds': bet365_yazhi[2],  # 初客队水位
                        'time': bet365_yazhi[7] if len(bet365_yazhi) > 7 else None,  # 初盘时间
                    },
                    'close': {
                        'handicap': bet365_yazhi[4],  # 终让球
                        'home_odds': bet365_yazhi[3],  # 终主队水位
                        'away_odds': bet365_yazhi[5],  # 终客队水位
                        'time': bet365_yazhi[6] if len(bet365_yazhi) > 6 else None,  # 终盘时间
                    }
                }
            }
            # 如果有大小球数据也加上
            if bet365_daxiao:
                bet365_data['total'] = {
                    'open': {
                        'line': bet365_daxiao[1],
                        'over_odds': bet365_daxiao[0],
                        'under_odds': bet365_daxiao[2],
                        'time': bet365_daxiao[7] if len(bet365_daxiao) > 7 else None,
                    },
                    'close': {
                        'line': bet365_daxiao[4],
                        'over_odds': bet365_daxiao[3],
                        'under_odds': bet365_daxiao[5],
                        'time': bet365_daxiao[6] if len(bet365_daxiao) > 6 else None,
                    }
                }
            result['bet365'] = bet365_data
        
        # 提取 Pinnacle 数据
        pinnacle_yazhi = _extract_company_odds(yazhi_html, 'Pinnacle', is_total=False)
        pinnacle_daxiao = _extract_company_odds(daxiao_html, 'Pinnacle', is_total=True)
        
        # 只要有亚盘数据就返回（大小球可选）
        if pinnacle_yazhi:
            pinnacle_data = {
                'asian': {
                    'open': {
                        'handicap': pinnacle_yazhi[1],  # 初让球
                        'home_odds': pinnacle_yazhi[0],  # 初主队水位
                        'away_odds': pinnacle_yazhi[2],  # 初客队水位
                        'time': pinnacle_yazhi[7] if len(pinnacle_yazhi) > 7 else None,  # 初盘时间
                    },
                    'close': {
                        'handicap': pinnacle_yazhi[4],  # 终让球
                        'home_odds': pinnacle_yazhi[3],  # 终主队水位
                        'away_odds': pinnacle_yazhi[5],  # 终客队水位
                        'time': pinnacle_yazhi[6] if len(pinnacle_yazhi) > 6 else None,  # 终盘时间
                    }
                }
            }
            # 如果有大小球数据也加上
            if pinnacle_daxiao:
                pinnacle_data['total'] = {
                    'open': {
                        'line': pinnacle_daxiao[1],
                        'over_odds': pinnacle_daxiao[0],
                        'under_odds': pinnacle_daxiao[2],
                        'time': pinnacle_daxiao[7] if len(pinnacle_daxiao) > 7 else None,
                    },
                    'close': {
                        'line': pinnacle_daxiao[4],
                        'over_odds': pinnacle_daxiao[3],
                        'under_odds': pinnacle_daxiao[5],
                        'time': pinnacle_daxiao[6] if len(pinnacle_daxiao) > 6 else None,
                    }
                }
            result['pinnacle'] = pinnacle_data
        
        log.info(f"独赔数据抓取完成: Bet365={'有' if result['bet365'] else '无'}, Pinnacle={'有' if result['pinnacle'] else '无'}")
        
    except Exception as e:
        log.warning(f"抓取独赔数据失败: {e}")
    
    return result


def fetch_yazhi(match_id):
    """抓取亚盘数据。平均值行格式: 初水位 初让球 初水位 终水位 终让球 终水位"""
    html, nums = _fetch_avg_page(match_id, 'yazhi')

    segment = _html_to_text(html)
    idx = segment.find('平均值')
    segment = segment[idx:idx + 200]

    open_hcap_raw = _extract_handicap_from_segment(segment, nums[0], nums[2])
    close_hcap_raw = _extract_handicap_from_segment(
        segment[segment.find(str(nums[3])):] if str(nums[3]) in segment else segment,
        nums[3], nums[5]
    )

    # 500.com 数值让球为负表示主让，取反以符合脚本惯例（正=主让）
    # 平均值行第一组为初盘、第二组为终盘
    return {
        'open': {
            'handicap': -open_hcap_raw,
            'home_odds': nums[0],
            'away_odds': nums[2],
        },
        'close': {
            'handicap': -close_hcap_raw,
            'home_odds': nums[3],
            'away_odds': nums[5],
        }
    }


def _extract_handicap_from_segment(segment, before_val, after_val):
    """从文本片段中提取两个数字之间的让球值（可能是文本或数字）"""
    pat = re.compile(
        rf'{re.escape(str(before_val))}\s+([^\d\s]+(?:/[^\d\s]+)?)\s+{re.escape(str(after_val))}'
    )
    m = pat.search(segment)
    if m:
        handicap_str = m.group(1)
        # 尝试数字解析
        try:
            return float(handicap_str)
        except ValueError:
            return parse_handicap(handicap_str)

    # 如果上面的模式没匹配，尝试直接数字匹配
    pat2 = re.compile(
        rf'{re.escape(str(before_val))}\s+(-?[\d.]+)\s+{re.escape(str(after_val))}'
    )
    m2 = pat2.search(segment)
    if m2:
        return float(m2.group(1))

    return 0


def _parse_odds_value(value, field_name, match_id):
    """解析赔率值，确保为有效正数"""
    if value is None:
        raise ValueError(f"赔率值为空: {field_name} (match_id={match_id})")
    
    try:
        val = float(value)
        if val <= 0:
            raise ValueError(f"赔率值必须为正数: {field_name} = {val} (match_id={match_id})")
        return val
    except (ValueError, TypeError):
        raise ValueError(f"赔率值解析失败: {field_name} = {repr(value)} (match_id={match_id})")


def fetch_ouzhi(match_id):
    """抓取欧赔平均值（JSON 时间序列）。每条为 [主, 平, 客, 返还率, 时间, ...]"""
    url = f'{OUZHI_JSON_URL}?fid={match_id}&cid=0&type=europe&r=1'
    referer = f'{BASE}/fenxi/ouzhi-{match_id}.shtml'
    
    try:
        series = _fetching_mod.fetch_json(url, referer=referer)
    except Exception as e:
        raise ValueError(f"抓取欧赔数据失败: {e} (match_id={match_id})")
    
    # 数据有效性检查
    if not isinstance(series, list):
        raise ValueError(f"欧赔数据格式错误，期望列表但得到: {type(series)} (match_id={match_id})")
    
    if len(series) == 0:
        raise ValueError(f"欧赔数据为空列表 (match_id={match_id})")
    
    # 检查数据点格式
    close = series[0]
    open_ = series[-1]
    
    if not isinstance(close, (list, tuple)) or len(close) < 3:
        raise ValueError(f"终盘数据格式错误: {close} (match_id={match_id})")
    
    if not isinstance(open_, (list, tuple)) or len(open_) < 3:
        raise ValueError(f"初盘数据格式错误: {open_} (match_id={match_id})")

    try:
        return {
            'open': {
                'home': _parse_odds_value(open_[0], 'open_home', match_id),
                'draw': _parse_odds_value(open_[1], 'open_draw', match_id),
                'away': _parse_odds_value(open_[2], 'open_away', match_id),
                'return_rate': float(open_[3]) if len(open_) > 3 and open_[3] else None,
            },
            'close': {
                'home': _parse_odds_value(close[0], 'close_home', match_id),
                'draw': _parse_odds_value(close[1], 'close_draw', match_id),
                'away': _parse_odds_value(close[2], 'close_away', match_id),
                'return_rate': float(close[3]) if len(close) > 3 and close[3] else None,
            },
            'series': series,
        }
    except ValueError as e:
        raise ValueError(f"欧赔数据解析失败: {e} (match_id={match_id})")


def fetch_daxiao(match_id):
    """抓取大小球数据。平均值行盘口线为纯数字，第一组为初盘、第二组为终盘"""
    _, nums = _fetch_avg_page(match_id, 'daxiao')
    return {
        'open': {
            'line': nums[1],
            'over_odds': nums[0],
            'under_odds': nums[2],
        },
        'close': {
            'line': nums[4],
            'over_odds': nums[3],
            'under_odds': nums[5],
        }
    }


RECENT_FORM_PAT = re.compile(
    r'近(\d+)场战绩.*?'
    r'<span class="ying">(\d+)胜</span>.*?'
    r'<span class="ping">(\d+)平</span>.*?'
    r'<span class="shu">(\d+)负</span>.*?'
    r'进<span class="ying">(\d+)球</span>失<span class="shu">(\d+)球</span>',
    re.DOTALL,
)


def _team_in_context(ctx, name):
    """队名与上下文模糊匹配（兼容简称）"""
    if not name:
        return False
    if name in ctx:
        return True
    for n in (4, 3, 2):
        if len(name) >= n and name[-n:] in ctx:
            return True
    return False


def get_live_league_profile(league_name: str, recent_matches: int = 200) -> Dict:
    """
    从最近比赛数据计算实时联赛画像
    
    参数：
        league_name: 联赛名称
        recent_matches: 最近比赛数量
    
    返回：
        实时画像字典
    """
    try:
        from .data_loader import fetch_league_matches
        
        matches = fetch_league_matches(league_name, limit=recent_matches)
        
        if not matches:
            return None
        
        total_matches = 0
        total_goals = 0
        draw_count = 0
        home_win_count = 0
        btts_count = 0
        over25_count = 0
        
        for match in matches:
            score = match.get('score')
            if score:
                parts = score.split('-')
                if len(parts) == 2:
                    try:
                        h, a = map(int, parts)
                        total_matches += 1
                        total_goals += h + a
                        if h == a:
                            draw_count += 1
                        if h > a:
                            home_win_count += 1
                        if h > 0 and a > 0:
                            btts_count += 1
                        if h + a >= 3:
                            over25_count += 1
                    except:
                        pass
        
        if total_matches == 0:
            return None
        
        return {
            'avg_goal': total_goals / (total_matches * 2),  # 场均进球（单队）
            'draw_rate': draw_count / total_matches,
            'home_win_rate': home_win_count / total_matches,
            'btts_rate': btts_count / total_matches,
            'over25_rate': over25_count / total_matches,
            'sample_size': total_matches,
            'source': 'live'
        }
        
    except Exception as e:
        log.debug(f"计算实时联赛画像失败: {e}")
        return None


def resolve_league_profile(league_name):
    """按联赛名称匹配画像，用于场均进球与比分先验（融合静态+实时）"""
    name = (league_name or '').strip()
    
    # 获取静态画像
    static_profile = dict(LEAGUE_PROFILES['default'])
    for key in sorted(LEAGUE_PROFILES, key=len, reverse=True):
        if key != 'default' and key in name:
            static_profile.update(LEAGUE_PROFILES[key])
            break
    
    # 获取实时画像
    live_profile = get_live_league_profile(name)
    
    if live_profile and live_profile.get('sample_size', 0) >= 50:
        # 融合静态和实时画像：70% 静态 + 30% 实时
        blended_profile = {
            'avg_goal': 0.7 * static_profile.get('avg_goal', 1.42) + 0.3 * live_profile.get('avg_goal', 1.42),
            'home_boost': static_profile.get('home_boost', 1.0),
            'low_score': static_profile.get('low_score', 1.0),
            'draw_mult': 0.7 * static_profile.get('draw_mult', 1.0) + 0.3 * (live_profile.get('draw_rate', 0.25) / 0.25),
            'name': name or 'default',
            'live_sample': live_profile.get('sample_size', 0),
            'source': 'blended'
        }
        return blended_profile
    
    static_profile['name'] = name or 'default'
    static_profile['source'] = 'static'
    return static_profile


def _parse_recent_form(groups):
    n, w, d, l = int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
    gf, ga = int(groups[4]), int(groups[5])
    n = max(n, 1)
    pts = w * 3 + d
    return {
        'games': n, 'wins': w, 'draws': d, 'losses': l,
        'gf': gf, 'ga': ga, 'attack': gf / n, 'defense': ga / n,
        'form_pts': pts / n,
    }


def fetch_team_strength(match_id, home, away, league_profile=None):
    """
    从数据分析页抓取主客队近10场及主客场进球/失球，换算攻防强度。
    返回 None 表示页面无数据（不影响主流程）。
    
    集成 ELO 评分系统：
    - 获取球队 ELO 评分
    - 将 ELO 转换为进球期望值 (xG)
    - 返回包含 ELO 信息的综合实力数据
    """
    try:
        html = _fetching_mod.fetch(f'{BASE}/fenxi/shuju-{match_id}.shtml')
    except (urllib.error.URLError, ValueError, OSError):
        return None

    tagged = []
    for m in RECENT_FORM_PAT.finditer(html):
        # 仅用紧邻战绩前的短上下文识别队名，避免多场数据串台
        ctx = _html_to_text(html[max(0, m.start() - 140):m.start()])
        tagged.append({'ctx': ctx, 'stats': _parse_recent_form(m.groups())})

    if len(tagged) < 2:
        return None

    home_all = away_all = home_venue = away_venue = None
    for item in tagged:
        ctx, st = item['ctx'], item['stats']
        if _team_in_context(ctx, home):
            if home_all is None:
                home_all = st
            elif home_venue is None:
                home_venue = st
        elif _team_in_context(ctx, away):
            if away_all is None:
                away_all = st
            elif away_venue is None:
                away_venue = st

    if not home_all or not away_all:
        return None

    hv = home_venue or home_all
    av = away_venue or away_all
    attack_home = _blend_close_open(hv['attack'], home_all['attack'], 0.68)
    defense_home = _blend_close_open(hv['defense'], home_all['defense'], 0.68)
    attack_away = _blend_close_open(av['attack'], away_all['attack'], 0.68)
    defense_away = _blend_close_open(av['defense'], away_all['defense'], 0.68)

    form_diff = home_all['form_pts'] - away_all['form_pts']
    
    # ELO 评分集成
    elo_home = elo_away = None
    elo_xg_home = elo_xg_away = None
    elo_strength_home = elo_strength_away = None
    elo_prediction = None
    
    if ELO_AVAILABLE:
        try:
            elo = get_elo_system()
            elo_home = elo.get_rating(home)
            elo_away = elo.get_rating(away)
            
            # 计算基于 ELO 的进球期望值
            elo_xg_home = elo_to_goals_expected(elo_home, elo_away)
            elo_xg_away = elo_to_goals_expected(elo_away, elo_home)
            
            # 计算实力因子
            elo_strength_home = elo_to_strength_factor(elo_home)
            elo_strength_away = elo_to_strength_factor(elo_away)
            
            # 获取 ELO 预测
            league_type = league_profile.get('name', '联赛') if league_profile else '联赛'
            elo_prediction = elo.predict_match(home, away, league_type)
            
            log.debug(f"ELO 评分: {home}={elo_home:.2f}, {away}={elo_away:.2f}")
            log.debug(f"ELO xG: {home}={elo_xg_home:.2f}, {away}={elo_xg_away:.2f}")
        except Exception as e:
            log.error(f"ELO 计算失败: {e}")

    result = {
        'home_recent': home_all,
        'away_recent': away_all,
        'home_venue': hv,
        'away_venue': av,
        'attack_home': attack_home,
        'defense_home': defense_home,
        'attack_away': attack_away,
        'defense_away': defense_away,
        'form_diff': form_diff,
        'momentum_supremacy': max(-0.35, min(0.35, form_diff * 0.12)),
        'summary': (
            f"主队近{home_all['games']}场 进{home_all['gf']}失{home_all['ga']}（{home_all['form_pts']:.1f}分/场）；"
            f"客队近{away_all['games']}场 进{away_all['gf']}失{away_all['ga']}（{away_all['form_pts']:.1f}分/场）"
        ),
    }
    
    # 添加 ELO 相关数据
    if ELO_AVAILABLE and elo_home is not None:
        result.update({
            'elo_home': elo_home,
            'elo_away': elo_away,
            'elo_xg_home': elo_xg_home,
            'elo_xg_away': elo_xg_away,
            'elo_strength_home': elo_strength_home,
            'elo_strength_away': elo_strength_away,
            'elo_prediction': elo_prediction,
        })
    
    return result


def _blend_close_open(close_val, open_val, close_weight=CLOSE_BLEND_WEIGHT):
    """终盘为主、初盘为辅的线性融合"""
    if open_val is None:
        return close_val
    w = close_weight
    return w * close_val + (1.0 - w) * open_val


