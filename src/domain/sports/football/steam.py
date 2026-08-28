# -*- coding: utf-8 -*-
"""临场资金流突变（steam move）与诱盘（trap）识别。

纯计算——**只用 `strptime` 解析传入的时间串，不读 `now()`**（已实测：
全模块零处 `now()`），所以整块搬得动。落盘与调度留在适配层。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

log = logging.getLogger('domain.football.steam')


# 迁移当时的真实常量（从 steam_move.py 原样搬来）
CRITICAL_TIME_WINDOW = 30  # 赛前30分钟为关键期

STEAM_FAST_THRESHOLD = 0.02    # 快速变化阈值

STEAM_CRITICAL_THRESHOLD = 0.05  # 急速变化阈值

class SteamSignal:
    """资金流信号"""
    
    def __init__(self, signal_type: str, confidence: float, description: str, details: dict = None):
        self.signal_type = signal_type  # 'steam_rise', 'steam_drop', 'trap', 'stable'
        self.confidence = confidence    # 置信度 0~1
        self.description = description  # 描述
        self.details = details or {}    # 详细信息
    
    def to_dict(self) -> Dict:
        return {
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'description': self.description,
            'details': self.details,
        }


def _analyze_asian_steam(asian_data: Dict, match_time: str) -> Dict:
    """
    分析亚盘资金流
    
    返回：
        亚盘资金流分析结果
    """
    result = {
        'handicap_speed': 0.0,
        'handicap_acceleration': 0.0,
        'water_speed': 0.0,
        'water_acceleration': 0.0,
        'time_remaining': None,
        'is_critical_period': False,
        'signals': [],
        'trap_analysis': None,
    }
    
    # 获取初盘和终盘数据
    open_hcap = asian_data.get('open_handicap')
    close_hcap = asian_data.get('handicap')
    open_time = asian_data.get('open_time')
    close_time = asian_data.get('close_time')
    
    open_water_home = asian_data.get('open_water', {}).get('home')
    close_water_home = asian_data.get('close_water', {}).get('home')
    
    # 计算时间差（分钟）
    time_diff = _calculate_time_diff(open_time, close_time)
    
    if time_diff is None or time_diff <= 0:
        time_diff = 360  # 默认6小时
    
    # 判断是否处于关键期（赛前30分钟）
    time_remaining = _calculate_time_remaining(match_time, close_time)
    result['time_remaining'] = time_remaining
    result['is_critical_period'] = time_remaining is not None and time_remaining <= CRITICAL_TIME_WINDOW
    
    # 计算让球变化速度
    if open_hcap is not None and close_hcap is not None:
        hcap_change = close_hcap - open_hcap
        result['handicap_speed'] = hcap_change / time_diff * 60  # 每分钟变化量
        
        # 检测急升/急跌
        abs_speed = abs(result['handicap_speed'])
        if abs_speed >= STEAM_CRITICAL_THRESHOLD:
            signal_type = 'steam_rise' if hcap_change > 0 else 'steam_drop'
            confidence = min(1.0, abs_speed / STEAM_CRITICAL_THRESHOLD)
            change_desc = '升高' if hcap_change > 0 else '降低'
            desc = f"让球急速变化: {open_hcap:+.2f} → {close_hcap:+.2f}（{change_desc}）"
            result['signals'].append(SteamSignal(signal_type, confidence, desc, {
                'speed': result['handicap_speed'],
                'change': hcap_change,
                'time_diff': time_diff,
            }))
        elif abs_speed >= STEAM_FAST_THRESHOLD:
            signal_type = 'steam_rise' if hcap_change > 0 else 'steam_drop'
            confidence = min(0.8, abs_speed / STEAM_FAST_THRESHOLD)
            desc = f"让球快速变化: {open_hcap:+.2f} → {close_hcap:+.2f}"
            result['signals'].append(SteamSignal(signal_type, confidence, desc, {
                'speed': result['handicap_speed'],
                'change': hcap_change,
            }))
    
    # 计算水位变化速度
    if open_water_home is not None and close_water_home is not None:
        water_change = close_water_home - open_water_home
        result['water_speed'] = water_change / time_diff * 60
        
        # 检测水位急升/急跌
        abs_water_speed = abs(result['water_speed'])
        if abs_water_speed >= 0.03:  # 水位每分钟变化0.03以上
            signal_type = 'steam_rise' if water_change > 0 else 'steam_drop'
            confidence = min(1.0, abs_water_speed / 0.03)
            desc = f"水位急速变化: {open_water_home:.2f} → {close_water_home:.2f}"
            result['signals'].append(SteamSignal(signal_type, confidence, desc, {
                'speed': result['water_speed'],
                'change': water_change,
            }))
    
    # 诱盘分析
    result['trap_analysis'] = _analyze_trap_pattern(asian_data)
    
    # 将信号转换为字典以便JSON序列化
    result['signals'] = [sig.to_dict() for sig in result['signals']]
    
    return result


def _analyze_total_steam(total_data: Dict, match_time: str) -> Dict:
    """
    分析大小球资金流
    """
    result = {
        'line_speed': 0.0,
        'over_water_speed': 0.0,
        'under_water_speed': 0.0,
        'time_remaining': None,
        'is_critical_period': False,
        'signals': [],
        'trap_analysis': None,
    }
    
    open_line = total_data.get('open_line')
    close_line = total_data.get('close_line')
    open_time = total_data.get('open_time')
    close_time = total_data.get('close_time')
    
    open_over = total_data.get('open_water', {}).get('over')
    close_over = total_data.get('close_water', {}).get('over')
    
    time_diff = _calculate_time_diff(open_time, close_time)
    if time_diff is None or time_diff <= 0:
        time_diff = 360
    
    time_remaining = _calculate_time_remaining(match_time, close_time)
    result['time_remaining'] = time_remaining
    result['is_critical_period'] = time_remaining is not None and time_remaining <= CRITICAL_TIME_WINDOW
    
    # 计算大小球线变化速度
    if open_line is not None and close_line is not None:
        line_change = close_line - open_line
        result['line_speed'] = line_change / time_diff * 60
        
        if abs(result['line_speed']) >= 0.02:
            signal_type = 'steam_rise' if line_change > 0 else 'steam_drop'
            confidence = min(1.0, abs(result['line_speed']) / 0.02)
            desc = f"大小球线快速变化: {open_line} → {close_line}"
            result['signals'].append(SteamSignal(signal_type, confidence, desc, {
                'speed': result['line_speed'],
                'change': line_change,
            }))
    
    # 计算大球水位变化
    if open_over is not None and close_over is not None:
        over_change = close_over - open_over
        result['over_water_speed'] = over_change / time_diff * 60
        
        if abs(result['over_water_speed']) >= 0.03:
            signal_type = 'steam_rise' if over_change > 0 else 'steam_drop'
            confidence = min(1.0, abs(result['over_water_speed']) / 0.03)
            desc = f"大球水位快速变化: {open_over:.2f} → {close_over:.2f}"
            result['signals'].append(SteamSignal(signal_type, confidence, desc))
    
    # 诱盘分析
    result['trap_analysis'] = _analyze_total_trap(total_data)
    
    # 将信号转换为字典以便JSON序列化
    result['signals'] = [sig.to_dict() for sig in result['signals']]
    
    return result


def _analyze_trap_pattern(asian_data: Dict) -> Dict:
    """
    分析诱盘模式
    
    诱盘特征：
    1. 让球与水位反向变化
    2. 临开场前快速反转
    3. 高水方突然降水
    """
    result = {
        'is_trap': False,
        'trap_type': None,  # 'reverse', 'late_swing', 'high_water_drop'
        'confidence': 0.0,
        'reason': '',
        'details': {},
    }
    
    open_hcap = asian_data.get('open_handicap')
    close_hcap = asian_data.get('handicap')
    open_water_home = asian_data.get('open_water', {}).get('home')
    close_water_home = asian_data.get('close_water', {}).get('home')
    open_water_away = asian_data.get('open_water', {}).get('away')
    close_water_away = asian_data.get('close_water', {}).get('away')
    
    if None in [open_hcap, close_hcap, open_water_home, close_water_home]:
        return result
    
    hcap_change = close_hcap - open_hcap
    water_change_home = close_water_home - open_water_home
    
    factors = []
    confidence = 0.0
    
    # 特征1：让球与水位反向变化（诱盘常见模式）
    # 让球升高但主队水位也升高 → 可能诱盘
    if hcap_change > 0.25 and water_change_home > 0.08:
        factors.append("让球升高但主队水位同步上升")
        confidence += 0.3
    
    # 特征2：让球降低但主队水位下降 → 可能诱下盘
    if hcap_change < -0.25 and water_change_home < -0.08:
        factors.append("让球降低但主队水位同步下降")
        confidence += 0.3
    
    # 特征3：高水方突然降水（可能是诱盘）
    if open_water_home > 2.0 and close_water_home < open_water_home - 0.15:
        factors.append("高水方(>2.0)突然大幅降水")
        confidence += 0.25
    
    # 特征4：水位交叉（主队和客队水位反转）
    if open_water_home < open_water_away and close_water_home > close_water_away:
        factors.append("水位交叉反转")
        confidence += 0.2
    
    # 特征5：让球方向反转
    if open_hcap * close_hcap < 0:
        factors.append("让球方向完全反转")
        confidence += 0.3
    
    if confidence >= 0.5:
        result['is_trap'] = True
        result['confidence'] = min(1.0, confidence)
        result['reason'] = "; ".join(factors)
        
        if confidence >= 0.7:
            result['trap_type'] = 'strong_trap'
        else:
            result['trap_type'] = 'possible_trap'
    
    return result


def _analyze_total_trap(total_data: Dict) -> Dict:
    """
    分析大小球诱盘模式
    """
    result = {
        'is_trap': False,
        'trap_type': None,
        'confidence': 0.0,
        'reason': '',
    }
    
    open_line = total_data.get('open_line')
    close_line = total_data.get('close_line')
    open_over = total_data.get('open_water', {}).get('over')
    close_over = total_data.get('close_water', {}).get('over')
    
    if None in [open_line, close_line, open_over, close_over]:
        return result
    
    line_change = close_line - open_line
    over_change = close_over - open_over
    
    confidence = 0.0
    factors = []
    
    # 特征1：大球水位下降但大小球线升高 → 诱大
    if over_change < -0.1 and line_change > 0:
        factors.append("大球水位下降但大小球线升高（诱大）")
        confidence += 0.35
    
    # 特征2：小球水位下降但大小球线降低 → 诱小
    if over_change > 0.1 and line_change < 0:
        factors.append("小球水位下降但大小球线降低（诱小）")
        confidence += 0.35
    
    # 特征3：高水方突然降水
    if open_over > 2.0 and close_over < open_over - 0.15:
        factors.append("大球高水突然降水")
        confidence += 0.2
    
    if confidence >= 0.5:
        result['is_trap'] = True
        result['confidence'] = min(1.0, confidence)
        result['reason'] = "; ".join(factors)
    
    return result


def _calculate_time_diff(start_time: str, end_time: str) -> Optional[float]:
    """
    计算时间差（分钟）
    """
    try:
        from datetime import datetime
        
        if start_time and end_time:
            fmt = '%Y-%m-%d %H:%M:%S'
            t1 = datetime.strptime(start_time, fmt)
            t2 = datetime.strptime(end_time, fmt)
            diff = (t2 - t1).total_seconds() / 60
            return max(0.1, diff)
    except Exception:
        pass
    return None


def _calculate_time_remaining(match_time: str, current_time: str) -> Optional[float]:
    """
    计算距离比赛开始的剩余时间（分钟）
    """
    try:
        from datetime import datetime
        
        if match_time and current_time:
            fmt = '%Y-%m-%d %H:%M:%S'
            match_dt = datetime.strptime(match_time, fmt)
            current_dt = datetime.strptime(current_time, fmt)
            diff = (match_dt - current_dt).total_seconds() / 60
            return max(0, diff)
    except Exception:
        pass
    return None


def _summarize_signals(signals: List) -> Dict:
    """
    汇总所有信号（支持字典形式的信号）
    """
    if not signals:
        return {
            'has_strong_signal': False,
            'signal_count': 0,
            'dominant_signal': 'stable',
            'confidence': 0.0,
            'recommendation': '无明显资金流信号',
        }
    
    result = {
        'has_strong_signal': False,
        'signal_count': len(signals),
        'dominant_signal': None,
        'confidence': 0.0,
        'recommendation': '',
    }
    
    # 统计各类信号
    signal_counts = {}
    total_confidence = 0.0
    
    for signal in signals:
        # 支持字典和对象两种形式
        if isinstance(signal, dict):
            signal_type = signal.get('signal_type', 'unknown')
            confidence = signal.get('confidence', 0.0)
        else:
            signal_type = signal.signal_type
            confidence = signal.confidence
        
        signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        total_confidence += confidence
        
        if confidence >= 0.7:
            result['has_strong_signal'] = True
    
    # 找到主导信号
    dominant_type = max(signal_counts, key=signal_counts.get)
    result['dominant_signal'] = dominant_type
    result['confidence'] = total_confidence / len(signals)
    
    # 生成建议
    if dominant_type == 'steam_rise':
        result['recommendation'] = "资金流入明显，注意热门方向"
    elif dominant_type == 'steam_drop':
        result['recommendation'] = "资金流出明显，注意冷门方向"
    elif dominant_type == 'trap':
        result['recommendation'] = "疑似诱盘，建议反向操作"
    
    return result


def _signal_to_dict(s) -> Dict:
    """兼容函数：将信号转换为字典（支持字典和对象两种形式）"""
    return s if isinstance(s, dict) else s.to_dict()


def _normalize_match_time(match_time: str, now=None) -> str:
    """
    标准化比赛时间格式为 YYYY-MM-DD HH:MM:SS
    
    输入格式支持：
    - YYYY-MM-DD HH:MM:SS（已有年份）
    - MM-DD HH:MM（需要补年份）
    """
    if not match_time:
        return None
    
    from datetime import datetime
    
    # 尝试解析不同格式
    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m-%d %H:%M']
    
    for fmt in formats:
        try:
            dt = datetime.strptime(match_time, fmt)
            # 如果只有月日，补充当前年份
            if fmt == '%m-%d %H:%M':
                # **年份由调用方注入**（判据 16）：源站的时间串常常不带年，
                # 补的是"当前年"——这是这个模块里唯一的时钟依赖。
                current_year = (now or datetime.now()).year
                dt = dt.replace(year=current_year)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    
    return match_time
