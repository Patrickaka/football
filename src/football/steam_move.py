#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临场资金流模型 - Steam Move Detector
====================================

功能：
1. 检测盘口变化速度（急跌/急升）
2. 识别诱盘模式
3. 综合资金流信号分析

这是职业模型的核心模块之一，临场资金流动是非常强的信号。
"""

import math
from typing import Dict, List, Tuple, Optional, Any

# ==================== 常量配置 ====================

# 时间阈值（分钟）
CRITICAL_TIME_WINDOW = 30  # 赛前30分钟为关键期
IMPORTANT_TIME_WINDOW = 60  # 赛前60分钟为重要期

# 变化速度阈值（每分钟变化量）
STEAM_FAST_THRESHOLD = 0.02    # 快速变化阈值
STEAM_CRITICAL_THRESHOLD = 0.05  # 急速变化阈值

# 诱盘识别阈值
诱盘水位反转阈值 = 0.15  # 水位反转超过15%可能是诱盘
诱盘让球反转阈值 = 0.5   # 让球反转超过0.5球可能是诱盘

# ==================== 资金流信号类型 ====================

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


# ==================== 资金流检测器 ====================

def steam_move_detector(asian_data: Dict, total_data: Dict = None, match_time: str = None) -> Dict:
    """
    临场资金流检测器 - 主接口
    
    参数：
        asian_data: 亚盘数据（包含初盘、终盘、时间戳）
        total_data: 大小球数据（可选）
        match_time: 比赛时间（用于计算剩余时间）
    
    返回：
        资金流分析结果字典
    """
    result = {
        'asian': _analyze_asian_steam(asian_data, match_time),
        'total': _analyze_total_steam(total_data, match_time) if total_data else None,
        'summary': None,
        'signals': [],
    }
    
    # 汇总信号（已经是字典形式）
    signals = []
    
    # 亚盘信号
    asian_result = result['asian']
    if asian_result:
        signals.extend(asian_result.get('signals', []))
    
    # 大小球信号
    if result['total']:
        signals.extend(result['total'].get('signals', []))
    
    # 综合分析
    result['signals'] = signals
    result['summary'] = _summarize_signals(signals)
    
    return result


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


# ==================== 集成接口 ====================

def integrate_steam_signal(asian: Dict, euro: Dict, total: Dict, match: Dict) -> Dict:
    """
    将资金流信号集成到现有分析数据中
    
    参数：
        asian: 亚盘分析结果
        euro: 欧赔分析结果
        total: 大小球分析结果
        match: 比赛信息
    
    返回：
        包含资金流信号的综合结果
    """
    # 获取时间戳
    match_time = match.get('time')
    
    # 运行资金流检测
    steam_result = steam_move_detector(asian, total, match_time)
    
    # 提取关键信号
    asian_steam = steam_result.get('asian', {})
    total_steam = steam_result.get('total', {})
    
    # 更新asian字典
    asian['steam_speed'] = asian_steam.get('handicap_speed', 0.0)
    asian['water_speed'] = asian_steam.get('water_speed', 0.0)
    asian['is_critical_period'] = asian_steam.get('is_critical_period', False)
    asian['trap_analysis'] = asian_steam.get('trap_analysis')
    asian['steam_signals'] = [s.to_dict() for s in asian_steam.get('signals', [])]
    
    # 更新total字典
    if total:
        total['steam_speed'] = total_steam.get('line_speed', 0.0)
        total['is_critical_period'] = total_steam.get('is_critical_period', False)
        total['trap_analysis'] = total_steam.get('trap_analysis')
        total['steam_signals'] = [s.to_dict() for s in total_steam.get('signals', [])]
    
    # 返回综合结果
    return {
        'asian': asian,
        'total': total,
        'steam_summary': steam_result.get('summary'),
        'all_signals': [s.to_dict() for s in steam_result.get('signals', [])],
    }


# ==================== 命令行测试 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='临场资金流检测器')
    parser.add_argument('--test', action='store_true', help='运行测试')
    
    args = parser.parse_args()
    
    if args.test:
        # 测试案例1：急跌信号
        print("=== 测试案例1：赛前30分钟水位急跌 ===")
        asian_data1 = {
            'open_handicap': -0.75,
            'handicap': -0.75,
            'open_water': {'home': 1.85, 'away': 1.90},
            'close_water': {'home': 1.55, 'away': 2.20},
            'open_time': '2026-06-10 18:00:00',
            'close_time': '2026-06-10 19:30:00',  # 赛前30分钟
        }
        
        result1 = steam_move_detector(asian_data1, match_time='2026-06-10 20:00:00')
        print(f"信号数量: {len(result1['signals'])}")
        for sig in result1['signals']:
            print(f"  - {sig.description} (置信度: {sig.confidence:.2%})")
        print(f"诱盘分析: {result1['asian']['trap_analysis']}")
        
        # 测试案例2：让球急速升高
        print("\n=== 测试案例2：让球急速升高 ===")
        asian_data2 = {
            'open_handicap': -0.5,
            'handicap': -1.0,
            'open_water': {'home': 1.90, 'away': 1.85},
            'close_water': {'home': 2.05, 'away': 1.70},
            'open_time': '2026-06-10 19:00:00',
            'close_time': '2026-06-10 19:45:00',
        }
        
        result2 = steam_move_detector(asian_data2, match_time='2026-06-10 20:00:00')
        print(f"信号数量: {len(result2['signals'])}")
        for sig in result2['signals']:
            print(f"  - {sig.description} (置信度: {sig.confidence:.2%})")
        
        # 测试案例3：疑似诱盘
        print("\n=== 测试案例3：疑似诱盘 ===")
        asian_data3 = {
            'open_handicap': -0.75,
            'handicap': -0.25,  # 让球降低
            'open_water': {'home': 2.00, 'away': 1.80},
            'close_water': {'home': 1.75, 'away': 2.05},  # 主队水位下降
            'open_time': '2026-06-10 18:00:00',
            'close_time': '2026-06-10 19:45:00',
        }
        
        result3 = steam_move_detector(asian_data3, match_time='2026-06-10 20:00:00')
        trap = result3['asian']['trap_analysis']
        print(f"诱盘检测: {'是' if trap['is_trap'] else '否'} (置信度: {trap['confidence']:.2%})")
        print(f"理由: {trap['reason']}")


if __name__ == '__main__':
    main()
