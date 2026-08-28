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

from ..domain.sports.football import steam as _steam

# 纯计算转发给领域层
_analyze_asian_steam = _steam._analyze_asian_steam
_analyze_total_steam = _steam._analyze_total_steam
_analyze_trap_pattern = _steam._analyze_trap_pattern
_analyze_total_trap = _steam._analyze_total_trap
_calculate_time_diff = _steam._calculate_time_diff
_calculate_time_remaining = _steam._calculate_time_remaining
_summarize_signals = _steam._summarize_signals
_signal_to_dict = _steam._signal_to_dict
_normalize_match_time = _steam._normalize_match_time

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
    # 获取时间戳并标准化格式
    match_time = _normalize_match_time(match.get('time'))
    
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
    asian['steam_signals'] = [_signal_to_dict(s) for s in asian_steam.get('signals', [])]
    
    # 更新total字典
    if total:
        total['steam_speed'] = total_steam.get('line_speed', 0.0)
        total['is_critical_period'] = total_steam.get('is_critical_period', False)
        total['trap_analysis'] = total_steam.get('trap_analysis')
        total['steam_signals'] = [_signal_to_dict(s) for s in total_steam.get('signals', [])]
    
    # 返回综合结果
    return {
        'asian': asian,
        'total': total,
        'steam_summary': steam_result.get('summary'),
        'all_signals': [_signal_to_dict(s) for s in steam_result.get('signals', [])],
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