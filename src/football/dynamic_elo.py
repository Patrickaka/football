#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态ELO系统 - 考虑近期状态
=============================

功能：
1. 长期ELO：传统ELO评分
2. 近期ELO：最近N场比赛的ELO趋势
3. 综合ELO：0.7*长期 + 0.3*近期

这样可以捕捉球队的状态变化。
"""

import os
import json
from typing import Dict, List, Tuple, Optional

# ==================== 常量配置 ====================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
ELO_DB_FILE = os.path.join(DATA_DIR, 'dynamic_elo_db.json')

RECENT_GAMES = 10  # 近期比赛数量
LONG_TERM_WEIGHT = 0.7  # 长期ELO权重
SHORT_TERM_WEIGHT = 0.3  # 近期ELO权重

class DynamicELO:
    """动态ELO评分系统"""
    
    def __init__(self):
        self.teams = {}  # {team_name: {'elo': 1500, 'history': [...]}
        self._load()
    
    def _load(self):
        """加载ELO数据库"""
        if os.path.exists(ELO_DB_FILE):
            try:
                with open(ELO_DB_FILE, 'r', encoding='utf-8') as f:
                    self.teams = json.load(f)
                print(f"已加载动态ELO数据库，{len(self.teams)} 支球队")
            except Exception as e:
                print(f"加载ELO数据库失败: {e}")
                self.teams = {}
    
    def save(self):
        """保存ELO数据库"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(ELO_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.teams, f, ensure_ascii=False, indent=2)
    
    def get_elo(self, team_name: str) -> Tuple[float, float, float]:
        """
        获取球队的ELO评分
        
        返回：(长期ELO, 近期ELO, 综合ELO)
        """
        if team_name not in self.teams:
            # 默认初始ELO
            return 1500.0, 1500.0, 1500.0
        
        team = self.teams[team_name]
        long_term = team.get('elo', 1500.0)
        
        # 计算近期ELO（最近10场的平均趋势）
        history = team.get('history', [])
        if len(history) >= 3:
            # 使用最近几场的趋势
            recent_games = history[-RECENT_GAMES:]
            if len(recent_games) >= 3:
                # 计算趋势：最近3场的平均变化
                recent_changes = []
                for i in range(1, len(recent_games)):
                    recent_changes.append(recent_games[i] - recent_games[i-1])
                if recent_changes:
                    avg_change = sum(recent_changes) / len(recent_changes)
                    short_term = long_term + avg_change * 3  # 趋势外推
                else:
                    short_term = long_term
            else:
                short_term = long_term
        else:
            short_term = long_term
        
        # 综合ELO
        combined = LONG_TERM_WEIGHT * long_term + SHORT_TERM_WEIGHT * short_term
        
        return long_term, short_term, combined
    
    def update_elo(self, team_name: str, new_elo: float):
        """
        更新球队ELO
        
        参数：
            team_name: 球队名称
            new_elo: 新的ELO评分
        """
        if team_name not in self.teams:
            self.teams[team_name] = {'elo': 1500.0, 'history': []}
        
        team = self.teams[team_name]
        old_elo = team['elo']
        
        # 更新历史记录
        history = team.get('history', [])
        history.append(old_elo)
        if len(history) > 50:
            history = history[-50:]  # 保留最近50场
        team['history'] = history
        
        # 更新当前ELO
        team['elo'] = new_elo
    
    def get_elo_diff(self, home_team: str, away_team: str) -> float:
        """
        获取两队的综合ELO差值
        
        返回：主队ELO - 客队ELO
        """
        _, _, home_elo = self.get_elo(home_team)
        _, _, away_elo = self.get_elo(away_team)
        return home_elo - away_elo

# ==================== 全局实例 ====================
_elo_system = None

def get_elo_system() -> DynamicELO:
    """获取全局ELO系统实例"""
    global _elo_system
    if _elo_system is None:
        _elo_system = DynamicELO()
    return _elo_system

def get_team_elo(team_name: str) -> Tuple[float, float, float]:
    """获取球队ELO的便捷接口"""
    return get_elo_system().get_elo(team_name)

def get_elo_difference(home_team: str, away_team: str) -> float:
    """获取两队ELO差值的便捷接口"""
    return get_elo_system().get_elo_diff(home_team, away_team)
