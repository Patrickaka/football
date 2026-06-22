#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML训练数据集构建器
===================

功能：
1. 读取Football-Data原始CSV数据
2. 按日期排序构建赛前特征
3. 输出JSONL格式训练数据
4. 确保无未来数据泄漏
"""

import os
import csv
import json
import math
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime


# ==================== 常量配置 ====================

# 联赛映射
LEAGUE_MAP = {
    'E0': '英超',
    'SP1': '西甲',
    'D1': '德甲',
    'I1': '意甲',
    'F1': '法甲',
}

# 赛季列表
SEASONS = ['2425', '2526']

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_DIR = DATA_DIR
OUTPUT_FILE = os.path.join(DATA_DIR, 'ml_training_data.jsonl')

# 默认值
DEFAULT_ELO = 1500.0
K_FACTOR = 24


# ==================== 工具函数 ====================

def parse_date(date_str: str) -> str:
    """统一日期格式为ISO格式"""
    date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y']
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return date_str


def odds_to_prob(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    """
    将欧赔转换为去水后的概率
    
    参数：
        home_odds: 主胜赔率
        draw_odds: 平局赔率
        away_odds: 客胜赔率
    
    返回：
        (主胜概率, 平局概率, 客胜概率)
    """
    try:
        p_home = 1.0 / home_odds if home_odds > 1 else 0.0
        p_draw = 1.0 / draw_odds if draw_odds > 1 else 0.0
        p_away = 1.0 / away_odds if away_odds > 1 else 0.0
        
        total = p_home + p_draw + p_away
        if total > 0:
            return p_home / total, p_draw / total, p_away / total
        else:
            return 0.333, 0.333, 0.334
    except:
        return 0.333, 0.333, 0.334


def asian_odds_to_prob(handicap: float, home_odds: float, away_odds: float) -> Tuple[float, float]:
    """
    将亚盘转换为去水后的概率
    
    参数：
        handicap: 让球盘口
        home_odds: 主队赔率
        away_odds: 客队赔率
    
    返回：
        (主队概率, 客队概率)
    """
    try:
        p_home = 1.0 / home_odds if home_odds > 1 else 0.0
        p_away = 1.0 / away_odds if away_odds > 1 else 0.0
        
        total = p_home + p_away
        if total > 0:
            return p_home / total, p_away / total
        else:
            return 0.5, 0.5
    except:
        return 0.5, 0.5


def total_odds_to_prob(over_odds: float, under_odds: float) -> Tuple[float, float]:
    """
    将大小球赔率转换为概率
    
    参数：
        over_odds: 大球赔率
        under_odds: 小球赔率
    
    返回：
        (大球概率, 小球概率)
    """
    try:
        p_over = 1.0 / over_odds if over_odds > 1 else 0.0
        p_under = 1.0 / under_odds if under_odds > 1 else 0.0
        
        total = p_over + p_under
        if total > 0:
            return p_over / total, p_under / total
        else:
            return 0.5, 0.5
    except:
        return 0.5, 0.5


# ==================== ELO计算类 ====================

class EloCalculator:
    """ELO评分计算器"""
    
    def __init__(self):
        self.ratings: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    
    def get_rating(self, team: str) -> float:
        """获取球队ELO评分"""
        return self.ratings[team]
    
    def update_rating(self, home_team: str, away_team: str, 
                      home_goals: int, away_goals: int):
        """
        根据比赛结果更新ELO评分
        
        参数：
            home_team: 主队名称
            away_team: 客队名称
            home_goals: 主队进球
            away_goals: 客队进球
        """
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]
        
        # 计算预期结果
        expected_home = 1.0 / (1.0 + math.pow(10, (away_rating - home_rating) / 400))
        expected_away = 1.0 / (1.0 + math.pow(10, (home_rating - away_rating) / 400))
        
        # 实际结果
        if home_goals > away_goals:
            actual_home = 1.0
            actual_away = 0.0
        elif home_goals < away_goals:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
        
        # 进球数修正因子
        goals_diff = abs(home_goals - away_goals)
        if goals_diff <= 1:
            k_multiplier = 1.0
        elif goals_diff == 2:
            k_multiplier = 1.5
        else:
            k_multiplier = 1.5 + (goals_diff - 2) * 0.5
        
        # 更新评分
        self.ratings[home_team] = home_rating + K_FACTOR * k_multiplier * (actual_home - expected_home)
        self.ratings[away_team] = away_rating + K_FACTOR * k_multiplier * (actual_away - expected_away)


# ==================== 球队状态跟踪器 ====================

class TeamStatsTracker:
    """球队状态跟踪器"""
    
    def __init__(self):
        # 存储球队历史记录 [(date, home_team, away_team, home_goals, away_goals, result), ...]
        self.team_matches: Dict[str, List[Dict]] = defaultdict(list)
        self.league_matches: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_match(self, match: Dict):
        """添加一场比赛记录"""
        home_team = match['home_team']
        away_team = match['away_team']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        result = match['result']
        date = match['date']
        league = match['league']
        
        # 为主队添加记录
        self.team_matches[home_team].append({
            'date': date,
            'is_home': True,
            'goals_for': home_goals,
            'goals_against': away_goals,
            'result': result,
        })
        
        # 为客队添加记录
        self.team_matches[away_team].append({
            'date': date,
            'is_home': False,
            'goals_for': away_goals,
            'goals_against': home_goals,
            'result': 'A' if result == 'H' else ('H' if result == 'A' else 'D'),
        })
        
        # 为联赛添加记录
        self.league_matches[league].append({
            'date': date,
            'goals_total': home_goals + away_goals,
            'is_draw': (result == 'D'),
        })
    
    def get_recent_stats(self, team: str, n: int = 5, is_home: bool = None) -> Dict:
        """
        获取球队最近n场比赛的统计数据
        
        参数：
            team: 球队名称
            n: 最近n场
            is_home: None-所有比赛, True-主场, False-客场
        
        返回：
            统计字典
        """
        matches = self.team_matches.get(team, [])
        
        # 过滤主客场
        if is_home is not None:
            matches = [m for m in matches if m['is_home'] == is_home]
        
        # 取最近n场
        recent = matches[-n:]
        
        if not recent:
            return {
                'matches_count': 0,
                'avg_goals_for': 1.5,
                'avg_goals_against': 1.5,
                'avg_points': 1.5,
                'win_rate': 0.33,
                'draw_rate': 0.33,
            }
        
        total_goals_for = sum(m['goals_for'] for m in recent)
        total_goals_against = sum(m['goals_against'] for m in recent)
        total_points = sum(3 if m['result'] == 'H' else (1 if m['result'] == 'D' else 0) for m in recent)
        wins = sum(1 for m in recent if m['result'] == 'H')
        draws = sum(1 for m in recent if m['result'] == 'D')
        
        count = len(recent)
        return {
            'matches_count': count,
            'avg_goals_for': total_goals_for / count,
            'avg_goals_against': total_goals_against / count,
            'avg_points': total_points / count,
            'win_rate': wins / count,
            'draw_rate': draws / count,
        }
    
    def get_league_stats(self, league: str, n: int = 100) -> Dict:
        """
        获取联赛最近n场比赛的统计数据
        
        参数：
            league: 联赛名称
            n: 最近n场
        
        返回：
            统计字典
        """
        matches = self.league_matches.get(league, [])
        recent = matches[-n:]
        
        if not recent:
            return {
                'avg_goals': 2.7,
                'draw_rate': 0.26,
            }
        
        total_goals = sum(m['goals_total'] for m in recent)
        total_draws = sum(1 for m in recent if m['is_draw'])
        count = len(recent)
        
        return {
            'avg_goals': total_goals / count,
            'draw_rate': total_draws / count,
        }


# ==================== 数据集构建器 ====================

class DatasetBuilder:
    """ML训练数据集构建器"""
    
    def __init__(self):
        self.elo_calculator = EloCalculator()
        self.stats_tracker = TeamStatsTracker()
        self.raw_matches = []
    
    def load_raw_csv(self, league_code: str, season: str):
        """
        加载原始CSV文件
        
        参数：
            league_code: 联赛代码（如E0, SP1）
            season: 赛季（如2025-26）
        """
        filename = f"{league_code}_{season}.csv"
        filepath = os.path.join(RAW_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"警告：文件不存在 {filepath}")
            return
        
        print(f"加载文件: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # 跳过标题行
                if row.get('Date') in ['Date', '']:
                    continue
                
                try:
                    match = self._parse_csv_row(row, league_code, season)
                    if match:
                        self.raw_matches.append(match)
                except Exception as e:
                    print(f"解析行失败: {e}")
    
    def _parse_csv_row(self, row: Dict[str, str], league_code: str, season: str) -> Optional[Dict]:
        """
        解析CSV行数据
        
        参数：
            row: CSV行数据
            league_code: 联赛代码
            season: 赛季
        
        返回：
            解析后的比赛字典，无效返回None
        """
        # 检查必需字段
        date_str = row.get('Date', '')
        home_team = row.get('HomeTeam', '').strip()
        away_team = row.get('AwayTeam', '').strip()
        
        if not date_str or not home_team or not away_team:
            return None
        
        # 解析比分
        try:
            fthg = int(row.get('FTHG', 0))
            ftag = int(row.get('FTAG', 0))
        except:
            return None
        
        # 解析半场比分（可选）
        try:
            hthg = int(row.get('HTHG', 0))
            htag = int(row.get('HTAG', 0))
        except:
            hthg = 0
            htag = 0
        
        # 确定结果
        if fthg > ftag:
            result = 'H'
        elif fthg < ftag:
            result = 'A'
        else:
            result = 'D'
        
        # 解析欧赔
        try:
            avg_h = float(row.get('AvgH', row.get('B365H', 2.5)))
            avg_d = float(row.get('AvgD', row.get('B365D', 3.2)))
            avg_a = float(row.get('AvgA', row.get('B365A', 3.2)))
        except:
            avg_h = 2.5
            avg_d = 3.2
            avg_a = 3.2
        
        # 解析亚盘
        try:
            ah_h = float(row.get('AHh', 0))
            avg_ahh = float(row.get('AvgAHH', row.get('B365AHH', 1.9)))
            avg_aha = float(row.get('AvgAHA', row.get('B365AHA', 1.9)))
        except:
            ah_h = 0.0
            avg_ahh = 1.9
            avg_aha = 1.9
        
        # 解析大小球
        try:
            over_odds = float(row.get('Avg>2.5', row.get('B365>2.5', 1.9)))
            under_odds = float(row.get('Avg<2.5', row.get('B365<2.5', 1.9)))
        except:
            over_odds = 1.9
            under_odds = 1.9
        
        # 检查是否有有效的赔率数据
        has_euro = avg_h > 1 and avg_d > 1 and avg_a > 1
        has_asian = ah_h != 0 or (avg_ahh > 1 and avg_aha > 1)
        has_total = over_odds > 1 and under_odds > 1
        
        return {
            'date': parse_date(date_str),
            'league_code': league_code,
            'league': LEAGUE_MAP.get(league_code, league_code),
            'season': season,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': fthg,
            'away_goals': ftag,
            'result': result,
            'half_home_goals': hthg,
            'half_away_goals': htag,
            # 欧赔
            'euro_avg_h': avg_h,
            'euro_avg_d': avg_d,
            'euro_avg_a': avg_a,
            # 亚盘
            'asian_handicap': ah_h,
            'asian_avg_h': avg_ahh,
            'asian_avg_a': avg_aha,
            # 大小球
            'total_over_odds': over_odds,
            'total_under_odds': under_odds,
            # 缺失标记
            'has_euro_odds': 1 if has_euro else 0,
            'has_asian_odds': 1 if has_asian else 0,
            'has_total_odds': 1 if has_total else 0,
        }
    
    def build(self) -> List[Dict]:
        """
        构建训练数据集
        
        返回：
            训练样本列表
        """
        # 按日期排序
        self.raw_matches.sort(key=lambda x: x['date'])
        
        # 按日期分组处理（同一天的比赛一起处理）
        date_groups = defaultdict(list)
        for match in self.raw_matches:
            date_groups[match['date']].append(match)
        
        training_samples = []
        
        for date, day_matches in sorted(date_groups.items()):
            # 第一步：为当天所有比赛生成赛前特征
            day_features = []
            for match in day_matches:
                features = self._build_features(match)
                day_features.append((match, features))
            
            # 第二步：保存当天所有比赛的训练样本
            for match, features in day_features:
                sample = self._create_sample(match, features)
                training_samples.append(sample)
            
            # 第三步：更新ELO和球队状态（使用当天真实结果）
            for match in day_matches:
                self.elo_calculator.update_rating(
                    match['home_team'],
                    match['away_team'],
                    match['home_goals'],
                    match['away_goals']
                )
                self.stats_tracker.add_match(match)
        
        print(f"共生成 {len(training_samples)} 个训练样本")
        return training_samples
    
    def _build_features(self, match: Dict) -> Dict:
        """
        为单场比赛构建赛前特征
        
        参数：
            match: 比赛数据
        
        返回：
            特征字典
        """
        features = {}
        
        # ELO特征
        elo_home = self.elo_calculator.get_rating(match['home_team'])
        elo_away = self.elo_calculator.get_rating(match['away_team'])
        features['elo_home'] = elo_home
        features['elo_away'] = elo_away
        features['elo_diff'] = elo_home - elo_away
        
        # 主队最近5场状态
        home_stats_5 = self.stats_tracker.get_recent_stats(match['home_team'], n=5)
        features['home_attack_5'] = home_stats_5['avg_goals_for']
        features['home_defense_5'] = home_stats_5['avg_goals_against']
        features['home_form_points_5'] = home_stats_5['avg_points']
        features['home_win_rate_5'] = home_stats_5['win_rate']
        features['home_draw_rate_5'] = home_stats_5['draw_rate']
        features['home_matches_count'] = home_stats_5['matches_count']
        
        # 客队最近5场状态
        away_stats_5 = self.stats_tracker.get_recent_stats(match['away_team'], n=5)
        features['away_attack_5'] = away_stats_5['avg_goals_for']
        features['away_defense_5'] = away_stats_5['avg_goals_against']
        features['away_form_points_5'] = away_stats_5['avg_points']
        features['away_win_rate_5'] = away_stats_5['win_rate']
        features['away_draw_rate_5'] = away_stats_5['draw_rate']
        features['away_matches_count'] = away_stats_5['matches_count']
        
        # 主队最近10场状态
        home_stats_10 = self.stats_tracker.get_recent_stats(match['home_team'], n=10)
        features['home_win_rate_10'] = home_stats_10['win_rate']
        features['home_goals_for_10'] = home_stats_10['avg_goals_for']
        features['home_goals_against_10'] = home_stats_10['avg_goals_against']
        
        # 客队最近10场状态
        away_stats_10 = self.stats_tracker.get_recent_stats(match['away_team'], n=10)
        features['away_win_rate_10'] = away_stats_10['win_rate']
        features['away_goals_for_10'] = away_stats_10['avg_goals_for']
        features['away_goals_against_10'] = away_stats_10['avg_goals_against']
        
        # 主队主场最近5场
        home_h_stats = self.stats_tracker.get_recent_stats(match['home_team'], n=5, is_home=True)
        features['home_h_goals_for_5'] = home_h_stats['avg_goals_for']
        features['home_h_goals_against_5'] = home_h_stats['avg_goals_against']
        
        # 客队客场最近5场
        away_a_stats = self.stats_tracker.get_recent_stats(match['away_team'], n=5, is_home=False)
        features['away_a_goals_for_5'] = away_a_stats['avg_goals_for']
        features['away_a_goals_against_5'] = away_a_stats['avg_goals_against']
        
        # 欧赔特征（去水后概率）
        euro_home_prob, euro_draw_prob, euro_away_prob = odds_to_prob(
            match['euro_avg_h'],
            match['euro_avg_d'],
            match['euro_avg_a']
        )
        features['euro_home_prob'] = euro_home_prob
        features['euro_draw_prob'] = euro_draw_prob
        features['euro_away_prob'] = euro_away_prob
        
        # 亚盘特征
        asian_home_prob, asian_away_prob = asian_odds_to_prob(
            match['asian_handicap'],
            match['asian_avg_h'],
            match['asian_avg_a']
        )
        features['asian_handicap'] = match['asian_handicap']
        features['asian_home_prob'] = asian_home_prob
        features['asian_away_prob'] = asian_away_prob
        
        # 大小球特征
        over_prob, under_prob = total_odds_to_prob(
            match['total_over_odds'],
            match['total_under_odds']
        )
        features['total_line'] = 2.5  # 默认值，实际应该从数据中提取
        features['over_prob'] = over_prob
        features['under_prob'] = under_prob
        
        # 联赛特征
        league_stats = self.stats_tracker.get_league_stats(match['league'], n=100)
        features['league_avg_goals_100'] = league_stats['avg_goals']
        features['league_draw_rate_100'] = league_stats['draw_rate']
        
        # 是否主队热门（根据欧赔判断）
        features['is_home_favorite'] = 1 if euro_home_prob > euro_away_prob else 0
        
        # 缺失标记
        features['has_euro_odds'] = match['has_euro_odds']
        features['has_asian_odds'] = match['has_asian_odds']
        features['has_total_odds'] = match['has_total_odds']
        
        return features
    
    def _create_sample(self, match: Dict, features: Dict) -> Dict:
        """
        创建训练样本
        
        参数：
            match: 比赛数据
            features: 特征字典
        
        返回：
            训练样本字典
        """
        match_id = f"{match['league_code']}_{match['date']}_{match['home_team']}_{match['away_team']}"
        
        return {
            'match_id': match_id,
            'match_date': match['date'],
            'league': match['league'],
            'season': match['season'],
            'features': features,
            'target': {
                'result': match['result'],
                'home_goals': match['home_goals'],
                'away_goals': match['away_goals'],
                'total_goals': match['home_goals'] + match['away_goals'],
            },
            'raw_data': {
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'half_home_goals': match['half_home_goals'],
                'half_away_goals': match['half_away_goals'],
            }
        }
    
    def save_to_jsonl(self, samples: List[Dict], filepath: str = None):
        """
        保存训练数据到JSONL文件
        
        参数：
            samples: 训练样本列表
            filepath: 输出文件路径
        """
        if filepath is None:
            filepath = OUTPUT_FILE
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"训练数据已保存到: {filepath}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    builder = DatasetBuilder()
    
    # 加载所有联赛和赛季的数据
    for league_code in LEAGUE_MAP.keys():
        for season in SEASONS:
            builder.load_raw_csv(league_code, season)
    
    # 构建数据集
    samples = builder.build()
    
    # 保存到文件
    builder.save_to_jsonl(samples)
    
    print("\n数据集构建完成！")
    print(f"总样本数: {len(samples)}")


if __name__ == '__main__':
    main()