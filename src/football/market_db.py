#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
盘口比分频率数据库系统
=======================

功能：
1. 数据采集 - 从Football-Data.co.uk下载联赛数据
2. 数据解析 - 读取CSV文件提取字段
3. 盘口标准化 - 统一亚盘和大小球格式
4. 比分频率统计 - 统计各盘口组合的比分概率
5. 数据库持久化 - JSON文件存储与加载
6. 查询接口 - get_market_score_prob()
7. 融合预测模型 - 泊松+历史概率融合

支持联赛：英超(E0)、西甲(SP1)、德甲(D1)、意甲(I1)、法甲(F1)
时间范围：2015-2026
"""

import os
import re
import json
import csv
import urllib.request
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# ==================== 常量配置 ====================

# 数据来源URL
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# 支持的联赛
LEAGUES = {
    'E0': '英超',
    'SP1': '西甲',
    'D1': '德甲',
    'I1': '意甲',
    'F1': '法甲',
}

# 赛季格式转换
SEASONS = [f"{y}{y+1}" for y in range(15, 27)]  # 1516, 1617, ..., 2627

# 标准亚盘值（主队视角，负数为主队让球）
STANDARD_ASIAN = [
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
    0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
]

# 标准大小球值
STANDARD_OU = [
    1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0
]

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DB_FILE = os.path.join(DATA_DIR, 'market_score_db.json')
CHANGE_DB_FILE = os.path.join(DATA_DIR, 'market_change_db.json')

# 默认融合权重
DEFAULT_WEIGHTS = {
    'poisson': 0.8,
    'market': 0.2,
}


# ==================== 数据采集模块 ====================

def download_league_data(league: str, season: str, save_dir: str = DATA_DIR) -> bool:
    """
    下载指定联赛和赛季的数据
    
    参数：
        league: 联赛代码（E0, SP1, D1, I1, F1）
        season: 赛季（如 '2122' 表示2021-2022赛季）
        save_dir: 保存目录
    
    返回：
        是否下载成功
    """
    if league not in LEAGUES:
        print(f"不支持的联赛: {league}")
        return False
    
    url = BASE_URL.format(season=season, league=league)
    filename = f"{league}_{season}.csv"
    filepath = os.path.join(save_dir, filename)
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        print(f"下载成功: {filepath}")
        return True
    except Exception as e:
        print(f"下载失败 {league}_{season}: {e}")
        return False


def download_all_leagues(start_season: int = 15, end_season: int = 26) -> int:
    """
    下载所有支持联赛的数据
    
    参数：
        start_season: 起始赛季（2015=15）
        end_season: 结束赛季（2026=26）
    
    返回：
        成功下载的文件数量
    """
    count = 0
    for league in LEAGUES:
        for year in range(start_season, end_season + 1):
            season = f"{year}{year+1}"
            if download_league_data(league, season):
                count += 1
    return count


# ==================== 数据解析模块 ====================

def parse_csv_file(filepath: str) -> List[Dict]:
    """
    解析CSV文件，提取所需字段
    
    参数：
        filepath: CSV文件路径
    
    返回：
        比赛记录列表
    """
    records = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = parse_match_row(row)
                if record:
                    records.append(record)
    except Exception as e:
        print(f"解析文件失败 {filepath}: {e}")
    
    return records


def parse_match_row(row: Dict) -> Optional[Dict]:
    """
    解析单行比赛数据
    
    返回字段：
        Date: 日期
        HomeTeam: 主队
        AwayTeam: 客队
        FTHG: 主队进球
        FTAG: 客队进球
        AHh: 亚盘让球
        AvgAHH/AvgAHA: 平均亚盘赔率（主/客）
        Avg>2.5/Avg<2.5: 平均大小球赔率
    """
    try:
        # 必需字段
        date = row.get('Date', '')
        home_team = row.get('HomeTeam', '').strip()
        away_team = row.get('AwayTeam', '').strip()
        
        # 比分
        fthg = int(row['FTHG']) if 'FTHG' in row and row['FTHG'].isdigit() else None
        ftag = int(row['FTAG']) if 'FTAG' in row and row['FTAG'].isdigit() else None
        
        if fthg is None or ftag is None:
            return None
        
        # 亚盘数据
        ahh = parse_handicap_value(row.get('AHh', ''))
        
        # 亚盘赔率（优先使用Avg，其次B365）
        avg_ahh = parse_odds_value(row.get('AvgAHH'))
        avg_aha = parse_odds_value(row.get('AvgAHA'))
        if avg_ahh is None:
            avg_ahh = parse_odds_value(row.get('B365AHH'))
        if avg_aha is None:
            avg_aha = parse_odds_value(row.get('B365AHA'))
        
        # 大小球数据
        avg_over = parse_odds_value(row.get('Avg>2.5'))
        avg_under = parse_odds_value(row.get('Avg<2.5'))
        if avg_over is None:
            avg_over = parse_odds_value(row.get('B365>2.5'))
        if avg_under is None:
            avg_under = parse_odds_value(row.get('B365<2.5'))
        
        # 开盘亚盘（如果有）
        o_ahh = parse_handicap_value(row.get('OAhh', ''))
        
        # 开盘大小球（如果有）
        o_over = parse_odds_value(row.get('O>2.5'))
        o_under = parse_odds_value(row.get('O<2.5'))
        
        return {
            'Date': date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': fthg,
            'FTAG': ftag,
            'AHh': ahh,
            'AvgAHH': avg_ahh,
            'AvgAHA': avg_aha,
            'AvgOver': avg_over,
            'AvgUnder': avg_under,
            'OAhh': o_ahh,
            'OOver': o_over,
            'OUnder': o_under,
        }
    
    except Exception as e:
        print(f"解析行失败: {e}")
        return None


def parse_handicap_value(value: str) -> Optional[float]:
    """解析让球值"""
    if not value or value.strip() == '':
        return None
    try:
        return round(float(value), 2)
    except ValueError:
        return None


def parse_odds_value(value: str) -> Optional[float]:
    """解析赔率值"""
    if not value or value.strip() == '':
        return None
    try:
        return round(float(value), 2)
    except ValueError:
        return None


# ==================== 盘口标准化模块 ====================

def normalize_asian(handicap: float) -> float:
    """
    将亚盘值标准化到标准值列表
    
    参数：
        handicap: 原始让球值
    
    返回：
        标准化后的让球值
    """
    if handicap is None:
        return None
    
    # 找到最近的标准值
    nearest = min(STANDARD_ASIAN, key=lambda x: abs(x - handicap))
    return round(nearest, 2)


def normalize_ou(line: float) -> float:
    """
    将大小球值标准化到标准值列表
    
    参数：
        line: 原始大小球线
    
    返回：
        标准化后的大小球线
    """
    if line is None:
        return None
    
    # 找到最近的标准值
    nearest = min(STANDARD_OU, key=lambda x: abs(x - line))
    return round(nearest, 2)


def normalize_handicap_from_odds(home_odds: float, away_odds: float) -> float:
    """
    从赔率反推让球值（当没有直接让球数据时）
    
    参数：
        home_odds: 主队赔率
        away_odds: 客队赔率
    
    返回：
        反推的让球值
    """
    if home_odds is None or away_odds is None:
        return 0.0
    
    # 简化的让球反推公式
    odds_ratio = away_odds / home_odds
    # 粗略转换：赔率比与让球的关系
    handicap = (1 - odds_ratio) * 0.5
    return normalize_asian(handicap)


# ==================== 比分频率统计模块 ====================

class MarketScoreDB:
    """
    盘口比分频率数据库
    
    结构：
        {
            "asian_ou_key": {
                "score": probability,
                ...
            }
        }
    """
    
    def __init__(self):
        self.db: Dict[str, Dict[str, float]] = {}
        self.sample_counts: Dict[str, int] = {}  # 记录每个盘口组合的样本数
        self._load()
    
    def _load(self):
        """从文件加载数据库（内部方法）"""
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.db = data.get('probabilities', {})
                    self.sample_counts = data.get('sample_counts', {})
                print(f"已加载数据库，{len(self.db)} 个盘口组合")
            except Exception as e:
                print(f"加载数据库失败: {e}")
                self.db = {}
                self.sample_counts = {}
    
    def load(self):
        """从文件加载数据库（公开方法）"""
        self._load()
    
    def save(self):
        """保存数据库到文件"""
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            'probabilities': self.db,
            'sample_counts': self.sample_counts,
        }
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据库已保存，{len(self.db)} 个盘口组合")
    
    def _get_key(self, asian: float, ou: float) -> str:
        """生成盘口组合的唯一键"""
        return f"{asian:.2f}_{ou:.2f}"
    
    def add_record(self, asian: float, ou: float, score: str):
        """
        添加一条比分记录
        
        参数：
            asian: 标准化的亚盘值
            ou: 标准化的大小球值
            score: 比分字符串（如 "2-1"）
        """
        key = self._get_key(asian, ou)
        
        if key not in self.db:
            self.db[key] = {}
            self.sample_counts[key] = 0
        
        # 确保 score 存在
        if score not in self.db[key]:
            self.db[key][score] = 0.0
        
        self.db[key][score] += 1.0
        self.sample_counts[key] += 1
    
    def add_match_result(self, asian: float, ou: float, score: str):
        """
        赛后结算时添加比赛结果（兼容接口）
        
        参数：
            asian: 亚盘让球值
            ou: 大小球线
            score: 比分字符串（如 "2-1"）
        """
        asian = normalize_asian(asian)
        ou = normalize_ou(ou)
        if asian is None or ou is None or not score:
            return
        self.add_record(asian, ou, score)
        self._normalize_all()
    
    def add_records(self, records: List[Dict]):
        """批量添加记录"""
        for record in records:
            asian = record.get('asian')
            ou = record.get('ou')
            score = record.get('score')
            if asian is not None and ou is not None and score:
                self.add_record(asian, ou, score)
    
    def build_from_csv_files(self, csv_dir: str = DATA_DIR):
        """
        从CSV文件批量构建数据库
        
        参数：
            csv_dir: CSV文件所在目录
        """
        count = 0
        for filename in os.listdir(csv_dir):
            if filename.endswith('.csv') and '_' in filename:
                filepath = os.path.join(csv_dir, filename)
                records = parse_csv_file(filepath)
                
                for record in records:
                    # 标准化盘口
                    asian = record.get('AHh')
                    if asian is None:
                        # 尝试从赔率反推
                        asian = normalize_handicap_from_odds(
                            record.get('AvgAHH'), record.get('AvgAHA')
                        )
                    asian = normalize_asian(asian)
                    
                    # 计算大小球（从赔率反推）
                    ou = self._implied_total_from_odds(
                        record.get('AvgOver'), record.get('AvgUnder')
                    )
                    ou = normalize_ou(ou)
                    
                    if asian is not None and ou is not None:
                        score = f"{record['FTHG']}-{record['FTAG']}"
                        self.add_record(asian, ou, score)
                        count += 1
        
        # 归一化所有概率
        self._normalize_all()
        print(f"从CSV构建完成，共 {count} 条记录")
    
    def _implied_total_from_odds(self, over_odds: float, under_odds: float) -> float:
        """从大小球赔率反推期望总进球"""
        if over_odds is None or under_odds is None:
            return 2.5
        
        # 简化计算：赔率比对应的总进球
        try:
            p_over = 1.0 / over_odds
            p_under = 1.0 / under_odds
            total = p_over + p_under
            p_over_normalized = p_over / total
            
            # 从大球概率反推总进球（简化版）
            # P(over 2.5) = 1 - P(0) - P(1) - P(2)
            # 这里用线性近似
            return 2.5 + (p_over_normalized - 0.5) * 2.0
        except:
            return 2.5
    
    def _normalize_all(self):
        """归一化所有盘口组合的概率"""
        for key in self.db:
            total = sum(self.db[key].values())
            if total > 0:
                self.db[key] = {k: v / total for k, v in sorted(
                    self.db[key].items(), key=lambda x: -x[1]
                )}
    
    def get_prob(self, asian: float, ou: float) -> Optional[Dict[str, float]]:
        """
        获取指定盘口组合的比分概率
        
        参数：
            asian: 亚盘让球
            ou: 大小球线
        
        返回：
            比分概率字典，如果没有精确匹配则返回None
        """
        key = self._get_key(normalize_asian(asian), normalize_ou(ou))
        return self.db.get(key)
    
    def get_prob_with_nearest(self, asian: float, ou: float) -> Dict[str, float]:
        """
        获取盘口组合的比分概率，支持模糊匹配
        
        如果精确匹配不存在，自动查找最近邻盘口
        
        参数：
            asian: 亚盘让球
            ou: 大小球线
        
        返回：
            比分概率字典
        """
        target_asian = normalize_asian(asian)
        target_ou = normalize_ou(ou)
        key = self._get_key(target_asian, target_ou)
        
        # 精确匹配
        if key in self.db:
            return self.db[key]
        
        # 查找最近邻
        nearest_key = self._find_nearest_key(target_asian, target_ou)
        if nearest_key:
            print(f"未找到精确匹配 {key}，使用最近邻 {nearest_key}")
            return self.db[nearest_key]
        
        return {}
    
    def _find_nearest_key(self, asian: float, ou: float) -> Optional[str]:
        """查找最近的盘口组合键"""
        min_distance = float('inf')
        nearest = None
        
        for key in self.db:
            parts = key.split('_')
            if len(parts) != 2:
                continue
            try:
                db_asian = float(parts[0])
                db_ou = float(parts[1])
                distance = abs(db_asian - asian) + abs(db_ou - ou)
                if distance < min_distance:
                    min_distance = distance
                    nearest = key
            except ValueError:
                continue
        
        return nearest
    
    def get_sample_count(self, asian: float, ou: float) -> int:
        """获取指定盘口组合的样本数量"""
        key = self._get_key(normalize_asian(asian), normalize_ou(ou))
        return self.sample_counts.get(key, 0)
    
    def get_top_scores(self, asian: float, ou: float, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取指定盘口组合的TOP N比分
        
        参数：
            asian: 亚盘让球
            ou: 大小球线
            top_n: 返回数量
        
        返回：
            排序后的比分概率列表
        """
        prob = self.get_prob_with_nearest(asian, ou)
        if not prob:
            return []
        
        sorted_scores = sorted(prob.items(), key=lambda x: -x[1])
        return sorted_scores[:top_n]
    
    def get_htf_probs(self, asian: float, ou: float) -> Dict[str, float]:
        """
        获取指定盘口组合的半全场概率分布
        
        参数：
            asian: 亚盘让球
            ou: 大小球线
        
        返回：
            半全场概率字典
        """
        prob = self.get_prob_with_nearest(asian, ou)
        if not prob:
            return {}
        
        htf_probs = {}
        
        def get_half_result(h, a):
            if h > a:
                return 'H'
            elif h < a:
                return 'A'
            else:
                return 'D'
        
        def get_full_result(h, a):
            if h > a:
                return 'H'
            elif h < a:
                return 'A'
            else:
                return 'D'
        
        for score, score_prob in prob.items():
            h, a = map(int, score.split('-'))
            # 简单假设半场进球是全场的40%
            half_h = int(h * 0.4)
            half_a = int(a * 0.4)
            
            half_res = get_half_result(half_h, half_a)
            full_res = get_full_result(h, a)
            key = f"{half_res}{full_res}"
            
            if key not in htf_probs:
                htf_probs[key] = 0.0
            htf_probs[key] += score_prob
        
        # 归一化
        total = sum(htf_probs.values())
        if total > 0:
            htf_probs = {k: v / total for k, v in sorted(htf_probs.items(), key=lambda x: -x[1])}
        
        return htf_probs
    
    def get_goal_count_dist(self, asian: float, ou: float) -> Dict[int, float]:
        """
        获取指定盘口组合的进球数分布
        
        参数：
            asian: 亚盘让球
            ou: 大小球线
        
        返回：
            进球数概率字典
        """
        prob = self.get_prob_with_nearest(asian, ou)
        if not prob:
            return {}
        
        goal_dist = {}
        for score, score_prob in prob.items():
            h, a = map(int, score.split('-'))
            total_goals = h + a
            
            if total_goals not in goal_dist:
                goal_dist[total_goals] = 0.0
            goal_dist[total_goals] += score_prob
        
        # 归一化
        total = sum(goal_dist.values())
        if total > 0:
            goal_dist = {k: v / total for k, v in sorted(goal_dist.items())}
        
        return goal_dist
    
    def merge_with(self, other_db: 'MarketScoreDB'):
        """合并另一个数据库"""
        for key, probs in other_db.db.items():
            if key not in self.db:
                self.db[key] = defaultdict(float)
            
            for score, prob in probs.items():
                # 按样本数加权合并
                self.db[key][score] += prob * other_db.sample_counts.get(key, 1)
            
            self.sample_counts[key] = self.sample_counts.get(key, 0) + other_db.sample_counts.get(key, 0)
        
        self._normalize_all()
    
    def clear(self):
        """清空数据库"""
        self.db = {}
        self.sample_counts = {}


# ==================== 盘口变化数据库 ====================

class MarketChangeDB:
    """
    盘口变化数据库（用于升降盘分析）
    
    结构：
        {
            "asian_change_ou_change": {
                "score": count,
                ...
            }
        }
    """
    
    def __init__(self):
        self.db: Dict[str, Dict[str, int]] = {}
        self._load()
    
    def _load(self):
        """从文件加载数据库"""
        if os.path.exists(CHANGE_DB_FILE):
            try:
                with open(CHANGE_DB_FILE, 'r', encoding='utf-8') as f:
                    self.db = json.load(f)
                print(f"已加载盘口变化数据库，{len(self.db)} 个变化组合")
            except Exception as e:
                print(f"加载盘口变化数据库失败: {e}")
                self.db = {}
    
    def save(self):
        """保存数据库到文件"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CHANGE_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, ensure_ascii=False, indent=2)
        print(f"盘口变化数据库已保存")
    
    def _get_key(self, asian_from: float, asian_to: float, ou_from: float, ou_to: float) -> str:
        """生成盘口变化的唯一键"""
        return f"{asian_from:.2f}→{asian_to:.2f}_{ou_from:.2f}→{ou_to:.2f}"
    
    def add_record(self, asian_from: float, asian_to: float, ou_from: float, ou_to: float, score: str):
        """
        添加一条盘口变化记录
        
        参数：
            asian_from: 开盘亚盘
            asian_to: 收盘亚盘
            ou_from: 开盘大小球
            ou_to: 收盘大小球
            score: 最终比分
        """
        key = self._get_key(asian_from, asian_to, ou_from, ou_to)
        
        if key not in self.db:
            self.db[key] = defaultdict(int)
        
        self.db[key][score] += 1
    
    def get_change_stats(self, asian_from: float, asian_to: float, 
                        ou_from: float, ou_to: float) -> Optional[Dict[str, float]]:
        """
        获取盘口变化后的比分统计
        
        返回：
            归一化的比分概率字典
        """
        key = self._get_key(asian_from, asian_to, ou_from, ou_to)
        counts = self.db.get(key)
        
        if not counts:
            return None
        
        total = sum(counts.values())
        return {score: count / total for score, count in sorted(
            counts.items(), key=lambda x: -x[1]
        )}


# ==================== 查询接口 ====================

def get_market_score_prob(asian_handicap: float, over_under: float, 
                         top_n: int = 10) -> Dict[str, Any]:
    """
    查询盘口组合的比分概率
    
    参数：
        asian_handicap: 亚盘让球
        over_under: 大小球线
        top_n: 返回前N个比分
    
    返回：
        包含比分概率、样本数等信息的字典
    """
    db = MarketScoreDB()
    
    # 获取比分概率
    prob = db.get_prob_with_nearest(asian_handicap, over_under)
    top_scores = sorted(prob.items(), key=lambda x: -x[1])[:top_n]
    
    # 获取样本数
    sample_count = db.get_sample_count(asian_handicap, over_under)
    
    return {
        'asian_handicap': normalize_asian(asian_handicap),
        'over_under': normalize_ou(over_under),
        'top_scores': top_scores,
        'sample_count': sample_count,
        'probabilities': prob,
    }


# ==================== 融合预测模型 ====================

def blend_predictions(poisson_probs: Dict[str, float], 
                     market_probs: Dict[str, float],
                     weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    融合泊松预测和历史市场概率
    
    参数：
        poisson_probs: 泊松模型预测的比分概率
        market_probs: 历史市场比分概率
        weights: 权重配置 {'poisson': 0.8, 'market': 0.2}
    
    返回：
        融合后的比分概率
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # 合并所有比分
    all_scores = set(poisson_probs.keys()) | set(market_probs.keys())
    
    blended = {}
    for score in all_scores:
        p_poisson = poisson_probs.get(score, 0.0)
        p_market = market_probs.get(score, 0.0)
        blended[score] = weights['poisson'] * p_poisson + weights['market'] * p_market
    
    # 归一化
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    
    # 排序
    blended = dict(sorted(blended.items(), key=lambda x: -x[1]))
    
    return blended


def predict_with_market(asian_handicap: float, over_under: float,
                       poisson_probs: Dict[str, float],
                       weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    使用历史市场数据增强泊松预测
    
    参数：
        asian_handicap: 亚盘让球
        over_under: 大小球线
        poisson_probs: 泊松模型预测结果
        weights: 融合权重
    
    返回：
        融合后的预测结果
    """
    # 获取历史市场概率
    market_result = get_market_score_prob(asian_handicap, over_under)
    market_probs = market_result.get('probabilities', {})
    
    # 融合预测
    blended_probs = blend_predictions(poisson_probs, market_probs, weights)
    
    # 获取TOP10
    top_10 = list(blended_probs.items())[:10]
    
    return {
        'asian_handicap': market_result['asian_handicap'],
        'over_under': market_result['over_under'],
        'top_scores': top_10,
        'sample_count': market_result['sample_count'],
        'blended_probabilities': blended_probs,
        'poisson_probabilities': poisson_probs,
        'market_probabilities': market_probs,
        'weights': weights or DEFAULT_WEIGHTS,
    }


# ==================== 命令行工具 ====================

def build_database():
    """构建完整的盘口比分数据库"""
    print("="*60)
    print("构建盘口比分频率数据库")
    print("="*60)
    
    db = MarketScoreDB()
    db.build_from_csv_files()
    db.save()
    
    print(f"\n数据库构建完成！")
    print(f"盘口组合数: {len(db.db)}")
    print(f"总样本数: {sum(db.sample_counts.values())}")


def query_database():
    """交互式查询数据库"""
    print("="*60)
    print("盘口比分数据库查询")
    print("="*60)
    
    while True:
        try:
            asian = float(input("请输入亚盘让球（如 -0.75）: "))
            ou = float(input("请输入大小球（如 2.75）: "))
            
            result = get_market_score_prob(asian, ou)
            
            print(f"\n盘口组合: 亚盘{result['asian_handicap']} | 大小球{result['over_under']}")
            print(f"历史样本数: {result['sample_count']}")
            print("\nTOP10 比分概率:")
            for i, (score, prob) in enumerate(result['top_scores'], 1):
                print(f"{i}. {score}  {prob*100:.1f}%")
            
            cont = input("\n继续查询? (y/n): ")
            if cont.lower() != 'y':
                break
        except ValueError:
            print("输入格式错误，请输入数字")
        except KeyboardInterrupt:
            break


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='盘口比分频率数据库系统')
    parser.add_argument('--download', action='store_true', help='下载数据')
    parser.add_argument('--build', action='store_true', help='构建数据库')
    parser.add_argument('--query', action='store_true', help='查询数据库')
    parser.add_argument('--download-all', action='store_true', help='下载所有联赛数据')
    
    args = parser.parse_args()
    
    if args.download_all:
        count = download_all_leagues()
        print(f"下载完成，共 {count} 个文件")
    elif args.download:
        league = input("输入联赛代码 (E0/SP1/D1/I1/F1): ").strip()
        season = input("输入赛季 (如 2122): ").strip()
        download_league_data(league, season)
    elif args.build:
        build_database()
    elif args.query:
        query_database()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
