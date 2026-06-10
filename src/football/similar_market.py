 #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相似盘口数据库系统 - KNN匹配引擎
==================================

功能：
1. 存储历史比赛的完整盘口特征（欧赔、亚盘、大小球）
2. KNN算法查找相似盘口比赛
3. 统计相似比赛的1X2结果分布
4. 提供专业级别的盘口匹配服务

这是职业模型的核心模块，提升比ML更大。
"""

import os
import json
import csv
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# ==================== 常量配置 ====================

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
SIMILAR_DB_FILE = os.path.join(DATA_DIR, 'similar_market_db.json')

# KNN默认参数
DEFAULT_K = 1000  # 默认取最近1000场
MIN_SAMPLE_SIZE = 50  # 最小样本数

# 特征权重（标准化后）
FEATURE_WEIGHTS = {
    'asian': 1.0,      # 亚盘让球
    'asian_odds': 0.8, # 亚盘赔率
    'total': 0.8,      # 大小球
    'total_odds': 0.6, # 大小球赔率
    'euro_home': 1.2,  # 主胜赔率
    'euro_draw': 0.5,  # 平局赔率
    'euro_away': 1.2,  # 客胜赔率
}

# ==================== 数据结构 ====================

class MatchRecord:
    """
    单场比赛记录
    
    特征向量：
    - asian: 亚盘让球（主队视角）
    - asian_odds_home: 亚盘主队赔率
    - asian_odds_away: 亚盘客队赔率
    - total: 大小球线
    - total_over: 大球赔率
    - total_under: 小球赔率
    - euro_home: 主胜赔率
    - euro_draw: 平局赔率
    - euro_away: 客胜赔率
    
    标签：
    - result: 比赛结果 (H/D/A)
    - goals_home: 主队进球
    - goals_away: 客队进球
    """
    
    def __init__(self, data: Dict):
        # 盘口特征
        self.asian = data.get('asian', 0.0)
        self.asian_odds_home = data.get('asian_odds_home', 0.0)
        self.asian_odds_away = data.get('asian_odds_away', 0.0)
        self.total = data.get('total', 2.5)
        self.total_over = data.get('total_over', 0.0)
        self.total_under = data.get('total_under', 0.0)
        self.euro_home = data.get('euro_home', 0.0)
        self.euro_draw = data.get('euro_draw', 0.0)
        self.euro_away = data.get('euro_away', 0.0)
        
        # 比赛结果
        self.result = data.get('result', '')  # H/D/A
        self.goals_home = data.get('goals_home', 0)
        self.goals_away = data.get('goals_away', 0)
        
        # 元数据
        self.date = data.get('date', '')
        self.league = data.get('league', '')
        self.home_team = data.get('home_team', '')
        self.away_team = data.get('away_team', '')
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'asian': self.asian,
            'asian_odds_home': self.asian_odds_home,
            'asian_odds_away': self.asian_odds_away,
            'total': self.total,
            'total_over': self.total_over,
            'total_under': self.total_under,
            'euro_home': self.euro_home,
            'euro_draw': self.euro_draw,
            'euro_away': self.euro_away,
            'result': self.result,
            'goals_home': self.goals_home,
            'goals_away': self.goals_away,
            'date': self.date,
            'league': self.league,
            'home_team': self.home_team,
            'away_team': self.away_team,
        }


# ==================== KNN匹配引擎 ====================

class SimilarMarketDB:
    """
    相似盘口数据库 - KNN匹配引擎
    
    核心功能：
    1. 存储历史比赛的完整盘口特征
    2. KNN算法查找相似盘口
    3. 统计相似比赛的结果分布
    """
    
    def __init__(self):
        self.records: List[MatchRecord] = []
        self._load()
    
    def _load(self):
        """从文件加载数据库"""
        if os.path.exists(SIMILAR_DB_FILE):
            try:
                with open(SIMILAR_DB_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data.get('records', []):
                        self.records.append(MatchRecord(record_data))
                print(f"已加载相似盘口数据库，{len(self.records)} 条记录")
            except Exception as e:
                print(f"加载相似盘口数据库失败: {e}")
                self.records = []
    
    def save(self):
        """保存数据库到文件"""
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            'records': [r.to_dict() for r in self.records],
            'version': '1.0',
            'count': len(self.records),
        }
        with open(SIMILAR_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"相似盘口数据库已保存，{len(self.records)} 条记录")
    
    def add_record(self, record: MatchRecord):
        """添加一条记录"""
        self.records.append(record)
    
    def add_records(self, records: List[MatchRecord]):
        """批量添加记录"""
        self.records.extend(records)
    
    def _extract_features(self, record: MatchRecord) -> List[float]:
        """
        提取特征向量（已标准化）
        
        返回：[asian_norm, asian_odds_norm, total_norm, total_odds_norm, 
               euro_home_norm, euro_draw_norm, euro_away_norm]
        """
        features = []
        
        # 亚盘让球 (-3.0 ~ +3.0) → (-1 ~ +1)
        asian_norm = max(-1.0, min(1.0, record.asian / 3.0))
        features.append(asian_norm * FEATURE_WEIGHTS['asian'])
        
        # 亚盘赔率比值 (转换为概率后标准化)
        if record.asian_odds_home > 0 and record.asian_odds_away > 0:
            p_home = 1.0 / record.asian_odds_home
            p_away = 1.0 / record.asian_odds_away
            total = p_home + p_away
            if total > 0:
                p_home_norm = (p_home / total - 0.5) * 2  # (-1 ~ +1)
                features.append(p_home_norm * FEATURE_WEIGHTS['asian_odds'])
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 大小球 (1.0 ~ 5.0) → (-1 ~ +1)
        total_norm = max(-1.0, min(1.0, (record.total - 3.0) / 2.0))
        features.append(total_norm * FEATURE_WEIGHTS['total'])
        
        # 大小球赔率比值
        if record.total_over > 0 and record.total_under > 0:
            p_over = 1.0 / record.total_over
            p_under = 1.0 / record.total_under
            total = p_over + p_under
            if total > 0:
                p_over_norm = (p_over / total - 0.5) * 2
                features.append(p_over_norm * FEATURE_WEIGHTS['total_odds'])
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 主胜赔率 (转换为概率的对数)
        if record.euro_home > 1.0:
            p_home = 1.0 / record.euro_home
            # 将概率压缩到 (-1 ~ +1) 范围
            home_norm = math.tanh((p_home - 0.5) * 4)
            features.append(home_norm * FEATURE_WEIGHTS['euro_home'])
        else:
            features.append(0.0)
        
        # 平局赔率
        if record.euro_draw > 1.0:
            p_draw = 1.0 / record.euro_draw
            draw_norm = math.tanh((p_draw - 0.33) * 6)
            features.append(draw_norm * FEATURE_WEIGHTS['euro_draw'])
        else:
            features.append(0.0)
        
        # 客胜赔率
        if record.euro_away > 1.0:
            p_away = 1.0 / record.euro_away
            away_norm = math.tanh((p_away - 0.5) * 4)
            features.append(away_norm * FEATURE_WEIGHTS['euro_away'])
        else:
            features.append(0.0)
        
        return features
    
    def _distance(self, record1: MatchRecord, record2: MatchRecord) -> float:
        """
        计算两条记录之间的距离（欧氏距离）
        
        参数：
            record1: 第一条记录
            record2: 第二条记录
        
        返回：
            距离值（越小越相似）
        """
        f1 = self._extract_features(record1)
        f2 = self._extract_features(record2)
        
        if len(f1) != len(f2):
            return float('inf')
        
        distance = 0.0
        for i in range(len(f1)):
            distance += (f1[i] - f2[i]) ** 2
        
        return math.sqrt(distance)
    
    def find_similar(self, query: MatchRecord, k: int = DEFAULT_K) -> List[Tuple[float, MatchRecord]]:
        """
        KNN查找相似比赛
        
        参数：
            query: 查询记录（待预测的比赛）
            k: 返回最近的k条记录
        
        返回：
            排序后的相似记录列表 [(距离, 记录), ...]
        """
        if not self.records:
            return []
        
        # 计算所有距离
        distances = []
        for record in self.records:
            # 跳过结果为空的记录
            if not record.result:
                continue
            dist = self._distance(query, record)
            distances.append((dist, record))
        
        # 按距离排序
        distances.sort(key=lambda x: x[0])
        
        # 返回前k条
        return distances[:k]
    
    def get_similar_stats(self, query: MatchRecord, k: int = DEFAULT_K) -> Dict:
        """
        获取相似比赛的统计结果
        
        参数：
            query: 查询记录
            k: 使用的近邻数量
        
        返回：
            统计结果字典
        """
        similar = self.find_similar(query, k)
        
        if not similar:
            return {
                'count': 0,
                'avg_distance': 0.0,
                'result_dist': {'H': 0, 'D': 0, 'A': 0},
                'probabilities': {'H': 0.0, 'D': 0.0, 'A': 0.0},
                'goals_dist': {},
                'confidence': 0.0,
            }
        
        total = len(similar)
        
        # 统计结果分布
        result_counts = defaultdict(int)
        goals_dist = defaultdict(int)
        total_distance = 0.0
        
        for dist, record in similar:
            result_counts[record.result] += 1
            goals_dist[f"{record.goals_home}-{record.goals_away}"] += 1
            total_distance += dist
        
        # 计算概率
        probabilities = {}
        for res in ['H', 'D', 'A']:
            probabilities[res] = result_counts[res] / total
        
        # 计算置信度（基于距离和样本量）
        avg_distance = total_distance / total
        # 距离越小、样本量越大，置信度越高
        distance_confidence = max(0.0, 1.0 - avg_distance * 2)
        sample_confidence = min(1.0, total / 500)
        confidence = (distance_confidence + sample_confidence) / 2
        
        # 排序比分分布
        sorted_goals = sorted(goals_dist.items(), key=lambda x: -x[1])[:10]
        goals_prob = {k: v / total for k, v in sorted_goals}
        
        return {
            'count': total,
            'avg_distance': round(avg_distance, 4),
            'result_dist': dict(result_counts),
            'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
            'goals_dist': goals_prob,
            'confidence': round(confidence, 4),
        }


# ==================== CSV数据导入 ====================

def parse_football_data_csv(filepath: str, league: str = '') -> List[MatchRecord]:
    """
    解析football-data.co.uk格式的CSV文件
    
    参数：
        filepath: CSV文件路径
        league: 联赛代码
    
    返回：
        MatchRecord列表
    """
    records = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                record = parse_football_data_row(row, league)
                if record:
                    records.append(record)
    
    except Exception as e:
        print(f"解析CSV失败 {filepath}: {e}")
    
    return records


def parse_football_data_row(row: Dict, league: str = '') -> Optional[MatchRecord]:
    """
    解析单行CSV数据
    
    返回：
        MatchRecord或None
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
        
        # 确定结果
        if fthg > ftag:
            result = 'H'
        elif fthg < ftag:
            result = 'A'
        else:
            result = 'D'
        
        # 亚盘数据
        ahh = parse_float(row.get('AHh'))
        avg_ahh = parse_float(row.get('AvgAHH')) or parse_float(row.get('B365AHH'))
        avg_aha = parse_float(row.get('AvgAHA')) or parse_float(row.get('B365AHA'))
        
        # 如果没有亚盘让球，从欧赔反推
        if ahh is None:
            home_odds = parse_float(row.get('AvgH')) or parse_float(row.get('B365H'))
            away_odds = parse_float(row.get('AvgA')) or parse_float(row.get('B365A'))
            if home_odds and away_odds:
                ahh = infer_handicap(home_odds, away_odds)
        
        # 大小球数据
        avg_over = parse_float(row.get('Avg>2.5')) or parse_float(row.get('B365>2.5'))
        avg_under = parse_float(row.get('Avg<2.5')) or parse_float(row.get('B365<2.5'))
        
        # 估算大小球线
        total = estimate_total(avg_over, avg_under)
        
        # 欧赔数据
        euro_home = parse_float(row.get('AvgH')) or parse_float(row.get('B365H'))
        euro_draw = parse_float(row.get('AvgD')) or parse_float(row.get('B365D'))
        euro_away = parse_float(row.get('AvgA')) or parse_float(row.get('B365A'))
        
        # 验证必需数据
        if ahh is None or total is None:
            return None
        
        return MatchRecord({
            'asian': ahh,
            'asian_odds_home': avg_ahh or 0.0,
            'asian_odds_away': avg_aha or 0.0,
            'total': total,
            'total_over': avg_over or 0.0,
            'total_under': avg_under or 0.0,
            'euro_home': euro_home or 0.0,
            'euro_draw': euro_draw or 0.0,
            'euro_away': euro_away or 0.0,
            'result': result,
            'goals_home': fthg,
            'goals_away': ftag,
            'date': date,
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
        })
    
    except Exception as e:
        return None


def parse_float(value: str) -> Optional[float]:
    """解析浮点数值"""
    if not value or value.strip() == '':
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def infer_handicap(home_odds: float, away_odds: float) -> float:
    """从欧赔反推亚盘让球"""
    if home_odds is None or away_odds is None:
        return 0.0
    
    # 简化公式：让球 ≈ (客胜赔率 - 主胜赔率) / (主胜赔率 + 客胜赔率) * 2
    try:
        ratio = away_odds / home_odds
        handicap = (1 - ratio) * 0.8
        # 限制范围
        return max(-2.0, min(2.0, round(handicap * 4) / 4))  # 标准化到0.25间隔
    except:
        return 0.0


def estimate_total(over_odds: float, under_odds: float) -> float:
    """从大小球赔率估算大小球线"""
    if over_odds is None or under_odds is None:
        return 2.5
    
    try:
        p_over = 1.0 / over_odds
        p_under = 1.0 / under_odds
        total_prob = p_over + p_under
        
        if total_prob <= 0:
            return 2.5
        
        p_over_norm = p_over / total_prob
        
        # 从大球概率反推总进球线
        # P(over 2.5) ≈ 0.5 对应 2.5球
        # P(over 2.5) ≈ 0.7 对应 3.0球
        # P(over 2.5) ≈ 0.3 对应 2.0球
        total_line = 2.5 + (p_over_norm - 0.5) * 3.0
        
        # 标准化到标准大小球线
        standard_lines = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
        return min(standard_lines, key=lambda x: abs(x - total_line))
    
    except:
        return 2.5


# ==================== 批量构建工具 ====================

def build_from_csv_directory(db: SimilarMarketDB, csv_dir: str = DATA_DIR):
    """
    从CSV目录批量构建数据库
    
    参数：
        db: SimilarMarketDB实例
        csv_dir: CSV文件目录
    """
    import re
    
    count = 0
    league_pattern = re.compile(r'^([A-Za-z0-9]+)_(\d{4})\.csv$')
    
    for filename in os.listdir(csv_dir):
        match = league_pattern.match(filename)
        if match:
            league = match.group(1)
            filepath = os.path.join(csv_dir, filename)
            print(f"处理文件: {filename}")
            
            records = parse_football_data_csv(filepath, league)
            db.add_records(records)
            count += len(records)
    
    print(f"批量导入完成，共 {count} 条记录")


# ==================== 查询接口 ====================

def similar_market_match(asian: float, total: float, euro_home: float, 
                         euro_draw: float, euro_away: float, 
                         asian_odds_home: float = 0.0, 
                         asian_odds_away: float = 0.0,
                         total_over: float = 0.0,
                         total_under: float = 0.0,
                         k: int = DEFAULT_K) -> Dict:
    """
    相似盘口匹配主接口
    
    参数：
        asian: 亚盘让球（主队视角，负数为主队让球）
        total: 大小球线
        euro_home: 主胜赔率
        euro_draw: 平局赔率
        euro_away: 客胜赔率
        asian_odds_home: 亚盘主队赔率（可选）
        asian_odds_away: 亚盘客队赔率（可选）
        total_over: 大球赔率（可选）
        total_under: 小球赔率（可选）
        k: 近邻数量（默认1000）
    
    返回：
        统计结果字典：
        {
            'count': 匹配数量,
            'avg_distance': 平均距离,
            'result_dist': {'H': 数量, 'D': 数量, 'A': 数量},
            'probabilities': {'H': 概率, 'D': 概率, 'A': 概率},
            'goals_dist': {比分: 概率},
            'confidence': 置信度(0~1),
        }
    """
    # 创建查询记录
    query = MatchRecord({
        'asian': asian,
        'asian_odds_home': asian_odds_home,
        'asian_odds_away': asian_odds_away,
        'total': total,
        'total_over': total_over,
        'total_under': total_under,
        'euro_home': euro_home,
        'euro_draw': euro_draw,
        'euro_away': euro_away,
    })
    
    # 加载数据库并查询
    db = SimilarMarketDB()
    
    if not db.records:
        print("警告：相似盘口数据库为空，请先构建数据库")
        return {
            'count': 0,
            'avg_distance': 0.0,
            'result_dist': {'H': 0, 'D': 0, 'A': 0},
            'probabilities': {'H': 0.333, 'D': 0.333, 'A': 0.334},
            'goals_dist': {},
            'confidence': 0.0,
        }
    
    return db.get_similar_stats(query, k)


# ==================== 命令行工具 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='相似盘口数据库工具')
    parser.add_argument('--build', action='store_true', help='从CSV构建数据库')
    parser.add_argument('--query', action='store_true', help='查询相似盘口')
    parser.add_argument('--asian', type=float, default=-0.75, help='亚盘让球')
    parser.add_argument('--total', type=float, default=2.75, help='大小球线')
    parser.add_argument('--home', type=float, default=1.72, help='主胜赔率')
    parser.add_argument('--draw', type=float, default=3.40, help='平局赔率')
    parser.add_argument('--away', type=float, default=4.50, help='客胜赔率')
    parser.add_argument('--k', type=int, default=1000, help='近邻数量')
    
    args = parser.parse_args()
    
    if args.build:
        db = SimilarMarketDB()
        build_from_csv_directory(db)
        db.save()
    
    elif args.query:
        result = similar_market_match(
            asian=args.asian,
            total=args.total,
            euro_home=args.home,
            euro_draw=args.draw,
            euro_away=args.away,
            k=args.k
        )
        
        print(f"\n=== 相似盘口匹配结果 ===")
        print(f"查询条件: 亚盘 {args.asian:+0.2f} | 大小球 {args.total}")
        print(f"欧赔: 主胜 {args.home} | 平局 {args.draw} | 客胜 {args.away}")
        print(f"\n匹配数量: {result['count']} 场")
        print(f"平均距离: {result['avg_distance']:.4f}")
        print(f"置信度: {result['confidence']:.2%}")
        
        print(f"\n=== 结果分布 ===")
        print(f"主胜: {result['result_dist']['H']} 场 ({result['probabilities']['H']:.2%})")
        print(f"平局: {result['result_dist']['D']} 场 ({result['probabilities']['D']:.2%})")
        print(f"客胜: {result['result_dist']['A']} 场 ({result['probabilities']['A']:.2%})")
        
        if result['goals_dist']:
            print(f"\n=== 热门比分 ===")
            for score, prob in list(result['goals_dist'].items())[:5]:
                print(f"  {score}: {prob:.2%}")


if __name__ == '__main__':
    main()
