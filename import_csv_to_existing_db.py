"""
将CSV数据导入现有数据库（结合项目已有表结构）

导入目标：
1. similar_market - 相似盘口样本库（已有表）
2. matches - 比赛历史表（新增，存储完整比赛数据）

使用方法：
python import_csv_to_existing_db.py
"""
import csv
import glob
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from common.db import get_connection, execute


# 联赛代码映射
LEAGUE_MAP = {
    'E0': 'Premier League',
    'D1': 'Bundesliga',
    'I1': 'Serie A',
    'F1': 'Ligue 1',
    'SP1': 'La Liga',
}


def parse_date(date_str):
    """解析日期格式 DD/MM/YYYY"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), '%d/%m/%Y').strftime('%Y-%m-%d')
    except ValueError:
        return None


def parse_time(time_str):
    """解析时间格式 HH:MM"""
    if not time_str:
        return None
    try:
        return datetime.strptime(time_str.strip(), '%H:%M').strftime('%H:%M')
    except ValueError:
        return None


def safe_float(val):
    """安全转换为浮点数"""
    if not val or val == '':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    """安全转换为整数"""
    if not val or val == '':
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def create_matches_table(cursor):
    """创建matches表（存储完整比赛数据）"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id              BIGINT AUTO_INCREMENT PRIMARY KEY,
            match_id        VARCHAR(256) UNIQUE NOT NULL,
            league          VARCHAR(64),
            league_code     VARCHAR(8),
            match_date      DATE,
            match_time      TIME,
            home_team       VARCHAR(128),
            away_team       VARCHAR(128),
            fthg            INT,
            ftag            INT,
            ftr             VARCHAR(1),
            hthg            INT,
            htag            INT,
            htr             VARCHAR(1),
            odds            JSON,
            stats           JSON,
            settled         TINYINT(1) DEFAULT 0,
            created_at      VARCHAR(64),
            updated_at      VARCHAR(64),
            INDEX idx_league (league),
            INDEX idx_date (match_date),
            INDEX idx_settled (settled),
            INDEX idx_home_team (home_team),
            INDEX idx_away_team (away_team)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)


def import_to_similar_market(row, cursor):
    """导入数据到similar_market表（已有表）"""
    try:
        # 提取盘口数据
        asian = safe_float(row.get('AHh'))
        total = safe_float(row.get('B365>2.5'))  # 用B365的大小球
        
        if asian is None and total is None:
            return None  # 跳过没有盘口数据的记录
        
        # 亚盘赔率
        asian_odds_home = safe_float(row.get('B365AHH'))
        asian_odds_away = safe_float(row.get('B365AHA'))
        
        # 大小球
        total_over = safe_float(row.get('B365>2.5'))
        total_under = safe_float(row.get('B365<2.5'))
        
        # 欧赔
        euro_home = safe_float(row.get('B365H'))
        euro_draw = safe_float(row.get('B365D'))
        euro_away = safe_float(row.get('B365A'))
        
        # 结果和比分
        ftr = row.get('FTR', '')
        goals_home = safe_int(row.get('FTHG'))
        goals_away = safe_int(row.get('FTAG'))
        
        # 日期和联赛
        match_date = parse_date(row.get('Date', ''))
        league_code = row.get('Div', '')
        league = LEAGUE_MAP.get(league_code, league_code)
        
        # 球队
        home_team = row.get('HomeTeam', '').strip()
        away_team = row.get('AwayTeam', '').strip()
        
        # 插入数据
        cursor.execute("""
            INSERT INTO similar_market (
                asian, asian_odds_home, asian_odds_away,
                total, total_over, total_under,
                euro_home, euro_draw, euro_away,
                result, goals_home, goals_away,
                date, league, home_team, away_team
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            asian, asian_odds_home, asian_odds_away,
            total, total_over, total_under,
            euro_home, euro_draw, euro_away,
            ftr, goals_home, goals_away,
            match_date, league, home_team, away_team
        ))
        
        return 1
        
    except Exception as e:
        return None


def import_to_matches(row, cursor):
    """导入数据到matches表（完整比赛数据）"""
    try:
        # 基础信息
        div = row.get('Div', '')
        league = LEAGUE_MAP.get(div, div)
        match_date = parse_date(row.get('Date', ''))
        match_time = parse_time(row.get('Time', ''))
        home_team = row.get('HomeTeam', '').strip()
        away_team = row.get('AwayTeam', '').strip()
        
        # 比分
        fthg = safe_int(row.get('FTHG'))
        ftag = safe_int(row.get('FTAG'))
        ftr = row.get('FTR', '')
        
        # 半场
        hthg = safe_int(row.get('HTHG'))
        htag = safe_int(row.get('HTAG'))
        htr = row.get('HTR', '')
        
        # 生成match_id
        match_id = f"{div}_{match_date}_{home_team}_{away_team}".replace(' ', '_')
        
        # 构建赔率数据
        odds = {}
        
        # 欧赔
        for prefix in ['B365', 'BW', 'PS', 'WH', '1XB']:
            for outcome in ['H', 'D', 'A']:
                key = f"{prefix}{outcome}"
                if row.get(key):
                    odds[key] = safe_float(row.get(key))
        
        # 大小球
        for prefix in ['B365', 'Max', 'Avg', 'P']:
            over_key = f"{prefix}>2.5"
            under_key = f"{prefix}<2.5"
            if row.get(over_key):
                odds[over_key] = safe_float(row.get(over_key))
                odds[under_key] = safe_float(row.get(under_key))
        
        # 亚盘
        ahh = safe_float(row.get('AHh'))
        if ahh is not None:
            odds['asian_handicap'] = ahh
            for prefix in ['B365', 'PA', 'Max', 'Avg']:
                odds[f'{prefix}AHH'] = safe_float(row.get(f'{prefix}AHH'))
                odds[f'{prefix}AHA'] = safe_float(row.get(f'{prefix}AHA'))
        
        # 统计数据
        stats = {
            'referee': row.get('Referee', ''),
            'hs': safe_int(row.get('HS')),
            'as': safe_int(row.get('AS')),
            'hst': safe_int(row.get('HST')),
            'ast': safe_int(row.get('AST')),
            'hf': safe_int(row.get('HF')),
            'af': safe_int(row.get('AF')),
            'hc': safe_int(row.get('HC')),
            'ac': safe_int(row.get('AC')),
            'hy': safe_int(row.get('HY')),
            'ay': safe_int(row.get('AY')),
            'hr': safe_int(row.get('HR')),
            'ar': safe_int(row.get('AR')),
        }
        
        # 判断是否已结算
        settled = 1 if fthg is not None and ftag is not None else 0
        
        now = datetime.now().isoformat()
        
        # 插入或更新
        cursor.execute("""
            INSERT INTO matches (
                match_id, league, league_code,
                match_date, match_time,
                home_team, away_team,
                fthg, ftag, ftr,
                hthg, htag, htr,
                odds, stats, settled,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                settled = VALUES(settled),
                odds = VALUES(odds),
                stats = VALUES(stats),
                updated_at = VALUES(updated_at)
        """, (
            match_id, league, div,
            match_date, match_time,
            home_team, away_team,
            fthg, ftag, ftr,
            hthg, htag, htr,
            json.dumps(odds), json.dumps(stats), settled,
            now, now
        ))
        
        return 1
        
    except Exception as e:
        return None


def import_csv_file(csv_file, cursor):
    """导入单个CSV文件"""
    filename = os.path.basename(csv_file)
    print(f"导入文件: {filename}")
    
    matches_count = 0
    market_count = 0
    errors = 0
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # 导入到matches表
            if import_to_matches(row, cursor):
                matches_count += 1
            
            # 导入到similar_market表
            if import_to_similar_market(row, cursor):
                market_count += 1
    
    return matches_count, market_count, errors


def main():
    """主函数"""
    print("=" * 60)
    print("将CSV数据导入现有数据库结构")
    print("=" * 60)
    
    # 获取数据库连接
    try:
        conn = get_connection()
        cursor = conn.cursor()
        print("✅ 数据库连接成功")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print("请检查环境变量设置")
        return
    
    # 创建matches表（如果不存在）
    print("\n创建/更新表结构...")
    create_matches_table(cursor)
    conn.commit()
    print("✅ 表结构准备完成")
    
    # CSV文件路径
    data_dir = Path(__file__).parent / 'data'
    
    # 支持的联赛
    leagues = ['E0', 'D1', 'I1', 'F1', 'SP1']
    
    total_matches = 0
    total_market = 0
    
    print("\n开始导入数据...")
    print("-" * 60)
    
    for league in leagues:
        pattern = f"{league}_*.csv"
        csv_files = sorted(glob.glob(str(data_dir / pattern)))
        
        if not csv_files:
            print(f"\n联赛 {league}: 无CSV文件")
            continue
        
        print(f"\n联赛 {league} ({LEAGUE_MAP.get(league, league)}):")
        
        for csv_file in csv_files:
            matches_count, market_count, errors = import_csv_file(csv_file, cursor)
            total_matches += matches_count
            total_market += market_count
            if matches_count > 0:
                print(f"  ✅ matches: {matches_count} 条")
                print(f"  ✅ similar_market: {market_count} 条")
    
    # 提交事务
    conn.commit()
    
    print("\n" + "=" * 60)
    print("导入完成!")
    print(f"✅ 导入 matches 表: {total_matches} 条")
    print(f"✅ 导入 similar_market 表: {total_market} 条")
    print("=" * 60)
    
    # 统计
    cursor.execute("SELECT COUNT(*) FROM matches")
    matches_total = cursor.fetchone()['COUNT(*)']
    
    cursor.execute("SELECT COUNT(*) FROM similar_market")
    market_total = cursor.fetchone()['COUNT(*)']
    
    print("\n数据库统计:")
    print(f"  matches 表: {matches_total} 条记录")
    print(f"  similar_market 表: {market_total} 条记录")
    
    # 关闭连接
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
