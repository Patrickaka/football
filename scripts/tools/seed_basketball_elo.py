#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
篮球 ELO 预填充工具
==================
从公开数据源获取 NBA/CBA/WNBA 近期比赛结果，批量初始化 ELO 评分系统。

数据源优先级：
1. balldontlie.io (免费NBA API)
2. 备用: 直接从 okooo/500 历史页抓取

使用方法:
    python seed_basketball_elo.py              # 默认获取最近一赛季
    python seed_basketball_elo.py --seasons 2   # 获取最近2赛季
    python seed_basketball_elo.py --league CBA  # 仅CBA
"""

import argparse
import json
import math
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

sys.path.insert(0, '.')
from src.domain.sports.basketball.elo import (
    BasketballELORatingSystem, HOME_ADVANTAGE, INITIAL_ELO, _sanitize_team_name,
)
from src.domain.sports.basketball.elo_store import EloStore
from src.foundation.store import Database, make_engine, mysql_url_from_env

_elo_system = None


def get_elo_system():
    """本工具进程内的 Elo 单例。

    评分与历史落在 foundation/store，与线上服务读的是同一张表——这正是
    预填充的意义：跑完之后线上立刻就能用上这批评分。
    """
    global _elo_system
    if _elo_system is None:
        db = Database(make_engine(mysql_url_from_env()))
        _elo_system = BasketballELORatingSystem(store=EloStore(db))
    return _elo_system


# ==================== 数据源 ====================

BALLDONTLIE_GAMES_URL = "https://api.balldontlie.io/v1/games"
BALLDONTLITE_API_KEY = ""  # 免费层不需要 key，但有限制

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}


def fetch_nba_games_season(season_year: int) -> List[Dict]:
    """
    从 balldontlie.io 获取指定赛季的 NBA 比赛结果。
    
    注意：免费版有速率限制 (60次/分钟)，需要分页。
    """
    all_games = []
    
    # API 参数
    params = {
        'season': season_year,       # 如 2024 表示 2024-25 赛季
        'per_page': 100,
        'page': 0,
    }
    
    while True:
        query = '&'.join(f'{k}={v}' for k, v in params.items())
        url = f"{BALLDONTLIE_GAMES_URL}?{query}"
        
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode('utf-8'))
            
            games = data.get('data', [])
            if not games:
                break
                
            all_games.extend(games)
            
            meta = data.get('meta', {})
            total = meta.get('total_count', 0)
            current = len(all_games)
            
            print(f"  已获取 {current}/{total} 场...")
            
            if current >= total:
                break
                
            params['page'] += 1
            time.sleep(0.6)  # 尊重速率限制
            
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("  速率限制，等待 60 秒后重试...")
                time.sleep(60)
                continue
            print(f"  HTTP 错误 {e.code}: {e.reason}")
            break
        except Exception as e:
            print(f"  获取失败: {e}")
            break
    
    return all_games


def parse_balldontlie_game(game: Dict) -> Dict:
    """解析单场比赛为标准格式"""
    home_team = game.get('home_team', {})
    away_team = game.get('visitor_team', {})
    
    home_name = _sanitize_team_name(home_team.get('full_name') or home_team.get('name', ''))
    away_name = _sanitize_team_name(away_team.get('full_name') or away_team.get('name', ''))
    
    if not home_name or not away_name:
        return None
    
    home_score = int(game.get('home_team_score', 0) or 0)
    away_score = int(game.get('visitor_team_score', 0) or 0)
    
    date_str = game.get('date', '')[:10] if game.get('date') else ''
    
    return {
        'home': home_name,
        'away': away_name,
        'home_score': home_score,
        'away_score': away_score,
        'date': date_str,
        'league': 'NBA',
        'status': game.get('status', ''),
        'raw': game,
    }


def seed_elo_from_games(games: List[Dict], verbose: bool = True) -> Dict:
    """
    用比赛列表批量更新 ELO 系统。
    
    返回统计信息。
    """
    elo = get_elo_system()
    
    stats = {
        'total_games': 0,
        'updated_games': 0,
        'skipped_games': 0,
        'teams_before': len(elo.ratings),
        'new_teams': set(),
        'errors': [],
    }
    
    for i, game in enumerate(games):
        try:
            home = game.get('home', '')
            away = game.get('away', '')
            hs = game.get('home_score', 0)
            as_ = game.get('away_score', 0)
            league = game.get('league', 'NBA')
            
            if not home or not away or hs == 0 or as_ == 0:
                stats['skipped_games'] += 1
                continue
            
            # 记录新球队
            if home not in elo.ratings:
                stats['new_teams'].add(home)
            if away not in elo.ratings:
                stats['new_teams'].add(away)
            
            # 更新 ELO
            elo.update_ratings(home, away, hs, as_, league)
            stats['updated_games'] += 1
            stats['total_games'] += 1
            
            # 每100场打印进度
            if verbose and (i + 1) % 100 == 0:
                print(f"  进度: {i + 1}/{len(games)} 场已处理")
                
        except Exception as e:
            stats['errors'].append(str(e))
            stats['skipped_games'] += 1
    
    stats['teams_after'] = len(elo.ratings)
    
    return stats


def print_elo_summary(stats: Dict):
    """打印预填充后的 ELO 统计"""
    elo = get_elo_system()
    sorted_teams = sorted(elo.ratings.items(), key=lambda x: -x[1])
    
    print("\n" + "=" * 50)
    print("ELO 预填充完成！")
    print("=" * 50)
    print(f"  处理比赛数: {stats['updated_games']} (跳过 {stats['skipped_games']})")
    print(f"  新增球队:   {len(stats['new_teams'])}")
    print(f"  总球队数:   {stats['teams_before']} → {stats['teams_after']}")
    
    if sorted_teams:
        vals = [r for _, r in sorted_teams]
        print(f"  ELO 范围:   {sorted_teams[-1][1]:.1f} ~ {sorted_teams[0][1]:.1f}")
        print(f"  ELO 均值:   {sum(vals)/len(vals):.1f} ± {math.sqrt(sum(v*v for v in vals)/len(vals) - (sum(vals)/len(vals))**2):.1f}")
    
    print("\nTop 15 球队:")
    for team, rating in sorted_teams[:15]:
        games = elo.games_played(team)
        print(f"  {rating:7.1f}  {team:<25s} ({games}场)")
    
    print("\nBottom 5 球队:")
    for team, rating in sorted_teams[-5:]:
        games = elo.games_played(team)
        print(f"  {rating:7.1f}  {team:<25s} ({games}场)")


def main():
    parser = argparse.ArgumentParser(description='篮球 ELO 预填充工具')
    parser.add_argument('--seasons', type=int, default=1, help='获取最近N个完整赛季 (默认1)')
    parser.add_argument('--league', type=str, default='NBA', help='联赛类型 (默认NBA)')
    parser.add_argument('--dry-run', action='store_true', help='只获取不写入')
    parser.add_argument('--skip-fetch', action='store_true', help='跳过抓取，仅用已有ELO数据做统计')
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"  篮球 ELO 预填充 - v3.3")
    print(f"{'='*50}")
    
    if args.skip_fetch:
        print("\n跳过数据抓取，显示当前 ELO 状态:")
        elo = get_elo_system()
        stats = {'teams_after': len(elo.ratings), 'updated_games': 0, 'skipped_games': 0}
        print_elo_summary({'updated_games': 0, 'skipped_games': 0, 
                           'teams_before': len(elo.ratings), 
                           'new_teams': set(), 'teams_after': len(elo.ratings), 'errors': []})
        return
    
    current_year = datetime.now().year
    target_seasons = []
    
    for i in range(args.seasons):
        # NBA 赛季标识: 2024 = 2024-25 赛季
        season = current_year - 1 - i
        target_seasons.append(season)
    
    print(f"\n目标联赛: {args.league}")
    print(f"目标赛季: {target_seasons}")
    
    all_games = []
    total_fetched = 0
    
    for season in target_seasons:
        print(f"\n--- 获取 {season}-{season+1} 赛季 ---")
        
        if args.league.upper() in ('NBA',):
            games = fetch_nba_games_season(season)
        else:
            print(f"  暂不支持 {args.league} 数据自动获取")
            continue
        
        parsed = []
        for g in games:
            pg = parse_balldontlie_game(g)
            if pg:
                parsed.append(pg)
        
        print(f"  有效比赛: {len(parsed)}/{len(games)}")
        all_games.extend(parsed)
        total_fetched += len(parsed)
    
    print(f"\n总共获取: {total_fetched} 场有效比赛")
    
    if not all_games:
        print("没有获取到任何比赛数据，退出。")
        return
    
    if args.dry_run:
        print("(dry-run模式，不写入ELO)")
        print(f"\n示例比赛:")
        for g in all_games[:5]:
            print(f"  {g['date']} {g['home']} {g['home_score']} - {g['away_score']} {g['away']}")
        return
    
    # 按日期排序（确保历史顺序）
    all_games.sort(key=lambda x: x.get('date', ''))
    
    print(f"\n开始批量更新 ELO...")
    stats = seed_elo_from_games(all_games)
    
    if stats.get('errors'):
        print(f"\n⚠️ {len(stats['errors'])} 个错误（前3个）:")
        for e in stats['errors'][:3]:
            print(f"  - {e}")
    
    print_elo_summary(stats)


if __name__ == '__main__':
    main()
