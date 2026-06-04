#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELO 评分系统演示脚本
展示如何使用 ELO 系统进行球队实力评估和比赛结果更新
"""

from elo import get_elo_system, elo_to_goals_expected, elo_to_strength_factor


def demo():
    print("="*60)
    print("ELO 评分系统演示")
    print("="*60)
    
    # 获取 ELO 系统实例
    elo = get_elo_system()
    
    # 1. 初始状态
    print("\n1. 初始状态")
    print(f"已记录球队数: {elo.get_team_count()}")
    
    # 2. 更新比赛结果
    print("\n2. 更新比赛结果")
    matches = [
        ('曼联', '利物浦', 2, 1, '英超'),
        ('利物浦', '阿森纳', 1, 1, '英超'),
        ('阿森纳', '曼联', 0, 3, '英超'),
        ('切尔西', '曼联', 1, 2, '英超'),
        ('利物浦', '切尔西', 3, 1, '英超'),
    ]
    
    for home, away, h_score, a_score, league in matches:
        elo.update_ratings(home, away, h_score, a_score, league)
        print(f"  {home} {h_score}-{a_score} {away}")
    
    # 3. 获取球队评分
    print("\n3. 球队 ELO 评分")
    teams = ['曼联', '利物浦', '阿森纳', '切尔西']
    for team in teams:
        rating = elo.get_rating(team)
        strength = elo_to_strength_factor(rating)
        print(f"  {team}: {rating:.2f} (实力因子: {strength:.2f})")
    
    # 4. 预测比赛
    print("\n4. 比赛预测")
    prediction = elo.predict_match('曼联', '利物浦', '英超')
    print(f"  曼联 vs 利物浦")
    print(f"    主胜概率: {prediction['home']:.2%}")
    print(f"    平局概率: {prediction['draw']:.2%}")
    print(f"    客胜概率: {prediction['away']:.2%}")
    print(f"    ELO差距: {prediction['rating_diff']:.2f}")
    
    # 5. 进球期望值
    print("\n5. 进球期望值 (xG)")
    xg_home = elo_to_goals_expected(prediction['home_rating'], prediction['away_rating'])
    xg_away = elo_to_goals_expected(prediction['away_rating'], prediction['home_rating'])
    print(f"  曼联 xG: {xg_home:.2f}")
    print(f"  利物浦 xG: {xg_away:.2f}")
    
    # 6. ELO 排名
    print("\n6. ELO 排名")
    top_teams = elo.get_top_teams(5)
    for i, item in enumerate(top_teams):
        print(f"  {i+1}. {item['team']}: {item['rating']}")
    
    # 7. 球队历史记录
    print("\n7. 曼联历史记录 (最近5条)")
    history = elo.get_team_history('曼联', 5)
    for record in history:
        change = record.get('change', 0)
        change_str = f" ({'+' if change > 0 else ''}{change:.2f})" if change else ""
        print(f"  {record['date'][:10]}: {record['rating']:.2f}{change_str} - {record['event']}")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == '__main__':
    demo()