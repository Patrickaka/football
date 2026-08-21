# -*- coding: utf-8 -*-
"""足球命令行入口"""

import sys
import os
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('football')

from .config import (
    AVG_LEAGUE_GOAL,
)
from .fetching import (
    fetch_match_list, search_match,
)
from .pipeline import (
    analyze_match,
)

def main():
    print("=" * 65)
    print("  足球比分预测脚本 - 动态赔率分析")
    print("  数据来源: odds.500.com（多家博彩公司平均值）")
    print("=" * 65)

    # ── 输入球队名称 ──
    home_kw = input("请输入主队名称（支持关键词）: ").strip()
    away_kw = input("请输入客队名称（支持关键词）: ").strip()

    if not home_kw and not away_kw:
        print("  ⚠ 未输入任何球队名，退出。")
        return

    # ── 抓取比赛列表 ──
    try:
        matches = fetch_match_list()
    except Exception as e:
        print(f"\n  ✗ 获取比赛列表失败: {e}")
        print("  请检查网络连接后重试。")
        return

    if not matches:
        print("\n  ✗ 未找到任何比赛数据，请稍后重试。")
        return

    print(f"  共找到 {len(matches)} 场比赛")

    # ── 搜索匹配比赛 ──
    found = search_match(matches, home_kw, away_kw)

    if not found:
        print(f"\n  ✗ 未找到匹配 '{home_kw} vs {away_kw}' 的比赛。")
        print(f"\n  今日比赛列表：")
        for i, m in enumerate(matches, 1):
            league_str = f"[{m.get('league', '?')}]" if m.get('league') else ''
            time_str = f" {m.get('time', '')}" if m.get('time') else ''
            print(f"  {i:2d}. {league_str} {m['home']} vs {m['away']}{time_str}  (ID:{m['match_id']})")
        return

    if len(found) == 1:
        match = found[0]
    else:
        print(f"\n  找到 {len(found)} 场匹配比赛：")
        for i, m in enumerate(found, 1):
            league_str = f"[{m.get('league', '?')}]" if m.get('league') else ''
            print(f"  {i}. {league_str} {m['home']} vs {m['away']}  (ID:{m['match_id']})")
        try:
            choice = int(input(f"  请选择 (1-{len(found)}): ").strip())
            match = found[choice - 1]
        except (ValueError, IndexError):
            print("  输入无效，默认选择第1场")
            match = found[0]

    print(f"\n  已选择: {match['home']} vs {match['away']} (ID:{match['match_id']})")
    print("\n  正在抓取赔率数据...")

    try:
        result = analyze_match(match)
    except Exception as e:
        print(f"  ✗ 赔率数据获取失败: {e}")
        return

    render_cli(result)


def _heat_tag(heat):
    return {'cold': '❄冷', 'hot': '🔥热', 'neutral': '—'}.get(heat, '—')


def render_cli(result):
    """将 analyze_match 的结果渲染为命令行报告"""
    match, asian, euro, total, model = (
        result['match'], result['asian'], result['euro'], result['total'], result['model']
    )
    team = result.get('team')
    confidence = result.get('confidence')
    lp = result.get('league_profile') or {}
    home, away = match['home'], match['away']

    print("\n" + "=" * 65)
    print(f"  赔率分析 | {home} vs {away}")
    if lp.get('name') and lp['name'] != 'default':
        print(f"  联赛模型: {lp['name']}（场均进球基准 {lp.get('avg_goal', AVG_LEAGUE_GOAL):.2f}）")
    
    # 显示模型配置信息
    model_info = []
    if model.get('model_type'):
        model_type_name = {
            'poisson': '泊松分布',
            'negative_binomial': '负二项分布',
            'bayesian': '贝叶斯推断',
            'ensemble': '多模型集成'
        }.get(model['model_type'], model['model_type'])
        model_info.append(f"模型: {model_type_name}")
    
    if model.get('calibrated'):
        calib_method_name = {
            'platt': 'Platt缩放',
            'isotonic': '等渗回归'
        }.get(model.get('calibration_method'), model.get('calibration_method'))
        model_info.append(f"校准: {calib_method_name}")
    
    if model.get('ensemble_size'):
        model_info.append(f"集成规模: {model['ensemble_size']}模型")
    
    if model_info:
        print(f"  模型配置: {', '.join(model_info)}")
    
    if confidence:
        print(f"  预测置信度: {confidence['label']} ({confidence['score']*100:.0f}%)")
        if confidence.get('notes'):
            print(f"  说明: {'；'.join(confidence['notes'])}")
    print("=" * 65)

    op, cl = asian['open_prob'], asian['close_prob']
    print("\n【亚盘分析】（多家博彩公司平均值）")
    print(f"  让球变化: {asian['handicap_trend']}")
    print(f"  水位变化: {asian['water_trend']}")
    # 根据让球方向显示正确的标签
    if asian['handicap'] > 0:
        print(f"  初盘真实概率: 主让球方 {op.get('home_give', op.get('home', 0.5))*100:.1f}% / 客受让方 {op.get('away_recv', op.get('away', 0.5))*100:.1f}%")
        print(f"  终盘真实概率: 主让球方 {cl.get('home_give', cl.get('home', 0.5))*100:.1f}% / 客受让方 {cl.get('away_recv', cl.get('away', 0.5))*100:.1f}%")
    elif asian['handicap'] < 0:
        print(f"  初盘真实概率: 主受让方 {op.get('home_recv', op.get('home', 0.5))*100:.1f}% / 客让球方 {op.get('away_give', op.get('away', 0.5))*100:.1f}%")
        print(f"  终盘真实概率: 主受让方 {cl.get('home_recv', cl.get('home', 0.5))*100:.1f}% / 客让球方 {cl.get('away_give', cl.get('away', 0.5))*100:.1f}%")
    else:
        print(f"  初盘真实概率: 主队 {op.get('home', 0.5)*100:.1f}% / 客队 {op.get('away', 0.5)*100:.1f}%")
        print(f"  终盘真实概率: 主队 {cl.get('home', 0.5)*100:.1f}% / 客队 {cl.get('away', 0.5)*100:.1f}%")
    print(f"  终盘判断: {asian['favor_desc']}，{asian['diff_desc']}")
    if asian.get('implied_supremacy') is not None:
        print(f"  反推净胜球: {asian['implied_supremacy']:+.2f}（非盘口线 {asian['handicap']:+.2f}）")

    eo, ec = euro['open'], euro['close']
    print("\n【欧赔分析】（多家博彩公司平均值）")
    print(f"  初盘: 主胜{eo['home']*100:.1f}% | 平{eo['draw']*100:.1f}% | 客胜{eo['away']*100:.1f}%")
    print(f"  终盘: 主胜{ec['home']*100:.1f}% | 平{ec['draw']*100:.1f}% | 客胜{ec['away']*100:.1f}%")
    print(f"  变化趋势: {', '.join(euro['changes']) if euro['changes'] else '赔率稳定'}")
    mom = euro.get('momentum') or {}
    if mom.get('summary'):
        print(f"  欧赔走势: {mom['summary']}")
    if euro.get('implied_supremacy') is not None:
        print(f"  欧赔反推净胜球: {euro['implied_supremacy']:+.2f}")
    el = euro.get('implied_lambdas')
    if el:
        print(f"  欧赔隐含 λ: 主{el['home']:.2f} / 客{el['away']:.2f}")

    kelly = euro.get('kelly')
    if kelly:
        ko, kc = kelly['open'], kelly['close']
        rr = kelly['return_rate']['close']
        print("\n【凯利指数分析】（欧赔 × 去水概率 × 100）")
        print(f"  理论返还率: 初盘{kelly['return_rate']['open']:.1f}% → 终盘{rr:.1f}%")
        print(f"  初盘凯利: 主胜{ko['home']:.1f} | 平{ko['draw']:.1f} | 客胜{ko['away']:.1f}")
        print(f"  终盘凯利: 主胜{kc['home']:.1f} | 平{kc['draw']:.1f} | 客胜{kc['away']:.1f}")
        if kelly['kelly_changes']:
            print(f"  凯利变化: {', '.join(kelly['kelly_changes'])}")
        if kelly['risks']:
            print(f"  风险提示: {'；'.join(kelly['risks'])}")
        if kelly['favors']:
            print(f"  相对看好: {'；'.join(kelly['favors'])}")
        print(f"  综合: {kelly['summary']}")
        
        # 新增：凯利时序趋势
        kelly_trend = kelly.get('trend')
        if kelly_trend and kelly_trend['summary'] != '数据不足':
            print(f"  凯利走势: {kelly_trend['summary']}")
            if kelly_trend.get('crossing_events'):
                for event in kelly_trend['crossing_events']:
                    print(f"    {event['desc']}")

    to, tc = total['open_prob'], total['close_prob']
    print("\n【大小球分析】（多家博彩公司平均值）")
    print(f"  初盘: 线{total['open_line']} | 大{to['over']*100:.1f}% / 小{to['under']*100:.1f}%")
    print(f"  终盘: 线{total['close_line']} | 大{tc['over']*100:.1f}% / 小{tc['under']*100:.1f}%")
    print(f"  判断: {total['lean_desc']}")
    print(f"  泊松反推总进球: {total.get('implied_total', 0):.2f}")
    print(f"  期望总进球区间: {total['expected_goals'][0]}-{total['expected_goals'][1]}球")

    if team:
        print("\n【球队攻防强度】（500.com 近10场 + 主客场）")
        print(f"  {team['summary']}")
        print(f"  主队进攻{team['attack_home']:.2f}球/场 防守{team['defense_home']:.2f}失/场")
        print(f"  客队进攻{team['attack_away']:.2f}球/场 防守{team['defense_away']:.2f}失/场")

    # 新增：联合异常特征
    anomaly = result.get('anomaly')
    if anomaly:
        joint_water = anomaly.get('joint_water')
        euro_asian_dev = anomaly.get('euro_asian_deviation')
        
        print("\n【联合异常特征分析】")
        if joint_water:
            print(f"  水位变化乘积: 主队水位变化{joint_water['asian_water_change']:+.3f} × 大球水位变化{joint_water['total_water_change']:+.3f} = {joint_water['joint_water_feature']:+.4f}")
            if joint_water.get('hint_desc'):
                print(f"  ⚡ {joint_water['hint_desc']}")
        
        if euro_asian_dev:
            print(f"  欧赔亚盘偏差: 欧赔隐含让球{euro_asian_dev['implied_handicap']:+.2f} vs 实际盘口{euro_asian_dev['actual_handicap']:+.2f}，偏差{euro_asian_dev['deviation']:+.2f}")

    print("\n" + "=" * 65)
    print("【综合信号汇总】")
    print("=" * 65)
    hcap = asian['handicap']
    if hcap > 0:
        print(f"  强弱判断: 主队较强（主让{hcap}球）")
    elif hcap < 0:
        print(f"  强弱判断: 客队较强（客让{abs(hcap)}球）")
    else:
        print("  强弱判断: 双方实力接近")
    dominant = max([('主胜', ec['home']), ('平局', ec['draw']), ('客胜', ec['away'])], key=lambda x: x[1])
    print(f"  欧赔最高概率结果: {dominant[0]} ({dominant[1]*100:.1f}%)")
    print(f"  期望总进球: {total['expected_goals'][0]}-{total['expected_goals'][1]} 球")
    if model.get('supremacy_blended') is not None:
        print(f"  融合净胜球: 亚{model.get('supremacy_asian', 0):+.2f} + 欧{model.get('supremacy_euro', 0):+.2f} → {model['supremacy_blended']:+.2f}")
    print(f"  泊松期望进球: 主队 λ={model['lam_home']:.2f} / 客队 λ={model['lam_away']:.2f}")
    
    # 显示概率校准和集成信息
    if model.get('calibrated'):
        print(f"  ✓ 概率已校准: 应用{model['calibration_method']}方法")
    
    if model.get('ensemble_size'):
        print(f"  ✓ 多模型集成: 融合{model['ensemble_size']}个模型输出")

    print("\n" + "=" * 65)
    print("【Top 5 候选比分】（含冷热标记）")
    print("=" * 65)
    top_prob = model['top_scores'][0]['prob']
    for i, s in enumerate(model['top_scores'], 1):
        bar = "█" * int(s['prob'] / top_prob * 20)
        heat = _heat_tag(s.get('heat', 'neutral'))
        print(f"  #{i}: {home} {s['home']} - {s['away']} {away}  [{s['result']}]  "
              f"概率:{s['prob']*100:.1f}%  {heat}  {bar}")

    rec_n = len(model['recommend'])
    print("\n" + "=" * 65)
    print(f"【推荐：最可能的{rec_n}个比分】")
    print("=" * 65)
    for rank, s in enumerate(model['recommend'], 1):
        print(f"\n  第{rank}推荐: ★ {home} {s['home']} : {s['away']} {away} ★")
        print(f"           结果: {s['result']}  |  比分概率: {s['prob']*100:.1f}%")
        print(f"           理由: {' / '.join(s['reasons'])}")

    print("\n")
    print("  ⚠  免责声明：以上分析仅为概率统计参考，体育赛事结果受多种")
    print("     不确定因素影响，不构成任何投注建议。请理性对待！")
    print("=" * 65)


if __name__ == '__main__':
    main()


