#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：用真实重构后的北单函数跑离线回测
（monkeypatch 掉网络抓取，用 CSV 隐含赔率驱动）
确认 analyze_zjq / analyze_bifen 实际产出与优化预期一致。
"""
import os
import csv
import math
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.beidan as beidan

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
DIV_LEAGUE = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}


def safe_float(v, default=None):
    try:
        if v is None or v == '':
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


# ---- monkeypatch 网络抓取 ----
def fake_ouzhi(match_id):
    # 返回 None 会触发 fallback 到 match 自带 spf 赔率；这里直接构造
    return fake_ouzhi._store.get(match_id)


fake_ouzhi._store = {}


def load_matches():
    rows = []
    for fn in sorted(os.listdir(CSV_DIR)):
        if not (fn.endswith('.csv') and fn[:2] in DIV_LEAGUE):
            continue
        div = fn[:2]
        league = DIV_LEAGUE[div]
        with open(os.path.join(CSV_DIR, fn), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                hg = safe_float(r.get('FTHG'))
                ag = safe_float(r.get('FTAG'))
                if hg is None or ag is None:
                    continue
                ah, ad, aa = safe_float(r.get('AvgH')), safe_float(r.get('AvgD')), safe_float(r.get('AvgA'))
                if not (ah and ad and aa):
                    continue
                mid = f"{div}_{r['Date']}_{r['HomeTeam']}_{r['AwayTeam']}"
                fake_ouzhi._store[mid] = {'home': ah, 'draw': ad, 'away': aa}
                rows.append({
                    'match_id': mid, 'league': league,
                    'home': r['HomeTeam'], 'away': r['AwayTeam'],
                    'hg': int(hg), 'ag': int(ag),
                    'over': safe_float(r.get('Avg>2.5')), 'under': safe_float(r.get('Avg<2.5')),
                })
    return rows


def main():
    beidan.fetch_ouzhi_odds = fake_ouzhi
    matches = load_matches()
    print(f"样本: {len(matches)} 场")

    goal_keys = [str(i) for i in range(8)] + ['7+']
    score_keys = [(h, a) for h in range(8) for a in range(8)]

    z_ll = z_brier = z_top2 = z_n = 0.0
    s_ll = s_brier = s_top1 = s_top3 = s_n = 0.0

    for m in matches:
        mid = m['match_id']
        over, under = m['over'], m['under']
        goals_data = {'history': [{'over_odds': over, 'under_odds': under}]} if (over and under) else None

        # 总进球（真实重构函数）
        zr = beidan.analyze_zjq({'id': mid, 'league': m['league'], 'home': m['home'], 'away': m['away'],
                                 'num': '', 'time': ''}, zjq_odds=None, asian_data=None, goals_data=goals_data)
        if 'probabilities' in zr:
            dist = zr['probabilities']
            at = m['hg'] + m['ag']
            ak = '7+' if at >= 7 else str(at)
            p = max(dist.get(ak, 0.0), 1e-9)
            z_ll += -math.log(p)
            z_brier += sum((dist.get(k, 0.0) - (1.0 if k == ak else 0.0)) ** 2 for k in goal_keys)
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            if ak in [k for k, _ in ranked[:2]]:
                z_top2 += 1
            z_n += 1

        # 比分（真实重构函数，无市场赔率 → 模型路径）
        br = beidan.analyze_bifen({'id': mid, 'league': m['league'], 'home': m['home'], 'away': m['away'],
                                   'num': '', 'time': '', 'handicap': 0}, bifen_odds=None, asian_data=None, goals_data=goals_data)
        if 'probabilities' in br:
            dist = br['probabilities']
            ak = (m['hg'], m['ag'])
            p = max(dist.get(ak, 0.0), 1e-9)
            s_ll += -math.log(p)
            s_brier += sum((dist.get(k, 0.0) - (1.0 if k == ak else 0.0)) ** 2 for k in score_keys)
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            if ranked and ranked[0][0] == ak:
                s_top1 += 1
            if ak in [k for k, _ in ranked[:3]]:
                s_top3 += 1
            s_n += 1

    print("\n" + "=" * 70)
    print("真实重构后北单函数离线回测")
    print("=" * 70)
    print(f"总进球 zjq : LogLoss={z_ll/z_n:.4f}  Brier={z_brier/z_n:.4f}  Top2={100*z_top2/z_n:.2f}%  (n={int(z_n)})")
    print(f"比分 bifen : LogLoss={s_ll/s_n:.4f}  Brier={s_brier/s_n:.4f}  Top1={100*s_top1/s_n:.2f}%  Top3={100*s_top3/s_n:.2f}%  (n={int(s_n)})")
    print("\n对照（回测脚本理论值）：")
    print("  zjq_dc  : LogLoss=1.8589  Top2=45.41%")
    print("  score_dc: LogLoss=2.9263  Top1=11.88%  Top3=31.78%")


if __name__ == '__main__':
    main()
