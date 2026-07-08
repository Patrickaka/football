#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线回测基线 (Offline Backtest Baseline)
=========================================

用真实历史 CSV (football-data.co.uk 格式) 驱动生产预测管线，量化三大市场命中率：
  - 胜平负 (1X2)
  - 进球数 (Top2 命中 + 大小球 2.5)
  - 半全场 (Top1 / Top3 命中)
并输出 Brier / LogLoss。

设计原则（诚实、零泄漏）：
  - 仅用赛前市场信号（欧赔/亚盘/大小球去水概率），不注入球队特征，
    因此不存在用赛果反推特征的数据泄漏。
  - 与生产同款调用：predict_scores(..., model_type='negative_binomial',
    enable_ensemble=True, ensemble_size=5)，再从 candidates 派生三市场。
  - 不含 analyze_match 后处理（残差模型/贝叶斯校准/盘口移动）——这些在当前
    生产环境要么无数据(空表)要么需要盘口移动/实时数据，离线不可复现。
    故本基线衡量的是「核心市场模型」，也正是这些改动实际作用的层。

关键约定：
  - football-data 的 AHh 符号与生产相反（AHh 负=主让），生产 parse_handicap
    是「正=主让」，故 handicap = -AHh 必须反转。
"""

import os
import sys
import csv
import math
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.football as fb

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

LEAGUE_MAP = {'E0': '英超', 'SP1': '西甲', 'D1': '德甲', 'I1': '意甲', 'F1': '法甲'}
SEASONS = ['2425', '2526']

HTF_CODES = ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']


def _f(row, key, default=None):
    """从 CSV 行安全取浮点。"""
    v = row.get(key, '')
    if v is None or v == '':
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _result(h, a):
    return 'H' if h > a else ('A' if h < a else 'D')


def build_inputs(row):
    """把一行 CSV 转成 predict_scores 需要的 asian/euro/total 三个 dict。
    返回 None 表示赔率不完整，跳过该场。"""
    # 欧赔（优先 Avg 平均，退化到 B365）
    oh = _f(row, 'AvgH') or _f(row, 'B365H')
    od = _f(row, 'AvgD') or _f(row, 'B365D')
    oa = _f(row, 'AvgA') or _f(row, 'B365A')
    if not (oh and od and oa and oh > 1 and od > 1 and oa > 1):
        return None
    ph, pd, pa = fb.remove_vig(oh, od, oa)

    # 大小球 2.5
    over = _f(row, 'Avg>2.5') or _f(row, 'B365>2.5')
    under = _f(row, 'Avg<2.5') or _f(row, 'B365<2.5')
    if not (over and under and over > 1 and under > 1):
        return None
    p_over, p_under = fb.remove_vig(over, under)

    # 亚盘（AHh 符号反转：football-data 负=主让 → 生产 正=主让）
    ahh = _f(row, 'AvgAHH') or _f(row, 'B365AHH')
    aha = _f(row, 'AvgAHA') or _f(row, 'B365AHA')
    ah_line = _f(row, 'AHh')
    if ah_line is None or not (ahh and aha and ahh > 1 and aha > 1):
        return None
    handicap = -ah_line  # 关键反转
    p_ah_home, p_ah_away = fb.remove_vig(ahh, aha)

    euro = {
        'close': {'home': ph, 'draw': pd, 'away': pa},
        'open': {'home': ph, 'draw': pd, 'away': pa},
    }
    asian = {
        'handicap': handicap,
        'open_handicap': handicap,
        'close_prob': {'home': p_ah_home, 'away': p_ah_away},
        'open_prob': {'home': p_ah_home, 'away': p_ah_away},
    }
    total = {
        'close_line': 2.5,
        'open_line': 2.5,
        'close_prob': {'over': p_over, 'under': p_under},
        'open_prob': {'over': p_over, 'under': p_under},
    }
    return asian, euro, total


def candidates_to_1x2(candidates):
    p = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    for (h, a), prob in candidates:
        p[_result(h, a)] += prob
    s = sum(p.values())
    if s > 0:
        p = {k: v / s for k, v in p.items()}
    return p


def candidates_to_goal_dist(candidates):
    d = defaultdict(float)
    for (h, a), prob in candidates:
        d[h + a] += prob
    s = sum(d.values())
    if s > 0:
        d = {k: v / s for k, v in d.items()}
    return dict(d)


def run(limit=None, leagues=None, verbose=False, per_file=None):
    files = []
    for code in (leagues or LEAGUE_MAP.keys()):
        for season in SEASONS:
            fn = os.path.join(DATA_DIR, f'{code}_{season}.csv')
            if os.path.exists(fn):
                files.append((code, fn))

    agg = {
        'n': 0, 'skipped': 0, 'errors': 0,
        '1x2_hit': 0,
        'goal_top2_hit': 0, 'ou25_hit': 0,
        'htf_top1_hit': 0, 'htf_top3_hit': 0, 'htf_n': 0,
        'brier_1x2': 0.0, 'logloss_1x2': 0.0,
        'brier_ou': 0.0, 'logloss_ou': 0.0,
        'logloss_htf': 0.0,
    }
    per_league = defaultdict(lambda: {'n': 0, '1x2_hit': 0, 'goal_top2_hit': 0,
                                      'ou25_hit': 0, 'htf_top1_hit': 0, 'htf_top3_hit': 0})

    for code, fn in files:
        league = LEAGUE_MAP.get(code, code)
        file_n = 0
        with open(fn, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and agg['n'] >= limit:
                    break
                if per_file and file_n >= per_file:
                    break
                fthg, ftag = _f(row, 'FTHG'), _f(row, 'FTAG')
                if fthg is None or ftag is None:
                    agg['skipped'] += 1
                    continue
                built = build_inputs(row)
                if built is None:
                    agg['skipped'] += 1
                    continue
                asian, euro, total = built

                try:
                    candidates, lam_h, lam_a, meta = fb.predict_scores(
                        asian, euro, total,
                        team_strength=None, league_profile={'name': league},
                        model_type='negative_binomial',
                        enable_draw_calibration=True, enable_calibration=True,
                        calibration_method='platt',
                        enable_ensemble=True, ensemble_size=5,
                    )
                except Exception as e:
                    agg['errors'] += 1
                    if verbose:
                        print(f"[ERR] {league} {row.get('HomeTeam')} vs {row.get('AwayTeam')}: {e}")
                    continue

                fthg, ftag = int(fthg), int(ftag)
                actual_1x2 = _result(fthg, ftag)
                actual_goals = fthg + ftag
                agg['n'] += 1
                file_n += 1
                per_league[league]['n'] += 1

                # ---- 1X2 ----
                p1x2 = candidates_to_1x2(candidates)
                pred_1x2 = max(p1x2, key=p1x2.get)
                if pred_1x2 == actual_1x2:
                    agg['1x2_hit'] += 1
                    per_league[league]['1x2_hit'] += 1
                agg['logloss_1x2'] += -math.log(max(1e-15, p1x2.get(actual_1x2, 0.0)))
                agg['brier_1x2'] += sum((p1x2.get(k, 0.0) - (1.0 if k == actual_1x2 else 0.0)) ** 2
                                        for k in ('H', 'D', 'A'))

                # ---- 进球数 ----
                gd = candidates_to_goal_dist(candidates)
                top2 = sorted(gd, key=gd.get, reverse=True)[:2]
                if actual_goals in top2:
                    agg['goal_top2_hit'] += 1
                    per_league[league]['goal_top2_hit'] += 1
                # 大小球 2.5
                p_over_model = sum(v for k, v in gd.items() if k >= 3)
                pred_over = p_over_model >= 0.5
                actual_over = actual_goals >= 3
                if pred_over == actual_over:
                    agg['ou25_hit'] += 1
                    per_league[league]['ou25_hit'] += 1
                p_ou = p_over_model if actual_over else (1 - p_over_model)
                agg['logloss_ou'] += -math.log(max(1e-15, p_ou))
                agg['brier_ou'] += (p_over_model - (1.0 if actual_over else 0.0)) ** 2

                # ---- 半全场 ----
                hthg, htag = _f(row, 'HTHG'), _f(row, 'HTAG')
                if hthg is not None and htag is not None:
                    actual_htf = _result(int(hthg), int(htag)) + actual_1x2
                    try:
                        htf = fb.calculate_half_full_time_probs(
                            candidates, team_strength=None, asian=asian, total=total,
                            home_team=row.get('HomeTeam', ''), away_team=row.get('AwayTeam', ''),
                            league=league,
                        )
                        htf_probs = fb._half_full_probs_to_dict(htf) or {}
                    except Exception:
                        htf_probs = {}
                    if htf_probs:
                        agg['htf_n'] += 1
                        ranked = sorted(htf_probs, key=htf_probs.get, reverse=True)
                        if ranked and ranked[0] == actual_htf:
                            agg['htf_top1_hit'] += 1
                            per_league[league]['htf_top1_hit'] += 1
                        if actual_htf in ranked[:3]:
                            agg['htf_top3_hit'] += 1
                            per_league[league]['htf_top3_hit'] += 1
                        agg['logloss_htf'] += -math.log(max(1e-15, htf_probs.get(actual_htf, 0.0)))

    _report(agg, per_league)
    return agg


def _pct(x, n):
    return f"{100.0 * x / n:.2f}%" if n else "n/a"


def _report(agg, per_league):
    n = agg['n']
    print("=" * 64)
    print("离线回测基线结果 (核心市场模型, 仅赛前赔率, 零泄漏)")
    print("=" * 64)
    print(f"有效场次: {n}  | 跳过(赔率不全): {agg['skipped']}  | 预测异常: {agg['errors']}")
    if not n:
        print("无有效样本。")
        return
    print("-" * 64)
    print(f"胜平负 1X2 命中率      : {_pct(agg['1x2_hit'], n)}  ({agg['1x2_hit']}/{n})")
    print(f"  1X2 Brier / LogLoss  : {agg['brier_1x2']/n:.4f} / {agg['logloss_1x2']/n:.4f}")
    print("-" * 64)
    print(f"进球数 Top2 命中率     : {_pct(agg['goal_top2_hit'], n)}  ({agg['goal_top2_hit']}/{n})")
    print(f"大小球 2.5 命中率      : {_pct(agg['ou25_hit'], n)}  ({agg['ou25_hit']}/{n})")
    print(f"  OU2.5 Brier / LogLoss: {agg['brier_ou']/n:.4f} / {agg['logloss_ou']/n:.4f}")
    print("-" * 64)
    hn = agg['htf_n']
    print(f"半全场 Top1 命中率     : {_pct(agg['htf_top1_hit'], hn)}  ({agg['htf_top1_hit']}/{hn})")
    print(f"半全场 Top3 命中率     : {_pct(agg['htf_top3_hit'], hn)}  ({agg['htf_top3_hit']}/{hn})")
    print(f"  HTF LogLoss          : {agg['logloss_htf']/hn:.4f}" if hn else "  HTF LogLoss: n/a")
    print("=" * 64)
    print("分联赛 1X2 / 进球Top2 / 大小球 / 半全场Top1:")
    for lg, s in sorted(per_league.items(), key=lambda kv: -kv[1]['n']):
        m = s['n']
        if not m:
            continue
        print(f"  {lg:<4} n={m:<4} "
              f"1X2={_pct(s['1x2_hit'], m):>7} "
              f"进球={_pct(s['goal_top2_hit'], m):>7} "
              f"大小={_pct(s['ou25_hit'], m):>7} "
              f"半全={_pct(s['htf_top1_hit'], m):>7}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='足球离线回测基线')
    ap.add_argument('--limit', type=int, default=None, help='最多回测场次(全局)')
    ap.add_argument('--per-file', type=int, default=None, dest='per_file',
                    help='每个联赛/赛季文件最多回测场次(用于均衡各联赛样本)')
    ap.add_argument('--leagues', nargs='*', default=None, help='联赛代码 E0/SP1/D1/I1/F1')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    run(limit=args.limit, leagues=args.leagues, verbose=args.verbose, per_file=args.per_file)
