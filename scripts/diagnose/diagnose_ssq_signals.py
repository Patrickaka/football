#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSQ 新信号消融实验 (v3.4 → v3.5)
================================
目标：提升单注中奖率（真实奖级，非面子指标）

测试信号（每个独立加权 vs 当前基线）：
  A. 遗漏值(gap)：号码距上次出现的间隔，冷号"回补"假设
  B. 冷热反转：反频率加权（赌冷号）
  C. 蓝球一阶马尔可夫：上期蓝球→本期蓝球转移概率
  D. 前区共现矩阵：号码两两共现频率加权
  E. 邻号/重号：上期开奖号的±1邻号和本身
  F. AC值约束过滤：覆盖结构生成后按AC值过滤

判定标准：walk-forward 2000 期，z-score > 2 (p<0.05) 才采纳
"""
import sys, json, math, random, logging
from collections import defaultdict, Counter
from itertools import combinations

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.ssq import (
    load_history, _analyze, _predict_sets, _is_valid_red,
    _weighted_sample, RED_RANGE, BLUE_RANGE, RED_COUNT, RECENT_WINDOW,
    ssq_prize_tier,
)

TRIALS = 2000


def sigma(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def predict_with_signal(train, analysis, n=5, seed=None, signal=None, **kw):
    """带可选信号的预测，保持 v3.2 覆盖结构（union=30 蛇形分组）不变，
    只改权重计算方式。signal=None 时退化为当前基线。"""
    rng = random.Random(seed)
    prev_red = set(train[-1]['red']) if train else set()
    prev_blue = train[-1]['blue'] if train else None

    # 基线权重
    red_w = [analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5
             for x in RED_RANGE]

    if signal == 'gap':
        # 遗漏值：距上次出现的期数，越长越"应该出"（回补假设）
        gap = kw.get('gap_scores', [0] * 33)
        # 归一化 gap 到 [0.5, 2.0] 乘子
        max_g = max(gap) if gap and max(gap) > 0 else 1
        red_w = [red_w[x - 1] * (0.5 + 1.5 * gap[x - 1] / max_g) for x in RED_RANGE]

    elif signal == 'anti_freq':
        # 冷热反转：反频率加权
        total = sum(analysis['red_freq'].values())
        red_w = [total / (analysis['red_freq'][x] + 1) for x in RED_RANGE]

    elif signal == 'cooccurrence':
        # 共现矩阵：号码两两出现频率作为权重
        cooc = kw.get('cooc_matrix', {})
        prev_red_list = list(prev_red)
        score = [red_w[x - 1] for x in RED_RANGE]
        for num in RED_RANGE:
            extra = sum(cooc.get((min(num, p), max(num, p)), 0) for p in prev_red_list)
            score[num - 1] += extra * 0.3
        red_w = score

    elif signal == 'adjacent':
        # 邻号/重号：上期开奖号±1 邻号加权
        adj_bonus = kw.get('adj_bonus', 2.0)
        score = list(red_w)
        for num in prev_red:
            for delta in (-1, 0, 1):
                nn = num + delta
                if 1 <= nn <= 33:
                    score[nn - 1] += adj_bonus
        red_w = score

    elif signal == 'markov_blue':
        # 蓝球马尔可夫不改变红球，只改蓝球选择
        pass  # 蓝球在下面处理

    # 红球排名池前30码蛇形分组（覆盖结构不变）
    ranking = sorted(RED_RANGE, key=lambda x: -red_w[x - 1])[:30]
    groups = [set() for _ in range(n)]
    for i, num in enumerate(ranking):
        row, col = i // n, i % n
        idx = col if row % 2 == 0 else (n - 1 - col)
        groups[idx].add(num)

    # 主推注合法性检查
    main = sorted(groups[0])
    if not _is_valid_red(main) or set(main) == prev_red:
        w_pool = [red_w[x - 1] for x in ranking]
        tries = 0
        while tries < 300:
            tries += 1
            red = sorted(_weighted_sample(ranking, w_pool, RED_COUNT, rng))
            if _is_valid_red(red) and set(red) != prev_red:
                main = red
                break
        groups[0] = set(main)
        rest = [x for x in ranking if x not in groups[0]]
        for g in groups[1:]:
            g.clear()
        for i, num in enumerate(rest):
            row, col = i // (n - 1), i % (n - 1)
            idx = col if row % 2 == 0 else (n - 2 - col)
            groups[1 + idx].add(num)

    # 蓝球选择
    if signal == 'markov_blue':
        # 蓝球一阶马尔可夫：上期蓝球→本期转移概率
        trans = kw.get('blue_transitions', {})
        prev_b = prev_blue
        if prev_b and prev_b in trans:
            probs = trans[prev_b]
            # 按转移概率加权抽样
            blues_pool = []
            weights = [probs.get(b, 0.1) for b in BLUE_RANGE]
            avail = [b for b in BLUE_RANGE if b != prev_b]
            w2 = [probs.get(b, 0.1) for b in avail]
            if sum(w2) > 0:
                blues_pool = [_weighted_sample(avail, w2, 1, rng)[0]
                               for _ in range(n)]
            else:
                blues_pool = rng.sample(avail, min(n, len(avail)))
            blues_pool = list(dict.fromkeys(blues_pool))  # 去重保序
            while len(blues_pool) < n:
                blues_pool.append(rng.choice(BLUE_RANGE))
            blues_pool = blues_pool[:n]
        else:
            avail_blues = [b for b in BLUE_RANGE if b != prev_blue]
            blues_pool = rng.sample(avail_blues, min(n, len(avail_blues)))
            while len(blues_pool) < n:
                blues_pool.append(rng.choice(BLUE_RANGE))
    else:
        avail_blues = [b for b in BLUE_RANGE if b != prev_blue]
        if len(avail_blues) < n:
            avail_blues = list(BLUE_RANGE)
        blues_pool = rng.sample(avail_blues, min(n, len(avail_blues)))
        while len(blues_pool) < n:
            blues_pool.append(rng.choice(BLUE_RANGE))

    return [{'red': sorted(groups[i]), 'blue': blues_pool[i]} for i in range(n)]


def compute_gap_scores(history):
    """计算每个红球的遗漏值（距上次出现多少期）"""
    last_seen = {n: len(history) for n in RED_RANGE}
    for i in range(len(history) - 1, -1, -1):
        for n in set(history[i]['red']):
            if last_seen[n] == len(history):
                last_seen[n] = len(history) - 1 - i
    return [last_seen[n] for n in RED_RANGE]


def compute_cooc_matrix(history, max_history=500):
    """计算红球两两共现频率（用最近 max_history 期防止过远历史稀释）"""
    recent = history[-max_history:]
    cooc = defaultdict(int)
    for rec in recent:
        for a, b in combinations(sorted(set(rec['red'])), 2):
            cooc[(a, b)] += 1
    return dict(cooc)


def compute_blue_transitions(history):
    """计算蓝球一阶马尔可夫转移矩阵"""
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(history) - 1):
        prev = history[i]['blue']
        curr = history[i + 1]['blue']
        trans[prev][curr] += 1
    # 归一化为概率
    result = {}
    for prev, counts in trans.items():
        total = sum(counts.values())
        result[prev] = {b: c / total for b, c in counts.items()}
    return result


def compute_adjacent_bonus(history):
    """计算邻号/重号是否对预测有信号"""
    pass  # 奖励值在 predict 中直接使用


def run_experiment():
    history = sorted(load_history(), key=lambda x: x['period'])
    print(f"[SSQ] 全量历史 {len(history)} 期")

    signals = {
        'baseline': None,
        'gap': 'gap',
        'anti_freq': 'anti_freq',
        'cooccurrence': 'cooccurrence',
        'adjacent': 'adjacent',
        'markov_blue': 'markov_blue',
    }

    results = {}
    for name, sig in signals.items():
        any_prize = 0  # 任1注中奖率（真实奖级 > 0）
        any3_red = 0   # 任1注红球≥3
        blue_any = 0   # 任1注蓝球命中
        n = 0
        for i in range(len(history) - TRIALS, len(history)):
            train = history[:i]
            if len(train) < 200:
                continue
            analysis = _analyze(train)
            kw = {}
            if sig == 'gap':
                kw['gap_scores'] = compute_gap_scores(train)
            elif sig == 'cooccurrence':
                kw['cooc_matrix'] = compute_cooc_matrix(train)
            elif sig == 'markov_blue':
                kw['blue_transitions'] = compute_blue_transitions(train)
            elif sig == 'adjacent':
                kw['adj_bonus'] = 2.0

            sets = predict_with_signal(train, analysis, n=5,
                                       seed=int(history[i]['period']),
                                       signal=sig, **kw)
            ar = set(history[i]['red'])
            ab = history[i]['blue']
            n += 1
            # 真实奖级
            won = False
            red3 = False
            for s in sets:
                rh = len(set(s['red']) & ar)
                bh = s['blue'] == ab
                tier = ssq_prize_tier(rh, bh)
                if tier > 0:
                    won = True
                if rh >= 3:
                    red3 = True
            if won: any_prize += 1
            if red3: any3_red += 1
            if any(s['blue'] == ab for s in sets): blue_any += 1

        # 精确随机基准
        from math import comb
        t = comb(33, 6)
        # 单注中奖率（六等奖：蓝球命中 6.25% + 五等：4红/3+蓝 约0.78% + ...）
        # 简化：单注任意奖 ≈ 6.71%（精确计算蓝球命中+红球≥3+蓝）
        p_blue = 1/16
        p_red3_blue = comb(6,3)*comb(27,3)/t * p_blue
        p_red4 = (comb(6,4)*comb(27,2) + comb(6,5)*27 + 1) / t
        p_red4_blue = p_red4 * p_blue
        p_red5_blue = comb(6,5)*27/t * p_blue
        p_red6_blue = 1/t * p_blue
        p_single_any = p_blue + (comb(6,3)*comb(27,3) + comb(6,4)*comb(27,2) +
                                 comb(6,5)*27 + 1) / t * (1 - p_blue) / t * 0  # 近似
        # 更精确：单注中奖 = P(蓝球命中) + P(蓝球未中 ∧ 红球≥4)
        p_single_prize = p_blue + (1 - p_blue) * (
            (comb(6,4)*comb(27,2) + comb(6,5)*27 + 1) / t
        )
        # 5注蓝球互异：任1注中蓝 = 5/16 = 31.25%
        # 5注任1注中奖 ≈ 1 - (1-p_single_prize_no_blue)^5 * (1-5/16)
        # 简化用实测对照
        p5_blue = 5/16
        p_single_red_ge3 = sum(comb(6,k)*comb(27,6-k) for k in range(3,7)) / t
        p5_red_ge3 = 1 - (1 - p_single_red_ge3) ** 5  # 独立近似（覆盖结构更高）

        results[name] = {
            'n': n,
            'any_prize_rate': any_prize / n,
            'any_red_ge3_rate': any3_red / n,
            'blue_any_hit_rate': blue_any / n,
        }
        r = results[name]
        s_prize = sigma(r['any_prize_rate'], n)
        z_prize = (r['any_prize_rate'] - 0.0671 * 5 * 0.3) / s_prize if s_prize else 0  # 粗糙基准
        print(f"\n[{name}] n={n}")
        print(f"  任1注中奖率:   {r['any_prize_rate']:.4f}")
        print(f"  任1注红≥3:     {r['any_red_ge3_rate']:.4f}  (随机基准≈{1-(1-p_single_red_ge3)**5:.4f})")
        print(f"  任1注蓝球命中: {r['blue_any_hit_rate']:.4f}  (基准=0.3125)")

    # 对照表
    print("\n" + "=" * 70)
    print(f"{'信号':<16} {'任1注中奖':>10} {'任1注红≥3':>10} {'蓝球命中':>10}")
    print("-" * 70)
    base = results.get('baseline', {})
    for name, r in results.items():
        diff_p = (r['any_prize_rate'] - base.get('any_prize_rate', 0)) * 100
        diff_r = (r['any_red_ge3_rate'] - base.get('any_red_ge3_rate', 0)) * 100
        diff_b = (r['blue_any_hit_rate'] - base.get('blue_any_hit_rate', 0)) * 100
        marker = " ←基线" if name == 'baseline' else ""
        print(f"{name:<16} {r['any_prize_rate']:>9.2%} {r['any_red_ge3_rate']:>9.2%} "
              f"{r['blue_any_hit_rate']:>9.2%}{marker}")
        if name != 'baseline':
            print(f"  {'':14} Δ={diff_p:+.2f}pp {'':5} Δ={diff_r:+.2f}pp {'':5} Δ={diff_b:+.2f}pp")

    # 保存
    out = {'date': '2026-08-24', 'trials': TRIALS, 'results': results}
    path = 'data/diagnose_ssq_signals_20260824.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")
    return out


if __name__ == '__main__':
    run_experiment()
