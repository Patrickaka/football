#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLT v4.6 六注后区全覆盖 — walk-forward 差分验证
================================================
对比同一期次下:
  v4.5 = 前5注(primary+balanced/rank/hot/cold), 后区覆盖10/12码
  v4.6 = +第6注 back_cover(后区=剩余2码), 后区覆盖12/12码

验证点:
1. v4.6 任1注后区≥1 应=100%（鸽笼原理）
2. 第6注对 any_prize 的独立贡献（只有第6注中奖的期数）
3. 差分用配对统计（同批期次）

注意: 训练数据固定（saved 只加载一次），规避单例截断 bug。
"""
import sys, json, math, logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery.analyzer import get_lottery_analyzer
from src.lottery.records import dlt_prize_tier

TRIALS = 500


def main():
    print("[DLT] 加载分析器...")
    a = get_lottery_analyzer()
    saved = list(a.history_data)
    print(f"[DLT] 历史 {len(saved)} 期, 跑 {TRIALS} 期\n")

    n = 0
    v45 = {'back_ge1': 0, 'prize': 0, 'j21': 0}
    v46 = {'back_ge1': 0, 'prize': 0, 'j21': 0}
    only_cover_prize = 0    # 只有第6注中奖的期数
    only_cover_j21 = 0      # 只有第6注 2+1 的期数
    cover_front_hits = []   # 第6注前区命中数分布

    for i in range(TRIALS):
        if i >= len(saved) - 81:
            break
        a.history_data = list(saved[i + 1:])
        a.update_statistics()
        multi = a.generate_multi_strategy_recommendations(
            voting_result=a.multi_model_voting(front_n=20, back_n=10, skip_ml=True))
        recs = [x for x in multi['recommendations']
                if not x['strategy'].startswith('picked')]
        core = [x for x in recs if x['strategy'] != 'back_cover']
        cover = [x for x in recs if x['strategy'] == 'back_cover']
        af = set(saved[i]['front'])
        ab = set(saved[i]['back'])
        n += 1

        def eval_notes(notes):
            b1 = prize = j21 = False
            for r in notes:
                hf = len(af & set(r['front']))
                hb = len(ab & set(r['back']))
                if hb >= 1: b1 = True
                if dlt_prize_tier(hf, hb) > 0: prize = True
                if hf >= 2 and hb >= 1: j21 = True
            return b1, prize, j21

        b5, p5, j5 = eval_notes(core)
        b6, p6, j6 = eval_notes(recs)  # 全部6注
        if b5: v45['back_ge1'] += 1
        if p5: v45['prize'] += 1
        if j5: v45['j21'] += 1
        if b6: v46['back_ge1'] += 1
        if p6: v46['prize'] += 1
        if j6: v46['j21'] += 1
        if cover:
            # 第6注独立指标
            cb, cp, cj = eval_notes(cover)
            if cp and not p5: only_cover_prize += 1
            if cj and not j5: only_cover_j21 += 1
            cover_front_hits.append(len(af & set(cover[0]['front'])))

    print("=" * 62)
    print(f"{'指标':<16} {'v4.5(5注)':>10} {'v4.6(6注)':>10} {'Δpp':>7}  说明")
    print("-" * 62)
    rows = [
        ('任1注后区≥1', v45['back_ge1'], v46['back_ge1'], '鸽笼保证, 应=100%'),
        ('任1注中奖', v45['prize'], v46['prize'], '第6注独立贡献'),
        ('任1注2+1', v45['j21'], v46['j21'], ''),
    ]
    for name, c5, c6, note in rows:
        p5r, p6r = c5 / n, c6 / n
        print(f"{name:<16} {p5r:>10.1%} {p6r:>10.1%} {(p6r - p5r) * 100:>+6.1f}  {note}")

    print("-" * 62)
    print(f"第6注独立中奖的期数(仅第6注中, 前5注全不中): {only_cover_prize}/{n} = {only_cover_prize/n:.1%}")
    print(f"第6注独立2+1的期数: {only_cover_j21}/{n} = {only_cover_j21/n:.1%}")
    if cover_front_hits:
        avg = sum(cover_front_hits) / len(cover_front_hits)
        ge2 = sum(1 for x in cover_front_hits if x >= 2) / len(cover_front_hits)
        print(f"第6注前区命中均值: {avg:.3f} (随机期望 5*5/35=0.714), 前区≥2率: {ge2:.1%} (单注随机基准≈13.9%)")

    # 配对 McNemar: v4.6 vs v4.5 on any_prize — 重算需要 per-period, 这里直接由独立贡献给出
    # only_cover_prize 即 b (v4.6 赢 v4.5 输), c=0 (第6注只增不减, v4.5 中奖则 v4.6 必中奖)
    z = only_cover_prize / math.sqrt(only_cover_prize) if only_cover_prize else 0.0
    print(f"\n配对 McNemar (任1注中奖, v4.6 vs v4.5): b={only_cover_prize}, c=0, z=+{z:.2f}")
    print("结论: 第6注只增不减(超集), 中奖率提升全部来自后区全覆盖的结构性收益")

    out = {'date': '2026-08-24', 'trials': n,
           'v45': {k: v / n for k, v in v45.items()},
           'v46': {k: v / n for k, v in v46.items()},
           'only_cover_prize': only_cover_prize / n,
           'only_cover_j21': only_cover_j21 / n}
    path = 'data/diagnose_dlt_v46_diff_20260824.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"已保存: {path}")


if __name__ == '__main__':
    main()
