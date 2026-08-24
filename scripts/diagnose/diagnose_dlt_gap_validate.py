#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLT gap=0.15 发现的严格验证
============================
上轮扫描: gap=0.15 在 500 期中 2+1=29.0% (z=4.65)。但存在两个疑点:
1. 非单调: gap=0.06 比 0.00/0.12 都差 → 可能过拟合
2. z 分数按独立样本算, 实际是同一批期次的配对样本 → 显著性被高估

本脚本:
1. W1=最近500期(原始窗口, 样本内), W2/W3=更早各500期(样本外, 从未参与发现)
2. gap ∈ {0.00, 0.12(基线), 0.15} 三窗口 × 配对 McNemar 检验
3. 每期记录二值结果, 输出 per-period 配对差分
"""
import sys, json, math, logging

sys.path.insert(0, '.')
logging.disable(logging.WARNING)

from src.lottery.config import FEATURE_WEIGHTS, BACK_FEATURE_WEIGHTS
from src.lottery.analyzer import get_lottery_analyzer
from src.lottery.records import dlt_prize_tier

MIN_TRAIN = 81  # 与上轮一致: len(saved)-i-1 >= 80


def set_gap_weight(new_gap):
    fw = dict(FEATURE_WEIGHTS)
    orig_total = sum(fw.values())
    without_gap = orig_total - fw['gap']
    scale = (orig_total - new_gap) / without_gap if without_gap > 0 else 1.0
    return {k: (v * scale if k != 'gap' else new_gap) for k, v in fw.items()}


def run_dlt_wf(saved, i_start, i_end, front_weights=None, label=''):
    """跑 [i_start, i_end) 期, 返回每期二值结果列表"""
    from src.lottery import config as dlt_config
    orig_fw = dict(dlt_config.FEATURE_WEIGHTS)
    if front_weights is not None:
        dlt_config.FEATURE_WEIGHTS.clear()
        dlt_config.FEATURE_WEIGHTS.update(front_weights)
    try:
        a = get_lottery_analyzer()
        outcomes = []  # 每期: dict(issue, j21, prize)
        for i in range(i_start, i_end):
            if i >= len(saved) - MIN_TRAIN:
                break
            a.history_data = list(saved[i + 1:])
            a.update_statistics()
            multi = a.generate_multi_strategy_recommendations(
                voting_result=a.multi_model_voting(front_n=20, back_n=10, skip_ml=True))
            recs = [x for x in multi['recommendations']
                    if not x['strategy'].startswith('picked')]
            af = set(saved[i]['front'])
            ab = set(saved[i]['back'])
            prize = j21 = False
            for r in recs:
                hf = len(af & set(r['front']))
                hb = len(ab & set(r['back']))
                if dlt_prize_tier(hf, hb) > 0: prize = True
                if hf >= 2 and hb >= 1: j21 = True
            outcomes.append({'issue': saved[i]['issue'], 'j21': j21, 'prize': prize})
        return outcomes
    finally:
        dlt_config.FEATURE_WEIGHTS.clear()
        dlt_config.FEATURE_WEIGHTS.update(orig_fw)


def mcnemar(out_a, out_b, key):
    """配对检验: a 赢 b 输=b, a 输 b 赢=c, z=(b-c)/sqrt(b+c)"""
    assert len(out_a) == len(out_b)
    b = c = 0
    for x, y in zip(out_a, out_b):
        if x[key] and not y[key]: b += 1
        elif y[key] and not x[key]: c += 1
    z = (b - c) / math.sqrt(b + c) if (b + c) > 0 else 0.0
    return b, c, z


def main():
    print("[DLT] 加载分析器...")
    a = get_lottery_analyzer()
    saved = list(a.history_data)
    total = len(saved)
    print(f"[DLT] 历史 {total} 期")
    print(f"[DLT] 窗口: W1=[0,500)样本内  W2=[500,1000)样本外  W3=[1000,1500)样本外\n")

    windows = {'W1(样本内)': (0, 500), 'W2(样本外)': (500, 1000), 'W3(样本外)': (1000, 1500)}
    gaps = {'gap=0.00': set_gap_weight(0.00),
            'gap=0.12(基线)': None,
            'gap=0.15': set_gap_weight(0.15)}

    all_results = {}
    for wname, (s, e) in windows.items():
        all_results[wname] = {}
        for gname, fw in gaps.items():
            print(f"  {wname} [{gname}]...", end='', flush=True)
            out = run_dlt_wf(saved, s, e, front_weights=fw, label=gname)
            n = len(out)
            j21 = sum(1 for x in out if x['j21']) / n if n else 0
            pz = sum(1 for x in out if x['prize']) / n if n else 0
            all_results[wname][gname] = {'outcomes': out, 'j21': j21, 'prize': pz, 'n': n}
            print(f" n={n} 2+1={j21:.1%} 中奖={pz:.1%}")
        print()

    print("=" * 78)
    print("配对 McNemar 检验 (gap=0.15 vs gap=0.12 基线, 同一期次配对)")
    print("-" * 78)
    print(f"{'窗口':<14} {'0.15命中率':>9} {'0.12命中率':>9} {'b(0.15赢)':>9} {'c(0.12赢)':>9} {'z':>6}  判定")
    print("-" * 78)
    summary = []
    for wname in windows:
        r15 = all_results[wname]['gap=0.15']
        r12 = all_results[wname]['gap=0.12(基线)']
        b21, c21, z21 = mcnemar(r15['outcomes'], r12['outcomes'], 'j21')
        verdict = "显著正" if z21 > 2 else ("显著负" if z21 < -2 else "噪声")
        print(f"{wname:<14} {r15['j21']:>9.1%} {r12['j21']:>9.1%} {b21:>9} {c21:>9} {z21:>+6.2f}  {verdict}")
        summary.append({'window': wname, 'p15': r15['j21'], 'p12': r12['j21'],
                        'b': b21, 'c': c21, 'z': z21, 'verdict': verdict})

    print()
    print("=" * 78)
    print("跨窗口一致性总表 (2+1 命中率)")
    print("-" * 78)
    print(f"{'窗口':<14} {'gap=0.00':>9} {'gap=0.12':>9} {'gap=0.15':>9}")
    for wname in windows:
        row = all_results[wname]
        print(f"{wname:<14} {row['gap=0.00']['j21']:>9.1%} "
              f"{row['gap=0.12(基线)']['j21']:>9.1%} {row['gap=0.15']['j21']:>9.1%}")

    # 判定结论
    consistent = all(s_['z'] > 0 for s_ in summary)
    significant_all = all(s_['z'] > 2 for s_ in summary)
    print()
    if significant_all:
        print("结论: gap=0.15 三窗口全部显著正 → 稳健信号, 可考虑更新配置")
    elif consistent:
        print("结论: gap=0.15 三窗口方向一致但非全部显著 → 弱信号, 谨慎")
    else:
        print("结论: gap=0.15 跨窗口不一致 → 判定为过拟合/噪声, 不更新配置")

    out = {'date': '2026-08-24', 'mcnemar': summary,
           'windows': {w: {g: {'j21': all_results[w][g]['j21'], 'prize': all_results[w][g]['prize'],
                               'n': all_results[w][g]['n']}
                           for g in all_results[w]} for w in windows}}
    path = 'data/diagnose_dlt_gap_validate_20260824.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")


if __name__ == '__main__':
    main()
