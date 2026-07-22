"""快乐8 选五/选六 多注复式 覆盖回测（walk-forward，真实历史）

目的：量化"昨日6码/5码全军覆没"是否是统计预期，并比较不同组数配置下
"至少一组中N"的真实命中率与成本，为调整提供依据。

方法：对每个目标期，只用它之前的真实历史生成多注复式（与线上完全一致），
再用该期实开奖号打分；滚动遍历最近 N 期取平均。
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kl8 import KL8Analyzer, generate_multi_slips

WARMUP = 60          # 前 60 期不评（保证历史窗口充足）
TAIL = 200           # 评最近 200 期
NSLIPS = [8, 12, 16]
PICK = 6             # 每组 6 码


def best_match(slips, draw):
    ds = set(draw)
    return max(len(set(s) & ds) for s in slips) if slips else 0


def evaluate(select_n):
    a = KL8Analyzer()
    hist = a.history_data  # 降序：hist[0] 最新
    n = len(hist)
    start = WARMUP
    end = min(n, TAIL + WARMUP)
    agg = {N: defaultdict(int) for N in NSLIPS}
    total = 0
    for p in range(start, end):
        target = hist[p]
        draw = target['numbers']
        # 只用 target 之前的数据（降序中 hist[0:p] 比 target 更新）
        a.history_data = hist[0:p]
        for N in NSLIPS:
            slips = generate_multi_slips(a, select_n, n_slips=N, pick_size=PICK)
            bm = best_match(slips, draw)
            # 选五：看中3/4/5；选六：看中4/5/6
            floors = (3, 4, 5) if select_n == 5 else (4, 5, 6)
            for f in floors:
                if bm >= f:
                    agg[N][f] += 1
        total += 1
    a.history_data = hist
    return agg, total


def main():
    t0 = time.time()
    for select_n in (5, 6):
        agg, total = evaluate(select_n)
        print(f"\n===== 选{select_n} 多注复式（每组{PICK}码，walk-forward {total}期）=====")
        label = "中3+ 中4+ 中5" if select_n == 5 else "中4+ 中5 中6"
        hdr = f"{'组数':>5}{'成本(元/期)':>12}" + "".join(f"{l:>9}" for l in label.split())
        print(hdr)
        combos = (6, 5, 1)[0] if select_n == 5 else 1  # 选五每组C(6,5)=6注, 选六每组1注
        for N in NSLIPS:
            cost = N * combos * 2
            rates = []
            for f in ((3, 4, 5) if select_n == 5 else (4, 5, 6)):
                rates.append(f"{100.0*agg[N][f]/total:6.2f}%")
            print(f"{N:>5}{cost:>12}" + "".join(f"{r:>9}" for r in rates))
    print(f"\n耗时 {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
