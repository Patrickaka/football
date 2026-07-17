"""
快乐8 选5复式(从选6号码池取号) 覆盖面调优实测。

用户玩法：取选6输出的 N 个号码 -> 打选5复式(C(N,5)注)。
中5(选5一等奖) = 该 N 号码集合包含 >=5 个当期开奖号。

本回测用真实历史做 walk-forward：
  对每期 t，用 t 之前的所有历史生成多注方案，再统计“至少一组含>=5个开奖号”的期数占比。
对比不同 (组数, 每组码数) 配置，给出组合中5率与每期成本。
"""
import sys, json, logging, math
logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')
import src.kl8 as kl8
from src.kl8 import KL8Analyzer

raw = json.load(open('data/kl8_history.json', encoding='utf-8'))['results']
raw = sorted(raw, key=lambda r: r['issue'])  # 升序：oldest..newest
# 转成 (issue, set) 用于快速判断
pairs = [(r['issue'], set(r['numbers'])) for r in raw]


def walkforward_combo_rate(n_slips, pick_size, step=1):
    """返回 (组合中5率, 平均最优组命中数, 覆盖期数)。"""
    combo5 = 0
    best_hits_sum = 0
    total = 0
    for i in range(60, len(pairs), step):  # 前60期热身
        # 历史(至今i-1期)，newest-first 供 analyzer 使用
        hist = [{'issue': pairs[j][0], 'numbers': sorted(pairs[j][1])} for j in range(i)]
        hist = list(reversed(hist))
        winners = pairs[i][1]
        a = KL8Analyzer(history_file=None)
        a.history_data = hist
        a.using_simulated_data = False
        try:
            a.update_statistics()
        except Exception:
            continue
        try:
            slips = kl8.generate_multi_slips(a, 6, n_slips=n_slips, pick_size=pick_size)
        except Exception:
            slips = []
        if not slips:
            continue
        best = max(len(set(s) & winners) for s in slips)
        if best >= 5:
            combo5 += 1
        best_hits_sum += best
        total += 1
    return (combo5 / total if total else 0.0, best_hits_sum / total if total else 0.0, total)


CONFIGS = [
    ('8×6(原基线)', 8, 6),
    ('12×6', 12, 6),
    ('16×6', 16, 6),
    ('20×6', 20, 6),
    ('25×6', 25, 6),
    ('6×7(对照)', 6, 7),
]

print('walk-forward 实测（true walk-forward，每期只用历史；子采样 step=3）：')
print(f'{"配置":<24} {"组合中5率":>10} {"平均最优组命中":>14} {"每期注数":>10} {"每期成本":>10} {"性价比(率/百元)":>16}')
for name, ns, ps in CONFIGS:
    rate, avg_best, total = walkforward_combo_rate(ns, ps, step=3)
    bets = ns * math.comb(ps, 5)  # 选5复式每注=C(ps,5)
    cost = bets * 2
    print(f'{name:<24} {100*rate:>9.2f}% {avg_best:>13.2f} {bets:>10} {cost:>9}元 {100*rate/cost*100:>15.2f}')

print('\n说明：组合中5率 = 该期多注中“至少有一组含>=5个开奖号”的期数占比；')
print('每期注数 = 组数 × C(每组码数,5)，成本按2元/注。')
