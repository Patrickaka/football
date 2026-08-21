"""
快乐8 "预测-揭晓-复盘" demo（真实历史，非作弊）。

做法：
  1. 选一期目标期 T，只使用 T 之前的所有历史生成预测（真预测，不开天眼）。
  2. 揭晓 T 期真实开奖 20 个号，统计选6主推的6个号命中几个。
  3. 跑 12×7 覆盖面方案（同口径，只用 T 前历史），看最好一组撞了几个。
  4. 复盘：要“中5”需要什么——以及为何“开奖后调整”是事后诸葛亮。

重点：本脚本预测阶段绝不读取 T 及之后的数据。
"""
import sys, json, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')
import src.kl8 as kl8
from src.kl8 import KL8Analyzer

raw = json.load(open('data/kl8_history.json', encoding='utf-8'))['results']
raw = sorted(raw, key=lambda r: r['issue'])  # 升序
pairs = [(r['issue'], set(r['numbers'])) for r in raw]

# 选一个靠后的目标期（保留充足历史，且结果已知）
IDX = len(pairs) - 6
T_issue, T_winners = pairs[IDX]
print('=' * 70)
print(f'目标期(预测对象): {T_issue}   真实开奖20个号:')
print('  ', sorted(T_winners))
print('=' * 70)

# —— 只用 T 之前的历史（真预测）——
hist = list(reversed([raw[j] for j in range(IDX)]))  # newest-first 供 analyzer，保留完整字段
a = KL8Analyzer(history_file=None)
a.history_data = hist
a.using_simulated_data = False
a.update_statistics()

res = a.predict_all()
s6 = res.get('select_6', {})
main6 = s6.get('numbers', [])
slips = s6.get('multi_slips', [])

print('\n【1】选6主推 6 个号（只用历史预测）:')
print('   ', main6)
main_hit = len(set(main6) & T_winners)
print(f'   → 命中当期开奖号: {main_hit} 个  (理论期望 6×0.25=1.5)')
print(f'   → 命中的号: {sorted(set(main6) & T_winners)}')

print('\n【2】覆盖面方案 12组×7码（同口径预测）最好的一组:')
best = max(slips, key=lambda s: len(set(s) & T_winners)) if slips else []
best_hit = len(set(best) & T_winners)
for i, s in enumerate(slips):
    h = len(set(s) & T_winners)
    mark = '  ← 最佳' if s == best else ''
    print(f'   第{i+1:>2}组 命中{h}个: {s}{mark}')
print(f'\n   12组里最好一组命中 {best_hit} 个。')

print('\n【3】复盘：要“中5”需要什么？')
print(f'   当期20个开奖号中，选6主推的6个只覆盖到 {main_hit} 个。')
print(f'   要让6个号里含5个开奖号，必须恰好押中这20个里的5个 —— 单注概率仅≈0.32%/期。')
print(f'   覆盖面方案把“某组撞上5+”的概率抬到约9.62%/期（历史回测），但单期里多数时候最好组只中3~4个。')
print(f'   本期最佳组命中 {best_hit} 个，属于“非9.62%命中案例”的正常波动。')

print('\n【4】重要提醒')
print('   开奖后你当然能挑5个当期开奖号说“中5”——但那是事后诸葛亮，')
print('   不能用于预测未来。真正可复用的只有“多组覆盖”这个物理杠杆。')
