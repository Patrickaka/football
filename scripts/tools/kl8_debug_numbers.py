"""检查快乐8号码是否真的每天相同"""
import os
import sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.kl8 import get_kl8_analyzer

a = get_kl8_analyzer()

# 检查最新几期开奖
print("=== 最近5期开奖 ===")
for i in range(min(5, len(a.history_data))):
    r = a.history_data[i]
    print(f"  {r.get('issue')}: {r.get('numbers')}")

# 250期窗口内频率Top10(冷号得分最高)
print("\n=== 250期窗口 频率最低的10个号(冷号,被优先推荐) ===")
recent = a.history_data[:250]
from collections import Counter
freq = Counter()
for rec in recent:
    for n in rec['numbers']:
        freq[n] += 1
expected = 250 * 20 / 80  # 62.5
sorted_freq = sorted(freq.items(), key=lambda x: (x[1], x[0]))
for num, count in sorted_freq[:10]:
    print(f"  号码{num:02d}: 出现{count}次 (期望{expected:.1f}, 偏离{(count/expected-1)*100:+.1f}%)")

print(f"\n=== 最新期号: {a.history_data[0]['issue']} ===")
print(f"总历史: {len(a.history_data)}期")
