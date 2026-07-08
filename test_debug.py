from src.beidan import fetch_okooo_schedule, generate_beidan_recommendations
import json

print("=== 测试okooo赛程抓取 ===")
matches = fetch_okooo_schedule('2026-07-08')
print(f"抓取到 {len(matches)} 场比赛")

for m in matches:
    print(f"\n比赛 {m.get('num', 'N/A')}:")
    print(f"  ID: {m.get('id')}")
    print(f"  日期: {m.get('date')}")
    print(f"  时间: {m.get('time')}")
    print(f"  联赛: {m.get('league')}")
    print(f"  主队: {m.get('home')}")
    print(f"  客队: {m.get('away')}")
    print(f"  让球: {m.get('handicap')}")
    print(f"  状态: {m.get('status')}")

print("\n\n=== 测试完整推荐 ===")
result = generate_beidan_recommendations(date='2026-07-08', source='okooo', bet_types=['spf', 'zjq'])
print(json.dumps(result, ensure_ascii=False, indent=2))
