import sys
sys.path.insert(0, 'src')

from src.football.result_sync import get_history

history = get_history()
records = history.records

print(f"总记录数: {len(records)}")
print()

# 检查前几条记录的结构
for i, record in enumerate(records[:3]):
    print(f"=== 记录 {i+1} ===")
    print(f"有 ml_1x2: {'ml_1x2' in record}")
    print(f"有 actual_result: {'actual_result' in record}")
    print(f"有 evaluation: {'evaluation' in record}")
    
    if 'evaluation' in record:
        eval_keys = list(record['evaluation'].keys())
        print(f"evaluation 包含: {eval_keys}")
        
        # 检查关键指标
        for metric in ['base_1x2_logloss', 'ml_1x2_logloss', 'base_1x2_brier', 'ml_1x2_brier']:
            if metric in record['evaluation']:
                print(f"  {metric}: {record['evaluation'][metric]}")
            else:
                print(f"  {metric}: 缺失")
    print()
