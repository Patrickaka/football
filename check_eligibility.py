import sys
sys.path.insert(0, 'src')

import src.football.ml as ml_module
ml_module.load_trained_ml_model()
test_samples = ml_module._trained_ml_metadata.get('test_count', 0) if ml_module._trained_ml_metadata else 0

from src.football.result_sync import get_history, check_ml_fusion_eligibility
history = get_history()
stats = history.get_ml_evaluation_stats()
eligibility = check_ml_fusion_eligibility(stats, test_samples)

print('合格:', eligibility['eligible'])
print('影子样本:', eligibility['shadow_samples'])
print('测试集样本:', eligibility['test_set_samples'])
print()
print('各条件状态:')
for k, v in eligibility['conditions'].items():
    status = '✅' if v['passed'] else '❌'
    print(f'  {k}: {status} - {v["reason"]}')
