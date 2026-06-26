"""检查各玩法预测状态"""
import sys, json
sys.path.insert(0, r'D:\devcode\pythoncode\football')
from src.kl8 import get_kl8_analyzer

a = get_kl8_analyzer()
r = a.predict_all()

keys = ['select_3', 'select_4', 'select_5', 'select_6', 'select_7', 'fu_shi_7']
for k in keys:
    d = r.get(k, {})
    mode = d.get('prediction_mode', '?')
    status = d.get('status', '?')
    warning = d.get('warning', 'none')
    nums = d.get('numbers', [])
    print(f'{k}: mode={mode}, status={status}, warning={warning}, numbers={nums}')

print(f'\nsignal_status: {r.get("statistics", {}).get("signal_status")}')
print(f'backfill: {r.get("statistics", {}).get("backfill_progress")}')
