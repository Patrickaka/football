# 测试 ML API
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.lottery3d.ml as lottery3d_ml

print("测试 ML 预测...")

# 获取数据
data = lottery3d_ml.fetch_data()
print(f"获取到 {len(data)} 期数据")

numbers = [x[2] for x in data]
print(f"号码数据: {len(numbers)} 个")

# 运行预测
print("开始预测...")
result = lottery3d_ml.predict_current(numbers)

if 'error' in result:
    print(f"错误: {result['error']}")
else:
    print(f"预测成功!")
    print(f"训练样本: {result['total_samples']} (正例: {result['pos_samples']}, 负例: {result['neg_samples']})")
    print(f"推荐数量: {len(result['recommendations'])}")
    print(f"Top 3:")
    for i, rec in enumerate(result['top3'], 1):
        print(f"  {i}. {rec['num']} - {rec['probability']*100:.2f}%")