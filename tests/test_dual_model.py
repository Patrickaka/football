# 测试双模型预测
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.lottery3d.ml as ml

print("=" * 70)
print("福彩 3D ML 预测 - 双模型测试")
print("=" * 70)

# 获取数据
data = ml.fetch_data()
numbers = [x[2] for x in data]
print(f"\n获取到 {len(numbers)} 期历史数据")
print(f"LightGBM 可用：{ml.HAS_LIGHTGBM}")

# 测试自动模型选择
print("\n测试自动模型选择...")
result = ml.predict_current(numbers, model_type="auto")

if 'error' in result:
    print(f"错误：{result['error']}")
else:
    print(f"\n模型类型：{result['model_type']}")
    print(f"模型信息：{result['model_info']}")
    print(f"训练样本：{result['total_samples']} (正例：{result['pos_samples']}, 负例：{result['neg_samples']})")
    
    print(f"\nTop 3 推荐:")
    for i, rec in enumerate(result['top3'], 1):
        print(f"  {i}. {rec['num']} - 概率 {rec['probability']*100:.2f}%")
    
    print(f"\n特征重要性 Top 5:")
    for i, (name, score) in enumerate(result['feature_importance'][:5], 1):
        if isinstance(score, (int, float)):
            print(f"  {i}. {name} ({score*100:.1f}%)")
        else:
            print(f"  {i}. {name}")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
