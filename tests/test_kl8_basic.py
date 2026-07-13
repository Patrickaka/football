"""快乐8模块基本功能测试"""
import sys
sys.path.insert(0, 'src')

from src.kl8 import get_kl8_analyzer, run_prediction, KL8_PREDICTOR_VERSION, SELECT_TYPES, KL8Analyzer

def main():
    print(f"版本: {KL8_PREDICTOR_VERSION}")

    # 获取分析器
    analyzer = get_kl8_analyzer()
    print(f"数据量: {len(analyzer.history_data)} 期")
    print(f"使用模拟数据: {analyzer.using_simulated_data}")

    # 运行预测；基础测试不写快照，避免产生测试副作用。
    original_save = KL8Analyzer._save_prediction_snapshot
    try:
        KL8Analyzer._save_prediction_snapshot = lambda self, prediction_result: None
        result = run_prediction(force_refresh=True)
    finally:
        KL8Analyzer._save_prediction_snapshot = original_save

    # 打印各选型结果
    for st in SELECT_TYPES:
        key = f'select_{st}'
        sel = result[key]
        print(f"\n选{st}: {sel['numbers']}")

    # 选5复式
    fushi = result['fu_shi_7']
    print(f"\n选5复式7码:")
    print(f"  Top7号码: {fushi['top7_numbers']}")
    print(f"  组合数量: {fushi['total_combinations']}")
    print(f"  前5组示例:")
    for i, combo in enumerate(fushi['combinations'][:5]):
        print(f"    组{i+1}: {combo}")

    # 最近开奖
    if result['recent_results']:
        print(f"\n最近开奖(第1期): {result['recent_results'][0]['numbers']}")

    # 统计信息
    stats = result['statistics']
    print(f"\n统计信息:")
    print(f"  期数: {stats['total_periods']}")
    print(f"  期望频率: {stats['expected_freq']}")
    print(f"  期望遗漏: {stats['expected_gap']}")

    # 排名Top10
    print(f"\n排名Top10:")
    for item in result['ranking'][:10]:
        score = item.get('ranking_score', item.get('score', 0))
        print(f"  号码{item['num']:02d}: 得分={score:.4f}")

    print("\n[OK] 快乐8模块基本功能测试通过!")

if __name__ == '__main__':
    main()
