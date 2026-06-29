"""
快乐8滚动回测脚本
=================
严格离线回测：每一期预测时只能使用该期之前的历史数据，
不能用未来数据调权重。

核心指标按玩法分别统计：
- 平均命中数
- 命中 ≥1、≥2、≥3 的概率
- 各中奖档命中率
- 理论随机基线
- 相对基线提升 Lift
- 95% 置信区间
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kl8 import (
    KL8Analyzer, KL8_PREDICTOR_VERSION,
    KL8_NUM_RANGE, KL8_DRAW_COUNT,
    FEATURE_WEIGHTS, MODEL_WEIGHTS,
    SELECT_TYPES,
    normalize_record,
)
from src.common.paths import data_path


# ─── 理论基线计算 ───

def theoretical_baseline(pick_n: int) -> dict:
    """计算各玩法的理论随机基线

    快乐8: 80选20, 玩家选pick_n个号码
    每个号码被开出的概率 p = 20/80 = 0.25
    命中数服从超几何分布 Hypergeometric(N=80, K=20, n=pick_n)

    E[命中] = pick_n * 20/80 = pick_n * 0.25
    """
    N = KL8_NUM_RANGE  # 80
    K = KL8_DRAW_COUNT  # 20
    n = pick_n

    p = K / N  # = 0.25

    # 期望命中数
    expected_hits = n * p

    # P(命中=k) = C(K,k)*C(N-K,n-k) / C(N,n)
    def hypergeom_pmf(k):
        if k > min(K, n) or k < max(0, n - N + K):
            return 0.0
        from math import comb
        return comb(K, k) * comb(N - K, n - k) / comb(N, n)

    # P(命中>=k) = sum of P(命中=j) for j >= k
    def hypergeom_cdf_inv(k):
        return sum(hypergeom_pmf(j) for j in range(k, min(K, n) + 1))

    return {
        'pick_n': n,
        'expected_hits': expected_hits,
        'p_ge1': hypergeom_cdf_inv(1),
        'p_ge2': hypergeom_cdf_inv(2),
        'p_ge3': hypergeom_cdf_inv(3),
        'p_ge4': hypergeom_cdf_inv(4),
        'p_ge5': hypergeom_cdf_inv(5),
        'pmf': {k: hypergeom_pmf(k) for k in range(min(K, n) + 1)},
    }


# ─── 置信区间 ───

def confidence_interval(values: list, confidence: float = 0.95) -> dict:
    """计算均值置信区间"""
    n = len(values)
    if n < 2:
        return {'mean': values[0] if values else 0, 'ci_low': 0, 'ci_high': 0, 'n': n}

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std_err = math.sqrt(variance / n)

    # 95% CI using t-distribution approximation (for n>30, z≈1.96)
    if n > 30:
        z = 1.96
    else:
        # Approximate t-value for small samples
        z = 2.0  # rough approximation

    ci_low = mean - z * std_err
    ci_high = mean + z * std_err

    return {
        'mean': mean,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'std_err': std_err,
        'n': n,
    }


# ─── 滚动回测 ───

def rolling_backtest(
    history_data: list,
    start_offset: int = 50,   # 最少用50期历史开始预测
    max_trials: int = None,    # None = 尽可能多
):
    """严格滚动回测

    第 start_offset+1 期预测时，只能用前 start_offset 期数据；
    第 start_offset+2 期预测时，只能用前 start_offset+1 期数据；
    严格离线，不使用未来数据。
    """
    n_total = len(history_data)
    if n_total < start_offset + 1:
        print(f'数据不足: 需要{start_offset + 1}期，实际{n_total}期')
        return None

    # history_data: 最新在前 → 需要反转
    # 回测从旧到新: history_data[n-1]是最旧的，history_data[0]是最新的
    data_oldest_first = list(reversed(history_data))

    max_trials = max_trials or (n_total - start_offset)
    trials = min(max_trials, n_total - start_offset)

    print(f'=== 快乐8滚动回测 ===')
    print(f'版本: {KL8_PREDICTOR_VERSION}')
    print(f'总数据: {n_total}期, 回测范围: 第{start_offset+1}期 ~ 第{start_offset+trials}期')
    print(f'回测期数: {trials}')
    print(f'特征权重: {FEATURE_WEIGHTS}')
    print(f'模型权重: {MODEL_WEIGHTS}')
    print()

    # 每个play_type的回测结果
    results_by_type = defaultdict(lambda: {
        'hit_counts': [],      # 每期命中数
        'actual_numbers': [],  # 每期实际开奖号码
        'predicted_numbers': [],  # 每期预测号码
        'issues': [],
    })

    for trial_idx in range(trials):
        # 预测期 = 第 (start_offset + trial_idx) 期 (oldest-first index)
        predict_idx = start_offset + trial_idx
        actual = set(data_oldest_first[predict_idx]['numbers'])
        issue = data_oldest_first[predict_idx]['issue']

        # 构造训练数据: 只用predict_idx之前的期数（最新在前格式）
        train_data = list(reversed(data_oldest_first[:predict_idx]))

        # 创建analyzer（只用历史数据）
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_file = ''
        analyzer.using_simulated_data = False
        analyzer.history_data = train_data
        analyzer.statistics = {}
        analyzer.update_statistics()

        # 各玩法预测
        for pick_n in SELECT_TYPES:
            ranking = analyzer.get_ensemble_ranking(top_n=pick_n)
            predicted = set(r['num'] for r in ranking[:pick_n])
            hits = len(actual & predicted)

            results_by_type[pick_n]['hit_counts'].append(hits)
            results_by_type[pick_n]['actual_numbers'].append(sorted(actual))
            results_by_type[pick_n]['predicted_numbers'].append(sorted(predicted))
            results_by_type[pick_n]['issues'].append(issue)

        # 选5复式7码
        ranking7 = analyzer.get_ensemble_ranking(top_n=7)
        top7 = [r['num'] for r in ranking7[:7]]
        top7_set = set(top7)

        results_by_type['fushi5_7']['hit_counts'].append(len(actual & top7_set))
        results_by_type['fushi5_7']['actual_numbers'].append(sorted(actual))
        results_by_type['fushi5_7']['predicted_numbers'].append(top7)
        results_by_type['fushi5_7']['issues'].append(issue)

        if (trial_idx + 1) % 20 == 0 or trial_idx == trials - 1:
            print(f'  进度: {trial_idx + 1}/{trials}')

    # ─── 统计汇总 ───

    print()
    print('=' * 70)
    print('回测统计汇总')
    print('=' * 70)

    baselines = {}
    for pick_n in SELECT_TYPES:
        baselines[pick_n] = theoretical_baseline(pick_n)

    # 选5复式7码: 7个号码命中>=5的概率基线
    N, K = 80, 20
    from math import comb
    fushi_baseline_ge1 = sum(
        comb(K, j) * comb(N - K, 7 - j) / comb(N, 7)
        for j in range(1, min(K, 7) + 1)
    )
    fushi_baseline_expected = 7 * 0.25  # = 1.75

    for play_type in [*SELECT_TYPES, 'fushi5_7']:
        r = results_by_type[play_type]
        hits = r['hit_counts']
        n_trials = len(hits)

        if play_type == 'fushi5_7':
            bl_expected = fushi_baseline_expected
            bl_ge1 = fushi_baseline_ge1
            desc = '选5复式7码'
        else:
            bl = baselines[play_type]
            bl_expected = bl['expected_hits']
            bl_ge1 = bl['p_ge1']
            desc = f'选{play_type}'

        mean_hits = sum(hits) / n_trials
        ge1_count = sum(1 for h in hits if h >= 1)
        ge2_count = sum(1 for h in hits if h >= 2)
        ge3_count = sum(1 for h in hits if h >= 3)
        ge4_count = sum(1 for h in hits if h >= 4)
        ge5_count = sum(1 for h in hits if h >= 5)

        ge1_rate = ge1_count / n_trials
        ge2_rate = ge2_count / n_trials
        ge3_rate = ge3_count / n_trials

        ci = confidence_interval(hits)

        # Lift = (实际 - 基线) / 基线
        lift_expected = (mean_hits - bl_expected) / bl_expected if bl_expected > 0 else 0
        lift_ge1 = (ge1_rate - bl_ge1) / bl_ge1 if bl_ge1 > 0 else 0

        print(f'\n--- {desc} ({n_trials}期) ---')
        print(f'  平均命中: {mean_hits:.3f} (基线={bl_expected:.3f}, Lift={lift_expected:+.1%})')
        print(f'  95% CI: [{ci["ci_low"]:.3f}, {ci["ci_high"]:.3f}]')
        print(f'  ≥1命中: {ge1_rate:.3f} (基线={bl_ge1:.3f}, Lift={lift_ge1:+.1%})')
        print(f'  ≥2命中: {ge2_rate:.3f} ({ge2_count}/{n_trials})')
        print(f'  ≥3命中: {ge3_rate:.3f} ({ge3_count}/{n_trials})')
        if play_type != 'fushi5_7':
            bl = baselines[play_type]
            print(f'  ≥4命中: {ge4_count/n_trials:.3f} ({ge4_count}/{n_trials}, 基线={bl["p_ge4"]:.3f})')
            print(f'  ≥5命中: {ge5_count/n_trials:.3f} ({ge5_count}/{n_trials}, 基线={bl["p_ge5"]:.3f})')

        # 判断是否显著超过基线
        if ci['ci_low'] > bl_expected:
            print(f'  [OK] 平均命中显著超过基线 (CI下界>{bl_expected:.3f})')
        elif ci['ci_high'] < bl_expected:
            print(f'  [NO] 平均命中显著低于基线')
        else:
            print(f'  [--] 平均命中与基线无显著差异')

    return results_by_type


def load_data():
    """加载历史数据"""
    # 尝试从文件加载
    path = Path(data_path('kl8_history.json'))
    if path.exists():
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            data = raw.get('results', raw.get('data', []))
        else:
            data = raw

        validated = []
        for r in data:
            normed = normalize_record(r)
            if normed:
                validated.append(normed)

        validated.sort(key=lambda x: x['issue'], reverse=True)
        return validated

    return []


if __name__ == '__main__':
    data = load_data()
    if len(data) < 51:
        print(f'数据不足: {len(data)}期，至少需要51期')
        print('请先通过API抓取数据: curl http://localhost:9004/api/kl8/fetch')
        sys.exit(1)

    print(f'加载了{len(data)}期有效历史数据')
    print(f'最新期: {data[0]["issue"]} ({data[0]["date"]})')
    print(f'最旧期: {data[-1]["issue"]} ({data[-1]["date"]})')
    print()

    # 打印理论基线
    print('=== 理论随机基线 ===')
    for pick_n in SELECT_TYPES:
        bl = theoretical_baseline(pick_n)
        print(f'  选{pick_n}: 平均命中={bl["expected_hits"]:.2f}, ≥1={bl["p_ge1"]:.4f}, ≥2={bl["p_ge2"]:.4f}, ≥3={bl["p_ge3"]:.4f}')
    print()

    results = rolling_backtest(data, start_offset=50)
