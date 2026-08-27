"""策略准入：够不够格进实盘融合。

**默认是不够格。** 每一项都要单独达标，`all(...)` 一票否决——这不是保守，
是因为 3D 没有可预测性，一个「看起来不错」的策略绝大多数时候是过拟合。
让它进实盘的成本是真金白银，拦下来的成本只是继续回测。

每项都同时报出**实际值与门槛**，而不只是过没过：过不了的时候，人需要看到
差多少才知道是接近了还是差得远。
"""

# 理论基准命中率：Top30 覆盖 1000 注里的 30 注。
# **用固定值而不是某一次随机回测的结果**——单次抽样的波动能轻易把门槛
# 抬高或压低几个百分点，那样「达标」就变成了跟哪一次随机数比。
THEORY_BASELINE = 0.03
# 平均真实排名的上限。1000 注随机排的期望是 500，所以这条等于「不比随机差」。
MAX_AVERAGE_RANK = 500
# 置换检验的显著性门槛。0.10 而不是 0.05：样本只有一百来期，
# 卡到 0.05 会把所有策略都挡在门外，那这项检查就没有意义了。
MAX_PVALUE = 0.10


def _check(passed, actual, required, reason):
    return {'passed': passed, 'actual': actual, 'required': required, 'reason': reason}


def evaluate(served_rate, raw_rate, average_rank, baseline=None, significance=None):
    """逐项检查，全部通过才算够格。

    `significance` 是置换检验的返回字典；**缺 `pvalue` 时按 1.0 处理**，
    也就是不通过——没有证据不等于有利证据。
    """
    baseline = THEORY_BASELINE if baseline is None else baseline
    checks = {
        'served_top30_last100_above_baseline': _check(
            served_rate >= baseline, round(served_rate, 4), round(baseline, 4),
            f'近100期 served Top30 不低于理论基准({baseline * 100:.1f}%)'),
        'raw_top30_last100_above_baseline': _check(
            raw_rate >= baseline, round(raw_rate, 4), round(baseline, 4),
            f'近100期 raw Top30 不低于理论基准({baseline * 100:.1f}%)'),
        'avg_rank_below_500': _check(
            average_rank < MAX_AVERAGE_RANK, average_rank, MAX_AVERAGE_RANK,
            '平均真实号码排名 < 500'),
    }
    if significance is not None:
        checks['permutation_significant'] = _check(
            significance.get('pvalue', 1.0) < MAX_PVALUE,
            significance.get('pvalue'), MAX_PVALUE,
            '置换检验 p 值 < 0.10')
    return {'eligible': all(item['passed'] for item in checks.values()),
            'checks': checks}
