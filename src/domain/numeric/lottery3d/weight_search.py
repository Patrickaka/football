"""评分权重的随机搜索：采样、局部扰动、留出测试集。

**这一层不知道怎么评估一组权重。** `evaluate` 由调用方注入——评估要跑一整
轮回测，那是几秒到几分钟的事，而搜索本身只是「采样、比较、保留最优」。
分开之后，测一条「refine 会不会认下更差的解」用一个假的评估函数就够了。

**搜出来的提升多半是过拟合。** 3D 没有可预测性，在同一段历史上试上百组
参数，总有一组看起来更好。留出测试集（`split_series`）不是为了让数字更
好看，是为了让「好看」这件事有机会被推翻。
"""

# 两类可调参数，取值方式完全不同：
#   倍率类 —— 范围是相对基线的**倍数**，采样值是 base * uniform(lo, hi)
#   绝对类 —— 范围就是**取值本身**的上下界，采样值是 uniform(lo, hi)
# 旧实现靠「键名以 _SIGMA 结尾」来区分，那是配置层的命名约定，不该长在
# 算法里。改成由调用方把绝对类的键名传进来。
DEFAULT_ABSOLUTE_KEYS = frozenset()
DEFAULT_RANGE = (0.5, 2.0)
# 倍率类扰动后的下限。权重滑到 0 或负数会让那一项特征彻底失效，
# 而回测照跑不误——只是搜索空间里多了一片没有意义的区域。
MULTIPLIER_FLOOR = 0.1
DEFAULT_MUTATION_SCALE = 0.15

RANDOM_PHASE = 'random'
REFINE_PHASE = 'refine'


def sample_weights(base, tunable, ranges, rng, absolute_keys=DEFAULT_ABSOLUTE_KEYS):
    """在基线附近随机采一组候选参数。**每个键都重采**，不是只动一个。"""
    candidate = {}
    for key in tunable:
        low, high = ranges.get(key, DEFAULT_RANGE)
        if key in absolute_keys:
            candidate[key] = rng.uniform(low, high)
        else:
            candidate[key] = base[key] * rng.uniform(low, high)
    return candidate


def mutate_weights(weights, tunable, ranges, rng,
                   scale=DEFAULT_MUTATION_SCALE, absolute_keys=DEFAULT_ABSOLUTE_KEYS):
    """在当前解附近扰动**一个**键。

    一次只动一个是有意的：动多个的话，分数变好了也说不清是哪一项带来的，
    而 refine 阶段的意义正是「沿着单个方向往下探」。

    **签名比迁移前少了 `base`**：旧实现要一个 `base` 参数，函数体里从头到尾
    没用过它，而调用方在认真地传。倍率类的钳制只有一个下限，没有拿 `base`
    去把结果收回 `ranges` 划的区间里——所以 refine 能把权重推到采样阶段
    根本到不了的地方。这里保持这个行为（改掉会动线上搜索结果），
    但把那个从未生效的参数删掉。
    """
    candidate = dict(weights)
    key = rng.choice(tunable)
    low, high = ranges.get(key, DEFAULT_RANGE)
    if key in absolute_keys:
        delta = (high - low) * scale * rng.uniform(-1, 1)
        candidate[key] = max(low, min(high, candidate[key] + delta))
    else:
        candidate[key] = max(MULTIPLIER_FLOOR,
                             candidate[key] * (1 + scale * rng.uniform(-1, 1)))
    return candidate


def split_series(numbers, test_ratio):
    """按时间切成 (训练段, 测试段)。**测试段永远在后面**——3D 是时序数据，
    随机切分会把「未来」漏进搜索里，而那样搜出来的参数在实盘一文不值。
    """
    train_size = int(len(numbers) * (1 - test_ratio))
    return numbers[:train_size], numbers[train_size:]


def search(base, tunable, ranges, evaluate, rng,
           iterations, refine_rounds,
           absolute_keys=DEFAULT_ABSOLUTE_KEYS,
           scale=DEFAULT_MUTATION_SCALE, on_improve=None):
    """随机采样 + 局部 refine，返回最优解与全过程。

    `evaluate(weights)` 返回 (分数, 回测详情)。**严格大于才换**：相等就换的话
    会在等值的平台上无谓地漂移，最后报出来的「最优参数」是随机的哪一个。

    基线自己也参与比较——采样了上百组还不如什么都不改，这个结论必须留得住。
    """
    baseline_score, baseline_detail = evaluate(base)
    best = {'weights': dict(base), 'score': baseline_score, 'detail': baseline_detail}
    history = []

    for phase, rounds, propose in (
        (RANDOM_PHASE, iterations,
         lambda: sample_weights(base, tunable, ranges, rng, absolute_keys)),
        (REFINE_PHASE, refine_rounds,
         lambda: mutate_weights(best['weights'], tunable, ranges, rng, scale, absolute_keys)),
    ):
        for index in range(rounds):
            candidate = propose()
            score, detail = evaluate(candidate)
            history.append({'phase': phase, 'score': score, 'weights': candidate})
            if score > best['score']:
                best = {'weights': candidate, 'score': score, 'detail': detail}
                if on_improve is not None:
                    on_improve(phase, index + 1, score, detail)

    return {
        'baseline': {'weights': dict(base), 'score': baseline_score, 'detail': baseline_detail},
        'best': best,
        'improvement': best['score'] - baseline_score,
        'history': history,
    }
