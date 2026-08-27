"""窗口权重：哪个回看窗口近期更靠谱，就给它多少票。

短窗口跟得紧但噪声大，长窗口稳但迟钝，而「哪个更准」本身随时间变。这里
不去赌一个窗口，而是**逐期回测每个窗口单独的表现**，按表现分配权重。

**这一层不碰缓存、不碰持久化、不读时钟。** 算一次权重要跑几十期回测，
所以外面一定会缓存它——但缓存该在外面。把 `time.time()` 和 kv 读写混进来，
这个函数就再也没法在测试里稳定复现了。

**这不是在预测。** 权重回答的是「哪个窗口最近描述得更准」，不是「下期会
开什么」。全部窗口表现一样时权重就退化成均分，那才是常态。
"""

# 每个窗口的先验票数。没有它，一次零命中的回测会把某个窗口的权重压到 0，
# 而它只是运气差——先验的意义是「别让几十期的样本量下结论下得太死」。
# 命中一注计 1 分，两位重合计 0.25 分，所以先验取在同一量纲上。
FULL_HIT_SCORE = 1.0
PARTIAL_HIT_SCORE = 0.25
# 判定「部分命中」需要重合几个数字
PARTIAL_HIT_DIGITS = 2

MIN_TRIALS = 10
# 回测至少要留出的余量：最长窗口之外还得有几期能当验证集
TRAIN_MARGIN = 10
TRIAL_MARGIN = 5


def default_weights(windows):
    """均分。**没有证据时就该均分**，而不是偏向某一个窗口。"""
    return {window: 1.0 / len(windows) for window in windows}


def has_enough_history(numbers, windows):
    return len(numbers) >= max(windows) + TRAIN_MARGIN


def trial_count(numbers, windows, requested):
    """能跑几期回测。至少 `MIN_TRIALS`，否则算出来的只是噪声。"""
    return max(MIN_TRIALS,
               min(requested, len(numbers) - max(windows) - TRIAL_MARGIN))


def score_windows(numbers, windows, trials, predict, actual_label):
    """逐期回测每个窗口，返回各自的原始得分。

    **每一期只用它之前的数据**——`predict(train, window)` 拿到的是 `numbers[:i]`，
    当期开奖绝不进训练集。这条泄漏一旦破了，所有窗口都会「表现很好」，
    权重就退化成随机。

    `predict` 返回该窗口给出的推荐号码列表；`actual_label` 把开奖转成同样
    的形式。两者都由调用方给：怎么预测是上面几层的事。
    """
    raw = {window: 0.0 for window in windows}
    start = len(numbers) - trials
    for index in range(start, len(numbers)):
        train = numbers[:index]
        actual = actual_label(numbers[index])
        for window in windows:
            if len(train) < window:
                continue
            raw[window] += _period_score(actual, predict(train, window))
    return raw


def _period_score(actual, predicted):
    """一期的得分：中了给满分，只重合两位给部分分，其余为 0。

    **部分分不是安慰奖**：三位全中是 1/1000 的事，只按全中打分的话几十期
    回测里绝大多数窗口都是 0 分，权重就无从区分了。
    """
    if actual in predicted:
        return FULL_HIT_SCORE
    if _max_overlap(actual, predicted) >= PARTIAL_HIT_DIGITS:
        return PARTIAL_HIT_SCORE
    return 0.0


def _max_overlap(actual, predicted):
    from src.domain.numeric.lottery3d.recommendations import max_digit_overlap
    return max_digit_overlap(actual, predicted)


def normalise(raw, windows, prior):
    """原始得分加上先验后归一化成权重。"""
    total = sum(raw[window] + prior for window in windows)
    return {window: (raw[window] + prior) / total for window in windows}
