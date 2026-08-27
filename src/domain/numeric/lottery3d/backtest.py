"""福彩 3D 的滚动回测：命中判定与统计聚合。

**这一层不知道怎么预测。** 每期的排名池由调用方算好喂进来——出一次号要读
配置、要走完评分与选号的整条链路，而回测本身只做三件事：切出滚动窗口、
判断实际开奖落在哪些池子里、把计数聚合成比率。分开之后，测一条「Top30
命中怎么算」不必先造一台评分器。

**回测不会让号码更准。** 直选恒定 1/1000，这里每个比率回答的都只是
「这套排序把真号排得靠前吗」，而线上实测的平均排名是 558——比随机期望的
500 还差。读这些数字时别默认模型有优势。

**训练集只能是这一期之前的历史。** 回测里最容易犯、又最不会报错的错是让
模型看见未来：命中率会好得不真实，而每一行代码看起来都很正常。
`rolling_slices` 存在的意义就是把这件事收进一处。
"""
import random

from src.domain.numeric.lottery3d import draw as _draw
from src.domain.numeric.lottery3d import recommendations as _recommend
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS

# 实际号码没出现在排名里时记的名次。全部 1000 注排完也只到 1000，
# 记 1001 是为了让「没排进来」在平均值里比「排最后」更差一点，且可区分。
MISS_RANK = DIGIT_SPACE.size ** POSITIONS + 1
TOP100_SIZE = 100

# 回测前要留出的训练余量：最长窗口 + 几期缓冲。第一期回测也得有完整的窗口
# 可看，否则前几期算出来的分和后面的不是同一个量。
TRIALS_GUARD = 5
# 压缩后的下限。低于这个期数，命中率的抽样噪声比信号大得多。
MIN_TRIALS = 20

# 「至少中两个数字」的门槛。3D 一注只有三位，中两位已经能落到组选奖里。
GE2_DIGITS = 2


def resolve_trials(total, requested, max_window,
                   guard=TRIALS_GUARD, floor=MIN_TRIALS):
    """历史不够长时把回测期数压回去。

    **压缩后可能仍然大于总期数**（`floor` 是硬下限），所以 `rolling_slices`
    还得自己夹一次——两处都留着不是重复：这里保证「不为了凑期数牺牲训练窗口」，
    那里保证「不会切出负索引」。
    """
    if total >= requested + max_window + guard:
        return requested
    return max(floor, total - max_window - guard)


def rolling_slices(numbers, trials):
    """产出 (train, actual)：每期只看它之前的历史。

    起点夹到 0。不夹的话 `len(numbers) - trials` 为负时 `range` 会从负数开始，
    `numbers[:-3]` 这样的切片照样能取到东西，于是实际跑的期数比 `trials` 多、
    而且前几期的 train 与 actual 对不上——**全程不报错**。
    """
    start = max(0, len(numbers) - trials)
    for index in range(start, len(numbers)):
        yield numbers[:index], numbers[index]


def as_key(triple):
    """三元组写成 '385' 这样的字符串。排名池里一律用这个形式比对。"""
    return ''.join(str(digit) for digit in triple)


def rank_of(key, ranked, miss_rank=MISS_RANK):
    """实际号码在排名里的名次，从 1 开始。不在里面记 `miss_rank`。"""
    for index, num in enumerate(ranked):
        if num == key:
            return index + 1
    return miss_rank


class RankingBacktest:
    """滚动回测的累加器：每期喂一次观测，最后汇总。

    **raw 与 served 两条 Top30 分开统计**：raw 是纯排序能力，served 是实盘
    真正推出去的那一池（加了多样性、去相关）。混成一个数字之后，命中率掉了
    就分不清是模型退步还是那些后处理伤了它。
    """

    def __init__(self, top3_size, recommend_size,
                 zu6_four_size, zu6_pool_size,
                 rng=None, top100_size=TOP100_SIZE, miss_rank=MISS_RANK):
        self.top3_size = top3_size
        self.recommend_size = recommend_size
        self.zu6_four_size = zu6_four_size
        self.zu6_pool_size = zu6_pool_size
        self.top100_size = top100_size
        self.miss_rank = miss_rank
        # 组六随机对照用的随机源。**跨期连续**，不是每期重开一个——每期重置
        # 会让「随机」变成同一组数字抽很多遍，方差被压掉，对照就失真了。
        self.rng = rng if rng is not None else random.Random()

        self.actual_ranks = []
        self.raw_hits = []
        self.served_hits = []
        self.hit_top3 = 0
        self.hit_top100 = 0
        self.hit_ge2 = 0
        self.zu6_draws = 0
        self.zu6_four_hit = 0
        self.zu6_pool_hit = 0
        self.zu6_ge2_hit = 0
        self.random_zu6_four_hit = 0
        self.random_zu6_pool_hit = 0

    @property
    def trials(self):
        """**实际评估的期数，不是请求的期数。** 比率的分母只能是这个——
        用请求值当分母，短序列上算出来的命中率会被系统性低估。"""
        return len(self.actual_ranks)

    def observe(self, actual, ranked, top3, raw_top, served_top):
        """记一期直选结果。`ranked` 是全部候选的排名（名次即下标）。"""
        key = as_key(actual)
        self.actual_ranks.append(rank_of(key, ranked, self.miss_rank))

        raw_hit = key in raw_top
        served_hit = key in served_top
        self.raw_hits.append(1 if raw_hit else 0)
        self.served_hits.append(1 if served_hit else 0)

        if key in top3:
            self.hit_top3 += 1
        if key in ranked[:self.top100_size]:
            self.hit_top100 += 1
        if _recommend.max_digit_overlap(key, served_top) >= GE2_DIGITS:
            self.hit_ge2 += 1

    def observe_zu6(self, actual, four, pool):
        """记一期组六四码/五码结果。**非组六期不该调这个**——调用方按形态判。

        随机对照在这里抽而不是让调用方传：它必须与真实池子在同一期、用同一个
        随机源抽，否则两边的样本量对不上，比出来的差值没有意义。
        """
        digits = set(actual)
        self.zu6_draws += 1
        if digits <= set(four):
            self.zu6_four_hit += 1
        if digits <= set(pool):
            self.zu6_pool_hit += 1
        if len(digits & set(four)) >= GE2_DIGITS:
            self.zu6_ge2_hit += 1

        universe = list(DIGIT_SPACE.numbers())
        if digits <= set(self.rng.sample(universe, self.zu6_four_size)):
            self.random_zu6_four_hit += 1
        if digits <= set(self.rng.sample(universe, self.zu6_pool_size)):
            self.random_zu6_pool_hit += 1

    def is_zu6(self, actual):
        """这一期是不是组六。放在这里是为了让调用方不必自己 import draw。"""
        return _draw.classify_form(actual) == _draw.ZU6

    def summarise(self, random_rate, random_hit, admission, last_window=100):
        """汇成对外的那份统计。

        `admission` 与 `random_*` 由调用方传：准入要读门槛、随机对照要跑另一
        轮回测，两者都不属于「这一批观测说明了什么」。
        """
        total = self.trials
        served_rate = self._rate(self.hit_served_top30, total)
        recent = min(last_window, total)
        raw_last = self._rate(sum(self.raw_hits[-recent:]), recent) if recent else 0.0
        served_last = self._rate(sum(self.served_hits[-recent:]), recent) if recent else 0.0

        ranks = sorted(self.actual_ranks)
        rank_avg = sum(self.actual_ranks) / total if total else 0.0
        rank_median = ranks[len(ranks) // 2] if ranks else 0

        return {
            'trials': total,
            'strategy': 'stable_baseline',
            'top3_hit': self.hit_top3,
            'top3_rate': self._rate(self.hit_top3, total),
            'top3_rate_baseline': round(self.top3_size / (DIGIT_SPACE.size ** POSITIONS), 4),
            'raw_top30_hit': self.hit_raw_top30,
            'raw_top30_rate': self._rate(self.hit_raw_top30, total),
            'served_top30_hit': self.hit_served_top30,
            'served_top30_rate': served_rate,
            'raw_top30_last100_rate': round(raw_last, 4),
            'served_top30_last100_rate': round(served_last, 4),
            # 主展示指标取 served——那才是实盘真正推出去的一池
            'top30_hit': self.hit_served_top30,
            'top30_rate': served_rate,
            'top30_rate_baseline': self._baseline_rate(),
            # 兼容旧字段名。页面与已保存的历史记录都还在读这三个
            'top_hit': self.hit_served_top30,
            'top_rate': served_rate,
            'top_rate_baseline': self._baseline_rate(),
            'top100_hit': self.hit_top100,
            'top100_rate': self._rate(self.hit_top100, total),
            'actual_rank_avg': round(rank_avg, 1),
            'actual_rank_median': rank_median,
            'actual_rank_top100_rate': self._rate(
                sum(1 for rank in self.actual_ranks if rank <= self.top100_size), total),
            'actual_rank_top300_rate': self._rate(
                sum(1 for rank in self.actual_ranks if rank <= 300), total),
            'ge2_digit_rate': self._rate(self.hit_ge2, total),
            'zu6_draws': self.zu6_draws,
            'zu6_four_hit': self.zu6_four_hit,
            'zu6_four_rate': self._rate(self.zu6_four_hit, self.zu6_draws),
            'zu6_pool_hit': self.zu6_pool_hit,
            'zu6_pool_rate': self._rate(self.zu6_pool_hit, self.zu6_draws),
            'zu6_ge2_hit': self.zu6_ge2_hit,
            'zu6_ge2_rate': self._rate(self.zu6_ge2_hit, self.zu6_draws),
            'zu6_random_four_rate': self._rate(self.random_zu6_four_hit, self.zu6_draws),
            'zu6_random_pool_rate': self._rate(self.random_zu6_pool_hit, self.zu6_draws),
            'random_rate': round(random_rate, 4),
            'random_hit': random_hit,
            'admission': admission,
        }

    @property
    def hit_raw_top30(self):
        return sum(self.raw_hits)

    @property
    def hit_served_top30(self):
        return sum(self.served_hits)

    def recent_rates(self, last_window=100):
        """近 N 期的 (raw, served) 命中率。准入判定要拿它当输入。"""
        recent = min(last_window, self.trials)
        if not recent:
            return 0.0, 0.0
        return (sum(self.raw_hits[-recent:]) / recent,
                sum(self.served_hits[-recent:]) / recent)

    @property
    def average_rank(self):
        return sum(self.actual_ranks) / self.trials if self.trials else 0.0

    def _baseline_rate(self):
        return round(self.recommend_size / (DIGIT_SPACE.size ** POSITIONS), 4)

    @staticmethod
    def _rate(hit, total):
        return round(hit / total, 4) if total else 0.0


def random_baseline(numbers, trials, top_n, rng):
    """随机对照：每期从全部候选里随机抽 `top_n` 注，看真号中没中。

    **对照必须跟被测的回测跑在同一段历史上**，期数也要一样——换一段时间窗
    比出来的差值里混着运气，说明不了模型好坏。
    """
    hit = 0
    evaluated = 0
    universe = [as_key((a, b, c))
                for a in DIGIT_SPACE.numbers()
                for b in DIGIT_SPACE.numbers()
                for c in DIGIT_SPACE.numbers()]
    for _, actual in rolling_slices(numbers, trials):
        evaluated += 1
        if as_key(actual) in set(rng.sample(universe, top_n)):
            hit += 1
    return {
        'trials': evaluated,
        'random_hit': hit,
        'random_rate': hit / evaluated if evaluated else 0.0,
    }


def permutation_summary(perm_rates, observed_rate, baseline_rate, alpha=0.05):
    """置换检验的汇总：打乱历史顺序后还能不能达到这个命中率。

    **p 值分子分母都加 1**：观测本身也是一个排列，不把它算进去，
    shuffles 次全部低于观测时 p 会是 0，而 0 是不可能的。

    3D 是独立均匀摇奖，期间本来就没有时序可学。打乱后命中率不降，说明模型
    抓到的不是信号——**这一项不通过比通过更常见，也更可信**。
    """
    count = len(perm_rates)
    at_least = sum(1 for rate in perm_rates if rate >= observed_rate)
    pvalue = (at_least + 1) / (count + 1)
    return {
        'shuffles': count,
        'observed_rate': observed_rate,
        'shuffled_mean_rate': sum(perm_rates) / count if count else 0.0,
        'shuffled_max_rate': max(perm_rates) if perm_rates else 0.0,
        'baseline_rate': baseline_rate,
        'pvalue': pvalue,
        'significant': pvalue < alpha,
    }


def shuffled_series(numbers, shuffles, rng):
    """产出 `shuffles` 份打乱顺序的历史。

    在上一份的基础上继续洗，不每次从原序列重来——**两种写法等价**：对已经
    打乱的序列再洗一次，得到的仍是均匀随机排列，且与上一份独立。
    """
    seq = [list(number) for number in numbers]
    for _ in range(shuffles):
        rng.shuffle(seq)
        yield seq


# ─── 目标函数 ───

# 复合目标里三项的权重。Top3 占大头是因为它最难提升，也最能区分策略；
# ≥2 码只占 15%——它太容易达标，权重给高了会把搜索引向「广撒网」。
COMPOSITE_WEIGHTS = {'top3_rate': 0.55, 'top30_rate': 0.30, 'ge2_digit_rate': 0.15}
COMPOSITE = 'composite'
# 页面与旧配置里管 Top30 叫 top_rate。两个名字指同一件事
LEGACY_TOP_RATE = 'top_rate'
CANONICAL_TOP_RATE = 'top30_rate'


def objective(result, metric=CANONICAL_TOP_RATE, composite_weights=None):
    """从回测结果里取出要优化的那个数。

    **未知的 metric 直接抛**，不回退到默认值：权重搜索会拿这个数当唯一的
    方向感，静默换成另一个指标的话，搜出来的参数是在优化别的东西，
    而结果看起来完全正常。
    """
    if metric == LEGACY_TOP_RATE:
        metric = CANONICAL_TOP_RATE
    if metric == COMPOSITE:
        weights = composite_weights or COMPOSITE_WEIGHTS
        return sum(weight * result[key] for key, weight in weights.items())
    if metric not in result:
        raise ValueError(f'未知 metric: {metric}')
    return result[metric]


class TopKBacktest:
    """按若干个 TopK 门槛统计命中，外加实际号码的名次。

    与 `RankingBacktest` 的分工：那个服务规则模型，要分 raw/served 两条池子、
    还要统计组六四码；这里只有一条排名，因为 ML 模型的输出就是一条——
    **合并成一个类会让两边都长出一半用不上的参数**。
    """

    def __init__(self, tiers, miss_rank=MISS_RANK):
        self.tiers = tiers
        self.hits = {tier: 0 for tier in tiers}
        self.ranks = []
        self.miss_rank = miss_rank

    @property
    def trials(self):
        """实际评估的期数。训练不出模型的那些期被跳过，不该进分母。"""
        return len(self.ranks)

    def observe(self, actual, ranked):
        key = as_key(actual)
        self.ranks.append(rank_of(key, ranked, self.miss_rank))
        for tier in self.tiers:
            if key in ranked[:tier]:
                self.hits[tier] += 1

    def summarise(self):
        total = self.trials
        ordered = sorted(self.ranks)
        result = {
            'trials': total,
            # 与 trials 同值。两个名字都在历史记录里，谁也不知道哪个先被读
            'evaluated': total,
            'actual_rank_avg': round(sum(self.ranks) / total, 1) if total else 0.0,
            'actual_rank_median': ordered[len(ordered) // 2] if ordered else 0,
        }
        for tier in self.tiers:
            result[f'top{tier}_hit'] = self.hits[tier]
            result[f'top{tier}_rate'] = round(self.hits[tier] / total, 4) if total else 0
        return result


def tier_baseline(tier):
    """某个 TopK 门槛下随机命中的概率：K 注覆盖 1000 注里的 K 注。"""
    return tier / (DIGIT_SPACE.size ** POSITIONS)
