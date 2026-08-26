"""号码组合的「形态」：区间分布、奇偶、大小、与上期的重号。

形态**不是预测信号**，是结构约束。一组号码全落在 1~20、或者八个号全是
奇数，看起来就不像一期开奖——但它并不比别的组合更不容易中。这里的目标值
一律取中性（四个区平分、奇偶各半、大小各半），存在的意义是拦住畸形的推荐，
不是拿它去猜下期。

三个函数分工：`targets` 说「应该长什么样」，`profile` 说「实际长什么样」，
`penalty` 把两者的差算成一个越小越好的数。
"""
from src.domain.numeric.kl8.space import SPACE

ZONE20_SIZE = 20   # 四个 20 码大区，形态的主尺度
ZONE10_SIZE = 10   # 八个 10 码小区，只在画像里报告，不参与惩罚
BIG_SMALL_THRESHOLD = 40

# 惩罚权重。四个区偏离得最狠，所以系数最大；大小比奇偶宽松一点，是因为
# 大小的界（40）本身就是人为切的，而奇偶是号码自带的属性。
ZONE20_PENALTY = 1.15
ODD_PENALTY = 1.00
SMALL_PENALTY = 0.85
REPEAT_PENALTY = 0.95
# 号码个数对不上时的惩罚。不是「很大的分」，是「这组根本没法比」的标记。
MALFORMED_PENALTY = 999.0


def targets(target_size):
    """中性形态：四个大区尽量平分，奇偶与大小各占一半。

    除不尽的余数分给靠前的区——分法必须固定，否则同一组输入两次运行会得到
    不同的目标值。
    """
    zone_base, zone_remainder = divmod(max(target_size, 0), 4)
    zone_targets = [zone_base + (1 if idx < zone_remainder else 0) for idx in range(4)]
    half_low = target_size // 2
    half_high = target_size - half_low
    return {
        'zone20_targets': zone_targets,
        # 区间而非单点：7 个号分不出「奇偶各半」，3 和 4 都算正常。
        'odd_range': (half_low, half_high),
        'small_range': (half_low, half_high),
    }


def profile(numbers, last_numbers=None):
    """这组号码实际的形态画像。"""
    nums = sorted(int(n) for n in numbers)
    last_numbers = last_numbers or set()
    odd = sum(1 for n in nums if n % 2 == 1)
    small = sum(1 for n in nums if n <= BIG_SMALL_THRESHOLD)
    return {
        'zone20': _bucket_counts(nums, ZONE20_SIZE),
        'zone10': _bucket_counts(nums, ZONE10_SIZE),
        'odd_even': {'odd': odd, 'even': len(nums) - odd},
        'big_small': {'small': small, 'big': len(nums) - small},
        'sum': sum(nums),
        'repeat_from_last': sum(1 for n in nums if n in last_numbers),
    }


def _bucket_counts(numbers, bucket_size):
    """按固定宽度切桶后每桶几个号。

    空桶也要占位，否则位置就对不上了。**空间外的号码直接不计**——它进不了
    任何一个桶，硬塞进首尾桶会凭空造出一个形态。
    """
    bucket_count = SPACE.size // bucket_size
    counts = [0] * bucket_count
    for number in numbers:
        if not SPACE.contains(number):
            continue
        counts[(number - SPACE.low) // bucket_size] += 1
    return counts


def penalty(numbers, target_size, last_numbers, repeat_cap):
    """越小越好。个数对不上直接判出局——那不是形态差，是根本没选够。"""
    if len(numbers) != target_size:
        return MALFORMED_PENALTY

    actual = profile(numbers, last_numbers)
    target = targets(target_size)
    score = sum(abs(count - want) for count, want
                in zip(actual['zone20'], target['zone20_targets'])) * ZONE20_PENALTY
    score += _range_miss(actual['odd_even']['odd'], target['odd_range']) * ODD_PENALTY
    score += _range_miss(actual['big_small']['small'], target['small_range']) * SMALL_PENALTY
    # 重号只罚超出的一侧：重号偏少不是毛病，偏多才会让推荐变成上期的复读。
    score += max(0, actual['repeat_from_last'] - repeat_cap) * REPEAT_PENALTY
    return score


def _range_miss(actual, bounds):
    """落在区间内不罚，出界按距离罚。两侧对称——只罚一侧会让形态系统性偏斜。"""
    low, high = bounds
    if actual < low:
        return low - actual
    if actual > high:
        return actual - high
    return 0
