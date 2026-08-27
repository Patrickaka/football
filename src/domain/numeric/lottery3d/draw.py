"""一注号码自身的属性：跨度、形态、奇偶比、大小比、连号、邻号、路数。

**这些函数只看一注，不看历史**——传进来一个三元组，回答一个关于它的问题。
需要历史的量在 `history.py`。分开的理由是它们的稳定性完全不同：这里的定义
（什么叫组三、什么叫连号）几乎不会变，那边的窗口与衰减系数是调参对象。

**没有一个函数能提高中奖概率。** 直选恒定 1/1000，形态只决定推荐「长得
像不像一次真实开奖」。
"""
from src.domain.numeric.lottery3d.space import DIGIT_SPACE

BIG_SMALL_THRESHOLD = 5   # 0~4 小、5~9 大
ROAD_MODULUS = 3          # 012 路

ZU6, ZU3, BAOZI = 'zu6', 'zu3', 'baozi'

FORM_LABELS = {ZU6: '组六', ZU3: '组三', BAOZI: '豹子'}

# 三个数字互不相同的概率是 0.72，两个相同 0.27，三个相同 0.01。
# 这是**组合数算出来的常数**，不是从历史拟合的——拿它当先验才成立。
THEORY_FORM_P = {ZU6: 0.72, ZU3: 0.27, BAOZI: 0.01}


def span(triple):
    """跨度：最大位与最小位之差。"""
    return max(triple) - min(triple)


def classify_form(triple):
    """形态：三个数字几不相同。组六 / 组三 / 豹子。"""
    distinct = len(set(triple))
    if distinct == 3:
        return ZU6
    if distinct == 2:
        return ZU3
    return BAOZI


def odd_even_key(triple):
    """奇偶比 (奇数个数, 偶数个数)。"""
    odds = sum(1 for digit in triple if digit % 2 == 1)
    return odds, len(triple) - odds


def big_small_key(triple):
    """大小比 (大数个数, 小数个数)。"""
    bigs = sum(1 for digit in triple if digit >= BIG_SMALL_THRESHOLD)
    return bigs, len(triple) - bigs


def ratio_label(key, kind='oe'):
    """把比例元组写成中文。`oe` 是奇偶，其余一律按大小写。"""
    first, second = key
    if kind == 'oe':
        return f'{first}奇{second}偶'
    return f'{first}大{second}小'


def has_consecutive_digits(*digits):
    """是否存在相差 1 的两位。

    **9 与 0 不算连号**：这里比的是数值大小，不是转盘上的相邻。绕回一旦算上，
    「连号」就跟 `neighbor` 混成一件事了，而那两个是不同的特征。
    """
    values = list(digits)
    return any(abs(values[i] - values[j]) == 1
               for i in range(len(values))
               for j in range(i + 1, len(values)))


def neighbor(digit):
    """相邻的两个数字，**绕回**——0 的邻居是 9 和 1。

    与 `has_consecutive_digits` 刚好相反：那个不绕，这个绕。两者都对，
    因为它们回答的不是同一个问题。
    """
    size = DIGIT_SPACE.size
    return {(digit - 1) % size, (digit + 1) % size}


def road(digit):
    """012 路：对 3 取模。"""
    return digit % ROAD_MODULUS


def digit_overlap(left, right):
    """两注共有几个数字，按重数算。

    `112` 与 `122` 共有两个（一个 1、一个 2），不是三个——按集合算会把
    「同一个数字出现两次」和「出现一次」当成一回事。
    """
    remaining = list(right)
    shared = 0
    for digit in left:
        if digit in remaining:
            remaining.remove(digit)
            shared += 1
    return shared
