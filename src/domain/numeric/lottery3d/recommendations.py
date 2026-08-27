"""对「最近推荐过什么」的处理：重复推荐降权，以及与开奖号的重合度。

这一层与开奖历史无关，看的是**我们自己发过什么**。存在的理由是连着几期推
同一注会让人以为系统「看好」它，而实际上每一注的中奖概率都是 1/1000——
降权是为了让推荐轮换起来，不是因为推过的号更不容易开出。
"""
from collections import Counter

from src.domain.numeric.lottery3d.draw import digit_overlap


def recent_numbers(history, window):
    """把最近若干期的推荐摊平成 (出现过的号码集合, 各自出现次数)。

    两种记录格式都要认：新格式是 `{"period": ..., "recommendations": [...]}`，
    旧格式直接是号码列表。**线上两种都还在**，只认一种会让一半历史被当成空。
    """
    seen, counts = set(), Counter()
    for entry in history[-window:]:
        numbers = entry.get('recommendations', []) if isinstance(entry, dict) else entry
        for number in numbers:
            seen.add(number)
            counts[number] += 1
    return seen, counts


def penalise_repeats(pool, history, window, repeat_penalty, consecutive_penalty):
    """对最近推过的号码降权，连续推过的再降一档。

    空历史时原样返回——给一个「都没推过」的空集合去扣分，结果一样，
    但会白跑一遍整个池子。
    """
    if not history:
        return pool

    seen, counts = recent_numbers(history, window)
    penalised = []
    for weight, number in pool:
        penalty = 0.0
        if number in seen:
            penalty -= repeat_penalty
        if counts.get(number, 0) >= 2:
            penalty -= consecutive_penalty
        penalised.append((weight + penalty, number))
    return penalised


def max_digit_overlap(actual, candidates):
    """候选里与开奖号重合最多的那注重合了几个数字。

    按**重数**算：开奖 `112`、候选 `122` 重合两个（一个 1、一个 2），
    不是三个。按集合算会把「1 开出两次」和「开出一次」混为一谈，
    回测的重合度分布就会整体偏高。
    """
    if not candidates:
        return 0
    return max(digit_overlap(actual, candidate) for candidate in candidates)
