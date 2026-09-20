"""Optional multi-window ranking for exclusion rounds only."""
from math import isfinite


def exclusion_candidates(analyzer, strategy, pool_size):
    windows = strategy.get('exclusion_windows')
    if windows is None:
        return analyzer.build_pool_by_strategy(strategy, pool_size=pool_size).get('candidates', [])
    if (not isinstance(windows, (list, tuple)) or not windows
            or any(type(w) is not int or w < 50 for w in windows)
            or len(set(windows)) != len(windows)):
        raise ValueError('exclusion_windows must contain distinct integer windows >= 50')
    scores = {number: 0.0 for number in range(1, 81)}
    for window in windows:
        ranking = analyzer.build_pool_by_strategy(
            {**strategy, 'window_size': window}, pool_size=80,
        )['candidates']
        if (len(ranking) != 80 or {n for n, _ in ranking} != set(scores)
                or any(not isfinite(score) for _, score in ranking)):
            raise ValueError('multi-window ranking requires all 80 finite number scores')
        # Equal scores receive equal midranks; numeric tie breaking must not
        # manufacture consensus for low-numbered balls.
        ranking = sorted(ranking, key=lambda item: -item[1])
        start = 0
        while start < 80:
            end = start + 1
            while end < 80 and ranking[end][1] == ranking[start][1]:
                end += 1
            percentile = 1.0 - (start + end - 1) / 158.0
            for number, _ in ranking[start:end]:
                scores[number] += percentile / len(windows)
            start = end
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:pool_size]
