"""从 beidan 的 markets 走势/因子实现生成黄金快照。

这些函数是纯计算（`markets.py` 里唯一的 `datetime.now()` 在
`_beidan_market_snapshot`，只用来打时间戳，不在这一组里）。

**语料按每个阈值的两侧铺开**：走势判定的门槛是 0.02/0.03/0.05，
只喂「明显在动」或「完全没动」的样本，把门槛改一点点是测不出来的（判据 5）。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_trends_golden.py /tmp/beidan_trends_old.json
"""
import json
import sys

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.markets as markets


def _series(start_home, step_home, start_away, step_away, count=6):
    """构造一段亚盘水位序列，两侧各按固定步长走。"""
    return [{'home_odds': round(start_home + step_home * i, 4),
             'away_odds': round(start_away + step_away * i, 4),
             'handicap': '-0.5'}
            for i in range(count)]


def _ou_series(start_over, step_over, start_under, step_under, count=6, line=2.5):
    return [{'over_odds': round(start_over + step_over * i, 4),
             'under_odds': round(start_under + step_under * i, 4),
             'line': line}
            for i in range(count)]


# 亚盘序列：围绕 0.02（调整门槛）与 0.03（方向门槛）两侧构造
ASIAN_HISTORIES = {
    'flat': _series(0.90, 0.0, 0.95, 0.0),
    'home_backing_strong': _series(1.05, -0.05, 0.85, 0.03),   # 主队降赔，过 0.03
    'home_backing_weak': _series(0.98, -0.025, 0.92, 0.01),    # 落在 0.02~0.03 之间
    'home_laying': _series(0.85, 0.05, 1.05, -0.03),
    'away_backing': _series(0.92, 0.005, 1.02, -0.05),
    'away_laying': _series(0.95, 0.0, 0.88, 0.05),
    'tiny_move': _series(0.95, -0.015, 0.95, 0.005),           # 低于 0.02，两个门槛都够不到
    'single_entry': _series(0.95, 0.0, 0.95, 0.0, count=1),
    'missing_side': [{'home_odds': 0.95}, {'home_odds': 0.90}, {'away_odds': 1.0}],
    'empty': [],
}

GOALS_HISTORIES = {
    'flat': _ou_series(0.95, 0.0, 0.95, 0.0),
    'over_backing': _ou_series(1.05, -0.08, 0.85, 0.04),
    'over_backing_weak': _ou_series(1.0, -0.04, 0.9, 0.01),    # 落在 0.05 门槛下方
    'over_laying': _ou_series(0.85, 0.08, 1.05, -0.04),
    'under_backing': _ou_series(0.95, 0.01, 1.05, -0.08),
    'lean_over_prices': _ou_series(0.80, 0.0, 1.05, 0.0),      # 大球贴水明显低
    'lean_under_prices': _ou_series(1.20, 0.0, 0.65, 0.0),     # 差值超过 0.5
    'near_margin': _ou_series(1.10, 0.0, 0.68, 0.0),           # 差值 0.42，够不到 0.5
    'single_entry': _ou_series(0.95, 0.0, 0.95, 0.0, count=1),
    'missing_side': [{'over_odds': 0.95}, {'over_odds': 0.90}],
    'empty': [],
}

# 亚盘水位之和：分档门槛是 3.6 / 4.0 / 4.4 / 4.8，每档两侧各取一个
ASIAN_SUM_HISTORIES = {
    f'sum_{total}': [{'home_odds': total / 2, 'away_odds': total / 2}] * 4
    for total in (3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.2)
}

CS_HISTORIES = {
    'active': [{'score': '1-0', 'odds': 8.5}, {'score': '1-1', 'odds': 7.2},
               {'score': '1-0', 'odds': 8.0}, {'score': '2-1', 'odds': 9.5},
               {'score': '1-1', 'odds': 7.5}],
    'drifting': [{'score': '1-0', 'odds': 8.0}, {'score': '1-0', 'odds': 8.3}],
    'firming': [{'score': '1-0', 'odds': 8.3}, {'score': '1-0', 'odds': 8.0}],
    'hairline': [{'score': '1-0', 'odds': 8.0}, {'score': '1-0', 'odds': 8.05}],
    'single_entry': [{'score': '1-0', 'odds': 8.0}],
    'malformed': [{'score': None, 'odds': 8.0}, {'odds': 7.0}, {'score': '1-1'}],
    'empty': [],
}

TOP_SCORES = {
    'dicts': [{'score': '1-0', 'probability': 0.12},
              {'score': '1-1', 'probability': 0.11},
              {'score': '2-1', 'probability': 0.09}],
    'tuples': [('1-0', 0.12), ('1-1', 0.11), ('2-1', 0.09)],
    'unparsable': [{'score': 'x-y', 'probability': 0.12},
                   {'score': '1-1', 'probability': 0.11}],
}

BUCKETS = {
    'even': {'0': 0.05, '1': 0.15, '2': 0.22, '3': 0.20,
             '4': 0.15, '5': 0.10, '6': 0.08, '7+': 0.05},
    'sparse': {'1': 0.4, '3': 0.6},
}

PROBS = (0.45, 0.28, 0.27)


def entries():
    for name, history in ASIAN_HISTORIES.items():
        yield f'asian_trend:{name}', markets.analyze_asian_trend(history)
        yield (f'adjust_probs:{name}',
               markets.adjust_probs_by_asian(*PROBS, history))
        yield f'asian_goal_factor:{name}', markets.calculate_asian_goal_factor(history)

    for name, history in ASIAN_SUM_HISTORIES.items():
        yield f'asian_goal_factor:{name}', markets.calculate_asian_goal_factor(history)

    for name, history in GOALS_HISTORIES.items():
        yield f'goals_trend:{name}', markets.analyze_goals_trend(history)
        yield f'goals_factor:{name}', markets.calculate_goals_factor(history)
        for bucket_name, buckets in BUCKETS.items():
            # 迁移前这个函数会改写入参，所以每次都要给一份新的
            yield (f'adjust_zjq:{name}:{bucket_name}',
                   markets.adjust_zjq_by_goals(dict(buckets), history))

    for name, history in CS_HISTORIES.items():
        yield f'cs_trend:{name}', markets.analyze_cs_trend(history)
        for scores_name, scores in TOP_SCORES.items():
            prediction = {'top3': [dict(item) if isinstance(item, dict) else item
                                   for item in scores]}
            yield (f'enhance_scores:{name}:{scores_name}',
                   markets.enhance_scores_with_cs(prediction, history))

    for name, history in GOALS_HISTORIES.items():
        yield (f'latest_ou:{name}',
               markets._latest_ou_market({'history': history}))
        yield (f'latest_ou_odds:{name}',
               markets._latest_ou_odds({'history': history}))
    yield 'latest_ou:no_history', markets._latest_ou_market({})
    yield 'latest_ou:none', markets._latest_ou_market(None)


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
