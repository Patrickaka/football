"""从 beidan 的联合市场状态实现生成黄金快照。

这几个函数是纯计算（`markets.py` 唯一的时钟在 `_beidan_market_snapshot`，
不在这一组里）。

**语料要走到 `constrain` 的三条出口**：正常收敛、目标落在支撑集之外
（所有结果同号）、以及两侧价格缺失导致整体不应用。只喂「正常」那条，
把二分搜索改坏都测不出来。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_market_state_golden.py /tmp/beidan_ms_old.json
"""
import json
import sys

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.markets as markets
import src.beidan.modeling as modeling


def _asian(handicaps, home_odds, away_odds):
    return {'history': [{'handicap': h, 'home_odds': ho, 'away_odds': ao}
                        for h, ho, ao in zip(handicaps, home_odds, away_odds)]}


def _goals(lines, over_odds, under_odds):
    return {'history': [{'line': l, 'over_odds': o, 'under_odds': u}
                        for l, o, u in zip(lines, over_odds, under_odds)]}


# 亚盘：方向、盘口线移动、缺价、单期
ASIAN_CASES = {
    'home_backing_line_up': _asian(['-0.5'] * 3 + ['-0.75'] * 3,
                                   [1.05, 1.00, 0.95, 0.92, 0.90, 0.88],
                                   [0.85, 0.88, 0.92, 0.95, 0.98, 1.02]),
    'away_backing_line_down': _asian(['-0.75'] * 3 + ['-0.5'] * 3,
                                     [0.85, 0.90, 0.95, 1.00, 1.05, 1.08],
                                     [1.05, 1.00, 0.95, 0.90, 0.85, 0.82]),
    'flat': _asian(['-0.5'] * 4, [0.95] * 4, [0.95] * 4),
    'pk': _asian(['0'] * 4, [0.98, 0.96, 0.94, 0.92], [0.92, 0.94, 0.96, 0.98]),
    'no_handicap_field': {'history': [{'home_odds': 0.95, 'away_odds': 0.95}] * 3},
    'missing_prices': {'history': [{'handicap': '-0.5'}] * 3},
    'single': _asian(['-0.5'], [0.95], [0.95]),
    'empty': {'history': []},
    'none': None,
}

# 大小球：水位与线的方向组合，含「水位与线打架」那种冲突
GOALS_CASES = {
    'over_backing_line_up': _goals([2.5, 2.5, 2.75, 2.75],
                                   [1.05, 0.98, 0.92, 0.85],
                                   [0.85, 0.92, 0.98, 1.05]),
    'over_backing_line_down': _goals([2.75, 2.75, 2.5, 2.5],   # 水位与线打架
                                     [1.05, 0.98, 0.92, 0.85],
                                     [0.85, 0.92, 0.98, 1.05]),
    'under_backing_line_down': _goals([2.75, 2.75, 2.5, 2.5],
                                      [0.85, 0.92, 0.98, 1.05],
                                      [1.05, 0.98, 0.92, 0.85]),
    'flat': _goals([2.5] * 4, [0.95] * 4, [0.95] * 4),
    'split_line': _goals(['2.5/3'] * 4, [0.95] * 4, [0.95] * 4),
    'unparsable_line': _goals(['abc'] * 4, [0.95] * 4, [0.95] * 4),
    'missing_prices': {'history': [{'line': 2.5}] * 3},
    'single': _goals([2.5], [0.95], [0.95]),
    'empty': {'history': []},
    'none': None,
}

# 比分矩阵：正常、只有主胜（让约束目标落到支撑集之外）、脏键、空
MATRICES = {
    'balanced': modeling.build_dixon_coles_matrix(1.4, 1.1),
    'home_heavy': modeling.build_dixon_coles_matrix(2.2, 0.6),
    'low_scoring': modeling.build_dixon_coles_matrix(0.7, 0.6),
    'only_home_wins': {(1, 0): 0.4, (2, 0): 0.35, (3, 0): 0.25},
    'malformed_keys': {'abc': 0.5, (1, 0): 0.3, (0, 1): 0.2},
    'empty': {},
}

SECTIONS = {
    'strong': {'quality': {'level': 'strong'}, 'probability': 0.68},
    'medium': {'quality': {'level': 'medium'}, 'probability': 0.55},
    'low': {'quality': {'level': 'low'}, 'probability': 0.40},
    'missing': {},
}


def entries():
    for asian_name, asian in ASIAN_CASES.items():
        for goals_name, goals in GOALS_CASES.items():
            yield (f'joint_state:{asian_name}:{goals_name}',
                   markets.build_beidan_joint_market_state(asian, goals))

    for matrix_name, matrix in MATRICES.items():
        for asian_name in ('home_backing_line_up', 'flat', 'missing_prices', 'none'):
            for goals_name in ('over_backing_line_up', 'flat', 'missing_prices', 'none'):
                yield (f'apply:{matrix_name}:{asian_name}:{goals_name}',
                       markets.apply_beidan_joint_market_state(
                           matrix, ASIAN_CASES[asian_name], GOALS_CASES[goals_name]))

    for goals_name, goals in GOALS_CASES.items():
        yield f'latest_ou:{goals_name}', markets._latest_ou_market(goals)
        yield f'latest_ou_odds:{goals_name}', markets._latest_ou_odds(goals)

    for section_name, section in SECTIONS.items():
        for bet_type in ('spf', 'rqspf', 'zjq', 'bifen'):
            for asian_name in ('home_backing_line_up', 'flat', 'none'):
                yield (f'admission:{section_name}:{bet_type}:{asian_name}',
                       markets.build_beidan_market_admission(
                           section, bet_type, ASIAN_CASES[asian_name],
                           GOALS_CASES['flat']))

    spf = {'probabilities': {'胜': 0.45, '平': 0.28, '负': 0.27},
           'score_probs': MATRICES['balanced']}
    for handicap in ('-0.5', '-1', '0', '+0.25', None):
        yield (f'water_market:{handicap}',
               markets.build_water_market_prediction(spf, handicap))


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
