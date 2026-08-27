"""从 beidan 的 modeling 实现生成黄金快照。

迁移前后各跑一次，逐条比对——**差异要么为零，要么每一条都能说清为什么**。

`modeling` 无副作用（AST 查过：零网络、零存储、零时钟），所以直接调用即可，
不需要只读护栏。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_modeling_golden.py /tmp/beidan_modeling_old.json
"""
import json
import sys

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.modeling as m

# 三档赔率：强主、均势、强客。用 1X2 概率表示（已去水）
PROB_SETS = {
    'home_strong': (0.62, 0.23, 0.15),
    'balanced': (0.36, 0.29, 0.35),
    'away_strong': (0.18, 0.24, 0.58),
}
# 让球盘：平手、半球、一球、平半（分盘，结算线有两条）
HANDICAPS = ('0', '-0.5', '-1', '+0.25')
# 大小球线：整数、半球、四分之一
TOTAL_LINES = (2.5, 2.75, 3.0)
# 大小球赔率：偏大、均衡、偏小
OU_ODDS = {'over_lean': (0.85, 1.05), 'level': (0.95, 0.95), 'under_lean': (1.08, 0.82)}
# 四个联赛档案，覆盖高进球/低进球与高平局/低平局
LEAGUES = ('英超', '德甲', '意甲', '西甲')
# Dixon-Coles 的 rho。**线上是 0.0（修正不生效）**，所以额外取两个非零值——
# 配置随时会改回来，那条分支必须有语料（判据 9 第二类）
RHOS = (0.0, -0.05, 0.08)


def entries():
    for key, (ph, pd, pa) in PROB_SETS.items():
        yield f'lambdas:{key}', m.euro_implied_lambdas(ph, pd, pa, 2.6)
        for handicap in HANDICAPS:
            yield (f'draw_calib:{key}:{handicap}',
                   m.calibrate_draw_probability(ph, pd, pa, handicap))

    for k in range(8):
        for mu in (0.4, 1.2, 2.7):
            yield f'poisson:{k}:{mu}', m.poisson_pmf(k, mu)

    for handicap in HANDICAPS:
        yield f'parse_handicap:{handicap}', m.parse_beidan_handicap(handicap)

    for line in TOTAL_LINES:
        yield f'line_parts:{line}', m._asian_line_parts(line)
        yield f'line_value:{line}', m._parse_total_line_value(line)
        for goals in range(6):
            yield (f'over_profit:{line}:{goals}',
                   m._asian_over_profit(goals, line, 1.9))

    for water in (0.85, 0.95, 1.05):
        yield f'euro_odds:{water}', m._to_euro_odds(water)

    for name, (over, under) in OU_ODDS.items():
        for line in TOTAL_LINES:
            yield (f'implied_total:{name}:{line}',
                   m.implied_total_from_ou(over, under, line))

    for lam_home in (0.9, 1.6):
        for lam_away in (0.7, 1.4):
            for rho in RHOS:
                matrix = m.build_dixon_coles_matrix(lam_home, lam_away, rho=rho)
                yield f'dc_matrix:{lam_home}:{lam_away}:{rho}', matrix
                yield (f'goals_agg:{lam_home}:{lam_away}:{rho}',
                       m.aggregate_goals_from_scores(matrix))
                for handicap in HANDICAPS:
                    yield (f'rqspf:{lam_home}:{lam_away}:{rho}:{handicap}',
                           m.rqspf_probs_from_score_probs(matrix, handicap))

    # `anchor_score_outcomes` 要的是带别名键的 dict（胜/平/负、H/D/A、home/draw/away
    # 三套都认），不是三元组。三套别名各取一组语料——**别名解析写错了不会报错，
    # 只会让锚定悄悄退化成 `applied: False`**
    base_matrix = m.build_dixon_coles_matrix(1.4, 1.1)
    alias_forms = {
        'cn': lambda p: {'胜': p[0], '平': p[1], '负': p[2]},
        'letter': lambda p: {'H': p[0], 'D': p[1], 'A': p[2]},
        'word': lambda p: {'home': p[0], 'draw': p[1], 'away': p[2]},
    }
    for key, probs in PROB_SETS.items():
        for form, build in alias_forms.items():
            for strength in (0.0, 0.5, 1.0):
                yield (f'anchor:{key}:{form}:{strength}',
                       m.anchor_score_outcomes(base_matrix, build(probs),
                                               strength=strength))
    yield 'anchor:empty_target', m.anchor_score_outcomes(base_matrix, {})
    yield 'anchor:empty_dist', m.anchor_score_outcomes({}, {'胜': 0.4, '平': 0.3, '负': 0.3})

    for league in LEAGUES:
        for name, (over, under) in OU_ODDS.items():
            for line in TOTAL_LINES:
                yield (f'target_total:{league}:{name}:{line}',
                       m.match_target_total(league=league, total_over_odds=over,
                                            total_under_odds=under, total_line=line))

    # `split` 参数已删——它在迁移前的函数体里从没出现过，三个不同的值算出
    # 完全一样的结果，三处调用方也一个都没传过。键名保留 `:0.45`（迁移前的
    # 默认值）以便与旧黄金逐条对齐；旧黄金里的 `:0.35` / `:0.55` 会成为
    # 「仅旧有」，比对脚本会验证它们的值与 `:0.45` 相同——那正是参数无效的证据
    for key, (ph, pd, pa) in PROB_SETS.items():
        for target in (2.2, 2.8, 3.4):
            yield (f'match_lambdas:{key}:{target}:0.45',
                   m.match_lambdas(ph, pd, pa, target))

    for key, (ph, pd, pa) in PROB_SETS.items():
        for league in LEAGUES[:2]:
            for handicap in HANDICAPS:
                for use_dc in (True, False):
                    yield (f'predict_scores:{key}:{league}:{handicap}:{use_dc}',
                           m.predict_scores_by_poisson(
                               ph, pd, pa, league=league, handicap=handicap,
                               total_over_odds=0.95, total_under_odds=0.95,
                               use_dc=use_dc, total_line=2.5))


def main(out_path):
    golden = {}
    for key, value in entries():
        golden[key] = as_comparable(value)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
