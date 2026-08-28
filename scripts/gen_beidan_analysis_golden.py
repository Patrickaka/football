"""从 beidan 的 recommending 实现生成「分析与组装」这半批的黄金快照。

覆盖 4-T5 前半的六个对象：本地赛果分析（`build_beidan_match_analysis`）、
总进球分组、半全场、总进球准入门槛的输入、候选日期、价值投注筛选。

**比分分布不是编的**：语料里的 `score_probs` 一律由 `predict_scores_by_poisson`
真实算出来（判据 10），只有需要专门撞某条分支时才手工构造——
比如空分布、JSON 列表形态、爆冷候选写坏了的那几条。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_analysis_golden.py /tmp/beidan_analysis_old.json
"""
import json
import sys

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.recommending as rec_mod
from src.beidan.modeling import predict_scores_by_poisson
from src.beidan.quality import assess_recommendation_quality
from src.beidan.upset import assess_upset_risk, pick_upset_scores

# ── 真实比分分布：三种力量对比 × 两种联赛 ────────────────────────
# 联赛只影响 `avg_goals` / `draw_rate` 两个先验，取一高一低两个档
PROB_SETS = {
    'home_strong': (0.62, 0.22, 0.16),
    'balanced': (0.36, 0.30, 0.34),
    'away_strong': (0.18, 0.26, 0.56),
    'draw_lean': (0.30, 0.40, 0.30),
    # 下面三组专门用来分离 `build_decision` 与 `build_score_strategy` 两道判定
    # ——它们的门槛不同，只喂「都过」或「都不过」的样本分不出是谁在起作用。
    # 实测（判据 28）：英超 0.72 组首选 0.7019、top1 比分 0.1374 → 单选 ✓ 单比分 ✗；
    # 意甲 0.62 组首选 0.6204、top1 比分 0.1450 → 单选 ✗ 单比分 ✓。
    'single_only': (0.72, 0.18, 0.10),
    'score_only': (0.62, 0.22, 0.16),
    'both_strong': (0.80, 0.13, 0.07),
}
# 上面三组各自只在一个联赛上成立——低进球联赛的比分分布更集中
SPLIT_LEAGUES = {'single_only': '英超', 'score_only': '意甲', 'both_strong': '意甲'}


def _spf_result(name, probs, league='英超', handicap=0, with_upset=True,
                quality_probs=None, odds=True, asian=None, lambdas=True):
    """按 `analyze_spf` 真实产出的形状拼一条 spf 结果。

    只走纯计算那几步（比分分布、爆冷、质量分档），不碰欧赔抓取与历史校准
    ——那两处是适配层的事，也正是这一批要摘出去的。
    """
    home, draw, away = probs
    prediction = predict_scores_by_poisson(home, draw, away, league=league,
                                           handicap=handicap)
    score_probs = prediction['score_probs']
    marginals = {
        '胜': sum(p for (h, a), p in score_probs.items() if h > a),
        '平': sum(p for (h, a), p in score_probs.items() if h == a),
        '负': sum(p for (h, a), p in score_probs.items() if h < a),
    }
    upset = assess_upset_risk(quality_probs or marginals)
    upset['candidates'] = pick_upset_scores(score_probs, upset.get('favorite'), top_n=2)
    result = {
        'match_id': '1320957', 'num': '1', 'home': '安山小绿人', 'away': '大邱FC',
        'league': league, 'time': '18:30', 'type': 'spf',
        'probabilities': marginals,
        'prediction': max(marginals, key=marginals.get),
        'confidence': max(marginals.values()),
        'score_probs': [[h, a, round(p, 6)] for (h, a), p in score_probs.items()],
        'scores': prediction['top3'],
        'lambda_home': prediction['lambda_home'] if lambdas else None,
        'lambda_away': prediction['lambda_away'] if lambdas else None,
        'quality': assess_recommendation_quality(quality_probs or marginals),
        'upset': upset if with_upset else {},
    }
    if odds:
        result['odds'] = {'胜': round(1 / home, 2), '平': round(1 / draw, 2),
                          '负': round(1 / away, 2)}
    if asian is not None:
        result['asian_trend'] = asian
    # 元组键的原始分布也留一份：`build_beidan_match_analysis` 两种形态都认，
    # **而认错了不会报错，只会让整份分析静默换一个输入**。
    # 两种形态的输出不会逐字相同——列表形态本身是 `round(p, 6)` 存的，
    # 差异全落在第 6 位小数上。它们覆盖的是「两条入口都能走通」，不是相等。
    result['_tuple_score_probs'] = score_probs
    return result


def spf_results():
    cases = {}
    for name, probs in PROB_SETS.items():
        for league in ('英超', '挪超'):
            cases[f'{name}:{league}'] = _spf_result(name, probs, league=league)
    for name, league in SPLIT_LEAGUES.items():
        cases[f'{name}:{league}:split'] = _spf_result(
            name, PROB_SETS[name], league=league)
    # 爆冷预警那一支：`score_picks` 会多出一个「防冷」
    cases['upset_alert'] = _spf_result(
        'upset', PROB_SETS['balanced'],
        quality_probs={'胜': 0.42, '平': 0.33, '负': 0.25})
    # 热门稳胆那一支：走的是另一条理由叙述
    cases['upset_confident'] = _spf_result(
        'confident', PROB_SETS['home_strong'],
        quality_probs={'胜': 0.62, '平': 0.22, '负': 0.16})
    cases['no_upset_section'] = _spf_result('plain', PROB_SETS['balanced'],
                                            with_upset=False)
    cases['no_odds'] = _spf_result('no_odds', PROB_SETS['balanced'], odds=False)
    cases['no_lambdas'] = _spf_result('no_lam', PROB_SETS['balanced'], lambdas=False)
    cases['with_asian_trend'] = _spf_result(
        'asian', PROB_SETS['balanced'], asian={'direction': '主队receive', 'strength': 12.5})
    cases['asian_trend_no_direction'] = _spf_result(
        'asian2', PROB_SETS['balanced'], asian={'strength': 12.5})
    return cases


def analysis_inputs():
    """喂给 `build_beidan_match_analysis` 的输入，含各种残缺形态。"""
    cases = {}
    for name, result in spf_results().items():
        tuple_probs = result.pop('_tuple_score_probs')
        cases[f'list_form:{name}'] = result
        # 同一份数据的元组键形态：覆盖的是「两条入口都走得通」，不是相等
        cases[f'tuple_form:{name}'] = dict(result, score_probs=tuple_probs)

    base = _spf_result('base', PROB_SETS['balanced'])
    base.pop('_tuple_score_probs')
    cases['none'] = None
    cases['empty'] = {}
    cases['error'] = {'error': '欧赔数据不可用'}
    cases['no_score_probs'] = dict(base, score_probs={})
    cases['score_probs_missing'] = {k: v for k, v in base.items() if k != 'score_probs'}
    # 列表里混着长度不对的项——旧实现只收长度为 3 的
    cases['ragged_list'] = dict(base, score_probs=[[1, 0, 0.2], [1, 1], [0, 1, 0.3]])
    # 爆冷候选的比分写坏了：旧实现 catch 掉 ValueError / KeyError 后继续
    alert = _spf_result('a', PROB_SETS['balanced'],
                        quality_probs={'胜': 0.42, '平': 0.33, '负': 0.25})
    alert.pop('_tuple_score_probs')
    cases['upset_candidate_unparsable'] = dict(
        alert, upset=dict(alert['upset'], alert=True, candidates=[{'score': 'x-y'}]))
    cases['upset_candidate_missing_key'] = dict(
        alert, upset=dict(alert['upset'], alert=True, candidates=[{'no_score': 1}]))
    cases['upset_alert_without_candidates'] = dict(
        alert, upset=dict(alert['upset'], alert=True, candidates=[]))
    # 质量分档影响 `build_decision` 与 `build_score_strategy` 的 confidence
    for level in ('strong', 'medium', 'split', 'low', None):
        cases[f'quality:{level}'] = dict(base, quality={'level': level})
    cases['quality_missing'] = {k: v for k, v in base.items() if k != 'quality'}
    cases['confidence_missing'] = {k: v for k, v in base.items() if k != 'confidence'}
    return cases


# ── 总进球分组 ───────────────────────────────────────────────────
# 线上真实键集是 '0'~'6' 与 '7+'。**'2' 同时属于小球组与中位组**，
# 三组概率之和大于 1，这是有意的重叠分组，不是漏写
ZJQ_PROB_SETS = {
    'live_shape': {'0': 0.033, '1': 0.053, '2': 0.143, '3': 0.214,
                   '4': 0.247, '5': 0.140, '6': 0.099, '7+': 0.071},
    'small_heavy': {'0': 0.20, '1': 0.30, '2': 0.30, '3': 0.10, '4': 0.10},
    'big_heavy': {'3': 0.30, '4': 0.30, '5': 0.20, '6': 0.10, '7+': 0.10},
    'middle_heavy': {'2': 0.45, '3': 0.45, '0': 0.05, '5': 0.05},
    'string_none': {'0': None, '1': '0.3', '2': 0.2},
    'missing_options': {'3': 1.0},
    'empty': {},
}

# ── 半全场 ───────────────────────────────────────────────────────
BQC_MATCH = {'id': '1320957', 'num': '1', 'home': '安山小绿人',
             'away': '大邱FC', 'league': 'K2联赛', 'time': '18:30'}
BQC_ODDS = {
    'present': {'1320957': {'胜胜': 3.2, '胜平': 15.0, '胜负': 41.0,
                            '平胜': 7.5, '平平': 6.1, '平负': 12.0,
                            '负胜': 51.0, '负平': 17.0, '负负': 4.3}},
    'dead_odds': {'1320957': {'胜胜': 0, '平平': None, '负负': 4.3}},
    'all_dead': {'1320957': {'胜胜': 0, '平平': None}},
    'missing': {},
}

# ── 总进球准入门槛的输入 ─────────────────────────────────────────
GOALS_DATA = {
    'none': None,
    'no_history': {'history': []},
    'over_under': {'history': [
        {'line': '2.5', 'over_odds': 0.95, 'under_odds': 0.90},
        {'line': '3', 'over_odds': 0.85, 'under_odds': 1.00},
    ]},
    'latest_incomplete': {'history': [
        {'line': '2.5', 'over_odds': 0.95, 'under_odds': 0.90},
        {'line': '3', 'over_odds': None, 'under_odds': 1.00},
    ]},
    'zero_water': {'history': [{'line': '2.5', 'over_odds': 0.0, 'under_odds': 0.0}]},
    'bad_water': {'history': [{'line': '2.5', 'over_odds': 'x', 'under_odds': 'y'}]},
}
GATE_SECTIONS = {
    'live_shape': {'probabilities': ZJQ_PROB_SETS['live_shape']},
    'small_heavy': {'probabilities': ZJQ_PROB_SETS['small_heavy']},
    'dirty_keys': {'probabilities': {'2': 0.4, '7+': 0.3, 'x': 0.2, '3': 'y'}},
    'no_probabilities': {},
    'none': None,
}

# ── 候选日期 ─────────────────────────────────────────────────────
DATE_CASES = [
    ('normal', '2026-08-28', True, 2),
    ('no_fallback', '2026-08-28', False, 2),
    ('zero_days', '2026-08-28', True, 0),
    ('five_days', '2026-08-28', True, 5),
    ('month_boundary', '2026-08-30', True, 3),
    ('year_boundary', '2026-12-30', True, 3),
    ('leap_day', '2028-02-27', True, 3),
    ('unparsable', 'not-a-date', True, 2),
    ('none', None, True, 2),
]

# ── 价值投注 ─────────────────────────────────────────────────────
# `probabilities` 与 `odds` 的键必须对上，否则一注都挑不出来
VALUE_RECOMMENDATIONS = [
    {'num': '1', 'home': 'A', 'away': 'B',
     'spf': {'probabilities': {'胜': 0.50, '平': 0.28, '负': 0.22},
             'odds': {'胜': 2.60, '平': 3.30, '负': 4.10}},
     'zjq': {'probabilities': {'0': 0.05, '1': 0.12, '2': 0.20, '7+': 0.10},
             'odds': {'0': 12.0, '1': 7.0, '2': 4.2, '7+': 30.0}}},
    # 让球分节存在但没有 odds 键——**总进球那支用 .get 容错，另外两支直接下标**
    {'num': '2', 'home': 'C', 'away': 'D',
     'spf': {'probabilities': {'胜': 0.34, '平': 0.33, '负': 0.33},
             'odds': {'胜': 3.00, '平': 3.00, '负': 3.00}},
     'rqspf': {'probabilities': {'让胜': 0.40, '让平': 0.25, '让负': 0.35},
               'odds': {'让胜': 2.20, '让平': 3.60, '让负': 3.10}}},
    # 脏赔率：0、None、负数各一
    {'num': '3', 'home': 'E', 'away': 'F',
     'spf': {'probabilities': {'胜': 0.60, '平': 0.20, '负': 0.20},
             'odds': {'胜': 0, '平': None, '负': -2.0}}},
    # 没有 probabilities 的分节整条跳过
    {'num': '4', 'home': 'G', 'away': 'H', 'spf': {'odds': {'胜': 2.0}}},
    # 完全没有这几个分节
    {'num': '5', 'home': 'I', 'away': 'J', 'bifen': {'probabilities': {'1-0': 0.2}}},
    # 总进球里 '7+' 那一档必须被跳过，哪怕它有正的 edge
    {'num': '6', 'home': 'K', 'away': 'L',
     'zjq': {'probabilities': {'7+': 0.90, '2': 0.05},
             'odds': {'7+': 30.0, '2': 4.0}}},
]


def entries():
    for name, spf_result in analysis_inputs().items():
        yield (f'match_analysis:{name}',
               rec_mod.build_beidan_match_analysis(spf_result))

    for name, probs in ZJQ_PROB_SETS.items():
        yield f'zjq_group:{name}', rec_mod.build_zjq_group_recommendation(probs)

    for name, odds in BQC_ODDS.items():
        yield f'bqc:{name}', rec_mod.analyze_bqc(BQC_MATCH, odds)

    for section_name, section in GATE_SECTIONS.items():
        for goals_name, goals in GOALS_DATA.items():
            for league in ('英超', '未知联赛'):
                yield (f'gate:{section_name}:{goals_name}:{league}',
                       rec_mod.build_beidan_total_goals_accuracy_gate(
                           section, goals, league))

    for name, date, allow, days in DATE_CASES:
        yield (f'dates:{name}',
               rec_mod._candidate_beidan_dates(date, allow_fallback=allow, days=days))

    for threshold in (-1.0, 0.0, 0.02, 0.05, 0.20, 0.90):
        yield (f'value_bets:{threshold}',
               _value_bets_of(VALUE_RECOMMENDATIONS, threshold))


def _value_bets_of(recommendations, threshold):
    """迁移前这段逻辑埋在 `find_value_bets` 里、抓完网络才跑得到。

    这里原样重跑那段循环，好让新旧两版比对同一件事——**它是被提取的对象，
    不是被测的实现**，所以刻意抄成与旧实现同构。
    """
    value_bets = []
    for match in recommendations:
        for bet_type in ['spf', 'rqspf', 'zjq']:
            if bet_type not in match:
                continue
            data = match[bet_type]
            if 'probabilities' not in data:
                continue
            probs = data['probabilities']
            if bet_type == 'zjq':
                odds_map = data.get('odds') or {}
                for key, prob in probs.items():
                    if key == '7+':
                        continue
                    odd = odds_map.get(key)
                    if odd and odd > 0:
                        edge = prob - 1 / odd
                        if edge > threshold:
                            value_bets.append({
                                'num': match['num'], 'home': match['home'],
                                'away': match['away'], 'type': bet_type,
                                'option': key, 'probability': prob, 'odd': odd,
                                'implied_probability': 1 / odd, 'edge': edge})
            else:
                for key, prob in probs.items():
                    odd = data['odds'].get(key)
                    if odd and odd > 0:
                        edge = prob - 1 / odd
                        if edge > threshold:
                            value_bets.append({
                                'num': match['num'], 'home': match['home'],
                                'away': match['away'], 'type': bet_type,
                                'option': key, 'probability': prob, 'odd': odd,
                                'implied_probability': 1 / odd, 'edge': edge})
    return sorted(value_bets, key=lambda item: -item['edge'])


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
