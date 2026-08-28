"""从 beidan 的 recommending 实现生成四种玩法推荐的黄金快照。

两处外部依赖用打桩替掉，**打在被测路径之外**（判据：打在路径上就等于把
路径删了）：`fetch_ouzhi_odds` 是网络，`apply_beidan_history_calibration`
要读已结算历史。校准的桩给两种：一种什么也不做、一种真的改动概率——
只喂前者的话，「校准的结果被后面覆盖掉了没有」根本看不出来。

语料按**四条流水线各自的分叉**铺：欧赔的两个来源与全缺、让球值有无、
亚盘/大小球/比分走势三份历史的有无、市场报价的有无。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_recommendation_golden.py /tmp/beidan_rec_old.json
"""
import json
import sys
from unittest import mock

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.recommending as rec_mod

# ── 比赛 ─────────────────────────────────────────────────────────
# 字段名来自线上真实赛程（`fetch_okooo_schedule` 的产出）
BASE_MATCH = {
    'id': '1320957', 'num': '1', 'home': '安山小绿人', 'away': '大邱FC',
    'league': '英超', 'time': '18:30', 'handicap': '(-1)',
}

MATCHES = {
    'plain': BASE_MATCH,
    # 让球值：线上真实的四种文本形态 + 无盘口
    'handicap_plus_one': dict(BASE_MATCH, handicap='(+1)'),
    'handicap_minus_two': dict(BASE_MATCH, handicap='(-2)'),
    'handicap_zero': dict(BASE_MATCH, handicap='0'),
    'no_handicap': dict(BASE_MATCH, handicap=None),
    'unparsable_handicap': dict(BASE_MATCH, handicap='公司'),
    # 赛程自带的欧赔：三个都在 / 缺一个
    'with_okooo_main': dict(BASE_MATCH, spf_sp=2.10, spf_s=3.30, spf_f=3.60),
    'partial_okooo_main': dict(BASE_MATCH, spf_sp=2.10, spf_s=3.30),
    # 官方让球赔率的两种来源与拼装
    'official_rqspf': dict(BASE_MATCH,
                           rqspf_odds={'让胜': 2.20, '让平': 3.40, '让负': 3.10}),
    'lottery_rqspf': dict(BASE_MATCH,
                          lottery_rqspf_odds={'让胜': 2.05, '让平': 3.50, '让负': 3.30}),
    'rqspf_prices': dict(BASE_MATCH, rqspf_sp=2.20, rqspf_s=3.40, rqspf_f=3.10),
    # 三个单价里有一个不到 1.0 —— 整组都不认
    'rqspf_prices_below_one': dict(BASE_MATCH, rqspf_sp=2.20, rqspf_s=0.95,
                                   rqspf_f=3.10),
    'rqspf_prices_dirty': dict(BASE_MATCH, rqspf_sp='x', rqspf_s=3.40, rqspf_f=3.10),
    # 联赛档案里没有的联赛 → 走兜底档案
    'unknown_league': dict(BASE_MATCH, league='某不知名联赛'),
    'low_scoring_league': dict(BASE_MATCH, league='意甲'),
}

# ── 欧赔 ─────────────────────────────────────────────────────────
OUZHI = {
    'home_favourite': {'home': 1.80, 'draw': 3.60, 'away': 4.20},
    'balanced': {'home': 2.70, 'draw': 3.20, 'away': 2.70},
    # 归一后平局恰好 0.274 —— **落在联赛平局率唯一起作用的那条窄带上**。
    # 英超 0.28 时不削平局、0.30 时要削（门槛分别是 0.2704 与 0.2773），
    # 别的取值两个档案算出来一模一样，联赛档案改坏了也测不出来（判据 23）。
    'draw_rate_sensitive': {'home': 2.20, 'draw': 3.363, 'away': 3.00},
    'away_favourite': {'home': 4.50, 'draw': 3.70, 'away': 1.75},
    'missing': None,
}

# ── 三份走势历史 ─────────────────────────────────────────────────
ASIAN = {
    'none': None,
    'empty': {'history': []},
    'rising': {'history': [
        {'ts': '2026-08-28T09:00:00', 'handicap': '-0.5', 'home_odds': 0.95,
         'away_odds': 0.90},
        {'ts': '2026-08-28T10:00:00', 'handicap': '-0.75', 'home_odds': 0.88,
         'away_odds': 0.98},
    ]},
    'falling': {'history': [
        {'ts': '2026-08-28T09:00:00', 'handicap': '-0.75', 'home_odds': 0.90,
         'away_odds': 0.95},
        {'ts': '2026-08-28T10:00:00', 'handicap': '-0.5', 'home_odds': 1.00,
         'away_odds': 0.85},
    ]},
}
GOALS = {
    'none': None,
    'empty': {'history': []},
    'over_laying': {'history': [
        {'ts': '2026-08-28T09:00:00', 'line': '2.5', 'over_odds': 0.95,
         'under_odds': 0.90},
        {'ts': '2026-08-28T10:00:00', 'line': '2.5', 'over_odds': 0.80,
         'under_odds': 1.05},
    ]},
    'line_moved': {'history': [
        {'ts': '2026-08-28T09:00:00', 'line': '2.5', 'over_odds': 0.95,
         'under_odds': 0.90},
        {'ts': '2026-08-28T10:00:00', 'line': '3', 'over_odds': 0.92,
         'under_odds': 0.93},
    ]},
    # 最后一条缺价 —— `_latest_ou_market` 要往回找
    'latest_incomplete': {'history': [
        {'ts': '2026-08-28T09:00:00', 'line': '2.5', 'over_odds': 0.95,
         'under_odds': 0.90},
        {'ts': '2026-08-28T10:00:00', 'line': '3', 'over_odds': None,
         'under_odds': 0.93},
    ]},
}
# 比分盘历史。**条目的键是 `time` / `score` / `odds` 三个平铺字段**，
# 不是一个 `scores` 字典——第一版按命名猜成了后者，于是 `market_odds` 恒为空、
# 整条融合从来没跑过，而 `cs_adjusted` 照样是 True（判据 10、23）。
# 真实形状来自 `fetch_okooo_cs_history`（`fetching.py:513`）。
CS = {
    'none': None,
    'empty': {'history': []},
    # 只有一条 —— 融合要求至少两条
    'single_entry': {'history': [{'time': '09:00', 'score': '1-0', 'odds': 8.0}]},
    # 键名写错的那种：融合会原样返回，`cs_adjusted` 仍是 True
    'wrong_shape': {'history': [{'ts': 't1', 'scores': {'1-0': 8.0}},
                                {'ts': 't2', 'scores': {'1-0': 7.5}}]},
    'present': {'history': [
        {'time': '09:00', 'score': '1-0', 'odds': 8.0},
        {'time': '09:30', 'score': '1-1', 'odds': 7.5},
        {'time': '10:00', 'score': '2-1', 'odds': 9.0},
        {'time': '10:30', 'score': '0-1', 'odds': 11.0},
    ]},
    # 盘口上有、模型 top3 里没有的比分 —— 走打折补入那一支
    'new_scores_only': {'history': [
        {'time': '09:00', 'score': '4-3', 'odds': 60.0},
        {'time': '09:30', 'score': '3-3', 'odds': 45.0},
    ]},
    # 超出窗口的条目要被截掉
    'long_history': {'history': [
        {'time': f'{9 + i}:00', 'score': f'{i}-0', 'odds': 8.0 + i}
        for i in range(8)]},
}

# ── 市场报价 ─────────────────────────────────────────────────────
BIFEN_ODDS = {
    'none': None,
    'empty': {},
    'other_match': {'999': {'1-0': 8.0}},
    'present': {'1320957': {'1-0': 8.0, '1-1': 7.5, '2-1': 9.0, '0-1': 11.0}},
    'all_dead': {'1320957': {'1-0': 0, '1-1': None}},
}
ZJQ_ODDS = {
    'none': None,
    'empty': {},
    'present': {'1320957': {'0': 11.0, '1': 5.6, '2': 3.9, '3': 4.3,
                            '4': 6.5, '5': 11.0, '6': 21.0, '7+': 26.0}},
    'partial': {'1320957': {'2': 3.9, '3': 4.3}},
    'all_dead': {'1320957': {'0': 0, '1': None}},
}


def _no_calibration(probabilities, bet_type, league=None):
    return probabilities, {'applied': False, 'reason': 'stub_no_history'}


def _tilting_calibration(probabilities, bet_type, league=None):
    """真的改动概率的桩：把每个选项按名字长度轻微加权再归一。

    **规则本身没有意义，要的是「它确实动了」**——只喂恒等桩的话，
    「校准结果被后面覆盖掉了没有」这件事在黄金里根本看不出来。
    """
    if not probabilities:
        return probabilities, {'applied': False, 'reason': 'stub_empty'}
    factors = {key: 1.0 + 0.02 * (len(str(key)) % 3) for key in probabilities}
    adjusted = {key: float(value or 0.0) * factors[key]
                for key, value in probabilities.items()}
    total = sum(adjusted.values())
    if total <= 0:
        return probabilities, {'applied': False, 'reason': 'stub_zero'}
    return ({key: value / total for key, value in adjusted.items()},
            {'applied': True, 'reason': 'stub_tilt',
             'factors': {str(k): round(v, 6) for k, v in factors.items()}})


CALIBRATORS = {'flat': _no_calibration, 'tilt': _tilting_calibration}


def _run(fn, ouzhi, calibrator, *args, **kwargs):
    with mock.patch.object(rec_mod, 'fetch_ouzhi_odds', return_value=ouzhi), \
         mock.patch.object(rec_mod, 'apply_beidan_history_calibration',
                           side_effect=calibrator):
        return fn(*args, **kwargs)


def entries():
    # 胜平负：欧赔来源 × 三份走势
    for match_name in ('plain', 'with_okooo_main', 'partial_okooo_main',
                       'unknown_league', 'low_scoring_league'):
        for odds_name, ouzhi in OUZHI.items():
            for cal_name, calibrator in CALIBRATORS.items():
                yield (f'spf:{match_name}:{odds_name}:{cal_name}',
                       _run(rec_mod.analyze_spf, ouzhi, calibrator,
                            MATCHES[match_name]))

    for asian_name, asian in ASIAN.items():
        for goals_name, goals in GOALS.items():
            yield (f'spf_market:{asian_name}:{goals_name}',
                   _run(rec_mod.analyze_spf, OUZHI['home_favourite'],
                        _no_calibration, MATCHES['plain'],
                        asian_data=asian, goals_data=goals))
    for cs_name, cs in CS.items():
        yield (f'spf_cs:{cs_name}',
               _run(rec_mod.analyze_spf, OUZHI['home_favourite'],
                    _no_calibration, MATCHES['plain'], cs_data=cs))

    # 让球胜平负：让球值的每一种形态 × 官方赔率的每一种来源
    for match_name in ('plain', 'handicap_plus_one', 'handicap_minus_two',
                       'handicap_zero', 'no_handicap', 'unparsable_handicap',
                       'official_rqspf', 'lottery_rqspf', 'rqspf_prices',
                       'rqspf_prices_below_one', 'rqspf_prices_dirty',
                       'with_okooo_main', 'partial_okooo_main'):
        for odds_name, ouzhi in OUZHI.items():
            for cal_name, calibrator in CALIBRATORS.items():
                yield (f'rqspf:{match_name}:{odds_name}:{cal_name}',
                       _run(rec_mod.analyze_rqspf, ouzhi, calibrator,
                            MATCHES[match_name]))
    for asian_name, asian in ASIAN.items():
        for goals_name, goals in GOALS.items():
            yield (f'rqspf_market:{asian_name}:{goals_name}',
                   _run(rec_mod.analyze_rqspf, OUZHI['balanced'],
                        _no_calibration, MATCHES['plain'],
                        asian_data=asian, goals_data=goals))

    # 比分：市场报价的有无是这一路最重要的分叉（两套键在那里并列）
    for match_name in ('plain', 'unknown_league', 'with_okooo_main',
                       'partial_okooo_main'):
        for odds_name, ouzhi in OUZHI.items():
            for market_name, market in BIFEN_ODDS.items():
                for cal_name, calibrator in CALIBRATORS.items():
                    yield (f'bifen:{match_name}:{odds_name}:{market_name}:{cal_name}',
                           _run(rec_mod.analyze_bifen, ouzhi, calibrator,
                                MATCHES[match_name], market))
    for asian_name, asian in ASIAN.items():
        for goals_name, goals in GOALS.items():
            yield (f'bifen_market:{asian_name}:{goals_name}',
                   _run(rec_mod.analyze_bifen, OUZHI['home_favourite'],
                        _no_calibration, MATCHES['plain'],
                        BIFEN_ODDS['present'], asian, goals))

    # 总进球
    for match_name in ('plain', 'unknown_league', 'with_okooo_main',
                       'partial_okooo_main'):
        for odds_name, ouzhi in OUZHI.items():
            for market_name, market in ZJQ_ODDS.items():
                for cal_name, calibrator in CALIBRATORS.items():
                    yield (f'zjq:{match_name}:{odds_name}:{market_name}:{cal_name}',
                           _run(rec_mod.analyze_zjq, ouzhi, calibrator,
                                MATCHES[match_name], market))
    for asian_name, asian in ASIAN.items():
        for goals_name, goals in GOALS.items():
            yield (f'zjq_market:{asian_name}:{goals_name}',
                   _run(rec_mod.analyze_zjq, OUZHI['away_favourite'],
                        _no_calibration, MATCHES['plain'],
                        ZJQ_ODDS['present'], asian, goals))


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
