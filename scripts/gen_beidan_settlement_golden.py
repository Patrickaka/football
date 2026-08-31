"""从 beidan 的 settling 实现生成黄金快照。

**语料的形状来自线上真实记录**（判据 10）：2026-08-28 读 `beidan_prediction_history`
的 500 条，顶层字段是 `key/source/match_id/date/num/time/league/home/away/
handicap/created_at/updated_at/settled/professional_snapshot/spf/zjq/rqspf/
market_layers`，三个玩法分节各带 `probabilities`。

线上读到的三件事直接决定了语料怎么铺：

1. **`handicap` 存的是字符串 `'(-1)'`**（143 条有值：`(-1)` 83、`(+1)` 32、
   `(-2)` 23、`(+2)` 5；另 357 条为 `None`），不是数字。所以让球胜平负那组
   必须同时喂字符串盘口与数字盘口。
2. **500 条 `settled` 全是 False**，仓库里没有任何一处会把它置 True。
   已结算记录在线上根本不存在，所以校准那组的语料只能构造——
   构造时按 `save_beidan_prediction_snapshot` 会保留的 `actual`/`settlement`
   两个字段来写。
3. **`zjq` 的概率键是 `'0'`~`'6'` 与 `'7+'` 的字符串**，`bifen` 分节一条都没有。

用法：
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 \\
        scripts/gen_beidan_settlement_golden.py /tmp/beidan_settlement_old.json
"""
import json
import sys
from unittest import mock

sys.path.insert(0, '.')

from tests.domain.golden import as_comparable

import src.beidan.settling as settling_mod

# ── 赔率 → 去水概率 ──────────────────────────────────────────────
# 含 0、None、负数三种脏值：它们各自走 `if o and o > 0` 的不同分支，
# 而分母与分子用的是两套过滤条件（判据 17 的形状），只喂干净赔率测不出来
ODDS_SETS = {
    'spf': {'胜': 2.10, '平': 3.30, '负': 3.60},
    'two_way': {'大': 1.85, '小': 1.95},
    'with_zero': {'胜': 2.10, '平': 0, '负': 3.60},
    'with_none': {'胜': 2.10, '平': None, '负': 3.60},
    'with_negative': {'胜': 2.10, '平': -3.0, '负': 3.60},
    'all_dead': {'胜': 0, '平': None},
    'single': {'胜': 1.01},
    'empty': {},
}

# ── 已结算记录：五个取值候选各自单独出现一次 ──────────────────────
# 旧实现在四个提取函数里写了四份候选链，而 `_actual_rqspf_from_record` 的
# 顺序与另外三个不同（`record['actual_score']` 在它那里排最后、在别处排最前）。
# 每个候选单独给一条语料，顺序改了才会有一条变红。
def _record(**fields):
    """按线上真实顶层字段拼一条记录，只覆盖测试关心的那几个。"""
    base = {
        'key': '2026-08-28|1|安山小绿人|大邱FC',
        'source': 'zgzcw', 'match_id': '1320957',
        'date': '2026-08-28', 'num': '1', 'time': '18:30',
        'league': 'K2联赛', 'home': '安山小绿人', 'away': '大邱FC',
        'handicap': None,
        'created_at': '2026-08-28T10:37:15',
        'updated_at': '2026-08-28T10:37:15',
        'settled': True,
    }
    base.update(fields)
    return base


RECORDS = {
    'top_actual_score': _record(actual_score='2-1'),
    'actual_dot_score': _record(actual={'score': '2-1'}),
    'actual_dot_actual_score': _record(actual={'actual_score': '2-1'}),
    'settlement_dot_score': _record(settlement={'score': '2-1'}),
    'settlement_dot_actual_score': _record(settlement={'actual_score': '2-1'}),
    # 两处同时有值且不一致——**这一条是唯一能分辨候选顺序的语料**
    'top_and_actual_disagree': _record(actual_score='2-1', actual={'score': '0-3'}),
    # 直取字段：spf 与 zjq 有这条捷径，bifen 与 rqspf 没有
    'direct_spf': _record(actual_spf='平', actual_score='2-1'),
    'direct_spf_invalid': _record(actual_spf='胜负', actual_score='2-1'),
    'direct_zjq_str': _record(actual_zjq='3', actual_score='2-1'),
    'direct_zjq_int': _record(actual_zjq=5, actual_score='2-1'),
    'direct_zjq_plus': _record(actual_zjq='7+', actual_score='2-1'),
    'direct_zjq_out_of_range': _record(actual_zjq='9', actual_score='2-1'),
    # `0 or ...` 会把「零球」当成缺失继续往下找候选——真实结果里 0 球很常见
    'direct_zjq_zero': _record(actual_zjq=0, actual_score='2-1'),
    'direct_zjq_zero_no_score': _record(actual_zjq=0),
    # 比分本身的各种形态
    'draw': _record(actual_score='1-1'),
    'away_win': _record(actual_score='0-2'),
    'goalless': _record(actual_score='0-0'),
    'seven_goals': _record(actual_score='4-3'),
    'eight_goals': _record(actual_score='5-3'),
    'no_dash': _record(actual_score='21'),
    'colon': _record(actual_score='2:1'),
    'not_int': _record(actual_score='a-b'),
    'empty_score': _record(actual_score=''),
    'missing': _record(),
    # `actual` / `settlement` 不是 dict 时旧实现会退回 {}
    'actual_not_dict': _record(actual='2-1'),
    'settlement_not_dict': _record(settlement=['2-1']),
    # 让球：线上真实形态是带括号的字符串
    'hc_str_minus_one': _record(handicap='(-1)', actual={'score': '1-0'}),
    'hc_str_plus_one': _record(handicap='(+1)', actual={'score': '0-1'}),
    'hc_str_minus_two': _record(handicap='(-2)', actual={'score': '3-1'}),
    'hc_str_plus_two': _record(handicap='(+2)', actual={'score': '0-2'}),
    'hc_fullwidth': _record(handicap='（-1）', actual={'score': '1-0'}),
    'hc_numeric': _record(handicap=-1, actual={'score': '1-0'}),
    'hc_numeric_float': _record(handicap=-1.0, actual={'score': '2-1'}),
    'hc_numeric_str': _record(handicap='-1', actual={'score': '1-0'}),
    'hc_zero': _record(handicap=0, actual={'score': '1-1'}),
    'hc_none': _record(handicap=None, actual={'score': '1-0'}),
    'hc_garbage': _record(handicap='公司', actual={'score': '1-0'}),
    'hc_str_no_score': _record(handicap='(-1)'),
    'hc_str_bad_score': _record(handicap='(-1)', actual={'score': 'x'}),
    # rqspf 的候选链把顶层 actual_score 排在最后
    'hc_top_score_only': _record(handicap='(-1)', actual_score='1-0'),
}

# ── 校准语料 ─────────────────────────────────────────────────────
# 概率键用线上真实的键集：spf 三档、rqspf 三档、zjq 八档（含 '7+'）。
SPF_PROBS = {'胜': 0.40, '平': 0.30, '负': 0.30}
RQSPF_PROBS = {'让胜': 0.30, '让平': 0.18, '让负': 0.52}
ZJQ_PROBS = {'0': 0.03, '1': 0.05, '2': 0.14, '3': 0.21,
             '4': 0.25, '5': 0.14, '6': 0.10, '7+': 0.08}
# bifen 的调用方（`recommending.py:852`）在归一化成 'h-a' 字符串**之前**
# 就调了校准，所以传进来的是元组键的矩阵。这一条不是臆造的形态。
BIFEN_TUPLE_PROBS = {(1, 0): 0.12, (1, 1): 0.11, (0, 1): 0.09}
BIFEN_STR_PROBS = {'1-0': 0.12, '1-1': 0.11, '0-1': 0.09}


def _settled(bet_type, probs, score, league='K2联赛', handicap=None):
    return _record(league=league, handicap=handicap, actual={'score': score},
                   **{bet_type: {'probabilities': probs}})


def _history(bet_type, probs, scores, league='K2联赛', handicap=None):
    return [_settled(bet_type, probs, s, league=league, handicap=handicap)
            for s in scores]


# 主场三连胜：实际全是 '胜'，而预测里 '胜' 只占 0.40 —— 因子应当抬高 '胜'
SPF_HOME_HEAVY = ['2-1', '3-0', '1-0', '2-0', '4-1', '1-0', '3-1', '2-0', '1-0', '2-1']
SPF_MIXED = ['2-1', '1-1', '0-2', '1-0', '1-1', '0-1', '3-1', '1-1', '0-2', '2-0']

HISTORIES = {
    'empty': [],
    # 一条都没结算 —— **这正是线上此刻的状态**（500 条 settled 全是 False）
    'none_settled': [dict(r, settled=False) for r in
                     _history('spf', SPF_PROBS, SPF_HOME_HEAVY)],
    'spf_home_heavy': _history('spf', SPF_PROBS, SPF_HOME_HEAVY),
    'spf_mixed': _history('spf', SPF_PROBS, SPF_MIXED),
    # 样本刚好差一条（min_samples=8 时给 7 条）
    'spf_seven': _history('spf', SPF_PROBS, SPF_HOME_HEAVY[:7]),
    'spf_eight': _history('spf', SPF_PROBS, SPF_HOME_HEAVY[:8]),
    # 联赛加权：同联赛每条算 1.25，异联赛算 1.0
    'spf_other_league': _history('spf', SPF_PROBS, SPF_HOME_HEAVY, league='英超'),
    # 分节存在但没有 probabilities / 分节不是 dict
    'spf_no_probs': [_record(spf={'prediction': '胜'}) for _ in range(10)],
    'spf_section_not_dict': [_record(spf='胜') for _ in range(10)],
    # 实际结果解析不出来 → 每条都被 `actual not in expected` 挡掉
    'spf_unparsable': _history('spf', SPF_PROBS, ['x'] * 10),
    'zjq': _history('zjq', ZJQ_PROBS, SPF_MIXED),
    'rqspf_str_handicap': _history('rqspf', RQSPF_PROBS, SPF_HOME_HEAVY,
                                   handicap='(-1)'),
    'rqspf_numeric_handicap': _history('rqspf', RQSPF_PROBS, SPF_HOME_HEAVY,
                                       handicap=-1),
    'rqspf_no_handicap': _history('rqspf', RQSPF_PROBS, SPF_HOME_HEAVY),
    'bifen': _history('bifen', BIFEN_STR_PROBS, ['1-0', '1-1', '0-1'] * 4),
}

# (用例名, 历史语料名, 概率, bet_type, league)
CALIBRATION_CASES = [
    ('empty_probs', 'spf_home_heavy', {}, 'spf', None),
    ('no_history', 'empty', SPF_PROBS, 'spf', None),
    ('none_settled', 'none_settled', SPF_PROBS, 'spf', None),
    ('spf_home_heavy', 'spf_home_heavy', SPF_PROBS, 'spf', None),
    ('spf_home_heavy_league', 'spf_home_heavy', SPF_PROBS, 'spf', 'K2联赛'),
    ('spf_mixed', 'spf_mixed', SPF_PROBS, 'spf', None),
    ('spf_seven', 'spf_seven', SPF_PROBS, 'spf', None),
    ('spf_eight', 'spf_eight', SPF_PROBS, 'spf', None),
    ('spf_other_league', 'spf_other_league', SPF_PROBS, 'spf', 'K2联赛'),
    ('spf_no_probs', 'spf_no_probs', SPF_PROBS, 'spf', None),
    ('spf_section_not_dict', 'spf_section_not_dict', SPF_PROBS, 'spf', None),
    ('spf_unparsable', 'spf_unparsable', SPF_PROBS, 'spf', None),
    ('zjq', 'zjq', ZJQ_PROBS, 'zjq', None),
    ('rqspf_str_handicap', 'rqspf_str_handicap', RQSPF_PROBS, 'rqspf', None),
    ('rqspf_numeric_handicap', 'rqspf_numeric_handicap', RQSPF_PROBS, 'rqspf', None),
    ('rqspf_no_handicap', 'rqspf_no_handicap', RQSPF_PROBS, 'rqspf', None),
    ('bifen_tuple_keys', 'bifen', BIFEN_TUPLE_PROBS, 'bifen', None),
    ('bifen_str_keys', 'bifen', BIFEN_STR_PROBS, 'bifen', None),
    # 未知玩法：提取器返回 None，一条都攒不上
    ('unknown_bet_type', 'spf_home_heavy', SPF_PROBS, 'dxq', None),
]

# 记录键：`num` 缺失、字段含分隔符等
KEY_MATCHES = {
    'full': {'date': '2026-08-28', 'num': '1', 'home': '安山小绿人', 'away': '大邱FC'},
    'missing_num': {'date': '2026-08-28', 'home': '安山小绿人', 'away': '大邱FC'},
    'all_missing': {},
    'extra_fields': {'date': '2026-08-28', 'num': 3, 'home': 'A', 'away': 'B', 'x': 'y'},
    'pipe_in_name': {'date': '2026-08-28', 'num': '1', 'home': 'A|B', 'away': 'C'},
}


def entries():
    for name, odds in ODDS_SETS.items():
        yield f'implied:{name}', settling_mod.calculate_implied_probability(odds)

    for name, record in RECORDS.items():
        yield f'spf:{name}', settling_mod._actual_spf_from_record(record)
        yield f'zjq:{name}', settling_mod._actual_zjq_from_record(record)
        yield f'bifen:{name}', settling_mod._actual_bifen_from_record(record)
        yield f'rqspf:{name}', settling_mod._actual_rqspf_from_record(record)

    for name, history_name, probs, bet_type, league in CALIBRATION_CASES:
        history = HISTORIES[history_name]
        with mock.patch.object(settling_mod, '_load_beidan_history',
                               return_value=history):
            adjusted, meta = settling_mod.apply_beidan_history_calibration(
                probs, bet_type, league=league)
        yield f'calib:{name}:probs', adjusted
        yield f'calib:{name}:meta', meta
        # 样本门槛的两侧：默认 8，另取 1 与 40
        for min_samples in (1, 40):
            with mock.patch.object(settling_mod, '_load_beidan_history',
                                   return_value=history):
                yield (f'calib:{name}:min{min_samples}',
                       settling_mod.apply_beidan_history_calibration(
                           probs, bet_type, league=league,
                           min_samples=min_samples)[1])
        # 截断窗口：limit 小于样本数时只看前几条
        for limit in (3, 200):
            with mock.patch.object(settling_mod, '_load_beidan_history',
                                   return_value=history):
                yield (f'calib:{name}:limit{limit}',
                       settling_mod.apply_beidan_history_calibration(
                           probs, bet_type, league=league, limit=limit)[1])

    for name, match in KEY_MATCHES.items():
        yield f'key:{name}', settling_mod._beidan_record_key(match)


def main(out_path):
    golden = {key: as_comparable(value) for key, value in entries()}
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(golden, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f'共 {len(golden)} 条 → {out_path}')


if __name__ == '__main__':
    main(sys.argv[1])
