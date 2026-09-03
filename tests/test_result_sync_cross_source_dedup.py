# -*- coding: utf-8 -*-
"""换源不得给同一场比赛再建一条记录。

赛程源换成竞彩官网时 match_id 从 500 的数字 fid 变成 `sporttery_*`，
只按 match_id 查重会把同一场比赛写成两条。竞彩编号+开赛时间是两边都稳定的
业务键，应当作为次级查重键复用已有记录。
"""

from src.football.result_sync import PredictionHistory


MATCH_TIME = '2020-09-03 00:00'


def _history():
    history = PredictionHistory()
    history._save = lambda: None
    history._save_record = lambda saved: 'memory'
    history.records = []
    return history


def _predict(history, match_id, league, **overrides):
    kwargs = {
        'match_id': match_id,
        'league': league,
        'home': '乌迪内斯',
        'away': '威尼斯',
        'match_time': MATCH_TIME,
        'match_num': '周三006',
        'predicted_scores': {'2-1': 0.11},
        'predicted_1x2': {'H': 0.53, 'D': 0.26, 'A': 0.21},
    }
    kwargs.update(overrides)
    return history.add_prediction(**kwargs)


def test_same_lottery_number_reuses_the_existing_record_across_sources():
    history = _history()
    _predict(history, '1474126', '意杯')

    _predict(history, 'sporttery_2041241', '意大利杯', lottery_handicap=-1,
             predicted_rqspf={'让胜': 0.53, '让平': 0.47})

    assert len(history.records) == 1
    record = history.records[0]
    assert record['league'] == '意大利杯'
    assert record['lottery_handicap'] == -1
    assert record['predicted_rqspf'] == {'让胜': 0.53, '让平': 0.47}


def test_reused_record_keeps_its_original_match_id_and_records_the_alias():
    """身份先到先得：换 match_id 要删旧行，反而制造第二种不一致。"""
    history = _history()
    _predict(history, '1474126', '意杯')

    _predict(history, 'sporttery_2041241', '意大利杯')

    record = history.records[0]
    assert record['match_id'] == '1474126'
    assert 'sporttery_2041241' in record.get('alias_match_ids', [])


def test_different_lottery_number_still_creates_a_separate_record():
    history = _history()
    _predict(history, '1474126', '意杯')

    _predict(history, 'sporttery_2041242', '德国杯', match_num='周三010',
             home='奥斯纳', away='拜仁')

    assert len(history.records) == 2


def test_missing_lottery_number_falls_back_to_match_id_only():
    """没有竞彩编号时不能瞎并：非竞彩场次的编号字段是空的。"""
    history = _history()
    _predict(history, '1474126', '意杯', match_num=None)

    _predict(history, 'sporttery_2041241', '意大利杯', match_num=None)

    assert len(history.records) == 2
