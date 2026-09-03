# -*- coding: utf-8 -*-
"""跨源重复记录结算时不得二次灌入派生库。

同一场比赛在换源期间留下了两条记录（500 数字 fid 一条、竞彩官网
`sporttery_*` 一条）。先结算的那条已经把 ELO/校准器/盘口库等派生库写过一遍，
保留下来的那条再结算时只应补齐赛果与命中，不能把同一场比赛再灌一次。
"""

from datetime import datetime
from unittest import mock

from src.football.result_sync import PredictionHistory


SETTLED_MATCH_TIME = '2020-09-02 21:00'
AFTER_MATCH = datetime(2020, 9, 3, 10, 0)

TRAINING_STORE_UPDATES = (
    '_update_calibrator',
    '_update_market_db',
    '_update_score_frequency_db',
    '_update_elo_ratings',
    '_update_half_time_stats',
    '_update_goal_count_stats',
    '_update_market_change_db',
)


def _history_with(record):
    history = PredictionHistory()
    history._save = lambda: None
    history._save_record = lambda saved: 'memory'
    history.records = [record]
    return history


def _record(**overrides):
    record = {
        'match_id': 'sporttery_2041215',
        'league': '沙职',
        'home': '利雅新月',
        'away': '吉达国民',
        'match_time': SETTLED_MATCH_TIME,
        'sync_status': 'ready',
        'settled': False,
        'predicted_1x2': {'H': 0.62, 'D': 0.22, 'A': 0.16},
    }
    record.update(overrides)
    return record


def _settle(history, record):
    with mock.patch.multiple(
            history, **{name: mock.DEFAULT for name in TRAINING_STORE_UPDATES}
    ) as stores:
        ok = history.update_result(
            record['match_id'], '3-0', 'H', source='live_team', now=AFTER_MATCH,
        )
    return ok, stores


def test_flagged_record_settles_but_skips_every_training_store():
    record = _record(skip_training_ingest=True)
    history = _history_with(record)

    ok, stores = _settle(history, record)

    assert ok is True
    assert record['settled'] is True
    assert record['actual_score'] == '3-0'
    assert record['hit_1x2'] is True
    for name in TRAINING_STORE_UPDATES:
        stores[name].assert_not_called()


def test_ordinary_record_still_feeds_every_training_store():
    """反向守卫：抑制开关不能把正常记录的派生库写入一起关掉。"""
    record = _record()
    history = _history_with(record)

    ok, stores = _settle(history, record)

    assert ok is True
    for name in TRAINING_STORE_UPDATES:
        stores[name].assert_called_once()
