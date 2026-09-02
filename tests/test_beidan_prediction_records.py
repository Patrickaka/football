# -*- coding: utf-8 -*-
"""北单预测记录：列表契约、赛果回填与前端入口。"""

from datetime import datetime
from pathlib import Path
from unittest import mock

from src.beidan import settling


def _pending_record():
    return {
        'key': '2026-08-28|301|主队|客队',
        'match_id': '12345',
        'date': '2026-08-28',
        'time': '12:30',
        'num': '301',
        'league': '测试联赛',
        'home': '主队',
        'away': '客队',
        'handicap': '(-1)',
        'created_at': '2026-08-28T10:00:00',
        'settled': False,
        'spf': {'prediction': '胜', 'probabilities': {'胜': 0.6, '平': 0.2, '负': 0.2}},
        'rqspf': {'prediction': '让平', 'probabilities': {'让胜': 0.2, '让平': 0.6, '让负': 0.2}},
        'zjq': {'prediction': '1', 'probabilities': {'0': 0.1, '1': 0.5, '2': 0.4}},
    }


def test_beidan_record_view_normalizes_legacy_fields():
    with mock.patch.object(settling, '_load_beidan_history', return_value=[_pending_record()]):
        rows = settling.get_beidan_prediction_records()
    assert rows[0]['match_num'] == '301'
    assert rows[0]['match_time'] == '2026-08-28 12:30'
    assert rows[0]['sync_status'] == 'pending'
    assert 'market_layers' not in rows[0]


def test_beidan_result_sync_settles_all_saved_markets():
    records = [_pending_record()]
    saved = []
    with mock.patch.object(settling, '_load_beidan_history', return_value=records), \
         mock.patch.object(settling, '_save_beidan_history', side_effect=lambda rows: saved.extend(rows)), \
         mock.patch.object(settling, 'log'):
        result = settling.sync_beidan_results(
            fetch_by_id=lambda match_id, match_time: {'score': '1-0', 'source': 'test'},
            fetch_by_team=lambda home, away, match_time: None,
            now=datetime(2026, 8, 29, 12, 0),
        )
    record = records[0]
    assert result['synced'] == 1
    assert record['settled'] is True
    assert record['sync_status'] == 'synced'
    assert record['actual_spf'] == '胜'
    assert record['actual_rqspf'] == '让平'
    assert record['actual_zjq'] == '1'
    assert record['hit_spf'] is True
    assert record['hit_rqspf'] is True
    assert record['hit_zjq'] is True
    assert saved


def test_beidan_result_sync_waits_until_three_hours_after_kickoff():
    records = [_pending_record()]
    fetch_by_id = mock.Mock(return_value={'score': '1-0'})
    with mock.patch.object(settling, '_load_beidan_history', return_value=records), \
         mock.patch.object(settling, '_save_beidan_history') as save:
        result = settling.sync_beidan_results(
            fetch_by_id=fetch_by_id,
            fetch_by_team=mock.Mock(),
            now=datetime(2026, 8, 28, 14, 0),
        )
    assert result['total'] == 0
    fetch_by_id.assert_not_called()
    save.assert_not_called()


def test_beidan_frontend_has_record_switch_calendar_and_sync():
    html = Path('web/index.html').read_text(encoding='utf-8')
    assert "openBeidanPredictionRecords()" in html
    assert "fetchJson('/api/beidan/records')" in html
    assert "fetchJson('/api/beidan/sync', { method: 'POST' })" in html
    assert '北单预测记录' in html
