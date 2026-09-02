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


def test_manual_sync_can_drain_more_than_the_periodic_batch_limit():
    records = []
    for index in range(35):
        record = _pending_record()
        record.update({
            'key': f'2026-08-28|{index}|主队{index}|客队{index}',
            'match_id': str(10000 + index),
            'num': str(index),
        })
        records.append(record)

    with mock.patch.object(settling, '_load_beidan_history', return_value=records), \
         mock.patch.object(settling, '_save_beidan_history'):
        result = settling.sync_beidan_results(
            fetch_by_id=lambda match_id, match_time: {
                'score': '1-0', 'source': 'test',
            },
            fetch_by_team=mock.Mock(),
            now=datetime(2026, 8, 29, 12, 0),
            batch_size=None,
        )

    assert result['total'] == 35
    assert result['synced'] == 35
    assert result['remaining'] == 0


def test_periodic_sync_reports_the_true_unprocessed_remainder():
    records = []
    for index in range(35):
        record = _pending_record()
        record.update({
            'key': f'2026-08-28|{index}|主队{index}|客队{index}',
            'match_id': str(10000 + index),
        })
        records.append(record)

    with mock.patch.object(settling, '_load_beidan_history', return_value=records), \
         mock.patch.object(settling, '_save_beidan_history'):
        result = settling.sync_beidan_results(
            fetch_by_id=lambda match_id, match_time: {
                'score': '1-0', 'source': 'test',
            },
            fetch_by_team=mock.Mock(),
            now=datetime(2026, 8, 29, 12, 0),
            batch_size=20,
        )

    assert result['total'] == 20
    assert result['remaining'] == 15


def test_long_sync_merges_into_latest_history_without_dropping_new_predictions():
    original = _pending_record()
    refreshed = _pending_record()
    refreshed['spf'] = {'prediction': '平', 'probabilities': {'平': 1.0}}
    newly_added = _pending_record()
    newly_added.update({
        'key': '2026-08-29|302|新主队|新客队',
        'match_id': '99999',
        'date': '2026-08-29',
    })
    saved = []

    with mock.patch.object(
            settling, '_load_beidan_history',
            side_effect=[[original], [refreshed, newly_added]]), \
         mock.patch.object(
             settling, '_save_beidan_history',
             side_effect=lambda rows: saved.extend(rows)):
        result = settling.sync_beidan_results(
            fetch_by_id=lambda match_id, match_time: {
                'score': '1-0', 'source': 'test',
            },
            fetch_by_team=mock.Mock(),
            now=datetime(2026, 8, 29, 12, 0),
        )

    assert result['synced'] == 1
    assert len(saved) == 2
    assert newly_added in saved
    merged = next(record for record in saved if record['key'] == original['key'])
    assert merged['settled'] is True
    assert merged['spf']['prediction'] == '平'


def test_non_500_sources_skip_invalid_cross_site_match_id_lookup():
    record = _pending_record()
    record['source'] = 'okooo'
    fetch_by_id = mock.Mock(return_value=None)
    fetch_by_team = mock.Mock(return_value={'score': '2-1', 'source': 'live_team'})

    with mock.patch.object(settling, '_load_beidan_history', return_value=[record]), \
         mock.patch.object(settling, '_save_beidan_history'):
        result = settling.sync_beidan_results(
            fetch_by_id=fetch_by_id,
            fetch_by_team=fetch_by_team,
            now=datetime(2026, 8, 29, 12, 0),
        )

    assert result['synced'] == 1
    fetch_by_id.assert_not_called()
    fetch_by_team.assert_called_once_with('主队', '客队', '2026-08-28 12:30')


def test_manual_sync_retries_backoff_and_terminal_failures():
    retrying = _pending_record()
    retrying.update({
        'sync_status': 'retry',
        'sync_attempts': 2,
        'next_sync_at': '2026-08-30T12:00:00',
    })
    failed = _pending_record()
    failed.update({
        'key': '2026-08-28|302|主队2|客队2',
        'match_id': '12346',
        'sync_status': 'failed',
        'sync_attempts': 5,
    })
    records = [retrying, failed]

    with mock.patch.object(settling, '_load_beidan_history', return_value=records), \
         mock.patch.object(settling, '_save_beidan_history'):
        result = settling.sync_beidan_results(
            fetch_by_id=lambda match_id, match_time: {
                'score': '1-0', 'source': 'test',
            },
            fetch_by_team=mock.Mock(),
            now=datetime(2026, 8, 29, 12, 0),
            batch_size=None,
            force_retry=True,
        )

    assert result['synced'] == 2
    assert all(record['settled'] for record in records)


def test_beidan_frontend_has_record_switch_calendar_and_sync():
    html = Path('web/index.html').read_text(encoding='utf-8')
    assert "openBeidanPredictionRecords()" in html
    assert "fetchJson('/api/beidan/records')" in html
    assert "fetchJson('/api/beidan/sync', { method: 'POST' })" in html
    assert "fetchJson('/api/beidan/sync/status')" in html
    assert '北单预测记录' in html
