from copy import deepcopy
from datetime import datetime, timedelta, timezone
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.football.storage_maintenance import (
    archive_stale_prediction_records, run_football_storage_maintenance,
)

NOW = datetime(2035, 3, 1, tzinfo=timezone.utc)


def history(count=5):
    # Deliberately thin memory rows: archive must read persisted documents.
    return SimpleNamespace(_records_lock=RLock(), records=[
        {'match_id': str(index), 'settled': True,
         'settled_at': (NOW-timedelta(days=35)).isoformat(),
         '_timeline_offloaded': True, 'market_timeline': [{'last': True}]}
        for index in range(count)])


def test_archive_batch_rotates_and_never_writes_a_thin_memory_copy():
    subject = history()
    before = deepcopy(subject.records)
    with patch('src.common.repositories.football_prediction_archive_one',
               return_value={'status': 'archived', 'bytes_saved': 100}) as archive:
        one = archive_stale_prediction_records(history=subject, now=NOW, max_records=2)
        two = archive_stale_prediction_records(history=subject, now=NOW, max_records=2)
        assert [call.args[0] for call in archive.call_args_list] == ['0', '1', '2', '3']
        assert all(not call.args[1:] for call in archive.call_args_list)
    assert one['archived_count'] == two['archived_count'] == 2
    assert one['bytes_saved_estimate'] == 200
    assert subject.records == before


def test_dry_run_does_not_advance_cursor_and_errors_do_not_stop_other_rows():
    subject = history(3)
    before = deepcopy(subject.__dict__['records'])
    with patch('src.common.repositories.football_prediction_archive_one', side_effect=[
        OSError('broken archive'), {'status': 'would_archive', 'bytes_saved': 123},
        {'status': 'already_archived'},
    ]) as archive:
        result = archive_stale_prediction_records(history=subject, now=NOW, dry_run=True)
        assert all(call.kwargs['dry_run'] for call in archive.call_args_list)
    assert result['would_archive_count'] == 1
    assert result['already_archived_count'] == 1
    assert result['bytes_saved_estimate'] == 123
    assert len(result['errors']) == 1
    assert not hasattr(subject, '_storage_archive_cursor')
    assert subject.records == before


def test_archive_excludes_live_recent_future_and_unknown_settlement_time():
    subject = history(0)
    subject.records = [
        {'match_id': 'live', 'settled': False, 'settled_at': '2034-01-01T00:00:00Z'},
        {'match_id': 'recent', 'settled': True, 'settled_at': '2035-02-28T00:00:00Z'},
        {'match_id': 'future', 'settled': True, 'settled_at': '2036-01-01T00:00:00Z'},
        {'match_id': 'unknown', 'settled': True},
        {'match_id': 'ambiguous', 'settled': True, 'settled_at': '2034-01-01 00:00:00'},
    ]
    with patch('src.common.repositories.football_prediction_archive_one') as archive:
        result = archive_stale_prediction_records(history=subject, now=NOW)
    archive.assert_not_called()
    assert result['eligible_count'] == result['checked_count'] == 0


def test_stages_are_independent_and_propagate_dry_run():
    with patch('src.football.intelligence.retention.cleanup_intelligence_cache',
               side_effect=OSError('cache failure')), \
         patch('src.football.storage_maintenance.archive_stale_prediction_records',
               return_value={'archived_count': 0}) as archive:
        result = run_football_storage_maintenance(dry_run=True, now=NOW)
    archive.assert_called_once_with(history=None, dry_run=True, now=NOW)
    assert result['intelligence_cache']['errors'] == ['cache failure']
    assert result['prediction_archive']['archived_count'] == 0


def test_corrupt_archive_is_not_loaded_as_empty_history():
    from src.common.football_storage import FootballStorageError
    from src.football.result_sync import PredictionHistory
    subject = PredictionHistory.__new__(PredictionHistory)
    subject.records = [{'match_id': 'existing'}]
    with patch('src.common.repositories.football_prediction_load',
               side_effect=FootballStorageError('invalid hash')):
        with pytest.raises(FootballStorageError):
            subject._load()
    assert subject.records == [{'match_id': 'existing'}]


def test_nested_archives_export_every_original_event_and_preserve_evaluation():
    from src.common import repositories
    from src.common.football_storage import encode_record, is_archived
    from src.football import result_sync
    from src.football.prediction_events import append_prediction_event, hydrate_prediction_events
    from src.domain.sports.football.prediction_evaluation import evaluate_frozen_events
    from tests.test_football_prediction_events import AS_OF, payload, record

    row = record()
    row['settled'] = False
    first = deepcopy(row['prediction_events'][0])
    original_events = [first]
    for index in range(1, 14):
        when = AS_OF + timedelta(minutes=index)
        candidate = payload(when)
        candidate['kickoff_at'] = first['kickoff_at']
        candidate['context']['market_update'] = index
        result = append_prediction_event(row, candidate, now=when+timedelta(seconds=1))
        assert result['appended'], result
        original_events.append(deepcopy(row['prediction_events'][-1]))
    row['settled'] = True
    assert len(row['prediction_events']) == 8
    assert hydrate_prediction_events(row) == original_events
    assert row['prediction_events'][0] == first
    before = evaluate_frozen_events([row], include_confidence_intervals=False)
    stored = encode_record(row, now=NOW)
    assert is_archived(stored)
    with patch('src.common.doc_store.load_all', return_value=[stored]):
        restored = repositories.football_prediction_load()[0]
    assert restored == row
    assert evaluate_frozen_events([restored], include_confidence_intervals=False) == before
    subject = result_sync.PredictionHistory.__new__(result_sync.PredictionHistory)
    subject.records = [restored]
    subject.get_stats = lambda: {}
    subject.get_frozen_evaluation_stats = lambda: before
    with patch.object(result_sync, '_global_history', subject):
        exported = result_sync.get_prediction_export()['records'][0]
    assert exported['prediction_events'] == original_events
    assert exported['selected_prediction_event_id'] == first['event_id']
    assert 'prediction_event_archive' not in exported
    assert 'prediction_event_retention' not in exported
    assert evaluate_frozen_events([exported], include_confidence_intervals=False) == before
    from src.common.football_storage import FootballStorageError
    subject.records = [deepcopy(restored)]
    subject.records[0].pop('prediction_event_archive')
    with patch.object(result_sync, '_global_history', subject):
        with pytest.raises(FootballStorageError):
            result_sync.get_prediction_export()
