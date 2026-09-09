"""Monitoring must not run the unrelated, expensive export statistics."""
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.football import result_sync


def test_records_only_export_preserves_rows_and_skips_statistics():
    history = SimpleNamespace(
        _records_lock=RLock(),
        records=[{'match_id': 'speed-test', 'match_time': '2026-09-09 20:00',
                  'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2}}],
        _persistable=lambda row: row,
        get_stats=Mock(return_value={'total': 1}),
        get_frozen_evaluation_stats=Mock(return_value={'n': 0}),
    )
    with patch.object(result_sync, '_global_history', history):
        light = result_sync.get_prediction_export(include_stats=False)
        history.get_stats.assert_not_called()
        history.get_frozen_evaluation_stats.assert_not_called()
        full = result_sync.get_prediction_export()
    assert light['records'] == full['records']
    assert light['stats'] == {}
    assert full['stats'] == {'total': 1, 'frozen_event_evaluation': {'n': 0}}
    history.get_stats.assert_called_once()
    history.get_frozen_evaluation_stats.assert_called_once()
