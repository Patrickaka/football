"""已结算多日的记录只在内存里留时间线的最后一条快照。

线上 830 条记录常驻内存约 420 MB，其中 71% 是 market_timeline（每条最多
80 个快照、每个 12.5 KB）。老记录的完整时间线只在写库与导出时按需从库读回，
内存副本永远不能把截断后的时间线写回库里。
"""
import unittest
from datetime import datetime
from unittest import mock

from src.football import result_sync
from src.football.result_sync import PredictionHistory

NOW = datetime(2026, 9, 7, 10, 0)


def _timeline(n):
    return [{'captured_at': f'2026-08-20T1{i}:00:00', 'odds': {'i': i}, 'signature': str(i)}
            for i in range(n)]


def _record(match_id, match_time, settled, timeline_len=5):
    return {
        'match_id': match_id, 'match_time': match_time, 'settled': settled,
        'home': 'A', 'away': 'B', 'market_timeline': _timeline(timeline_len),
    }


def _history(records):
    history = PredictionHistory.__new__(PredictionHistory)
    history.records = records
    return history


class OffloadTests(unittest.TestCase):

    def test_old_settled_records_keep_only_the_last_snapshot(self):
        old_settled = _record('old', '2026-08-20 20:00', True)
        recent_settled = _record('recent', '2026-09-06 20:00', True)
        old_unsettled = _record('pending', '2026-08-20 20:00', False)
        history = _history([old_settled, recent_settled, old_unsettled])

        offloaded = history.offload_stale_timelines(now=NOW)

        self.assertEqual(offloaded, 1)
        self.assertEqual(old_settled['market_timeline'], _timeline(5)[-1:])
        self.assertTrue(old_settled[result_sync.TIMELINE_OFFLOADED])
        self.assertEqual(len(recent_settled['market_timeline']), 5)
        self.assertEqual(len(old_unsettled['market_timeline']), 5)
        self.assertNotIn(result_sync.TIMELINE_OFFLOADED, recent_settled)

    def test_offloading_twice_is_idempotent(self):
        record = _record('old', '2026-08-20 20:00', True)
        history = _history([record])
        history.offload_stale_timelines(now=NOW)

        self.assertEqual(history.offload_stale_timelines(now=NOW), 0)
        self.assertEqual(len(record['market_timeline']), 1)


class PersistTests(unittest.TestCase):

    def test_saving_an_offloaded_record_writes_the_full_timeline_back(self):
        record = _record('old', '2026-08-20 20:00', True)
        history = _history([record])
        history.offload_stale_timelines(now=NOW)
        record['sync_status'] = 'ignored'
        stored = {'match_id': 'old', 'market_timeline': _timeline(5)}
        with mock.patch.object(result_sync.repositories, 'football_prediction_get',
                               return_value=stored), \
             mock.patch.object(result_sync.repositories, 'football_prediction_upsert',
                               return_value='mysql') as upsert:
            history._save_record(record)

        written = upsert.call_args.args[0]
        self.assertEqual(written['market_timeline'], _timeline(5))
        self.assertEqual(written['sync_status'], 'ignored')
        self.assertNotIn(result_sync.TIMELINE_OFFLOADED, written)
        self.assertEqual(len(record['market_timeline']), 1)

    def test_a_record_that_was_never_offloaded_is_written_as_is(self):
        record = _record('recent', '2026-09-06 20:00', True)
        history = _history([record])
        with mock.patch.object(result_sync.repositories, 'football_prediction_get') as get, \
             mock.patch.object(result_sync.repositories, 'football_prediction_upsert',
                               return_value='mysql') as upsert:
            history._save_record(record)

        get.assert_not_called()
        self.assertIs(upsert.call_args.args[0], record)

    def test_whole_table_save_merges_every_offloaded_record(self):
        old = _record('old', '2026-08-20 20:00', True)
        recent = _record('recent', '2026-09-06 20:00', True)
        history = _history([old, recent])
        history.offload_stale_timelines(now=NOW)
        with mock.patch.object(result_sync.repositories, 'football_prediction_get',
                               return_value={'match_id': 'old', 'market_timeline': _timeline(5)}), \
             mock.patch.object(result_sync.repositories, 'football_prediction_upsert',
                               return_value='mysql') as upsert, \
             mock.patch.object(result_sync.repositories, 'football_prediction_save') as save:
            history._save()

        save.assert_not_called()
        written = {call.args[0]['match_id']: call.args[0] for call in upsert.call_args_list}
        self.assertEqual(len(written['old']['market_timeline']), 5)
        self.assertEqual(len(written['recent']['market_timeline']), 5)


class HydrateTests(unittest.TestCase):

    def test_hydrating_restores_the_timeline_in_memory(self):
        record = _record('old', '2026-08-20 20:00', True)
        history = _history([record])
        history.offload_stale_timelines(now=NOW)
        with mock.patch.object(result_sync.repositories, 'football_prediction_get',
                               return_value={'match_id': 'old', 'market_timeline': _timeline(5)}):
            history._hydrate_timeline(record)

        self.assertEqual(record['market_timeline'], _timeline(5))
        self.assertNotIn(result_sync.TIMELINE_OFFLOADED, record)

    def test_export_carries_the_full_timeline_without_touching_memory(self):
        record = _record('old', '2026-08-20 20:00', True)
        history = _history([record])
        history.offload_stale_timelines(now=NOW)
        with mock.patch.object(result_sync, '_global_history', history), \
             mock.patch.object(result_sync.repositories, 'football_prediction_get',
                               return_value={'match_id': 'old', 'market_timeline': _timeline(5)}):
            exported = result_sync.get_prediction_export()

        self.assertEqual(len(exported['records'][0]['market_timeline']), 5)
        self.assertEqual(len(record['market_timeline']), 1)


if __name__ == '__main__':
    unittest.main()
