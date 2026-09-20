# -*- coding: utf-8 -*-
"""已结算多日的记录，内存里对几个大字段只留下游真正要用的那点信息。

线上实测 1060 条常驻 299 MB，骨架就占 197.6 MB。骨架里最大的是
`odds_layers`（库里 25.55 MB），而它的唯一消费者 `monitoring` 只判断
四个键有没有值——不读任何内容。`last_prematch_odds_snapshot` 全仓没有
消费者，`closing_odds_snapshot` 也只看真假。

**摘要绝不能写回库**：写回前一律从库读回全文，读不回来就拒绝写，
绝不拿摘要覆盖——那会让全文永久消失，而且一声不响。
"""
import unittest
from datetime import datetime
from unittest import mock

from src.football import result_sync
from src.football.result_sync import PredictionHistory

NOW = datetime(2026, 9, 7, 10, 0)
OLD = '2026-08-20 20:00'
RECENT = '2026-09-06 20:00'


def _layer(tag):
    return {'odds': [1.9] * 40, 'captured_at': f'2026-08-20T{tag}', 'bulk': 'x' * 400}


def _record(match_id, match_time=OLD, settled=True, **overrides):
    record = {
        'match_id': match_id, 'match_time': match_time, 'settled': settled,
        'home': 'A', 'away': 'B',
        'market_timeline': [{'i': i} for i in range(5)],
        'odds_layers': {'T-24h': _layer('01'), 'T-6h': None, 'T-15min': _layer('02')},
        'closing_odds_snapshot': {'home': 1.9, 'draw': 3.4, 'away': 4.1},
        'last_prematch_odds_snapshot': {'home': 2.0, 'bulk': 'y' * 800},
        'time_layers': {'T-1h': {'1-0': .12}},
    }
    record.update(overrides)
    return record


def _history(records):
    history = PredictionHistory.__new__(PredictionHistory)
    history.records = records
    return history


class TrimTests(unittest.TestCase):
    def test_odds_layers_keeps_only_what_monitoring_checks(self):
        """monitoring 的两个判断只看这四个键的真假，从不读内容。"""
        record = _record('old')
        _history([record]).trim_stale_records(now=NOW)

        layers = record['odds_layers']
        self.assertTrue(any(layers.get(k) for k in ('T-24h', 'T-6h', 'T-1h', 'T-15min')))
        self.assertTrue(layers.get('T-15min'))
        self.assertFalse(layers.get('T-6h'))
        self.assertNotIn('odds', layers.get('T-24h') if isinstance(layers.get('T-24h'), dict) else {})

    def test_closing_snapshot_keeps_its_truthiness(self):
        record = _record('old')
        _history([record]).trim_stale_records(now=NOW)
        self.assertTrue(record['closing_odds_snapshot'])

    def test_a_falsy_closing_snapshot_stays_falsy(self):
        record = _record('old', closing_odds_snapshot=None)
        _history([record]).trim_stale_records(now=NOW)
        self.assertFalse(record.get('closing_odds_snapshot'))

    def test_the_unread_prematch_snapshot_is_dropped(self):
        record = _record('old')
        _history([record]).trim_stale_records(now=NOW)
        self.assertNotIn('last_prematch_odds_snapshot', record)

    def test_fields_that_are_still_read_in_full_are_untouched(self):
        """领域层回测读 time_layers 的内容，它必须原样留着。"""
        record = _record('old')
        _history([record]).trim_stale_records(now=NOW)
        self.assertEqual(record['time_layers'], {'T-1h': {'1-0': .12}})

    def test_recent_and_unsettled_records_are_untouched(self):
        recent = _record('recent', match_time=RECENT)
        pending = _record('pending', settled=False)
        _history([recent, pending]).trim_stale_records(now=NOW)

        for record in (recent, pending):
            self.assertIn('last_prematch_odds_snapshot', record)
            self.assertEqual(len(record['market_timeline']), 5)

    def test_trimming_twice_changes_nothing_more(self):
        record = _record('old')
        history = _history([record])
        history.trim_stale_records(now=NOW)
        self.assertEqual(history.trim_stale_records(now=NOW), 0)

    def test_trimmed_fields_are_recorded_for_hydration(self):
        record = _record('old')
        _history([record]).trim_stale_records(now=NOW)
        self.assertEqual(
            set(record[result_sync.TRIMMED_FIELDS]),
            {'market_timeline', 'odds_layers', 'closing_odds_snapshot',
             'last_prematch_odds_snapshot'})


class PersistTests(unittest.TestCase):
    def _stored(self):
        full = _record('old')
        full.pop('_timeline_offloaded', None)
        return full

    def test_every_trimmed_field_is_restored_before_writing(self):
        record = _record('old')
        history = _history([record])
        history.trim_stale_records(now=NOW)

        with mock.patch.object(result_sync.repositories, 'football_prediction_get',
                               return_value=self._stored()), \
             mock.patch.object(result_sync.repositories, 'football_prediction_upsert',
                               return_value='mysql') as upsert:
            history._save_record(record)

        written = upsert.call_args.args[0]
        self.assertEqual(len(written['market_timeline']), 5)
        self.assertEqual(written['odds_layers']['T-24h']['odds'], [1.9] * 40)
        self.assertEqual(written['closing_odds_snapshot'], {'home': 1.9, 'draw': 3.4, 'away': 4.1})
        self.assertIn('last_prematch_odds_snapshot', written)
        self.assertNotIn(result_sync.TRIMMED_FIELDS, written)
        self.assertNotIn(result_sync.TIMELINE_OFFLOADED, written)
        # 内存副本不受影响
        self.assertEqual(len(record['market_timeline']), 1)

    def test_a_record_whose_full_text_cannot_be_read_is_not_written(self):
        """读不回全文就拒绝写——拿摘要覆盖会让全文永久消失。"""
        record = _record('old')
        history = _history([record])
        history.trim_stale_records(now=NOW)

        with mock.patch.object(result_sync.repositories, 'football_prediction_get',
                               return_value=None), \
             mock.patch.object(result_sync.repositories, 'football_prediction_upsert') as upsert:
            history._save_record(record)

        upsert.assert_not_called()


class HydrateTests(unittest.TestCase):
    def test_hydrating_restores_every_trimmed_field_in_memory(self):
        record = _record('old')
        history = _history([record])
        history.trim_stale_records(now=NOW)

        with mock.patch.object(result_sync.repositories, 'football_prediction_get',
                               return_value=_record('old')):
            history._hydrate_timeline(record)

        self.assertEqual(len(record['market_timeline']), 5)
        self.assertEqual(record['odds_layers']['T-24h']['odds'], [1.9] * 40)
        self.assertIn('last_prematch_odds_snapshot', record)
        self.assertNotIn(result_sync.TRIMMED_FIELDS, record)
        self.assertNotIn(result_sync.TIMELINE_OFFLOADED, record)
