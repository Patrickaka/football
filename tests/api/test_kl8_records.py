# -*- coding: utf-8 -*-
"""快乐8预测记录的分页与快速展示守卫。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.kl8 as kl8_module
from src.api.services import kl8 as service


class RecordsPagination(unittest.TestCase):

    def test_dedupe_happens_before_page_hydration(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot_dir = root / 'snapshots'
            settlement_dir = root / 'settlements'
            snapshot_dir.mkdir()
            settlement_dir.mkdir()

            snapshots = []
            for number in range(1, 26):
                issue = f'2026{number:03d}'
                filename = f'snapshot_{number}.json'
                (snapshot_dir / filename).write_text(
                    json.dumps({
                        'select_5': [1, 2, 3, 4, 5],
                        'fu_shi_7': {'top7_numbers': [1, 2, 3, 4, 5, 6, 7]},
                    }),
                    encoding='utf-8',
                )
                snapshots.append({
                    'file': filename,
                    'snapshot_id': f'id-{number}',
                    'target_issue': issue,
                    'based_on_issue': f'2026{number - 1:03d}',
                    'predicted_at': f'2026-08-{number:02d}T10:00:00',
                    'version': 'test',
                    'has_settlement': number % 2 == 0,
                })

            # 同一期旧快照必须在读取完整内容之前被丢掉。
            snapshots.append(dict(
                snapshots[-1],
                file='missing-old-duplicate.json',
                snapshot_id='old-duplicate',
                predicted_at='2026-01-01T00:00:00',
            ))
            recalculations = [
                {'source_snapshot_id': 'id-17', 'play_type': 'select_6',
                 'round': 1, 'numbers': [1, 2, 3, 4, 5, 6]},
                {'source_snapshot_id': 'id-25', 'play_type': 'select_6',
                 'round': 1, 'numbers': [7, 8, 9, 10, 11, 12]},
            ]

            with mock.patch.object(service, 'kl8_list_snapshots', return_value=snapshots), \
                    mock.patch.object(service, 'kl8_list_recalculations',
                                      return_value=recalculations), \
                    mock.patch.object(service, '_schedule_kl8_records_maintenance',
                                      return_value=False), \
                    mock.patch.object(service, '_load_kl8_record',
                                      wraps=service._load_kl8_record) as hydrate, \
                    mock.patch.object(kl8_module, 'KL8_SNAPSHOT_DIR', snapshot_dir), \
                    mock.patch.object(kl8_module, 'KL8_SETTLEMENT_DIR', settlement_dir):
                payload = service.kl8_records_payload(
                    {'page': ['2'], 'page_size': ['8']}
                )

            result = payload['result']
            self.assertEqual(result['count'], 25)
            self.assertEqual(result['page'], 2)
            self.assertEqual(result['page_size'], 8)
            self.assertEqual(result['total_pages'], 4)
            self.assertEqual(hydrate.call_count, 8)
            self.assertEqual(
                [row['target_issue'] for row in result['records']],
                [f'2026{number:03d}' for number in range(17, 9, -1)],
            )
            self.assertEqual(
                result['records'][0]['exclude_recalculations'][0]['round'], 1,
            )
            self.assertFalse(any(
                item.get('source_snapshot_id') == 'id-25'
                for row in result['records']
                for item in row['exclude_recalculations']
            ))

    def test_no_page_parameters_keep_the_legacy_full_result(self):
        self.assertEqual(service._kl8_records_page_options({}, 25), (1, 25, 1, False))

    def test_page_size_is_bounded_and_bad_values_use_defaults(self):
        self.assertEqual(
            service._kl8_records_page_options(
                {'page': ['bad'], 'page_size': ['9999']}, 120,
            ),
            (1, 50, 3, True),
        )


class ManualRecalculationContext(unittest.TestCase):

    def test_cached_payload_supplies_snapshot_context_for_both_recorded_plays(self):
        prediction = {
            'snapshot_file': 'snapshot_context-id.json',
            'statistics': {'version': 'kl8-test-version'},
            'select_6': {'numbers': [1, 2, 3, 4, 5, 6]},
            'fu_shi_7': {'top7_numbers': [1, 2, 3, 4, 5, 6, 7]},
        }
        analyzer = mock.Mock()
        analyzer.recalculate_play_excluding.return_value = {'ok': True}

        with mock.patch.dict(
                service._CACHE['kl8'], {'data': None, 'timestamp': 0}), \
                mock.patch.object(service.kl8_cache, 'predict',
                                  return_value=prediction), \
                mock.patch.object(service, 'kl8_run_prediction') as calculate:
            self.assertEqual(service.kl8_payload(), {'result': prediction})
            calculate.assert_not_called()
            self.assertIs(service._CACHE['kl8']['data'], prediction)
            self.assertGreater(service._CACHE['kl8']['timestamp'], 0)

            with mock.patch.object(service, 'get_kl8_analyzer',
                                   return_value=analyzer):
                service.kl8_exclude_recalculate_payload({
                    'play_type': ['select_6'], 'numbers': ['20'],
                })
                service.kl8_exclude_recalculate_payload({
                    'play_type': ['fu_shi_7'], 'numbers': ['21'],
                })

        select6_call, fushi7_call = analyzer.recalculate_play_excluding.call_args_list
        self.assertEqual(select6_call.args[:2], ('select_6', [20]))
        self.assertEqual(fushi7_call.args[:2], ('fu_shi_7', [21]))
        self.assertEqual(select6_call.kwargs['record_context'], {
            'source_snapshot_id': 'context-id',
            'source_version': 'kl8-test-version',
            'generation_mode': 'manual',
            'initial_numbers': [1, 2, 3, 4, 5, 6],
        })
        self.assertEqual(fushi7_call.kwargs['record_context'], {
            'source_snapshot_id': 'context-id',
            'source_version': 'kl8-test-version',
            'generation_mode': 'manual',
            'initial_numbers': [1, 2, 3, 4, 5, 6, 7],
        })


class MaintenanceFastExit(unittest.TestCase):

    def test_future_pending_issue_does_not_initialize_the_analyzer(self):
        records = [{
            'file': 'snapshot_future.json',
            'target_issue': '2026999',
            'has_settlement': False,
            'settlement': None,
        }]
        with mock.patch.object(service, '_kl8_draw_map_from_history_file',
                               return_value={'2026232': list(range(1, 21))}), \
                mock.patch.object(service, 'get_kl8_analyzer') as analyzer:
            service.kl8_backfill_settlements(records)
        analyzer.assert_not_called()

    def test_no_settlements_do_not_initialize_the_analyzer(self):
        with mock.patch.object(service, 'get_kl8_analyzer') as analyzer:
            service.kl8_rebuild_stale_settlements([
                {'target_issue': '2026999', 'settlement': None},
            ])
        analyzer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
