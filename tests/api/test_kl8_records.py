# -*- coding: utf-8 -*-
"""快乐8预测记录的分页与快速展示守卫。"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import src.kl8 as kl8_module
from src.api.services import kl8 as service
from src.kl8 import fetch as kl8_fetch


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
        current_version = service._current_kl8_predictor_version()
        current_config = service._current_kl8_config_fingerprint()
        based_on_issue = '2026232'
        target_issue = '2026233'
        prediction = {
            'snapshot_file': 'snapshot_context-id.json',
            'based_on_issue': based_on_issue,
            'target_issue': target_issue,
            'strategy_config_fingerprint': current_config,
            'statistics': {
                'version': current_version,
                'based_on_issue': based_on_issue,
                'target_issue': target_issue,
            },
            'select_6': {'numbers': [1, 2, 3, 4, 5, 6]},
            'fu_shi_7': {'top7_numbers': [1, 2, 3, 4, 5, 6, 7]},
        }
        analyzer = mock.Mock()
        analyzer.history_data = [{'issue': based_on_issue}]
        analyzer.recalculate_play_excluding.return_value = {'ok': True}
        source_params = {
            'source_snapshot_id': ['context-id'],
            'source_version': [current_version],
            'source_target_issue': [target_issue],
            'source_based_on_issue': [based_on_issue],
            'source_config_fingerprint': [current_config],
        }
        snapshots = [{
            'snapshot_id': 'context-id',
            'target_issue': target_issue,
            'based_on_issue': based_on_issue,
            'version': current_version,
            'strategy_config_fingerprint': current_config,
            'predicted_at_ns': 100,
        }]

        with mock.patch.dict(
                service._CACHE['kl8'], {'data': None, 'timestamp': 0}), \
                mock.patch.object(service.kl8_cache, 'predict',
                                  return_value=prediction), \
                mock.patch.object(service, 'kl8_latest_issue',
                                  return_value=based_on_issue), \
                mock.patch.object(service, 'kl8_list_snapshots',
                                  return_value=snapshots), \
                mock.patch.object(service, 'kl8_run_prediction') as calculate:
            self.assertEqual(service.kl8_payload(), {'result': prediction})
            calculate.assert_not_called()
            self.assertIs(service._CACHE['kl8']['data'], prediction)
            self.assertGreater(service._CACHE['kl8']['timestamp'], 0)

            with mock.patch.object(service, 'get_kl8_analyzer',
                                   return_value=analyzer):
                service.kl8_exclude_recalculate_payload({
                    **source_params,
                    'play_type': ['select_6'], 'numbers': ['20'],
                })
                service.kl8_exclude_recalculate_payload({
                    **source_params,
                    'play_type': ['fu_shi_7'], 'numbers': ['21'],
                })

        select6_call, fushi7_call = analyzer.recalculate_play_excluding.call_args_list
        self.assertEqual(select6_call.args[:2], ('select_6', [20]))
        self.assertEqual(fushi7_call.args[:2], ('fu_shi_7', [21]))
        self.assertEqual(select6_call.kwargs['record_context'], {
            'source_snapshot_id': 'context-id',
            'source_version': current_version,
            'generation_mode': 'manual',
            'initial_numbers': [1, 2, 3, 4, 5, 6],
        })
        self.assertEqual(fushi7_call.kwargs['record_context'], {
            'source_snapshot_id': 'context-id',
            'source_version': current_version,
            'generation_mode': 'manual',
            'initial_numbers': [1, 2, 3, 4, 5, 6, 7],
        })

    def test_manual_recalculation_rejects_stale_snapshot_context(self):
        current_version = service._current_kl8_predictor_version()
        current_config = service._current_kl8_config_fingerprint()
        prediction = {
            'snapshot_file': 'snapshot_cached-experiment.json',
            'based_on_issue': '2026232',
            'target_issue': '2026233',
            'strategy_config_fingerprint': current_config,
            'statistics': {
                'version': current_version,
                'based_on_issue': '2026232',
                'target_issue': '2026233',
            },
            'select_6': {'numbers': [1, 2, 3, 4, 5, 6]},
        }
        analyzer = mock.Mock()
        analyzer.history_data = [{'issue': '2026232'}]
        analyzer.recalculate_play_excluding.return_value = {'ok': True}
        snapshots = [
            {
                'snapshot_id': 'cached-experiment',
                'target_issue': '2026233',
                'based_on_issue': '2026232',
                'version': current_version,
                'strategy_config_fingerprint': current_config,
                'predicted_at': '2026-09-01T10:00:00',
            },
            {
                'snapshot_id': 'scheduler-formal',
                'target_issue': '2026233',
                'based_on_issue': '2026232',
                'version': current_version,
                'strategy_config_fingerprint': current_config,
                'predicted_at': '2026-09-01T11:00:00',
            },
        ]

        with mock.patch.dict(
                service._CACHE['kl8'],
                {'data': prediction, 'timestamp': 1},
                clear=True,
        ), mock.patch.object(
                service, 'get_kl8_analyzer', return_value=analyzer,
        ), mock.patch.object(
                service, 'kl8_list_snapshots', return_value=snapshots,
        ), mock.patch.object(
                service, 'kl8_latest_issue', return_value='2026232',
        ):
            payload = service.kl8_exclude_recalculate_payload({
                'play_type': ['select_6'],
                'numbers': ['20'],
                'source_snapshot_id': ['cached-experiment'],
                'source_version': [current_version],
                'source_target_issue': ['2026233'],
                'source_based_on_issue': ['2026232'],
                'source_config_fingerprint': [current_config],
            })

        self.assertIn('预测记录已更新', payload['error'])
        analyzer.recalculate_play_excluding.assert_not_called()

    def test_manual_recalculation_fails_closed_without_page_context(self):
        payload = service.kl8_exclude_recalculate_payload({
            'play_type': ['select_6'], 'numbers': ['20'],
        })
        self.assertIn('页面预测上下文不完整', payload['error'])

    def test_manual_recalculation_fails_closed_when_snapshot_index_is_empty(self):
        current_version = service._current_kl8_predictor_version()
        current_config = service._current_kl8_config_fingerprint()
        prediction = {
            'snapshot_file': 'snapshot_context-id.json',
            'based_on_issue': '2026232',
            'target_issue': '2026233',
            'strategy_config_fingerprint': current_config,
            'statistics': {
                'version': current_version,
                'based_on_issue': '2026232',
                'target_issue': '2026233',
            },
            'select_6': {'numbers': [1, 2, 3, 4, 5, 6]},
        }
        analyzer = mock.Mock()
        with mock.patch.dict(
                service._CACHE['kl8'], {'data': prediction, 'timestamp': 1}, clear=True,
        ), mock.patch.object(
                service, 'kl8_latest_issue', return_value='2026232',
        ), mock.patch.object(
                service, 'kl8_list_snapshots', return_value=[],
        ), mock.patch.object(
                service, 'get_kl8_analyzer', return_value=analyzer,
        ):
            payload = service.kl8_exclude_recalculate_payload({
                'play_type': ['select_6'],
                'numbers': ['20'],
                'source_snapshot_id': ['context-id'],
                'source_version': [current_version],
                'source_target_issue': ['2026233'],
                'source_based_on_issue': ['2026232'],
                'source_config_fingerprint': [current_config],
            })
        self.assertIn('当前预测记录不存在', payload['error'])
        analyzer.assert_not_called()


    def test_history_save_waits_for_manual_recalculation_context(self):
        current_version = service._current_kl8_predictor_version()
        current_config = service._current_kl8_config_fingerprint()
        based_on_issue = '2026232'
        target_issue = '2026233'
        prediction = {
            'snapshot_file': 'snapshot_context-id.json',
            'based_on_issue': based_on_issue,
            'target_issue': target_issue,
            'strategy_config_fingerprint': current_config,
            'statistics': {
                'version': current_version,
                'based_on_issue': based_on_issue,
                'target_issue': target_issue,
            },
            'select_6': {'numbers': [1, 2, 3, 4, 5, 6]},
        }
        snapshots = [{
            'snapshot_id': 'context-id',
            'target_issue': target_issue,
            'based_on_issue': based_on_issue,
            'version': current_version,
            'strategy_config_fingerprint': current_config,
            'predicted_at_ns': 100,
        }]
        source_params = {
            'play_type': ['select_6'],
            'numbers': ['20'],
            'source_snapshot_id': ['context-id'],
            'source_version': [current_version],
            'source_target_issue': [target_issue],
            'source_based_on_issue': [based_on_issue],
            'source_config_fingerprint': [current_config],
        }
        manual_entered = threading.Event()
        release_manual = threading.Event()
        saver_started = threading.Event()
        save_body_entered = threading.Event()
        errors = []
        manual_payloads = []

        analyzer = mock.Mock()
        analyzer.history_data = [{'issue': based_on_issue}]

        def recalculate(*_args, **_kwargs):
            manual_entered.set()
            if not release_manual.wait(5):
                raise RuntimeError('测试未释放手动重算线程')
            return {'ok': True}

        analyzer.recalculate_play_excluding.side_effect = recalculate

        def run_manual():
            try:
                manual_payloads.append(
                    service.kl8_exclude_recalculate_payload(source_params)
                )
            except Exception as exc:  # pragma: no cover - 线程异常回传
                errors.append(exc)

        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / 'kl8_history.json'
            history_file.write_text(json.dumps({'results': [{
                'issue': based_on_issue,
                'numbers': list(range(1, 21)),
                'date': '2026-08-31',
            }]}), encoding='utf-8')
            original_save_body = kl8_fetch._save_kl8_data_locked

            def save_body(data):
                save_body_entered.set()
                return original_save_body(data)

            def save_new_draw():
                saver_started.set()
                try:
                    kl8_fetch.save_kl8_data([{
                        'issue': target_issue,
                        'numbers': list(range(21, 41)),
                        'date': '2026-09-01',
                    }])
                except Exception as exc:  # pragma: no cover - 线程异常回传
                    errors.append(exc)

            with mock.patch.dict(
                    service._CACHE['kl8'],
                    {'data': prediction, 'timestamp': 1},
                    clear=True,
            ), mock.patch.object(
                    service, 'kl8_latest_issue', return_value=based_on_issue,
            ), mock.patch.object(
                    service, 'kl8_list_snapshots', return_value=snapshots,
            ), mock.patch.object(
                    service, 'get_kl8_analyzer', return_value=analyzer,
            ), mock.patch.object(
                    kl8_fetch, 'KL8_HISTORY_FILE', str(history_file),
            ), mock.patch.object(
                    kl8_fetch, '_save_kl8_data_locked', side_effect=save_body,
            ), mock.patch.object(
                    kl8_fetch, '_mirror_to_store',
            ), mock.patch.object(
                    kl8_fetch, 'clear_cache',
            ):
                manual_thread = threading.Thread(target=run_manual)
                save_thread = threading.Thread(target=save_new_draw)
                try:
                    manual_thread.start()
                    self.assertTrue(manual_entered.wait(2))
                    save_thread.start()
                    self.assertTrue(saver_started.wait(2))
                    self.assertFalse(
                        save_body_entered.wait(0.1),
                        '新期开奖写入穿透了手动重算的来源核验区间',
                    )
                    before = json.loads(history_file.read_text(encoding='utf-8'))
                    self.assertEqual(before['results'][0]['issue'], based_on_issue)
                finally:
                    release_manual.set()
                    manual_thread.join(3)
                    save_thread.join(3)

            self.assertFalse(manual_thread.is_alive() or save_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(manual_payloads, [{'result': {'ok': True}}])
            after = json.loads(history_file.read_text(encoding='utf-8'))
            self.assertEqual(after['results'][0]['issue'], target_issue)


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
