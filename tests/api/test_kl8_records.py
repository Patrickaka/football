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
            'fu_shi_7_recalculation_chain': {'records': [{
                'excluded_numbers': [1, 2, 3, 4, 5, 6, 7],
                'select6_round': {
                    'numbers': [7, 8, 9, 10, 11, 12],
                    'excluded_numbers': [1, 2, 3, 4, 5, 6],
                },
            }]},
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
                service.kl8_exclude_recalculate_payload({
                    **source_params,
                    'play_type': ['fu_shi_7'], 'numbers': ['1,2,3,4,5,6,7'],
                })

        select6_call, fushi7_call, linked_call = analyzer.recalculate_play_excluding.call_args_list
        self.assertEqual(linked_call.kwargs['record_context']['select6_round'], {
            'numbers': [7, 8, 9, 10, 11, 12],
            'excluded_numbers': [1, 2, 3, 4, 5, 6],
        })
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


class CurrentVersionReference(unittest.TestCase):
    runtime_version = 'kl8-current-reference-test'

    def _snapshot(self, identity, hour, *, version=None, experiment=False, start=1, **updates):
        return {
            'snapshot_id': identity, 'file': f'snapshot_{identity}.json',
            'target_issue': '2026242', 'based_on_issue': '2026241',
            'version': version or self.runtime_version, 'is_experiment': experiment,
            'predicted_at': f'2026-09-08T{hour:02d}:00:00+08:00',
            'has_settlement': False,
            'select_6': list(range(start, start + 6)),
            'fu_shi_7': list(range(start, start + 7)),
            **updates,
        }

    def _payload(self, snapshots, *, draws=None, rounds=None, missing=(), file_overrides=None):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot_dir, settlement_dir = root / 'snapshots', root / 'settlements'
            snapshot_dir.mkdir()
            settlement_dir.mkdir()
            for snapshot in snapshots:
                if snapshot['snapshot_id'] in missing:
                    continue
                raw = (file_overrides or {}).get(snapshot['snapshot_id'], snapshot)
                (snapshot_dir / snapshot['file']).write_text(json.dumps(raw), encoding='utf-8')
            before = {path.name: path.read_bytes() for path in snapshot_dir.iterdir()}
            with mock.patch.object(service, 'kl8_list_snapshots', return_value=snapshots), \
                    mock.patch.object(service, 'kl8_list_recalculations', return_value=rounds or []), \
                    mock.patch.object(service, '_current_kl8_predictor_version', return_value=self.runtime_version), \
                    mock.patch('src.kl8.records._load_record_draws', return_value=draws or {}), \
                    mock.patch.object(service, '_schedule_kl8_records_maintenance', return_value=False) as maintenance, \
                    mock.patch.object(service, 'kl8_run_prediction') as predict, \
                    mock.patch.object(kl8_module, 'KL8_SNAPSHOT_DIR', snapshot_dir), \
                    mock.patch.object(kl8_module, 'KL8_SETTLEMENT_DIR', settlement_dir):
                payload = service.kl8_records_payload({'page': ['1'], 'page_size': ['8']})
            predict.assert_not_called()
            self.assertEqual(before, {path.name: path.read_bytes() for path in snapshot_dir.iterdir()})
            self.assertEqual(list(settlement_dir.iterdir()), [])
            self.assertNotIn('error', payload)
            return payload['result'], maintenance.call_args.args[0]

    def test_keeps_original_and_reads_only_latest_current_formal_reference(self):
        original = self._snapshot('original', 10, version='kl8-v10.17')
        first_new = self._snapshot('new-first', 11, start=11)
        latest_new = self._snapshot('new-latest', 12, start=21)
        experiment = self._snapshot('experiment', 13, experiment=True, start=41)
        result, maintained = self._payload([experiment, latest_new, original, first_new])
        self.assertEqual(result['runtime_version'], self.runtime_version)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['pending_count'], 1)
        record = result['records'][0]
        self.assertEqual(record['record_role'], 'original_forecast')
        self.assertEqual(record['runtime_version'], self.runtime_version)
        self.assertEqual(record['version'], 'kl8-v10.17')
        self.assertEqual(record['snapshot_id'], 'original')
        self.assertEqual(record['predicted']['select_6'], list(range(1, 7)))
        reference = record['current_version_reference']
        self.assertEqual(reference['snapshot_id'], 'new-latest')
        self.assertEqual(reference['version'], self.runtime_version)
        self.assertEqual(reference['predicted_at'], latest_new['predicted_at'])
        self.assertEqual(reference['predicted']['select_6'], list(range(21, 27)))
        self.assertEqual(reference['predicted']['fu_shi_7'], list(range(21, 28)))
        self.assertEqual(reference['record_role'], 'current_version_reference')
        self.assertTrue(reference['reference_only'])
        self.assertFalse(reference['accuracy_eligible'])
        self.assertEqual(reference['accuracy_exclusion_reason'], 'current_version_reference')
        self.assertEqual([row['snapshot_id'] for row in maintained], ['original'])

    def test_original_and_reference_keep_their_own_recalculation_chains(self):
        original = self._snapshot('original', 10, version='kl8-v10.17')
        reference = self._snapshot('new', 11, start=21)
        rounds = [
            {'source_snapshot_id': 'original', 'play_type': 'select_6', 'round': 1,
             'numbers': [2, 3, 4, 5, 6, 7]},
            {'source_snapshot_id': 'new', 'play_type': 'select_6', 'round': 1,
             'numbers': [22, 23, 24, 25, 26, 27]},
            {'source_snapshot_id': 'hidden', 'play_type': 'select_6', 'round': 1,
             'numbers': [42, 43, 44, 45, 46, 47]},
        ]
        result, _ = self._payload([reference, original], rounds=rounds)
        record = result['records'][0]
        self.assertEqual([row['source_snapshot_id'] for row in record['exclude_recalculations']], ['original'])
        self.assertEqual([row['source_snapshot_id'] for row in
                          record['current_version_reference']['exclude_recalculations']], ['new'])

    def test_post_draw_reference_retains_time_audit_but_never_counts_as_primary(self):
        original = self._snapshot('original', 10, version='kl8-v10.17')
        after_draw = self._snapshot('after-draw', 22, start=21)
        draws = {'2026242': {'issue': '2026242', 'date': '2026-09-08',
                             'numbers': list(range(1, 21))}}
        result, maintained = self._payload([after_draw, original], draws=draws)
        record = result['records'][0]
        self.assertEqual(record['snapshot_id'], 'original')
        reference = record['current_version_reference']
        self.assertEqual(reference['snapshot_id'], 'after-draw')
        self.assertEqual(reference['prediction_audit']['reason'], 'prediction_at_or_after_draw')
        self.assertFalse(reference['accuracy_eligible'])
        self.assertEqual([row['snapshot_id'] for row in maintained], ['original'])

    def test_no_reference_when_current_snapshot_is_itself_the_original(self):
        current = self._snapshot('current', 10)
        result, _ = self._payload([current])
        self.assertIsNone(result['records'][0]['current_version_reference'])

    def test_absent_unreadable_and_inconsistent_files_cannot_supply_reference_metadata(self):
        original = self._snapshot('original', 10, version='kl8-v10.17')
        current = self._snapshot('current', 11)
        for replacement in (None, [], {}, {**current, 'version': 'different-version'},
                            {**current, 'snapshot_id': 'different-id'},
                            {**current, 'is_experiment': True},
                            {**current, 'predicted_at': '2026-09-08T09:00:00+08:00'},
                            {**current, 'select_6': []},
                            {**current, 'select_6': [1, 2, 3]},
                            {**current, 'fu_shi_7': [1, 2, 3]}):
            with self.subTest(replacement=replacement):
                result, _ = self._payload([original, current],
                                           file_overrides={'current': replacement})
                self.assertIsNone(result['records'][0]['current_version_reference'])
        result, _ = self._payload([original, current], missing={'current'})
        self.assertIsNone(result['records'][0]['current_version_reference'])

    def test_skips_invalid_future_data_boundaries_and_unknown_prediction_times(self):
        original = self._snapshot('original', 10, version='kl8-v10.17')
        invalid_candidates = [
            self._snapshot('same-issue', 11, based_on_issue='2026242'),
            self._snapshot('future-issue', 12, based_on_issue='2026243'),
            self._snapshot('invalid-issue', 13, based_on_issue='unknown'),
            self._snapshot('unknown-time', 14, predicted_at='2026-09-08T14:00:00'),
            self._snapshot('old-time', 9),
        ]
        result, _ = self._payload([original, *invalid_candidates])
        self.assertIsNone(result['records'][0]['current_version_reference'])

    def test_missing_newest_file_falls_back_to_latest_valid_saved_reference(self):
        original = self._snapshot('original', 10, version='kl8-v10.17')
        valid = self._snapshot('valid', 11, start=11)
        missing = self._snapshot('missing', 12, start=21)
        result, _ = self._payload([original, missing, valid], missing={'missing'})
        self.assertEqual(result['records'][0]['current_version_reference']['snapshot_id'], 'valid')


if __name__ == '__main__':
    unittest.main()
