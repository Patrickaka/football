# -*- coding: utf-8 -*-
"""快乐8模型版本升级时的预测快照与调度回归。"""

import json
import threading
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import src.kl8 as kl8_module
from src.api.services import kl8 as service
from src.kl8 import config as kl8_config
from src.kl8 import fetch as kl8_fetch
from src.kl8 import scheduler as kl8_scheduler
from src.kl8 import snapshots as kl8_snapshots
from src.kl8.analyzer import KL8Analyzer
from src.kl8.records import (
    _compute_next_issue,
    _resolved_strategies_fingerprint,
)


def _history(latest_issue='2026232'):
    latest = int(latest_issue)
    return [
        {
            'issue': str(latest),
            'numbers': list(range(1, 21)),
            'date': '2026-08-31',
        },
        {
            'issue': str(latest - 1),
            'numbers': list(range(21, 41)),
            'date': '2026-08-30',
        },
    ]


class CurrentVersionRecordSelectionTests(unittest.TestCase):

    def test_same_issue_prefers_latest_current_version_snapshot(self):
        current = service._current_kl8_predictor_version()
        snapshots = [
            {
                'snapshot_id': 'legacy-latest',
                'target_issue': '2026233',
                'predicted_at': '2026-09-01T13:00:00',
                'version': 'kl8-legacy',
                'is_experiment': False,
            },
            {
                'snapshot_id': 'current-experiment',
                'target_issue': '2026233',
                'predicted_at': '2026-09-01T12:00:00',
                'version': current,
                'is_experiment': True,
            },
            {
                'snapshot_id': 'current-formal',
                'target_issue': '2026233',
                'predicted_at': '2026-09-01T09:00:00',
                'version': current,
                'is_experiment': False,
            },
            {
                'snapshot_id': 'other-older',
                'target_issue': '2026234',
                'predicted_at': '2026-09-01T08:00:00',
                'version': 'kl8-legacy',
                'is_experiment': False,
            },
            {
                'snapshot_id': 'other-latest',
                'target_issue': '2026234',
                'predicted_at': '2026-09-01T10:00:00',
                'version': 'kl8-legacy',
                'is_experiment': True,
            },
        ]

        selected = service._dedupe_kl8_snapshots(snapshots)
        by_issue = {row['target_issue']: row for row in selected}

        self.assertEqual(by_issue['2026233']['snapshot_id'], 'current-experiment')
        self.assertEqual(
            by_issue['2026234']['snapshot_id'],
            'other-latest',
            '没有当前版本正式快照时仍按生成时间选择，不能删掉历史期记录',
        )

    def test_snapshot_index_preserves_nanosecond_order_for_same_second(self):
        current = service._current_kl8_predictor_version()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_dir = root / 'snapshots'
            settlement_dir = root / 'settlements'
            snapshot_dir.mkdir()
            settlement_dir.mkdir()
            common = {
                'target_issue': '2026233',
                'based_on_issue': '2026232',
                'predicted_at': '2026-09-01T12:00:00',
                'version': current,
            }
            for snapshot_id, predicted_at_ns in (
                ('same-second-old', 100),
                ('same-second-new', 200),
            ):
                (snapshot_dir / f'snapshot_{snapshot_id}.json').write_text(
                    json.dumps({
                        **common,
                        'snapshot_id': snapshot_id,
                        'predicted_at_ns': predicted_at_ns,
                    }),
                    encoding='utf-8',
                )

            with mock.patch.object(
                    kl8_config, 'KL8_SNAPSHOT_DIR', snapshot_dir,
            ), mock.patch.object(
                    kl8_config, 'KL8_SETTLEMENT_DIR', settlement_dir,
            ):
                indexed = kl8_snapshots.list_prediction_snapshots()

        self.assertEqual(
            {item['snapshot_id']: item['predicted_at_ns'] for item in indexed},
            {'same-second-old': 100, 'same-second-new': 200},
        )
        selected = service._dedupe_kl8_snapshots(indexed)
        self.assertEqual(selected[0]['snapshot_id'], 'same-second-new')

    def test_same_version_prefers_current_strategy_configuration(self):
        current = service._current_kl8_predictor_version()
        current_config = service._current_kl8_config_fingerprint()
        selected = service._dedupe_kl8_snapshots([
            {
                'snapshot_id': 'stale-config-newer-clock',
                'target_issue': '2026233',
                'version': current,
                'strategy_config_fingerprint': 'stale-config',
                'predicted_at_ns': 999,
            },
            {
                'snapshot_id': 'current-config',
                'target_issue': '2026233',
                'version': current,
                'strategy_config_fingerprint': current_config,
                'predicted_at_ns': 100,
            },
        ])

        self.assertEqual(selected[0]['snapshot_id'], 'current-config')


class SnapshotIdentityAcrossVersionsTests(unittest.TestCase):

    def test_persisted_fingerprint_controls_duplicate_identity(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = _history()
        strategy = {
            'feature_weights': {'frequency': 1.0},
            'model_weights': {'rank': 1.0},
            'window_size': 50,
        }
        current_fp = _resolved_strategies_fingerprint({'select_5': strategy})
        target_issue = _compute_next_issue(
            analyzer.history_data[0]['issue'], analyzer.history_data,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = Path(temp_dir)
            legacy = {
                'snapshot_id': 'legacy-id',
                'target_issue': target_issue,
                'predicted_at': '2026-09-01T08:00:00',
                'version': 'kl8-legacy',
                'strategy_fingerprint': 'legacy-persisted-fingerprint',
                'is_experiment': False,
                # 策略内容故意与当前相同。旧快照的身份只能读落盘指纹；
                # 用当前代码重算它会错误地得到 current_fp。
                'resolved_strategies': {'select_5': strategy},
            }
            (snapshot_dir / 'snapshot_legacy-id.json').write_text(
                json.dumps(legacy, ensure_ascii=False), encoding='utf-8',
            )

            with mock.patch.object(kl8_config, 'KL8_SNAPSHOT_DIR', temp_dir):
                first_name = analyzer._save_prediction_snapshot({
                    'resolved_strategies': {'select_5': strategy},
                })
                first = json.loads(
                    (snapshot_dir / first_name).read_text(encoding='utf-8')
                )
                second_name = analyzer._save_prediction_snapshot({
                    'resolved_strategies': {'select_5': strategy},
                })
                second = json.loads(
                    (snapshot_dir / second_name).read_text(encoding='utf-8')
                )

        self.assertEqual(first['strategy_fingerprint'], current_fp)
        self.assertFalse(
            first['is_experiment'],
            '旧版本旧指纹不能把新版首次正式预测误标为实验',
        )
        self.assertTrue(
            second['is_experiment'],
            '同目标期、同当前落盘指纹的后续快照才应标为实验',
        )


class PredictionSerializationTests(unittest.TestCase):

    def test_force_predictions_do_not_run_predict_all_concurrently(self):
        entered = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()
        guard = threading.Lock()
        calls = 0
        active = 0
        max_active = 0
        errors = []

        class Analyzer:
            history_data = _history()

            def predict_all(self):
                nonlocal calls, active, max_active
                with guard:
                    calls += 1
                    call_number = calls
                    active += 1
                    max_active = max(max_active, active)
                if call_number == 1:
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('测试未释放首个预测')
                else:
                    second_entered.set()
                with guard:
                    active -= 1
                return {'call': call_number}

        def invoke():
            try:
                kl8_snapshots.run_prediction(force_refresh=True)
            except Exception as exc:  # pragma: no cover - 仅用于把线程异常带回主线程
                errors.append(exc)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                kl8_snapshots, 'get_kl8_analyzer', return_value=Analyzer(),
            ))
            stack.enter_context(mock.patch.object(
                kl8_snapshots, '_strategy_config_fingerprint', return_value='fp',
            ))
            stack.enter_context(mock.patch.object(
                kl8_snapshots, '_history_signature', return_value=[1, 2],
            ))
            stack.enter_context(mock.patch.object(kl8_snapshots, '_persist_prediction'))
            first = threading.Thread(target=invoke)
            second = threading.Thread(target=invoke)
            first.start()
            self.assertTrue(entered.wait(2))
            second.start()
            self.assertFalse(
                second_entered.wait(0.1),
                '第二轮 predict_all 在第一轮结束前并发进入',
            )
            release.set()
            first.join(3)
            second.join(3)

        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(calls, 2)
        self.assertEqual(max_active, 1)

    def test_scheduler_degradation_waits_for_inflight_prediction(self):
        prediction_entered = threading.Event()
        release_prediction = threading.Event()
        mark_attempted = threading.Event()
        mark_finished = threading.Event()
        errors = []
        observed_statuses = []

        class Analyzer:
            history_data = _history()

            def predict_all(self):
                observed_statuses.append(
                    kl8_config.ACTIVE_STRATEGIES['select_6']['degradation_status']
                )
                prediction_entered.set()
                if not release_prediction.wait(5):
                    raise RuntimeError('测试未释放预测线程')
                observed_statuses.append(
                    kl8_config.ACTIVE_STRATEGIES['select_6']['degradation_status']
                )
                return {'statistics': {'signal_status': 'validated'}}

        def run_prediction():
            try:
                kl8_snapshots.run_prediction(force_refresh=True)
            except Exception as exc:  # pragma: no cover - 线程异常回传
                errors.append(exc)

        def run_degradation():
            try:
                kl8_scheduler._check_strategy_degradation()
            except Exception as exc:  # pragma: no cover - 线程异常回传
                errors.append(exc)
            finally:
                mark_finished.set()

        with tempfile.TemporaryDirectory() as temp_dir:
            settlement_dir = Path(temp_dir)
            for offset in range(100):
                (settlement_dir / f'settlement_{offset:03d}.json').write_text(
                    json.dumps({
                        'actual_issue': f'2026{offset + 1:03d}',
                        'strategy_ids': {'select_6': 'select-6-current'},
                        'hit_select_6': 0,
                    }),
                    encoding='utf-8',
                )

            real_mark = kl8_snapshots.mark_strategy_degradation

            def mark_with_probe(*args, **kwargs):
                mark_attempted.set()
                return real_mark(*args, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(
                    kl8_config.ACTIVE_STRATEGIES,
                    {
                        'select_6': {
                            'strategy_id': 'select-6-current',
                            'degradation_status': 'normal',
                        },
                    },
                    clear=True,
                ))
                stack.enter_context(mock.patch.object(
                    kl8_scheduler, 'data_path', return_value=str(settlement_dir),
                ))
                stack.enter_context(mock.patch.object(
                    kl8_scheduler,
                    'mark_strategy_degradation',
                    side_effect=mark_with_probe,
                ))
                stack.enter_context(mock.patch.object(
                    kl8_snapshots, 'get_kl8_analyzer', return_value=Analyzer(),
                ))
                stack.enter_context(mock.patch.object(
                    kl8_snapshots, '_history_signature', return_value=[1, 2],
                ))
                stack.enter_context(mock.patch.object(
                    kl8_snapshots, '_persist_prediction',
                ))
                persist = stack.enter_context(mock.patch.object(
                    kl8_snapshots, '_persist_active_strategies',
                ))
                clear = stack.enter_context(mock.patch.object(
                    kl8_snapshots, 'clear_cache',
                ))

                prediction_thread = threading.Thread(target=run_prediction)
                degradation_thread = threading.Thread(target=run_degradation)
                try:
                    prediction_thread.start()
                    self.assertTrue(prediction_entered.wait(2))
                    degradation_thread.start()
                    self.assertTrue(mark_attempted.wait(3))
                    self.assertFalse(
                        mark_finished.wait(0.1),
                        '策略降级写入穿透了正在执行的预测',
                    )
                    self.assertEqual(
                        kl8_config.ACTIVE_STRATEGIES['select_6'][
                            'degradation_status'
                        ],
                        'normal',
                    )
                finally:
                    release_prediction.set()
                    prediction_thread.join(3)
                    degradation_thread.join(3)

                self.assertFalse(
                    prediction_thread.is_alive() or degradation_thread.is_alive()
                )
                self.assertEqual(errors, [])
                self.assertEqual(observed_statuses, ['normal', 'normal'])
                self.assertEqual(
                    kl8_config.ACTIVE_STRATEGIES['select_6'][
                        'degradation_status'
                    ],
                    'yellow_watch',
                )
                persist.assert_called_once_with()
                clear.assert_called_once_with()

    def test_degradation_does_not_mark_a_replaced_strategy(self):
        with mock.patch.dict(
                kl8_config.ACTIVE_STRATEGIES,
                {'select_6': {'strategy_id': 'new-strategy'}},
                clear=True,
        ), mock.patch.object(
                kl8_snapshots, '_persist_active_strategies',
        ) as persist, mock.patch.object(
                kl8_snapshots, 'clear_cache',
        ) as clear:
            changed = kl8_snapshots.mark_strategy_degradation(
                'select_6', 'old-strategy', -0.5, 0.1,
            )

        self.assertFalse(changed)
        persist.assert_not_called()
        clear.assert_not_called()


class HistoryWriteSerializationTests(unittest.TestCase):

    def test_concurrent_saves_keep_both_new_draws(self):
        first_body_entered = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        second_body_entered = threading.Event()
        call_guard = threading.Lock()
        call_count = 0
        errors = []

        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / 'kl8_history.json'
            history_file.write_text(
                json.dumps({'results': _history('2026230')}),
                encoding='utf-8',
            )
            original_save_body = kl8_fetch._save_kl8_data_locked

            def gated_save_body(data):
                nonlocal call_count
                with call_guard:
                    call_count += 1
                    current_call = call_count
                if current_call == 1:
                    first_body_entered.set()
                    if not release_first.wait(5):
                        raise RuntimeError('测试未释放首个保存线程')
                else:
                    second_body_entered.set()
                return original_save_body(data)

            def save(record, started=None):
                if started is not None:
                    started.set()
                try:
                    kl8_fetch.save_kl8_data([record])
                except Exception as exc:  # pragma: no cover - 线程异常回传
                    errors.append(exc)

            draw_one = {
                'issue': '2026231',
                'numbers': list(range(1, 21)),
                'date': '2026-08-30',
            }
            draw_two = {
                'issue': '2026232',
                'numbers': list(range(21, 41)),
                'date': '2026-08-31',
            }

            with mock.patch.object(
                    kl8_fetch, 'KL8_HISTORY_FILE', str(history_file),
            ), mock.patch.object(
                    kl8_fetch,
                    '_save_kl8_data_locked',
                    side_effect=gated_save_body,
            ), mock.patch.object(
                    kl8_fetch, '_mirror_to_store',
            ), mock.patch.object(
                    kl8_fetch, 'clear_cache',
            ) as clear:
                first = threading.Thread(target=save, args=(draw_one,))
                second = threading.Thread(
                    target=save,
                    args=(draw_two, second_started),
                )
                try:
                    first.start()
                    self.assertTrue(first_body_entered.wait(2))
                    second.start()
                    self.assertTrue(second_started.wait(2))
                    self.assertFalse(
                        second_body_entered.wait(0.1),
                        '第二个历史保存在线程一提交前进入了读合并阶段',
                    )
                finally:
                    release_first.set()
                    first.join(3)
                    second.join(3)

                self.assertFalse(first.is_alive() or second.is_alive())
                self.assertEqual(errors, [])
                stored = json.loads(history_file.read_text(encoding='utf-8'))
                issues = {item['issue'] for item in stored['results']}
                self.assertTrue({'2026230', '2026231', '2026232'} <= issues)
                self.assertFalse(history_file.with_suffix('.json.tmp').exists())
                self.assertEqual(clear.call_count, 2)


class RecalculationPersistencePerformanceTests(unittest.TestCase):

    def test_automatic_round_with_explicit_number_does_not_scan_directory(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = _history()
        result = {
            'play_type': 'select_6',
            'excluded_numbers': [1, 2, 3, 4, 5, 6],
            'numbers': [7, 8, 9, 10, 11, 12],
            'quality': {},
        }

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(kl8_config, 'KL8_RECALCULATION_DIR', temp_dir), \
                mock.patch.object(
                    Path,
                    'glob',
                    side_effect=AssertionError('自动轮次不应扫描历史记录目录'),
                ):
            record = analyzer._save_exclude_recalculation(
                result,
                record_context={
                    'round': 3,
                    'source_snapshot_id': 'snapshot-current',
                    'source_version': kl8_module.KL8_PREDICTOR_VERSION,
                    'generation_mode': 'automatic',
                },
            )

        self.assertEqual(record['round'], 3)
        self.assertEqual(record['generation_mode'], 'automatic')

    def test_manual_round_continues_from_sidecar_without_directory_scan(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = _history()
        source_id = 'snapshot-current'
        version = kl8_module.KL8_PREDICTOR_VERSION
        first_result = {
            'play_type': 'select_6',
            'excluded_numbers': [1, 2, 3, 4, 5, 6],
            'numbers': [7, 8, 9, 10, 11, 12],
            'quality': {},
        }
        manual_result = {
            'play_type': 'select_6',
            'excluded_numbers': list(range(1, 13)),
            'numbers': [13, 14, 15, 16, 17, 18],
            'quality': {},
        }

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                kl8_config, 'KL8_RECALCULATION_DIR', temp_dir,
        ):
            analyzer._save_exclude_recalculation(
                first_result,
                record_context={
                    'round': 12,
                    'source_snapshot_id': source_id,
                    'source_version': version,
                    'generation_mode': 'automatic',
                },
            )
            with mock.patch.object(
                    Path,
                    'glob',
                    side_effect=AssertionError('已有轮次索引时不应扫描历史目录'),
            ):
                record = analyzer._save_exclude_recalculation(
                    manual_result,
                    record_context={
                        'source_snapshot_id': source_id,
                        'source_version': version,
                        'generation_mode': 'manual',
                    },
                )

        self.assertEqual(record['round'], 13)
        self.assertEqual(record['generation_mode'], 'manual')


class SchedulerVersionRefreshTests(unittest.TestCase):

    def _run_same_issue(self, snapshots):
        history = _history()
        analyzer = mock.Mock()
        analyzer.history_data = history
        prediction = {'statistics': {'signal_status': 'validated'}}

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                kl8_scheduler, 'get_kl8_analyzer', return_value=analyzer,
            ))
            stack.enter_context(mock.patch.object(
                kl8_scheduler, 'fetch_or_load_kl8_data', return_value=history,
            ))
            stack.enter_context(mock.patch.object(
                kl8_module, 'list_prediction_snapshots', return_value=snapshots,
            ))
            # 兼容实现选择顶层导入或函数内惰性导入两种写法。
            if hasattr(kl8_scheduler, 'list_prediction_snapshots'):
                stack.enter_context(mock.patch.object(
                    kl8_scheduler, 'list_prediction_snapshots',
                    return_value=snapshots,
                ))
            clear = stack.enter_context(mock.patch.object(kl8_scheduler, 'clear_cache'))
            predict = stack.enter_context(mock.patch.object(
                kl8_scheduler, 'run_prediction', return_value=prediction,
            ))
            publish = stack.enter_context(mock.patch.object(
                kl8_scheduler, '_publish_runtime_prediction',
            ))
            original_last_issue = kl8_scheduler._last_processed_issue
            kl8_scheduler._last_processed_issue = history[0]['issue']
            try:
                kl8_scheduler.refresh_kl8_and_predict()
            finally:
                kl8_scheduler._last_processed_issue = original_last_issue
        return clear, predict, publish

    def test_same_draw_repredicts_when_current_formal_snapshot_is_missing(self):
        target_issue = _compute_next_issue('2026232', _history())
        clear, predict, publish = self._run_same_issue([{
            'snapshot_id': 'legacy-formal',
            'based_on_issue': '2026232',
            'target_issue': target_issue,
            'version': 'kl8-legacy',
            'is_experiment': False,
        }])

        clear.assert_called_once_with()
        predict.assert_called_once_with(force_refresh=True)
        publish.assert_called_once_with({
            'statistics': {'signal_status': 'validated'},
        })

    def test_same_draw_skips_when_current_formal_snapshot_exists(self):
        target_issue = _compute_next_issue('2026232', _history())
        _, predict, publish = self._run_same_issue([{
            'snapshot_id': 'current-formal',
            'based_on_issue': '2026232',
            'target_issue': target_issue,
            'version': kl8_module.KL8_PREDICTOR_VERSION,
            'strategy_config_fingerprint': (
                kl8_module._prediction_config_fingerprint()
            ),
            'is_experiment': False,
        }])

        predict.assert_not_called()
        publish.assert_not_called()

    def test_same_version_repredicts_after_strategy_config_changes(self):
        target_issue = _compute_next_issue('2026232', _history())
        clear, predict, publish = self._run_same_issue([{
            'snapshot_id': 'same-version-old-config',
            'based_on_issue': '2026232',
            'target_issue': target_issue,
            'version': kl8_module.KL8_PREDICTOR_VERSION,
            'strategy_config_fingerprint': 'stale-config',
            'is_experiment': False,
        }])

        clear.assert_called_once_with()
        predict.assert_called_once_with(force_refresh=True)
        publish.assert_called_once()


if __name__ == '__main__':
    unittest.main()
