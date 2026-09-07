"""策略验证只在开奖数据变化后重跑。

线上实测：验证每 2 小时跑一轮、每轮约 50 分钟 CPU，38 个候选在同一份
2060 期数据上每次全部落选，而两轮之间根本没有新开奖。这里守住
「没有新期号就不重算」。
"""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import src.kl8 as kl8_module
from src.kl8 import records as kl8_records
from src.kl8 import scheduler as kl8_scheduler


class _Analyzer:
    def __init__(self, latest_issue):
        self.history_data = [{'issue': latest_issue}, {'issue': '2026001'}]


class _Backtest:
    calls = []

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def run_candidate_tournament_per_play_type(self, play_type, candidate_strategies=None):
        _Backtest.calls.append(play_type)
        return {'all_failed': True}


class VerificationSkipTests(unittest.TestCase):

    def setUp(self):
        _Backtest.calls = []
        self.tmp = tempfile.TemporaryDirectory()
        self.state_file = str(Path(self.tmp.name) / 'kl8_verification_state.json')
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(
            kl8_records, 'KL8_VERIFICATION_STATE_FILE', self.state_file))
        self.stack.enter_context(mock.patch.object(
            kl8_scheduler, 'count_valid_history_periods', return_value=2060))
        self.stack.enter_context(mock.patch.object(kl8_scheduler, 'clear_cache'))
        self.stack.enter_context(mock.patch.object(kl8_module, 'ACTIVE_STRATEGIES', {}))
        self.stack.enter_context(mock.patch.object(kl8_module, 'KL8RollingBacktest', _Backtest))
        self.stack.enter_context(mock.patch.object(
            kl8_module, 'get_kl8_analyzer', return_value=_Analyzer('2026239')))

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def test_first_run_verifies_every_play_type_and_records_the_issue(self):
        kl8_scheduler.run_verified_strategy_selection_if_needed()

        self.assertEqual(_Backtest.calls, [
            'select_3', 'select_4', 'select_5', 'select_6', 'select_7', 'fu_shi_7'])
        state = json.loads(Path(self.state_file).read_text(encoding='utf-8'))
        self.assertEqual(state['latest_issue'], '2026239')
        self.assertEqual(state['play_types'], _Backtest.calls)

    def test_same_issue_does_not_rerun_the_tournament(self):
        kl8_scheduler.run_verified_strategy_selection_if_needed()
        _Backtest.calls = []

        kl8_scheduler.run_verified_strategy_selection_if_needed()

        self.assertEqual(_Backtest.calls, [])

    def test_a_new_issue_reruns_the_tournament(self):
        kl8_scheduler.run_verified_strategy_selection_if_needed()
        _Backtest.calls = []
        self.stack.enter_context(mock.patch.object(
            kl8_module, 'get_kl8_analyzer', return_value=_Analyzer('2026240')))

        kl8_scheduler.run_verified_strategy_selection_if_needed()

        self.assertEqual(len(_Backtest.calls), 6)

    def test_a_play_type_left_out_last_time_still_gets_verified(self):
        Path(self.state_file).write_text(json.dumps(
            {'latest_issue': '2026239', 'play_types': ['select_3']}), encoding='utf-8')

        kl8_scheduler.run_verified_strategy_selection_if_needed()

        self.assertEqual(_Backtest.calls, [
            'select_4', 'select_5', 'select_6', 'select_7', 'fu_shi_7'])


if __name__ == '__main__':
    unittest.main()
