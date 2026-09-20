# -*- coding: utf-8 -*-
"""kl8 策略试验记录对领域层的适配。

这层替掉的是「33839 条全量常驻内存 + 每条新增整表重写 26.7MB JSON」。
测试盯住三件会静默出错的事：定位当前条不能再靠对象身份、读失败不能降级成
空列表、以及落库失败时当前条仍要参与本次 FDR。
"""
import unittest
from unittest import mock

from src.domain.numeric.repository import create_all
from src.domain.numeric.trial_store import TrialStore
from src.foundation.store import Database, make_engine
from src.kl8 import trial_sync
from tests.domain.numeric.test_trial_store import REAL


def _trial(**overrides):
    return {**REAL, **overrides}


class _Base(unittest.TestCase):
    def setUp(self):
        db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(db)
        self.store = TrialStore(db, game='kl8')
        trial_sync.set_store(self.store)
        self.addCleanup(trial_sync.reset_store)


class FamilyTests(_Base):
    def test_main_plays_form_one_family(self):
        self.assertEqual(set(trial_sync.family_play_types('select_6')),
                         {'select_6', 'fu_shi_7'})

    def test_other_plays_stand_alone(self):
        self.assertEqual(trial_sync.family_play_types('select_3'), ('select_3',))

    def test_exposures_are_excluded_from_the_testing_family(self):
        trial_sync.record_trial(_trial(play_type='select_6',
                                       tournament_round='holdout_exposure'))
        trial_sync.record_trial(_trial(play_type='select_6',
                                       tournament_round='validation'))
        rounds = [t['tournament_round'] for t in trial_sync.family_trials('select_6')]
        self.assertEqual(rounds, ['validation'])

    def test_exposures_are_visible_when_asked_for(self):
        trial_sync.record_trial(_trial(play_type='select_6',
                                       tournament_round='holdout_exposure'))
        self.assertEqual(len(trial_sync.family_trials('select_6',
                                                      include_exposures=True)), 1)


class LocateTests(_Base):
    def test_locates_by_key_not_identity(self):
        """查库返回的是新对象，`is` 比较不再成立。"""
        trial = _trial()
        self.assertEqual(trial_sync.index_of([dict(trial)], trial), 0)

    def test_returns_none_when_absent(self):
        self.assertIsNone(trial_sync.index_of([], _trial()))

    def test_distinguishes_trials_sharing_everything_but_tested_at(self):
        first = _trial(tested_at='2026-06-26T14:00:01')
        second = _trial(tested_at='2026-06-26T14:00:02')
        self.assertEqual(trial_sync.index_of([first, second], second), 1)


class CurrentTrialTests(_Base):
    def test_current_trial_is_present_after_a_successful_write(self):
        trial = _trial(play_type='select_3')
        trial_sync.record_trial(trial)
        trials, index = trial_sync.family_with_current('select_3', trial)
        self.assertEqual(trial_sync.trial_key(trials[index]),
                         trial_sync.trial_key(trial))

    def test_current_trial_still_counts_when_the_write_failed(self):
        """落库失败时也要参与本次校正，否则 FDR 按少一条的族算，p 值偏松。"""
        trial = _trial(play_type='select_3')
        trials, index = trial_sync.family_with_current('select_3', trial)
        self.assertEqual(trials[index], trial)
        self.assertEqual(len(trials), 1)


class FailureModeTests(_Base):
    def test_a_write_failure_is_reported_not_raised(self):
        with mock.patch.object(self.store, 'append', side_effect=RuntimeError('boom')):
            self.assertFalse(trial_sync.record_trial(_trial()))

    def test_a_read_failure_raises_instead_of_degrading(self):
        """读不到历史试验时 FDR 只剩当前一条，等于不校正——必须让它失败。"""
        with mock.patch.object(self.store, 'family_trials',
                               side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                trial_sync.family_trials('select_3')
