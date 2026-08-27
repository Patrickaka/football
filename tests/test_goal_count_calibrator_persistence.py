#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GoalCountCalibrator 跨存储往返的行为。

**进球数是 int，而 kv_store 底层走 JSON，JSON 的对象键只能是字符串。**
存进去的 `{2: 1.17}` 读回来是 `{"2": 1.17}`，于是：

1. `factors.get(2)` 查不到 `"2"` → 因子按 1.0 处理，**校准等于没做**；
2. `set(goal_dist) | set(factors)` 混进两种类型的键，
   `sorted()` 当场抛 `'<' not supported between instances of 'str' and 'int'`，
   被上层 `except` 吞成「进球数校准失败，使用原始分布」。

线上三天 2294 条该告警，434 个分桶里 74 个有因子——**一次都没生效过**。

既有的池化测试都在内存里 `record_result` 完立刻读，没经过存储，
所以这条路径此前完全没有测试。这个文件专门补上「存过再读」。
"""
import json
import unittest
from unittest import mock

from src.common import kv_store
from src.football.goal_count_calibrator import GoalCountCalibrator


def _reloaded(calibrator):
    """模拟一次真实的存取：db 经 JSON 往返后，由 `_load()` 重新读进来。

    **必须走 `_load()`**——修复的位置就在那道防腐层上：外部表示（字符串键）
    在进门时转成领域表示（int 键）。测试若直接给 `db` 赋值就绕过了它，
    修没修都测不出来。`kv_store.load` 被打桩，不碰真实存储。
    """
    raw = json.loads(json.dumps(calibrator.db))
    with mock.patch.object(kv_store, 'load', return_value=raw):
        return GoalCountCalibrator()


class PersistedKeyTypeTests(unittest.TestCase):
    """线上 kv 里的真实形状：因子与历史分布的键都是字符串。"""

    LEAGUE = '瑞典超'
    LINE = 2.75
    ASIAN = 0.5
    EXPECTED = 2.5
    # 模型给出的分布，键是 int——这一侧永远是 int，它不经过存储
    MODEL_DIST = {0: 0.10, 1: 0.25, 2: 0.30, 3: 0.20, 4: 0.15}

    def _calibrator_with_history(self, samples=12):
        calibrator = GoalCountCalibrator()
        calibrator.db = {}
        for _ in range(samples):
            calibrator.record_result(
                self.LEAGUE, self.LINE, self.MODEL_DIST,
                actual_total_goals=3, expected_total_goals=self.EXPECTED,
                asian=self.ASIAN, sample_weight=1.0)
        return calibrator

    def test_factors_survive_the_roundtrip_as_usable_keys(self):
        """存过再读，因子仍要能被 int 进球数查到。

        这是这个 bug 的核心：**键的类型在往返中变了，而查询端没变。**
        """
        calibrator = _reloaded(self._calibrator_with_history())

        key = calibrator._get_bucket_key(self.LEAGUE, self.LINE, self.ASIAN, self.EXPECTED)
        factors = calibrator.db[key]['calibration_factors']
        self.assertTrue(factors, '前置条件：这份历史应当产出因子')
        self.assertEqual(sorted({type(k).__name__ for k in factors}), ['int'],
                         '往返之后因子的键必须仍是 int')

    def test_calibration_does_not_raise_after_roundtrip(self):
        """**迁移前这里抛 TypeError**，被上层吞成「使用原始分布」。"""
        calibrator = _reloaded(self._calibrator_with_history())

        calibrated = calibrator.calibrate_goal_dist(
            league=self.LEAGUE, total_line=self.LINE, goal_dist=self.MODEL_DIST,
            expected_total=self.EXPECTED, asian=self.ASIAN, min_samples=4)
        self.assertEqual(sorted({type(k).__name__ for k in calibrated}), ['int'],
                         '输出的键必须全是 int，混了类型下游 sorted 就会炸')

    def test_calibration_actually_changes_the_distribution(self):
        """**不抛异常还不够，得真的校准。**

        往返之后即使不抛，`factors.get(2)` 查不到 `"2"` 也会让每个因子都回落
        到 1.0——分布原样返回，看起来一切正常。这条用例区分「没崩」和「生效」。
        """
        calibrator = _reloaded(self._calibrator_with_history())

        calibrated = calibrator.calibrate_goal_dist(
            league=self.LEAGUE, total_line=self.LINE, goal_dist=self.MODEL_DIST,
            expected_total=self.EXPECTED, asian=self.ASIAN, min_samples=4)
        self.assertNotEqual(calibrated, self.MODEL_DIST)
        # 实际总是开 3 球，而模型只给 3 球 0.20——校准该把它抬上去
        self.assertGreater(calibrated[3], self.MODEL_DIST[3])

    def test_roundtrip_result_matches_in_memory_result(self):
        """存过一轮和没存过，校准结果应当一致。

        这是最强的那条：**持久化不该改变行为**。两边任何一处漂了它都会红。
        """
        fresh = self._calibrator_with_history()
        in_memory = fresh.calibrate_goal_dist(
            league=self.LEAGUE, total_line=self.LINE, goal_dist=self.MODEL_DIST,
            expected_total=self.EXPECTED, asian=self.ASIAN, min_samples=4)

        persisted = _reloaded(self._calibrator_with_history())
        after = persisted.calibrate_goal_dist(
            league=self.LEAGUE, total_line=self.LINE, goal_dist=self.MODEL_DIST,
            expected_total=self.EXPECTED, asian=self.ASIAN, min_samples=4)

        self.assertEqual(sorted(in_memory), sorted(after))
        for goals in in_memory:
            self.assertAlmostEqual(in_memory[goals], after[goals], places=9)

    def test_recording_onto_persisted_history_keeps_int_keys(self):
        """在读回来的历史上继续累积样本，键不能又退回字符串。

        真实的循环是「读 → 加一场 → 存」，只在读的时候修一次是不够的。
        """
        calibrator = _reloaded(self._calibrator_with_history())
        calibrator.record_result(
            self.LEAGUE, self.LINE, self.MODEL_DIST,
            actual_total_goals=2, expected_total_goals=self.EXPECTED,
            asian=self.ASIAN, sample_weight=1.0)

        key = calibrator._get_bucket_key(self.LEAGUE, self.LINE, self.ASIAN, self.EXPECTED)
        bucket = calibrator.db[key]
        self.assertEqual(sorted({type(k).__name__ for k in bucket['calibration_factors']}),
                         ['int'])
        for dist in bucket['predicted_distributions']:
            self.assertEqual(sorted({type(k).__name__ for k in dist}), ['int'])

    def test_historical_distributions_survive_the_roundtrip(self):
        """`predicted_distributions` 里每条分布的键同样是进球数。

        线上实测它们也全是字符串——因子算得对不对，取决于这一份读得对不对。
        """
        calibrator = _reloaded(self._calibrator_with_history())

        key = calibrator._get_bucket_key(self.LEAGUE, self.LINE, self.ASIAN, self.EXPECTED)
        for dist in calibrator.db[key]['predicted_distributions']:
            self.assertEqual(sorted({type(k).__name__ for k in dist}), ['int'])


if __name__ == '__main__':
    unittest.main()
