# -*- coding: utf-8 -*-
"""按比赛循环的热路径不得每次重建 PredictionHistory。

`PredictionHistory()` 的构造函数会整表读 football_prediction（doc 列合计约
30MB / 742 行）。`analyze_match` 里每场比赛构造一次，一次首页刷新 16 场就是
十几次整表读 + JSON 解析，3.6GB 内存的线上机直接被打满（页面 Failed to fetch、
SSH 握手超时）。历史记录在进程内本来就是共享的，热路径必须复用同一个实例。
"""

import re
import unittest
from pathlib import Path
from unittest import mock


class SharedHistoryInstance(unittest.TestCase):

    def test_get_history_reuses_the_process_wide_instance(self):
        from src.football import result_sync

        with mock.patch(
                'src.common.repositories.football_prediction_load',
                side_effect=AssertionError('共享实例不得重新整表读'),
        ):
            first = result_sync.get_history()
            second = result_sync.get_history()

        self.assertIs(first, second)
        self.assertIs(first, result_sync._global_history)


class HotPathsDoNotRebuildHistory(unittest.TestCase):
    """源码守卫：构造点只允许留全局实例那一处。"""

    def _construction_count(self, path):
        source = Path(path).read_text(encoding='utf-8')
        return len(re.findall(r'PredictionHistory\(\)', source))

    def test_analyze_match_does_not_construct_history(self):
        self.assertEqual(self._construction_count('src/football/pipeline.py'), 0)

    def test_api_service_does_not_construct_history(self):
        self.assertEqual(self._construction_count('src/api/services/football.py'), 0)

    def test_result_sync_constructs_only_the_global_instance(self):
        source = Path('src/football/result_sync.py').read_text(encoding='utf-8')
        self.assertEqual(len(re.findall(r'PredictionHistory\(\)', source)), 1)
        self.assertIn('_global_history = PredictionHistory()', source)


if __name__ == '__main__':
    unittest.main()
