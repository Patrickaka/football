# -*- coding: utf-8 -*-
"""缓存目录的创建必须能扛住并发。

`-n 4` 的四个 pytest worker 同时 import src.football.cache_manager，模块级的
`_global_cache_manager = FootballCacheManager()` 会同时走 `os.path.exists` →
`os.makedirs`：后到的进程撞上 FileExistsError，异常被 config.py 的
`except Exception` 吞掉，只留下一句「缓存管理器模块未导入」，最终表现为
`ImportError: cannot import name 'get_cache' from 'src.football.config'`，
整个 CI 收集阶段全红。
"""

import os
import tempfile
import unittest
from unittest import mock

from src.football.cache_manager import FootballCacheManager


class CacheDirCreationIsRaceProof(unittest.TestCase):

    def test_existing_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            FootballCacheManager(cache_dir=tmp)  # 已存在，不得抛错

    def test_directory_created_between_check_and_makedirs(self):
        """另一个进程在 exists() 与 makedirs() 之间把目录建好——正是线上那次的时序。"""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'cache')
            os.makedirs(target)
            with mock.patch('src.football.cache_manager.os.path.exists', return_value=False):
                FootballCacheManager(cache_dir=target)


if __name__ == '__main__':
    unittest.main()
