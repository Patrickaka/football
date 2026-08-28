# -*- coding: utf-8 -*-
"""`import src.football` 的导入期守卫

两条断言，都在子进程里做——本进程一旦 import 过，sys.modules 就被污染了，
在同一个进程里测「有没有被导入」永远是假绿。

**为什么不断言墙钟耗时**：CI 机器比本地慢且抖动，一条时间阈值会因为与代码
无关的原因变红，而「失败集合一条都不该红」这条判据正是靠稳定性才有信号
（判据 20b 吃过一次亏：`requirements.txt` 用 `>=`，CI 装的库比本地新，
黄金里第三方算出来的数直接红 5 条）。所以这里断言的是**原因**
（重库有没有进 sys.modules），不是**表征**（快不快）。原因是确定性的。
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(code, extra_env=None):
    """在干净子进程里跑一段代码，返回 stdout"""
    env = {
        'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
        'PYTHONPATH': str(REPO_ROOT),
        'PYTHONDONTWRITEBYTECODE': '1',
        'HOME': os.environ.get('HOME', '/tmp'),
    }
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=180,
    )
    if result.returncode != 0:
        raise AssertionError(f'子进程失败 rc={result.returncode}\n{result.stderr}')
    return result.stdout.strip()


class FootballImportIsCheap(unittest.TestCase):

    def test_importing_football_does_not_pull_in_the_heavy_ml_libraries(self):
        """import src.football 不得把 xgboost / lightgbm / sklearn 拉进来

        它们只有 dynamic_weights.MetaWeightModel 用得到，而构造它是惰性的。
        三个库合计约 0.76 秒，而整个 import src.football 原本才 0.85 秒——
        也就是说导入代价的九成花在一段当前走不到的路上。
        """
        loaded = _run(
            'import sys, src.football;'
            "print(','.join(m for m in ('xgboost', 'lightgbm', 'sklearn') if m in sys.modules))"
        )
        self.assertEqual(loaded, '', f'这些重库不该在导入期被拉进来: {loaded}')

    def test_importing_football_does_not_reconfigure_the_process_stdout(self):
        """import src.football 不得改掉整个进程的 sys.stdout 编码

        `config.py` 曾在模块级调用 `sys.stdout.reconfigure(encoding='utf-8')`：
        import 一个业务包，顺手改掉调用方进程的标准输出编码，属越界副作用。
        用 PYTHONIOENCODING 钉一个非 utf-8 的编码进来，import 之后它必须还在。
        """
        before = _run('import sys; print(sys.stdout.encoding)',
                      {'PYTHONIOENCODING': 'latin-1'})
        after = _run('import sys, src.football; print(sys.stdout.encoding)',
                     {'PYTHONIOENCODING': 'latin-1'})
        self.assertEqual(before, after,
                         f'import 前是 {before}，import 后变成了 {after}')

    def test_the_lazy_meta_model_is_still_reachable_and_still_builds(self):
        """惰性化不能把东西弄丢：访问 _global_meta_model 仍要拿到一个实例，
        且此时三个 ML 库才被真正导入（判据 27：既断言不报错，也断言真生效）。
        """
        out = _run(
            'import sys;'
            'import src.football.dynamic_weights as dw;'
            "assert 'xgboost' not in sys.modules, '访问之前不该导入';"
            'm = dw._global_meta_model;'
            "print(type(m).__name__, dw._global_meta_model is m, 'xgboost' in sys.modules)"
        )
        self.assertEqual(out, 'MetaWeightModel True True')


if __name__ == '__main__':
    unittest.main()
