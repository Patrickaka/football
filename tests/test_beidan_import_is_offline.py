"""导入 beidan 不该联网。

迁移前 `config.py` 在模块级调用 `_init_okooo_session()`，于是 `import
src.beidan` 就发两次 HTTP 请求加一次 `sleep(0.5)`。后果不只是慢：

- 任何 import beidan 的测试都在联网，CI 因此随第三方站点波动；
- okooo 不可达时，导入要等两个 10 秒超时才继续；
- **import 期还不知道这次运行要不要用 okooo**——跑一个纯计算的单元测试
  也得先跟人家握两次手。

初始化本身没问题，问题是它发生在 import 期。改成首次真正要用时才做。
"""
import importlib
import sys
import time
import unittest
from unittest import mock

BEIDAN_MODULES = ('src.beidan', 'src.beidan.config')


def _reimport(name):
    """把 beidan 全家从 sys.modules 里摘掉再导入，让模块级代码重新执行。"""
    for loaded in [m for m in sys.modules if m.startswith('src.beidan')]:
        del sys.modules[loaded]
    return importlib.import_module(name)


class ImportIsOfflineTests(unittest.TestCase):

    def tearDown(self):
        # 别把摘干净的状态留给后面的用例
        _reimport('src.beidan.config')

    def test_importing_config_makes_no_request(self):
        with mock.patch('requests.Session.get') as get:
            _reimport('src.beidan.config')
        get.assert_not_called()

    def test_importing_package_makes_no_request(self):
        """整包导入同样不该联网——`__init__.py` 会把 config 一起拉进来。"""
        with mock.patch('requests.Session.get') as get:
            _reimport('src.beidan')
        get.assert_not_called()

    def test_importing_does_not_sleep(self):
        """预热里还有一次 `sleep(0.5)`。**它和请求是一对**，
        只掐掉请求而留下 sleep，导入照样白等半秒。"""
        with mock.patch('requests.Session.get'), \
             mock.patch('time.sleep') as slept:
            _reimport('src.beidan.config')
        slept.assert_not_called()

    def test_reimport_is_instant_once_dependencies_are_loaded(self):
        """依赖加载完之后，重新导入应当是瞬时的。

        **量的是第二次而不是第一次**：首次导入要连带加载 requests、numpy 等，
        本机 1.37 秒，而那与本模块无关，会随机器和依赖版本漂。第二次只跑
        模块自身的代码——惰性化之后是 0.001 秒量级，真联网的话两次请求
        加 sleep(0.5) 一定漏出来。
        """
        _reimport('src.beidan.config')          # 先把依赖都加载进来
        started = time.time()
        _reimport('src.beidan.config')
        self.assertLess(time.time() - started, 0.2)


class SessionStillWorksTests(unittest.TestCase):
    """惰性化不能把初始化弄丢——只是推迟。"""

    def test_ensure_initialises_on_first_call(self):
        config = _reimport('src.beidan.config')
        with mock.patch('requests.Session.get') as get, \
             mock.patch('time.sleep'):
            config.ensure_okooo_session()
        self.assertEqual(get.call_count, 2, '预热要访问首页与单场页各一次')

    def test_ensure_is_idempotent(self):
        """**第二次调用不该再发请求。** 每次取数都重新握手的话，
        惰性化就从「省掉一次」变成「每次都来一遍」。"""
        config = _reimport('src.beidan.config')
        with mock.patch('requests.Session.get') as get, \
             mock.patch('time.sleep'):
            config.ensure_okooo_session()
            config.ensure_okooo_session()
            config.ensure_okooo_session()
        self.assertEqual(get.call_count, 2)

    def test_waf_block_skips_initialisation(self):
        """撞了 WAF 就别再去敲门——这是迁移前 `_init_okooo_session` 开头
        那道判断，惰性化之后要仍然生效。"""
        config = _reimport('src.beidan.config')
        with mock.patch.object(config, '_is_okooo_waf_blocked', return_value=True), \
             mock.patch('requests.Session.get') as get, \
             mock.patch('time.sleep'):
            config.ensure_okooo_session()
        get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
