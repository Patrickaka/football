# -*- coding: utf-8 -*-
"""`describe_exception` 的判据。

黄金文件里存过异常消息原文，于是 2026-08-31 用 homebrew 的 Python 3.14
重新生成之后，CI 的 3.13 整片对不上——3.14 改了 `math domain error` 等
一批措辞。这里守的就是「哪些消息该留、哪些该收敛成类型名」。
"""
import math
import unittest

from tests.domain.golden import describe_exception


def _raised(fn):
    try:
        fn()
    except Exception as exc:
        return describe_exception(exc)
    raise AssertionError('没有抛异常')


class ProjectMessagesSurvive(unittest.TestCase):
    """项目自己 raise 的消息是契约，必须逐字留下。"""

    def test_a_single_line_raise_keeps_its_message(self):
        def fn():
            raise ValueError("赔率值解析失败: f = '' (match_id=m1)")
        self.assertEqual(_raised(fn),
                         "ValueError: 赔率值解析失败: f = '' (match_id=m1)")

    def test_a_multiline_raise_keeps_its_message(self):
        """跨行写法的行号落在 `raise` 那一行，判据同样成立。"""
        def fn():
            raise ValueError(
                "未找到'平均值'行")
        self.assertEqual(_raised(fn), "ValueError: 未找到'平均值'行")

    def test_a_wrapped_reraise_keeps_its_message(self):
        def fn():
            try:
                int('x')
            except ValueError as exc:
                raise ValueError('欧赔数据解析失败') from exc
        self.assertEqual(_raised(fn), 'ValueError: 欧赔数据解析失败')


class InterpreterMessagesCollapse(unittest.TestCase):
    """解释器措辞随版本改，只留类型名。

    每一条都是 3.13 与 3.14 措辞不同的真实例子——留原文的话，换个解释器
    跑 regen 就会让黄金整片变红。
    """

    def test_math_domain_error_collapses(self):
        self.assertEqual(_raised(lambda: math.sqrt(-0.001)), 'ValueError')

    def test_float_division_by_zero_collapses(self):
        self.assertEqual(_raised(lambda: 1.0 / 0), 'ZeroDivisionError')

    def test_membership_on_none_collapses(self):
        self.assertEqual(_raised(lambda: 'x' in None), 'TypeError')

    def test_attribute_on_none_collapses(self):
        self.assertEqual(_raised(lambda: None.get('x')), 'AttributeError')

    def test_unpacking_mismatch_collapses(self):
        def fn():
            a, b, c, d, e, f = (1, 2, 3)
        self.assertEqual(_raised(fn), 'ValueError')

    def test_missing_key_collapses(self):
        self.assertEqual(_raised(lambda: {}['aliases']), 'KeyError')


class NoTracebackIsTolerated(unittest.TestCase):
    def test_an_exception_never_raised_collapses_to_its_type(self):
        """手工构造、没有 traceback 的异常不能让规范化本身炸掉。"""
        self.assertEqual(describe_exception(ValueError('凭空造的')), 'ValueError')


if __name__ == '__main__':
    unittest.main()
