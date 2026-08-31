# -*- coding: utf-8 -*-
"""领域层对外输出只能是原生 JSON 类型。

**这条守卫替代了响应层的兜底。** 切换入口当天线上每条接口 500，
`TypeError: vars() argument must have __dict__ attribute`——旧入口有一层
自己的 JSON 清洗，新入口直接 `return payload` 走 FastAPI 的
`jsonable_encoder`，而它处理不了 numpy。

当时的止血是在响应层兜底。**兜底等于承认领域层可以吐任何东西**，
而且漏掉一条路由就是一个 500。现在改成在领域层的输出边界修好，
响应层回到 FastAPI 原生写法。

生产数据上实测下来只有两处（判据 10：读实测值，别猜"哪里会有 numpy"）：
- `model.dixon_coles.matrix` 是 `numpy.ndarray`——计算时它就该是 numpy，
  但进对外 `result` 时要转成 list。
- `model.half_full_time.sample_info.distance` 是 `inf`——那是"没找到匹配
  盘口"的内部哨兵，判完 quality 就没用了。**JSON 没有 Infinity 字面量**，
  输出它浏览器的 `JSON.parse` 直接抛错，整页数据都拿不到。
"""
import math
import unittest
from copy import deepcopy

import numpy as np

from src.domain.sports.football.analysis_result import _native, build_analysis_result
from tests.domain.sports.football._pipeline_corpus import BASE_PARTS

NATIVE_TYPES = (str, int, float, bool, type(None))


def offending_paths(value, path='$'):
    """列出所有不是原生 JSON 类型、或是非有限浮点的位置。"""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(offending_paths(item, f'{path}.{key}'))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(offending_paths(item, f'{path}[{index}]'))
    elif isinstance(value, bool):
        pass
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(f'{path} = {value}（JSON 没有这个字面量）')
    elif not isinstance(value, NATIVE_TYPES):
        found.append(f'{path} = {type(value).__module__}.{type(value).__name__}')
    return found


class NativeConversion(unittest.TestCase):

    def test_arrays_become_lists(self):
        self.assertEqual(_native(np.array([[1.0, 2.0], [3.0, 4.0]])),
                         [[1.0, 2.0], [3.0, 4.0]])

    def test_scalars_become_plain_numbers(self):
        self.assertEqual(_native(np.float64(0.5)), 0.5)
        self.assertEqual(_native(np.int64(3)), 3)
        self.assertIsInstance(_native(np.float64(0.5)), float)

    def test_it_reaches_into_nested_structures(self):
        self.assertEqual(_native({'a': [np.float32(1.5)], 'b': {'c': np.array([1])}}),
                         {'a': [1.5], 'b': {'c': [1]}})

    def test_native_values_pass_through_unchanged(self):
        payload = {'s': '文本', 'b': True, 'n': None, 'f': 1.5, 'i': 3}
        self.assertEqual(_native(payload), payload)

    def test_strings_are_not_mistaken_for_arrays(self):
        """`str` / `bytes` 也有 `tolist`-like 的迭代性——别把它们拆了。"""
        self.assertEqual(_native('abc'), 'abc')
        self.assertEqual(_native(b'abc'), b'abc')

    def test_booleans_stay_booleans(self):
        """`bool` 有 `.item()` 吗？没有；但 numpy 的 `bool_` 有。
        原生 `True` 不能被 `.item()` 那条分支吃掉。
        """
        self.assertIs(_native(True), True)


class AnalysisResultIsSerialisable(unittest.TestCase):

    def test_the_output_has_no_numpy_and_no_infinities(self):
        """**这是那次线上故障的直接守卫。**"""
        result = build_analysis_result(**deepcopy(BASE_PARTS))
        self.assertEqual(offending_paths(result, 'result'), [])

    def test_a_numpy_matrix_in_the_input_comes_out_native(self):
        """Dixon-Coles 的矩阵进来时就是 numpy——那是它计算时该有的样子。"""
        parts = deepcopy(BASE_PARTS)
        parts['dixon_coles_result'] = {'matrix': np.array([[0.1, 0.2], [0.3, 0.4]]),
                                       'rho': np.float64(-0.05)}
        result = build_analysis_result(**parts)
        self.assertEqual(offending_paths(result, 'result'), [])
        self.assertEqual(result['model']['dixon_coles']['matrix'],
                         [[0.1, 0.2], [0.3, 0.4]])

    def test_the_whole_payload_survives_json_dumps(self):
        """最终判准：标准库能不能把它变成合法 JSON。

        `allow_nan=False` 是关键——默认的 `True` 会输出 `Infinity`/`NaN`，
        那不是合法 JSON，浏览器解析不了。
        """
        import json

        parts = deepcopy(BASE_PARTS)
        parts['dixon_coles_result'] = {'matrix': np.array([[0.1]]), 'rho': np.float64(0.0)}
        json.dumps(build_analysis_result(**parts), allow_nan=False, ensure_ascii=False)


class SampleDistanceHasNoInfinity(unittest.TestCase):
    """`distance` 的 `inf` 哨兵不能出现在对外字段里。"""

    def test_the_conversion_is_in_place(self):
        """`calculate_half_full_time_probs` 要注入半场统计库，按设计留在
        适配层 `src/football/scoring.py`——**输出边界不一定在领域层**，
        它在"最后一个碰这个值的地方"。
        """
        import pathlib
        source = pathlib.Path('src/football/scoring.py').read_text(encoding='utf-8')
        self.assertIn('if math.isfinite(distance) else None', source,
                      'sample_info.distance 的 inf 转换被去掉了')

    def test_the_quality_judgement_still_uses_the_raw_distance(self):
        """**转换必须在判完 quality 之后**：`distance <= 0.3` 这类比较要拿
        原始的 inf 参与，换成 None 会直接 TypeError。
        """
        import pathlib
        source = pathlib.Path('src/football/scoring.py').read_text(encoding='utf-8')
        quality_line = source.index("quality = 'high' if")
        conversion_line = source.index('if math.isfinite(distance) else None')
        self.assertLess(quality_line, conversion_line)

    def test_json_cannot_express_infinity(self):
        """说明这条守卫为什么必要——不是风格问题，是 JSON 规范。"""
        import json

        with self.assertRaises(ValueError):
            json.dumps({'d': math.inf}, allow_nan=False)


class NoFallbackInTheResponseLayer(unittest.TestCase):

    def test_the_custom_response_class_is_gone(self):
        """响应层回到 FastAPI 原生——**兜底删掉了，靠的是领域层不吐脏数据**。"""
        import pathlib
        self.assertFalse(pathlib.Path('src/api/responses.py').exists())

    def test_routes_use_the_native_style(self):
        import pathlib
        # lottery 路由已随大乐透、双色球、福彩 3D 和排列五功能一起下线；
        # 这里只检查仍在提供服务的业务路由。
        for module in ('basketball', 'beidan', 'kl8', 'football'):
            source = pathlib.Path(f'src/api/routers/{module}.py').read_text(encoding='utf-8')
            with self.subTest(module=module):
                self.assertNotIn('json_result', source)
                self.assertIn('await run_blocking(', source)


if __name__ == '__main__':
    unittest.main()
