# -*- coding: utf-8 -*-
"""`get_dynamic_weights` / `fuse_predictions` 的行为守卫

**这个文件原来是判据 1 的活标本**：它在 setUp/tearDown 里存取
`dynamic_weights.NUMPY_AVAILABLE`，在每个用例里把它设成 False，名字叫
「works_without_numpy」——而这两个函数**从头到尾没有读过这个标志**
（AST 实测）。monkeypatch 什么也没做，用例测的其实只是「函数能跑通」。

2026-08-28 删掉 `MetaWeightModel` 之后本模块不再依赖 numpy，那个标志也没了。
这里保留真正在断言的部分（四路权重和为 1、融合后归一），去掉那层虚构的
「没有 numpy」外壳。
"""

import unittest

import src.football.dynamic_weights as dynamic_weights


class DynamicWeightsTests(unittest.TestCase):

    def test_get_dynamic_weights_returns_four_weights_summing_to_one(self):
        weights = dynamic_weights.get_dynamic_weights(0.6)

        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[0], 0)

    def test_fuse_predictions_normalises_over_the_union_of_scores(self):
        fused = dynamic_weights.fuse_predictions(
            {'1-0': 0.6, '1-1': 0.4},
            {'1-0': 0.5, '0-1': 0.5},
            {'1-1': 0.7, '0-1': 0.3},
            confidence=0.6,
        )

        self.assertAlmostEqual(sum(fused.values()), 1.0)
        self.assertEqual(set(fused), {'1-0', '1-1', '0-1'})

    def test_the_module_no_longer_carries_the_ml_libraries(self):
        """删掉 MetaWeightModel 的收益：本模块现在只用标准库。

        断言的是原因（模块里没有这些名字），不是表征——判据 27。
        """
        for name in ('MetaWeightModel', 'init_meta_model', '_load_ml_libs',
                     'XGBOOST_AVAILABLE', 'SKLEARN_AVAILABLE', 'np'):
            self.assertFalse(hasattr(dynamic_weights, name),
                             f'{name} 应当已随 MetaWeightModel 一并删除')


if __name__ == '__main__':
    unittest.main()
