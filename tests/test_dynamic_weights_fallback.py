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

    def test_fuse_predictions_takes_the_union_of_all_four_sources(self):
        """并集要真的是并集：**每一路都给一个只有它才有的比分**。

        判据 23——第一版语料里 elo 的两个比分被前两路完全覆盖，
        把 `| set(elo_pred.keys())` 删掉照样全绿，用例看起来在测并集。
        """
        fused = dynamic_weights.fuse_predictions(
            {'1-0': 1.0},
            {'2-0': 1.0},
            {'0-1': 1.0},
            ml_pred={'3-3': 1.0},
            confidence=0.6,
        )

        self.assertEqual(set(fused), {'1-0', '2-0', '0-1', '3-3'})

    def test_fuse_predictions_normalises_inputs_that_do_not_sum_to_one(self):
        """归一化要真的被执行到。

        判据 23——第一版语料里三路各自和为 1、权重也和为 1，融合结果本来就是
        1.0，把 `total = sum(...)` 改成 `total = 0.0`（即跳过归一）照样全绿。
        这里喂三路各自和为 10 的输入，不归一就会得到 10。
        """
        fused = dynamic_weights.fuse_predictions(
            {'1-0': 6.0, '1-1': 4.0},
            {'1-0': 5.0, '0-1': 5.0},
            {'1-1': 7.0, '0-1': 3.0},
            confidence=0.6,
        )

        self.assertAlmostEqual(sum(fused.values()), 1.0)

    def test_ml_weight_is_carved_out_of_the_other_three_proportionally(self):
        """ML 有资格参与融合时，另三路要按比例缩减腾出空间。

        这条路在测试环境里从来走不到——`ml_weight` 恒为 0，因为
        `check_ml_fusion_eligibility` 拿不到足够样本。所以按比例扣减那一步
        （`if ml_weight > 0 and ml_weight < 1.0`）改成 `if False:` 也不会红。
        判据 9 的第三行「当前调用方碰巧不触发」——有合法输入能走到，补用例。
        """
        import src.football.result_sync as result_sync

        original_eligibility = result_sync.check_ml_fusion_eligibility
        original_weight = result_sync.get_ml_fusion_weight
        result_sync.check_ml_fusion_eligibility = lambda *a, **k: {
            'eligible': True, 'shadow_samples': 500,
        }
        result_sync.get_ml_fusion_weight = lambda *a, **k: 0.25
        try:
            weights = dynamic_weights.get_dynamic_weights(0.6)
        finally:
            result_sync.check_ml_fusion_eligibility = original_eligibility
            result_sync.get_ml_fusion_weight = original_weight

        market_w, team_w, elo_w, ml_w = weights
        self.assertAlmostEqual(ml_w, 0.25)
        self.assertAlmostEqual(sum(weights), 1.0)
        # 另三路按比例缩减：0.6/0.25/0.15 各乘 0.75
        self.assertAlmostEqual(market_w, 0.45)
        self.assertAlmostEqual(team_w, 0.1875)
        self.assertAlmostEqual(elo_w, 0.1125)

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
