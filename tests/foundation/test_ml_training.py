import unittest

from src.foundation.ml.training import (
    blend_weights,
    lightgbm_eval_set,
    split_by_group,
    xgboost_eval_set,
)


class SplitByGroupTests(unittest.TestCase):
    def test_splits_by_group_boundary(self):
        X = [[i] for i in range(20)]
        y = [i % 2 for i in range(20)]
        groups = [i // 2 for i in range(20)]
        split = split_by_group(X, y, groups, ratio=0.8, min_valid=2)
        self.assertTrue(split.use_valid)
        self.assertEqual(len(split.X_train) + len(split.X_valid), 20)

    def test_no_sample_appears_in_both_sides(self):
        X = [[i] for i in range(20)]
        y = [0] * 20
        groups = [i // 2 for i in range(20)]
        split = split_by_group(X, y, groups, ratio=0.8, min_valid=2)
        self.assertEqual(len(split.X_train) + len(split.X_valid), len(X))

    def test_falls_back_when_too_few_groups(self):
        X = [[i] for i in range(6)]
        y = [0] * 6
        groups = [0, 0, 1, 1, 2, 2]
        split = split_by_group(X, y, groups, ratio=0.8, min_valid=5)
        self.assertFalse(split.use_valid)
        self.assertEqual(split.X_train, X)
        self.assertIsNone(split.X_valid)

    def test_falls_back_when_group_ids_missing(self):
        X = [[i] for i in range(20)]
        y = [0] * 20
        split = split_by_group(X, y, None, ratio=0.8, min_valid=2)
        self.assertTrue(split.use_valid)
        self.assertEqual(len(split.X_train), 16)


class EvalSetShapeTests(unittest.TestCase):
    """XGBoost 要求 [(X, y)] 列表；传单个元组会报
    'too many values to unpack (expected 2)'。此坑 2026-08-25 在线上发生过。"""

    def test_xgboost_wraps_tuple_in_list(self):
        self.assertEqual(xgboost_eval_set(([[1]], [0])), [([[1]], [0])])

    def test_xgboost_none_stays_none(self):
        self.assertIsNone(xgboost_eval_set(None))

    def test_xgboost_accepts_already_wrapped(self):
        self.assertEqual(xgboost_eval_set([([[1]], [0])]), [([[1]], [0])])

    def test_lightgbm_wraps_tuple_in_list(self):
        self.assertEqual(lightgbm_eval_set(([[1]], [0])), [([[1]], [0])])

    def test_lightgbm_none_stays_none(self):
        self.assertIsNone(lightgbm_eval_set(None))


class BlendWeightsTests(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(blend_weights([0.6, 0.7, 0.8])), 1.0)

    def test_higher_score_gets_higher_weight(self):
        weights = blend_weights([0.6, 0.9])
        self.assertGreater(weights[1], weights[0])

    def test_equal_scores_get_equal_weights(self):
        self.assertEqual(blend_weights([0.5, 0.5]), [0.5, 0.5])

    def test_empty_scores_return_empty(self):
        self.assertEqual(blend_weights([]), [])

    def test_all_zero_scores_fall_back_to_uniform(self):
        self.assertEqual(blend_weights([0, 0]), [0.5, 0.5])


if __name__ == '__main__':
    unittest.main()
