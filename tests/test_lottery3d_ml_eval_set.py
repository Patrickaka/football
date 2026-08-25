import random
import unittest

import src.lottery3d.ml as ml


def _make_dataset(n_features=4, n_train=60, n_valid=20):
    rng = random.Random(0)
    make_rows = lambda n: [[rng.random() for _ in range(n_features)] for _ in range(n)]
    make_labels = lambda n: [rng.randint(0, 1) for _ in range(n)]
    return make_rows(n_train), make_labels(n_train), make_rows(n_valid), make_labels(n_valid)


@unittest.skipUnless(ml.HAS_XGBOOST, "xgboost 未安装")
class XGBoostEvalSetTests(unittest.TestCase):
    def test_xgboost_trains_with_eval_set(self):
        X, y, X_valid, y_valid = _make_dataset()

        model, used_name = ml.train_single_model(
            X, y, "xgboost", eval_set=(X_valid, y_valid)
        )

        self.assertIsNotNone(model, "xgboost 应能在带 eval_set 时完成训练")
        self.assertEqual(used_name, "xgboost")

    def test_xgboost_trains_without_eval_set(self):
        X, y, _, _ = _make_dataset()

        model, used_name = ml.train_single_model(X, y, "xgboost", eval_set=None)

        self.assertIsNotNone(model, "样本不足以切出验证集时 xgboost 仍应完成训练")
        self.assertEqual(used_name, "xgboost")

    def test_xgboost_joins_ensemble(self):
        X, y, _, _ = _make_dataset(n_train=200)
        group_ids = [i // 4 for i in range(len(X))]

        trained, _ = ml.train_ensemble(
            X, y, models_to_try=["xgboost"], group_ids=group_ids
        )

        self.assertIn("xgboost", [name for _, name, _ in trained])


if __name__ == '__main__':
    unittest.main()
