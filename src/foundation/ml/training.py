from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TrainValidSplit:
    X_train: list
    y_train: list
    X_valid: Optional[list]
    y_valid: Optional[list]
    use_valid: bool

    def eval_set(self):
        """统一的验证集表示：(X, y) 元组或 None。各库形状由适配函数转换。"""
        if not self.use_valid:
            return None
        return (self.X_valid, self.y_valid)


def split_by_group(X, y, group_ids, ratio=0.8, min_valid=5, min_groups=10):
    """按分组（如期号）做时序切分，避免同组样本跨越训练/验证边界造成泄漏。

    分组不足时退化为按样本位置切分。
    """
    if group_ids and len(group_ids) == len(X):
        unique = sorted(set(group_ids))
        if len(unique) >= min_groups:
            boundary = unique[int(len(unique) * ratio)]
            train_idx = [i for i, g in enumerate(group_ids) if g <= boundary]
            valid_idx = [i for i, g in enumerate(group_ids) if g > boundary]
            if len(valid_idx) >= min_valid:
                return TrainValidSplit(
                    X_train=[X[i] for i in train_idx],
                    y_train=[y[i] for i in train_idx],
                    X_valid=[X[i] for i in valid_idx],
                    y_valid=[y[i] for i in valid_idx],
                    use_valid=True,
                )
        return TrainValidSplit(X, y, None, None, use_valid=False)

    split = max(min_valid, int(len(X) * ratio))
    if len(X) - split < min_valid:
        return TrainValidSplit(X, y, None, None, use_valid=False)
    return TrainValidSplit(
        X_train=X[:split],
        y_train=y[:split],
        X_valid=X[split:],
        y_valid=y[split:],
        use_valid=True,
    )


def xgboost_eval_set(eval_set):
    """XGBoost 要求 Sequence[Tuple[X, y]]。

    传入单个 (X, y) 元组会让 XGBoost 按样本行解包，报
    'too many values to unpack (expected 2)'。
    """
    return _as_pair_list(eval_set)


def lightgbm_eval_set(eval_set):
    """LightGBM 同样要求列表形式。"""
    return _as_pair_list(eval_set)


def _as_pair_list(eval_set):
    if eval_set is None:
        return None
    if isinstance(eval_set, list):
        return eval_set
    return [eval_set]


def blend_weights(scores):
    """按验证得分归一化为融合权重。全零时退化为均匀分布。"""
    scores = list(scores)
    if not scores:
        return []
    total = sum(scores)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [s / total for s in scores]
