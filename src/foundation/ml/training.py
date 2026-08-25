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
    """把统一的 (X, y) 表示转成各库要求的 [(X, y)] 列表。

    刻意不靠 list/tuple 类型来猜测调用方意图——[X, y] 与 [(X, y)] 在类型上
    无法区分，猜错会静默把错误形状喂给训练器，正是本模块要消除的那类问题。
    """
    if eval_set is None:
        return None
    if isinstance(eval_set, tuple) and len(eval_set) == 2:
        return [eval_set]
    if isinstance(eval_set, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in eval_set):
            return eval_set          # 已是 [(X, y), ...]
        raise ValueError(
            'eval_set 为 list 时必须是 [(X, y), ...] 形式；'
            '单个验证集请传 (X, y) 元组，不要传 [X, y]'
        )
    raise ValueError(f'不支持的 eval_set 类型: {type(eval_set).__name__}')


def blend_weights(scores):
    """按验证得分归一化为融合权重。

    负分截断为 0 再归一化：验证得分理论上非负（AUC/准确率类指标），但函数
    接收外部输入，不应假设。截断而非报错，是因为个别模型跑出负分（用了
    log-loss、相关系数等可能为负的指标，或模型本身表现差于基线）是训练期的
    正常噪声，不代表调用方传参出错——直接拒绝会让上游融合流程为了一个模型
    的低分而整体炸掉，代价大于把它的权重记为 0（等同于"不参与融合"）。
    截断后全零（含空列表之外的情况）时退化为均匀分布。
    """
    scores = list(scores)
    if not scores:
        return []
    clipped = [max(0.0, s) for s in scores]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [s / total for s in clipped]
