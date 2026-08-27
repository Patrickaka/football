"""纯 Python 的决策树与随机森林：三个梯度提升库都装不上时的降级路径。

**它存在的意义是「装不上库也能出号」，不是「和那三个库一样好」。** 分裂点
按分位数采样而不是全枚举，树也很浅——慢十倍还差一截，但不需要任何编译型
依赖。线上三个库都在，这条路走不到；CI 里也装了，所以它长期没有运行时信号，
**这正是它必须有测试的理由**。

迁移前这条路径实际上是断的：调用它的 `train_ensemble` 在告警时给一个模块级
标志赋值却没写 `global`，于是 Python 把那个名字当成局部变量，读它的那一行
必然抛 `UnboundLocalError`——降级从来没有真正发生过，只是被上层的 except
吞成了「训练失败」。
"""
import random

# 每个特征最多试这么多个分裂点。全枚举是 O(样本数)，分位数采样把它压成常数，
# 代价是可能错过最优分裂——降级路径要的是「能跑完」，不是「最优」。
DEFAULT_MAX_SPLITS = 20
# 叶子节点在没有样本可依据时给出的概率。0.5 表示「不知道」
UNKNOWN_PROBABILITY = 0.5
# 单棵树默认用满特征
ALL_FEATURES = 1.0


def gini(labels):
    """基尼不纯度。全同为 0，对半分最大（二分类下是 0.5）。"""
    if not labels:
        return 0
    positive = sum(labels) / len(labels)
    return 1 - (1 - positive) ** 2 - positive ** 2


class DecisionTree:
    """一棵分类树。**接收注入的随机源**——特征子集是随机抽的，
    自带随机源会让同一份数据两次训练出不同的树，回测就没法比了。
    """

    def __init__(self, max_depth, min_samples_split, feature_subset_ratio=ALL_FEATURES,
                 rng=None, max_splits_per_feature=DEFAULT_MAX_SPLITS):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.max_splits_per_feature = max_splits_per_feature
        # 单棵树默认看全部特征。随机子集是森林用来给树之间制造差异的手段，
        # 一棵孤立的树没有理由自缚手脚
        self.rng = rng if rng is not None else random.Random()
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        return [self._predict_one(row, self.tree) for row in X]

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return self._leaf(y)
        if len(set(y)) == 1:
            return {'leaf': True, 'value': y[0]}

        feature_count = len(X[0])
        chosen = max(1, int(feature_count * self.feature_subset_ratio))
        candidates = self.rng.sample(range(feature_count), chosen)

        feature, threshold, gain = self._best_split(X, y, candidates)
        # 增益为 0 也停：再分下去只是把噪声刻进树里
        if feature is None or gain <= 0:
            return self._leaf(y)

        left = [index for index in range(len(X)) if X[index][feature] <= threshold]
        right = [index for index in range(len(X)) if X[index][feature] > threshold]
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build([X[i] for i in left], [y[i] for i in left], depth + 1),
            'right': self._build([X[i] for i in right], [y[i] for i in right], depth + 1),
        }

    @staticmethod
    def _leaf(labels):
        return {'leaf': True,
                'value': sum(labels) / len(labels) if labels else UNKNOWN_PROBABILITY}

    def _best_split(self, X, y, candidates):
        """增益最大的那个 (特征, 阈值)。没有可用分裂时返回 (None, None, -1)。"""
        best_gain, best_feature, best_threshold = -1, None, None
        current = gini(y)
        total = len(y)

        for feature in candidates:
            values = [X[index][feature] for index in range(total)]
            order = sorted(range(total), key=lambda index: values[index])
            step = max(1, total // (self.max_splits_per_feature + 1))
            for cut in range(step, total - 1, step):
                threshold = (values[order[cut]] + values[order[cut + 1]]) / 2
                left = [y[order[index]] for index in range(cut + 1)]
                right = [y[order[index]] for index in range(cut + 1, total)]
                if not left or not right:
                    continue
                gain = current - (len(left) / total * gini(left)
                                  + len(right) / total * gini(right))
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold, best_gain

    def _predict_one(self, row, node):
        while not node['leaf']:
            node = (node['left'] if row[node['feature']] <= node['threshold']
                    else node['right'])
        return node['value']


class RandomForest:
    """多棵树的平均。每棵树看一份 bootstrap 重采样和一个随机特征子集。

    **所有树共用一个随机源**，不是每棵树各开一个：共用时整片森林由一个种子
    决定，可复现；各开一个的话，要么每棵树完全相同（同一个种子），
    要么整体不可复现。
    """

    def __init__(self, n_trees, max_depth, min_samples_split,
                 feature_subset_ratio=0.6, max_splits_per_feature=DEFAULT_MAX_SPLITS,
                 rng=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.max_splits_per_feature = max_splits_per_feature
        self.rng = rng if rng is not None else random.Random()
        self.trees = []

    def fit(self, X, y):
        size = len(X)
        self.trees = []
        for _ in range(self.n_trees):
            picked = [self.rng.randint(0, size - 1) for _ in range(size)]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_subset_ratio=self.feature_subset_ratio,
                max_splits_per_feature=self.max_splits_per_feature,
                rng=self.rng)
            tree.fit([X[index] for index in picked], [y[index] for index in picked])
            self.trees.append(tree)
        return self

    def predict(self, X):
        if not self.trees:
            return [UNKNOWN_PROBABILITY] * len(X)
        votes = [tree.predict(X) for tree in self.trees]
        return [sum(row[index] for row in votes) / len(votes)
                for index in range(len(X))]
