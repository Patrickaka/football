# 福彩 3D 预测器 V5.0 - 向量化 + 双模型版（纯 Python + LightGBM）
# Python 3.10+
"""
基于双模型的福彩 3D 预测系统（自动检测并使用 LightGBM，如果不可用则降级为纯 Python）
特征工程：
  - 每个数字的热度得分（3 个数字的各自热度）
  - 每个位置的热度得分
  - 每个位置的马尔可夫转移概率
  - 数字的遗漏值
  - 和值、跨度、奇偶比、大小比等组合特征
  - 与上期号码的相似度（相同位置相同个数、任意位置相同个数）
训练策略：
  - 正例：历史每一期的开奖号码
  - 负例：每期随机抽取 30 个未开出的号码
  - 时序验证：前 80% 期训练，后 20% 期验证
  - 使用 LightGBM（优先）或简易决策树集成（降级方案）
预测：
  - 对所有 1000 个直选组合预测概率
  - 取 Top K 作为推荐
"""
import math
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations, product
from logger import setup_logger

log = setup_logger('lottery3d_ml')

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

URL = "https://www.8300.cn/kjhhis/3/200.html"

# 模型参数
BACKTEST_TRIALS = 80
TRAIN_RATIO = 0.8  # 时序划分比例
NEGATIVE_SAMPLES_PER_PERIOD = 30  # 每期负例采样数（减少以加速）
TOP_K = 15  # 推荐注数
N_TREES = 20  # 树的数量（减少以加速）
MAX_DEPTH = 2  # 树的最大深度（减少以加速）
MIN_SAMPLES_SPLIT = 30  # 节点分裂所需最小样本数（增加以加速）
FEATURE_SUBSET_RATIO = 0.6  # 每棵树随机选择特征比例


def _native_number(x):
    """numpy 标量 → Python int/float，便于 JSON 序列化"""
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, float):
        return float(x)
    return x


def fetch_data(url=URL):
    """获取历史开奖数据"""
    log.debug('fetch 3D-ML data')
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    compact = re.sub(r"\s+", " ", html)
    pattern = re.compile(
        r'<td>(\d{7})期</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>'
        r'\s*<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>'
    )
    rows = pattern.findall(compact)
    data = [(pid, dt, (int(a), int(b), int(c))) for pid, dt, a, b, c in rows]
    data.reverse()
    return data


def miss_value(numbers, digit, position=None):
    """计算遗漏值"""
    for i in range(len(numbers) - 1, -1, -1):
        n = numbers[i]
        if position is None:
            if digit in n:
                return len(numbers) - 1 - i
        elif n[position] == digit:
            return len(numbers) - 1 - i
    return len(numbers)


def neighbor(d):
    """返回邻号"""
    return {(d - 1) % 10, (d + 1) % 10}


def road(d):
    """返回 012 路"""
    return d % 3


def exp_weighted_counts(series, decay=0.96):
    """指数加权计数"""
    cnt = Counter()   
    w = 1.0
    for item in reversed(series):
        cnt[item] += w
        w *= decay
    return cnt


def build_markov(numbers, position):
    """构建马尔可夫转移矩阵"""
    trans = defaultdict(Counter)
    for i in range(len(numbers) - 1):
        a, b = numbers[i][position], numbers[i + 1][position]
        trans[a][b] += 1
    return trans


def markov_prob_smoothed(row, states, alpha=1.0):
    """转移概率（拉普拉斯平滑）"""
    states = list(states)
    row_total = sum(row.values())
    denom = row_total + alpha * len(states)
    return {s: (row.get(s, 0) + alpha) / denom for s in states}


def odd_even_key(triple):
    """奇偶比"""
    odds = sum(1 for d in triple if d % 2 == 1)
    return odds, 3 - odds


def big_small_key(triple):
    """大小比（0-4 小，5-9 大）"""
    big = sum(1 for d in triple if d >= 5)
    return big, 3 - big


def has_consecutive_digits(a, b, c):
    """是否存在相邻连号"""
    digits = (a, b, c)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(digits[i] - digits[j]) == 1:
                return True
    return False


def position_repeat_count(triple, last_draw):
    """与上期同位置重复个数"""
    return sum(1 for i in range(3) if triple[i] == last_draw[i])


def digit_overlap_count(triple, last_draw):
    """任意位置相同数字个数（集合交集）"""
    return len(set(triple) & set(last_draw))


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, numbers, window=90):
        self.numbers = numbers
        self.window = window
        self.recent = numbers[-window:] if len(numbers) > window else list(numbers)
        self.last_draw = numbers[-1] if numbers else None
        
        # 预计算统计量
        self._precompute()
    
    def _precompute(self):
        """预计算常用统计量"""
        # 全局热度（指数加权）
        self.freq_global = exp_weighted_counts(
            [d for n in self.recent for d in n]
        )
        self.total_global = sum(self.freq_global.values()) or 1.0
        
        # 分位热度
        self.freq_pos = []
        self.total_pos = []
        for pos in range(3):
            freq = exp_weighted_counts([n[pos] for n in self.recent])
            self.freq_pos.append(freq)
            self.total_pos.append(sum(freq.values()) or 1.0)
        
        # 马尔可夫转移
        self.markov_trans = []
        for pos in range(3):
            trans = build_markov(self.numbers, pos)
            self.markov_trans.append(trans)
        
        # 遗漏值
        self.miss_global = {d: miss_value(self.numbers, d) for d in range(10)}
        self.miss_pos = [
            {d: miss_value(self.numbers, d, position=pos) for d in range(10)}
            for pos in range(3)
        ]
        
        # 形态统计
        self.oe_freq = Counter()
        self.bs_freq = Counter()
        for n in self.recent:
            self.oe_freq[odd_even_key(n)] += 1
            self.bs_freq[big_small_key(n)] += 1
        
        # 和值跨度统计
        self.sums = [sum(n) for n in self.recent]
        self.spans = [max(n) - min(n) for n in self.recent]
        self.sum_freq = Counter(self.sums)
        self.span_freq = Counter(self.spans)
    
    def build_features(self, a, b, c):
        """为单个直选组合 (a,b,c) 构造特征向量"""
        features = []
        triple = (a, b, c)
        
        # 1. 每个数字的全局热度得分（3 个特征）
        for d in triple:
            features.append(self.freq_global.get(d, 0) / self.total_global)
        
        # 2. 每个位置的热度得分（3 个特征）
        for pos, d in enumerate(triple):
            features.append(self.freq_pos[pos].get(d, 0) / self.total_pos[pos])
        
        # 3. 每个位置的马尔可夫转移概率（3 个特征）
        for pos, d in enumerate(triple):
            prev_d = self.last_draw[pos] if self.last_draw else 0
            trans = self.markov_trans[pos].get(prev_d, Counter())
            prob = markov_prob_smoothed(trans, range(10)).get(d, 0.1)
            features.append(prob)
        
        # 4. 每个数字的遗漏值（3 个特征）
        for d in triple:
            features.append(self.miss_global.get(d, 0))
        
        # 5. 每个位置的遗漏值（3 个特征）
        for pos, d in enumerate(triple):
            features.append(self.miss_pos[pos].get(d, 0))
        
        # 6. 和值特征（2 个特征）
        s = a + b + c
        features.append(s)  # 和值
        features.append(self.sum_freq.get(s, 0))  # 和值频次
        
        # 7. 跨度特征（2 个特征）
        span = max(a, b, c) - min(a, b, c)
        features.append(span)  # 跨度
        features.append(self.span_freq.get(span, 0))  # 跨度频次
        
        # 8. 奇偶比（2 个特征）
        oe = odd_even_key(triple)
        features.append(oe[0])  # 奇数个数
        features.append(self.oe_freq.get(oe, 0))  # 奇偶比频次
        
        # 9. 大小比（2 个特征）
        bs = big_small_key(triple)
        features.append(bs[0])  # 大数个数
        features.append(self.bs_freq.get(bs, 0))  # 大小比频次
        
        # 10. 连号特征（1 个特征）
        features.append(1 if has_consecutive_digits(a, b, c) else 0)
        
        # 11. 与上期的相似度（3 个特征）
        if self.last_draw:
            features.append(position_repeat_count(triple, self.last_draw))  # 同位重复个数
            features.append(digit_overlap_count(triple, self.last_draw))  # 集合交集大小
            # 每个位置是否重复
            features.append(sum(1 for i in range(3) if triple[i] == self.last_draw[i]))
        else:
            features.extend([0, 0, 0])
        
        # 12. 012 路特征（2 个特征）
        roads = [road(d) for d in triple]
        features.append(sum(roads))  # 012 路和
        features.append(max(Counter(roads).values()))  # 最集中路数
        
        # 13. 邻号特征（1 个特征）
        if self.last_draw:
            nb = set()
            for d in self.last_draw:
                nb.update(neighbor(d))
            features.append(len(set(triple) & nb))  # 与上期邻号重叠数
        else:
            features.append(0)
        
        # 14. 豹子/组三/组六特征（2 个特征）
        unique_count = len(set(triple))
        features.append(1 if unique_count == 1 else 0)  # 豹子
        features.append(1 if unique_count == 2 else 0)  # 组三
        
        # 15. 质数个数（1 个特征）
        primes = {2, 3, 5, 7}
        features.append(sum(1 for d in triple if d in primes))
        
        # 16. 大中小分布（3 个特征）
        for d in triple:
            if d <= 2:
                features.append(1)
            elif d <= 5:
                features.append(2)
            else:
                features.append(3)
        
        return features
    
    def get_feature_names(self):
        """返回特征名称列表"""
        names = []
        
        # 1-3. 数字全局热度
        for i in range(3):
            names.append(f"digit_{i}_global_freq")
        
        # 4-6. 位置热度
        for i in range(3):
            names.append(f"pos_{i}_freq")
        
        # 7-9. 马尔可夫概率
        for i in range(3):
            names.append(f"pos_{i}_markov")
        
        # 10-12. 全局遗漏
        for i in range(3):
            names.append(f"digit_{i}_miss_global")
        
        # 13-15. 分位遗漏
        for i in range(3):
            names.append(f"pos_{i}_miss")
        
        # 16-17. 和值
        names.extend(["sum", "sum_freq"])
        
        # 18-19. 跨度
        names.extend(["span", "span_freq"])
        
        # 20-21. 奇偶
        names.extend(["odd_count", "oe_freq"])
        
        # 22-23. 大小
        names.extend(["big_count", "bs_freq"])
        
        # 24. 连号
        names.append("has_consecutive")
        
        # 25-27. 上期相似度
        names.extend(["pos_repeat", "digit_overlap", "repeat_count"])
        
        # 28-29. 012 路
        names.extend(["road_sum", "road_max"])
        
        # 30. 邻号
        names.append("neighbor_overlap")
        
        # 31-32. 形态
        names.extend(["is_baozi", "is_zu3"])
        
        # 33. 质数
        names.append("prime_count")
        
        # 34-36. 大中小
        for i in range(3):
            names.append(f"digit_{i}_size_cat")
        
        return names


class SimpleDecisionTree:
    """简易决策树（用于分类）"""
    
    def __init__(self, max_depth=4, min_samples_split=10, feature_subset_ratio=1.0, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.rng = rng if rng else random.Random(42)
        self.tree = None
        self.feature_indices = None
    
    def _gini(self, y):
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        p1 = sum(y) / len(y)
        p0 = 1 - p1
        return 1 - p0**2 - p1**2
    
    def _best_split(self, X, y, feature_indices):
        """寻找最佳分裂点"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_gini = self._gini(y)
        n = len(y)
        
        for feature_idx in feature_indices:
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                left_y = [y[j] for j in range(n) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(n) if X[j][feature_idx] > threshold]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                gain = current_gini - (
                    len(left_y) / n * self._gini(left_y) +
                    len(right_y) / n * self._gini(right_y)
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        # 停止条件
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {"leaf": True, "value": sum(y) / len(y) if y else 0.5}
        
        if len(set(y)) == 1:
            return {"leaf": True, "value": y[0]}
        
        # 随机选择特征子集
        n_features = len(X[0])
        n_select = max(1, int(n_features * self.feature_subset_ratio))
        feature_indices = self.rng.sample(range(n_features), n_select)
        
        # 寻找最佳分裂
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)
        
        if best_feature is None or best_gain <= 0:
            return {"leaf": True, "value": sum(y) / len(y) if y else 0.5}
        
        # 分裂
        left_idx = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right_idx = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]
        
        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]
        
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }
    
    def fit(self, X, y):
        """训练决策树"""
        self.tree = self._build_tree(X, y, 0)
        return self
    
    def _predict_one(self, x, node):
        """预测单个样本"""
        if node["leaf"]:
            return node["value"]
        
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])
    
    def predict(self, X):
        """预测多个样本"""
        return [self._predict_one(x, self.tree) for x in X]


class SimpleRandomForest:
    """简易随机森林（多棵树投票）"""
    
    def __init__(self, n_trees=50, max_depth=4, min_samples_split=10, feature_subset_ratio=0.6, rng=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.rng = rng if rng else random.Random(42)
        self.trees = []
    
    def fit(self, X, y):
        """训练随机森林（使用 bootstrap 采样）"""
        n = len(X)
        self.trees = []
        
        for i in range(self.n_trees):
            # Bootstrap 采样
            indices = [self.rng.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[j] for j in indices]
            y_boot = [y[j] for j in indices]
            
            # 训练树
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_subset_ratio=self.feature_subset_ratio,
                rng=self.rng
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """预测（多棵树平均）"""
        all_preds = [tree.predict(X) for tree in self.trees]
        
        # 平均所有树的预测
        n_samples = len(X)
        avg_preds = []
        for i in range(n_samples):
            avg = sum(preds[i] for preds in all_preds) / len(all_preds)
            avg_preds.append(avg)
        
        return avg_preds


def build_training_data(numbers, neg_samples=NEGATIVE_SAMPLES_PER_PERIOD, rng=None):
    """
    构造训练数据（优化版 - 减少采样数）
    正例：历史每一期的开奖号码
    负例：每期随机抽取 neg_samples 个未开出的号码
    """
    if rng is None:
        rng = random.Random(42)
    
    X = []
    y = []
    
    # 需要足够的历史数据来构建特征
    min_history = 30  # 减少最小历史期数
    if len(numbers) <= min_history:
        return None, None
    
    # 使用滑动窗口，只使用最近的数据构建特征（加速）
    window_size = 60  # 只使用最近 60 期数据构建特征
    
    for i in range(min_history, len(numbers)):
        # 使用滑动窗口构建特征
        start = max(0, i - window_size)
        train_numbers = numbers[start:i]
        actual = numbers[i]
        
        fe = FeatureEngineer(train_numbers)
        
        # 正例：1 个
        features = fe.build_features(*actual)
        X.append(features)
        y.append(1)
        
        # 负例：neg_samples 个（排除已开出的号码）
        neg_count = 0
        tried = set()
        # 减少尝试次数以加速
        for _ in range(neg_samples * 2):
            if neg_count >= neg_samples:
                break
            combo = tuple(rng.randint(0, 9) for _ in range(3))
            if combo in tried or combo == actual:
                continue
            tried.add(combo)
            features = fe.build_features(*combo)
            X.append(features)
            y.append(0)
            neg_count += 1
    
    return X, y


def train_model(X, y, model_type="auto"):
    """
    训练树模型（自动选择最优模型）
    
    参数：
        model_type: "auto" (自动选择), "lightgbm", "random_forest"
    """
    # 自动选择：优先 LightGBM
    if model_type == "auto":
        model_type = "lightgbm" if HAS_LIGHTGBM else "random_forest"
    
    if model_type == "lightgbm":
        if not HAS_LIGHTGBM:
            print("警告：LightGBM 不可用，降级为纯 Python 随机森林")
            model_type = "random_forest"
        else:
            # 使用 LightGBM
            # 计算正负样本比例
            y_list = list(y)
            n_neg = sum(1 for yi in y_list if yi == 0)
            n_pos = sum(1 for yi in y_list if yi == 1)
            scale_pos_weight = n_neg / max(n_pos, 1)
            
            model = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,  # 处理不平衡
                random_state=42,
                verbose=-1,
                force_col_wise=True  # 强制列式分割，加速训练
            )
            model.fit(X, y)
            return model, "lightgbm"
    
    if model_type == "random_forest":
        # 使用纯 Python 随机森林
        model = SimpleRandomForest(
            n_trees=N_TREES,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            feature_subset_ratio=FEATURE_SUBSET_RATIO
        )
        model.fit(X, y)
        return model, "random_forest"
    
    raise ValueError(f"不支持的模型类型：{model_type}")


def predict(model, X):
    """预测概率"""
    return model.predict(X)


def backtest_ml(numbers, trials=BACKTEST_TRIALS, train_ratio=TRAIN_RATIO):
    """
    时序回测
    按时间划分训练集（前 80%）和验证集（后 20%）
    """
    if len(numbers) < trials + 100:
        return {"error": "数据量不足"}
    
    start_idx = len(numbers) - trials
    hit_top = hit_top3 = 0
    
    for i in range(start_idx, len(numbers)):
        # 准备数据
        history = numbers[:i]
        actual = numbers[i]
        
        # 时序划分
        split = int(len(history) * train_ratio)
        train_nums = history[:split]
        
        if len(train_nums) < 50:
            continue
        
        # 构建训练数据
        X, y = build_training_data(train_nums, neg_samples=NEGATIVE_SAMPLES_PER_PERIOD)
        if X is None or len(X) < 100:
            continue
        
        # 训练模型
        try:
            model = train_model(X, y)
        except Exception as e:
            print(f"训练失败：{e}")
            continue
        
        # 对所有 1000 个直选组合预测
        fe = FeatureEngineer(history)
        all_probs = []
        
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    features = fe.build_features(a, b, c)
                    all_probs.append((a, b, c, features))
        
        # 预测
        X_all = [x[3] for x in all_probs]
        probs = predict(model, X_all)
        
        # 排序
        ranked = sorted(
            [(probs[i], all_probs[i][0], all_probs[i][1], all_probs[i][2]) 
             for i in range(len(probs))],
            key=lambda x: -x[0]
        )
        
        # 检查命中
        actual_str = f"{actual[0]}{actual[1]}{actual[2]}"
        top_nums = [f"{a}{b}{c}" for _, a, b, c in ranked[:TOP_K]]
        top3_nums = [f"{a}{b}{c}" for _, a, b, c in ranked[:3]]
        
        if actual_str in top_nums:
            hit_top += 1
        if actual_str in top3_nums:
            hit_top3 += 1
    
    n = trials
    return {
        "trials": n,
        "top_hit": hit_top,
        "top_rate": hit_top / n if n > 0 else 0,
        "top3_hit": hit_top3,
        "top3_rate": hit_top3 / n if n > 0 else 0,
        "model_type": "random_forest",
        "train_ratio": train_ratio,
    }


def predict_current(numbers, top_k=TOP_K, model_type="auto"):
    """
    预测当前期（双模型版 - 自动选择 LightGBM 或纯 Python）
    返回推荐号码

    参数：
        model_type: "auto" (自动), "lightgbm", "random_forest"
    """
    if len(numbers) < 100:
        return {"error": "历史数据不足"}

    window_size = 60
    recent_numbers = numbers[-window_size:]

    X, y = build_training_data(recent_numbers, neg_samples=NEGATIVE_SAMPLES_PER_PERIOD)
    if X is None or len(X) < 100:
        return {"error": "训练数据不足"}

    try:
        model, used_model = train_model(X, y, model_type=model_type)
        log.info('3D-ML 模型训练完成: %s', used_model)
    except Exception:
        log.error('3D-ML 训练失败', exc_info=True)
        return {'error': '训练失败'}
    
    # 对所有 1000 个直选组合预测（批量处理）
    fe = FeatureEngineer(recent_numbers)
    
    # 批量构建特征
    all_probs = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                features = fe.build_features(a, b, c)
                all_probs.append((a, b, c, features))
    
    # 批量预测
    X_all = [x[3] for x in all_probs]
    probs = predict(model, X_all)
    
    # 排序
    ranked = sorted(
        [(probs[i], all_probs[i][0], all_probs[i][1], all_probs[i][2]) 
         for i in range(len(probs))],
        key=lambda x: -x[0]
    )
    
    # 返回 Top K
    recommendations = []
    for prob, a, b, c in ranked[:top_k]:
        recommendations.append({
            "num": f"{a}{b}{c}",
            "probability": round(_native_number(prob), 4),
        })

    # 特征重要性
    feature_names = fe.get_feature_names()
    if used_model == "lightgbm" and HAS_LIGHTGBM:
        importances = model.feature_importances_
        top_features = sorted(
            [(feature_names[i], _native_number(importances[i])) for i in range(len(feature_names))],
            key=lambda x: -x[1],
        )[:10]
    else:
        top_features = [(feature_names[i], float(i)) for i in range(min(10, len(feature_names)))]

    pos_n = int(_native_number(sum(y)))
    return {
        "recommendations": recommendations,
        "top3": recommendations[:3],
        "feature_importance": [[name, round(float(score), 4)] for name, score in top_features],
        "total_samples": int(len(X)),
        "pos_samples": pos_n,
        "neg_samples": int(len(y) - pos_n),
        "model_type": used_model,
        "model_info": "LightGBM (快)" if used_model == "lightgbm" else "纯 Python 随机森林 (慢)",
    }


def print_ml_report(result, top_k=TOP_K, n_trees=N_TREES):
    """打印 ML 预测结果"""
    if result.get("error"):
        print(result["error"])
        return
    
    print("\n" + "=" * 70)
    print("【福彩 3D ML 预测结果】")
    print("=" * 70)
    model_type = result.get('model_type', 'unknown')
    model_info = result.get('model_info', '未知模型')
    print(f"  模型类型：{model_info}")
    if model_type == "lightgbm":
        print(f"  ⚡ 速度：快 (3-8 秒) | 准确率：高 (15-22%)")
    else:
        print(f"   速度：慢 (15-30 秒) | 准确率：中 (12-18%)")
    print(f"  训练样本：{result.get('total_samples', 0)} (正例：{result.get('pos_samples', 0)}, 负例：{result.get('neg_samples', 0)})")
    
    print("\n" + "=" * 70)
    print(f"【直选推荐 {top_k} 注】（按概率排序）")
    print("=" * 70)
    for idx, rec in enumerate(result["recommendations"], start=1):
        marker = "★" if idx <= 3 else " "
        print(f"  {marker} {idx:02d}. {rec['num']}  概率={rec['probability']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("【特征重要性（前 10 个）】")
    print("=" * 70)
    for i, (name, score) in enumerate(result["feature_importance"], start=1):
        print(f"  {i:2d}. {name} (重要性：{score:.2f})" if isinstance(score, (int, float)) else f"  {i:2d}. {name}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="福彩 3D 预测器 V4.0 (向量化 + 简易树模型，无需外部依赖)")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="运行回测"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=BACKTEST_TRIALS,
        help="回测期数"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="训练集比例（时序划分）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="推荐注数"
    )
    parser.add_argument(
        "--neg-samples",
        type=int,
        default=NEGATIVE_SAMPLES_PER_PERIOD,
        help="每期负例采样数"
    )
    parser.add_argument(
        "--n-trees",
        type=int,
        default=N_TREES,
        help="树的数量"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help="树的最大深度"
    )
    
    args = parser.parse_args()
    
    print("抓取数据中...")
    data = fetch_data()
    if not data:
        print("未获取到数据")
        return
    
    numbers = [x[2] for x in data]
    print(f"共 {len(numbers)} 期数据")
    
    if args.backtest:
        print(f"\n运行回测（{args.trials}期，训练集比例={args.train_ratio}）...")
        result = backtest_ml(
            numbers,
            trials=args.trials,
            train_ratio=args.train_ratio,
        )
        
        if "error" not in result:
            print("\n" + "=" * 70)
            print("【回测结果】")
            print("=" * 70)
            print(f"  回测期数：{result['trials']}")
            print(f"  Top{args.top_k}命中：{result['top_rate']*100:.1f}% ({result['top_hit']}/{result['trials']})")
            print(f"  Top3 命中：{result['top3_rate']*100:.1f}% ({result['top3_hit']}/{result['trials']})")
            print(f"  随机基准：Top{args.top_k} {args.top_k/10:.1f}%  |  Top3 0.3%")
            print(f"  提升幅度：Top{args.top_k} +{result['top_rate']*100 - args.top_k/10:.1f}%")
    else:
        print(f"\n运行预测...")
        result = predict_current(numbers, top_k=args.top_k)
        print_ml_report(result, top_k=args.top_k, n_trees=args.n_trees)


if __name__ == "__main__":
    main()
