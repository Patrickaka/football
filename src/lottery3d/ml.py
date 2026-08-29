# 福彩 3D 预测器 V7.0 - 多模型集成版（CatBoost/XGBoost/LightGBM）
"""基于多模型集成的福彩 3D 预测：旧名字对领域层的适配层。

算法本身在 `src/domain/numeric/lottery3d/` 的三个模块里：
`ml_features`（54 维特征）、`ml_training`（正负采样、时间衰减、按期切分、
集成加权）、`ml_forest`（纯 Python 降级模型）。这里留下的是四样东西：

1. **旧的函数名与签名**——`webapp/lazy_modules.py`、`prediction.py`
   与存量测试都按这些名字导入。
2. **把配置常量喂进领域层**。特征窗口、衰减系数、冷热分档阈值一律由这里传。
3. **三个梯度提升库的调用**。它们是外部依赖，装没装、怎么调参、
   early stopping 怎么传，都是这一层的事。
4. **kv 缓存与 CLI**。读写存储是副作用，不进领域层。

**预测的分数只能用来排序。** 训练集按 1:100 采样，不是真实的 1:999 比例，
所以 `model_score` 不是概率——当成中奖概率看会差一个数量级。线上实测的
平均真实排名是 710，比随机期望的 500 还差。
"""
import random
import sys
import time

from ..common.logger import setup_logger
from ..common import kv_store
from ..common.data_cache import cached_fetch

log = setup_logger('lottery3d_ml')

from src.domain.numeric.lottery3d import backtest as _bt
from src.domain.numeric.lottery3d import ml_features as _features
from src.domain.numeric.lottery3d import ml_forest as _forest
from src.domain.numeric.lottery3d import ml_training as _training

from .config import URL

# 尝试导入机器学习库
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# stdout 的编码由**进程入口**设置（`main.py`），不在库模块顶层做。
# 放这里的代价：pytest 一旦有任何测试导入到本模块，就会把它的捕获流换掉，
# 之后每个用例的 setup/teardown 都报 UnicodeDecodeError——**从那一刻起
# 整套测试全红，而且看不出跟谁有关**。
#
# `try/except` 与 `hasattr` 挡不住这件事：`reconfigure` 在 pytest 下**会成功**，
# 成功本身就是问题。同样的坑 `src/football/config.py` 与
# `src/webapp/settings.py` 各踩过一次。

log.info("3D-ML 后端可用性: catboost=%s xgboost=%s lightgbm=%s sklearn=%s",
         HAS_CATBOOST, HAS_XGBOOST, HAS_LIGHTGBM, HAS_SKLEARN)

# 对「ML 后端不可用→降级」在整个进程里只告警一次，避免每次预测都刷屏。
# **必须配 `global` 才能在函数里改**——迁移前少了那一句，于是读它的那行
# 直接 UnboundLocalError，降级路径从来没有真正跑通过（被上层 except 吞成
# 「训练失败」）。
_ML_FALLBACK_WARNED = False

# ─── 模型参数（v7：更深的树、更大窗口、更丰富特征）───

BACKTEST_TRIALS = 60
NEGATIVE_SAMPLES_PER_PERIOD = 100   # 每期负例采样数
TOP_K = 15                          # 推荐注数
FEATURE_SUBSET_RATIO = 0.85         # 特征选择保留比例
MIN_VARIANCE = 0.0005               # 方差过滤阈值
TRAINING_WINDOW = 200               # 滚动训练样本窗口
FEATURE_HISTORY_WINDOW = 100        # 特征工程历史窗口

# 时间衰减：越近的期权重越高
TIME_DECAY_RECENT = 30
TIME_DECAY_RECENT_WEIGHT = 2.0
TIME_DECAY_MID = 60
TIME_DECAY_MID_WEIGHT = 1.4
TIME_DECAY_OLD_WEIGHT = 1.0

# 前多少期只当历史用，不产出训练样本。60 期以下算不出稳定的冷热与遗漏
MIN_HISTORY_PERIODS = 60
# 样本少于这么多条就不训练：切完验证集两边都太薄
MIN_TRAINING_ROWS = 100
# 预测所需的最少历史
MIN_PERIODS_FOR_PREDICTION = 100

# 指数加权的衰减系数。**与规则模型的 `config.EXP_DECAY` 恰好同值，
# 但不是同一个旋钮**——调那边不该影响这边，所以各留各的
ML_EXP_DECAY = 0.96
ML_MARKOV_ALPHA = 1.0
# 转移概率查不到时的兜底。1/10 即均匀分布，表示「没见过，不偏袒」
ML_FALLBACK_PROB = 0.1
# 冷热分档：达到均值这么多倍算热/温。**注意与规则模型的 1.2/0.8 不同**，
# 两处各调各的，名字却都叫「热号」
ML_HOT_RATIO = 1.3
ML_WARM_RATIO = 0.7
# 和值/跨度趋势的观察窗口
ML_TREND_WINDOW = 20
# 历史不足时的兜底：和值均值 13.5 是 0~27 的中点，跨度均值 4.5 同理
ML_DEFAULT_SUM_MEAN = 13.5
ML_DEFAULT_SPAN_MEAN = 4.5
ML_DEFAULT_DEVIATION = 1.0

# 纯 Python 降级森林的规模。比三个库小得多——它只是保证「装不上库也能出号」
FALLBACK_TREES = 30
FALLBACK_DEPTH = 4
FALLBACK_MIN_SPLIT = 15

# 回测统计的 TopK 门槛
ML_TOP_TIERS = (3, 15, 30, 100)
DEFAULT_MODELS = ("catboost", "xgboost", "lightgbm")
ENSEMBLE = "ensemble"
FALLBACK_MODEL_NAME = "random_forest"
NEUTRAL_SCORE = _training.NEUTRAL_SCORE

MODEL_VERSION = "ml-v7"
BACKTEST_HISTORY_KEY = "lottery3d_ml_backtest_history"
# 保留最近这么多次回测记录
BACKTEST_HISTORY_LIMIT = 30
ML_CACHE_KEY = "lottery3d_ml_prediction_cache"


def _feature_settings(window=FEATURE_HISTORY_WINDOW):
    return _features.FeatureSettings(
        history_window=window,
        decay=ML_EXP_DECAY,
        markov_alpha=ML_MARKOV_ALPHA,
        fallback_prob=ML_FALLBACK_PROB,
        hot_ratio=ML_HOT_RATIO,
        warm_ratio=ML_WARM_RATIO,
        trend_window=ML_TREND_WINDOW,
        default_sum_mean=ML_DEFAULT_SUM_MEAN,
        default_span_mean=ML_DEFAULT_SPAN_MEAN,
        default_deviation=ML_DEFAULT_DEVIATION,
    )


def _sampling_settings(neg_samples):
    return _training.SamplingSettings(
        neg_samples=neg_samples,
        min_history=MIN_HISTORY_PERIODS,
        feature_window=FEATURE_HISTORY_WINDOW,
        decay_tiers=((TIME_DECAY_RECENT, TIME_DECAY_RECENT_WEIGHT),
                     (TIME_DECAY_MID, TIME_DECAY_MID_WEIGHT)),
        base_weight=TIME_DECAY_OLD_WEIGHT,
    )


def FeatureEngineer(numbers, window=FEATURE_HISTORY_WINDOW):  # noqa: N802
    """旧名字。领域层的特征器要一份配置，这里把常量喂进去。"""
    return _features.FeatureEngineer(numbers, _feature_settings(window))


SimpleDecisionTree = _forest.DecisionTree
SimpleRandomForest = _forest.RandomForest
_validation_score = _training.validation_score


def _fetch_data_internal(url=URL):
    """内部数据抓取函数。

    **与 `fetching.py` 的同名函数不是一回事**，三处实质差异：缓存键
    （`lottery3d_ml` vs `lottery3d`）、重试（这里不重试，那边三次带退避）、
    超时（20s vs 30s）。合并要动缓存键与重试语义，得单独做并验证，
    不能夹在 ML 迁移里顺手改。
    """
    import re
    import urllib.request
    log.debug('fetch 3D-ML data')
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(request, timeout=20).read().decode("utf-8", "ignore")
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


def fetch_data(url=URL, force_refresh=False):
    """获取历史开奖数据（带缓存，每天只抓取一次）"""
    return cached_fetch('lottery3d_ml', lambda: _fetch_data_internal(url), force_refresh)


def _native_number(x):
    """numpy 标量 → Python int/float，便于 JSON 序列化。

    **不能写成「依次试 int()、float()」**：那样一个普通的 float 会先被
    int() 截断成整数，小数部分无声消失。
    """
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x)
    return x


def _seeded_rng(numbers):
    """按历史长度派生的随机源。**同一份数据必须得到同一个训练集**，
    否则两次回测的差异里混着采样噪声，比不出模型的好坏。
    """
    return random.Random(len(numbers) * 10007)


def build_training_data(numbers, neg_samples=NEGATIVE_SAMPLES_PER_PERIOD, rng=None):
    """构造训练数据：每期一条正例 + 分层抽样的负例，附时间衰减权重与期号分组。

    **返回值比迁移前多了一层含义**：迁移前数据不足时返回四个 `None`，
    而调用方的守卫写的是 `if result is None or len(result) < 4`——四元组
    既不是 None、长度也正好是 4，那道守卫一次都没拦下过，越过它之后才在
    `len(None)` 上炸掉。现在返回一个空的具名结果，`if not samples` 就够了。
    """
    rng = rng if rng is not None else _seeded_rng(numbers)
    return _training.build_training_samples(
        numbers,
        lambda history: _features.FeatureEngineer(history, _feature_settings()),
        _sampling_settings(neg_samples),
        rng)


def select_features(X, y, feature_names, keep_ratio=FEATURE_SUBSET_RATIO):
    """特征选择：先方差过滤，再按互信息取前若干。

    没装 sklearn 就全都留着——**少几个特征不影响正确性**，
    而为了这一步去装一个重依赖不划算。
    """
    if not HAS_SKLEARN or len(X) < 50:
        return list(range(len(feature_names))), feature_names

    import numpy as np
    rows = np.array(X)
    labels = np.array(y)

    variance = VarianceThreshold(threshold=MIN_VARIANCE)
    filtered = variance.fit_transform(rows)
    mask = variance.get_support()

    keep = int(len(feature_names) * keep_ratio)
    if keep < 5:
        keep = max(5, len(feature_names) // 2)

    try:
        scores = mutual_info_classif(filtered, labels, random_state=42)
        ranked = np.argsort(scores)[::-1][:keep]
        indices = np.where(mask)[0][ranked].tolist()
    except Exception:
        # 互信息算不出来时退回「方差过滤后的前 N 个」。这是降级，
        # 顺序变成了原始特征序而不是重要性序
        indices = np.where(mask)[0].tolist()[:keep]
    return indices, [feature_names[index] for index in indices]


def train_single_model(X, y, model_name, sample_weight=None, eval_set=None):
    """训练单个梯度提升模型。装不上或训练失败都返回 (None, None)。"""
    labels = list(y)
    negatives = sum(1 for label in labels if label == 0)
    positives = sum(1 for label in labels if label == 1)
    scale_pos_weight = negatives / max(positives, 1)

    try:
        if model_name == "catboost" and HAS_CATBOOST:
            model = CatBoostClassifier(
                iterations=100, depth=4, learning_rate=0.03, l2_leaf_reg=15,
                random_strength=1.5, scale_pos_weight=scale_pos_weight,
                random_state=42, verbose=False, task_type="CPU",
                early_stopping_rounds=20)
            if eval_set is not None:
                model.fit(X, y, sample_weight=sample_weight, eval_set=eval_set)
            else:
                model.fit(X, y, sample_weight=sample_weight)
            return model, "catboost"

        if model_name == "xgboost" and HAS_XGBOOST:
            model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.03,
                reg_lambda=15, reg_alpha=5, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight, random_state=42,
                eval_metric='logloss', verbosity=0,
                # early stopping 依赖验证集，无 eval_set 时传进去会直接报错
                early_stopping_rounds=20 if eval_set is not None else None)
            if eval_set is not None:
                # xgboost 要的是 [(X, y)] 列表，调用方给的是单个 (X, y) 元组
                model.fit(X, y, sample_weight=sample_weight, eval_set=[eval_set])
            else:
                model.fit(X, y, sample_weight=sample_weight)
            return model, "xgboost"

        if model_name == "lightgbm" and HAS_LIGHTGBM:
            return _fit_lightgbm(X, y, sample_weight, eval_set, scale_pos_weight)
    except Exception as exc:
        log.warning(f"训练 {model_name} 失败: {exc}")

    return None, None


def _fit_lightgbm(X, y, sample_weight, eval_set, scale_pos_weight):
    """lightgbm 要 numpy 数组。转不了就退回不带验证集训练。"""
    try:
        import numpy as np
        rows = np.array(X) if not isinstance(X, np.ndarray) else X
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        eval_data = None
        if eval_set is not None:
            eval_data = [(np.array(eval_set[0]), np.array(eval_set[1]))]
    except Exception:
        rows, labels, eval_data = X, y, None

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.03,
        reg_lambda=15, reg_alpha=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, random_state=42,
        verbose=-1, force_col_wise=True)
    if eval_data is not None:
        model.fit(rows, labels, sample_weight=sample_weight, eval_set=eval_data)
    else:
        model.fit(rows, labels, sample_weight=sample_weight)
    return model, "lightgbm"


def train_ensemble(X, y, models_to_try=None, sample_weight=None, group_ids=None):
    """训练多模型集成：时序切分 → 训练集选特征 → 验证集评权重 → 全量重训。

    验证得分只用来分配融合权重，不用来挑模型——三个模型学到的东西不一样，
    分数低的那个未必在每一注上都差。
    """
    models_to_try = list(models_to_try or DEFAULT_MODELS)
    feature_names = FeatureEngineer([]).get_feature_names()

    split = (_training.split_by_period(group_ids, len(X)) if group_ids
             else _training.fallback_split(len(X)))
    if split is None:
        train_idx, valid_idx = list(range(len(X))), []
    else:
        train_idx, valid_idx = split

    X_train_raw = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    w_train = [sample_weight[i] for i in train_idx] if sample_weight is not None else None

    selected, _ = select_features(X_train_raw, y_train, feature_names)
    X_train = _project(X_train_raw, selected)
    X_full = _project(X, selected)
    eval_set = None
    if valid_idx:
        eval_set = (_project([X[i] for i in valid_idx], selected),
                    [y[i] for i in valid_idx])

    scores = _probe_scores(X_train, y_train, w_train, eval_set, models_to_try)

    trained = []
    for name in models_to_try:
        model, used = train_single_model(X_full, y, name,
                                         sample_weight=sample_weight, eval_set=eval_set)
        if model:
            score = scores.get(used, NEUTRAL_SCORE)
            trained.append((model, used, score))
            log.info(f"全量重训 {used} 完成，融合权重依据验证得分: {score:.4f}")

    if not trained:
        trained.append(_fallback_model(X_full, y))
    return trained, selected


def _project(rows, indices):
    return [[row[index] for index in indices] for row in rows]


def _probe_scores(X_train, y_train, w_train, eval_set, models_to_try):
    """先在训练段上各训一遍，拿验证段给它们打分。"""
    scores = {}
    for name in models_to_try:
        probe, used = train_single_model(X_train, y_train, name,
                                         sample_weight=w_train, eval_set=eval_set)
        if not probe:
            continue
        scores[used] = _score_probe(probe, eval_set, X_train, y_train)
        log.info(f"验证 {used} 得分: {scores[used]:.4f}")
    return scores


def _score_probe(probe, eval_set, X_train, y_train):
    """有验证集就用验证集打分；没有就退回训练集——**那是虚高的分数**，
    但三个模型都虚高同样的量，用来分配权重仍然可比。
    """
    try:
        if not hasattr(probe, 'predict_proba'):
            return NEUTRAL_SCORE
        rows, labels = (eval_set[0], eval_set[1]) if eval_set else (X_train, y_train)
        return _training.validation_score(labels, probe.predict_proba(rows)[:, 1])
    except Exception:
        return NEUTRAL_SCORE


def _fallback_model(X, y):
    """三个库都不可用时的纯 Python 森林。

    **`global` 那一句是必需的**：迁移前少了它，Python 把 `_ML_FALLBACK_WARNED`
    当成局部变量，读它的那行必然 UnboundLocalError——这条降级路径从来没有
    真正跑通过，只是被上层的 except 吞成了「训练失败」。
    """
    global _ML_FALLBACK_WARNED
    if not _ML_FALLBACK_WARNED:
        log.warning(
            "ML 集成后端不可用（catboost=%s xgboost=%s lightgbm=%s sklearn=%s），"
            "已降级为纯 Python 随机森林；若期望启用 ML 集成，"
            "请用已安装这些库的虚拟环境运行服务。",
            HAS_CATBOOST, HAS_XGBOOST, HAS_LIGHTGBM, HAS_SKLEARN)
        _ML_FALLBACK_WARNED = True

    model = _forest.RandomForest(
        n_trees=FALLBACK_TREES, max_depth=FALLBACK_DEPTH,
        min_samples_split=FALLBACK_MIN_SPLIT, rng=random.Random(42))
    model.fit(X, y)
    return model, FALLBACK_MODEL_NAME, NEUTRAL_SCORE


def predict_single(model, X):
    """单个模型的预测概率。"""
    if hasattr(model, 'predict_proba'):
        try:
            import numpy as np
            rows = np.array(X) if not isinstance(X, np.ndarray) else X
            return model.predict_proba(rows)[:, 1].tolist()
        except Exception:
            pass
    return model.predict(X)


def ensemble_predict(models, X):
    """按验证得分加权融合各模型的预测。"""
    if len(models) == 1:
        return predict_single(models[0][0], X)
    weights = _training.blend_weights([score for _, _, score in models])
    return _training.blend_predictions(
        [predict_single(model, X) for model, _, _ in models], weights)


def _rank_all(models, selected, history):
    """给全部 1000 注打分并排序，返回 ('385', 分数) 的降序列表。"""
    engineer = _features.FeatureEngineer(
        history[-FEATURE_HISTORY_WINDOW:], _feature_settings())
    combos = list(_training.all_combos())
    # 先把 54 维算出来再投影。写成 `build_features(combo)[i] for i in selected`
    # 会为每一个被选中的维度重算一遍整注特征——1000 注 × 45 维 = 四万五千次
    rows = [[row[index] for index in selected]
            for row in (engineer.build_features(combo) for combo in combos)]
    probabilities = ensemble_predict(models, rows)
    ranked = sorted(zip(probabilities, combos), key=lambda pair: -pair[0])
    return [(_bt.as_key(combo), probability) for probability, combo in ranked]


def _train_for(numbers, neg_samples):
    """一次完整训练。数据不够就返回 None，由调用方决定怎么办。"""
    samples = build_training_data(numbers, neg_samples=neg_samples,
                                 rng=_seeded_rng(numbers))
    if not samples or len(samples.rows) < MIN_TRAINING_ROWS:
        return None
    return train_ensemble(samples.rows, samples.labels,
                          sample_weight=samples.weights, group_ids=samples.groups)


def backtest_ml(numbers, trials=BACKTEST_TRIALS, train_window=TRAINING_WINDOW,
                base_period=None, neg_samples=NEGATIVE_SAMPLES_PER_PERIOD):
    """滚动回测：每期用最近 `train_window` 期重训一遍，与实盘逻辑一致。

    **每期都重训**，所以这里很慢——一期一秒上下。做黄金语料时把 `trials`
    压到个位数，别用线上默认值。
    """
    if len(numbers) < trials + train_window:
        return {"error": f"数据量不足，需要至少 {trials + train_window} 期"}

    accumulator = _bt.TopKBacktest(ML_TOP_TIERS)
    for history, actual in _bt.rolling_slices(numbers, trials):
        window = history[-train_window:] if len(history) > train_window else history
        # 与 `build_training_data` 用同一个下限。迁移前这里写的是 `< 60`
        # 而那边是 `<= 60`，差一期就会漏进去，然后在 len(None) 上炸掉
        if len(window) <= MIN_HISTORY_PERIODS:
            continue
        try:
            trained = _train_for(window, neg_samples)
        except Exception as exc:
            log.warning(f"回测期训练失败: {exc}")
            continue
        if trained is None:
            continue
        models, selected = trained
        accumulator.observe(actual, [num for num, _ in _rank_all(models, selected, history)])

    result = accumulator.summarise()
    result.update({
        "model_type": ENSEMBLE,
        "train_window": train_window,
        "base_period": base_period,
        "baseline_top3_rate": _bt.tier_baseline(3),
        "baseline_top30_rate": _bt.tier_baseline(30),
    })
    save_ml_backtest_history(result)
    return result


def save_ml_backtest_history(result):
    """保存 ML 回测结果到历史记录。

    **这是 ML 准入闸门唯一的数据来源**，而写它的只有 CLI 的 `--backtest`：
    web 服务没有任何路径会调到这里。所以线上 `lottery3d_ml_backtest_history`
    至今不存在，`is_ml_eligible_from_backtest` 恒为 False，融合里的 ML 那半
    一直是空的。**这不是死代码，是没被触发过的闸门。**
    """
    try:
        history = kv_store.load(BACKTEST_HISTORY_KEY, [])
        history.append({
            "base_period": result.get("base_period"),
            "model_version": MODEL_VERSION,
            "top30_rate": result["top30_rate"],
            "top3_rate": result["top3_rate"],
            "top100_rate": result["top100_rate"],
            "actual_rank_avg": result["actual_rank_avg"],
            "actual_rank_median": result["actual_rank_median"],
            "trials": result["trials"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        kv_store.save(BACKTEST_HISTORY_KEY, history[-BACKTEST_HISTORY_LIMIT:])
        log.info("ML回测历史已保存")
    except Exception as exc:
        log.error(f"保存ML回测历史失败: {exc}")


def predict_current(numbers, top_k=TOP_K, model_type=ENSEMBLE,
                    neg_samples=NEGATIVE_SAMPLES_PER_PERIOD):
    """预测当前期，返回 Top K 推荐。

    `model_type` 指定单模型时，那个模型训不出来就退回集成。
    """
    if len(numbers) < MIN_PERIODS_FOR_PREDICTION:
        return {"error": "历史数据不足"}

    train_numbers = numbers[-TRAINING_WINDOW:]
    samples = build_training_data(train_numbers, neg_samples=neg_samples,
                                  rng=_seeded_rng(train_numbers))
    if not samples or len(samples.rows) < MIN_TRAINING_ROWS:
        return {"error": "训练数据不足"}

    try:
        models, selected = _train_by_type(samples, model_type)
    except Exception as exc:
        log.error(f'3D-ML 训练失败：{exc}', exc_info=True)
        return {'error': '训练失败'}

    ranked = _rank_all(models, selected, numbers)
    names = [name for _, name, _ in models]
    log.info(f'3D-ML 多模型集成完成：{names}')

    top = ranked[:top_k]
    # TopK 内部的分数占比。**不是概率**——采样比例不是真实比例
    total = sum(probability for _, probability in top) or 1.0
    recommendations = [{
        "num": num,
        "model_score": round(_native_number(probability), 4),
        "topk_score_share": round(_native_number(probability) / total, 4),
    } for num, probability in top]

    positives = int(_native_number(sum(samples.labels)))
    weights = _training.blend_weights([score for _, _, score in models])
    return {
        "recommendations": recommendations,
        "top3": recommendations[:3],
        "feature_importance": _importance(models, selected),
        "total_samples": len(samples.rows),
        "pos_samples": positives,
        "neg_samples": len(samples.labels) - positives,
        "model_type": "+".join(names),
        "model_weights": [round(weight, 4) for weight in weights],
        "model_info": f"多模型集成 ({', '.join(names)})",
        "num_models": len(models),
    }


def _train_by_type(samples, model_type):
    """按请求的模型类型训练。单模型训不出来就退回集成。"""
    if model_type != ENSEMBLE:
        model, used = train_single_model(samples.rows, samples.labels, model_type,
                                         sample_weight=samples.weights)
        if model:
            return [(model, used, 1.0)], list(range(len(samples.rows[0])))
        log.warning(f"{model_type} 不可用，降级为集成模式")
    return train_ensemble(samples.rows, samples.labels,
                          sample_weight=samples.weights, group_ids=samples.groups)


def _importance(models, selected):
    """第一个模型的特征重要性前十。

    名字来自领域层的同一次遍历，与特征值不可能错位——迁移前特征名和特征值
    是两份手工对齐的列表，错位了这张表就会张冠李戴而毫无痕迹。
    """
    # 特征名取一次就够：写在推导式里会为每个下标重建一次特征器
    all_names = FeatureEngineer([]).get_feature_names()
    names = [all_names[index] for index in selected]
    if models and hasattr(models[0][0], 'feature_importances_'):
        scores = models[0][0].feature_importances_
        ranked = sorted(((names[i], _native_number(scores[i]))
                         for i in range(len(names))), key=lambda pair: -pair[1])[:10]
    else:
        # 拿不到重要性时按原顺序给前十，分数用序号占位
        ranked = [(names[i], float(i)) for i in range(min(10, len(names)))]
    return [[name, round(float(score), 4)] for name, score in ranked]


def load_ml_cache():
    """加载 ML 预测缓存"""
    try:
        return kv_store.load(ML_CACHE_KEY, None)
    except Exception as exc:
        log.error(f"加载 ML 缓存失败: {exc}")
        return None


def save_ml_cache(data):
    """保存 ML 预测缓存"""
    try:
        kv_store.save(ML_CACHE_KEY, data)
        log.info("ML 缓存已保存")
    except Exception as exc:
        log.error(f"保存 ML 缓存失败: {exc}")


def clear_ml_cache():
    """清空持久化的 ML 预测缓存"""
    try:
        kv_store.save(ML_CACHE_KEY, None)
        log.info("ML prediction cache cleared")
    except Exception as exc:
        log.error(f"Clear ML prediction cache failed: {exc}")


def print_ml_report(result, top_k=TOP_K):
    """打印 ML 预测结果"""
    if result.get("error"):
        print(result["error"])
        return

    print("\n" + "=" * 70)
    print("【福彩 3D ML 预测结果 V7.0 - 多模型集成版】")
    print("=" * 70)
    print(f"  模型类型：{result.get('model_info', '未知模型')}")
    print(f"  模型数量：{result.get('num_models', 1)} 个")
    weights = result.get('model_weights', [])
    if weights:
        print(f"  模型权重：{', '.join(f'{w:.2f}' for w in weights)}")
    print(f"  训练样本：{result.get('total_samples', 0)} "
          f"(正例：{result.get('pos_samples', 0)}, 负例：{result.get('neg_samples', 0)})")

    print("\n" + "=" * 70)
    print(f"【直选推荐 {top_k} 注】（按模型分排序）")
    print("=" * 70)
    for index, row in enumerate(result["recommendations"], start=1):
        marker = "★" if index <= 3 else " "
        print(f"  {marker} {index:02d}. {row['num']}  模型分={row['model_score']:.4f}")

    print("\n" + "=" * 70)
    print("【特征重要性（前 10 个）】")
    print("=" * 70)
    for index, (name, score) in enumerate(result["feature_importance"], start=1):
        print(f"  {index:2d}. {name} (重要性：{score:.2f})")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="福彩 3D 预测器 V7.0 (多模型集成版：CatBoost/XGBoost/LightGBM)")
    parser.add_argument("--backtest", action="store_true", help="运行回测")
    parser.add_argument("--trials", type=int, default=BACKTEST_TRIALS, help="回测期数")
    parser.add_argument("--train-window", type=int, default=TRAINING_WINDOW,
                        help="滚动训练窗口大小（期数）")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="推荐注数")
    parser.add_argument("--neg-samples", type=int, default=NEGATIVE_SAMPLES_PER_PERIOD,
                        help="每期负例采样数")
    parser.add_argument("--model", type=str, default=ENSEMBLE,
                        choices=[ENSEMBLE, *DEFAULT_MODELS, FALLBACK_MODEL_NAME],
                        help="模型类型")
    args = parser.parse_args()

    print("抓取数据中...")
    data = fetch_data()
    if not data:
        print("未获取到数据")
        return

    numbers = [row[2] for row in data]
    print(f"共 {len(numbers)} 期数据")

    if args.backtest:
        print(f"\n运行回测（{args.trials}期，训练窗口={args.train_window}，"
              f"负例数={args.neg_samples}）...")
        result = backtest_ml(numbers, trials=args.trials, train_window=args.train_window,
                             base_period=data[-1][0], neg_samples=args.neg_samples)
        if result.get("error"):
            print(result["error"])
            return
        print("\n" + "=" * 70)
        print("【回测结果】")
        print("=" * 70)
        print(f"  回测期数：{result['trials']}")
        for tier in ML_TOP_TIERS:
            print(f"  Top{tier} 命中：{result[f'top{tier}_hit']} "
                  f"({result[f'top{tier}_rate'] * 100:.2f}%)")
        print(f"  平均真实排名：{result['actual_rank_avg']}")
        print(f"  中位真实排名：{result['actual_rank_median']}")
    else:
        print("\n运行预测...")
        print_ml_report(
            predict_current(numbers, top_k=args.top_k, model_type=args.model,
                            neg_samples=args.neg_samples),
            top_k=args.top_k)


if __name__ == "__main__":
    main()
