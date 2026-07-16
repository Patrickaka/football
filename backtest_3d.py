"""
福彩3D 全面回测与自动调优脚本
=============================
目标：量化当前命中率基线，然后系统性调优所有可调参数，
     在保证性能（不超时）的前提下最大化 Top3/Top30 命中率。

回测策略：
  - 滚动窗口：用前 N 期数据预测第 N+1 期
  - 测量指标：Top1/3/10/30 直选命中率、组选(组三+组六)命中率、≥2码命中
  - 对比基线：随机选号的理论期望 (N/1000)
  
调优维度：
  1) W_* 权重常数（网格搜索 / 坐标下降）
  2) DIST_CALIBRATION_LAM 强度
  3) FEATURE_FLAGS 开关组合
  4) 分布校准迭代次数
"""

import json
import math
import os
import sys
import time
from collections import Counter
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.lottery3d as L

# ================================================================
# 数据加载
# ================================================================
REAL_RAW = "data/_raw_3d_2000.json"

def load_numbers():
    rows = json.load(open(REAL_RAW, "r", encoding="utf-8"))
    nums = [tuple(int(d) for d in r[2:5]) for r in rows]
    return nums


def is_zu6(t):
    return len(set(t)) == 3

def is_zu3(t):
    return len(set(t)) == 2

def is_baozi(t):
    return len(set(t)) == 1


# ================================================================
# 复刻 run_prediction 管线（无网络、纯本地）
# ================================================================
MIN_PERIODS = 100  # 至少需要这么多期才能跑预测

def build_pipeline(numbers, period_idx):
    """复刻 run_prediction 的评分管线，返回 (score, meta, danma, kill, numbers_so_far)
    
    用 numbers[0:period_idx+1] 的数据来预测 numbers[period_idx]（即下一期）
    实际上我们用 numbers[:period_idx] 来预测 numbers[period_idx]
    """
    hist = list(numbers[:period_idx])
    if len(hist) < MIN_PERIODS:
        return None
    
    periods = [str(i) for i in range(len(hist))]  # fake periods
    sums = [sum(x) for x in hist]
    spans = [max(x) - min(x) for x in hist]

    try:
        ww, _ = L.resolve_window_weights(hist, compute_weights=False, period=periods[-1])
    except Exception:
        return None

    meta = L.build_ranking_meta(hist, ww, sums, spans, tail_top=5)
    score, freq_all = L.ensemble_digit_scores(hist, ww, dynamic=meta.get("dynamic"))
    ds = L._blend_dan_score(score, meta)
    danma, tuoma, kill, rank = L.pick_dan_tuo_kill(ds, enable_danma_random=False)

    # 把真实历史注入 meta，供分布校准使用
    meta["numbers"] = hist

    return {
        "score": score,
        "meta": meta,
        "danma": danma,
        "kill": kill,
        "hist": hist,
    }


def predict_topk(pipe, top_k=30):
    """从管线预测 TopK 结果"""
    result = L.rank_triplets(
        pipe["score"], pipe["danma"], pipe["kill"], pipe["meta"],
        top_n=top_k,
        enable_exploration=False,
        apply_noise=False,
        enable_cold_hot_balance=False,
        enable_diversity=False,
        enable_correlation=False,
        recent_recommendations=None,
    )
    return [(w, t) for w, t in result]


# ================================================================
# 回测引擎
# ================================================================
def run_backtest(numbers, start_idx=200, end_idx=-50, step=1, top_ks=[3, 10, 30]):
    """
    滚动回测：
      对于每个 period_idx in range(start_idx, end_idx, step):
        用 numbers[:period_idx] 构建 pipeline → 预测 → 与 numbers[period_idx] 比较
    返回统计结果。
    
    end_idx=-50 表示保留最近50期不参与回测（避免过拟合到最新走势）
    """
    stats = {k: {"hits": 0, "total": 0, "zu6_hits": 0, "zu3_hits": 0, "ge2_hits": 0}
             for k in top_ks}
    actual_end = len(numbers) + end_idx
    t0 = time.time()
    count = 0

    for idx in range(start_idx, actual_end, step):
        target = numbers[idx]
        pipe = build_pipeline(numbers, idx)
        if pipe is None:
            continue

        preds = predict_topk(pipe, top_k=max(top_ks))
        pred_set = set(t for _, t in preds)

        count += 1
        target_str = "".join(map(str, target))
        target_set = frozenset(target)

        for k in top_ks:
            topk = preds[:k]
            topk_set = set(t for _, t in topk)
            s = stats[k]
            s["total"] += 1
            # 直选命中
            if any(t == target_str for t in topk_set):
                s["hits"] += 1
            # 组六命中
            if is_zu6(target) and any(frozenset(t) == target_set for t in topk_set):
                s["zu6_hits"] += 1
            # 组三命中
            if is_zu3(target) and any(frozenset(t) == target_set for t in topk_set):
                s["zu3_hits"] += 1
            # >=2 码命中
            overlap = len(set(target_str) & set("".join(t for t in topk_set if len(set(t)) == 3)[:3]))
            # 更精确：检查每个推荐与目标的数字重叠
            max_overlap = 0
            for t in topk_set:
                o = len(set(str(target[0]) + str(target[1]) + str(target[2])) & set(t))
                if o > max_overlap:
                    max_overlap = o
            if max_overlap >= 2:
                s["ge2_hits"] += 1

        if count % 100 == 0:
            elapsed = time.time() - t0
            print(f"  已回测 {count} 期 (idx={idx}/{actual_end}), 耗时 {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  回测完成: {count} 期, 总耗时 {elapsed:.1f}s ({elapsed/count:.2f}s/期)")

    results = {}
    for k in top_ks:
        s = stats[k]
        n = s["total"]
        results[f"top{k}"] = {
            "直选命中率": s["hits"] / n if n else 0,
            "直选命中次数": s["hits"],
            "总期数": n,
            "组六命中": s["zu6_hits"],
            "组三命中": s["zu3_hits"],
            ">=2码命中": s["ge2_hits"],
            ">=2码率": s["ge2_hits"] / n if n else 0,
        }
    results["回测期数"] = count
    results["耗时秒"] = round(elapsed, 1)
    return results


def print_backtest_results(results, label=""):
    """打印回测结果"""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    total_periods = results.get("回测期数", 0)
    print(f"  回测期数: {total_periods}")
    
    keys = [k for k in results.keys() if k.startswith("top")]
    for k in sorted(keys, key=lambda x: int(x.replace("top",""))):
        v = results[k]
        n = v["总期数"]
        hit = v["直选命中率"]
        expected = int(k.replace("top","")) / 1000.0
        lift = (hit / expected - 1) * 100 if expected > 0 else 0
        print(f"  {k:>5s}: 直选={hit*100:.2f}% ({v['直选命中次数']}/{n}) "
              f"期望={expected*100:.1f}% 提升={lift:+.1f}% | "
              f">=2码={v.get('>=2码率',0)*100:.1f}%")
    return results


# ================================================================
# 主流程
# ================================================================
def main():
    print("=" * 70)
    print("  福彩3D 全面回测与调优")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/4] 加载历史数据...")
    numbers = load_numbers()
    print(f"  共 {len(numbers)} 期 ({numbers[0]} ~ {numbers[-1]})")

    # 2. 当前模型基线回测
    print(f"\n[2/4] 当前模型基线回测 (start=250, end=-50, step=5)...")
    print(f"  当前配置:")
    print(f"    DIST_CALIBRATION_ENABLED = {L.DIST_CALIBRATION_ENABLED}")
    print(f"    DIST_CALIBRATION_LAM = {L.DIST_CALIBRATION_LAM}")
    print(f"    FEATURE_FLAGS = {json.dumps(L.FEATURE_FLAGS, ensure_ascii=False)}")
    
    baseline = run_backtest(numbers, start_idx=250, end_idx=-50, step=5, top_ks=[3, 10, 30])
    print_backtest_results(baseline, label="【改造前】当前模型基线")

    # 保存基线
    baseline_json = {
        "label": "baseline_current",
        "config": {
            "DIST_CALIBRATION_LAM": L.DIST_CALIBRATION_LAM,
            "FEATURE_FLAGS": dict(L.FEATURE_FLAGS),
        },
        "results": baseline,
    }
    open("data/bt_baseline.json", "w", encoding="utf-8").write(
        json.dumps(baseline_json, ensure_ascii=False, indent=2)
    )

    # 3. 参数扫描：Lam 强度
    print(f"\n[3/4] Lam 强度扫描 (0.0 ~ 1.2) ...")
    lam_results = {}
    for lam_val in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]:
        old_lam = L.DIST_CALIBRATION_LAM
        old_en = L.DIST_CALIBRATION_ENABLED
        
        L.DIST_CALIBRATION_LAM = lam_val
        L.DIST_CALIBRATION_ENABLED = lam_val > 0
        
        res = run_backtest(numbers, start_idx=250, end_idx=-50, step=5, top_ks=[3, 30])
        
        lam_results[lam_val] = {
            "top3_hit": res.get("top3", {}).get("直选命中率", 0),
            "top30_hit": res.get("top30", {}).get("直选命中率", 0),
            "top30_ge2": res.get("top30", {}).get(">=2码率", 0),
        }
        
        print(f"  lam={lam_val}: Top3={res.get('top3',{}).get('直选命中率',0)*100:.2f}% "
              f"Top30={res.get('top30',{}).get('直选命中率',0)*100:.2f}% "
              f">=2码={res.get('top30',{}).get('>=2码率',0)*100:.1f}%")
        
        # 恢复
        L.DIST_CALIBRATION_LAM = old_lam
        L.DIST_CALIBRATION_ENABLED = old_en

    best_lam = max(lam_results.keys(), key=lambda k: lam_results[k]["top30_hit"])
    print(f"\n  最优 lam = {best_lam} (Top30命中率最高)")
    
    # 4. 应用最优 Lam 并做完整回测
    print(f"\n[4/4] 最终验证 (lam={best_lam}, 全量回测 step=2)...")
    
    L.DIST_CALIBRATION_LAM = best_lam
    L.DIST_CALIBRATION_ENABLED = best_lam > 0
    
    final = run_backtest(numbers, start_idx=250, end_idx=-50, step=2, top_ks=[3, 10, 30])
    print_backtest_results(final, label=f"【改造后】lam={best_lam}")

    # 汇总对比
    print(f"\n{'='*70}")
    print(f"  改造前后对比汇总")
    print(f"{'='*70}")
    for k in ["top3", "top10", "top30"]:
        b = baseline.get(k, {}).get("直选命中率", 0)
        f_ = final.get(k, {}).get("直选命中率", 0)
        print(f"  {k:>5s}: {b*100:.2f}% -> {f_*100:.2f}% ({(f_-b)*100:+.2f}%)")

    # 保存结果
    summary = {
        "baseline": baseline,
        "final": final,
        "best_lam": best_lam,
        "lam_scan": lam_results,
    }
    out_path = "data/bt_optimization.json"
    open(out_path, "w", encoding="utf-8").write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n  结果已保存: {out_path}")

    return summary


if __name__ == "__main__":
    main()
