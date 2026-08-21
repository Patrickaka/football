"""
福彩3D 最终验证回测
====================
对比「改造前」(原始参数) vs 「改造后」(当前代码全部修改) 的命中率差异。
使用 step=2 密集采样，确保统计显著性。
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.lottery3d as L

REAL_RAW = "data/_raw_3d_2000.json"
MIN_PERIODS = 100


def load_numbers():
    rows = json.load(open(REAL_RAW, "r", encoding="utf-8"))
    return [tuple(int(d) for d in r[2:5]) for r in rows]


def build_pipeline(numbers, period_idx):
    hist = list(numbers[:period_idx])
    if len(hist) < MIN_PERIODS:
        return None
    periods = [str(i) for i in range(len(hist))]
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
    meta["numbers"] = hist
    return {"score": score, "meta": meta, "danma": danma, "kill": kill, "hist": hist}


def predict_topk(pipe, top_k=30):
    result = L.rank_triplets(
        pipe["score"], pipe["danma"], pipe["kill"], pipe["meta"],
        top_n=top_k,
        enable_exploration=False, apply_noise=False,
        enable_cold_hot_balance=False, enable_diversity=False,
        enable_correlation=False, recent_recommendations=None,
    )
    return [(w, t) for w, t in result]


def full_backtest(numbers, start_idx=250, end_idx=-50, step=2, label=""):
    """完整回测：返回 Top3/10/30 的详细统计"""
    actual_end = len(numbers) + end_idx
    stats = {
        3: {"h": 0, "t": 0},
        10: {"h": 0, "t": 0},
        30: {"h": 0, "t": 0},
        "ge2_30": {"h": 0, "t": 0},
        "zu6_30": {"h": 0, "t": 0},   # 组六命中
        "zu3_30": {"h": 0, "t": 0},   # 组三命中
    }
    
    t0 = time.time()
    count = 0
    
    for idx in range(start_idx, actual_end, step):
        target = numbers[idx]
        target_str = "".join(map(str, target))
        pipe = build_pipeline(numbers, idx)
        if pipe is None:
            continue
        
        preds = predict_topk(pipe, top_k=30)
        pred_set = set(t for _, t in preds)
        count += 1

        target_set = frozenset(target)
        is_zu6 = len(target_set) == 3
        is_zu3 = len(target_set) == 2

        for k in [3, 10, 30]:
            topk_set = set(t for _, t in preds[:k])
            if any(t == target_str for t in topk_set):
                stats[k]["h"] += 1
            stats[k]["t"] += 1

        # >=2码命中
        max_ov = 0
        for t in pred_set:
            o = len(set(target_str) & set(t))
            if o > max_ov:
                max_ov = o
        if max_ov >= 2:
            stats["ge2_30"]["h"] += 1
        stats["ge2_30"]["t"] += 1

        # 组六/组三命中
        if is_zu6 and any(frozenset(t) == target_set for t in pred_set):
            stats["zu6_30"]["h"] += 1
        stats["zu6_30"]["t"] += 1
        if is_zu3 and any(frozenset(t) == target_set for t in pred_set):
            stats["zu3_30"]["h"] += 1
        stats["zu3_30"]["t"] += 1

        if count % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {label} 进度: {count} 期 ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    results = {
        "label": label,
        "total_periods": count,
        "elapsed_s": round(elapsed, 1),
    }
    
    for k in [3, 10, 30]:
        n = stats[k]["t"]
        hr = stats[k]["h"] / n if n else 0
        exp = k / 1000.0
        lift = (hr / exp - 1) * 100 if exp > 0 else 0
        results[f"top{k}_hit_rate"] = hr
        results[f"top{k}_hits"] = stats[k]["h"]
        results[f"top{k}_lift_pct"] = round(lift, 1)

    ge2_n = stats["ge2_30"]["t"]
    results["ge2_rate"] = stats["ge2_30"]["h"] / ge2_n if ge2_n else 0
    
    zu6_n = stats["zu6_30"]["t"]
    results["zu6_hit_rate"] = stats["zu6_30"]["h"] / zu6_n if zu6_n else 0
    
    zu3_n = stats["zu3_30"]["t"]
    results["zu3_hit_rate"] = stats["zu3_30"]["h"] / zu3_n if zu3_n else 0

    return results


def print_comparison(before, after):
    """打印前后对比"""
    print(f"\n{'='*70}")
    print(f"  福彩3D 改造前后对比（{after['total_periods']}期回测, step=2）")
    print(f"{'='*70}")
    
    print(f"\n  {'指标':<18s} {'改造前':>10s} {'改造后':>10s} {'变化':>10s}")
    print(f"  {'-'*50}")
    
    for k in ["top3", "top10", "top30"]:
        b = before.get(f"{k}_hit_rate", 0) * 100
        a = after.get(f"{k}_hit_rate", 0) * 100
        delta = a - b
        lift_b = before.get(f"{k}_lift_pct", 0)
        lift_a = after.get(f"{k}_lift_pct", 0)
        arrow = "✅" if delta > 0 else ("⚠️" if abs(delta) < 0.1 else "❌")
        print(f"  {k.upper():<18s} {b:>9.2f}% {a:>9.2f}% {delta:>+9.2f}% {arrow}"
              f"  (Lift: {lift_b:.0f}%→{lift_a:.0f}%)")
    
    b_ge2 = before.get("ge2_rate", 0) * 100
    a_ge2 = after.get("ge2_rate", 0) * 100
    print(f"  {'>=2码命中':<18s} {b_ge2:>9.1f}% {a_ge2:>9.1f}% {a_ge2-b_ge2:>+9.1f}%")
    
    b_zu6 = before.get("zu6_hit_rate", 0) * 100
    a_zu6 = after.get("zu6_hit_rate", 0) * 100
    print(f"  {'组六覆盖':<18s} {b_zu6:>9.1f}% {a_zu6:>9.1f}% {a_zu6-b_zu6:>+9.1f}%")
    
    print(f"\n  耗时: 改造前={before.get('elapsed_s',0):.1f}s | 改造后={after.get('elapsed_s',0):.1f}s")
    
    # 汇总提升
    top30_delta = (after.get("top30_hit_rate", 0) - before.get("top30_hit_rate", 0)) * 100
    print(f"\n  ★ Top30 命中率总提升: {top30_delta:+.2f}% "
          f"(相对提升 {(after.get('top30_hit_rate',0)/before.get('top30_hit_rate',1)-1)*100:+.1f}%)")


def main():
    print("=" * 70)
    print("  福彩3D 最终验证回测")
    print("=" * 70)

    numbers = load_numbers()
    print(f"\n  数据: {len(numbers)} 期")

    # 打印当前配置
    print(f"\n  当前配置:")
    print(f"    DIST_CALIBRATION_LAM = {L.DIST_CALIBRATION_LAM}")
    print(f"    DIST_CALIBRATION_ENABLED = {L.DIST_CALIBRATION_ENABLED}")
    print(f"    DIST_CALIBRATION_IPF_ITERS = {L.DIST_CALIBRATION_IPF_ITERS}")
    print(f"    W_CONSECUTIVE = {L.W_CONSECUTIVE}")
    print(f"    SUM_SOFT_SIGMA = {L.SUM_SOFT_SIGMA}")
    print(f"    SPAN_SOFT_SIGMA = {L.SPAN_SOFT_SIGMA}")
    print(f"    W_REALIZED = {L.W_REALIZED}")

    # ===== 改造后回测（当前代码） =====
    print(f"\n  [1/2] 改造后回测 (step=2) ...")
    after = full_backtest(numbers, start_idx=250, end_idx=-50, step=2, label="改造后")
    
    print(f"\n  改造后结果:")
    for k in ["top3", "top10", "top30"]:
        hr = after[f"{k}_hit_rate"]
        lift = after[f"{k}_lift_pct"]
        print(f"    {k.upper()}: {hr*100:.2f}% ({after[f'{k}_hits']}/{after['total_periods']}) Lift={lift:+.0f}%")
    print(f"    >=2码: {after['ge2_rate']*100:.1f}%")
    print(f"    组六: {after['zu6_hit_rate']*100:.1f}%")

    # 保存结果
    out_path = "data/bt_final_validation.json"
    open(out_path, "w", encoding="utf-8").write(json.dumps(after, ensure_ascii=False, indent=2))
    print(f"\n  结果已保存: {out_path}")

    return after


if __name__ == "__main__":
    main()
