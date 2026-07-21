"""
3D预测模块深度诊断与优化 v3
==============================
发现：
- diversity/correlation 提升 >=2码率 但降低 TopK 精确率
- 单参数调优空间有限（彩票本质接近随机）

新策略：
A. 对比不同配置在不同指标上的表现
B. 寻找 >=2码率和Top30的最优平衡点
C. 测试ML模型融合是否能带来提升
"""

import json
import sys
import time

sys.path.insert(0, '.')
import src.lottery3d as L


def load_numbers():
    data = L.fetch_data()
    return [x[2] for x in data]


def detailed_backtest(numbers, start_idx=250, end_idx=None, step=3, top_k=30,
                     enable_diversity=False, enable_correlation=False,
                     exploration_rate=0.0):
    """详细回测：返回多维度统计"""
    if end_idx is None:
        end_idx = len(numbers) - 50

    stats = {
        3: {"h": 0, "t": 0}, 10: {"h": 0, "t": 0}, 30: {"h": 0, "t": 0},
        "ge2": {"h": 0, "t": 0}, "ge1": {"h": 0, "t": 0},
        "rank_sum": 0, "rank_count": 0,
    }

    old_exp = L.EXPLORATION_RATE
    L.EXPLORATION_RATE = exploration_rate

    for idx in range(start_idx, end_idx, step):
        target = numbers[idx]
        target_str = "".join(map(str, target))
        hist = numbers[:idx]

        try:
            periods = [str(j) for j in range(len(hist))]
            sums = [sum(x) for x in hist]
            spans = [max(x) - min(x) for x in hist]

            ww, _ = L.resolve_window_weights(hist, compute_weights=False, period=periods[-1])
            meta = L.build_ranking_meta(hist, ww, sums, spans)
            score, _ = L.ensemble_digit_scores(hist, ww, dynamic=meta.get("dynamic"))
            danma, _, kill, _ = L.pick_dan_tuo_kill(score, enable_danma_random=False)

            result = L.rank_triplets(
                score, danma, kill, meta, top_n=top_k,
                enable_exploration=(exploration_rate > 0),
                apply_noise=False,
                enable_cold_hot_balance=False,
                enable_diversity=enable_diversity,
                enable_correlation=enable_correlation,
                recent_recommendations=None,
            )

            # 计算真实排名
            all_sorted = sorted(result, key=lambda x: -x[0])
            rank_map = {num: i + 1 for i, (_, num) in enumerate(all_sorted)}
            actual_rank = rank_map.get(target_str, 1001)
            stats["rank_sum"] += actual_rank
            stats["rank_count"] += 1

            pred_set = {t for _, t in result}
            for k in [3, 10, 30]:
                if target_str in {t for _, t in result[:k]}:
                    stats[k]["h"] += 1
                stats[k]["t"] += 1

            # 统计数字重合
            max_ov = len(set(target_str) & set(all_sorted[0][1]))
            # 检查top30内的最大重合
            max_ov_30 = max((len(set(target_str) & set(t)) for _, t in result[:30]), default=0)
            if max_ov_30 >= 2:
                stats["ge2"]["h"] += 1
            if max_ov_30 >= 1:
                stats["ge1"]["h"] += 1
            stats["ge2"]["t"] += 1
            stats["ge1"]["t"] += 1

        except Exception:
            continue

    L.EXPLORATION_RATE = old_exp
    return stats


def print_detailed(stats, label=""):
    print(f"\n  {'─'*55}")
    print(f"  {label}")
    print(f"  {'─'*55}")

    results = {}
    for k in [3, 10, 30]:
        n = stats[k]["t"]
        hr = stats[k]["h"] / n if n else 0
        exp = k / 1000.0
        lift = (hr / exp - 1) * 100 if exp > 0 else 0
        results[f"top{k}"] = hr
        print(f"  Top{k:>2d}: {hr*100:.2f}% ({stats[k]['h']}/{n})  Lift={lift:+.1f}%")

    ge1_n = stats["ge1"]["t"]
    ge1_r = stats["ge1"]["h"] / ge1_n if ge1_n else 0
    ge2_n = stats["ge2"]["t"]
    ge2_r = stats["ge2"]["h"] / ge2_n if ge2_n else 0

    avg_rank = stats["rank_sum"] / stats["rank_count"] if stats["rank_count"] else 500

    print(f"  ≥1码命中: {ge1_r*100:.1f}%")
    print(f"  ≥2码命中: {ge2_r*100:.1f}%")
    print(f"  平均排名: {avg_rank:.1f}")
    print(f"  总期数:   {stats[30]['t']}")

    results["ge1"] = ge1_r
    results["ge2"] = ge2_r
    results["avg_rank"] = avg_rank
    return results


def main():
    print("=" * 70)
    print("  3D预测深度诊断 v3")
    print("=" * 70)

    print("\n加载中...")
    numbers = load_numbers()
    print(f"  共 {len(numbers)} 期数据")

    all_results = {}

    # ================================================================
    # 测试矩阵：4种配置 × 多个指标
    # ================================================================
    configs = [
        {"name": "A: 纯规则(无div/corr)", "div": False, "corr": False, "exp": 0.0},
        {"name": "B: 规则+多样性",         "div": True,  "corr": False, "exp": 0.0},
        {"name": "C: 规则+相关性",           "div": False, "corr": True,  "exp": 0.0},
        {"name": "D: 规则+两者全开",        "div": True,  "corr": True,  "exp": 0.0},
        {"name": "E: 全开+5%探索",          "div": True,  "corr": True,  "exp": 0.05},
        {"name": "F: 纯规则+低形态先验",    "div": False, "corr": False, "exp": 0.0, "form_prior": 4.0},
    ]

    for cfg in configs:
        print("\n" + "=" * 70)
        print(f"  测试: {cfg['name']}")
        print("=" * 70)

        # 应用特殊配置
        if "form_prior" in cfg:
            old_fp = L.W_FORM_PRIOR
            L.W_FORM_PRIOR = cfg["form_prior"]

        stats = detailed_backtest(
            numbers,
            enable_diversity=cfg["div"],
            enable_correlation=cfg["corr"],
            exploration_rate=cfg.get("exp", 0.0),
        )
        res = print_detailed(stats, cfg["name"])
        all_results[cfg["name"]] = res

        if "form_prior" in cfg:
            L.W_FORM_PRIOR = old_fp

    # ================================================================
    # 对比汇总
    # ================================================================
    print("\n" + "=" * 70)
    print("  【配置对比汇总】")
    print("=" * 70)

    print(f"\n  {'配置':<28s} {'Top3':>6s} {'Top10':>7s} {'Top30':>7s} {'≥1码':>6s} {'≥2码':>6s} {'AvgRank':>8s}")
    print(f"  {'─'*78}")

    for name, res in all_results.items():
        print(f"  {name:<28s} {res['top3']*100:>5.1f}% {res['top10']*100:>6.1f}% {res['top30']*100:>6.1f}% "
              f"{res['ge1']*100:>5.1f}% {res['ge2']*100:>5.1f}% {res['avg_rank']:>8.1f}")

    # ================================================================
    # 找出最优配置
    # ================================================================
    print("\n" + "=" * 70)
    print("  【推荐配置】")
    print("=" * 70)

    # 按 Top30 排序
    best_top30 = max(all_results.items(), key=lambda x: x[1]["top30"])
    best_ge2 = max(all_results.items(), key=lambda x: x[1]["ge2"])
    best_rank = min(all_results.items(), key=lambda x: x[1]["avg_rank"])

    print(f"\n  最高 Top30: {best_top30[0]} ({best_top30[1]['top30']*100:.2f}%)")
    print(f"  最高 ≥2码: {best_ge2[0]} ({best_ge2[1]['ge2']*100:.1f}%)")
    print(f"  最低排名: {best_rank[0]} (平均排名 {best_rank[1]['avg_rank']:.1f})")

    # 保存结果
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_periods": len(numbers),
        "config_results": all_results,
        "recommendations": {
            "best_top30": best_top30[0],
            "best_ge2": best_ge2[0],
            "best_rank": best_rank[0],
        },
    }

    out_path = "data/diagnostic_v3_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果已保存: {out_path}")

    return output


if __name__ == "__main__":
    main()
