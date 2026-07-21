"""
3D预测模块保守优化脚本 v2
=============================
策略：
1. 只做理论上合理的、小幅度的参数调整
2. 每个修改都用全量回测(step=3)验证
3. 累积式修改：只有正向改进才保留

目标：将 Top30 从基线 ~2.94% 提升到 >3.5%
"""

import json
import sys
import time

sys.path.insert(0, '.')
import src.lottery3d as L


def load_numbers():
    data = L.fetch_data()
    return [x[2] for x in data]


def full_backtest(numbers, start_idx=250, end_idx=None, step=3, top_k=30):
    """全量回测（与基线相同的设置）"""
    if end_idx is None:
        end_idx = len(numbers) - 50

    stats = {3: {"h": 0, "t": 0}, 10: {"h": 0, "t": 0}, 30: {"h": 0, "t": 0}, "ge2": {"h": 0, "t": 0}}

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
                enable_exploration=(L.EXPLORATION_RATE > 0),
                apply_noise=False,
                enable_cold_hot_balance=False,
                enable_diversity=True,
                enable_correlation=True,
                recent_recommendations=None,
            )

            pred_set = {t for _, t in result}
            for k in [3, 10, 30]:
                if target_str in {t for _, t in result[:k]}:
                    stats[k]["h"] += 1
                stats[k]["t"] += 1

            max_ov = max((len(set(target_str) & set(t)) for t in pred_set), default=0)
            if max_ov >= 2:
                stats["ge2"]["h"] += 1
            stats["ge2"]["t"] += 1

        except Exception as e:
            continue

    return stats


def print_result(stats, label=""):
    """格式化输出回测结果"""
    print(f"\n  {label}")
    print(f"  {'─'*50}")

    results = {}
    for k in [3, 10, 30]:
        n = stats[k]["t"]
        hr = stats[k]["h"] / n if n else 0
        exp = k / 1000.0
        lift = (hr / exp - 1) * 100 if exp > 0 else 0
        results[f"top{k}"] = hr
        marker = " ✅" if lift > 5 else (" ⚠️" if lift < -5 else "")
        print(f"  Top{k:>2d}: {hr*100:.2f}% ({stats[k]['h']}/{n})  Lift={lift:+.1f}%{marker}")

    ge2_n = stats["ge2"]["t"]
    ge2_r = stats["ge2"]["h"] / ge2_n if ge2_n else 0
    print(f"  >=2码: {ge2_r*100:.1f}%")
    print(f"  总期数: {stats[30]['t']}")

    return results


def main():
    print("=" * 65)
    print("  3D预测保守优化 v2")
    print("=" * 65)

    print("\n加载中...")
    numbers = load_numbers()
    print(f"  共 {len(numbers)} 期数据")

    # ================================================================
    # 基线回测
    # ================================================================
    print("\n" + "=" * 65)
    print("  [基线] 当前默认配置")
    print("=" * 65)
    base_stats = full_backtest(numbers)
    base_results = print_result(base_stats, "基线")
    base_top30 = base_results["top30"]

    # 记录原始配置
    original_config = {
        "flags": dict(L.FEATURE_FLAGS),
        "weights": {},
        "params": {
            "EXPLORATION_RATE": L.EXPLORATION_RATE,
            "W_KILL_PENALTY": L.W_KILL_PENALTY,
            "W_FORM_PRIOR": L.W_FORM_PRIOR,
            "W_HOT_GLOBAL": L.W_HOT_GLOBAL,
            "W_HOT_POS": L.W_HOT_POS,
            "W_DANMA_HIT": L.W_DANMA_HIT,
            "SUM_SOFT_SIGMA": L.SUM_SOFT_SIGMA,
            "SPAN_SOFT_SIGMA": L.SPAN_SOFT_SIGMA,
            "W_CONSECUTIVE": L.W_CONSECUTIVE,
            "W_POS_REPEAT": L.W_POS_REPEAT,
            "W_RATIO_MATCH": L.W_RATIO_MATCH,
        },
    }

    # 累积最优结果
    best_top30 = base_top30
    best_stats = base_stats
    applied_changes = []

    # ================================================================
    # 改动1: 启用 road (012路匹配)
    # ================================================================
    print("\n" + "=" * 65)
    print("  [测试1] 启用 road (012路匹配)")
    print("=" * 65)
    L.FEATURE_FLAGS["road"] = True
    stats1 = full_backtest(numbers)
    res1 = print_result(stats1, "启用 road")

    if res1["top30"] > best_top30 + 0.002:
        print(f"  → 有效! Top30: {best_top30*100:.2f}% → {res1['top30']*100:.2f}%")
        best_top30 = res1["top30"]
        best_stats = stats1
        applied_changes.append("启用 road (012路)")
    else:
        print(f"  → 无效，回滚")
        L.FEATURE_FLAGS["road"] = False

    # ================================================================
    # 改动2: 降低 W_FORM_PRIOR (形态先验权重)
    # ================================================================
    print("\n" + "=" * 65)
    print("  [测试2] W_FORM_PRIOR: 6.0 → 4.0")
    print("=" * 65)
    old_fp = L.W_FORM_PRIOR
    L.W_FORM_PRIOR = 4.0
    stats2 = full_backtest(numbers)
    res2 = print_result(stats2, f"W_FORM_PRIOR={L.W_FORM_PRIOR}")

    if res2["top30"] > best_top30 + 0.002:
        print(f"  → 有效! Top30: {best_top30*100:.2f}% → {res2['top30']*100:.2f}%")
        best_top30 = res2["top30"]
        best_stats = stats2
        applied_changes.append(f"W_FORM_PRIOR={old_fp}→{L.W_FORM_PRIOR}")
    else:
        print(f"  → 无效，回滚")
        L.W_FORM_PRIOR = old_fp

    # ================================================================
    # 改动3: 降低 W_KILL_PENALTY (杀码惩罚)
    # ================================================================
    print("\n" + "=" * 65)
    print("  [测试3] W_KILL_PENALTY: {} → 3.0".format(L.W_KILL_PENALTY))
    print("=" * 65)
    old_kp = L.W_KILL_PENALTY
    L.W_KILL_PENALTY = 3.0
    stats3 = full_backtest(numbers)
    res3 = print_result(stats3, f"W_KILL_PENALTY={L.W_KILL_PENALTY}")

    if res3["top30"] > best_top30 + 0.002:
        print(f"  → 有效! Top30: {best_top30*100:.2f}% → {res3['top30']*100:.2f}%")
        best_top30 = res3["top30"]
        best_stats = stats3
        applied_changes.append(f"W_KILL_PENALTY={old_kp}→{L.W_KILL_PENALTY}")
    else:
        print(f"  → 无效，回滚")
        L.W_KILL_PENALTY = old_kp

    # ================================================================
    # 改动4: 微调和值sigma (收紧和值约束)
    # ================================================================
    print("\n" + "=" * 65)
    print("  [测试4] SUM_SOFT_SIGMA: {} → 2.6".format(L.SUM_SOFT_SIGMA))
    print("=" * 65)
    old_ss = L.SUM_SOFT_SIGMA
    L.SUM_SOFT_SIGMA = 2.6
    stats4 = full_backtest(numbers)
    res4 = print_result(stats4, f"SUM_SOFT_SIGMA={L.SUM_SOFT_SIGMA}")

    if res4["top30"] > best_top30 + 0.002:
        print(f"  → 有效! Top30: {best_top30*100:.2f}% → {res4['top30']*100:.2f}%")
        best_top30 = res4["top30"]
        best_stats = stats4
        applied_changes.append(f"SUM_SOFT_SIGMA={old_ss}→{L.SUM_SOFT_SIGMA}")
    else:
        print(f"  → 无效，回滚")
        L.SUM_SOFT_SIGMA = old_ss

    # ================================================================
    # 改动5: 小幅探索率 (增加推荐多样性)
    # ================================================================
    print("\n" + "=" * 65)
    print("  [测试5] EXPLORATION_RATE: 0 → 0.08")
    print("=" * 65)
    old_exp = L.EXPLORATION_RATE
    L.EXPLORATION_RATE = 0.08
    stats5 = full_backtest(numbers)
    res5 = print_result(stats5, f"EXPLORATION_RATE={L.EXPLORATION_RATE}")

    if res5["top30"] > best_top30 + 0.002:
        print(f"  → 有效! Top30: {best_top30*100:.2f}% → {res5['top30']*100:.2f}%")
        best_top30 = res5["top30"]
        best_stats = stats5
        applied_changes.append(f"EXPLORATION_RATE={old_exp}→{L.EXPLORATION_RATE}")
    else:
        print(f"  → 无效，回滚")
        L.EXPLORATION_RATE = old_exp

    # ================================================================
    # 改动6: 提高 W_HOT_GLOBAL (热号全局权重)
    # ================================================================
    print("\n" + "=" * 65)
    print("  [测试6] W_HOT_GLOBAL: {} → 3.0".format(L.W_HOT_GLOBAL))
    print("=" * 65)
    old_hg = L.W_HOT_GLOBAL
    L.W_HOT_GLOBAL = 3.0
    stats6 = full_backtest(numbers)
    res6 = print_result(stats6, f"W_HOT_GLOBAL={L.W_HOT_GLOBAL}")

    if res6["top30"] > best_top30 + 0.002:
        print(f"  → 有效! Top30: {best_top30*100:.2f}% → {res6['top30']*100:.2f}%")
        best_top30 = res6["top30"]
        best_stats = stats6
        applied_changes.append(f"W_HOT_GLOBAL={old_hg}→{L.W_HOT_GLOBAL}")
    else:
        print(f"  → 无效，回滚")
        L.W_HOT_GLOBAL = old_hg

    # ================================================================
    # 改动7: 关闭 consecutive (连号奖励)
    # ================================================================
    print("\n" + "=" * 65)
    print("  [测试7] 关闭 consecutive (连号奖励)")
    print("=" * 65)
    old_consec = L.FEATURE_FLAGS.get("consecutive", True)
    L.FEATURE_FLAGS["consecutive"] = False
    stats7 = full_backtest(numbers)
    res7 = print_result(stats7, "关闭 consecutive")

    if res7["top30"] > best_top30 + 0.002:
        print(f"  → 有效! Top30: {best_top30*100:.2f}% → {res7['top30']*100:.2f}%")
        best_top30 = res7["top30"]
        best_stats = stats7
        applied_changes.append("关闭 consecutive")
    else:
        print(f"  → 无效，回滚")
        L.FEATURE_FLAGS["consecutive"] = old_consec

    # ================================================================
    # 最终汇总
    # ================================================================
    print("\n" + "=" * 65)
    print("  【优化汇总】")
    print("=" * 65)

    print(f"\n  基线 Top30: {base_results['top30']*100:.2f}%")
    print(f"  最终 Top30: {best_top30*100:.2f}%")
    delta = (best_top30 - base_results['top30']) * 100
    print(f"  净提升: {delta:+.2f}%")

    print(f"\n  应用的改动 ({len(applied_changes)} 个):")
    for i, change in enumerate(applied_changes, 1):
        print(f"    {i}. {change}")

    # 当前最终配置
    final_config = {
        "flags": dict(L.FEATURE_FLAGS),
        "weights": {
            "W_FORM_PRIOR": L.W_FORM_PRIOR,
            "W_KILL_PENALTY": L.W_KILL_PENALTY,
            "SUM_SOFT_SIGMA": L.SUM_SOFT_SIGMA,
            "W_HOT_GLOBAL": L.W_HOT_GLOBAL,
        },
        "params": {
            "EXPLORATION_RATE": L.EXPLORATION_RATE,
        },
        "applied_changes": applied_changes,
        "baseline": {"top30": base_results["top30"], "top10": base_results["top10"], "top3": base_results["top3"]},
        "final": {"top30": best_top30, "top10": None, "top3": None},
    }

    # 保存结果
    out_path = "data/optimization_v2_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果已保存: {out_path}")

    # 恢复原始配置
    L.FEATURE_FLAGS = original_config["flags"]
    for k, v in original_config["weights"].items():
        setattr(L, k, v)
    for k, v in original_config["params"].items():
        setattr(L, k, v)
    print("\n  (原始配置已恢复)")

    return final_config


if __name__ == "__main__":
    main()
