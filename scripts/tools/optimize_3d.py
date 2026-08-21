"""
3D预测模块系统性优化脚本
===========================
基于基线回测结果（Top30≈随机），执行以下优化：

Phase 1: 特征开关消融测试（单独开启每个关闭的特征）
Phase 2: 核心权重坐标下降搜索
Phase 3: 新信号注入测试
Phase 4: 最终全量验证

输出：最优配置 + 回测对比
"""

import json
import sys
import time
from collections import Counter

sys.path.insert(0, '.')
import src.lottery3d as L

# ================================================================
# 数据加载
# ================================================================
def load_data():
    data = L.fetch_data()
    return [x[2] for x in data]


def quick_backtest(numbers, start_idx=250, end_idx=None, step=8, top_k=30):
    """快速回测：返回 TopK 命中率"""
    if end_idx is None:
        end_idx = len(numbers) - 50
    hits = total = 0

    for idx in range(start_idx, end_idx, step):
        target = numbers[idx]
        target_str = "".join(map(str, target))
        hist = numbers[:idx]
        if len(hist) < 100:
            continue

        try:
            periods = [str(i) for i in range(len(hist))]
            sums = [sum(x) for x in hist]
            spans = [max(x) - min(x) for x in hist]

            ww, _ = L.resolve_window_weights(hist, compute_weights=False, period=periods[-1])
            meta = L.build_ranking_meta(hist, ww, sums, spans)
            score, _ = L.ensemble_digit_scores(hist, ww, dynamic=meta.get("dynamic"))
            danma, _, kill, _ = L.pick_dan_tuo_kill(score, enable_danma_random=False)

            result = L.rank_triplets(
                score, danma, kill, meta, top_n=top_k,
                enable_exploration=False, apply_noise=False,
                enable_cold_hot_balance=False, enable_diversity=False,
                enable_correlation=False, recent_recommendations=None,
            )

            pred_set = {t for _, t in result[:top_k]}
            total += 1
            if target_str in pred_set:
                hits += 1
        except Exception as e:
            continue

    return {"hit_rate": hits / total if total else 0, "hits": hits, "total": total}


# ================================================================
# Phase 1: 特征开关消融测试
# ================================================================
def phase1_feature_flags(numbers):
    print("\n" + "=" * 65)
    print("  Phase 1: 特征开关消融测试")
    print("=" * 65)

    base = quick_backtest(numbers)
    print(f"  基线 Top30: {base['hit_rate']*100:.2f}% (hits={base['hits']}/{base['total']})")

    # 测试关闭的特征：单独开启
    open_tests = [
        ("miss", True, "遗漏加分"),
        ("neighbor", True, "邻号加分"),
        ("road", True, "012路匹配"),
        ("form_switch", True, "形态切换"),
        ("cold_hot_balance", True, "冷热平衡"),
    ]

    results_open = {}
    for flag_name, val, desc in open_tests:
        old_val = L.FEATURE_FLAGS.get(flag_name)
        if old_val == val:
            print(f"  ⏭️  {flag_name} ({desc}): 已经是开启状态")
            continue

        L.FEATURE_FLAGS[flag_name] = val
        try:
            res = quick_backtest(numbers)
            delta = (res["hit_rate"] - base["hit_rate"]) * 100
            marker = " ✅" if res["hit_rate"] > base["hit_rate"] + 0.003 else ""
            print(f"  {'开启':>4s} {flag_name:15s} ({desc}): {base['hit_rate']*100:.2f}% → {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
            results_open[flag_name] = {"rate": res["hit_rate"], "delta": delta, "action": "open"}
        except Exception as e:
            print(f"  ❌ {flag_name}: ERROR - {e}")
        finally:
            L.FEATURE_FLAGS[flag_name] = old_val

    # 测试开启的特征：单独关闭
    close_tests = [
        ("consecutive", False, "连号奖励"),
        ("lag1_repeat", False, "上期重复"),
        ("ratio", False, "奇偶大小比"),
        ("pair", False, "数字配对"),
        ("slope", False, "斜连走势"),
    ]

    results_close = {}
    for flag_name, val, desc in close_tests:
        old_val = L.FEATURE_FLAGS.get(flag_name)
        if old_val == val:
            continue

        L.FEATURE_FLAGS[flag_name] = val
        try:
            res = quick_backtest(numbers)
            delta = (res["hit_rate"] - base["hit_rate"]) * 100
            marker = " ✅" if res["hit_rate"] > base["hit_rate"] + 0.003 else ""
            print(f"  {'关闭':>4s} {flag_name:15s} ({desc}): {base['hit_rate']*100:.2f}% → {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
            results_close[flag_name] = {"rate": res["hit_rate"], "delta": delta, "action": "close"}
        except Exception as e:
            print(f"  ❌ {flag_name}: ERROR - {e}")
        finally:
            L.FEATURE_FLAGS[flag_name] = old_val

    return {"baseline": base, "open": results_open, "close": results_close}


# ================================================================
# Phase 2: 权重坐标下降搜索
# ================================================================
WEIGHT_SEARCH_SPACE = {
    "W_MARKOV":       [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "W_MARKOV2":      [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "W_HOT_GLOBAL":   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "W_HOT_POS":      [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    "W_DANMA_HIT":    [2.0, 3.0, 4.0, 5.0, 6.0],
    "W_KILL_PENALTY": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "W_CONSECUTIVE":  [0.8, 1.2, 1.5, 2.0, 2.5, 3.0],
    "W_FORM_PRIOR":   [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "W_POS_REPEAT":   [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    "W_RATIO_MATCH":  [1.0, 1.4, 1.8, 2.2, 2.5, 3.0],
    "SUM_SOFT_SIGMA": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    "SPAN_SOFT_SIGMA":[0.8, 1.0, 1.2, 1.4, 1.8, 2.2],
}


def phase2_weight_search(numbers):
    print("\n" + "=" * 65)
    print("  Phase 2: 权重坐标下降搜索")
    print("=" * 65)

    base = quick_backtest(numbers)
    print(f"  基线 Top30: {base['hit_rate']*100:.2f}%")

    best_changes = {}

    for attr_name, search_range in WEIGHT_SEARCH_SPACE.items():
        old_val = getattr(L, attr_name)
        best_v = old_val
        best_r = base["hit_rate"]

        for v in search_range:
            setattr(L, attr_name, v)
            try:
                res = quick_backtest(numbers)
                r = res["hit_rate"]
                if r > best_r + 0.002:
                    best_r = r
                    best_v = v
            except:
                continue

        setattr(L, attr_name, best_v)
        delta = (best_r - base["hit_rate"]) * 100
        if abs(delta) > 0.08:
            status = "✅ 提升" if delta > 0 else "⬇️"
            print(f"  {attr_name:18s}: {old_val:.1f} → {best_v:.1f}  "
                  f"{base['hit_rate']*100:.2f}%→{best_r*100:.2f}% ({delta:+.2f}%) {status}")
            best_changes[attr_name] = {"old": old_val, "new": best_v, "delta": delta}
        else:
            print(f"  {attr_name:18s}: 保持={old_val:.1f}  (变化<0.08%)")

    return best_changes


# ================================================================
# Phase 3: 新信号注入测试
# ================================================================
def phase3_new_signals(numbers):
    print("\n" + "=" * 65)
    print("  Phase 3: 新信号注入测试")
    print("=" * 65)

    base = quick_backtest(numbers)
    print(f"  基线 Top30: {base['hit_rate']*100:.2f}%")

    results = {}

    # 测试1：适度探索率
    for exp_rate in [0.05, 0.10, 0.15]:
        old_exp = L.EXPLORATION_RATE
        L.EXPLORATION_RATE = exp_rate
        try:
            res = quick_backtest(numbers)
            delta = (res["hit_rate"] - base["hit_rate"]) * 100
            marker = " ✅" if res["hit_rate"] > base["hit_rate"] + 0.003 else ""
            print(f"  EXPLORATION_RATE={exp_rate:.2f}: {base['hit_rate']*100:.2f}% → {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
            results[f"exp_{exp_rate}"] = {"rate": res["hit_rate"], "delta": delta}
        finally:
            L.EXPLORATION_RATE = old_exp

    # 测试2：回补奖励增强
    for rb in [0.8, 1.2, 1.5, 2.0]:
        old_rb = L.REBOUND_BONUS
        L.REBOUND_BONUS = rb
        try:
            res = quick_backtest(numbers)
            delta = (res["hit_rate"] - base["hit_rate"]) * 100
            marker = " ✅" if res["hit_rate"] > base["hit_rate"] + 0.003 else ""
            print(f"  REBOUND_BONUS={rb:.1f}: {base['hit_rate']*100:.2f}% → {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
            results[f"rb_{rb}"] = {"rate": res["hit_rate"], "delta": delta}
        finally:
            L.REBOUND_BONUS = old_rb

    # 测试3：杀码惩罚减弱/增强
    for kp in [2.0, 3.0, 5.0, 7.0]:
        old_kp = L.W_KILL_PENALTY
        L.W_KILL_PENALTY = kp
        try:
            res = quick_backtest(numbers)
            delta = (res["hit_rate"] - base["hit_rate"]) * 100
            marker = " ✅" if res["hit_rate"] > base["hit_rate"] + 0.003 else ""
            print(f"  W_KILL_PENALTY={kp:.1f}: {base['hit_rate']*100:.2f}% → {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
            results[f"kp_{kp}"] = {"rate": res["hit_rate"], "delta": delta}
        finally:
            L.W_KILL_PENALTY = old_kp

    # 测试4：和值sigma调整
    for ss in [2.2, 2.8, 3.5, 4.0]:
        old_ss = L.SUM_SOFT_SIGMA
        L.SUM_SOFT_SIGMA = ss
        try:
            res = quick_backtest(numbers)
            delta = (res["hit_rate"] - base["hit_rate"]) * 100
            marker = " ✅" if res["hit_rate"] > base["hit_rate"] + 0.003 else ""
            print(f"  SUM_SOFT_SIGMA={ss:.1f}: {base['hit_rate']*100:.2f}% → {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
            results[f"ss_{ss}"] = {"rate": res["hit_rate"], "delta": delta}
        finally:
            L.SUM_SOFT_SIGMA = old_ss

    # 测试5：跨度sigma调整
    for sps in [1.0, 1.2, 1.6, 2.0]:
        old_sps = L.SPAN_SOFT_SIGMA
        L.SPAN_SOFT_SIGMA = sps
        try:
            res = quick_backtest(numbers)
            delta = (res["hit_rate"] - base["hit_rate"]) * 100
            marker = " ✅" if res["hit_rate"] > base["hit_rate"] + 0.003 else ""
            print(f"  SPAN_SOFT_SIGMA={sps:.1f}: {base['hit_rate']*100:.2f}% → {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
            results[f"sps_{sps}"] = {"rate": res["hit_rate"], "delta": delta}
        finally:
            L.SPAN_SOFT_SIGMA = old_sps

    return results


# ================================================================
# Phase 4: 最优组合验证
# ================================================================
def phase4_final_validation(numbers, step=3):
    """最终全量验证"""
    print("\n" + "=" * 65)
    print("  Phase 4: 最终全量验证 (step={})".format(step))
    print("=" * 65)

    start_idx = 250
    end_idx = len(numbers) - 50

    stats = {3: {"h": 0, "t": 0}, 10: {"h": 0, "t": 0}, 30: {"h": 0, "t": 0}, "ge2": {"h": 0, "t": 0}}

    t0 = time.time()
    count = 0
    for i in range(start_idx, end_idx, step):
        target = numbers[i]
        target_str = "".join(map(str, target))
        hist = numbers[:i]

        try:
            periods = [str(j) for j in range(len(hist))]
            sums = [sum(x) for x in hist]
            spans = [max(x) - min(x) for x in hist]

            ww, _ = L.resolve_window_weights(hist, compute_weights=False, period=periods[-1])
            meta = L.build_ranking_meta(hist, ww, sums, spans)
            score, _ = L.ensemble_digit_scores(hist, ww, dynamic=meta.get("dynamic"))
            danma, _, kill, _ = L.pick_dan_tuo_kill(score, enable_danma_random=False)

            result = L.rank_triplets(
                score, danma, kill, meta, top_n=30,
                enable_exploration=False, apply_noise=False,
                enable_cold_hot_balance=False, enable_diversity=True,
                enable_correlation=True, recent_recommendations=None,
            )

            pred_set = {t for _, t in result}
            count += 1

            for k in [3, 10, 30]:
                topk = {t for _, t in result[:k]}
                if target_str in topk:
                    stats[k]["h"] += 1
                stats[k]["t"] += 1

            max_ov = max((len(set(target_str) & set(t)) for t in pred_set), default=0)
            if max_ov >= 2:
                stats["ge2"]["h"] += 1
            stats["ge2"]["t"] += 1

        except Exception as e:
            continue

        if count % 80 == 0:
            elapsed = time.time() - t0
            print(f"    进度: {count}期 ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  完成: {count}期, 耗时 {elapsed:.1f}s ({elapsed/count:.2f}s/期)")

    final = {}
    for k in [3, 10, 30]:
        n = stats[k]["t"]
        hr = stats[k]["h"] / n if n else 0
        exp = k / 1000.0
        lift = (hr / exp - 1) * 100 if exp > 0 else 0
        final[f"top{k}_rate"] = hr
        final[f"top{k}_lift"] = lift
        print(f"  Top{k:>2d}: {hr*100:.2f}% ({stats[k]['h']}/{n})  期望={exp*100:.1f}%  Lift={lift:+.1f}%")

    ge2_n = stats["ge2"]["t"]
    ge2_r = stats["ge2"]["h"] / ge2_n if ge2_n else 0
    final["ge2_rate"] = ge2_r
    print(f"  >=2码: {ge2_r*100:.1f}%")
    final["total_periods"] = count
    final["elapsed"] = round(elapsed, 1)

    return final


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("  福彩3D 系统性优化脚本")
    print("=" * 70)

    print("\n加载数据...")
    numbers = load_data()
    print(f"  共 {len(numbers)} 期数据")

    # 记录原始配置用于恢复
    original_flags = dict(L.FEATURE_FLAGS)
    original_weights = {}
    for attr_name in WEIGHT_SEARCH_SPACE:
        original_weights[attr_name] = getattr(L, attr_name)
    original_params = {
        "EXPLORATION_RATE": L.EXPLORATION_RATE,
        "REBOUND_BONUS": L.REBOUND_BONUS,
        "W_KILL_PENALTY": L.W_KILL_PENALTY,
        "SUM_SOFT_SIGMA": L.SUM_SOFT_SIGMA,
        "SPAN_SOFT_SIGMA": L.SPAN_SOFT_SIGMA,
    }

    # Phase 1
    p1_results = phase1_feature_flags(numbers)

    # Phase 2
    p2_results = phase2_weight_search(numbers)

    # Phase 3
    p3_results = phase3_new_signals(numbers)

    # Phase 4: 用当前已修改的配置跑最终验证
    p4_results = phase4_final_validation(numbers)

    # 汇总
    print("\n" + "=" * 70)
    print("  优化汇总")
    print("=" * 70)

    current_config = {
        "flags": dict(L.FEATURE_FLAGS),
        "weights": {},
        "params": {},
    }
    for attr_name in WEIGHT_SEARCH_SPACE:
        current_config["weights"][attr_name] = getattr(L, attr_name)
    for key in original_params:
        current_config["params"][key] = getattr(L, key)

    summary = {
        "phase1_feature_flags": p1_results,
        "phase2_weights": p2_results,
        "phase3_signals": p3_results,
        "phase4_final": p4_results,
        "current_config": current_config,
    }

    out_path = "data/optimization_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")

    # 恢复原始配置
    L.FEATURE_FLAGS = original_flags
    for attr_name, val in original_weights.items():
        setattr(L, attr_name, val)
    for key, val in original_params.items():
        setattr(L, key, val)
    print("(原始配置已恢复)")

    return summary


if __name__ == "__main__":
    main()
