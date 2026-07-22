#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
战术特征 walk-forward 诊断（足球贝叶斯 skill × 现有足球模块）
============================================================

目的（回应 B 方案）：
    验证 skill 框架里「实时/战术语境证据更新」到底有没有 lift、
    能不能进核心概率。

数据现实与诚实边界：
- 本仓库历史预测（src/football/data/prediction_history.json）含
  predicted_1x2（静态赛前概率底座） + time_layers（T-6h/T-1h/final
  多时段概率快照）+ actual_result（真实赛果）。
- 这些 time_layers 的概率演化 = skill「证据驱动更新」的**历史代理信号**
  （最接近实时盘口/伤停变动对概率的影响）。用它做 walk-forward 验证
  「实时更新机制」本身有没有 lift，是严谨且可跑的。
- 真正的 xG / Field Tilt / 首发 / 伤停 等**战术特征**在本仓库历史中
  不存在（没有历史 xG/阵容数据集）。脚本为此预留可插拔接口
  （--tactical tactical_history.json）：若该文件存在则一并纳入 walk-forward；
  否则显式标 UNAVAILABLE，不编造。

方法（零依赖，纯 Python）：
- 载入已结算场次。
- 构造候选模型：
    M_base   = 静态 predicted_1x2（= skill 的 P1 静态底座）
    M_late   = final 时段 1x2（= 实时更新后的概率，skill 似然更新代理）
    M_blend  = w*M_base + (1-w)*M_late   （调权融合，w 经 LOOCV 调参）
    M_tact   = 若战术特征存在，三路融合（base / late / tactical）
- 指标：logloss / brier / 1x2 命中率（argmax==actual）。
- 切分：LOOCV（样本小，N≈6，适合留一交叉验证）。
- 融合闸门：对照 ml_shadow_evaluator.should_integrate(min_samples=100)，
  明确本诊断 N 远小于生产融合门槛，属「方法论/可行性验证」而非生产决策。

用法：
  python diagnose_tactical_feature.py \
      [--history src/football/data/prediction_history.json] \
      [--tactical tactical_history.json] \
      [--out reports/diagnose_tactical_feature.json]
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

EPS = 1e-15
LABELS = ["H", "D", "A"]


# ------------------------- 基础工具 -------------------------
def scores_to_1x2(scores: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """把 'h-a': prob 的字典汇总为 {'H','D','A'} 概率。"""
    if not scores:
        return None
    out = {"H": 0.0, "D": 0.0, "A": 0.0}
    for k, v in scores.items():
        try:
            h, a = k.split("-")
            h, a = int(h), int(a)
        except Exception:
            continue
        if h > a:
            out["H"] += v
        elif h < a:
            out["A"] += v
        else:
            out["D"] += v
    s = out["H"] + out["D"] + out["A"]
    if s <= 0:
        return None
    return {k: v / s for k, v in out.items()}


def normalize(p: Dict[str, float]) -> Dict[str, float]:
    s = sum(p.values())
    return {k: v / s for k, v in p.items()} if s > 0 else {"H": 1/3, "D": 1/3, "A": 1/3}


def logloss(p: Dict[str, float], actual: str) -> float:
    prob = max(EPS, min(1 - EPS, p.get(actual, EPS)))
    return -math.log(prob)


def brier(p: Dict[str, float], actual: str) -> float:
    return sum((p.get(l, 0.0) - (1.0 if l == actual else 0.0)) ** 2 for l in LABELS)


def hit(p: Dict[str, float], actual: str) -> int:
    return 1 if max(p, key=p.get) == actual else 0


def load_settled(history_path: str) -> List[Dict]:
    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for r in data:
        if not r.get("actual_result") or r["actual_result"] not in LABELS:
            continue
        base = r.get("predicted_1x2")
        if not base:
            continue
        base = normalize(base)
        late = scores_to_1x2((r.get("time_layers") or {}).get("final"))
        out.append({
            "match_id": r.get("match_id"),
            "league": r.get("league"),
            "home": r.get("home"),
            "away": r.get("away"),
            "base": base,
            "late": late,
            "asian": r.get("asian"),
            "total_line": r.get("total_line"),
            "actual": r["actual_result"],
            "tactical": None,  # 由 --tactical 填充
        })
    return out


def load_tactical(tactical_path: str, records: List[Dict]) -> int:
    """把 tactical_history.json（match_id -> 战术特征）并入 records。返回命中数。"""
    if not tactical_path or not os.path.exists(tactical_path):
        return 0
    with open(tactical_path, "r", encoding="utf-8") as f:
        tmap = json.load(f)
    n = 0
    by_id = {r["match_id"]: r for r in records}
    for mid, tv in tmap.items():
        if mid in by_id:
            by_id[mid]["tactical"] = tv
            n += 1
    return n


def tactical_to_1x2(tv: Dict) -> Optional[Dict[str, float]]:
    """把战术特征粗略转为 1x2 先验（仅当字段齐全时使用）。

    用 xG 近似：主胜≈xg_home/(xg_home+xg_away) 的平滑 + 控球/伤停微调。
    这是一个透明、可调的启发式，不是黑箱；walk-forward 会验证它是否真有 lift。
    """
    xh, xa = tv.get("xg_home"), tv.get("xg_away")
    if xh is None or xa is None or (xh + xa) <= 0:
        return None
    ft = tv.get("field_tilt")  # 0..1，<0.5 表示推进质量差（控球陷阱）
    poss = tv.get("possession_home")
    # 基础实力比
    base_h = xh / (xh + xa)
    # 控球陷阱惩罚：控球高但 Field Tilt 低 → 下调主胜
    if poss and ft is not None and poss > 0.60 and ft < 0.50:
        base_h *= 0.92
    # 伤停影响（关键中轴缺阵）
    dih = tv.get("injury_impact_home", 0.0)
    dia = tv.get("injury_impact_away", 0.0)
    base_h = max(0.05, base_h - 0.03 * dih + 0.03 * dia)
    base_h = min(0.95, max(0.05, base_h))
    # 平局先验：弱队对强队更易平，简单取 (1-base_h)*0.28
    draw = (1 - base_h) * 0.28
    away = max(0.05, 1 - base_h - draw)
    return normalize({"H": base_h, "D": draw, "A": away})


# ------------------------- walk-forward -------------------------
def blend(a: Dict, b: Dict, w: float) -> Dict[str, float]:
    return normalize({l: w * a[l] + (1 - w) * b[l] for l in LABELS})


def tune_w(train: List[Dict], late_ok: bool) -> float:
    """在 train 上选使平均 logloss 最小的 w（网格搜索）。"""
    best_w, best_ll = 1.0, float("inf")
    for w in [i / 20 for i in range(21)]:
        if late_ok:
            ll = sum(logloss(blend(r["base"], r["late"], w), r["actual"]) for r in train) / len(train)
        else:
            ll = sum(logloss(r["base"], r["actual"]) for r in train) / len(train)
        if ll < best_ll:
            best_ll, best_w = ll, w
    return best_w


def walk_forward(records: List[Dict]) -> Dict:
    """LOOCV：对每场留一，其余调参，评估留一场。返回各模型指标。"""
    n = len(records)
    late_available = [r for r in records if r["late"]]
    has_late = len(late_available) > 0

    agg = {
        "base": {"logloss": [], "brier": [], "hit": []},
        "late": {"logloss": [], "brier": [], "hit": []},
        "blend_tuned": {"logloss": [], "brier": [], "hit": []},
        "tact": {"logloss": [], "brier": [], "hit": []},
    }
    drift_stats = {"correct": [], "wrong": []}

    for i in range(n):
        test = records[i]
        train = records[:i] + records[i + 1:]
        if not train:
            train = records  # 单样本退化保护
        # base
        agg["base"]["logloss"].append(logloss(test["base"], test["actual"]))
        agg["base"]["brier"].append(brier(test["base"], test["actual"]))
        agg["base"]["hit"].append(hit(test["base"], test["actual"]))
        # late（仅当有 late）
        if test["late"]:
            agg["late"]["logloss"].append(logloss(test["late"], test["actual"]))
            agg["late"]["brier"].append(brier(test["late"], test["actual"]))
            agg["late"]["hit"].append(hit(test["late"], test["actual"]))
        # blend tuned
        if has_late:
            w = tune_w(train, late_ok=True)
            p = blend(test["base"], test["late"], w) if test["late"] else test["base"]
            agg["blend_tuned"]["logloss"].append(logloss(p, test["actual"]))
            agg["blend_tuned"]["brier"].append(brier(p, test["actual"]))
            agg["blend_tuned"]["hit"].append(hit(p, test["actual"]))
        # tactical（仅当本场有战术特征）
        if test["tactical"]:
            t1x2 = tactical_to_1x2(test["tactical"])
            if t1x2 and test["late"]:
                # 三路融合：base/late/tactical 等权→再 walk-forward 简化用 1/3
                p = normalize({l: (test["base"][l] + test["late"][l] + t1x2[l]) / 3 for l in LABELS})
                agg["tact"]["logloss"].append(logloss(p, test["actual"]))
                agg["tact"]["brier"].append(brier(p, test["actual"]))
                agg["tact"]["hit"].append(hit(p, test["actual"]))
        # drift 描述统计
        if test["late"]:
            d = abs(test["late"]["H"] - test["base"]["H"])
            (drift_stats["correct"] if hit(test["base"], test["actual"]) else drift_stats["wrong"]).append(d)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    res = {}
    for k, v in agg.items():
        if not v["logloss"]:
            continue
        res[k] = {
            "n": len(v["logloss"]),
            "logloss": mean(v["logloss"]),
            "brier": mean(v["brier"]),
            "hit_rate": mean(v["hit"]),
        }
    return res, drift_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "football", "data", "prediction_history.json"))
    ap.add_argument("--tactical", default=None, help="可选：战术特征历史 JSON（match_id -> 特征）")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "diagnose_tactical_feature.json"))
    args = ap.parse_args()

    records = load_settled(args.history)
    if not records:
        print("[ERR] 无已结算场次，无法诊断。")
        sys.exit(1)
    n_tac = load_tactical(args.tactical, records)

    res, drift = walk_forward(records)

    # ---- 汇总报告 ----
    base = res.get("base", {})
    late = res.get("late")
    blend = res.get("blend_tuned")
    tact = res.get("tact")

    print("=" * 64)
    print("战术特征 walk-forward 诊断（足球贝叶斯 skill 验证）")
    print("=" * 64)
    print(f"已结算样本 N = {len(records)}（融合生产闸门需 ≥100，见 ml_shadow_evaluator）")
    print(f"含 late 时段概率的场次 = {sum(1 for r in records if r['late'])}")
    print(f"含真实战术特征(xG/FT/伤停)的场次 = {n_tac}"
          f"{'  → 已纳入三路融合' if n_tac else '  → UNAVAILABLE（无历史战术数据集，未编造）'}")
    print("-" * 64)
    print(f"{'模型':<16}{'logloss':>10}{'brier':>10}{'命中率':>10}")
    for name, m in [("M_base(静态)", base), ("M_late(实时代理)", late),
                    ("M_blend(调权融合)", blend), ("M_tact(含战术)", tact)]:
        if not m:
            continue
        print(f"{name:<16}{m['logloss']:>10.4f}{m['brier']:>10.4f}{m['hit_rate']*100:>9.1f}%")

    # lift
    print("-" * 64)
    if late and base["logloss"]:
        ll_lift = (base["logloss"] - late["logloss"]) / base["logloss"] * 100
        print(f"实时代理 vs 静态底座：logloss {'↓' if ll_lift>0 else '↑'} {abs(ll_lift):.1f}%"
              f"（{'实时更新有增益' if ll_lift>0 else '实时更新无增益'}）")
    if blend and base["logloss"]:
        ll_lift = (base["logloss"] - blend["logloss"]) / base["logloss"] * 100
        print(f"调权融合 vs 静态底座：logloss {'↓' if ll_lift>0 else '↑'} {abs(ll_lift):.1f}%")
    if tact and base["logloss"]:
        ll_lift = (base["logloss"] - tact["logloss"]) / base["logloss"] * 100
        print(f"含战术特征 vs 静态底座：logloss {'↓' if ll_lift>0 else '↑'} {abs(ll_lift):.1f}%")
    if drift["correct"] and drift["wrong"]:
        print(f"盘口漂移(|ΔH|)：命中场均值 {sum(drift['correct'])/len(drift['correct']):.3f}"
              f" / 错判场均值 {sum(drift['wrong'])/len(drift['wrong']):.3f}"
              f" → {'漂移大时更易错判（steam 信号值得跟进）' if sum(drift['wrong'])/len(drift['wrong']) > sum(drift['correct'])/len(drift['correct']) else '漂移与对错无单调关系'}")
    print("=" * 64)
    print("结论：本诊断为『实时更新机制可行性』验证。真正 xG/Field Tilt/伤停")
    print("特征需历史战术数据集（--tactical）；当前缺失则框架保留接口，不编造。")

    # JSON 输出
    out = {
        "generated_at": datetime.now().isoformat(),
        "n_settled": len(records),
        "n_with_late": sum(1 for r in records if r["late"]),
        "n_with_tactical": n_tac,
        "models": res,
        "drift_stats": {
            "correct_mean": (sum(drift["correct"]) / len(drift["correct"])) if drift["correct"] else None,
            "wrong_mean": (sum(drift["wrong"]) / len(drift["wrong"])) if drift["wrong"] else None,
        },
        "verdict": {
            "late_vs_base_logloss_lift_pct": (
                (base["logloss"] - late["logloss"]) / base["logloss"] * 100) if late else None,
            "tactical_features_available": n_tac > 0,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 诊断 JSON 已写出: {args.out}")


if __name__ == "__main__":
    main()
