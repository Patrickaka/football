#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
北单深度报告批量生成器（CLI 包装器）
================================
调用现有北单模块 generate_beidan_recommendations 获取今日推荐，把每场 rec 落盘到
src/beidan/cache/beidan_{match_id}.json，再交给 src.football.bayes_report.build_beidan_report
生成深度报告，并登记进 reports/football_bayes_manifest.json（按 match_id 索引）。

说明：
- 需要网络可达 zgzcw.com（与北单模块同源）。
- 本脚本不再是生产必需：server 的 /api/beidan 会自动持久化 rec，并在用户点击报告时按需生成。
  本脚本仅用于主动预生成 / 测试。

用法：
  python gen_beidan_reports.py [YYYY-MM-DD] [--source zgzcw] [--types spf,rqspf,zjq]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.football.bayes_report import build_beidan_report, load_live_context  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default=None, help="比赛日期 YYYY-MM-DD（默认取模块默认日期）")
    ap.add_argument("--source", default="zgzcw", help="数据源 zgzcw/dc/jczq")
    ap.add_argument("--types", default="spf,rqspf,zjq", help="投注类型，逗号分隔")
    args = ap.parse_args()

    from src.beidan import generate_beidan_recommendations
    bet_types = [t.strip() for t in args.types.split(",") if t.strip()]

    print(f"[*] 调用北单推荐（date={args.date}, source={args.source}, types={bet_types}）...")
    result = generate_beidan_recommendations(
        date=args.date, bet_types=bet_types, source=args.source, save_history=False
    )
    if "error" in result:
        print(f"[ERR] 北单推荐失败: {result['error']}")
        sys.exit(1)

    recs = result.get("recommendations", []) or []
    print(f"[*] 获取到 {len(recs)} 场推荐")

    beidan_cache = os.path.join(ROOT, "src", "beidan", "cache")
    os.makedirs(beidan_cache, exist_ok=True)
    reports_dir = os.path.join(ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    live_dir = reports_dir

    ok = skip = 0
    for rec in recs:
        # 北单 rec 的 match_id 嵌套在 spf 内，需上提
        mid = str(rec.get("match_id") or (rec.get("spf") or {}).get("match_id", "") or "")
        if not mid:
            skip += 1
            continue
        rec["match_id"] = mid  # 回填，供 build_beidan_report / manifest 使用
        try:
            with open(os.path.join(beidan_cache, f"beidan_{mid}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[WARN] rec 落盘失败 {mid}: {e}")

        live = load_live_context(mid)  # 若 reports/ 下存在 live_context_{mid}.json 则叠加
        out = os.path.join(reports_dir, f"beidan_bayes_{mid}.html")
        try:
            build_beidan_report(rec, live, out)
            ok += 1
        except Exception as e:
            print(f"[ERR] 生成北单报告失败 {mid}: {e}")
            skip += 1

    print(f"\n[北单批量完成] 成功={ok} / 跳过={skip} / 共={len(recs)}")


if __name__ == "__main__":
    main()
