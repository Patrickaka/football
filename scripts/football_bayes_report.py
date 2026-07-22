#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单场深度报告生成器（CLI 包装器）
================================
核心生成逻辑已迁入 src/football/bayes_report.py（生产环境 server 也调用它做按需生成），
本脚本仅作为命令行便捷入口，用于批量/手动预生成报告文件。

用法：
  # 单场（足球，吃模块缓存 pkl）
  python football_bayes_report.py --cache <match_analysis.pkl> \
         [--live <live_context.json>] [--out <report.html>]

  # 批量：覆盖某天所有场次（自动匹配 live_context_{match_id}.json）
  python football_bayes_report.py --date 2026-07-22 [--live-dir <dir>]

  # 北单单场（吃 generate_beidan_recommendations 返回的 rec JSON）
  python football_bayes_report.py --beidan-rec <rec.json> [--live <live_context.json>]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.football.bayes_report import (  # noqa: E402
    build_report, build_beidan_report, discover_daily_caches,
    load_module_cache, cleanup_old_reports,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="单场 match_analysis pkl 路径（足球模块缓存）")
    ap.add_argument("--live", default=None, help="live_context.json 路径（实时战术语境/伤停等）")
    ap.add_argument("--out", default=None, help="输出 HTML 路径（仅单场模式）")
    ap.add_argument("--date", default=None,
                    help="批量模式：扫描该日期（YYYY-MM-DD）所有最终分析缓存，逐场生成报告")
    ap.add_argument("--live-dir", default=None,
                    help="批量模式：实时数据目录，按 live_context_{match_id}.json 自动匹配")
    ap.add_argument("--beidan-rec", default=None,
                    help="北单单场：传入 rec JSON 路径，生成北单深度报告")
    ap.add_argument("--cleanup", action="store_true",
                    help="清理 reports/ 下生成时间超过 --days 天的深度报告（含配套 live_context 与 manifest 条目）")
    ap.add_argument("--days", type=int, default=3,
                    help="--cleanup 的留存天数阈值，默认 3 天")
    ap.add_argument("--dry-run", action="store_true",
                    help="与 --cleanup 配合：只统计将要删除的文件，不实际删除")
    args = ap.parse_args()

    # 留存清理模式
    if args.cleanup:
        res = cleanup_old_reports(max_age_days=args.days, dry_run=args.dry_run)
        print(f"\n[清理] reports 目录: {res['reports_dir']}")
        print(f"      留存阈值: {res['max_age_days']} 天  dry_run={res['dry_run']}")
        print(f"      保留: {res['kept_count']} 篇  删除: {res['deleted_count']} 篇  "
              f"释放: {res['freed_bytes']/1024:.1f} KB  manifest移除: {res['pruned_manifest']} 条")
        for d in res["deleted"]:
            tag = " [将删除]" if res["dry_run"] else " [已删除]"
            print(f"  - {d['file']}  match_id={d['mid']}  已生成{d['age_days']}天前  "
                  f"{d['size']/1024:.1f}KB{tag}")
        return

    # 北单单场模式
    if args.beidan_rec:
        if not os.path.exists(args.beidan_rec):
            ap.error(f"--beidan-rec 文件不存在: {args.beidan_rec}")
        with open(args.beidan_rec, "r", encoding="utf-8") as f:
            rec = json.load(f)
        live = {}
        if args.live:
            with open(args.live, "r", encoding="utf-8") as f:
                live = json.load(f)
        mid = str(rec.get("match_id", "beidan"))
        if not args.out:
            args.out = os.path.join(ROOT, "reports", f"beidan_bayes_{mid}.html")
        build_beidan_report(rec, live, args.out)
        return

    # 批量模式
    if args.date:
        caches = discover_daily_caches(args.date)
        if not caches:
            print(f"[WARN] 未找到 {args.date} 的任何最终分析缓存")
            return
        reports_dir = os.path.join(ROOT, "reports")
        live_dir = args.live_dir or reports_dir
        ok, skip = 0, 0
        for cp in caches:
            try:
                mid = str(load_module_cache(cp).get("match", {}).get("match_id", "") or "")
            except Exception as e:
                print(f"[SKIP] 无法读取 {os.path.basename(cp)}: {e}")
                skip += 1
                continue
            if not mid:
                print(f"[SKIP] 无 match_id: {os.path.basename(cp)}")
                skip += 1
                continue
            live = {}
            live_path = os.path.join(live_dir, f"live_context_{mid}.json")
            if os.path.exists(live_path):
                try:
                    with open(live_path, "r", encoding="utf-8") as f:
                        live = json.load(f)
                except Exception as e:
                    print(f"[WARN] 读取 {live_path} 失败: {e}")
            out_path = os.path.join(reports_dir, f"football_bayes_{mid}.html")
            try:
                build_report(cp, live, out_path)
                ok += 1
            except Exception as e:
                print(f"[ERR] 生成 {mid} 失败: {e}")
                skip += 1
        print(f"\n[批量完成] 日期={args.date} 成功={ok} / 跳过={skip} / 共={len(caches)}")
        return

    # 单场模式
    if not args.cache:
        ap.error("需提供 --cache（单场）或 --date（批量）或 --beidan-rec（北单）")
    live = {}
    if args.live:
        with open(args.live, "r", encoding="utf-8") as f:
            live = json.load(f)
    if not args.out:
        mid = load_module_cache(args.cache).get("match", {}).get("match_id", "match")
        args.out = os.path.join(ROOT, "reports", f"football_bayes_{mid}.html")
    build_report(args.cache, live, args.out)


if __name__ == "__main__":
    main()
