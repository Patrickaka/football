#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时数据自动化抓取管线（足球贝叶斯 skill × 现有足球模块）
========================================================

目的（回应 C 方案）：
    把 skill 要求的「联网补伤停/首发/xG/赛程」做成自动化管线，
    产出与 scripts/football_bayes_report.py 同构的 live_context.json，
    使单场深度报告可批量/半自动生成，而非每场手工拼 JSON。

合规（遵循 skill）：
- 任何数值/结论都可追溯到 tool_log（action/query/url/ts/hit）。
- 多源交叉验证：来源冲突显式标注 injury_conflict，按「更详细源优先」处理。
- 缺失即 UNAVAILABLE，绝不编造。

两种输入模式：
1) --agent-input <json>   （推荐 / 默认可用）
   agent（人或本助手）用 WebSearch/WebFetch 取得原始发现，写成:
     [{"source":"搜狐体育","query":"首尔FC 浦项制铁 伤停",
       "url":"https://...","snippet":"首尔FC主力中卫金某因伤缺阵...","ts":"2026-07-22T14:00"}, ...]
   本脚本负责解析 + 交叉验证 + 归一化 + 写 live_context.json。
   （这是 skill「人工逐场 web_search」与「自动化管线」之间的现实桥梁。）

2) --auto                  （需联网 + 可选密钥）
   用 urllib 抓取可配置端点；若设了 --api-key（或环境变量
   FOOTBALL_API_KEY）则走真实供应商（football-data.org 风格）。
   任何失败都优雅降级为 UNAVAILABLE，并记入 tool_log。

用法：
  python fetch_live_context.py --match-id 1373176 --home 首尔FC --away 浦项制铁 \
      --league "K1联赛" --date 2026-07-22 \
      --agent-input reports/_agent_findings_seoul_pohang.json \
      --out reports/live_context_1373176.json
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.football.live_context_quality import assess_live_context

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


# ------------------------- 启发式解析 -------------------------
INJURY_KW = ["伤停", "缺阵", "停赛", "受伤", "因伤", "主力伤", "suspended", "injured",
             "out for", "doubtful", "ruled out", "缺席"]
NO_INJURY_KW = ["无伤停", "无重大伤停", "没有伤停", "阵容齐整", "no injuries", "fully fit",
                "no injury concerns"]
LINEUP_KW = ["首发", "预计首发", "出场阵容", "xi", "lineup", "formation", "阵型"]
SCHEDULE_KW = ["周中", "3天2赛", "三天两赛", "赛程密集", "密集赛程", "轮换", "休整",
               "rest", "fixture congestion", "rotat"]
FORM_KW = ["近5场", "近五场", "最近5场", "近期战绩", "recent form", "last 5", "form guide"]
STYLE_KW = ["控球", "反击", "高中锋", "压迫", "阵地战", "边路", "pressing",
            "counter-attack", "counterattack", "high press", "possessive"]


def _find_team(token: str, home: str, away: str) -> Optional[str]:
    """在文本中识别主/客队提及。"""
    if home and home in token:
        return "home"
    if away and away in token:
        return "away"
    return None


def parse_injuries(findings: List[Dict], home: str, away: str) -> (List[Dict], Optional[str]):
    """从 agent findings 解析伤停，并检测来源冲突。返回 (injuries, conflict_msg)。"""
    injuries = []
    conflict = None
    team_none_assert = {}  # team -> source that says no injuries
    for f in findings:
        snip = (f.get("snippet") or "") + " " + (f.get("query") or "")
        src = f.get("source", "UNAVAILABLE")
        ts = f.get("ts", "UNAVAILABLE")
        low = snip.lower()
        # 无伤停断言
        if any(k in snip for k in NO_INJURY_KW):
            team = _find_team(snip, home, away) or "unknown"
            team_none_assert.setdefault(team, src)
        # 具体伤停条目
        for kw in INJURY_KW:
            if kw.lower() in low:
                team = _find_team(snip, home, away) or "unknown"
                # 抽取角色/球员（粗粒度）
                role = None
                for r in ["门将", "中卫", "后腰", "中锋", "边卫", "前锋", "后卫",
                          "中场", "defender", "midfielder", "forward", "goalkeeper", "striker"]:
                    if r.lower() in low:
                        role = r
                        break
                status = "停赛" if "停赛" in snip or "suspended" in low else "缺阵"
                impact = "高" if any(x in snip for x in ["主力", "关键", "key", "主力中卫", "核心"]) else "中"
                injuries.append({
                    "team": team, "player": "未具名" if "某" not in snip and "金" not in snip and "李" not in snip else "具名未核实",
                    "role": role or "未指明", "status": status, "impact": impact,
                    "source": src, "ts": ts,
                })
                break
    # 冲突检测：某队既被称「无伤停」又被列具体伤停
    teams_with_injury = {i["team"] for i in injuries if i["team"] != "unknown"}
    for team, src_none in team_none_assert.items():
        if team in teams_with_injury:
            detailed = next((i["source"] for i in injuries if i["team"] == team), "?")
            conflict = (f"{src_none} 称「{team}无伤停」，但 {detailed} 列具伤停；"
                        f"按 skill 规则以更详细源({detailed})为准并下调该队胜率，"
                        f"同时保留此冲突标注供人工复核。")
            break
    return injuries, conflict


def parse_schedule(findings: List[Dict], home: str, away: str) -> Dict:
    sd = {"source": "UNAVAILABLE", "ts": "UNAVAILABLE"}
    for f in findings:
        snip = f.get("snippet") or ""
        t = _find_team(snip, home, away)
        if any(k in snip for k in SCHEDULE_KW):
            if "休整" in snip or "rest" in snip.lower():
                val = "周中休整"
            else:
                val = "赛程密集/轮换"
            if t == "home":
                sd["home"] = val
            elif t == "away":
                sd["away"] = val
            else:
                sd["home"] = sd.get("home", val)
                sd["away"] = sd.get("away", val)
            sd["source"] = f.get("source", "UNAVAILABLE")
            sd["ts"] = f.get("ts", "UNAVAILABLE")
    return sd if (sd.get("home") or sd.get("away")) else {}


def parse_form(findings: List[Dict], home: str, away: str) -> Dict:
    form = {"source": "UNAVAILABLE", "ts": "UNAVAILABLE"}
    for f in findings:
        snip = f.get("snippet") or ""
        t = _find_team(snip, home, away)
        if any(k in snip for k in FORM_KW):
            if t == "home":
                form["home"] = snip.strip()[:60]
            elif t == "away":
                form["away"] = snip.strip()[:60]
            form["source"] = f.get("source", "UNAVAILABLE")
            form["ts"] = f.get("ts", "UNAVAILABLE")
    return form if (form.get("home") or form.get("away")) else {}


def parse_style(findings: List[Dict], home: str, away: str) -> Optional[str]:
    hits = []
    for f in findings:
        snip = f.get("snippet") or ""
        if any(k in snip for k in STYLE_KW):
            hits.append(snip.strip()[:80])
    if not hits:
        return None
    # 用首条风格描述做定性注记（不编造定量 xG）
    return "；".join(hits[:2])


# ------------------------- 自动抓取（可选） -------------------------
def auto_fetch(match_id: str, api_key: Optional[str]) -> List[Dict]:
    """尝试自动抓取；失败则返回空（调用方降级为 UNAVAILABLE）。

    设计：有 api_key 时走真实供应商；否则仅做连通性探测并记录，
    不臆造数据。所有异常被吞掉，返回 []，由主流程降级。
    """
    log = []
    if api_key:
        url = f"https://api.football-data.org/v4/matches/{match_id}"
        try:
            req = urllib.request.Request(url, headers={**UA, "X-Auth-Token": api_key})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8", "ignore"))
            log.append({"action": "api_get", "query": f"match {match_id}",
                        "url": url, "ts": datetime.now().isoformat(),
                        "hit": "ok", "raw": data})
        except Exception as e:
            log.append({"action": "api_get", "query": f"match {match_id}",
                        "url": url, "ts": datetime.now().isoformat(),
                        "hit": f"fail: {e}"})
    else:
        # 无密钥：仅记录「未配置自动源」，不发起可能违规的请求
        log.append({"action": "auto_skip", "query": f"match {match_id}",
                    "url": "-", "ts": datetime.now().isoformat(),
                    "hit": "no_api_key → 降级 UNAVAILABLE"})
    return log


# ------------------------- 组装 -------------------------
def build_context(match: Dict, findings: List[Dict], auto_log: List[Dict]) -> Dict:
    home, away = match.get("home"), match.get("away")
    injuries, conflict = parse_injuries(findings, home, away)
    sd = parse_schedule(findings, home, away)
    form = parse_form(findings, home, away)
    style = parse_style(findings, home, away)

    tool_log = list(auto_log)
    for f in findings:
        tool_log.append({
            "action": f.get("action", "web_search"),
            "query": f.get("query", ""),
            "url": f.get("url", "UNAVAILABLE"),
            "ts": f.get("ts", "UNAVAILABLE"),
            "hit": (f.get("snippet") or "")[:80],
        })

    ctx = {
        "tool_log": tool_log,
        "possession": None,   # K联赛无 FBref/Understat 定量 → UNAVAILABLE（不编造）
        "injuries": injuries,
        "lineup": {},
        "schedule_density": sd,
        "form": form,
        "style_notes": style,
    }
    if conflict:
        ctx["injury_conflict"] = conflict
    ctx["quality"] = assess_live_context(ctx)
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--league", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--agent-input", default=None, help="agent web 发现 JSON 路径")
    ap.add_argument("--auto", action="store_true", help="尝试自动抓取（需 --api-key 或环境变量）")
    ap.add_argument("--api-key", default=os.environ.get("FOOTBALL_API_KEY"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", f"live_context_{args.match_id}.json")

    findings = []
    if args.agent_input and os.path.exists(args.agent_input):
        with open(args.agent_input, "r", encoding="utf-8") as f:
            findings = json.load(f)
        print(f"[OK] 载入 agent 发现 {len(findings)} 条 @ {args.agent_input}")

    auto_log = auto_fetch(args.match_id, args.api_key if args.auto else None)
    ctx = build_context(
        {"home": args.home, "away": args.away, "league": args.league, "date": args.date},
        findings, auto_log)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)

    n_inj = len(ctx["injuries"])
    print(f"[OK] live_context 已写出: {out}")
    print(f"比赛: {args.league} {args.home} vs {args.away} ({args.match_id})")
    print(f"伤停条目: {n_inj} ｜ 赛程: {'有' if ctx['schedule_density'] else '无'} ｜ "
          f"风格注记: {'有' if ctx['style_notes'] else '无'}")
    print(f"控球陷阱定量: UNAVAILABLE（K联赛无 FBref/Understat）")
    if ctx.get("injury_conflict"):
        print(f"⚠️ 伤停冲突: {ctx['injury_conflict']}")
    print(f"工具调用凭证: {len(ctx['tool_log'])} 条")


if __name__ == "__main__":
    main()
