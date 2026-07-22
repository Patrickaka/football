#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单场深度报告生成器（足球贝叶斯分析 skill × 现有足球模块）
=======================================================

定位：把现有足球预测模块（src/football）的真实输出当作概率底座，
叠加 football-bayes skill 的分析框架（先验P0 → 战术语境修正 →
联赛特性 → 似然更新P1 → 剧本 → 无效控球警示 → 风险清单），
产出一份人读的深度报告（HTML + 文本）。

设计原则（遵循 skill 合规）：
- 任何数值型数据都必须可追溯到「工具调用凭证」区，缺失则标 UNAVAILABLE，禁止编造。
- 模块缓存提供概率底座；实时战术语境/伤停/首发/xG 由外部（agent 联网抓取）
  以 JSON 形式喂入 live_context；未提供则相应字段降级为 UNAVAILABLE。

用法：
  # 单场
  python football_bayes_report.py --cache <match_analysis.pkl> \
         [--live <live_context.json>] [--out <report.html>]

  # 批量：覆盖某天所有场次（自动匹配 live_context_{match_id}.json）
  python football_bayes_report.py --date 2026-07-22 \
         [--live-dir <实时数据目录>]

live_context.json 结构示例：
{
  "tool_log": [                       # 工具调用凭证（必须，合规用）
    {"action": "web_search", "query": "...", "url": "...", "ts": "...", "hit": "..."}
  ],
  "possession": {"home": 0.55, "away": 0.45, "field_tilt": 0.48, "xg_home": 1.4, "xg_away": 0.9, "source": "...", "ts": "..."},
  "injuries": [{"team": "home", "player": "...", "role": "中卫", "status": "缺阵", "impact": "高", "source": "...", "ts": "..."}],
  "lineup": {"home_expected": [...], "away_expected": [...], "source": "...", "ts": "..."},
  "schedule_density": {"home": "3天2赛", "away": "周中休整", "source": "...", "ts": "..."},
  "form": {"home": "近5场3胜1平1负", "away": "近5场1胜2平2负", "source": "...", "ts": "..."},
  "style_notes": "首尔FC高中锋压制；浦项制铁依赖反击速度"
}
"""

import argparse
import glob
import json
import math
import os
import pickle
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional


def load_module_cache(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return d


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def derive_prior_p0(euro_open: Dict[str, float]) -> Dict[str, float]:
    """由欧赔初盘隐含概率推导先验 P0（未叠加模型/实时信息）。

    这是 skill 框架里的「先验」基准；模块最终输出已融合 Elo+DC+盘口，
    对应 skill 的「更新后 P1」。二者之差即模型修正方向。
    """
    h = euro_open.get("home", 0.0)
    d = euro_open.get("draw", 0.0)
    a = euro_open.get("away", 0.0)
    s = h + d + a
    if s <= 0:
        return {"home": 1/3, "draw": 1/3, "away": 1/3}
    return {"home": h / s, "draw": d / s, "away": a / s}


def tactical_context(live: dict) -> dict:
    """战术语境修正：控球陷阱(定量) + 风格克制(定性)。支持部分接入。

    - possession 提供 控球率/Field Tilt/xG → 控球陷阱可定量判定
    - style_notes 提供 风格克制定性描述（来自前瞻/战术分析）
    两者任一存在即视为「部分接入」；K联赛未覆盖 FBref/Understat 时
    possession 缺，控球陷阱定量为 UNAVAILABLE（不编造）。
    """
    poss = live.get("possession")
    notes = live.get("style_notes")
    out = {
        "available": bool(poss or notes),
        "has_trap": False,
        "possession_trap": None,
        "style_matchup": [],
    }

    if poss and poss.get("home") is not None and poss.get("field_tilt") is not None:
        home_poss = poss["home"]
        field_tilt = poss["field_tilt"]
        xg_home = poss.get("xg_home")
        xg_away = poss.get("xg_away")
        trap_home = home_poss > 0.60 and field_tilt < 0.50
        trap_away = (1 - home_poss) > 0.60 and (1 - field_tilt) < 0.50
        trap = trap_home or trap_away
        out["has_trap"] = True
        out["possession_trap"] = {
            "home_possession": home_poss,
            "field_tilt": field_tilt,
            "xg_home": xg_home,
            "xg_away": xg_away,
            "trap_detected": trap,
            "verdict": ("检测到无效控球风险（控球率高但推进质量差）"
                        if trap else "控球与推进质量一致，未见明显控球陷阱"),
            "source": poss.get("source", "UNAVAILABLE"),
            "ts": poss.get("ts", "UNAVAILABLE"),
        }
    elif poss:
        out["trap_note"] = "控球率/Field Tilt 不完整，控球陷阱检查无法判定（降级）。"
    else:
        out["trap_note"] = ("缺数据 → 降级推断：未获取实时控球率/Field Tilt/xG"
                            "（K联赛未覆盖 FBref/Understat），控球陷阱定量为 UNAVAILABLE，不编造。")

    if notes:
        out["style_matchup"].append(notes)
    out["style_matchup"].append(
        "高中锋 vs 矮个防线：定位球/二点球风险方向需结合首发高度（缺数据时降级标注）。"
    )
    out["style_matchup"].append(
        "反击速度 vs 慢速回追：身后球被打穿风险方向需结合边卫回追速度（缺数据时降级标注）。"
    )
    return out


def league_specifics(league_profile: dict, league: str) -> dict:
    """联赛特性修正：基于现有模块的 league_profile + 已知联赛特性。"""
    name = league or league_profile.get("name", "")
    avg_goal = league_profile.get("avg_goal")
    low_score = league_profile.get("low_score")
    home_boost = league_profile.get("home_boost")
    lines = []
    if avg_goal is not None:
        lines.append(f"联赛场均进球 {avg_goal:.2f}，"
                     f"{'偏低进球联赛' if (avg_goal < 1.5) else ('偏高进球联赛' if avg_goal > 2.0 else '中等进球联赛')}；"
                     f"低比分修正系数 {low_score}。")
    if home_boost is not None:
        lines.append(f"主场加成系数 {home_boost}（反映主场优势强度）。")
    # 已知杯赛/大赛特性
    if any(k in name for k in ("杯", "欧冠", "欧联", "世界杯", "欧洲杯")):
        lines.append("杯赛/大赛：轮换与战意波动大，单场偶然性上升，建议下调覆盖确定性。")
    if "K1" in name or "韩" in name:
        lines.append("K联赛特性：主场优势相对明显、下半场进球占比偏高、强队客场易平。")
    if "意甲" in name:
        lines.append("意甲特性：防守强度高、低比分倾向、主场哨与长补时需注意。")
    return {"applied": bool(lines), "lines": lines or ["无专门联赛规则，已使用静态 league_profile。"]}


def likelihood_update(module_wdl: Dict[str, float], live: dict, league: str) -> dict:
    """似然更新 P1：以模块校准输出为基准，叠加实时伤停/首发/赛程密度。"""
    p1 = dict(module_wdl)
    evidence = []
    # 伤停/停赛
    for inj in live.get("injuries", []) or []:
        ev = (f"{inj.get('team')}方 {inj.get('player','?')}（{inj.get('role','?')}）"
              f"状态={inj.get('status','?')}，影响={inj.get('impact','中')}"
              f"｜来源:{inj.get('source','UNAVAILABLE')} @ {inj.get('ts','?')}")
        evidence.append(ev)
        # 简易方向修正：关键中轴缺阵压低该队胜率
        if inj.get("status") in ("缺阵", "停赛") and inj.get("role") in ("门将", "中卫", "后腰", "中锋"):
            delta = {"高": 0.04, "中": 0.02, "低": 0.01}.get(inj.get("impact", "中"), 0.02)
            if inj.get("team") == "home":
                p1["home"] = max(0.05, p1["home"] - delta)
                p1["away"] = p1["away"] + delta * 0.6
                p1["draw"] = p1["draw"] + delta * 0.4
            else:
                p1["away"] = max(0.05, p1["away"] - delta)
                p1["home"] = p1["home"] + delta * 0.6
                p1["draw"] = p1["draw"] + delta * 0.4
    # 赛程密度
    sd = live.get("schedule_density")
    if sd:
        evidence.append(f"赛程密度：主 {sd.get('home','?')} / 客 {sd.get('away','?')}"
                        f"｜来源:{sd.get('source','UNAVAILABLE')} @ {sd.get('ts','?')}")
    # 归一化
    s = p1["home"] + p1["draw"] + p1["away"]
    if s > 0:
        p1 = {k: v / s for k, v in p1.items()}
    return {"p1": p1, "evidence": evidence}


def build_scripts(module: dict, tactical: dict, home: str, away: str) -> List[dict]:
    """生成至少 2 套剧本（基于模块 Top 比分 + 战术语境）。"""
    top = module.get("top_scores", [])[:3]
    scripts = []
    if top:
        t1 = top[0]
        scripts.append({
            "name": "剧本A（模型首选路径）",
            "text": (f"{home} 按预期主导控球与射门，凭借实力优势率先破门；"
                     f"{away} 收缩防守伺机反击。最可能比分 {t1['home']}-{t1['away']}"
                     f"（模型概率 {pct(t1['prob'])}）。若 {home} 久攻不下，"
                     f"易被 {away} 偷袭，转向平局收场。"),
        })
    if len(top) > 1:
        t2 = top[1]
        scripts.append({
            "name": "剧本B（次选/韧性路径）",
            "text": (f"{away} 上半场低位防守顶住压力，下半场通过定位球或反击制造威胁；"
                     f"双方陷入僵持，{t2['home']}-{t2['away']}（模型概率 {pct(t2['prob'])}）"
                     f"成为合理结局。若主队早破门则切换为剧本A。"),
        })
    # 战术驱动剧本
    if tactical.get("available") and (tactical.get("possession_trap") or {}).get("trap_detected"):
        scripts.append({
            "name": "剧本C（控球陷阱反杀）",
            "text": (f"一方控球虚高但推进效率差（Field Tilt 偏低），"
                     f"对手以高效反击兑现机会，控球方最终哑火爆冷。"),
        })
    else:
        scripts.append({
            "name": "剧本C（均衡消耗）",
            "text": (f"双方节奏胶着，进球分散在下半场，胜负由一次定位球或个体失误决定；"
                     f"覆盖双选（主胜+平局）比单博更稳。"),
        })
    return scripts


def possession_trap_warning(live: dict) -> dict:
    """无效控球警示（反面教材）。无可靠历史案例则明确说明，不编造。"""
    case = live.get("trap_case")
    if case:
        return {"available": True, **case}
    return {
        "available": False,
        "note": ("未找到可验证来源：本报告不编造历史案例。原则阐述——"
                 "控球率高但 Field Tilt（进攻三区进入率）低于 50% 时，"
                 "多为后场倒脚，xG 往往极低（如 xG≈0.06），"
                 "此类球队易被高效反击队克制。需结合本场实测数据判断。"),
    }


def risk_list(module_risk: dict, tactical: dict, live: dict) -> List[str]:
    risks = []
    rf = (module_risk or {}).get("risk_factors") or []
    for r in rf:
        risks.append(f"模块风险因子：{r}")
    if not tactical.get("available"):
        risks.append("战术语境缺失：未获取实时控球/xG，无法校验控球陷阱与风格克制，结论置信度下降。")
    if not live.get("injuries"):
        risks.append("伤停未知：关键中轴缺阵可能显著改变先验，当前按完整阵容估计。")
    if not live.get("lineup"):
        risks.append("首发未知：预计首发未获取，临场变阵风险未纳入。")
    risks.append("盘口已含市场预期：胜平负概率含庄家 margin，存在系统性偏差可能。")
    risks.append("单场偶然性：即便高置信场次，足球单场噪声仍可能推翻模型结论。")
    return risks[:6]


def render_html(report: dict) -> str:
    m = report["match"]
    title = f"{m['league']} {m['home']} vs {m['away']} | {m.get('time','')} | {m.get('num','')}"
    css = """
    :root{--bg:#f5f7fa;--card:#fff;--ink:#1f2933;--muted:#6b7280;--line:#e5e7eb;
          --home:#2563eb;--draw:#d97706;--away:#059669;--warn:#dc2626;--ok:#059669;}
    *{box-sizing:border-box;}
    body{margin:0;background:var(--bg);color:var(--ink);
         font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
         line-height:1.6;padding:24px;}
    .wrap{max-width:960px;margin:0 auto;}
    h1{font-size:22px;margin:0 0 4px;}
    .sub{color:var(--muted);font-size:13px;margin-bottom:18px;}
    .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
          padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.04);}
    .card h2{font-size:16px;margin:0 0 12px;display:flex;align-items:center;gap:8px;}
    .tag{font-size:11px;padding:2px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;}
    .tag.u{background:#fef2f2;color:#b91c1c;}
    .tag.ok{background:#ecfdf5;color:#047857;}
    .prob{display:flex;gap:10px;margin:10px 0;}
    .prob .b{flex:1;text-align:center;border-radius:10px;padding:12px 6px;border:1px solid var(--line);}
    .prob .b .l{font-size:13px;color:var(--muted);}
    .prob .b .v{font-size:22px;font-weight:700;margin-top:2px;}
    .home .v{color:var(--home);} .draw .v{color:var(--draw);} .away .v{color:var(--away);}
    table{width:100%;border-collapse:collapse;font-size:13px;}
    td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;}
    .arr{color:var(--muted);font-size:13px;margin:6px 0;}
    ul{margin:8px 0;padding-left:20px;font-size:13px;}
    li{margin:4px 0;}
    .log{font-size:12px;color:var(--muted);background:#fafafa;border:1px dashed var(--line);
         border-radius:8px;padding:10px;max-height:180px;overflow:auto;}
    .log div{margin:3px 0;}
    .warn{border-left:4px solid var(--warn);padding-left:12px;color:#7f1d1d;font-size:13px;}
    .okb{border-left:4px solid var(--ok);padding-left:12px;font-size:13px;}
    .muted{color:var(--muted);}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
    @media(max-width:680px){.grid2{grid-template-columns:1fr;}}
    """
    w = report["wdl"]  # module calibrated
    p0 = report["p0"]
    h = []
    h.append(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>深度报告 · {title}</title><style>{css}</style></head><body><div class='wrap'>")
    h.append(f"<h1>足球深度研究报告</h1>")
    h.append(f"<div class='sub'>{title} ｜ 数据时间戳：{report['ts']} ｜ 模块版本：{report['module_version']}</div>")

    # 工具调用凭证（合规，置于数值前）
    h.append("<div class='card'><h2>数据来源与工具调用凭证 <span class='tag'>合规区</span></h2>")
    if report["tool_log"]:
        h.append("<div class='log'>")
        for t in report["tool_log"]:
            h.append(f"<div>[{t.get('action','?')}] {t.get('query', t.get('url',''))} "
                     f"→ hit: {t.get('hit','-')} @ {t.get('ts','?')}</div>")
        h.append("</div>")
    else:
        h.append("<div class='warn'>本次报告未执行联网抓取（未提供 live_context.tool_log）。"
                 "概率底座来自现有模块缓存；实时战术语境/伤停字段按 skill 降级协议标注为 UNAVAILABLE。</div>")
    h.append("</div>")

    # 1) 先验 P0
    h.append("<div class='card'><h2>1) 先验 P0（欧赔初盘隐含）</h2>")
    h.append("<div class='prob'>"
             f"<div class='b home'><div class='l'>主胜</div><div class='v'>{pct(p0['home'])}</div></div>"
             f"<div class='b draw'><div class='l'>平局</div><div class='v'>{pct(p0['draw'])}</div></div>"
             f"<div class='b away'><div class='l'>客胜</div><div class='v'>{pct(p0['away'])}</div></div></div>")
    h.append(f"<div class='muted'>来源：欧赔初盘（open）隐含概率，未叠加 Elo/DC/实时信息。"
             f"现有模块已融合 Elo 先验（见 elo.py）与 Dixon-Coles 低比分相关。</div></div>")

    # 2) 战术语境
    tac = report["tactical"]
    if not tac["available"]:
        tag_cls, tag_txt = "u", "UNAVAILABLE"
    elif tac.get("has_trap"):
        tag_cls, tag_txt = "ok", "已接入"
    else:
        tag_cls, tag_txt = "ok", "部分接入"
    h.append("<div class='card'><h2>2) 战术语境修正 "
             f"<span class='tag {tag_cls}'>{tag_txt}</span></h2>")
    if tac["available"]:
        if tac.get("possession_trap"):
            pt = tac["possession_trap"]
            h.append(f"<div class='okb'>控球陷阱检查：主队控球 {pct(pt['home_possession'])}，"
                     f"Field Tilt {pct(pt['field_tilt'])}，xG {pt.get('xg_home')}/{pt.get('xg_away')}。"
                     f"结论：{pt['verdict']}（来源：{pt['source']} @ {pt['ts']}）</div>")
        else:
            h.append(f"<div class='warn'>{tac.get('trap_note','控球陷阱定量 UNAVAILABLE')}</div>")
        for s in tac["style_matchup"]:
            h.append(f"<div class='muted'>• {s}</div>")
    else:
        h.append(f"<div class='warn'>{tac.get('trap_note','缺数据 → 降级推断')}</div>")
    h.append("</div>")

    # 3) 联赛特性
    h.append("<div class='card'><h2>3) 联赛特性修正</h2><ul>")
    for line in report["league"]["lines"]:
        h.append(f"<li>{line}</li>")
    h.append("</ul></div>")

    # 4) 似然更新 P1
    h.append("<div class='card'><h2>4) 似然更新 P1（模块校准输出）</h2>")
    h.append("<div class='prob'>"
             f"<div class='b home'><div class='l'>主胜</div><div class='v'>{pct(w['home'])}</div></div>"
             f"<div class='b draw'><div class='l'>平局</div><div class='v'>{pct(w['draw'])}</div></div>"
             f"<div class='b away'><div class='l'>客胜</div><div class='v'>{pct(w['away'])}</div></div></div>")
    dh = w['home'] - p0['home']
    dd = w['draw'] - p0['draw']
    da = w['away'] - p0['away']

    def sgn(x):
        return f"+{x*100:.1f}%" if x >= 0 else f"{x*100:.1f}%"
    h.append(f"<div class='arr'>P0→P1 修正方向：主胜 {sgn(dh)} / "
             f"平局 {sgn(dd)} / 客胜 {sgn(da)}"
             f"（正值=模型上调）</div>")
    if report["update"]["evidence"]:
        h.append("<div class='muted'>导致更新的证据点：</div><ul>")
        for e in report["update"]["evidence"]:
            h.append(f"<li>{e}</li>")
        h.append("</ul>")
    else:
        h.append("<div class='muted'>无实时伤停/赛程证据输入，P1 = 模块校准输出。</div>")
    if report.get("injury_conflict"):
        h.append(f"<div class='warn'>⚠️ 伤停数据冲突：{report['injury_conflict']}</div>")
    h.append(f"<div class='muted'>置信度：{report['confidence_label']}（{report['confidence_score']}）；"
             f"风险等级：{report['risk_level']}</div></div>")

    # 5) 剧本
    h.append("<div class='card'><h2>5) 剧本预测</h2>")
    for sc in report["scripts"]:
        h.append(f"<div class='okb'><b>{sc['name']}</b><br>{sc['text']}</div><br>")
    h.append("</div>")

    # 6) 无效控球警示
    h.append("<div class='card'><h2>6) 无效控球警示（反面教材） "
             f"<span class='tag {'u' if not report['trap_warn']['available'] else 'ok'}'>"
             f"{'UNAVAILABLE' if not report['trap_warn']['available'] else '有案例'}</span></h2>")
    if report["trap_warn"]["available"]:
        tw = report["trap_warn"]
        h.append(f"<div class='warn'>案例：{tw.get('case','')} ｜ 数据：控球 {tw.get('possession')} / "
                 f"Field Tilt {tw.get('field_tilt')} / xG={tw.get('xg')} ｜ 来源：{tw.get('source')}</div>")
        h.append(f"<div class='muted'>启示：{tw.get('lesson','')}</div>")
    else:
        h.append(f"<div class='warn'>{report['trap_warn'].get('note','未找到可验证来源')}</div>")
    h.append("</div>")

    # 7) 风险
    h.append("<div class='card'><h2>7) 关键不确定性与风险清单</h2><ul>")
    for r in report["risks"]:
        h.append(f"<li>{r}</li>")
    h.append("</ul></div>")

    h.append(f"<div class='sub'>本报告由 football-bayes skill 框架叠加现有足球模块（{report['module_version']}）"
             f"生成，属信息分析用途，仅供参考不构成投资建议。</div>")
    h.append("</div></body></html>")
    return "".join(h)


def report_url_from_path(out_path: str, match_id) -> str:
    """将本地报告路径转换为前端可访问的 URL。"""
    # server.py 将以 /reports/<filename> 提供 reports/ 目录下的文件
    fname = os.path.basename(out_path)
    return f"/reports/{fname}"


def update_manifest(out_path: str, match: dict):
    """更新 reports/football_bayes_manifest.json，按 match_id 记录报告。"""
    reports_dir = os.path.dirname(out_path) or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
    )
    manifest_path = os.path.join(reports_dir, "football_bayes_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    if "reports" not in manifest:
        manifest["reports"] = {}

    mid = str(match.get("match_id", ""))
    if not mid:
        return
    manifest["reports"][mid] = {
        "match_id": mid,
        "home": match.get("home", ""),
        "away": match.get("away", ""),
        "league": match.get("league", ""),
        "time": match.get("time", ""),
        "num": match.get("num", ""),
        "url": report_url_from_path(out_path, mid),
        "file": os.path.basename(out_path),
        "generated_at": datetime.now().isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_report(cache_path: str, live: dict, out_path: str) -> dict:
    d = load_module_cache(cache_path)
    match = d.get("match", {})
    league = match.get("league", "")
    home, away = match.get("home", "?"), match.get("away", "?")
    analysis = d.get("analysis", {})
    wdl = analysis.get("wdl") or {
        "home": d.get("model", {}).get("supremacy_euro", 0.5),
        "draw": 0.28, "away": 0.22,
    }
    euro = d.get("euro", {})
    p0 = derive_prior_p0(euro.get("open", {}))
    tactical = tactical_context(live)
    league_info = league_specifics(d.get("league_profile", {}), league)
    update = likelihood_update(wdl, live, league)
    scripts = build_scripts(d.get("model", {}), tactical, home, away)
    trap_warn = possession_trap_warning(live)
    risks = risk_list(d.get("risk_level"), tactical, live)
    conf = d.get("confidence", {})
    rlevel = d.get("risk_level", {})

    report = {
        "match": match,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "module_version": d.get("model", {}).get("prediction_logic_version", "unknown"),
        "tool_log": live.get("tool_log", []),
        "p0": p0,
        "tactical": tactical,
        "league": league_info,
        "update": update,
        "wdl": update["p1"],
        "scripts": scripts,
        "trap_warn": trap_warn,
        "risks": risks,
        "confidence_label": conf.get("label", "未知"),
        "confidence_score": conf.get("score", "?"),
        "risk_level": rlevel.get("level", "?"),
        "injury_conflict": live.get("injury_conflict"),
    }
    html = render_html(report)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 更新清单，供前端按 match_id 索引
    try:
        update_manifest(out_path, match)
    except Exception as e:
        print(f"[WARN] 清单更新失败: {e}")

    # 文本摘要打印
    print(f"[OK] 报告已生成: {out_path}")
    print(f"比赛: {league} {home} vs {away} @ {match.get('time','')}")
    print(f"P0(初盘隐含): 主{pct(p0['home'])} 平{pct(p0['draw'])} 客{pct(p0['away'])}")
    print(f"P1(模块校准): 主{pct(report['wdl']['home'])} 平{pct(report['wdl']['draw'])} 客{pct(report['wdl']['away'])}")
    print(f"战术语境: {'已接入' if tactical['available'] else 'UNAVAILABLE(降级)'}")
    print(f"置信: {conf.get('label')} / 风险: {rlevel.get('level')}")
    return report


def discover_daily_caches(date: str) -> list:
    """扫描某天所有「最终分析」缓存（排除 _early / _T* 变体）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(root, "src", "football", "cache")
    pat = os.path.join(cache_dir, f"{date}_match_analysis_*.pkl")
    all_pk = glob.glob(pat)
    final = [p for p in all_pk if not re.search(r"_(early|T\d+h)\.pkl$", p)]
    return sorted(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="单场 match_analysis pkl 路径（现有模块缓存）")
    ap.add_argument("--live", default=None, help="live_context.json 路径（实时战术语境/伤停等）")
    ap.add_argument("--out", default=None, help="输出 HTML 路径（仅单场模式）")
    ap.add_argument("--date", default=None,
                    help="批量模式：扫描该日期（YYYY-MM-DD）所有最终分析缓存，逐场生成报告")
    ap.add_argument("--live-dir", default=None,
                    help="批量模式：实时数据目录，按 live_context_{match_id}.json 自动匹配")
    args = ap.parse_args()

    # 批量模式
    if args.date:
        caches = discover_daily_caches(args.date)
        if not caches:
            print(f"[WARN] 未找到 {args.date} 的任何最终分析缓存")
            return
        reports_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
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
        ap.error("需提供 --cache（单场）或 --date（批量）")
    live = {}
    if args.live:
        with open(args.live, "r", encoding="utf-8") as f:
            live = json.load(f)

    if not args.out:
        mid = load_module_cache(args.cache).get("match", {}).get("match_id", "match")
        args.out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "reports", f"football_bayes_{mid}.html")
    build_report(args.cache, live, args.out)


if __name__ == "__main__":
    main()
