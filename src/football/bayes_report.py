#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单场深度报告生成模块（足球贝叶斯分析 skill × 现有足球/北单模块）
==============================================================

把现有预测模块（src/football、src/beidan）的真实输出当作概率底座，
叠加 football-bayes skill 的分析框架（先验P0 → 战术语境修正 →
联赛特性 → 似然更新P1 → 剧本 → 无效控球警示 → 风险清单），
产出一份人读的深度报告（HTML）。

本模块同时服务于两条链路：
1. 脚本链路（scripts/football_bayes_report.py、scripts/gen_beidan_reports.py）
   用于批量/手动预生成报告文件。
2. 服务端按需生成（server.py 调用 ensure_football_report /
   ensure_beidan_report）——当用户点击「深度报告」按钮而报告文件尚
   不存在时，由服务器现抓数据现生成，无需在生产机手动跑脚本。

设计原则（遵循 skill 合规）：
- 任何数值型数据都必须可追溯到「工具调用凭证」区，缺失则标 UNAVAILABLE，禁止编造。
- 模块缓存提供概率底座；实时战术语境/伤停/首发/xG 由外部（agent 联网抓取）
  以 JSON 形式喂入 live_context；未提供则相应字段降级为 UNAVAILABLE。
"""

import glob
import json
import math
import os
import pickle
import re
import time
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

# ----- 路径解析（本文件位于 src/football/）-----
from ..domain.sports.football.reporting import (  # noqa: F401
    DRIFT_THRESHOLD,
    REPORT_SCHEMA_VERSION,
    _REPORT_CSS,
    _beidan_p0_p1,
    _beidan_scripts,
    _extract_mid_from_report_path,
    _to_implied,
    build_scripts,
    derive_prior_p0,
    league_specifics,
    likelihood_update,
    pct,
    possession_trap_warning,
    render_beidan_html,
    render_html,
    report_url_from_path,
    risk_list,
    tactical_context,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(MODULE_DIR))
DEFAULT_REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
FOOTBALL_CACHE_DIR = os.path.join(REPO_ROOT, "src", "football", "cache")
BEIDAN_CACHE_DIR = os.path.join(REPO_ROOT, "src", "beidan", "cache")
MANIFEST_PATH = os.path.join(DEFAULT_REPORTS_DIR, "football_bayes_manifest.json")

# 防止并发请求同时生成同一报告
REPORT_GEN_LOCK = Lock()
_PRO_VALIDATION_CACHE = {"mtime": None, "value": {}, "checked_at": 0.0}


# ===================== 通用工具 =====================

def load_module_cache(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return d


def load_professional_validation_summary() -> dict:
    """Load the lightweight strict-OOS summary without rerunning a backtest."""
    from .professional_baseline import (
        BASELINE_GENERATED_AT,
        BASELINE_VERSION,
        bundled_professional_baseline,
    )
    path = os.path.join(DEFAULT_REPORTS_DIR, "professional_football_backtest.json")
    try:
        report_exists = os.path.exists(path)
        if (
            not report_exists
            and _PRO_VALIDATION_CACHE["value"]
            and time.time() - _PRO_VALIDATION_CACHE["checked_at"] < 60
        ):
            return dict(_PRO_VALIDATION_CACHE["value"])
        mtime = os.path.getmtime(path) if report_exists else None
        if _PRO_VALIDATION_CACHE["value"] and _PRO_VALIDATION_CACHE["mtime"] == mtime:
            return dict(_PRO_VALIDATION_CACHE["value"])
        if report_exists:
            with open(path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            generated_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            source = "runtime_report"
        else:
            from ..common import kv_store
            database_report, database_backend = kv_store.load_with_backend(
                "football_professional_validation",
            )
            if isinstance(database_report, dict) and database_report.get("model_metrics"):
                report = database_report
                generated_at = report.get("generated_at") or "database"
                source = (
                    "database_kv_store" if database_backend == "mysql"
                    else "local_kv_fallback"
                )
                mtime = f"{database_backend}:{generated_at}"
            else:
                report = bundled_professional_baseline()
                generated_at = BASELINE_GENERATED_AT
                source = "bundled_audited_baseline"
                mtime = "bundled"
        model = report.get("model_metrics") or {}
        market = report.get("market_baseline_metrics") or {}
        strategy = report.get("strategy") or {}
        checks = {
            "model_beats_market": (
                float(model.get("logloss", 99)) < float(market.get("logloss", 99))
            ),
            "positive_roi": float(strategy.get("roi", 0) or 0) > 0,
            "positive_clv": float(strategy.get("mean_clv", 0) or 0) > 0,
            "enough_samples": int(report.get("out_of_sample_n", 0) or 0) >= 1000,
            "enough_strategy_bets": int(strategy.get("bets", 0) or 0) >= 100,
        }
        value = {
            "available": True,
            "out_of_sample_n": report.get("out_of_sample_n", 0),
            "model": model,
            "market": market,
            "strategy": strategy,
            "checks": checks,
            "production_ready": all(checks.values()),
            "generated_at": generated_at,
            "source": source,
            "baseline_version": BASELINE_VERSION,
        }
        _PRO_VALIDATION_CACHE.update({
            "mtime": mtime, "value": value, "checked_at": time.time(),
        })
        return dict(value)
    except Exception as exc:
        return {"available": False, "production_ready": False, "reason": "internal_error",
                "error_type": type(exc).__name__}


















# ===================== 足球渲染 =====================







def update_manifest(out_path: str, match: dict):
    """更新 reports/football_bayes_manifest.json，按 match_id 记录报告。"""
    reports_dir = os.path.dirname(out_path) or DEFAULT_REPORTS_DIR
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
    from .live_context_quality import assess_live_context
    from .accuracy_gate import build_accuracy_gate

    live = dict(live or {})
    live["quality"] = assess_live_context(live)
    professional_validation = load_professional_validation_summary()
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
    accuracy_gate = (d.get("lottery") or {}).get("accuracy_gate")
    if not accuracy_gate:
        accuracy_gate = build_accuracy_gate(
            d.get("lottery") or {},
            confidence=conf,
            anomaly=d.get("anomaly") or {},
            league=league,
        )
    professional_evidence = d.get("professional_evidence")
    from .professional_readiness import build_match_evidence_profile
    evidence_input = dict(d)
    evidence_input["live_context"] = live
    # Always rebuild here: report-only live_context may be newer than the
    # prediction cache and must participate in the evidence audit.
    professional_evidence = build_match_evidence_profile(evidence_input)

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
        "live_context_quality": live.get("quality"),
        "professional_validation": professional_validation,
        "accuracy_gate": accuracy_gate,
        "professional_evidence": professional_evidence,
        "match_evidence": {
            "asian_handicap": (d.get("asian") or {}).get("handicap"),
            "asian_trend": (d.get("asian") or {}).get("handicap_trend"),
            "total_line": (
                (d.get("total") or {}).get("close_line")
                or (d.get("total") or {}).get("line")
            ),
            "euro_close": (d.get("euro") or {}).get("close") or {},
            "lottery_handicap": (d.get("match") or {}).get("lottery_handicap"),
        },
        "decision_gate": {
            "official_bet_allowed": (
                live.get("quality", {}).get("official_bet_allowed") is True
                and professional_validation.get("production_ready") is True
            ),
            "live_context_passed": live.get("quality", {}).get("official_bet_allowed") is True,
            "professional_validation_passed": professional_validation.get("production_ready") is True,
        },
    }
    html = render_html(report)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    try:
        update_manifest(out_path, match)
    except Exception as e:
        print(f"[WARN] 清单更新失败: {e}")

    print(f"[OK] 报告已生成: {out_path}")
    print(f"比赛: {league} {home} vs {away} @ {match.get('time','')}")
    print(f"P0(初盘隐含): 主{pct(p0['home'])} 平{pct(p0['draw'])} 客{pct(p0['away'])}")
    print(f"P1(模块校准): 主{pct(report['wdl']['home'])} 平{pct(report['wdl']['draw'])} 客{pct(report['wdl']['away'])}")
    print(f"战术语境: {'已接入' if tactical['available'] else 'UNAVAILABLE(降级)'}")
    print(f"置信: {conf.get('label')} / 风险: {rlevel.get('level')}")

    # 保存赔率快照（生成时刻的盘口），供后续变盘检测决定是否重生成
    mid = _extract_mid_from_report_path(out_path)
    if mid:
        euro = d.get("euro", {})
        save_odds_snapshot(mid, "football", euro.get("instant") or euro.get("open") or {})
    return report


# ===================== 北单（beidan）深度报告 =====================





def build_beidan_report(rec: dict, live: dict, out_path: str) -> dict:
    match = {
        "match_id": rec.get("match_id"),
        "home": rec.get("home"),
        "away": rec.get("away"),
        "league": rec.get("league", ""),
        "num": rec.get("num", ""),
        "time": rec.get("time", ""),
    }
    league = match["league"]
    home, away = match["home"], match["away"]
    spf = rec.get("spf") or {}
    p0, p1_init = _beidan_p0_p1(rec)
    tactical = tactical_context(live)
    league_info = league_specifics({}, league)
    update = likelihood_update(p1_init, live, league)
    scripts = _beidan_scripts(rec, tactical, home, away)
    trap_warn = possession_trap_warning(live)
    risks = risk_list({}, tactical, live)
    q = spf.get("quality") or {}
    conf_label = q.get("label") or ("高" if float(spf.get("confidence", 0) or 0) >= 0.7 else ("中" if float(spf.get("confidence", 0) or 0) >= 0.5 else "低"))
    conf_score = q.get("score", spf.get("confidence", "?"))
    risk_level = q.get("level", "中")

    report = {
        "match": match,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "module_version": "beidan",
        "tool_log": live.get("tool_log", []),
        "p0": p0,
        "tactical": tactical,
        "league": league_info,
        "update": update,
        "wdl": update["p1"],
        "scripts": scripts,
        "trap_warn": trap_warn,
        "risks": risks,
        "confidence_label": conf_label,
        "confidence_score": conf_score,
        "risk_level": risk_level,
        "injury_conflict": live.get("injury_conflict"),
        "beidan": {
            "spf": spf,
            "rqspf": rec.get("rqspf") or {},
            "zjq": rec.get("zjq") or {},
            "upset": rec.get("upset") or spf.get("upset") or {},
            "asian_trend": spf.get("asian_trend"),
        },
    }
    html = render_beidan_html(report)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    try:
        update_manifest(out_path, match)
    except Exception as e:
        print(f"[WARN] 清单更新失败: {e}")

    print(f"[OK] 北单报告已生成: {out_path}")
    print(f"比赛: {league} {home} vs {away} @ {match.get('time','')}")
    print(f"P0(赔率隐含): 主{pct(p0['home'])} 平{pct(p0['draw'])} 客{pct(p0['away'])}")
    print(f"P1(模型校准): 主{pct(report['wdl']['home'])} 平{pct(report['wdl']['draw'])} 客{pct(report['wdl']['away'])}")

    # 保存赔率快照（生成时刻的赔率），供后续变盘检测决定是否重生成
    mid = rec.get("match_id") or _extract_mid_from_report_path(out_path)
    if mid:
        save_odds_snapshot(mid, "beidan", (rec.get("spf") or {}).get("odds") or {})
    print(f"战术语境: {'已接入' if tactical['available'] else 'UNAVAILABLE(降级)'}")
    return report




# ===================== 批量扫描（scripts 用） =====================

def discover_daily_caches(date: str) -> list:
    """扫描某天所有「最终分析」缓存（排除 _early / _T* 变体）。"""
    pat = os.path.join(FOOTBALL_CACHE_DIR, f"{date}_match_analysis_*.pkl")
    all_pk = glob.glob(pat)
    final = [p for p in all_pk if not re.search(r"_(early|T\d+h)\.pkl$", p)]
    return sorted(final)


# ===================== 服务端按需生成接口 =====================

# 索引缓存有效期：超过此时间则强制重扫缓存目录，避免新生成的 pkl 长时间不可见。
_INDEX_TTL = 10 * 60  # 10 分钟


def _football_cache_index(force: bool = False) -> Dict[str, str]:
    """建立 match_id -> 缓存 pkl 文件名的索引（缓存到 JSON，避免每次全量扫描）。

    返回 dict: { match_id(str): pkl_filename(str) }

    为避免「索引一次性缓存、永不再刷新」导致新分析出的 pkl 长期不被识别
    （表现为部分比赛始终无法生成深度报告），这里引入 TTL：索引文件 mtime
    超过 _INDEX_TTL 时强制重扫。force=True 立即重扫。
    """
    idx_path = os.path.join(FOOTBALL_CACHE_DIR, "_report_index.json")
    if not force and os.path.exists(idx_path):
        try:
            if (time.time() - os.path.getmtime(idx_path)) < _INDEX_TTL:
                with open(idx_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
    idx: Dict[str, str] = {}
    if os.path.isdir(FOOTBALL_CACHE_DIR):
        for p in glob.glob(os.path.join(FOOTBALL_CACHE_DIR, "*match_analysis*.pkl")):
            if re.search(r"_(early|T\d+h)\.pkl$", p):
                continue
            try:
                with open(p, "rb") as f:
                    d = pickle.load(f)
                mid = str(d.get("match", {}).get("match_id", "") or "")
                if mid:
                    idx[mid] = os.path.basename(p)
            except Exception:
                # 个别 pkl 损坏或需缺失依赖（如未装 numpy）时跳过，不阻塞整体
                continue
    try:
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False)
    except Exception:
        pass
    return idx


def football_reportable_ids() -> set:
    """返回当前缓存目录中可生成报告的足球 match_id 集合。"""
    return set(_football_cache_index().keys())


def refresh_football_cache_index() -> set:
    """强制重扫缓存目录并刷新索引，返回最新可生成报告的 match_id 集合。

    供后台「自动补分析」流程在生成新 pkl 后调用，使按钮可见性及时更新。
    """
    return set(_football_cache_index(force=True).keys())


def load_live_context(mid: str) -> dict:
    """读取 reports/live_context_{mid}.json（若存在）。"""
    lp = os.path.join(DEFAULT_REPORTS_DIR, f"live_context_{mid}.json")
    if os.path.exists(lp):
        try:
            with open(lp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def ensure_football_report(mid: str, force: bool = False) -> Optional[str]:
    """确保某足球比赛的深度报告存在；若不存在则现生成；若已存在但赔率变盘明显则重生成。

    force=True 时无论是否存在、是否变盘都强制重生成。
    返回报告文件绝对路径；若无法生成（无缓存）返回 None。

    注：pkl 文件名后缀是哈希而非 match_id，无法按 id 直接 glob，故以索引
    （match_id -> pkl 文件名）定位。索引引入 TTL 自动刷新；若索引未命中
    （如刚分析完、索引尚滞后），这里强制重扫一次再试，保证自愈。
    """
    mid = str(mid)
    out_path = os.path.join(DEFAULT_REPORTS_DIR, f"football_bayes_{mid}.html")
    exists = os.path.exists(out_path)
    if exists and not force:
        try:
            with open(out_path, "r", encoding="utf-8") as existing_report:
                head = existing_report.read(512)
            if f"data-report-schema='{REPORT_SCHEMA_VERSION}'" not in head:
                os.remove(out_path)
                exists = False
        except OSError:
            exists = False
    if exists and not force:
        # 变盘检测：当前 pkl 的盘口与生成时快照偏差明显则重生成
        try:
            pkl = _lookup_pkl(mid)
            if pkl:
                d = load_module_cache(pkl)
                euro = d.get("euro", {})
                cur = euro.get("instant") or euro.get("open") or {}
                if odds_drifted(mid, "football", cur):
                    print(f"[INFO] 足球 {mid} 检测到变盘，重生成报告")
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                else:
                    return out_path
        except Exception:
            return out_path
    with REPORT_GEN_LOCK:
        # double-check after acquiring lock
        if os.path.exists(out_path) and not force:
            return out_path
        if force and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        pkl = _lookup_pkl(mid)
        if not pkl:
            return None
        try:
            build_report(pkl, load_live_context(mid), out_path)  # 内部存快照
            return out_path
        except Exception as e:
            print(f"[ERR] 足球报告生成失败 {mid}: {e}")
            return None


def _lookup_pkl(mid: str):
    """按 match_id 从索引定位 pkl 文件路径；索引未命中时强制重扫一次再试。

    因 pkl 文件名是哈希、不含 match_id，只能依赖索引。返回 pkl 绝对路径或 None。
    """
    mid = str(mid)
    idx = _football_cache_index()
    fn = idx.get(mid)
    if not fn:
        # 索引可能滞后（如刚跑完 analyze_match），强制刷新后重试
        fn = _football_cache_index(force=True).get(mid)
    if not fn:
        return None
    return os.path.join(FOOTBALL_CACHE_DIR, fn)


def ensure_beidan_report(mid: str, force: bool = False) -> Optional[str]:
    """确保某北单比赛的深度报告存在；若不存在则现生成；若已存在但赔率变盘明显则重生成。

    force=True 时无论是否存在、是否变盘都强制重生成。
    返回报告文件绝对路径；若无法生成（无 rec 缓存）返回 None。
    """
    mid = str(mid)
    out_path = os.path.join(DEFAULT_REPORTS_DIR, f"beidan_bayes_{mid}.html")
    exists = os.path.exists(out_path)
    if exists and not force:
        # 变盘检测：当前 rec 的赔率与生成时快照偏差明显则重生成
        rec_path = os.path.join(BEIDAN_CACHE_DIR, f"beidan_{mid}.json")
        if not os.path.exists(rec_path):
            return out_path  # 无 rec 可对比，保留旧报告
        try:
            with open(rec_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            odds = (rec.get("spf") or {}).get("odds") or {}
            if odds_drifted(mid, "beidan", odds):
                print(f"[INFO] 北单 {mid} 检测到变盘，重生成报告")
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            else:
                return out_path
        except Exception:
            return out_path
    with REPORT_GEN_LOCK:
        if os.path.exists(out_path):
            return out_path
        rec_path = os.path.join(BEIDAN_CACHE_DIR, f"beidan_{mid}.json")
        if not os.path.exists(rec_path):
            return None
        try:
            with open(rec_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"[ERR] 读取北单 rec 失败 {mid}: {e}")
            return None
        rec["match_id"] = mid
        live = load_live_context(mid)
        try:
            build_beidan_report(rec, live, out_path)  # 内部存快照
            return out_path
        except Exception as e:
            print(f"[ERR] 北单报告生成失败 {mid}: {e}")
            return None


def persist_beidan_recs(recs: List[dict]) -> List[str]:
    """把北单推荐 rec 落盘到 src/beidan/cache/beidan_{mid}.json，便于后续按需生成。

    返回成功持久化的 match_id 列表。
    """
    os.makedirs(BEIDAN_CACHE_DIR, exist_ok=True)
    persisted = []
    for rec in recs:
        mid = str(rec.get("match_id") or (rec.get("spf") or {}).get("match_id", "") or "")
        if not mid:
            continue
        rec["match_id"] = mid
        try:
            with open(os.path.join(BEIDAN_CACHE_DIR, f"beidan_{mid}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, default=str)
            persisted.append(mid)
        except Exception as e:
            print(f"[WARN] 北单 rec 落盘失败 {mid}: {e}")
    return persisted


# ===================== 赔率快照 & 变盘检测（自动重生成） =====================

# 变盘阈值：胜平负隐含概率任一方绝对偏差 >= 此值即视为「变盘明显」，需重生成报告。


def _snapshot_path(mid: str, kind: str) -> str:
    return os.path.join(DEFAULT_REPORTS_DIR, f"_snap_{kind}_{mid}.json")


def save_odds_snapshot(mid: str, kind: str, odds: dict):
    """保存生成报告时刻的赔率快照。

    kind='football': odds 为 {home,draw,away} 隐含概率（来自 pkl euro）；
    kind='beidan':   odds 为 {胜,平,负} 十进制赔率（来自 rec.spf.odds）。
    以原始值存储，比较时再归一化为隐含概率，避免双重换算误差。
    """
    try:
        os.makedirs(DEFAULT_REPORTS_DIR, exist_ok=True)
        with open(_snapshot_path(mid, kind), "w", encoding="utf-8") as f:
            json.dump({
                "mid": mid, "kind": kind, "odds": odds,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }, f, ensure_ascii=False)
    except Exception:
        pass


def load_odds_snapshot(mid: str, kind: str):
    try:
        with open(_snapshot_path(mid, kind), "r", encoding="utf-8") as f:
            return json.load(f).get("odds")
    except Exception:
        return None




def odds_drifted(mid: str, kind: str, current_odds, threshold: float = DRIFT_THRESHOLD) -> bool:
    """对比当前赔率与报告生成时的快照，判定是否变盘明显。

    返回 True 表示存在明显变盘（需重新生成报告）。
    """
    if not current_odds:
        return False
    cur = _to_implied(current_odds)
    if not cur:
        return False
    prev = load_odds_snapshot(mid, kind)
    if not prev:
        return False  # 无快照（首次生成）不视为变盘
    prev_i = _to_implied(prev)
    if not prev_i:
        return False
    drift = max(abs(cur[k] - prev_i[k]) for k in ("home", "draw", "away"))
    return drift >= threshold


# ===================== 批量后台同步（新拉取时预生成 + 变盘重生成） =====================

def sync_football_reports(mids):
    """批量同步足球深度报告（供 server 后台线程调用）。

    - 报告不存在 → 生成；
    - 报告已存在但赔率变盘明显 → 删除后重生成；
    已开赛比赛由调用方（server 列表过滤）保证不会传入，这里不再判断。
    """
    if not mids:
        return
    try:
        reportable = football_reportable_ids()
    except Exception:
        return
    for mid in mids:
        mid = str(mid)
        if mid not in reportable:
            continue
        try:
            ensure_football_report(mid)  # 内部含「不存在则生成 / 变盘则重生成」
        except Exception as e:
            print(f"[ERR] 后台同步足球报告失败 {mid}: {e}")


def sync_beidan_reports(recs):
    """批量同步北单深度报告（供 server 后台线程调用）。"""
    if not recs:
        return
    for rec in recs:
        mid = str(rec.get("match_id") or (rec.get("spf") or {}).get("match_id", "") or "")
        if not mid:
            continue
        try:
            ensure_beidan_report(mid)
        except Exception as e:
            print(f"[ERR] 后台同步北单报告失败 {mid}: {e}")


# ===================== 报告留存清理（retention） =====================



def _prune_manifest(reports_dir: str) -> int:
    """从 manifest 中移除文件已不存在的条目，返回移除数量。"""
    mp = os.path.join(reports_dir or DEFAULT_REPORTS_DIR, "football_bayes_manifest.json")
    if not os.path.exists(mp):
        return 0
    try:
        with open(mp, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return 0
    reps = manifest.get("reports", {})
    before = len(reps)
    kept = {}
    for mid, info in reps.items():
        fn = info.get("file", "")
        if fn and os.path.exists(os.path.join(reports_dir or DEFAULT_REPORTS_DIR, fn)):
            kept[mid] = info
    if len(kept) != before:
        manifest["reports"] = kept
        try:
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception:
            return 0
        return before - len(kept)
    return 0


def cleanup_old_reports(max_age_days: int = 3, reports_dir: str = None, dry_run: bool = False) -> dict:
    """删除 reports/ 下「生成时间」超过 max_age_days 天的深度报告内容。

    判定标准：以报告 HTML 文件自身的修改时间(mtime) 作为「报告生成时间」。
    任何超过 N 天未被刷新的报告文件视为过期，连同对应的 live_context_{mid}.json
    一并删除，并同步清理 football_bayes_manifest.json 中对应条目（文件不存在的
    条目均会被移除）。

    覆盖：football_bayes_*.html 与 beidan_bayes_*.html。
    返回结构化摘要，供 CLI / 日志展示。

    设计说明：
    - 北单报告标题不含比赛日期，足球报告虽含比赛日，但 mtime 策略对两类统一且
      稳健，故统一采用 mtime 判定。生产环境报告每日随赛程刷新，3 天未触动的即
      视为陈旧，可直接清理。
    - dry_run=True 时只统计不删除，便于先核对。
    """
    reports_dir = reports_dir or DEFAULT_REPORTS_DIR
    now = datetime.now().timestamp()
    cutoff = now - max_age_days * 86400.0

    deleted, freed, kept = [], 0, 0
    for pat in ("football_bayes_*.html", "beidan_bayes_*.html"):
        for p in sorted(glob.glob(os.path.join(reports_dir, pat))):
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            mid = _extract_mid_from_report_path(p)
            extra = []
            if mid:
                lp = os.path.join(reports_dir, f"live_context_{mid}.json")
                if os.path.exists(lp):
                    extra = [lp]
            if mtime < cutoff:
                age_days = round((now - mtime) / 86400.0, 2)
                size = os.path.getsize(p) + sum(os.path.getsize(x) for x in extra)
                if dry_run:
                    deleted.append({"file": os.path.basename(p), "mid": mid,
                                    "age_days": age_days, "size": size, "would_delete": True})
                    freed += size
                    continue
                try:
                    for x in [p] + extra:
                        os.remove(x)
                    deleted.append({"file": os.path.basename(p), "mid": mid,
                                    "age_days": age_days, "size": size})
                    freed += size
                except OSError as e:
                    print(f"[ERR] 删除失败 {p}: {e}")
            else:
                kept += 1

    pruned = 0
    if not dry_run:
        pruned = _prune_manifest(reports_dir)

    return {
        "reports_dir": reports_dir,
        "max_age_days": max_age_days,
        "dry_run": dry_run,
        "deleted": deleted,
        "deleted_count": len(deleted),
        "kept_count": kept,
        "freed_bytes": freed,
        "pruned_manifest": pruned,
    }

