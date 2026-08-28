# -*- coding: utf-8 -*-
"""足球赔率页解析的适配层：抓取 + 喂配置，算法在 `domain/sports/football/`

领域层不发请求、不读全局配置，所以本层负责：发 HTTP、把 `config.py` 的阈值
显式传进去、以及 ELO 这类需要外部状态的增补。

导出名与迁移前逐个一致（存量测试按名字导入）。下划线开头的那几个在领域层
去掉了下划线——它们本来就被 `__init__.py` 导出，"私有"只是名字上的错觉。
"""

import urllib.error

from ..common.logger import setup_logger
from ..domain.sports.football import lottery as _lot
from ..domain.sports.football import parsing as _p

log = setup_logger('football')
from . import fetching as _fetching_mod

from .config import (
    BASE, CLOSE_BLEND_WEIGHT, ELO_AVAILABLE, LEAGUE_PROFILES,
    LOTTERY_OFFICIAL_ODDS_WEIGHT, MIN_AVG_NUMBERS, ODDS_PAGES, OUZHI_JSON_URL,
    elo_to_goals_expected, elo_to_strength_factor, get_elo_system,
)

# ---- 纯计算：直接转发 ----
get_close_total_line = _p.get_close_total_line
parse_handicap = _p.parse_handicap
parse_total_line = _p.parse_total_line
parse_lottery_handicap = _lot.parse_lottery_handicap
_lottery_odds_probabilities = _lot.lottery_odds_probabilities
_spf_selection_profile = _lot.spf_selection_profile
_apply_lottery_market_availability = _lot.apply_lottery_market_availability
_html_to_text = _p.html_to_text
_extract_avg = _p.extract_avg_numbers
_handicap_text_to_num = _p.handicap_text_to_num
_extract_company_odds = _p.extract_company_odds
_extract_handicap_from_segment = _p.extract_handicap_from_segment
_parse_odds_value = _p.parse_odds_value
_team_in_context = _p.team_in_context
_parse_recent_form = _p.parse_recent_form
RECENT_FORM_PAT = _p.RECENT_FORM_PAT


def _blend_lottery_probabilities(model_probs, market_probs,
                                 market_weight=LOTTERY_OFFICIAL_ODDS_WEIGHT):
    return _lot.blend_lottery_probabilities(model_probs, market_probs, market_weight)


def lottery_market_probabilities(candidates, lottery_handicap=None,
                                 spf_odds=None, rqspf_odds=None):
    """Build JCZQ probabilities from scores and independently priced official markets."""
    return _lot.lottery_market_probabilities(
        candidates, lottery_handicap, spf_odds, rqspf_odds,
        market_weight=LOTTERY_OFFICIAL_ODDS_WEIGHT)


def calculate_bookmaker_consensus(bet365_data, pinnacle_data, avg_handicap):
    """计算博彩公司分歧指数"""
    return _p.bookmaker_consensus(bet365_data, pinnacle_data, avg_handicap)


def _blend_close_open(close_val, open_val, close_weight=CLOSE_BLEND_WEIGHT):
    """终盘为主、初盘为辅的线性融合"""
    return _p.blend_close_open(close_val, open_val, close_weight)


# ---- 抓取本体：留在适配层 ----

def _fetch_avg_page(match_id, page):
    """抓取指定赔率页，返回 (html, 平均值行数字列表)，并校验数据量"""
    label = ODDS_PAGES[page]
    html = _fetching_mod.fetch(f'{BASE}/fenxi/{page}-{match_id}.shtml')
    nums = _p.extract_avg_numbers(html)
    if len(nums) < MIN_AVG_NUMBERS:
        raise ValueError(f"{label}平均值数据不足 (match_id={match_id}), 获取到: {nums}")
    return html, nums


def fetch_yazhi(match_id):
    """抓取亚盘数据。平均值行格式: 初水位 初让球 初水位 终水位 终让球 终水位"""
    html, nums = _fetch_avg_page(match_id, 'yazhi')
    return _p.yazhi_from_page(html, nums)


def fetch_daxiao(match_id):
    """抓取大小球数据。平均值行盘口线为纯数字，第一组为初盘、第二组为终盘"""
    _, nums = _fetch_avg_page(match_id, 'daxiao')
    return _p.daxiao_from_avg_numbers(nums)


def fetch_ouzhi(match_id):
    """抓取欧赔平均值（JSON 时间序列）。每条为 [主, 平, 客, 返还率, 时间, ...]"""
    url = f'{OUZHI_JSON_URL}?fid={match_id}&cid=0&type=europe&r=1'
    referer = f'{BASE}/fenxi/ouzhi-{match_id}.shtml'
    try:
        series = _fetching_mod.fetch_json(url, referer=referer)
    except Exception as e:
        raise ValueError(f"抓取欧赔数据失败: {e} (match_id={match_id})")
    return _p.ouzhi_from_series(series, match_id)


def fetch_single_company_odds(match_id):
    """抓取 Bet365 和 Pinnacle 的独赔数据

    **线上从来没有取到过**：真实页面的纯文本里既没有公司名也没有任何别名
    （2026-08-28 实测），114 场缓存的 `bookmaker_consensus` 全是 None。
    两个页面与 `fetch_yazhi`/`fetch_daxiao` 是同一批 URL，`fetching.fetch`
    有 TTL 缓存与并发去重，所以不会多发请求——浪费的只是解析。
    详见交接文档 §四。
    """
    log.debug("抓取独赔数据: match_id=%s", match_id)
    result = {'bet365': None, 'pinnacle': None}
    try:
        yazhi_html = _fetching_mod.fetch(f'{BASE}/fenxi/yazhi-{match_id}.shtml')
        daxiao_html = _fetching_mod.fetch(f'{BASE}/fenxi/daxiao-{match_id}.shtml')

        for key, company in (('bet365', 'Bet365'), ('pinnacle', 'Pinnacle')):
            asian_row = _p.extract_company_odds(yazhi_html, company, is_total=False)
            total_row = _p.extract_company_odds(daxiao_html, company, is_total=True)
            if company == 'Bet365':
                log.debug("Bet365 亚盘原始数据: %s", asian_row)
            result[key] = _p.company_odds_to_markets(asian_row, total_row)

        log.debug(
            "独赔数据抓取完成: Bet365=%s, Pinnacle=%s",
            '有' if result['bet365'] else '无',
            '有' if result['pinnacle'] else '无',
        )
    except Exception as e:
        log.warning(f"抓取独赔数据失败: {e}")
    return result


def get_live_league_profile(league_name: str, recent_matches: int = 200):
    """从最近比赛数据计算实时联赛画像；取不到数据返回 None"""
    try:
        from .data_loader import fetch_league_matches
        return _p.league_profile_from_matches(
            fetch_league_matches(league_name, limit=recent_matches))
    except Exception as e:
        log.debug(f"计算实时联赛画像失败: {e}")
        return None


def resolve_league_profile(league_name):
    """按联赛名称匹配画像，用于场均进球与比分先验（融合静态+实时）"""
    name = (league_name or '').strip()
    static_profile = _p.resolve_static_league_profile(name, LEAGUE_PROFILES)
    live_profile = get_live_league_profile(name)
    return _p.blend_league_profiles(static_profile, live_profile, name)


def fetch_team_strength(match_id, home, away, league_profile=None):
    """从数据分析页抓取主客队近况并换算攻防强度，再叠加 ELO 评分

    返回 None 表示页面无数据（不影响主流程）。
    """
    try:
        html = _fetching_mod.fetch(f'{BASE}/fenxi/shuju-{match_id}.shtml')
    except (urllib.error.URLError, ValueError, OSError):
        return None

    result = _p.parse_team_strength(html, home, away, venue_weight=CLOSE_BLEND_WEIGHT)
    if result is None:
        return None

    if ELO_AVAILABLE:
        try:
            elo = get_elo_system()
            elo_home = elo.get_rating(home)
            elo_away = elo.get_rating(away)
            league_type = league_profile.get('name', '联赛') if league_profile else '联赛'
            result.update({
                'elo_home': elo_home,
                'elo_away': elo_away,
                'elo_xg_home': elo_to_goals_expected(elo_home, elo_away),
                'elo_xg_away': elo_to_goals_expected(elo_away, elo_home),
                'elo_strength_home': elo_to_strength_factor(elo_home),
                'elo_strength_away': elo_to_strength_factor(elo_away),
                'elo_prediction': elo.predict_match(home, away, league_type),
            })
            log.debug(f"ELO 评分: {home}={elo_home:.2f}, {away}={elo_away:.2f}")
            log.debug(f"ELO xG: {home}={result['elo_xg_home']:.2f}, {away}={result['elo_xg_away']:.2f}")
        except Exception as e:
            log.error(f"ELO 计算失败: {e}")

    return result
