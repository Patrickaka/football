"""
足球比分预测脚本 - 动态抓取赔率数据
====================================
数据来源: odds.500.com（Bet365等多家博彩公司平均值）

使用方法：
  运行脚本 → 输入主队和客队名称关键词 → 自动匹配比赛 → 抓取赔率 → 分析输出
"""

import sys
import math
import re
import gzip
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding='utf-8')

# ===================== 常量 =====================
BASE = 'https://odds.500.com'
INDEX_URL = f'{BASE}/index_jczq.shtml'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

# 平均值行至少应包含 初/终 各 3 个数值
MIN_AVG_NUMBERS = 6

# 趋势判定阈值：低于该幅度视为"无变化/稳定"
HANDICAP_TREND_EPS = 0.02
WATER_TREND_EPS = 0.05
EURO_PROB_TREND_EPS = 0.02

# 大小球倾向阈值：单边真实概率达到该值才判定为大/小球倾向
TOTAL_LEAN_THRESHOLD = 0.55

# 凯利指数：某项高于返还率超过该值视为「打出难度大」
KELLY_BIAS_EPS = 2.0

# 泊松比分矩阵枚举的单队最大进球数
MAX_GOALS = 7

# 初/终盘融合权重（终盘为主，初盘平滑噪声）
CLOSE_BLEND_WEIGHT = 0.72

# λ 网格拟合：粗搜步长、细搜步长与半径
LAMBDA_COARSE_STEP = 0.12
LAMBDA_FINE_STEP = 0.04
LAMBDA_FINE_RADIUS = 0.18

# 拟合目标权重：1X2 / 总进球 / 净胜球(反推) / 大小球分布 / 球队攻防先验
FIT_W_1X2 = 3.0
FIT_W_TOTAL = 1.4
FIT_W_SUPREMACY = 2.0
FIT_W_OU_DIST = 0.9
FIT_W_TEAM = 1.35

# 净胜球：亚盘反推 vs 欧赔反推 的融合权重（不再使用让球盘数值本身）
SUP_ASIAN_WEIGHT = 0.48
SUP_EURO_WEIGHT = 0.52

# 联赛场均进球基准（用于球队攻防强度归一化，可被联赛配置覆盖）
AVG_LEAGUE_GOAL = 1.35
HOME_VENUE_ATTACK_BOOST = 1.06

# 净胜球亚盘/欧赔严重分歧时改等权融合
SUPREMACY_CONFLICT_GAP = 0.75

# 欧赔走势对净胜球的修正幅度
MOMENTUM_SUPREMACY_WEIGHT = 0.22

# 坐标下降精调 λ 的迭代次数与步长
LAMBDA_REFINE_STEPS = 28
LAMBDA_REFINE_STEP0 = 0.07

# 预测置信度：低于该值仅推荐 1 个比分
CONFIDENCE_LOW_THRESHOLD = 0.52
CONFIDENCE_HIGH_THRESHOLD = 0.72

# 联赛画像：场均进球、主场加成、低比分倾向（乘在 0-2 球基准频率上）
LEAGUE_PROFILES = {
    'default': {'avg_goal': 1.35, 'home_boost': 1.06, 'low_score': 1.0, 'draw_mult': 1.0},
    '英超': {'avg_goal': 1.48, 'home_boost': 1.08, 'low_score': 0.92, 'draw_mult': 0.95},
    '英冠': {'avg_goal': 1.42, 'home_boost': 1.07, 'low_score': 0.94, 'draw_mult': 0.96},
    '西甲': {'avg_goal': 1.38, 'home_boost': 1.07, 'low_score': 1.02, 'draw_mult': 1.05},
    '意甲': {'avg_goal': 1.28, 'home_boost': 1.05, 'low_score': 1.12, 'draw_mult': 1.08},
    '德甲': {'avg_goal': 1.52, 'home_boost': 1.06, 'low_score': 0.90, 'draw_mult': 0.94},
    '法甲': {'avg_goal': 1.32, 'home_boost': 1.06, 'low_score': 1.05, 'draw_mult': 1.02},
    '荷甲': {'avg_goal': 1.55, 'home_boost': 1.05, 'low_score': 0.88, 'draw_mult': 0.93},
    '葡超': {'avg_goal': 1.30, 'home_boost': 1.06, 'low_score': 1.04, 'draw_mult': 1.03},
    '欧冠': {'avg_goal': 1.45, 'home_boost': 1.04, 'low_score': 0.96, 'draw_mult': 0.98},
    '欧联': {'avg_goal': 1.40, 'home_boost': 1.05, 'low_score': 0.98, 'draw_mult': 1.0},
    '世界杯': {'avg_goal': 1.38, 'home_boost': 1.03, 'low_score': 1.0, 'draw_mult': 1.0},
    '欧洲杯': {'avg_goal': 1.36, 'home_boost': 1.04, 'low_score': 1.02, 'draw_mult': 1.02},
    '友谊': {'avg_goal': 1.42, 'home_boost': 1.02, 'low_score': 0.98, 'draw_mult': 0.97},
    '国际': {'avg_goal': 1.40, 'home_boost': 1.03, 'low_score': 1.0, 'draw_mult': 1.0},
    '巴甲': {'avg_goal': 1.38, 'home_boost': 1.08, 'low_score': 0.95, 'draw_mult': 0.96},
    '阿甲': {'avg_goal': 1.32, 'home_boost': 1.07, 'low_score': 1.0, 'draw_mult': 0.98},
    '中超': {'avg_goal': 1.28, 'home_boost': 1.07, 'low_score': 1.06, 'draw_mult': 1.04},
    '日职': {'avg_goal': 1.30, 'home_boost': 1.06, 'low_score': 1.04, 'draw_mult': 1.03},
    '韩K': {'avg_goal': 1.26, 'home_boost': 1.06, 'low_score': 1.06, 'draw_mult': 1.04},
}

# 比分冷热：相对历史基准频率的比值阈值
HEAT_RATIO_HOT = 0.70
HEAT_RATIO_COLD = 1.32
HEAT_FILTER_PENALTY = 0.62
COLD_FILTER_BONUS = 1.18

# 常见比分历史基准频率（用于冷热，非市场赔率）
SCORE_BASELINE_FREQ = {
    (0, 0): 0.082, (1, 0): 0.095, (0, 1): 0.072, (1, 1): 0.118,
    (2, 0): 0.071, (0, 2): 0.048, (2, 1): 0.092, (1, 2): 0.058,
    (2, 2): 0.042, (3, 0): 0.038, (0, 3): 0.022, (3, 1): 0.048,
    (1, 3): 0.028, (3, 2): 0.032, (2, 3): 0.022, (4, 0): 0.018,
    (0, 4): 0.010, (4, 1): 0.022, (1, 4): 0.012, (3, 3): 0.014,
}

# 仅亚盘/大小球走 HTML 抓取（无平均值 JSON 接口）；欧赔走 JSON
ODDS_PAGES = {
    'yazhi': '亚盘',
    'daxiao': '大小球',
}

# 欧赔平均值时间序列 JSON 接口：cid=0 即"平均值"，按时间降序（首条=终盘）
OUZHI_JSON_URL = f'{BASE}/fenxi1/json/ouzhi.php'

# ===================== 工具函数 =====================

def fetch(url, encoding='gbk', referer=None):
    """抓取网页，自动处理 gzip 压缩和编码"""
    headers = {**HEADERS, 'Referer': referer} if referer else HEADERS
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except urllib.error.HTTPError:
        from http import cookiejar
        cj = cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        with opener.open(req, timeout=20) as resp:
            raw = resp.read()

    # 自动解压 gzip
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)

    for enc in [encoding, 'gb2312', 'gb18030', 'utf-8']:
        try:
            result = raw.decode(enc)
            # 清理 surrogate 字符
            result = result.encode('utf-8', errors='replace').decode('utf-8')
            return result
        except (UnicodeDecodeError, LookupError, UnicodeEncodeError):
            continue
    return raw.decode('utf-8', errors='replace')


def fetch_json(url, referer=None):
    """抓取并解析 JSON 接口"""
    return json.loads(fetch(url, encoding='utf-8', referer=referer))


def parse_handicap(text):
    """将让球文本转换为数值（正=主让，负=客让）"""
    t = text.strip()
    sign = -1 if '受' in t else 1
    t = t.replace('受', '')

    mapping = {
        '平手': 0, '半球': 0.5, '一球': 1.0, '球半': 1.5,
        '两球': 2.0, '两球半': 2.5, '三球': 3.0, '三球半': 3.5,
        '平手/半球': 0.25, '半球/一球': 0.75,
        '一球/球半': 1.25, '球半/两球': 1.75,
        '两球/两球半': 2.25, '两球半/三球': 2.75, '三球/三球半': 3.25,
    }
    if t in mapping:
        return sign * mapping[t]
    try:
        return sign * float(t)
    except ValueError:
        return 0


def parse_total_line(text):
    """解析大小球盘口线"""
    t = text.strip()
    mapping = {
        '0.5/1': 0.75, '1/1.5': 1.25, '1.5/2': 1.75,
        '2/2.5': 2.25, '2.5/3': 2.75, '3/3.5': 3.25, '3.5/4': 3.75,
    }
    if t in mapping:
        return mapping[t]
    try:
        return float(t)
    except ValueError:
        return 2.5


# ===================== 抓取比赛列表 =====================

def fetch_match_list():
    """抓取今日比赛列表，返回 [{home, away, match_id, league, time}, ...]"""
    print("  正在联网获取今日比赛列表...")
    html = fetch(INDEX_URL)

    matches = []

    # 方案A: shuju 链接在前、title 锚点紧随其后，故以 id 为锚向后取本场 title
    # 布局: ...shuju-<id>.shtml... title="主队名VS客队名数据分析"...
    title_pat = re.compile(
        r'shuju-(\d+)\.shtml.*?title="([^"]+?)VS([^"]+?)'
        r'(?:数据|盘口|百家|欧赔|亚赔|亚盘|指数|对比|分析)[^"]*"',
        re.DOTALL
    )
    for m in title_pat.finditer(html):
        match_id = m.group(1).strip()
        home_name = m.group(2).strip()
        away_name = m.group(3).strip()
        # 清理队名尾部残留后缀
        for suffix in ['百家', '欧赔', '亚赔', '亚盘', '数据', '盘口', '指数', '对比', '分析', '百家欧赔', '百家亚盘']:
            if home_name.endswith(suffix):
                home_name = home_name[:-len(suffix)].strip()
            if away_name.endswith(suffix):
                away_name = away_name[:-len(suffix)].strip()
        if home_name and away_name and match_id:
            matches.append({
                'home': home_name,
                'away': away_name,
                'match_id': match_id
            })

    # 如果 title 方案没找到，用方案B: 正则匹配 team 链接
    if not matches:
        row_pat = re.compile(
            r'<a[^>]*href="//liansai\.500\.com/team/\d+/"[^>]*>([^<]+)</a>'
            r'.*?VS.*?'
            r'<a[^>]*href="//liansai\.500\.com/team/\d+/"[^>]*>([^<]+)</a>'
            r'.*?shuju-(\d+)\.shtml',
            re.DOTALL
        )
        for m in row_pat.finditer(html):
            home_name = m.group(1).strip()
            away_name = m.group(2).strip()
            match_id = m.group(3)
            if home_name and away_name and match_id:
                matches.append({
                    'home': home_name,
                    'away': away_name,
                    'match_id': match_id
                })

    # 提取联赛和时间
    league_pat = re.compile(r'<a[^>]*href="//liansai\.500\.com/zuqiu-\d+/"[^>]*>([^<]+)</a>')
    time_pat = re.compile(r'(\d{2}-\d{2}\s+\d{2}:\d{2})')

    leagues = league_pat.findall(html)
    times = time_pat.findall(html)

    for i, match in enumerate(matches):
        if i < len(leagues):
            match['league'] = leagues[i].strip()
        if i < len(times):
            match['time'] = times[i]

    return matches


def search_match(matches, home_key, away_key):
    """搜索匹配的比赛，支持模糊匹配"""
    home_key = home_key.strip()
    away_key = away_key.strip()

    # 精确匹配
    exact = []
    partial = []
    for m in matches:
        home_match = home_key in m['home'] if home_key else True
        away_match = away_key in m['away'] if away_key else True
        if home_match and away_match:
            if m['home'] == home_key and m['away'] == away_key:
                exact.append(m)
            else:
                partial.append(m)
    return exact + partial


# ===================== 解析赔率 =====================

def _html_to_text(html):
    """去除标签与转义空白，压缩为单行纯文本"""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _extract_avg(html, keyword='平均值'):
    """从HTML中提取包含keyword行后续的全部数字"""
    text = _html_to_text(html)
    idx = text.find(keyword)
    if idx < 0:
        raise ValueError(f"未找到'{keyword}'行")
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text[idx:])
    return [float(n) for n in numbers]


def _fetch_avg_page(match_id, page):
    """抓取指定赔率页，返回 (html, 平均值行数字列表)，并校验数据量"""
    label = ODDS_PAGES[page]
    html = fetch(f'{BASE}/fenxi/{page}-{match_id}.shtml')
    nums = _extract_avg(html)
    if len(nums) < MIN_AVG_NUMBERS:
        raise ValueError(f"{label}平均值数据不足 (match_id={match_id}), 获取到: {nums}")
    return html, nums


def fetch_yazhi(match_id):
    """抓取亚盘数据。平均值行格式: 初水位 初让球 初水位 终水位 终让球 终水位"""
    html, nums = _fetch_avg_page(match_id, 'yazhi')

    segment = _html_to_text(html)
    idx = segment.find('平均值')
    segment = segment[idx:idx + 200]

    open_hcap_raw = _extract_handicap_from_segment(segment, nums[0], nums[2])
    close_hcap_raw = _extract_handicap_from_segment(
        segment[segment.find(str(nums[3])):] if str(nums[3]) in segment else segment,
        nums[3], nums[5]
    )

    # 500.com 数值让球为负表示主让，取反以符合脚本惯例（正=主让）
    # 平均值行第一组为初盘、第二组为终盘
    return {
        'open': {
            'handicap': -open_hcap_raw,
            'home_odds': nums[0],
            'away_odds': nums[2],
        },
        'close': {
            'handicap': -close_hcap_raw,
            'home_odds': nums[3],
            'away_odds': nums[5],
        }
    }


def _extract_handicap_from_segment(segment, before_val, after_val):
    """从文本片段中提取两个数字之间的让球值（可能是文本或数字）"""
    pat = re.compile(
        rf'{re.escape(str(before_val))}\s+([^\d\s]+(?:/[^\d\s]+)?)\s+{re.escape(str(after_val))}'
    )
    m = pat.search(segment)
    if m:
        handicap_str = m.group(1)
        # 尝试数字解析
        try:
            return float(handicap_str)
        except ValueError:
            return parse_handicap(handicap_str)

    # 如果上面的模式没匹配，尝试直接数字匹配
    pat2 = re.compile(
        rf'{re.escape(str(before_val))}\s+(-?[\d.]+)\s+{re.escape(str(after_val))}'
    )
    m2 = pat2.search(segment)
    if m2:
        return float(m2.group(1))

    return 0


def fetch_ouzhi(match_id):
    """抓取欧赔平均值（JSON 时间序列）。每条为 [主, 平, 客, 返还率, 时间, ...]"""
    url = f'{OUZHI_JSON_URL}?fid={match_id}&cid=0&type=europe&r=1'
    referer = f'{BASE}/fenxi/ouzhi-{match_id}.shtml'
    series = fetch_json(url, referer=referer)
    if not isinstance(series, list) or not series:
        raise ValueError(f"欧赔平均值 JSON 为空或异常 (match_id={match_id}): {series}")

    close, open_ = series[0], series[-1]
    return {
        'open': {
            'home': open_[0], 'draw': open_[1], 'away': open_[2],
            'return_rate': float(open_[3]) if len(open_) > 3 else None,
        },
        'close': {
            'home': close[0], 'draw': close[1], 'away': close[2],
            'return_rate': float(close[3]) if len(close) > 3 else None,
        },
        'series': series,
    }


def fetch_daxiao(match_id):
    """抓取大小球数据。平均值行盘口线为纯数字，第一组为初盘、第二组为终盘"""
    _, nums = _fetch_avg_page(match_id, 'daxiao')
    return {
        'open': {
            'line': nums[1],
            'over_odds': nums[0],
            'under_odds': nums[2],
        },
        'close': {
            'line': nums[4],
            'over_odds': nums[3],
            'under_odds': nums[5],
        }
    }


RECENT_FORM_PAT = re.compile(
    r'近(\d+)场战绩.*?'
    r'<span class="ying">(\d+)胜</span>.*?'
    r'<span class="ping">(\d+)平</span>.*?'
    r'<span class="shu">(\d+)负</span>.*?'
    r'进<span class="ying">(\d+)球</span>失<span class="shu">(\d+)球</span>',
    re.DOTALL,
)


def _team_in_context(ctx, name):
    """队名与上下文模糊匹配（兼容简称）"""
    if not name:
        return False
    if name in ctx:
        return True
    for n in (4, 3, 2):
        if len(name) >= n and name[-n:] in ctx:
            return True
    return False


def resolve_league_profile(league_name):
    """按联赛名称匹配画像，用于场均进球与比分先验"""
    name = (league_name or '').strip()
    profile = dict(LEAGUE_PROFILES['default'])
    for key in sorted(LEAGUE_PROFILES, key=len, reverse=True):
        if key != 'default' and key in name:
            profile.update(LEAGUE_PROFILES[key])
            profile['name'] = key
            return profile
    profile['name'] = 'default'
    return profile


def _parse_recent_form(groups):
    n, w, d, l = int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
    gf, ga = int(groups[4]), int(groups[5])
    n = max(n, 1)
    pts = w * 3 + d
    return {
        'games': n, 'wins': w, 'draws': d, 'losses': l,
        'gf': gf, 'ga': ga, 'attack': gf / n, 'defense': ga / n,
        'form_pts': pts / n,
    }


def fetch_team_strength(match_id, home, away):
    """
    从数据分析页抓取主客队近10场及主客场进球/失球，换算攻防强度。
    返回 None 表示页面无数据（不影响主流程）。
    """
    try:
        html = fetch(f'{BASE}/fenxi/shuju-{match_id}.shtml')
    except (urllib.error.URLError, ValueError, OSError):
        return None

    tagged = []
    for m in RECENT_FORM_PAT.finditer(html):
        # 仅用紧邻战绩前的短上下文识别队名，避免多场数据串台
        ctx = _html_to_text(html[max(0, m.start() - 140):m.start()])
        tagged.append({'ctx': ctx, 'stats': _parse_recent_form(m.groups())})

    if len(tagged) < 2:
        return None

    home_all = away_all = home_venue = away_venue = None
    for item in tagged:
        ctx, st = item['ctx'], item['stats']
        if _team_in_context(ctx, home):
            if home_all is None:
                home_all = st
            elif home_venue is None:
                home_venue = st
        elif _team_in_context(ctx, away):
            if away_all is None:
                away_all = st
            elif away_venue is None:
                away_venue = st

    if not home_all or not away_all:
        return None

    hv = home_venue or home_all
    av = away_venue or away_all
    attack_home = _blend_close_open(hv['attack'], home_all['attack'], 0.68)
    defense_home = _blend_close_open(hv['defense'], home_all['defense'], 0.68)
    attack_away = _blend_close_open(av['attack'], away_all['attack'], 0.68)
    defense_away = _blend_close_open(av['defense'], away_all['defense'], 0.68)

    form_diff = home_all['form_pts'] - away_all['form_pts']
    return {
        'home_recent': home_all,
        'away_recent': away_all,
        'home_venue': hv,
        'away_venue': av,
        'attack_home': attack_home,
        'defense_home': defense_home,
        'attack_away': attack_away,
        'defense_away': defense_away,
        'form_diff': form_diff,
        'momentum_supremacy': max(-0.35, min(0.35, form_diff * 0.12)),
        'summary': (
            f"主队近{home_all['games']}场 进{home_all['gf']}失{home_all['ga']}（{home_all['form_pts']:.1f}分/场）；"
            f"客队近{away_all['games']}场 进{away_all['gf']}失{away_all['ga']}（{away_all['form_pts']:.1f}分/场）"
        ),
    }


# ===================== 分析函数 =====================

def remove_vig(o1, o2, o3=None):
    """去水率，返回真实概率"""
    if o3 is None:
        p1, p2 = 1 / o1, 1 / o2
        total = p1 + p2
        return p1 / total, p2 / total
    else:
        p1, p2, p3 = 1 / o1, 1 / o2, 1 / o3
        total = p1 + p2 + p3
        return p1 / total, p2 / total, p3 / total


def analyze_asian(data):
    """解析亚盘，返回让球走势、水位走势、真实概率与强弱判断"""
    op, cl = data['open'], data['close']
    hcap = cl['handicap']

    dh = hcap - op['handicap']
    if dh > HANDICAP_TREND_EPS:
        handicap_trend = f"让球升高 {op['handicap']:+.2f} → {hcap:+.2f}（主队被看好）"
    elif dh < -HANDICAP_TREND_EPS:
        handicap_trend = f"让球降低 {op['handicap']:+.2f} → {hcap:+.2f}（客队被看好）"
    else:
        handicap_trend = f"让球不变 {hcap:+.2f}（盘口稳定）"

    dw = cl['home_odds'] - op['home_odds']
    if dw > WATER_TREND_EPS:
        water_trend = "主队水位上升 → 资金偏向客队"
    elif dw < -WATER_TREND_EPS:
        water_trend = "主队水位下降 → 资金偏向主队"
    else:
        water_trend = "水位基本稳定"

    hp_o, ap_o = remove_vig(op['home_odds'], op['away_odds'])
    hp_c, ap_c = remove_vig(cl['home_odds'], cl['away_odds'])

    if abs(hcap) <= 0.25:
        diff_range, diff_desc = [0, 0.5], "势均力敌"
    elif abs(hcap) <= 0.75:
        diff_range, diff_desc = [0.5, 1.5], "预期1球差"
    elif abs(hcap) <= 1.25:
        diff_range, diff_desc = [1, 2], "预期1-2球差"
    elif abs(hcap) <= 1.75:
        diff_range, diff_desc = [1.5, 2.5], "预期2球差"
    else:
        diff_range = [abs(hcap) - 0.25, abs(hcap) + 0.25]
        diff_desc = f"预期{abs(hcap):.1f}球差以上"

    if hcap > 0:
        favor, favor_desc = 'home', f"主队让 {hcap} 球（主强客弱）"
    elif hcap < 0:
        favor, favor_desc = 'away', f"客队让 {abs(hcap)} 球（客强主弱）"
    else:
        favor, favor_desc = 'even', "平手盘（势均力敌）"

    return {
        'handicap': hcap,
        'open_handicap': op['handicap'],
        'favor': favor, 'favor_desc': favor_desc, 'diff_desc': diff_desc,
        'diff_range': diff_range,
        'handicap_trend': handicap_trend, 'water_trend': water_trend,
        'open_prob': {'home_recv': hp_o, 'away_give': ap_o},
        'close_prob': {'home_recv': hp_c, 'away_give': ap_c},
    }


def _return_rate_from_odds(home, draw, away):
    """由欧赔估算理论返还率（%），JSON 无返还率字段时兜底"""
    total = 1.0 / home + 1.0 / draw + 1.0 / away
    return 100.0 / total if total > 0 else 92.0


def kelly_index_triple(home_odds, draw_odds, away_odds, p_home, p_draw, p_away):
    """三项凯利指数（%）= 赔率 × 去水概率 × 100，与 500.com 口径一致"""
    return {
        'home': home_odds * p_home * 100,
        'draw': draw_odds * p_draw * 100,
        'away': away_odds * p_away * 100,
    }


def _kelly_outcome_label(key):
    return {'home': '主胜', 'draw': '平局', 'away': '客胜'}[key]


def analyze_kelly(ouzhi_data, probs_open, probs_close):
    """
    欧赔凯利指数分析：初/终盘凯利、返还率对比、离散度与打出难度提示。
    probs 通常取同一组欧赔去水概率（与计算凯利的赔率对应）。
    """
    op, cl = ouzhi_data['open'], ouzhi_data['close']
    ph_o, pd_o, pa_o = probs_open
    ph_c, pd_c, pa_c = probs_close

    rr_o = op.get('return_rate') or _return_rate_from_odds(op['home'], op['draw'], op['away'])
    rr_c = cl.get('return_rate') or _return_rate_from_odds(cl['home'], cl['draw'], cl['away'])

    k_open = kelly_index_triple(op['home'], op['draw'], op['away'], ph_o, pd_o, pa_o)
    k_close = kelly_index_triple(cl['home'], cl['draw'], cl['away'], ph_c, pd_c, pa_c)
    delta = {k: k_close[k] - k_open[k] for k in k_close}

    labels = ('home', 'draw', 'away')
    spread = max(k_close.values()) - min(k_close.values())
    hardest = max(labels, key=lambda k: k_close[k] - rr_c)
    favored = min(labels, key=lambda k: k_close[k] - rr_c)

    risks, favors, kelly_changes = [], [], []
    for k in labels:
        name = _kelly_outcome_label(k)
        diff = k_close[k] - rr_c
        if diff > KELLY_BIAS_EPS:
            risks.append(f"{name}凯利{k_close[k]:.1f}高于返还率{rr_c:.1f}（+{diff:.1f}）→ 打出偏难")
        elif diff < -KELLY_BIAS_EPS:
            favors.append(f"{name}凯利{k_close[k]:.1f}低于返还率（{diff:.1f}）→ 相对看好")
        if abs(delta[k]) >= 1.0:
            arrow = '↑' if delta[k] > 0 else '↓'
            kelly_changes.append(f"{name}凯利{arrow}{abs(delta[k]):.1f}")

    if spread >= 4.0:
        bias_desc = f"凯利离散度{spread:.1f}，庄家态度分化明显"
    else:
        bias_desc = f"凯利离散度{spread:.1f}，三项较为均衡"

    summary_parts = [bias_desc, f"最难项倾向{_kelly_outcome_label(hardest)}"]
    if favors:
        summary_parts.append(favors[0])
    summary = '；'.join(summary_parts)

    return {
        'return_rate': {'open': rr_o, 'close': rr_c},
        'open': k_open,
        'close': k_close,
        'delta': delta,
        'spread': spread,
        'hardest': hardest,
        'favored': favored,
        'risks': risks,
        'favors': favors,
        'kelly_changes': kelly_changes,
        'summary': summary,
    }


def analyze_euro_momentum(series):
    """由欧赔时间序列提取主/客胜概率走势，用于修正净胜球"""
    if not series or len(series) < 2:
        return {'shift_supremacy': 0.0, 'summary': '欧赔走势数据不足'}

    chrono = list(reversed(series))
    first = remove_vig(chrono[0][0], chrono[0][1], chrono[0][2])
    last = remove_vig(chrono[-1][0], chrono[-1][1], chrono[-1][2])
    d_home = last[0] - first[0]
    d_away = last[2] - first[2]
    shift = max(-0.45, min(0.45, (d_home - d_away) * 1.8))

    parts = []
    if d_home > EURO_PROB_TREND_EPS:
        parts.append(f"主胜概率累积↑{d_home * 100:.1f}%")
    elif d_home < -EURO_PROB_TREND_EPS:
        parts.append(f"主胜概率累积↓{-d_home * 100:.1f}%")
    if d_away > EURO_PROB_TREND_EPS:
        parts.append(f"客胜概率累积↑{d_away * 100:.1f}%")
    elif d_away < -EURO_PROB_TREND_EPS:
        parts.append(f"客胜概率累积↓{-d_away * 100:.1f}%")

    return {
        'shift_supremacy': shift,
        'delta_home': d_home,
        'delta_away': d_away,
        'summary': '，'.join(parts) if parts else '欧赔走势平稳',
    }


def analyze_euro(data):
    """解析欧赔，返回初终盘 1X2 真实概率、凯利、走势与变化趋势"""
    op, cl = data['open'], data['close']

    ph_o, pd_o, pa_o = remove_vig(op['home'], op['draw'], op['away'])
    ph_c, pd_c, pa_c = remove_vig(cl['home'], cl['draw'], cl['away'])

    changes = []
    if ph_c - ph_o > EURO_PROB_TREND_EPS: changes.append(f"主胜概率↑{(ph_c-ph_o)*100:.1f}%")
    elif ph_c - ph_o < -EURO_PROB_TREND_EPS: changes.append(f"主胜概率↓{(ph_o-ph_c)*100:.1f}%")
    if pa_c - pa_o > EURO_PROB_TREND_EPS: changes.append(f"客胜概率↑{(pa_c-pa_o)*100:.1f}%")
    elif pa_c - pa_o < -EURO_PROB_TREND_EPS: changes.append(f"客胜概率↓{(pa_o-pa_c)*100:.1f}%")
    if pd_c - pd_o > EURO_PROB_TREND_EPS: changes.append(f"平局概率↑{(pd_c-pd_o)*100:.1f}%")
    elif pd_c - pd_o < -EURO_PROB_TREND_EPS: changes.append(f"平局概率↓{(pd_o-pd_c)*100:.1f}%")

    kelly = analyze_kelly(data, (ph_o, pd_o, pa_o), (ph_c, pd_c, pa_c))
    momentum = analyze_euro_momentum(data.get('series', []))

    return {
        'open': {'home': ph_o, 'draw': pd_o, 'away': pa_o},
        'close': {'home': ph_c, 'draw': pd_c, 'away': pa_c},
        'raw_odds': {'open': dict(op), 'close': dict(cl)},
        'kelly': kelly,
        'momentum': momentum,
        'changes': changes,
    }


def analyze_total(data):
    """解析大小球，返回盘口线、大小球真实概率、倾向与期望进球区间"""
    op, cl = data['open'], data['close']
    line = cl['line']

    po_o, pu_o = remove_vig(op['over_odds'], op['under_odds'])
    po_c, pu_c = remove_vig(cl['over_odds'], cl['under_odds'])

    if po_c >= TOTAL_LEAN_THRESHOLD:
        lean, lean_desc = 'over', f"大球倾向（大球概率{po_c*100:.1f}%）"
    elif pu_c >= TOTAL_LEAN_THRESHOLD:
        lean, lean_desc = 'under', f"小球倾向（小球概率{pu_c*100:.1f}%）"
    else:
        lean, lean_desc = None, f"大小球均衡（线{line}，各约50%）"

    over_lean = lean == 'over'
    if line <= 1.0:
        expected_goals = [1, 3]
    elif line <= 2.0:
        expected_goals = [1, 4]
    elif line <= 2.5:
        expected_goals = [2, 4] if over_lean else [1, 3]
    elif line <= 3.0:
        expected_goals = [2, 5] if over_lean else [1, 3]
    elif line <= 3.5:
        expected_goals = [3, 6] if over_lean else [2, 4]
    else:
        lo = max(0, int(line))
        expected_goals = [lo, lo + 2]

    implied_total = implied_total_goals(line, po_c)
    open_implied = implied_total_goals(op['line'], po_o)

    return {
        'open_line': op['line'], 'close_line': line,
        'implied_total': implied_total,
        'open_implied_total': open_implied,
        'lean': lean, 'lean_desc': lean_desc,
        'open_prob': {'over': po_o, 'under': pu_o},
        'close_prob': {'over': po_c, 'under': pu_c},
        'expected_goals': expected_goals,
    }


def _poisson_pmf(k, lam):
    """泊松概率质量函数 P(X=k)"""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _outcome(h, a):
    return 'home' if h > a else 'draw' if h == a else 'away'


def _blend_close_open(close_val, open_val, close_weight=CLOSE_BLEND_WEIGHT):
    """终盘为主、初盘为辅的线性融合"""
    if open_val is None:
        return close_val
    w = close_weight
    return w * close_val + (1.0 - w) * open_val


def _poisson_tail_over(lam_total, line):
    """泊松总进球模型下 P(总进球 > line)；四分盘按相邻半球盘各半权重"""
    frac = round((line * 4) % 4)
    if frac in (1, 3):
        low, high = line - 0.25, line + 0.25
        return 0.5 * _poisson_tail_over(lam_total, low) + 0.5 * _poisson_tail_over(lam_total, high)
    k_min = math.floor(line + 0.501)
    prob = 0.0
    for k in range(k_min, 30):
        prob += _poisson_pmf(k, lam_total)
    return min(1.0, prob)


def implied_total_goals(line, p_over, tol=1e-4):
    """由大小球盘口线与去水大球概率反推期望总进球 λ_total"""
    p_over = max(0.02, min(0.98, p_over))
    lo, hi = 0.3, 6.5
    for _ in range(48):
        mid = (lo + hi) / 2
        if _poisson_tail_over(mid, line) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _dc_tau(h, a, lam_home, lam_away, rho):
    """Dixon-Coles 低比分相关修正因子（修正独立泊松对 0-0/1-1 的偏差）"""
    if h > 1 or a > 1:
        return 1.0
    if h == 0 and a == 0:
        return 1.0 - lam_home * lam_away * rho
    if h == 0 and a == 1:
        return 1.0 + lam_home * rho
    if h == 1 and a == 0:
        return 1.0 + lam_away * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _matrix_margins(matrix):
    """从比分矩阵汇总 1X2 边缘概率"""
    margins = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    for (h, a), prob in matrix.items():
        margins[_outcome(h, a)] += prob
    return margins


def _asian_payout_home(diff, handicap):
    """亚盘主队结算单位：1=全赢, 0.5=半赢, 0=走水, -0.5=半输, -1=全输"""
    frac = round((handicap * 4) % 4)
    if frac in (1, 3):
        low, high = handicap - 0.25, handicap + 0.25
        return 0.5 * _asian_payout_home(diff, low) + 0.5 * _asian_payout_home(diff, high)
    adj = diff - handicap
    if adj > 1e-9:
        return 1.0
    if abs(adj) <= 1e-9:
        return 0.0
    return -1.0


def _asian_cover_prob(lam_home, lam_away, handicap, rho=0.0):
    """泊松比分矩阵下主队赢盘（含半赢）概率"""
    matrix = build_score_matrix(lam_home, lam_away, rho=rho)
    cover = 0.0
    for (h, a), prob in matrix.items():
        pay = _asian_payout_home(h - a, handicap)
        if pay > 0:
            cover += prob
        elif pay == 0.5:
            cover += 0.5 * prob
    return cover


def asian_implied_supremacy(
    handicap, p_home_cover, p_away_cover,
    total_hint=2.5, open_handicap=None, open_hp=None, open_ap=None,
):
    """
    由让球盘 + 上下盘真实概率反推期望净胜球（不再把盘口线当作净胜球）。
    在泊松框架下二分搜索 μ，使 P(主队赢盘) ≈ 去水后主胜概率。
    """
    p_home = max(0.05, min(0.95, p_home_cover))
    if open_handicap is not None and open_hp is not None:
        handicap = _blend_close_open(handicap, open_handicap)
        p_home = _blend_close_open(p_home, open_hp)

    lo, hi = -3.5, 3.5
    for _ in range(52):
        mid = (lo + hi) / 2
        lam_h = max(0.08, (total_hint + mid) / 2)
        lam_a = max(0.08, (total_hint - mid) / 2)
        pc = _asian_cover_prob(lam_h, lam_a, handicap)
        if pc < p_home:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def euro_implied_supremacy(p_home, p_draw, p_away, total_hint=2.5):
    """由欧赔 1X2 真实概率反推期望净胜球（独立于亚盘让球数值）"""
    p_home, p_draw, p_away = max(p_home, 0.02), max(p_draw, 0.02), max(p_away, 0.02)
    lo, hi = -3.5, 3.5
    for _ in range(52):
        mid = (lo + hi) / 2
        lam_h = max(0.08, (total_hint + mid) / 2)
        lam_a = max(0.08, (total_hint - mid) / 2)
        margins = _matrix_margins(build_score_matrix(lam_h, lam_a))
        if margins['home'] < p_home:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def euro_implied_lambdas(p_home, p_draw, p_away, total_hint):
    """由欧赔 1X2 直接拟合主客队 λ（作为球队强度融合的先验）"""
    return _fit_lambda_grid(
        euro_implied_supremacy(p_home, p_draw, p_away, total_hint),
        total_hint, p_home, p_draw, p_away, rho=0.0,
        ou_targets=None, team_lambdas=None,
    )


def blend_market_supremacy(sup_asian, sup_euro):
    """融合亚盘与欧赔反推净胜球；严重分歧时等权避免单边偏差"""
    if sup_asian * sup_euro < 0 or abs(sup_asian - sup_euro) >= SUPREMACY_CONFLICT_GAP:
        return 0.5 * sup_asian + 0.5 * sup_euro
    return SUP_ASIAN_WEIGHT * sup_asian + SUP_EURO_WEIGHT * sup_euro


def compute_prediction_confidence(asian, euro, total, team=None):
    """
    多市场信号一致性 → 置信度 0~1。
    低置信时减少推荐条数并降权排序。
    """
    score = 1.0
    notes = []
    sup_a = asian.get('implied_supremacy', 0.0)
    sup_e = euro.get('implied_supremacy', 0.0)

    if sup_a * sup_e < 0:
        score -= 0.32
        notes.append('亚盘与欧赔净胜球方向相反')
    elif abs(sup_a - sup_e) >= SUPREMACY_CONFLICT_GAP:
        score -= 0.22
        notes.append(f'净胜球分歧较大（亚{sup_a:+.2f}/欧{sup_e:+.2f}）')

    kelly = euro.get('kelly') or {}
    if kelly.get('spread', 99) < 2.5:
        score -= 0.12
        notes.append('凯利三项胶着')

    if team and euro.get('implied_lambdas'):
        target = total.get('implied_total', 2.5)
        tl = team_poisson_lambdas(team, target, team.get('league_profile'))
        el = euro['implied_lambdas']
        gap = abs(el['home'] - tl[0]) + abs(el['away'] - tl[1])
        if gap > 0.85:
            score -= 0.14
            notes.append('球队攻防λ与市场λ偏差大')

    score = max(0.0, min(1.0, score))
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        level, label = 'high', '高置信'
    elif score >= CONFIDENCE_LOW_THRESHOLD:
        level, label = 'medium', '中置信'
    else:
        level, label = 'low', '低置信（谨慎参考）'

    return {
        'score': round(score, 3),
        'level': level,
        'label': label,
        'notes': notes,
        'recommend_count': 2 if level != 'low' else 1,
    }


def team_poisson_lambdas(strength, total_target, league_profile=None):
    """
    用攻防强度构造 λ：主队进攻×客队防守×主场系数。
    defense 为场均失球（对手防守弱则失球多 → 因子更大）。
    """
    lp = league_profile or strength.get('league_profile') or LEAGUE_PROFILES['default']
    avg = lp.get('avg_goal', AVG_LEAGUE_GOAL)
    boost = lp.get('home_boost', HOME_VENUE_ATTACK_BOOST)
    atk_h = strength['attack_home'] / avg
    def_a = strength['defense_away'] / avg
    atk_a = strength['attack_away'] / avg
    def_h = strength['defense_home'] / avg
    lam_home = max(0.08, atk_h * def_a * avg * boost)
    lam_away = max(0.08, atk_a * def_h * avg)
    scale = total_target / max(lam_home + lam_away, 0.1)
    return lam_home * scale, lam_away * scale


def _ou_total_distribution(lam_total, max_k=6):
    return {_k: _poisson_pmf(_k, lam_total) for _k in range(max_k + 1)}


def _matrix_total_margins(matrix, max_k=6):
    margins = {k: 0.0 for k in range(max_k + 1)}
    for (h, a), prob in matrix.items():
        t = min(h + a, max_k)
        margins[t] += prob
    return margins


def estimate_lambdas(supremacy, total_line, min_lambda=0.05):
    """由净胜球与总进球快速解 λ（兜底）"""
    lam_home = max(min_lambda, (total_line + supremacy) / 2)
    lam_away = max(min_lambda, (total_line - supremacy) / 2)
    return lam_home, lam_away


def _lambda_fit_error(
    lam_pair, supremacy, target_total, targets, rho,
    ou_targets=None, team_lambdas=None,
):
    """λ 拟合目标函数（越小越好）"""
    lam_h, lam_a = lam_pair
    matrix = build_score_matrix(lam_h, lam_a, rho=rho)
    margins = _matrix_margins(matrix)
    err = (
        FIT_W_1X2 * sum((margins[k] - targets[i]) ** 2 for i, k in enumerate(('home', 'draw', 'away')))
        + FIT_W_TOTAL * (lam_h + lam_a - target_total) ** 2
        + FIT_W_SUPREMACY * (lam_h - lam_a - supremacy) ** 2
    )
    if ou_targets:
        model_ou = _matrix_total_margins(matrix)
        err += FIT_W_OU_DIST * sum((model_ou[k] - ou_targets[k]) ** 2 for k in ou_targets)
    if team_lambdas:
        err += FIT_W_TEAM * (
            (lam_h - team_lambdas[0]) ** 2 + (lam_a - team_lambdas[1]) ** 2
        )
    return err


def _fit_lambda_refine(
    start, supremacy, target_total, targets, rho,
    ou_targets=None, team_lambdas=None,
):
    """网格初解后的坐标下降精调"""
    lh, la = start
    err = _lambda_fit_error((lh, la), supremacy, target_total, targets, rho, ou_targets, team_lambdas)
    step = LAMBDA_REFINE_STEP0
    for _ in range(LAMBDA_REFINE_STEPS):
        improved = False
        for dh in (step, -step, 0):
            for da in (step, -step, 0):
                if dh == 0 and da == 0:
                    continue
                trial = (max(0.08, lh + dh), max(0.08, la + da))
                te = _lambda_fit_error(
                    trial, supremacy, target_total, targets, rho, ou_targets, team_lambdas,
                )
                if te + 1e-9 < err:
                    lh, la, err = trial[0], trial[1], te
                    improved = True
        if not improved:
            step *= 0.55
            if step < 0.008:
                break
    return lh, la


def _fit_lambda_grid(
    supremacy, target_total, p_home, p_draw, p_away, rho=0.0,
    ou_targets=None, team_lambdas=None, euro_lambdas=None,
):
    """λ 网格搜索：1X2 + 反推净胜球 + 大小球分布 + 球队/欧赔先验"""
    targets = (p_home, p_draw, p_away)
    if euro_lambdas:
        best = euro_lambdas
    elif team_lambdas:
        best = team_lambdas
    else:
        best = estimate_lambdas(supremacy, target_total)
    best_err = float('inf')

    def _search(step, center=None, radius=2.5):
        nonlocal best, best_err
        if center is None:
            starts = [i * step for i in range(int(2.6 / step) + 1)]
            pairs = ((lh, la) for lh in starts for la in starts)
        else:
            lh0, la0 = center
            n = int(radius / step) + 1
            pairs = (
                (max(0.08, lh0 + di * step), max(0.08, la0 + dj * step))
                for di in range(-n, n + 1)
                for dj in range(-n, n + 1)
            )
        for lam_h, lam_a in pairs:
            err = _lambda_fit_error(
                (lam_h, lam_a), supremacy, target_total, targets, rho, ou_targets, team_lambdas,
            )
            if err < best_err:
                best_err = err
                best = (lam_h, lam_a)
        return best

    lh, la = _search(LAMBDA_COARSE_STEP)
    lh, la = _search(LAMBDA_FINE_STEP, center=(lh, la), radius=LAMBDA_FINE_RADIUS)
    return _fit_lambda_refine(
        (lh, la), supremacy, target_total, targets, rho, ou_targets, team_lambdas,
    )


def _estimate_dc_rho(lam_home, lam_away, p_draw_target):
    """根据欧赔平局概率估计 Dixon-Coles 相关系数 ρ（负值抬高 0-0/1-1 权重）"""
    base = build_score_matrix(lam_home, lam_away, rho=0.0)
    p_draw_base = _matrix_margins(base)['draw']
    gap = p_draw_target - p_draw_base
    if gap > 0.025:
        return -0.16
    if gap < -0.015:
        return -0.06
    return -0.11


def build_score_matrix(lam_home, lam_away, max_goals=MAX_GOALS, rho=0.0):
    """泊松比分矩阵；rho≠0 时施加 Dixon-Coles 低比分修正并归一化"""
    cells = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            cells[(h, a)] = (
                _dc_tau(h, a, lam_home, lam_away, rho)
                * _poisson_pmf(h, lam_home)
                * _poisson_pmf(a, lam_away)
            )
    total = sum(cells.values())
    if total <= 0:
        return cells
    return {cell: prob / total for cell, prob in cells.items()}


def calibrate_to_euro(matrix, p_home, p_draw, p_away):
    """按欧赔 1X2 缩放矩阵（保留作兜底；主流程已用 λ 拟合替代）"""
    targets = {'home': p_home, 'draw': p_draw, 'away': p_away}
    model = _matrix_margins(matrix)
    adjusted = {}
    for (h, a), prob in matrix.items():
        outcome = _outcome(h, a)
        scale = targets[outcome] / model[outcome] if model[outcome] > 0 else 0.0
        adjusted[(h, a)] = prob * scale
    total = sum(adjusted.values())
    if total <= 0:
        return matrix
    return {cell: prob / total for cell, prob in adjusted.items()}


def fit_lambdas_from_markets(
    supremacy, total_line, p_over,
    p_home, p_draw, p_away,
    open_total_line=None, team_strength=None, euro_lambdas=None,
    league_profile=None,
):
    """大小球反推总进球 + 反推净胜球 + 欧赔/球队先验，网格+坐标下降拟合 λ"""
    line = _blend_close_open(total_line, open_total_line)
    lp = league_profile or LEAGUE_PROFILES['default']
    avg_goal = lp.get('avg_goal', AVG_LEAGUE_GOAL)
    target_total = implied_total_goals(line, p_over)
    target_total = max(avg_goal * 1.4, min(avg_goal * 3.2, target_total))
    ou_targets = _ou_total_distribution(target_total)
    team_lams = None
    if team_strength:
        team_lams = team_poisson_lambdas(team_strength, target_total, lp)
    lam_home, lam_away = _fit_lambda_grid(
        supremacy, target_total, p_home, p_draw, p_away, rho=0.0,
        ou_targets=ou_targets, team_lambdas=team_lams, euro_lambdas=euro_lambdas,
    )
    rho = _estimate_dc_rho(lam_home, lam_away, p_draw)
    lam_home, lam_away = _fit_lambda_grid(
        supremacy, target_total, p_home, p_draw, p_away, rho=rho,
        ou_targets=ou_targets, team_lambdas=team_lams, euro_lambdas=euro_lambdas,
    )
    return lam_home, lam_away, target_total, rho


def _baseline_freq(h, a, league_profile=None):
    base = SCORE_BASELINE_FREQ.get((h, a), 0.018)
    if not league_profile:
        return base
    low_mult = league_profile.get('low_score', 1.0)
    draw_mult = league_profile.get('draw_mult', 1.0)
    if h == a:
        return base * draw_mult
    if h + a <= 2:
        return base * low_mult
    if h + a >= 4:
        return base / max(low_mult, 0.85)
    return base


def score_heat_label(h, a, model_prob, league_profile=None):
    """
    比分冷热：模型概率 vs 历史常见比分基准。
    冷=相对基准偏高（模型更看好但市场常忽视）；热=相对基准偏低（过热难出）。
    """
    base = _baseline_freq(h, a, league_profile)
    if base <= 0:
        return 'neutral', 1.0
    ratio = model_prob / base
    if ratio >= HEAT_RATIO_COLD:
        return 'cold', ratio
    if ratio <= HEAT_RATIO_HOT:
        return 'hot', ratio
    return 'neutral', ratio


def _heat_filter_weight(heat):
    if heat == 'hot':
        return HEAT_FILTER_PENALTY
    if heat == 'cold':
        return COLD_FILTER_BONUS
    return 1.0


def predict_scores(asian, euro, total, team_strength=None, league_profile=None):
    """泊松 + DC：多市场反推净胜球 + 走势/状态修正 + 联合拟合 λ"""
    p_home = euro['close']['home']
    p_draw = euro['close']['draw']
    p_away = euro['close']['away']
    p_over = total['close_prob']['over']
    line = total['close_line']
    open_line = total.get('open_line')

    target_total_pre = total.get('implied_total') or implied_total_goals(line, p_over)
    sup_asian = asian.get('implied_supremacy')
    if sup_asian is None:
        sup_asian = asian_implied_supremacy(
            asian['handicap'], asian['close_prob']['home_recv'],
            asian['close_prob']['away_give'], target_total_pre,
            open_handicap=asian.get('open_handicap'),
            open_hp=asian['open_prob']['home_recv'],
            open_ap=asian['open_prob']['away_give'],
        )
    sup_euro = euro.get('implied_supremacy')
    if sup_euro is None:
        sup_euro = euro_implied_supremacy(p_home, p_draw, p_away, target_total_pre)
    supremacy = blend_market_supremacy(sup_asian, sup_euro)
    mom = euro.get('momentum') or {}
    supremacy += mom.get('shift_supremacy', 0) * MOMENTUM_SUPREMACY_WEIGHT
    if team_strength:
        supremacy += team_strength.get('momentum_supremacy', 0) * 0.35

    euro_lams = None
    el = euro.get('implied_lambdas')
    if el:
        euro_lams = (el['home'], el['away'])

    try:
        lam_home, lam_away, target_total, rho = fit_lambdas_from_markets(
            supremacy, line, p_over, p_home, p_draw, p_away,
            open_total_line=open_line, team_strength=team_strength, euro_lambdas=euro_lams,
            league_profile=league_profile,
        )
        matrix = build_score_matrix(lam_home, lam_away, rho=rho)
        margins = _matrix_margins(matrix)
        err = sum(
            (margins[k] - t) ** 2
            for k, t in zip(('home', 'draw', 'away'), (p_home, p_draw, p_away))
        )
        if err > 0.012:
            matrix = calibrate_to_euro(matrix, p_home, p_draw, p_away)
    except (ValueError, ZeroDivisionError, OverflowError):
        lam_home, lam_away = estimate_lambdas(supremacy, line)
        matrix = build_score_matrix(lam_home, lam_away)
        matrix = calibrate_to_euro(matrix, p_home, p_draw, p_away)
        target_total = line
        supremacy = supremacy

    candidates = sorted(matrix.items(), key=lambda kv: -kv[1])
    meta = {
        'supremacy_asian': sup_asian,
        'supremacy_euro': sup_euro,
        'supremacy_blended': supremacy,
        'target_total': target_total,
    }
    return candidates, lam_home, lam_away, meta


# ===================== 综合分析 =====================

def _result_label(h, a):
    return "主胜" if h > a else "平局" if h == a else "客胜"


def _score_entry(h, a, prob, heat_info=None):
    entry = {'home': h, 'away': a, 'prob': prob, 'result': _result_label(h, a)}
    if heat_info:
        entry['heat'] = heat_info[0]
        entry['heat_ratio'] = round(heat_info[1], 2)
    return entry


def _alignment_score(h, a, asian, euro, total):
    """赔率信号一致性得分（0~1），用于在概率接近时优选更贴合市场的比分"""
    diff = h - a
    favor, diff_range, hcap = asian['favor'], asian['diff_range'], asian['handicap']
    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']
    lo, hi = total['expected_goals']
    score = 0.0

    if favor == 'home' and diff > 0 and diff_range[0] <= diff <= diff_range[1]:
        score += 0.35
    elif favor == 'away' and diff < 0 and diff_range[0] <= -diff <= diff_range[1]:
        score += 0.35
    elif favor == 'even' and diff == 0:
        score += 0.25

    top = max(p_home, p_draw, p_away)
    if diff > 0 and p_home >= top - 0.03:
        score += 0.3
    elif diff < 0 and p_away >= top - 0.03:
        score += 0.3
    elif diff == 0 and p_draw >= top - 0.03:
        score += 0.3

    goals = h + a
    if lo <= goals <= hi:
        score += 0.25
    elif abs(goals - (lo + hi) / 2) <= 1.0:
        score += 0.12

    if total['lean'] == 'over' and goals >= total['close_line']:
        score += 0.1
    elif total['lean'] == 'under' and goals <= total['close_line']:
        score += 0.1

    return min(1.0, score)


def _recommend_reasons(h, a, asian, euro, total, team=None, heat=None):
    """为单个推荐比分生成理由列表"""
    diff = h - a
    favor, diff_range, hcap = asian['favor'], asian['diff_range'], asian['handicap']
    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']
    lo, hi = total['expected_goals']

    reasons = []
    if team:
        reasons.append(f"攻防强度 λ≈{team['attack_home']:.2f}/{team['attack_away']:.2f}进")
        if team.get('form_diff', 0) > 0.35 and diff > 0:
            reasons.append('主队近期状态更好')
        elif team.get('form_diff', 0) < -0.35 and diff < 0:
            reasons.append('客队近期状态更好')
    mom = euro.get('momentum') or {}
    if mom.get('summary') and mom['summary'] != '欧赔走势平稳':
        reasons.append(mom['summary'])
    if favor == 'home' and diff > 0 and diff_range[0] <= diff <= diff_range[1]:
        reasons.append(f"符合主让{hcap}球盘口预期")
    elif favor == 'away' and diff < 0 and diff_range[0] <= -diff <= diff_range[1]:
        reasons.append(f"符合客让{abs(hcap)}球盘口预期")
    elif diff == 0:
        reasons.append("欧赔平局概率支撑")
    if diff > 0 and p_home > 0.4:
        reasons.append(f"欧赔主胜概率{p_home*100:.0f}%")
    elif diff < 0 and p_away > 0.4:
        reasons.append(f"欧赔客胜概率{p_away*100:.0f}%")
    elif diff == 0 and p_draw > 0.3:
        reasons.append(f"欧赔平局概率{p_draw*100:.0f}%")
    if lo <= h + a <= hi:
        reasons.append(f"总进球{h+a}球在预期区间")
    if heat == 'cold':
        reasons.append("冷门口比分（模型概率高于历史基准）")
    elif heat == 'hot':
        reasons.append("热门比分（已降权）")
    kelly = euro.get('kelly')
    if kelly:
        fav = kelly.get('favored')
        hard = kelly.get('hardest')
        if fav == 'home' and diff > 0:
            reasons.append("凯利指数相对看好主胜")
        elif fav == 'away' and diff < 0:
            reasons.append("凯利指数相对看好客胜")
        elif fav == 'draw' and diff == 0:
            reasons.append("凯利指数相对看好平局")
        if hard == 'home' and diff > 0:
            reasons.append("凯利提示主胜打出难度偏大")
        elif hard == 'away' and diff < 0:
            reasons.append("凯利提示客胜打出难度偏大")
    return reasons or ["综合赔率推断"]


def _pick_recommendations(candidates, asian, euro, total, n=2, pool=12, confidence=None, league_profile=None):
    """Top 池内按 概率×一致性×冷热×置信度 选取"""
    if confidence:
        n = confidence.get('recommend_count', n)
    pool = min(pool, len(candidates))
    conf_w = confidence['score'] if confidence else 1.0
    scored = []
    for (h, a), prob in candidates[:pool]:
        align = _alignment_score(h, a, asian, euro, total)
        heat, _ = score_heat_label(h, a, prob, league_profile)
        w = _heat_filter_weight(heat)
        scored.append(((h, a), prob, align, heat, prob * (1.0 + 0.45 * align) * w * (0.65 + 0.35 * conf_w)))
    scored.sort(key=lambda x: -x[4])
    seen = set()
    picked = []
    for (h, a), prob, _, _heat, _ in scored:
        if (h, a) in seen:
            continue
        seen.add((h, a))
        picked.append((h, a, prob))
        if len(picked) >= n:
            break
    if len(picked) < n:
        for (h, a), prob in candidates:
            if (h, a) not in seen:
                picked.append((h, a, prob))
                if len(picked) >= n:
                    break
    return picked


def analyze_match(match):
    """抓取赔率 + 球队攻防 + 泊松模型，返回完整结果 dict"""
    mid = match['match_id']
    home, away = match.get('home', ''), match.get('away', '')
    league_profile = resolve_league_profile(match.get('league', ''))

    asian = analyze_asian(fetch_yazhi(mid))
    euro = analyze_euro(fetch_ouzhi(mid))
    total = analyze_total(fetch_daxiao(mid))
    team = fetch_team_strength(mid, home, away)
    if team:
        team['league_profile'] = league_profile

    target_total = total['implied_total']
    lp_avg = league_profile.get('avg_goal', AVG_LEAGUE_GOAL)
    target_total = max(lp_avg * 1.4, min(lp_avg * 3.2, target_total))
    total['implied_total'] = target_total

    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']

    asian['implied_supremacy'] = asian_implied_supremacy(
        asian['handicap'], asian['close_prob']['home_recv'], asian['close_prob']['away_give'],
        target_total, open_handicap=asian.get('open_handicap'),
        open_hp=asian['open_prob']['home_recv'], open_ap=asian['open_prob']['away_give'],
    )
    euro['implied_supremacy'] = euro_implied_supremacy(p_home, p_draw, p_away, target_total)
    euro['implied_lambdas'] = dict(
        zip(('home', 'away'), euro_implied_lambdas(p_home, p_draw, p_away, target_total))
    )

    confidence = compute_prediction_confidence(asian, euro, total, team)

    candidates, lam_home, lam_away, meta = predict_scores(
        asian, euro, total, team_strength=team, league_profile=league_profile,
    )

    top_scores = [
        _score_entry(h, a, prob, score_heat_label(h, a, prob, league_profile))
        for (h, a), prob in candidates[:5]
    ]
    recommend = []
    for h, a, prob in _pick_recommendations(
        candidates, asian, euro, total, confidence=confidence, league_profile=league_profile,
    ):
        heat, _ = score_heat_label(h, a, prob, league_profile)
        recommend.append({
            **_score_entry(h, a, prob, (heat, _)),
            'reasons': _recommend_reasons(h, a, asian, euro, total, team, heat=heat),
        })

    return {
        'match': {k: match.get(k) for k in ('home', 'away', 'league', 'time', 'match_id')},
        'league_profile': league_profile,
        'asian': asian,
        'euro': euro,
        'total': total,
        'team': team,
        'confidence': confidence,
        'model': {
            'lam_home': lam_home, 'lam_away': lam_away,
            'top_scores': top_scores, 'recommend': recommend,
            **meta,
        },
    }


# ===================== 主程序 =====================

def main():
    print("=" * 65)
    print("  足球比分预测脚本 - 动态赔率分析")
    print("  数据来源: odds.500.com（多家博彩公司平均值）")
    print("=" * 65)

    # ── 输入球队名称 ──
    home_kw = input("请输入主队名称（支持关键词）: ").strip()
    away_kw = input("请输入客队名称（支持关键词）: ").strip()

    if not home_kw and not away_kw:
        print("  ⚠ 未输入任何球队名，退出。")
        return

    # ── 抓取比赛列表 ──
    try:
        matches = fetch_match_list()
    except Exception as e:
        print(f"\n  ✗ 获取比赛列表失败: {e}")
        print("  请检查网络连接后重试。")
        return

    if not matches:
        print("\n  ✗ 未找到任何比赛数据，请稍后重试。")
        return

    print(f"  共找到 {len(matches)} 场比赛")

    # ── 搜索匹配比赛 ──
    found = search_match(matches, home_kw, away_kw)

    if not found:
        print(f"\n  ✗ 未找到匹配 '{home_kw} vs {away_kw}' 的比赛。")
        print(f"\n  今日比赛列表：")
        for i, m in enumerate(matches, 1):
            league_str = f"[{m.get('league', '?')}]" if m.get('league') else ''
            time_str = f" {m.get('time', '')}" if m.get('time') else ''
            print(f"  {i:2d}. {league_str} {m['home']} vs {m['away']}{time_str}  (ID:{m['match_id']})")
        return

    if len(found) == 1:
        match = found[0]
    else:
        print(f"\n  找到 {len(found)} 场匹配比赛：")
        for i, m in enumerate(found, 1):
            league_str = f"[{m.get('league', '?')}]" if m.get('league') else ''
            print(f"  {i}. {league_str} {m['home']} vs {m['away']}  (ID:{m['match_id']})")
        try:
            choice = int(input(f"  请选择 (1-{len(found)}): ").strip())
            match = found[choice - 1]
        except (ValueError, IndexError):
            print("  输入无效，默认选择第1场")
            match = found[0]

    print(f"\n  已选择: {match['home']} vs {match['away']} (ID:{match['match_id']})")
    print("\n  正在抓取赔率数据...")

    try:
        result = analyze_match(match)
    except Exception as e:
        print(f"  ✗ 赔率数据获取失败: {e}")
        return

    render_cli(result)


def _heat_tag(heat):
    return {'cold': '❄冷', 'hot': '🔥热', 'neutral': '—'}.get(heat, '—')


def render_cli(result):
    """将 analyze_match 的结果渲染为命令行报告"""
    match, asian, euro, total, model = (
        result['match'], result['asian'], result['euro'], result['total'], result['model']
    )
    team = result.get('team')
    confidence = result.get('confidence')
    lp = result.get('league_profile') or {}
    home, away = match['home'], match['away']

    print("\n" + "=" * 65)
    print(f"  赔率分析 | {home} vs {away}")
    if lp.get('name') and lp['name'] != 'default':
        print(f"  联赛模型: {lp['name']}（场均进球基准 {lp.get('avg_goal', AVG_LEAGUE_GOAL):.2f}）")
    if confidence:
        print(f"  预测置信度: {confidence['label']} ({confidence['score']*100:.0f}%)")
        if confidence.get('notes'):
            print(f"  说明: {'；'.join(confidence['notes'])}")
    print("=" * 65)

    op, cl = asian['open_prob'], asian['close_prob']
    print("\n【亚盘分析】（多家博彩公司平均值）")
    print(f"  让球变化: {asian['handicap_trend']}")
    print(f"  水位变化: {asian['water_trend']}")
    print(f"  初盘真实概率: 主受让方 {op['home_recv']*100:.1f}% / 让球方 {op['away_give']*100:.1f}%")
    print(f"  终盘真实概率: 主受让方 {cl['home_recv']*100:.1f}% / 让球方 {cl['away_give']*100:.1f}%")
    print(f"  终盘判断: {asian['favor_desc']}，{asian['diff_desc']}")
    if asian.get('implied_supremacy') is not None:
        print(f"  反推净胜球: {asian['implied_supremacy']:+.2f}（非盘口线 {asian['handicap']:+.2f}）")

    eo, ec = euro['open'], euro['close']
    print("\n【欧赔分析】（多家博彩公司平均值）")
    print(f"  初盘: 主胜{eo['home']*100:.1f}% | 平{eo['draw']*100:.1f}% | 客胜{eo['away']*100:.1f}%")
    print(f"  终盘: 主胜{ec['home']*100:.1f}% | 平{ec['draw']*100:.1f}% | 客胜{ec['away']*100:.1f}%")
    print(f"  变化趋势: {', '.join(euro['changes']) if euro['changes'] else '赔率稳定'}")
    mom = euro.get('momentum') or {}
    if mom.get('summary'):
        print(f"  欧赔走势: {mom['summary']}")
    if euro.get('implied_supremacy') is not None:
        print(f"  欧赔反推净胜球: {euro['implied_supremacy']:+.2f}")
    el = euro.get('implied_lambdas')
    if el:
        print(f"  欧赔隐含 λ: 主{el['home']:.2f} / 客{el['away']:.2f}")

    kelly = euro.get('kelly')
    if kelly:
        ko, kc = kelly['open'], kelly['close']
        rr = kelly['return_rate']['close']
        print("\n【凯利指数分析】（欧赔 × 去水概率 × 100）")
        print(f"  理论返还率: 初盘{kelly['return_rate']['open']:.1f}% → 终盘{rr:.1f}%")
        print(f"  初盘凯利: 主胜{ko['home']:.1f} | 平{ko['draw']:.1f} | 客胜{ko['away']:.1f}")
        print(f"  终盘凯利: 主胜{kc['home']:.1f} | 平{kc['draw']:.1f} | 客胜{kc['away']:.1f}")
        if kelly['kelly_changes']:
            print(f"  凯利变化: {', '.join(kelly['kelly_changes'])}")
        if kelly['risks']:
            print(f"  风险提示: {'；'.join(kelly['risks'])}")
        if kelly['favors']:
            print(f"  相对看好: {'；'.join(kelly['favors'])}")
        print(f"  综合: {kelly['summary']}")

    to, tc = total['open_prob'], total['close_prob']
    print("\n【大小球分析】（多家博彩公司平均值）")
    print(f"  初盘: 线{total['open_line']} | 大{to['over']*100:.1f}% / 小{to['under']*100:.1f}%")
    print(f"  终盘: 线{total['close_line']} | 大{tc['over']*100:.1f}% / 小{tc['under']*100:.1f}%")
    print(f"  判断: {total['lean_desc']}")
    print(f"  泊松反推总进球: {total.get('implied_total', 0):.2f}")
    print(f"  期望总进球区间: {total['expected_goals'][0]}-{total['expected_goals'][1]}球")

    if team:
        print("\n【球队攻防强度】（500.com 近10场 + 主客场）")
        print(f"  {team['summary']}")
        print(f"  主队进攻{team['attack_home']:.2f}球/场 防守{team['defense_home']:.2f}失/场")
        print(f"  客队进攻{team['attack_away']:.2f}球/场 防守{team['defense_away']:.2f}失/场")

    print("\n" + "=" * 65)
    print("【综合信号汇总】")
    print("=" * 65)
    hcap = asian['handicap']
    if hcap > 0:
        print(f"  强弱判断: 主队较强（主让{hcap}球）")
    elif hcap < 0:
        print(f"  强弱判断: 客队较强（客让{abs(hcap)}球）")
    else:
        print("  强弱判断: 双方实力接近")
    dominant = max([('主胜', ec['home']), ('平局', ec['draw']), ('客胜', ec['away'])], key=lambda x: x[1])
    print(f"  欧赔最高概率结果: {dominant[0]} ({dominant[1]*100:.1f}%)")
    print(f"  期望总进球: {total['expected_goals'][0]}-{total['expected_goals'][1]} 球")
    if model.get('supremacy_blended') is not None:
        print(f"  融合净胜球: 亚{model.get('supremacy_asian', 0):+.2f} + 欧{model.get('supremacy_euro', 0):+.2f} → {model['supremacy_blended']:+.2f}")
    print(f"  泊松期望进球: 主队 λ={model['lam_home']:.2f} / 客队 λ={model['lam_away']:.2f}")

    print("\n" + "=" * 65)
    print("【Top 5 候选比分】（含冷热标记）")
    print("=" * 65)
    top_prob = model['top_scores'][0]['prob']
    for i, s in enumerate(model['top_scores'], 1):
        bar = "█" * int(s['prob'] / top_prob * 20)
        heat = _heat_tag(s.get('heat', 'neutral'))
        print(f"  #{i}: {home} {s['home']} - {s['away']} {away}  [{s['result']}]  "
              f"概率:{s['prob']*100:.1f}%  {heat}  {bar}")

    rec_n = len(model['recommend'])
    print("\n" + "=" * 65)
    print(f"【推荐：最可能的{rec_n}个比分】")
    print("=" * 65)
    for rank, s in enumerate(model['recommend'], 1):
        print(f"\n  第{rank}推荐: ★ {home} {s['home']} : {s['away']} {away} ★")
        print(f"           结果: {s['result']}  |  比分概率: {s['prob']*100:.1f}%")
        print(f"           理由: {' / '.join(s['reasons'])}")

    print("\n")
    print("  ⚠  免责声明：以上分析仅为概率统计参考，体育赛事结果受多种")
    print("     不确定因素影响，不构成任何投注建议。请理性对待！")
    print("=" * 65)


if __name__ == '__main__':
    main()
