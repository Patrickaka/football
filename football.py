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

# 联赛场均进球基准（用于球队攻防强度归一化）
AVG_LEAGUE_GOAL = 1.35
HOME_VENUE_ATTACK_BOOST = 1.06

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
        'open': {'home': open_[0], 'draw': open_[1], 'away': open_[2]},
        'close': {'home': close[0], 'draw': close[1], 'away': close[2]},
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


def _parse_recent_form(groups):
    n, gf, ga = int(groups[0]), int(groups[4]), int(groups[5])
    n = max(n, 1)
    return {'games': n, 'gf': gf, 'ga': ga, 'attack': gf / n, 'defense': ga / n}


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

    return {
        'home_recent': home_all,
        'away_recent': away_all,
        'home_venue': hv,
        'away_venue': av,
        'attack_home': attack_home,
        'defense_home': defense_home,
        'attack_away': attack_away,
        'defense_away': defense_away,
        'summary': (
            f"主队近{home_all['games']}场 进{home_all['gf']}失{home_all['ga']}；"
            f"客队近{away_all['games']}场 进{away_all['gf']}失{away_all['ga']}"
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


def analyze_euro(data):
    """解析欧赔，返回初终盘 1X2 真实概率与变化趋势"""
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

    return {
        'open': {'home': ph_o, 'draw': pd_o, 'away': pa_o},
        'close': {'home': ph_c, 'draw': pd_c, 'away': pa_c},
        'raw_odds': {'open': dict(op), 'close': dict(cl)},
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
    """融合亚盘与欧赔反推的净胜球"""
    return SUP_ASIAN_WEIGHT * sup_asian + SUP_EURO_WEIGHT * sup_euro


def team_poisson_lambdas(strength, total_target):
    """
    用攻防强度构造 λ：主队进攻×客队防守×主场系数。
    defense 为场均失球（对手防守弱则失球多 → 因子更大）。
    """
    avg = AVG_LEAGUE_GOAL
    atk_h = strength['attack_home'] / avg
    def_a = strength['defense_away'] / avg
    atk_a = strength['attack_away'] / avg
    def_h = strength['defense_home'] / avg
    lam_home = max(0.08, atk_h * def_a * avg * HOME_VENUE_ATTACK_BOOST)
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
            matrix = build_score_matrix(lam_h, lam_a, rho=rho)
            margins = _matrix_margins(matrix)
            err_1x2 = sum((margins[k] - targets[i]) ** 2 for i, k in enumerate(('home', 'draw', 'away')))
            err_total = (lam_h + lam_a - target_total) ** 2
            err_sup = (lam_h - lam_a - supremacy) ** 2
            err = FIT_W_1X2 * err_1x2 + FIT_W_TOTAL * err_total + FIT_W_SUPREMACY * err_sup
            if ou_targets:
                model_ou = _matrix_total_margins(matrix)
                err += FIT_W_OU_DIST * sum(
                    (model_ou[k] - ou_targets[k]) ** 2 for k in ou_targets
                )
            if team_lambdas:
                err += FIT_W_TEAM * (
                    (lam_h - team_lambdas[0]) ** 2 + (lam_a - team_lambdas[1]) ** 2
                )
            if err < best_err:
                best_err = err
                best = (lam_h, lam_a)
        return best

    lh, la = _search(LAMBDA_COARSE_STEP)
    return _search(LAMBDA_FINE_STEP, center=(lh, la), radius=LAMBDA_FINE_RADIUS)


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
):
    """大小球反推总进球 + 反推净胜球 + 欧赔/球队先验，网格拟合 λ"""
    line = _blend_close_open(total_line, open_total_line)
    target_total = implied_total_goals(line, p_over)
    ou_targets = _ou_total_distribution(target_total)
    team_lams = None
    if team_strength:
        team_lams = team_poisson_lambdas(team_strength, target_total)
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


def _baseline_freq(h, a):
    return SCORE_BASELINE_FREQ.get((h, a), 0.018)


def score_heat_label(h, a, model_prob):
    """
    比分冷热：模型概率 vs 历史常见比分基准。
    冷=相对基准偏高（模型更看好但市场常忽视）；热=相对基准偏低（过热难出）。
    """
    base = _baseline_freq(h, a)
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


def predict_scores(asian, euro, total, team_strength=None):
    """泊松 + DC：亚盘/欧赔反推净胜球 + 大小球分布 + 球队攻防 → 拟合 λ"""
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

    euro_lams = None
    el = euro.get('implied_lambdas')
    if el:
        euro_lams = (el['home'], el['away'])

    try:
        lam_home, lam_away, target_total, rho = fit_lambdas_from_markets(
            supremacy, line, p_over, p_home, p_draw, p_away,
            open_total_line=open_line, team_strength=team_strength, euro_lambdas=euro_lams,
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
    return reasons or ["综合赔率推断"]


def _pick_recommendations(candidates, asian, euro, total, n=2, pool=12):
    """Top 池内按 概率×一致性×冷热权重 选取，过热比分降权"""
    pool = min(pool, len(candidates))
    scored = []
    for (h, a), prob in candidates[:pool]:
        align = _alignment_score(h, a, asian, euro, total)
        heat, _ = score_heat_label(h, a, prob)
        w = _heat_filter_weight(heat)
        scored.append(((h, a), prob, align, heat, prob * (1.0 + 0.45 * align) * w))
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
    asian = analyze_asian(fetch_yazhi(mid))
    euro = analyze_euro(fetch_ouzhi(mid))
    total = analyze_total(fetch_daxiao(mid))
    team = fetch_team_strength(mid, home, away)

    target_total = total['implied_total']
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

    candidates, lam_home, lam_away, meta = predict_scores(asian, euro, total, team_strength=team)

    top_scores = [
        _score_entry(h, a, prob, score_heat_label(h, a, prob))
        for (h, a), prob in candidates[:5]
    ]
    recommend = []
    for h, a, prob in _pick_recommendations(candidates, asian, euro, total):
        heat, _ = score_heat_label(h, a, prob)
        recommend.append({
            **_score_entry(h, a, prob, (heat, _)),
            'reasons': _recommend_reasons(h, a, asian, euro, total, team, heat=heat),
        })

    return {
        'match': {k: match.get(k) for k in ('home', 'away', 'league', 'time', 'match_id')},
        'asian': asian,
        'euro': euro,
        'total': total,
        'team': team,
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
    home, away = match['home'], match['away']

    print("\n" + "=" * 65)
    print(f"  赔率分析 | {home} vs {away}")
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
    if euro.get('implied_supremacy') is not None:
        print(f"  欧赔反推净胜球: {euro['implied_supremacy']:+.2f}")
    el = euro.get('implied_lambdas')
    if el:
        print(f"  欧赔隐含 λ: 主{el['home']:.2f} / 客{el['away']:.2f}")

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

    print("\n" + "=" * 65)
    print("【推荐：最可能的2个比分】")
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
