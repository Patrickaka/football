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

    return {
        'open_line': op['line'], 'close_line': line,
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


def estimate_lambdas(supremacy, total_line, min_lambda=0.05):
    """由净胜球(亚盘让球)与总进球(大小球盘口)解出主客队期望进球 λ"""
    lam_home = max(min_lambda, (total_line + supremacy) / 2)
    lam_away = max(min_lambda, (total_line - supremacy) / 2)
    return lam_home, lam_away


def build_score_matrix(lam_home, lam_away, max_goals=MAX_GOALS):
    """以独立泊松分布构建比分联合概率矩阵 {(h,a): prob}"""
    return {
        (h, a): _poisson_pmf(h, lam_home) * _poisson_pmf(a, lam_away)
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
    }


def calibrate_to_euro(matrix, p_home, p_draw, p_away):
    """按欧赔 1X2 真实概率缩放比分矩阵，使三种结果的边缘概率与欧赔一致"""
    targets = {'home': p_home, 'draw': p_draw, 'away': p_away}
    model = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    for (h, a), prob in matrix.items():
        model[_outcome(h, a)] += prob

    adjusted = {}
    for (h, a), prob in matrix.items():
        outcome = _outcome(h, a)
        scale = targets[outcome] / model[outcome] if model[outcome] > 0 else 0.0
        adjusted[(h, a)] = prob * scale

    total = sum(adjusted.values())
    if total <= 0:
        return matrix
    return {cell: prob / total for cell, prob in adjusted.items()}


def predict_scores(supremacy, total_line, p_home, p_draw, p_away):
    """泊松模型：解 λ → 构建比分矩阵 → 欧赔校准 → 按真实概率降序返回"""
    lam_home, lam_away = estimate_lambdas(supremacy, total_line)
    matrix = build_score_matrix(lam_home, lam_away)
    matrix = calibrate_to_euro(matrix, p_home, p_draw, p_away)
    candidates = sorted(matrix.items(), key=lambda kv: -kv[1])
    return candidates, lam_home, lam_away


# ===================== 综合分析 =====================

def _result_label(h, a):
    return "主胜" if h > a else "平局" if h == a else "客胜"


def _score_entry(h, a, prob):
    return {'home': h, 'away': a, 'prob': prob, 'result': _result_label(h, a)}


def _recommend_reasons(h, a, asian, euro, total):
    """为单个推荐比分生成理由列表"""
    diff = h - a
    favor, diff_range, hcap = asian['favor'], asian['diff_range'], asian['handicap']
    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']
    lo, hi = total['expected_goals']

    reasons = []
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
    return reasons or ["综合赔率推断"]


def analyze_match(match):
    """抓取指定比赛三类赔率并运行泊松模型，返回完整结果 dict"""
    mid = match['match_id']
    asian = analyze_asian(fetch_yazhi(mid))
    euro = analyze_euro(fetch_ouzhi(mid))
    total = analyze_total(fetch_daxiao(mid))

    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']
    candidates, lam_home, lam_away = predict_scores(
        asian['handicap'], total['close_line'], p_home, p_draw, p_away
    )

    top_scores = [_score_entry(h, a, prob) for (h, a), prob in candidates[:5]]
    recommend = [
        {**_score_entry(h, a, prob), 'reasons': _recommend_reasons(h, a, asian, euro, total)}
        for (h, a), prob in candidates[:2]
    ]

    return {
        'match': {k: match.get(k) for k in ('home', 'away', 'league', 'time', 'match_id')},
        'asian': asian,
        'euro': euro,
        'total': total,
        'model': {
            'lam_home': lam_home, 'lam_away': lam_away,
            'top_scores': top_scores, 'recommend': recommend,
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


def render_cli(result):
    """将 analyze_match 的结果渲染为命令行报告"""
    match, asian, euro, total, model = (
        result['match'], result['asian'], result['euro'], result['total'], result['model']
    )
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

    eo, ec = euro['open'], euro['close']
    print("\n【欧赔分析】（多家博彩公司平均值）")
    print(f"  初盘: 主胜{eo['home']*100:.1f}% | 平{eo['draw']*100:.1f}% | 客胜{eo['away']*100:.1f}%")
    print(f"  终盘: 主胜{ec['home']*100:.1f}% | 平{ec['draw']*100:.1f}% | 客胜{ec['away']*100:.1f}%")
    print(f"  变化趋势: {', '.join(euro['changes']) if euro['changes'] else '赔率稳定'}")

    to, tc = total['open_prob'], total['close_prob']
    print("\n【大小球分析】（多家博彩公司平均值）")
    print(f"  初盘: 线{total['open_line']} | 大{to['over']*100:.1f}% / 小{to['under']*100:.1f}%")
    print(f"  终盘: 线{total['close_line']} | 大{tc['over']*100:.1f}% / 小{tc['under']*100:.1f}%")
    print(f"  判断: {total['lean_desc']}")
    print(f"  期望总进球区间: {total['expected_goals'][0]}-{total['expected_goals'][1]}球")

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
    print(f"  泊松期望进球: 主队 λ={model['lam_home']:.2f} / 客队 λ={model['lam_away']:.2f}")

    print("\n" + "=" * 65)
    print("【Top 5 候选比分】（欧赔校准后真实概率）")
    print("=" * 65)
    top_prob = model['top_scores'][0]['prob']
    for i, s in enumerate(model['top_scores'], 1):
        bar = "█" * int(s['prob'] / top_prob * 20)
        print(f"  #{i}: {home} {s['home']} - {s['away']} {away}  [{s['result']}]  "
              f"概率:{s['prob']*100:.1f}%  {bar}")

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
