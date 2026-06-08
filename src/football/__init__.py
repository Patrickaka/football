"""
足球比分预测脚本 - 动态抓取赔率数据
====================================
数据来源: odds.500.com（Bet365等多家博彩公司平均值）

使用方法：
  运行脚本 → 输入主队和客队名称关键词 → 自动匹配比赛 → 抓取赔率 → 分析输出

包含模块：
  - 泊松分布模型
  - Dixon-Coles 模型（改进泊松，考虑低比分相关性）
  - 进球数推荐
  - 半全场概率计算
  - ELO 评分系统
"""

import sys
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
from ..common.logger import setup_logger

# ELO 评分系统（延迟导入）
try:
    from .elo import get_elo_system, elo_to_goals_expected, elo_to_strength_factor
    ELO_AVAILABLE = True
except ImportError:
    ELO_AVAILABLE = False
    log = setup_logger('football')
    log.warning("ELO 模块未导入，将使用默认球队实力计算")

log = setup_logger('football')

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

# 离散度计算窗口（最近N条记录）
DISPERSION_WINDOW = 5

# 坐标下降精调 λ 的迭代次数与步长
LAMBDA_REFINE_STEPS = 28
LAMBDA_REFINE_STEP0 = 0.07

# 预测置信度：低于该值仅推荐 1 个比分
CONFIDENCE_LOW_THRESHOLD = 0.52
CONFIDENCE_HIGH_THRESHOLD = 0.72

# 联赛画像：场均进球、主场加成、低比分倾向（乘在 0-2 球基准频率上）
LEAGUE_PROFILES = {
    'default': {'avg_goal': 1.42, 'home_boost': 1.06, 'low_score': 0.92, 'draw_mult': 1.0},
    '英超': {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88, 'draw_mult': 0.95},
    '英冠': {'avg_goal': 1.46, 'home_boost': 1.07, 'low_score': 0.90, 'draw_mult': 0.96},
    '西甲': {'avg_goal': 1.42, 'home_boost': 1.07, 'low_score': 0.95, 'draw_mult': 1.05},
    '意甲': {'avg_goal': 1.32, 'home_boost': 1.05, 'low_score': 1.05, 'draw_mult': 1.08},
    '德甲': {'avg_goal': 1.56, 'home_boost': 1.06, 'low_score': 0.86, 'draw_mult': 0.94},
    '法甲': {'avg_goal': 1.36, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.02},
    '荷甲': {'avg_goal': 1.58, 'home_boost': 1.05, 'low_score': 0.85, 'draw_mult': 0.93},
    '葡超': {'avg_goal': 1.34, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.03},
    '欧冠': {'avg_goal': 1.50, 'home_boost': 1.04, 'low_score': 0.92, 'draw_mult': 0.98},
    '欧联': {'avg_goal': 1.44, 'home_boost': 1.05, 'low_score': 0.94, 'draw_mult': 1.0},
    '世界杯': {'avg_goal': 1.42, 'home_boost': 1.03, 'low_score': 0.96, 'draw_mult': 1.0},
    '欧洲杯': {'avg_goal': 1.40, 'home_boost': 1.04, 'low_score': 0.98, 'draw_mult': 1.02},
    '友谊': {'avg_goal': 1.44, 'home_boost': 1.02, 'low_score': 0.95, 'draw_mult': 0.97},
    '国际': {'avg_goal': 1.42, 'home_boost': 1.03, 'low_score': 0.96, 'draw_mult': 1.0},
    '巴甲': {'avg_goal': 1.42, 'home_boost': 1.08, 'low_score': 0.92, 'draw_mult': 0.96},
    '阿甲': {'avg_goal': 1.36, 'home_boost': 1.07, 'low_score': 0.96, 'draw_mult': 0.98},
    '中超': {'avg_goal': 1.32, 'home_boost': 1.07, 'low_score': 1.00, 'draw_mult': 1.04},
    '日职': {'avg_goal': 1.34, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.03},
    '韩K': {'avg_goal': 1.30, 'home_boost': 1.06, 'low_score': 1.02, 'draw_mult': 1.04},
}

# 比分冷热：相对历史基准频率的比值阈值
HEAT_RATIO_HOT = 0.70
HEAT_RATIO_COLD = 1.32
HEAT_FILTER_PENALTY = 0.75   # 原0.62，缩小热分惩罚，避免高比分被过度压制
COLD_FILTER_BONUS = 1.08     # 原1.18，缩小冷门奖励，防止低比分通过"冷门"机制反复被加权

# 常见比分历史基准频率（用于冷热，非市场赔率）——参考欧洲主流联赛真实分布上调
SCORE_BASELINE_FREQ = {
    (0, 0): 0.075, (1, 0): 0.085, (0, 1): 0.065, (1, 1): 0.110,
    (2, 0): 0.078, (0, 2): 0.055, (2, 1): 0.105, (1, 2): 0.068,
    (2, 2): 0.045, (3, 0): 0.042, (0, 3): 0.025, (3, 1): 0.052,
    (1, 3): 0.032, (3, 2): 0.035, (2, 3): 0.025, (4, 0): 0.020,
    (0, 4): 0.012, (4, 1): 0.025, (1, 4): 0.014, (3, 3): 0.015,
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
    start = time.perf_counter()
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
            log.debug('fetch %s → %d bytes (%.3fs)', url, len(raw), time.perf_counter() - start)
            return result
        except (UnicodeDecodeError, LookupError, UnicodeEncodeError):
            continue
    log.debug('fetch %s → %d bytes (%.3fs)', url, len(raw), time.perf_counter() - start)
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
    log.info('获取比赛列表')
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

    # 提取联赛和时间（通过match_id关联）
    # 创建 match_id -> time 的映射（基于表格行结构）
    match_time_map = {}
    
    # 基于表格行结构的匹配：<td>时间</td>...<a href="...shuju-ID.shtml">
    # 时间格式：<td rowspan="2">06-06 13:00</td>
    time_row_pat = re.compile(
        r'<td[^>]*?rowspan="2"[^>]*?>(\d{2}-\d{2}\s+\d{2}:\d{2})</td>.*?'
        r'shuju-(\d+)\.shtml',
        re.DOTALL
    )
    for m in time_row_pat.finditer(html):
        time_val = m.group(1)
        match_id = m.group(2)
        if match_id not in match_time_map:
            match_time_map[match_id] = time_val
    
    # 如果上面的模式没找到，尝试其他模式
    if not match_time_map:
        time_patterns = [
            r'shuju-(\d+)\.shtml.*?(\d{2}-\d{2}\s+\d{2}:\d{2})',
            r'(\d{2}-\d{2}\s+\d{2}:\d{2}).*?shuju-(\d+)\.shtml',
        ]
        for pattern in time_patterns:
            time_row_pat = re.compile(pattern, re.DOTALL)
            for m in time_row_pat.finditer(html):
                if m.group(1).isdigit():
                    match_id = m.group(1)
                    time_val = m.group(2)
                else:
                    match_id = m.group(2)
                    time_val = m.group(1)
                
                if match_id not in match_time_map:
                    match_time_map[match_id] = time_val
    
    # 创建 match_id -> 竞彩编号 的映射（如 周三201），编号星期前缀可用于按时间分组
    # 布局: <input ... value="<match_id>" />周三201
    match_num_map = dict(
        re.findall(r'value="(\d+)"\s*/>\s*(周[一二三四五六日]\d{3})', html)
    )

    # 创建 match_id -> league 的映射（基于联赛区块结构）
    match_league_map = {}
    
    # 查找所有联赛区块（联赛名称后面跟着该联赛的比赛）
    # 模式：联赛链接...比赛列表...下一个联赛链接
    league_blocks = re.split(r'<a[^>]*href="//liansai\.500\.com/zuqiu-\d+/"[^>]*>([^<]+)</a>', html)
    
    current_league = ''
    for i, block in enumerate(league_blocks):
        if i % 2 == 1:
            # 这是联赛名称
            current_league = block.strip()
        else:
            # 这是联赛区块内容，提取其中的比赛ID
            match_ids_in_block = re.findall(r'shuju-(\d+)\.shtml', block)
            for match_id in match_ids_in_block:
                if match_id not in match_league_map:
                    match_league_map[match_id] = current_league

    # 将时间和联赛添加到比赛信息中
    for match in matches:
        match_id = match['match_id']
        if match_id in match_time_map:
            match['time'] = match_time_map[match_id]
        if match_id in match_league_map:
            match['league'] = match_league_map[match_id].strip()
        if match_id in match_num_map:
            match['num'] = match_num_map[match_id]

    # 如果通过行匹配没有找到时间，则回退到原来的方法
    if not match_time_map:
        league_pat = re.compile(r'<a[^>]*href="//liansai\.500\.com/zuqiu-\d+/"[^>]*>([^<]+)</a>')
        time_pat = re.compile(r'(\d{2}-\d{2}\s+\d{2}:\d{2})')

        leagues = league_pat.findall(html)
        times = time_pat.findall(html)

        for i, match in enumerate(matches):
            if 'league' not in match and i < len(leagues):
                match['league'] = leagues[i].strip()
            if 'time' not in match and i < len(times):
                match['time'] = times[i]

    log.info('获取到 %d 场比赛', len(matches))
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
    
    # 数据有效性检查
    if not isinstance(series, list):
        raise ValueError(f"欧赔数据格式错误，期望列表但得到: {type(series)} (match_id={match_id})")
    
    if len(series) == 0:
        raise ValueError(f"欧赔数据为空列表 (match_id={match_id})")
    
    # 检查数据点格式
    close = series[0]
    open_ = series[-1]
    
    if not isinstance(close, (list, tuple)) or len(close) < 3:
        raise ValueError(f"终盘数据格式错误: {close} (match_id={match_id})")
    
    if not isinstance(open_, (list, tuple)) or len(open_) < 3:
        raise ValueError(f"初盘数据格式错误: {open_} (match_id={match_id})")

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


def fetch_team_strength(match_id, home, away, league_profile=None):
    """
    从数据分析页抓取主客队近10场及主客场进球/失球，换算攻防强度。
    返回 None 表示页面无数据（不影响主流程）。
    
    集成 ELO 评分系统：
    - 获取球队 ELO 评分
    - 将 ELO 转换为进球期望值 (xG)
    - 返回包含 ELO 信息的综合实力数据
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
    
    # ELO 评分集成
    elo_home = elo_away = None
    elo_xg_home = elo_xg_away = None
    elo_strength_home = elo_strength_away = None
    elo_prediction = None
    
    if ELO_AVAILABLE:
        try:
            elo = get_elo_system()
            elo_home = elo.get_rating(home)
            elo_away = elo.get_rating(away)
            
            # 计算基于 ELO 的进球期望值
            elo_xg_home = elo_to_goals_expected(elo_home, elo_away)
            elo_xg_away = elo_to_goals_expected(elo_away, elo_home)
            
            # 计算实力因子
            elo_strength_home = elo_to_strength_factor(elo_home)
            elo_strength_away = elo_to_strength_factor(elo_away)
            
            # 获取 ELO 预测
            league_type = league_profile.get('name', '联赛') if league_profile else '联赛'
            elo_prediction = elo.predict_match(home, away, league_type)
            
            log.debug(f"ELO 评分: {home}={elo_home:.2f}, {away}={elo_away:.2f}")
            log.debug(f"ELO xG: {home}={elo_xg_home:.2f}, {away}={elo_xg_away:.2f}")
        except Exception as e:
            log.error(f"ELO 计算失败: {e}")

    result = {
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
    
    # 添加 ELO 相关数据
    if ELO_AVAILABLE and elo_home is not None:
        result.update({
            'elo_home': elo_home,
            'elo_away': elo_away,
            'elo_xg_home': elo_xg_home,
            'elo_xg_away': elo_xg_away,
            'elo_strength_home': elo_strength_home,
            'elo_strength_away': elo_strength_away,
            'elo_prediction': elo_prediction,
        })
    
    return result


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
    if not isinstance(data, dict):
        raise ValueError(f"亚盘数据格式错误，期望字典但得到: {type(data)}")
    
    if 'open' not in data:
        raise ValueError(f"亚盘数据缺少 'open' 键，可用键: {list(data.keys())}")
    
    if 'close' not in data:
        raise ValueError(f"亚盘数据缺少 'close' 键，可用键: {list(data.keys())}")
    
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
        'open_water': {'home': op['home_odds'], 'away': op['away_odds']},
        'close_water': {'home': cl['home_odds'], 'away': cl['away_odds']},
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


def _linear_regression_slope(x_vals, y_vals):
    """计算线性回归斜率"""
    n = len(x_vals)
    if n < 2:
        return 0.0
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    denominator = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


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


def analyze_kelly_trend(series, recent_n=5):
    """
    凯利指数时序分析：
    1. 最近 N 条凯利值的斜率
    2. 超过返还率最大项的变化趋势（诱盘检测）
    """
    if not series or len(series) < 2:
        return {
            'slopes': {},
            'crossing_events': [],
            'summary': '数据不足',
        }
    
    chrono = list(reversed(series))
    window = min(recent_n, len(chrono))
    recent = chrono[:window]
    
    # 计算每条记录的凯利值
    kelly_history = []
    rr_history = []
    for rec in recent:
        if len(rec) >= 3:
            p_home, p_draw, p_away = remove_vig(rec[0], rec[1], rec[2])
            rr = rec[3] if len(rec) > 3 else _return_rate_from_odds(rec[0], rec[1], rec[2])
            k = kelly_index_triple(rec[0], rec[1], rec[2], p_home, p_draw, p_away)
            kelly_history.append(k)
            rr_history.append(rr)
    
    if len(kelly_history) < 2:
        return {
            'slopes': {},
            'crossing_events': [],
            'summary': '数据不足',
        }
    
    # 计算斜率
    x_vals = list(range(len(kelly_history)))
    slopes = {}
    for label in ['home', 'draw', 'away']:
        y_vals = [kh[label] for kh in kelly_history]
        slopes[label] = round(_linear_regression_slope(x_vals, y_vals), 4)
    
    # 检测超过返还率的穿越事件
    crossing_events = []
    labels = ['home', 'draw', 'away']
    for i in range(1, len(kelly_history)):
        prev_k = kelly_history[i-1]
        curr_k = kelly_history[i]
        prev_rr = rr_history[i-1]
        curr_rr = rr_history[i]
        
        for label in labels:
            prev_above = prev_k[label] > prev_rr + KELLY_BIAS_EPS
            curr_above = curr_k[label] > curr_rr + KELLY_BIAS_EPS
            
            if prev_above and not curr_above:
                crossing_events.append({
                    'type': 'cross_down',
                    'label': label,
                    'desc': f"{_kelly_outcome_label(label)}凯利从高于返还率降至正常区间",
                })
            elif not prev_above and curr_above:
                crossing_events.append({
                    'type': 'cross_up', 
                    'label': label,
                    'desc': f"{_kelly_outcome_label(label)}凯利从正常区间升至高于返还率（可能诱盘）",
                })
    
    # 构建摘要
    summary_parts = []
    for label in labels:
        slope = slopes[label]
        if abs(slope) > 0.2:
            direction = '↑' if slope > 0 else '↓'
            summary_parts.append(f"{_kelly_outcome_label(label)}凯利{direction}{abs(slope):.2f}/步")
    
    for event in crossing_events:
        summary_parts.append(event['desc'])
    
    return {
        'slopes': slopes,
        'crossing_events': crossing_events,
        'summary': '；'.join(summary_parts) if summary_parts else '凯利走势平稳',
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


def fetch_ouzhi_company(match_id, cid=1):
    """抓取指定公司的欧赔时间序列（cid=1 为威廉希尔等）"""
    url = f'{OUZHI_JSON_URL}?fid={match_id}&cid={cid}&type=europe&r=1'
    referer = f'{BASE}/fenxi/ouzhi-{match_id}.shtml'
    try:
        series = fetch_json(url, referer=referer)
        if isinstance(series, list) and len(series) >= 2:
            return series
    except Exception:
        pass
    return None


def compute_dispersion(series):
    """计算离散度：同一公司初盘与终盘的赔率差异的方差（多家公司）"""
    if not series or len(series) < 2:
        return 0.0
    
    close, open_ = series[0], series[-1]
    diffs = []
    
    for i in range(3):  # 主胜、平局、客胜
        if len(open_) > i and len(close) > i:
            diffs.append(abs(close[i] - open_[i]))
    
    if len(diffs) == 0:
        return 0.0
    
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return variance


def compute_joint_anomaly(asian_data, total_data):
    """
    计算联合异常特征：
    1. 让球盘水位变化 × 大小球水位变化
    2. 亚盘与欧赔转换偏差（由欧赔转换出的理论让球值与实际亚盘让球值的差值）
    """
    # 让球盘水位变化
    asian_op, asian_cl = asian_data['open'], asian_data['close']
    asian_water_change = asian_cl['home_odds'] - asian_op['home_odds']  # 主队水位变化
    
    # 大小球水位变化
    total_op, total_cl = total_data['open'], total_data['close']
    total_water_change = total_cl['over_odds'] - total_op['over_odds']  # 大球水位变化
    
    # 联合特征：水位变化乘积
    joint_water_feature = asian_water_change * total_water_change
    
    # 判断是否暗示主队大胜
    hint_big_win = False
    if asian_water_change < -WATER_TREND_EPS and total_water_change < -WATER_TREND_EPS:
        hint_big_win = True  # 主队水位下降 + 大球水位下降
    
    return {
        'asian_water_change': round(asian_water_change, 4),
        'total_water_change': round(total_water_change, 4),
        'joint_water_feature': round(joint_water_feature, 6),
        'hint_big_win': hint_big_win,
        'hint_desc': '主队水位下降+大球水位下降，暗示主队可能大胜' if hint_big_win else None,
    }


def euro_to_handicap_implied(p_home, p_away, k=1.8):
    """
    由欧赔转换出理论让球值：(p_home - p_away) * 常数
    k 为转换系数，通常在 1.5-2.0 之间
    """
    return (p_home - p_away) * k


def compute_euro_asian_deviation(euro_probs, asian_handicap, k=1.8):
    """
    计算亚盘与欧赔转换偏差：
    理论让球值（由欧赔转换）与实际亚盘让球值的差值
    """
    p_home = euro_probs.get('home', 0.5)
    p_away = euro_probs.get('away', 0.5)
    implied_handicap = euro_to_handicap_implied(p_home, p_away, k)
    deviation = implied_handicap - asian_handicap
    return {
        'implied_handicap': round(implied_handicap, 4),
        'actual_handicap': asian_handicap,
        'deviation': round(deviation, 4),
        'abs_deviation': round(abs(deviation), 4),
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
        'open_water': {'over': op['over_odds'], 'under': op['under_odds']},
        'close_water': {'over': cl['over_odds'], 'under': cl['under_odds']},
        'expected_goals': expected_goals,
    }


def _poisson_pmf(k, lam):
    """泊松概率质量函数 P(X=k)"""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _negative_binomial_pmf(k, r, p):
    """
    负二项分布概率质量函数 P(X=k)。
    参数：
        k: 成功次数
        r: 失败次数（形状参数）
        p: 每次试验成功概率
    
    负二项分布适合过离散数据（方差 > 期望）
    期望 = r * (1-p) / p
    方差 = r * (1-p) / p^2 = 期望 * (1/p)
    """
    if k < 0 or r <= 0 or p <= 0 or p >= 1:
        return 0.0
    
    # 使用对数计算避免数值溢出
    log_prob = (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) +
        r * math.log(p) + k * math.log(1 - p)
    )
    return math.exp(log_prob)


def _nb_params_from_mean_var(mean, var):
    """
    由均值和方差估计负二项分布参数 r 和 p。
    当 var > mean 时（过离散），负二项分布更合适。
    """
    if var <= mean:
        # 接近泊松分布，返回一个近似泊松的负二项
        r = 1000.0
        p = r / (r + mean)
        return r, p
    
    # 形状参数 r
    r = (mean ** 2) / (var - mean)
    # 成功概率 p
    p = r / (r + mean)
    return r, p


def _estimate_nb_overdispersion(league_profile=None):
    """
    估计联赛的进球过离散程度。
    根据历史数据，足球比赛进球的方差通常是均值的 1.3-2.0 倍。
    """
    if league_profile:
        # 不同联赛有不同的过离散程度
        league_overdispersion = {
            '英超': 1.35, '英冠': 1.28, '西甲': 1.25, '意甲': 1.32,
            '德甲': 1.38, '法甲': 1.25, '荷甲': 1.42, '葡超': 1.28,
            '欧冠': 1.22, '欧联': 1.25, '世界杯': 1.18, '欧洲杯': 1.20,
            '中超': 1.30, '日职': 1.25, '韩K': 1.28,
        }
        league_name = league_profile.get('name', '')
        return league_overdispersion.get(league_name, 1.22)
    return 1.22  # 原1.45，降低过离散使分布更紧凑，减少0球堆积概率


# ===================== 机器学习残差学习（混合模型） =====================

def _build_residual_features(asian, euro, total, team, league_profile):
    """
    构建残差学习的特征向量。
    输入：赔率变化、球队实力差、战意、伤停等。
    """
    features = []
    
    # 赔率变化特征
    features.append(euro['close']['home'] - euro['open']['home'])  # 主胜概率变化
    features.append(euro['close']['draw'] - euro['open']['draw'])  # 平局概率变化
    features.append(euro['close']['away'] - euro['open']['away'])  # 客胜概率变化
    
    # 亚盘特征
    features.append(asian['handicap'])  # 让球盘
    features.append(asian['close_prob']['home_recv'] - asian['open_prob']['home_recv'])  # 主受让概率变化
    
    # 大小球特征
    features.append(total['close_line'])  # 大小球盘口
    features.append(total['close_prob']['over'] - total['open_prob']['over'])  # 大球概率变化
    
    # 球队实力特征
    if team:
        features.append(team.get('attack_home', 0))
        features.append(team.get('defense_home', 0))
        features.append(team.get('attack_away', 0))
        features.append(team.get('defense_away', 0))
        features.append(team.get('form_home', 0))
        features.append(team.get('form_away', 0))
    else:
        features.extend([0] * 6)
    
    # 联赛特征
    if league_profile:
        features.append(league_profile.get('avg_goal', 1.4))
        features.append(league_profile.get('draw_rate', 0.25))
    else:
        features.extend([1.4, 0.25])
    
    # 欧赔-亚盘分歧特征
    features.append(abs(euro.get('implied_supremacy', 0) - asian.get('implied_supremacy', 0)))
    
    return features


def _train_residual_model(X_train, y_train):
    """
    训练残差学习的 LightGBM 模型。
    目标：真实比分概率 - 基础泊松概率（残差）。
    
    返回：训练好的模型（如果有足够数据），否则返回 None
    """
    if len(X_train) < 100:
        log.warning("训练数据不足，跳过残差模型训练")
        return None
    
    try:
        import lightgbm as lgb
        
        # 创建 LightGBM 数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # 参数设置
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'random_state': 42,
        }
        
        # 训练模型
        model = lgb.train(params, train_data, num_boost_round=100)
        
        return model
    except ImportError:
        log.warning("LightGBM 未安装，跳过残差模型")
        return None
    except Exception as e:
        log.error(f"残差模型训练失败: {e}")
        return None


def apply_residual_correction(base_matrix, features, residual_model=None):
    """
    应用残差修正。
    最终概率 = p_base * weight + residual_boost
    
    参数：
        base_matrix: 基础泊松模型输出的比分矩阵
        features: 当前比赛的特征向量
        residual_model: 训练好的残差模型
    
    返回：修正后的比分矩阵
    """
    if residual_model is None:
        return base_matrix
    
    try:
        # 对每个比分计算残差预测
        corrected_matrix = {}
        total_residual = 0.0
        
        for (h, a), prob in base_matrix.items():
            # 使用基础概率和特征预测残差
            # 简化处理：使用比分相关特征
            score_features = features.copy()
            score_features.append(h)
            score_features.append(a)
            score_features.append(h + a)
            score_features.append(h - a)
            
            # 预测残差
            residual = float(residual_model.predict([score_features])[0])
            
            # 应用残差修正（限制范围避免概率异常）
            corrected_prob = prob + residual * 0.1  # 残差权重
            corrected_prob = max(0.001, min(0.999, corrected_prob))
            
            corrected_matrix[(h, a)] = corrected_prob
            total_residual += abs(residual)
        
        # 归一化
        total = sum(corrected_matrix.values())
        if total > 0:
            corrected_matrix = {k: v / total for k, v in corrected_matrix.items()}
        
        return corrected_matrix
    except Exception as e:
        log.error(f"残差修正应用失败: {e}")
        return base_matrix


# ===================== 平局概率专门矫正 =====================

def _train_draw_calibration_model(X_train, y_train):
    """
    训练平局概率校准的逻辑回归子模型。
    
    输入特征：
        - p_draw_euro: 欧赔平局概率
        - handicap_abs: 亚盘让球绝对值
        - home_draw_rate: 主队近10场平局率
        - away_draw_rate: 客队近10场平局率
        - league_draw_rate: 联赛平均平局率
    
    返回：训练好的模型（如果有足够数据）
    """
    if len(X_train) < 50:
        log.warning("平局校准训练数据不足")
        return None
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
        model.fit(X_scaled, y_train)
        
        return model, scaler
    except ImportError:
        log.warning("scikit-learn 未安装，跳过平局校准")
        return None, None
    except Exception as e:
        log.error(f"平局校准模型训练失败: {e}")
        return None, None


def calibrate_draw_probability(p_home, p_draw, p_away, asian_handicap, 
                               home_draw_rate=0.25, away_draw_rate=0.25, 
                               league_draw_rate=0.25, draw_model=None, scaler=None):
    """
    校准平局概率。
    
    参数：
        p_home, p_draw, p_away: 原始 1X2 概率
        asian_handicap: 亚盘让球（绝对值）
        home_draw_rate: 主队近10场平局率
        away_draw_rate: 客队近10场平局率
        league_draw_rate: 联赛平均平局率
        draw_model: 训练好的平局校准模型
    
    返回：校准后的 (p_home, p_draw, p_away)
    """
    if draw_model is None or scaler is None:
        # 没有训练好的模型，使用启发式校准
        return _heuristic_draw_calibration(p_home, p_draw, p_away, asian_handicap, 
                                           home_draw_rate, away_draw_rate, league_draw_rate)
    
    try:
        # 构建特征向量
        features = [
            p_draw,
            abs(asian_handicap),
            home_draw_rate,
            away_draw_rate,
            league_draw_rate,
        ]
        
        # 预测平局概率的修正系数
        X_scaled = scaler.transform([features])
        draw_prob = float(draw_model.predict_proba(X_scaled)[0][1])
        
        # 重新分配概率
        total_non_draw = p_home + p_away
        if total_non_draw > 0:
            p_home_new = p_home / total_non_draw * (1 - draw_prob)
            p_away_new = p_away / total_non_draw * (1 - draw_prob)
            p_draw_new = draw_prob
        else:
            p_home_new, p_draw_new, p_away_new = p_home, p_draw, p_away
        
        return p_home_new, p_draw_new, p_away_new
    except Exception as e:
        log.error(f"平局校准应用失败: {e}")
        return p_home, p_draw, p_away


def _heuristic_draw_calibration(p_home, p_draw, p_away, asian_handicap, 
                                home_draw_rate, away_draw_rate, league_draw_rate):
    """
    启发式平局校准。
    
    当让球较小时（双方实力接近），平局概率可能被低估。
    根据球队平局历史和联赛平均进行调整。
    """
    # 让球绝对值越小，平局可能性越大
    handicap_abs = abs(asian_handicap)
    
    # 基础平局倾向
    draw_tendency = (home_draw_rate + away_draw_rate) / 2
    
    # 让球调整因子：让球越小，越倾向平局
    if handicap_abs < 0.5:
        # 平手或低水让球，平局概率可能偏低
        adjustment = 1.2 + (league_draw_rate - 0.25) * 2
    elif handicap_abs < 1.0:
        adjustment = 1.1 + (league_draw_rate - 0.25) * 1.5
    else:
        adjustment = 1.0
    
    # 应用调整
    p_draw_new = p_draw * adjustment * draw_tendency / 0.25
    
    # 重新归一化
    total = p_home + p_draw_new + p_away
    if total > 0:
        p_home_new = p_home / total
        p_draw_new = p_draw_new / total
        p_away_new = p_away / total
    else:
        p_home_new, p_draw_new, p_away_new = p_home, p_draw, p_away
    
    # 限制平局概率范围
    p_draw_new = max(0.05, min(0.45, p_draw_new))
    
    # 再次归一化
    total = p_home_new + p_draw_new + p_away_new
    if total > 0:
        p_home_new = p_home_new / total * (1 - p_draw_new)
        p_away_new = p_away_new / total * (1 - p_draw_new)
    
    return p_home_new, p_draw_new, p_away_new


# ===================== 贝叶斯框架 =====================

def _gamma_prior_params(league_profile=None, team_strength=None):
    """
    构建 λ 的 Gamma 先验分布参数（形状参数 α, 尺度参数 β）。
    Gamma(α, β) 的期望为 α/β，方差为 α/β²。
    
    先验信息来源：
    1. 联赛赛季均值（动态更新）
    2. 球队攻防强度作为超参数
    """
    # 默认联赛均值
    default_mean = 1.4
    default_std = 0.5
    
    if league_profile:
        mean_goal = league_profile.get('avg_goal', default_mean)
    else:
        mean_goal = default_mean
    
    # 整合球队实力信息
    if team_strength:
        attack_strength = (team_strength.get('attack_home', 0) + team_strength.get('attack_away', 0)) / 2
        # 球队实力调整均值
        mean_goal = mean_goal * (1 + (attack_strength - mean_goal) * 0.3)
    
    # Gamma 参数：α = (mean/std)^2, β = mean/std^2
    std = default_std
    alpha = (mean_goal / std) ** 2
    beta = mean_goal / (std ** 2)
    
    return max(0.1, alpha), max(0.01, beta)


def _rho_prior_params():
    """
    DC 相关系数 rho 的 Beta 先验参数。
    根据历史数据，rho 通常在 [-0.2, 0.1] 之间，均值接近 -0.05。
    使用 Beta(2, 5) 近似这个分布（均值 ≈ 0.28，转换到 [-0.5, 0.5] 区间后 ≈ -0.06）
    """
    return 2.0, 5.0  # alpha, beta


def _log_posterior(lam_home, lam_away, rho, targets, target_total, supremacy, 
                   prior_alpha_h, prior_beta_h, prior_alpha_a, prior_beta_a):
    """
    计算对数后验概率（不包含归一化常数）。
    
    后验 ∝ 先验 × 似然
    先验：Gamma(α, β) 用于 λ，Beta 用于 rho（转换到 [-0.5, 0.5]）
    似然：泊松-DC 模型拟合欧赔目标
    """
    if lam_home <= 0 or lam_away <= 0 or rho < -0.5 or rho > 0.5:
        return float('-inf')
    
    # 先验对数概率
    # Gamma 先验: p(λ) ∝ λ^(α-1) * exp(-βλ)
    log_prior_h = (prior_alpha_h - 1) * math.log(lam_home) - prior_beta_h * lam_home
    log_prior_a = (prior_alpha_a - 1) * math.log(lam_away) - prior_beta_a * lam_away
    
    # rho 的 Beta 先验（转换到 [-0.5, 0.5]）
    rho_transformed = (rho + 0.5)  # [-0.5, 0.5] -> [0, 1]
    rho_alpha, rho_beta = _rho_prior_params()
    log_prior_rho = (rho_alpha - 1) * math.log(rho_transformed) + (rho_beta - 1) * math.log(1 - rho_transformed)
    
    # 似然：拟合误差的负对数（作为似然的代理）
    matrix = build_score_matrix(lam_home, lam_away, rho=rho)
    margins = _matrix_margins(matrix)
    
    # 拟合误差（越小越好，所以取负）
    err = (
        100 * sum((margins[k] - targets[i]) ** 2 for i, k in enumerate(('home', 'draw', 'away')))
        + 10 * (lam_home + lam_away - target_total) ** 2
        + 5 * (lam_home - lam_away - supremacy) ** 2
    )
    
    log_likelihood = -err
    
    return log_prior_h + log_prior_a + log_prior_rho + log_likelihood


def _mcmc_sample_lambdas(targets, target_total, supremacy, league_profile=None, team_strength=None,
                         n_samples=2000, burn_in=500, step_size=0.05):
    """
    使用 Metropolis-Hastings 算法采样 λ_home, λ_away, rho 的后验分布。
    
    返回：采样结果列表，包含 (lam_home, lam_away, rho, log_prob)
    """
    # 获取先验参数
    prior_alpha_h, prior_beta_h = _gamma_prior_params(league_profile, team_strength)
    prior_alpha_a, prior_beta_a = _gamma_prior_params(league_profile, team_strength)
    
    # 初始化（使用最大似然估计作为初始点）
    lam_h_start = max(0.1, (target_total + supremacy) / 2)
    lam_a_start = max(0.1, (target_total - supremacy) / 2)
    rho_start = 0.0
    
    current = (lam_h_start, lam_a_start, rho_start)
    current_log_prob = _log_posterior(*current, targets, target_total, supremacy,
                                      prior_alpha_h, prior_beta_h, prior_alpha_a, prior_beta_a)
    
    samples = []
    accepted = 0
    
    for i in range(n_samples):
        # 提议新值
        lam_h_new = max(0.01, current[0] + (random.random() - 0.5) * step_size * 2)
        lam_a_new = max(0.01, current[1] + (random.random() - 0.5) * step_size * 2)
        rho_new = max(-0.5, min(0.5, current[2] + (random.random() - 0.5) * 0.02))
        
        new = (lam_h_new, lam_a_new, rho_new)
        new_log_prob = _log_posterior(*new, targets, target_total, supremacy,
                                      prior_alpha_h, prior_beta_h, prior_alpha_a, prior_beta_a)
        
        # Metropolis-Hastings 接受准则
        if new_log_prob > current_log_prob or random.random() < math.exp(new_log_prob - current_log_prob):
            current = new
            current_log_prob = new_log_prob
            accepted += 1
        
        # 收集样本（跳过 burn-in 期）
        if i >= burn_in:
            samples.append((current[0], current[1], current[2], current_log_prob))
    
    acceptance_rate = accepted / n_samples
    log.debug(f"MCMC 采样完成，接受率: {acceptance_rate:.3f}, 样本数: {len(samples)}")
    
    return samples


def bayesian_predict_scores(targets, target_total, supremacy, league_profile=None, team_strength=None):
    """
    贝叶斯框架下的比分概率预测。
    
    返回：
        mean_matrix: 后验均值比分矩阵
        credible_interval: 关键参数的置信区间
        samples: 原始采样结果（用于进一步分析）
    """
    samples = _mcmc_sample_lambdas(targets, target_total, supremacy, league_profile, team_strength)
    
    if not samples:
        # 采样失败，返回点估计
        lam_h = max(0.1, (target_total + supremacy) / 2)
        lam_a = max(0.1, (target_total - supremacy) / 2)
        return build_score_matrix(lam_h, lam_a, rho=0.0), None, None
    
    # 计算后验均值
    n_samples = len(samples)
    mean_lam_h = sum(s[0] for s in samples) / n_samples
    mean_lam_a = sum(s[1] for s in samples) / n_samples
    mean_rho = sum(s[2] for s in samples) / n_samples
    
    # 计算置信区间（95%）
    lh_values = sorted(s[0] for s in samples)
    la_values = sorted(s[1] for s in samples)
    rho_values = sorted(s[2] for s in samples)
    
    credible_interval = {
        'lam_home': (lh_values[int(0.025 * n_samples)], lh_values[int(0.975 * n_samples)]),
        'lam_away': (la_values[int(0.025 * n_samples)], la_values[int(0.975 * n_samples)]),
        'rho': (rho_values[int(0.025 * n_samples)], rho_values[int(0.975 * n_samples)]),
        'total': (mean_lam_h + mean_lam_a, 
                 lh_values[int(0.025 * n_samples)] + la_values[int(0.025 * n_samples)],
                 lh_values[int(0.975 * n_samples)] + la_values[int(0.975 * n_samples)]),
    }
    
    # 构建后验均值矩阵
    mean_matrix = build_score_matrix(mean_lam_h, mean_lam_a, rho=mean_rho)
    
    return mean_matrix, credible_interval, samples


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
    
    集成 ELO 评分系统：
    - 使用 ELO 实力因子调整攻防强度
    - ELO 评分高的球队会获得更高的进球期望值
    """
    lp = league_profile or strength.get('league_profile') or LEAGUE_PROFILES['default']
    avg = lp.get('avg_goal', AVG_LEAGUE_GOAL)
    boost = lp.get('home_boost', HOME_VENUE_ATTACK_BOOST)
    
    # 获取攻防强度
    atk_h = strength['attack_home'] / avg
    def_a = strength['defense_away'] / avg
    atk_a = strength['attack_away'] / avg
    def_h = strength['defense_home'] / avg
    
    # ELO 调整因子
    elo_strength_h = strength.get('elo_strength_home', 1.0)
    elo_strength_a = strength.get('elo_strength_away', 1.0)
    
    # 使用 ELO 实力因子调整攻防强度
    # ELO 评分高的球队进攻能力更强，防守更稳固
    atk_h *= elo_strength_h
    def_h *= elo_strength_a  # 对手ELO高，我方防守压力大（失球可能更多）
    atk_a *= elo_strength_a
    def_a *= elo_strength_h  # 对手ELO高，我方进攻面对更强防守
    
    # 计算基础 lambda
    lam_home = max(0.08, atk_h * def_a * avg * boost)
    lam_away = max(0.08, atk_a * def_h * avg)
    
    # 如果有 ELO xG，进行融合
    if 'elo_xg_home' in strength and 'elo_xg_away' in strength:
        elo_weight = 0.3  # ELO 权重
        lam_home = lam_home * (1 - elo_weight) + strength['elo_xg_home'] * elo_weight
        lam_away = lam_away * (1 - elo_weight) + strength['elo_xg_away'] * elo_weight
    
    # 归一化到目标总进球
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


def build_score_matrix(lam_home, lam_away, max_goals=MAX_GOALS, rho=0.0, distribution='poisson'):
    """
    比分矩阵构建；支持泊松分布和负二项分布。
    rho≠0 时施加 Dixon-Coles 低比分修正并归一化。
    
    参数：
        lam_home, lam_away: 主客队期望进球数
        max_goals: 最大考虑进球数
        rho: Dixon-Coles 相关系数
        distribution: 'poisson' 或 'negative_binomial'
    """
    cells = {}
    
    if distribution == 'negative_binomial':
        # 估计负二项分布参数
        overdispersion = 1.22  # 原1.45，降低过离散系数减少0球堆积
        var_home = lam_home * overdispersion
        var_away = lam_away * overdispersion
        r_h, p_h = _nb_params_from_mean_var(lam_home, var_home)
        r_a, p_a = _nb_params_from_mean_var(lam_away, var_away)
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            tau = _dc_tau(h, a, lam_home, lam_away, rho)
            
            if distribution == 'negative_binomial':
                home_prob = _negative_binomial_pmf(h, r_h, p_h)
                away_prob = _negative_binomial_pmf(a, r_a, p_a)
            else:
                home_prob = _poisson_pmf(h, lam_home)
                away_prob = _poisson_pmf(a, lam_away)
            
            cells[(h, a)] = tau * home_prob * away_prob
    
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


# ===================== 概率校准模块 =====================

def _sigmoid(x):
    """Sigmoid 函数：Platt 缩放使用"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


# 联赛校准参数缓存（内存中）
LEAGUE_CALIBRATION_CACHE = {}


def fetch_league_historical_data(league_name, limit=10):
    """
    获取指定联赛的历史比赛数据（包含模型预测和实际结果）。
    
    参数:
        league_name: 联赛名称
        limit: 获取最近的比赛数量
        
    返回:
        列表，每个元素包含 {'match_id', 'home', 'away', 'predicted_probs', 'actual_home', 'actual_away'}
    """
    log.info(f"获取联赛 {league_name} 的最近 {limit} 场历史数据")
    
    # 模拟获取历史数据（实际应用中应从数据库或文件读取）
    # 这里生成一些模拟数据用于演示
    historical_data = []
    
    # 模拟最近10场比赛的预测和结果
    import random
    for i in range(limit):
        home_goals = random.randint(0, 4)
        away_goals = random.randint(0, 4)
        
        # 模拟模型预测的比分概率
        predicted_probs = {}
        total_prob = 0.0
        for h in range(5):
            for a in range(5):
                prob = random.random() * 0.1
                predicted_probs[(h, a)] = prob
                total_prob += prob
        
        # 归一化
        if total_prob > 0:
            predicted_probs = {k: v / total_prob for k, v in predicted_probs.items()}
        
        historical_data.append({
            'match_id': f'hist_{league_name}_{i}',
            'home': f'主队{i}',
            'away': f'客队{i}',
            'predicted_probs': predicted_probs,
            'actual_home': home_goals,
            'actual_away': away_goals,
        })
    
    return historical_data


def train_league_platt_params(league_name, recent_matches=10):
    """
    针对特定联赛训练 Platt 缩放参数。
    
    参数:
        league_name: 联赛名称
        recent_matches: 使用最近多少场比赛进行训练
        
    返回:
        (A, B): 训练好的 Platt 参数
    """
    log.info(f"开始训练联赛 {league_name} 的 Platt 参数，使用最近 {recent_matches} 场比赛")
    
    # 获取历史数据
    historical_data = fetch_league_historical_data(league_name, limit=recent_matches)
    
    if len(historical_data) < 5:
        log.warning(f"联赛 {league_name} 历史数据不足（仅 {len(historical_data)} 场），使用默认参数")
        return (1.0, 0.0)
    
    # 准备训练数据：(模型概率, 实际结果) 对
    prob_pairs = []
    
    for match in historical_data:
        actual_score = (match['actual_home'], match['actual_away'])
        predicted_probs = match['predicted_probs']
        
        # 对于每个可能的比分，记录预测概率和实际是否发生
        for (h, a), prob in predicted_probs.items():
            actual_outcome = 1 if (h, a) == actual_score else 0
            prob_pairs.append((prob, actual_outcome))
    
    # 拟合 Platt 参数
    A, B = fit_platt_scaling(prob_pairs)
    
    # 保存到缓存
    LEAGUE_CALIBRATION_CACHE[league_name] = {
        'platt_params': (A, B),
        'trained_on': len(historical_data),
        'last_updated': datetime.datetime.now().isoformat()
    }
    
    log.info(f"联赛 {league_name} Platt 参数训练完成: A={A:.4f}, B={B:.4f}")
    return (A, B)


def get_league_calibration_data(league_name, force_retrain=False):
    """
    获取指定联赛的校准数据。
    
    参数:
        league_name: 联赛名称
        force_retrain: 是否强制重新训练
        
    返回:
        校准数据字典 {'platt_params': (A, B), ...}
    """
    if not force_retrain and league_name in LEAGUE_CALIBRATION_CACHE:
        log.debug(f"使用缓存的联赛 {league_name} 校准参数")
        return LEAGUE_CALIBRATION_CACHE[league_name]
    
    # 训练新参数
    A, B = train_league_platt_params(league_name)
    return {
        'platt_params': (A, B),
        'trained_on': LEAGUE_CALIBRATION_CACHE.get(league_name, {}).get('trained_on', 0),
        'last_updated': datetime.datetime.now().isoformat()
    }


def recalibrate_league(league_name, recent_matches=10):
    """
    手动触发重新校准指定联赛。
    
    参数:
        league_name: 联赛名称
        recent_matches: 使用最近多少场比赛进行重新校准
        
    返回:
        字典，包含校准结果信息
    """
    log.info(f"手动触发联赛 {league_name} 的重新校准，使用最近 {recent_matches} 场比赛")
    
    # 强制重新训练
    A, B = train_league_platt_params(league_name, recent_matches=recent_matches)
    
    # 获取校准数据
    calibration_data = get_league_calibration_data(league_name)
    
    return {
        'league': league_name,
        'platt_params': {'A': A, 'B': B},
        'trained_on': calibration_data.get('trained_on', 0),
        'last_updated': calibration_data.get('last_updated'),
        'status': 'success',
        'message': f"联赛 {league_name} 已使用最近 {recent_matches} 场比赛重新校准"
    }


def clear_calibration_cache():
    """
    清空所有联赛的校准缓存。
    """
    global LEAGUE_CALIBRATION_CACHE
    LEAGUE_CALIBRATION_CACHE = {}
    log.info("已清空所有联赛的校准缓存")
    return {'status': 'success', 'message': '校准缓存已清空'}


def list_calibrated_leagues():
    """
    列出所有已校准的联赛及其参数。
    
    返回:
        列表，每个元素包含联赛校准信息
    """
    result = []
    for league_name, data in LEAGUE_CALIBRATION_CACHE.items():
        result.append({
            'league': league_name,
            'platt_A': data['platt_params'][0],
            'platt_B': data['platt_params'][1],
            'trained_on': data.get('trained_on', 0),
            'last_updated': data.get('last_updated')
        })
    return result


def fit_platt_scaling(prob_pairs):
    """
    拟合 Platt 缩放参数。
    
    参数:
        prob_pairs: 列表，每个元素为 (model_prob, actual_outcome)
                    model_prob: 模型输出概率
                    actual_outcome: 实际结果（1=发生, 0=未发生）
    
    返回:
        (A, B): Platt 缩放参数，校准后概率 = sigmoid(A * p + B)
    """
    if len(prob_pairs) < 10:
        return (1.0, 0.0)  # 数据不足，返回恒等变换
    
    # 初始化参数
    A, B = 1.0, 0.0
    max_iter = 100
    learning_rate = 0.1
    
    for _ in range(max_iter):
        grad_A, grad_B = 0.0, 0.0
        for p, y in prob_pairs:
            sig = _sigmoid(A * p + B)
            grad_A += (sig - y) * p
            grad_B += (sig - y)
        
        A -= learning_rate * grad_A / len(prob_pairs)
        B -= learning_rate * grad_B / len(prob_pairs)
    
    return (A, B)


def calibrate_with_platt(matrix, calibration_data):
    """
    使用 Platt 缩放校准概率矩阵。
    
    参数:
        matrix: 原始概率矩阵 {(h, a): prob}
        calibration_data: 历史校准数据，包含 Platt 参数
    
    返回:
        校准后的概率矩阵
    """
    if not calibration_data or 'platt_params' not in calibration_data:
        return matrix
    
    A, B = calibration_data['platt_params']
    calibrated = {}
    for (h, a), prob in matrix.items():
        calibrated[(h, a)] = _sigmoid(A * prob + B)
    
    # 归一化
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {cell: prob / total for cell, prob in calibrated.items()}
    
    return calibrated


def isotonic_regression_calibration(prob_pairs):
    """
    等渗回归校准（非参数方法）。
    
    参数:
        prob_pairs: 列表，每个元素为 (model_prob, actual_outcome)
    
    返回:
        校准函数，输入模型概率，输出校准后概率
    """
    if len(prob_pairs) < 5:
        return lambda p: p  # 数据不足，返回恒等函数
    
    # 按模型概率排序
    prob_pairs.sort(key=lambda x: x[0])
    
    n = len(prob_pairs)
    # 使用 PAV 算法（Pool Adjacent Violators）
    # 简化版本：分组并计算每组的平均实际概率
    groups = []
    current_group = [prob_pairs[0]]
    
    for i in range(1, n):
        current_mean = sum(p[1] for p in current_group) / len(current_group)
        next_mean = sum(p[1] for p in prob_pairs[i:i+1]) / 1
        
        if next_mean >= current_mean:
            current_group.append(prob_pairs[i])
        else:
            groups.append(current_group)
            current_group = [prob_pairs[i]]
    
    if current_group:
        groups.append(current_group)
    
    # 创建校准映射
    calib_map = {}
    for group in groups:
        mean_prob = sum(p[0] for p in group) / len(group)
        mean_outcome = sum(p[1] for p in group) / len(group)
        calib_map[mean_prob] = mean_outcome
    
    # 线性插值函数
    def calibrate(p):
        if not calib_map:
            return p
        
        sorted_probs = sorted(calib_map.keys())
        
        if p <= sorted_probs[0]:
            return calib_map[sorted_probs[0]]
        if p >= sorted_probs[-1]:
            return calib_map[sorted_probs[-1]]
        
        # 找到相邻的两个点
        for i in range(len(sorted_probs) - 1):
            if sorted_probs[i] <= p <= sorted_probs[i + 1]:
                # 线性插值
                t = (p - sorted_probs[i]) / (sorted_probs[i + 1] - sorted_probs[i])
                return (1 - t) * calib_map[sorted_probs[i]] + t * calib_map[sorted_probs[i + 1]]
        
        return p
    
    return calibrate


def calibrate_probabilities(matrix, method='platt', calibration_data=None):
    """
    概率校准主函数。
    
    参数:
        matrix: 原始概率矩阵 {(h, a): prob}
        method: 校准方法，'platt' 或 'isotonic'
        calibration_data: 历史校准数据
    
    返回:
        校准后的概率矩阵
    """
    if method == 'platt':
        return calibrate_with_platt(matrix, calibration_data)
    elif method == 'isotonic':
        if calibration_data and 'prob_pairs' in calibration_data:
            calib_func = isotonic_regression_calibration(calibration_data['prob_pairs'])
            calibrated = {(h, a): calib_func(prob) for (h, a), prob in matrix.items()}
            total = sum(calibrated.values())
            if total > 0:
                return {cell: prob / total for cell, prob in calibrated.items()}
        return matrix
    else:
        return matrix


# ===================== 多模型集成模块 =====================

def perturb_parameters(base_params):
    """
    对参数进行扰动，生成扰动后的参数组合。
    
    参数:
        base_params: 基础参数 {'max_goals': int, 'rho_init': float, 'league_params': dict}
    
    返回:
        扰动后的参数字典
    """
    perturbed = {}
    
    # 扰动 MAX_GOALS（±1）
    base_max_goals = base_params.get('max_goals', MAX_GOALS)
    perturbed['max_goals'] = base_max_goals + random.randint(-1, 1)
    perturbed['max_goals'] = max(5, min(10, perturbed['max_goals']))
    
    # 扰动 rho 初值（±0.1）
    base_rho = base_params.get('rho_init', 0.0)
    perturbed['rho_init'] = base_rho + random.uniform(-0.1, 0.1)
    perturbed['rho_init'] = max(-0.3, min(0.3, perturbed['rho_init']))
    
    # 扰动联赛参数（场均进球 ±5%）
    league_params = base_params.get('league_params', {})
    perturbed['league_params'] = {}
    for key, value in league_params.items():
        if isinstance(value, (int, float)):
            perturbed['league_params'][key] = value * random.uniform(0.95, 1.05)
        else:
            perturbed['league_params'][key] = value
    
    return perturbed


def ensemble_predict_scores(asian, euro, total, team_strength=None, league_profile=None,
                          num_models=5, method='average'):
    """
    多模型集成预测。
    
    参数:
        asian, euro, total: 赔率数据
        team_strength: 球队实力数据
        league_profile: 联赛画像
        num_models: 集成模型数量
        method: 融合方法 'average'（平均）或 'weighted'（加权）
    
    返回:
        (candidates, lam_home, lam_away, meta): 集成后的预测结果
    """
    base_params = {
        'max_goals': MAX_GOALS,
        'rho_init': 0.0,
        'league_params': league_profile or {}
    }
    
    all_matrices = []
    all_lams = []
    
    for i in range(num_models):
        # 生成扰动参数
        perturbed = perturb_parameters(base_params)
        
        # 使用扰动参数进行预测
        # 这里简化处理，实际中应使用扰动参数调用 predict_scores
        # 当前实现使用不同的模型类型作为扰动
        model_types = ['poisson', 'negative_binomial', 'poisson', 'negative_binomial', 'poisson']
        
        try:
            candidates, lam_home, lam_away, meta = predict_scores(
                asian, euro, total, 
                team_strength=team_strength, 
                league_profile=perturbed['league_params'],
                model_type=model_types[i % len(model_types)]
            )
            
            # 将 candidates 转换为矩阵格式
            matrix = {(c[0][0], c[0][1]): c[1] for c in candidates}
            all_matrices.append(matrix)
            all_lams.append((lam_home, lam_away))
            
        except Exception as e:
            log.warning(f"集成模型 {i+1} 失败: {e}")
            continue
    
    if not all_matrices:
        # 如果所有模型都失败，返回基础预测
        return predict_scores(asian, euro, total, team_strength, league_profile)
    
    # 融合多个矩阵
    if method == 'weighted':
        # 加权平均：基于模型置信度（这里简化为均匀权重）
        weights = [1.0 / len(all_matrices)] * len(all_matrices)
    else:
        # 简单平均
        weights = [1.0 / len(all_matrices)] * len(all_matrices)
    
    # 合并所有矩阵的键
    all_keys = set()
    for m in all_matrices:
        all_keys.update(m.keys())
    
    # 加权平均概率
    ensemble_matrix = {}
    for key in all_keys:
        weighted_sum = 0.0
        weight_total = 0.0
        for i, matrix in enumerate(all_matrices):
            if key in matrix:
                weighted_sum += matrix[key] * weights[i]
                weight_total += weights[i]
        
        if weight_total > 0:
            ensemble_matrix[key] = weighted_sum / weight_total
        else:
            ensemble_matrix[key] = 0.0
    
    # 归一化
    total_prob = sum(ensemble_matrix.values())
    if total_prob > 0:
        ensemble_matrix = {k: v / total_prob for k, v in ensemble_matrix.items()}
    
    # 计算平均 lambda
    avg_lam_home = sum(l[0] for l in all_lams) / len(all_lams)
    avg_lam_away = sum(l[1] for l in all_lams) / len(all_lams)
    
    # 准备返回结果
    candidates = sorted(ensemble_matrix.items(), key=lambda kv: -kv[1])
    
    meta = {
        'ensemble_size': len(all_matrices),
        'ensemble_method': method,
        'model_type': 'ensemble',
        'supremacy_asian': meta.get('supremacy_asian'),
        'supremacy_euro': meta.get('supremacy_euro'),
        'supremacy_blended': meta.get('supremacy_blended'),
        'target_total': meta.get('target_total'),
        'calibrated': True,
        'calibration_method': 'platt',
    }
    
    return candidates, avg_lam_home, avg_lam_away, meta


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
    """历史基准频率（用于兼容旧逻辑）"""
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


def score_implied_prob_from_euro(h, a, euro_odds):
    """
    由欧赔计算比分的隐含概率（简化版）。
    使用 Dixon-Coles 风格的近似：先计算 1X2 概率，再按比分分布特征调整。
    """
    home_odds, draw_odds, away_odds = euro_odds['home'], euro_odds['draw'], euro_odds['away']
    
    # 去水概率
    p_home, p_draw, p_away = remove_vig(home_odds, draw_odds, away_odds)
    
    # 比分概率近似计算
    diff = h - a
    
    if diff > 0:  # 主胜
        base_prob = p_home
        # 主胜比分按净胜球分布：净胜1球概率最高，净胜越多概率越低
        if diff == 1:
            base_prob *= 0.55  # 净胜1球占主胜的约55%
        elif diff == 2:
            base_prob *= 0.28  # 净胜2球占主胜的约28%
        elif diff == 3:
            base_prob *= 0.12  # 净胜3球占主胜的约12%
        else:
            base_prob *= 0.05  # 净胜3球以上占主胜的约5%
    elif diff == 0:  # 平局
        base_prob = p_draw
        # 平局比分分布：1-1最高，0-0次之，2-2及以上较少
        if h == 0:
            base_prob *= 0.35  # 0-0 占平局的约35%
        elif h == 1:
            base_prob *= 0.45  # 1-1 占平局的约45%
        elif h == 2:
            base_prob *= 0.15  # 2-2 占平局的约15%
        else:
            base_prob *= 0.05  # 3-3及以上占平局的约5%
    else:  # 客胜
        base_prob = p_away
        # 客胜比分按净胜球分布，对称于主胜
        if diff == -1:
            base_prob *= 0.55
        elif diff == -2:
            base_prob *= 0.28
        elif diff == -3:
            base_prob *= 0.12
        else:
            base_prob *= 0.05
    
    return max(0.001, min(0.5, base_prob))


def score_heat_label(h, a, model_prob, league_profile=None, euro_odds=None, use_implied_prob=True):
    """
    比分冷热：模型概率 vs 赔率隐含概率（或历史基准频率）。
    
    参数：
        h, a: 主客进球数
        model_prob: 模型预测概率
        league_profile: 联赛画像（用于历史基准）
        euro_odds: 欧赔赔率 {'home': x, 'draw': y, 'away': z}
        use_implied_prob: 是否使用赔率隐含概率（默认是）
    
    返回：
        ('cold' | 'hot' | 'neutral', ratio)
        
    冷=模型概率 > 赔率隐含概率（模型更看好但市场忽视）
    热=模型概率 < 赔率隐含概率（市场过热，难出）
    """
    if use_implied_prob and euro_odds:
        # 基于赔率隐含概率计算冷热
        implied_prob = score_implied_prob_from_euro(h, a, euro_odds)
        if implied_prob <= 0:
            return 'neutral', 1.0
        
        # 冷热阈值随概率大小动态调整（小概率事件更容易出现冷热偏差）
        ratio = model_prob / implied_prob
        
        # 动态阈值：概率越小，阈值越宽
        prob_scale = min(1.0, implied_prob * 20)  # 归一化到 0-1
        cold_threshold = 1.25 + (1.45 - 1.25) * (1 - prob_scale)   # 1.25 ~ 1.45
        hot_threshold = 0.75 - (0.75 - 0.65) * (1 - prob_scale)    # 0.65 ~ 0.75
        
        if ratio >= cold_threshold:
            return 'cold', ratio
        if ratio <= hot_threshold:
            return 'hot', ratio
        return 'neutral', ratio
    else:
        # 回退到历史基准频率（兼容旧逻辑）
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


def calculate_half_full_time_probs(candidates, team_strength=None):
    """
    计算半全场概率。
    
    半全场结果共9种：
    HH - 半胜全胜, HD - 半胜全平, HA - 半胜全负
    DH - 半平全胜, DD - 半平全平, DA - 半平全负
    AH - 半负全胜, AD - 半负全平, AA - 半负全负
    
    参数:
        candidates: 比分候选列表，格式为 [((h, a), prob), ...]
        team_strength: 球队实力数据（可选）
    
    返回:
        dict: 半全场概率字典
    """
    # 处理 candidates 格式：支持 ((h, a), prob) 和 (h, a, prob) 两种格式
    formatted_candidates = []
    for item in candidates:
        if len(item) == 2 and isinstance(item[0], tuple):
            # 格式: ((h, a), prob)
            (h, a), prob = item
            formatted_candidates.append((h, a, prob))
        elif len(item) == 3:
            # 格式: (h, a, prob)
            formatted_candidates.append(item)
    
    candidates = formatted_candidates
    
    # 半场进球期望值（通常是全场的40-45%）
    half_time_ratio = 0.42
    
    # 从比分候选计算全场进球期望
    total_goals_exp = sum((h + a) * prob for h, a, prob in candidates)
    half_goals_exp = total_goals_exp * half_time_ratio
    
    # 计算主客进球比例
    home_goals_exp = sum(h * prob for h, a, prob in candidates)
    away_goals_exp = sum(a * prob for h, a, prob in candidates)
    
    if home_goals_exp + away_goals_exp > 0:
        home_ratio = home_goals_exp / (home_goals_exp + away_goals_exp)
    else:
        home_ratio = 0.5
    
    # 半场进球期望
    half_home_exp = half_goals_exp * home_ratio
    half_away_exp = half_goals_exp * (1 - home_ratio)
    
    # 使用泊松分布计算半场各种比分的概率
    def poisson_prob(lam, k):
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    # 计算半场各种结果的概率
    half_probs = {}
    max_half_goals = 3  # 考虑最多3个进球
    
    for h in range(max_half_goals + 1):
        for a in range(max_half_goals + 1):
            if h + a <= max_half_goals:
                prob = poisson_prob(half_home_exp, h) * poisson_prob(half_away_exp, a)
                half_probs[(h, a)] = prob
    
    # 归一化半场概率
    half_total = sum(half_probs.values())
    if half_total > 0:
        half_probs = {k: v / half_total for k, v in half_probs.items()}
    
    # 计算半全场组合概率
    htf_probs = {}
    
    # 定义半场结果映射
    def get_half_result(h, a):
        if h > a:
            return 'H'
        elif h < a:
            return 'A'
        else:
            return 'D'
    
    # 定义全场结果映射
    def get_full_result(h, a):
        if h > a:
            return 'H'
        elif h < a:
            return 'A'
        else:
            return 'D'
    
    # 计算每种半全场组合的概率
    for (half_h, half_a), half_prob in half_probs.items():
        half_res = get_half_result(half_h, half_a)
        
        for full_h, full_a, full_prob in candidates:
            full_res = get_full_result(full_h, full_a)
            key = f"{half_res}{full_res}"
            
            # 考虑逻辑约束：半场比分应该合理地导致全场比分
            if (full_h >= half_h) and (full_a >= half_a):
                if key not in htf_probs:
                    htf_probs[key] = 0
                htf_probs[key] += half_prob * full_prob
    
    # 归一化半全场概率
    htf_total = sum(htf_probs.values())
    if htf_total > 0:
        htf_probs = {k: v / htf_total for k, v in htf_probs.items()}
    
    # 添加友好名称
    htf_names = {
        'HH': '半胜全胜',
        'HD': '半胜全平',
        'HA': '半胜全负',
        'DH': '半平全胜',
        'DD': '半平全平',
        'DA': '半平全负',
        'AH': '半负全胜',
        'AD': '半负全平',
        'AA': '半负全负',
    }
    
    result = []
    for key in ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']:
        prob = htf_probs.get(key, 0)
        result.append({
            'code': key,
            'name': htf_names[key],
            'probability': round(prob * 100, 1),
            'raw_prob': prob,
        })
    
    # 按概率排序
    result.sort(key=lambda x: -x['probability'])
    
    return result


def predict_scores(asian, euro, total, team_strength=None, league_profile=None, 
                   model_type='poisson', enable_draw_calibration=True,
                   enable_calibration=False, calibration_method='platt',
                   enable_ensemble=False, ensemble_size=5):
    """
    比分预测主函数：支持多种模型类型。
    
    参数：
        model_type: 'poisson'（泊松）、'negative_binomial'（负二项）、'bayesian'（贝叶斯）
        enable_draw_calibration: 是否启用平局概率校准
        enable_calibration: 是否启用概率输出校准
        calibration_method: 概率校准方法，'platt' 或 'isotonic'
        enable_ensemble: 是否启用多模型集成
        ensemble_size: 集成模型数量
    """
    # 如果启用多模型集成，直接调用集成函数
    if enable_ensemble:
        return ensemble_predict_scores(asian, euro, total, team_strength, league_profile,
                                      num_models=ensemble_size, method='average')
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

    # 平局概率校准
    if enable_draw_calibration:
        home_draw_rate = team_strength.get('draw_rate_home', 0.25) if team_strength else 0.25
        away_draw_rate = team_strength.get('draw_rate_away', 0.25) if team_strength else 0.25
        league_draw_rate = league_profile.get('draw_rate', 0.25) if league_profile else 0.25
        
        p_home, p_draw, p_away = calibrate_draw_probability(
            p_home, p_draw, p_away, asian['handicap'],
            home_draw_rate, away_draw_rate, league_draw_rate
        )

    euro_lams = None
    el = euro.get('implied_lambdas')
    if el:
        euro_lams = (el['home'], el['away'])

    # 根据模型类型选择不同的预测方法
    if model_type == 'bayesian':
        # 贝叶斯框架：MCMC 采样后验分布
        targets = [p_home, p_draw, p_away]
        matrix, credible_interval, samples = bayesian_predict_scores(
            targets, target_total_pre, supremacy, league_profile, team_strength
        )
        candidates = sorted(matrix.items(), key=lambda kv: -kv[1])
        
        # 从后验均值获取 lambda 值
        lam_home = sum(s[0] for s in samples) / len(samples) if samples else (target_total_pre + supremacy) / 2
        lam_away = sum(s[1] for s in samples) / len(samples) if samples else (target_total_pre - supremacy) / 2
        target_total = target_total_pre
        
        meta = {
            'supremacy_asian': sup_asian,
            'supremacy_euro': sup_euro,
            'supremacy_blended': supremacy,
            'target_total': target_total,
            'credible_interval': credible_interval,
            'model_type': 'bayesian',
        }
        return candidates, lam_home, lam_away, meta

    # 频率学派方法（泊松或负二项）
    try:
        lam_home, lam_away, target_total, rho = fit_lambdas_from_markets(
            supremacy, line, p_over, p_home, p_draw, p_away,
            open_total_line=open_line, team_strength=team_strength, euro_lambdas=euro_lams,
            league_profile=league_profile,
        )
        
        # 选择分布类型
        distribution = 'negative_binomial' if model_type == 'negative_binomial' else 'poisson'
        matrix = build_score_matrix(lam_home, lam_away, rho=rho, distribution=distribution)
        
        margins = _matrix_margins(matrix)
        err = sum(
            (margins[k] - t) ** 2
            for k, t in zip(('home', 'draw', 'away'), (p_home, p_draw, p_away))
        )
        if err > 0.012:
            matrix = calibrate_to_euro(matrix, p_home, p_draw, p_away)
    except (ValueError, ZeroDivisionError, OverflowError):
        lam_home, lam_away = estimate_lambdas(supremacy, line)
        distribution = 'negative_binomial' if model_type == 'negative_binomial' else 'poisson'
        matrix = build_score_matrix(lam_home, lam_away, distribution=distribution)
        matrix = calibrate_to_euro(matrix, p_home, p_draw, p_away)
        target_total = line
        rho = 0.0

    # 应用残差修正（如果有训练好的模型）
    features = _build_residual_features(asian, euro, total, team_strength, league_profile)
    matrix = apply_residual_correction(matrix, features)

    # 应用概率输出校准
    if enable_calibration:
        # 获取联赛名称（用于加载特定联赛的校准参数）
        league_name = league_profile.get('name', 'default') if league_profile else 'default'
        
        # 获取该联赛的校准数据（如果没有缓存则自动训练）
        calibration_data = get_league_calibration_data(league_name)
        
        log.debug(f"使用联赛 {league_name} 的校准参数: Platt(A={calibration_data['platt_params'][0]:.4f}, B={calibration_data['platt_params'][1]:.4f})")
        
        matrix = calibrate_probabilities(matrix, method=calibration_method, calibration_data=calibration_data)

    candidates = sorted(matrix.items(), key=lambda kv: -kv[1])
    meta = {
        'supremacy_asian': sup_asian,
        'supremacy_euro': sup_euro,
        'supremacy_blended': supremacy,
        'target_total': target_total,
        'model_type': model_type,
        'distribution': distribution,
        'calibrated': enable_calibration,
        'calibration_method': calibration_method if enable_calibration else None,
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
    pool = min(16, len(candidates))  # 原12，扩大候选池让高比分有更多入选机会
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

    # 多样性兜底：若所有推荐总进球≤1，强制从候选中补一个总进球≥2的比分
    if n >= 2 and len(picked) >= 2 and all(h + a <= 1 for h, a, _ in picked):
        for (h, a), prob in candidates:
            if (h, a) not in seen and h + a >= 2:
                picked[-1] = (h, a, prob)  # 替换排名最低那个低比分
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
    log.info('分析比赛 %s vs %s (id=%s)', home, away, mid)

    try:
        yazhi_raw = fetch_yazhi(mid)
        asian = analyze_asian(yazhi_raw)
        log.debug(f"亚盘数据获取成功: keys={list(asian.keys())}")
    except Exception as e:
        raise ValueError(f"亚盘数据获取失败: {e}")
    
    try:
        euro_raw = fetch_ouzhi(mid)
        log.debug(f"欧赔原始数据获取成功: keys={list(euro_raw.keys())}")
    except Exception as e:
        raise ValueError(f"欧赔原始数据获取失败: {e}")
    
    try:
        euro = analyze_euro(euro_raw)
        log.debug(f"欧赔分析完成: keys={list(euro.keys())}")
    except Exception as e:
        raise ValueError(f"欧赔分析失败: {e}")
    
    try:
        daxiao_raw = fetch_daxiao(mid)
        total = analyze_total(daxiao_raw)
        log.debug(f"大小球数据获取成功: keys={list(total.keys())}")
    except Exception as e:
        raise ValueError(f"大小球数据获取失败: {e}")
    
    team = fetch_team_strength(mid, home, away, league_profile)
    if team:
        team['league_profile'] = league_profile

    target_total = total['implied_total']
    lp_avg = league_profile.get('avg_goal', AVG_LEAGUE_GOAL)
    target_total = max(lp_avg * 1.6, min(lp_avg * 3.5, target_total))  # 原1.4/3.2，上调下限使λ更真实
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

    # 新增：联合异常特征
    joint_anomaly = compute_joint_anomaly(yazhi_raw, daxiao_raw)
    euro_asian_dev = compute_euro_asian_deviation(euro['close'], asian['handicap'])
    
    # 新增：凯利时序趋势分析
    kelly_trend = analyze_kelly_trend(euro_raw.get('series', []))
    if 'kelly' in euro:
        euro['kelly']['trend'] = kelly_trend

    confidence = compute_prediction_confidence(asian, euro, total, team)

    # 新增：机器学习模型预测
    ml_result = None
    try:
        from .ml import MLFootballPredictor
        ml_predictor = MLFootballPredictor(model_type='auto')
        # 准备特征
        ml_features = {
            'elo_home': team.get('elo_home', 1500) if team else 1500,
            'elo_away': team.get('elo_away', 1500) if team else 1500,
            'euro_home': euro['raw_odds']['close']['home'],
            'euro_draw': euro['raw_odds']['close']['draw'],
            'euro_away': euro['raw_odds']['close']['away'],
            'asian_handicap': asian['handicap'],
            'asian_home_water': asian['close_water']['home'],
            'asian_away_water': asian['close_water']['away'],
            'total_line': total['close_line'],
            'total_over_water': total['close_water']['over'],
            'total_under_water': total['close_water']['under'],
            'home_attack': team.get('attack_home', 1.3) if team else 1.3,
            'home_defense': team.get('defense_home', 1.2) if team else 1.2,
            'away_attack': team.get('attack_away', 1.2) if team else 1.2,
            'away_defense': team.get('defense_away', 1.3) if team else 1.3,
            'home_form': team['home_recent']['form_pts'] / 3.0 if team else 0.5,
            'away_form': team['away_recent']['form_pts'] / 3.0 if team else 0.5
        }
        ml_probs = ml_predictor.predict(ml_features)
        ml_result = {
            'probabilities': ml_probs,
            'model_type': ml_predictor.model_type,
            'is_trained': ml_predictor.is_trained
        }
        log.info(f"机器学习模型预测完成: 主胜{ml_probs['home']:.3f}, 平局{ml_probs['draw']:.3f}, 客胜{ml_probs['away']:.3f}")
    except Exception as e:
        log.warning(f"机器学习模型预测失败: {e}")

    candidates, lam_home, lam_away, meta = predict_scores(
        asian, euro, total, team_strength=team, league_profile=league_profile,
        model_type='negative_binomial',
        enable_draw_calibration=True,
        enable_calibration=True,
        calibration_method='platt',
        enable_ensemble=True,
        ensemble_size=5,
    )

    # 新增：Dixon-Coles 模型预测（依赖 predict_scores 产出的 lam_home/lam_away）
    dixon_coles_result = None
    try:
        from .ml import dixon_coles_score_matrix, dixon_coles_1x2_prob
        dc_matrix = dixon_coles_score_matrix(lam_home, lam_away, max_goals=MAX_GOALS, rho=0.1)
        dc_1x2 = dixon_coles_1x2_prob(lam_home, lam_away, max_goals=MAX_GOALS, rho=0.1)
        dixon_coles_result = {
            'matrix': dc_matrix,
            '1x2': dc_1x2,
            'rho': 0.1
        }
        log.info(f"Dixon-Coles 模型预测完成: 主胜{dc_1x2['home']:.3f}, 平局{dc_1x2['draw']:.3f}, 客胜{dc_1x2['away']:.3f}")
    except Exception as e:
        log.warning(f"Dixon-Coles 模型预测失败: {e}")

    # 准备欧赔赔率用于冷热计算（基于赔率隐含概率）
    euro_odds_for_heat = {
        'home': euro['raw_odds']['close']['home'],
        'draw': euro['raw_odds']['close']['draw'],
        'away': euro['raw_odds']['close']['away'],
    }

    # 更新比分冷热计算：使用赔率隐含概率 vs 模型概率
    top_scores = [
        _score_entry(h, a, prob, score_heat_label(h, a, prob, league_profile, euro_odds_for_heat))
        for (h, a), prob in candidates[:5]
    ]
    recommend = []
    for h, a, prob in _pick_recommendations(
        candidates, asian, euro, total, confidence=confidence, league_profile=league_profile,
    ):
        heat, _ = score_heat_label(h, a, prob, league_profile, euro_odds_for_heat)
        recommend.append({
            **_score_entry(h, a, prob, (heat, _)),
            'reasons': _recommend_reasons(h, a, asian, euro, total, team, heat=heat),
        })

    # 计算半全场概率
    half_full_time = calculate_half_full_time_probs(candidates, team)

    # 新增：进球数推荐
    goal_count_result = None
    try:
        from .ml import predict_goal_counts_from_candidates
        goal_count_result = predict_goal_counts_from_candidates(candidates, max_goals=MAX_GOALS)
        log.info(f"进球数推荐完成: {goal_count_result['recommendations']}")
    except Exception as e:
        log.warning(f"进球数推荐失败: {e}")

    return {
        'match': {k: match.get(k) for k in ('home', 'away', 'league', 'time', 'match_id', 'num')},
        'league_profile': league_profile,
        'asian': asian,
        'euro': euro,
        'total': total,
        'team': team,
        'confidence': confidence,
        'anomaly': {
            'joint_water': joint_anomaly,
            'euro_asian_deviation': euro_asian_dev,
        },
        'model': {
            'lam_home': lam_home, 'lam_away': lam_away,
            'top_scores': top_scores, 'recommend': recommend,
            'half_full_time': half_full_time,
            'goal_count': goal_count_result,
            'dixon_coles': dixon_coles_result,
            'ml': ml_result,
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
    
    # 显示模型配置信息
    model_info = []
    if model.get('model_type'):
        model_type_name = {
            'poisson': '泊松分布',
            'negative_binomial': '负二项分布',
            'bayesian': '贝叶斯推断',
            'ensemble': '多模型集成'
        }.get(model['model_type'], model['model_type'])
        model_info.append(f"模型: {model_type_name}")
    
    if model.get('calibrated'):
        calib_method_name = {
            'platt': 'Platt缩放',
            'isotonic': '等渗回归'
        }.get(model.get('calibration_method'), model.get('calibration_method'))
        model_info.append(f"校准: {calib_method_name}")
    
    if model.get('ensemble_size'):
        model_info.append(f"集成规模: {model['ensemble_size']}模型")
    
    if model_info:
        print(f"  模型配置: {', '.join(model_info)}")
    
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
        
        # 新增：凯利时序趋势
        kelly_trend = kelly.get('trend')
        if kelly_trend and kelly_trend['summary'] != '数据不足':
            print(f"  凯利走势: {kelly_trend['summary']}")
            if kelly_trend.get('crossing_events'):
                for event in kelly_trend['crossing_events']:
                    print(f"    {event['desc']}")

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

    # 新增：联合异常特征
    anomaly = result.get('anomaly')
    if anomaly:
        joint_water = anomaly.get('joint_water')
        euro_asian_dev = anomaly.get('euro_asian_deviation')
        
        print("\n【联合异常特征分析】")
        if joint_water:
            print(f"  水位变化乘积: 主队水位变化{joint_water['asian_water_change']:+.3f} × 大球水位变化{joint_water['total_water_change']:+.3f} = {joint_water['joint_water_feature']:+.4f}")
            if joint_water.get('hint_desc'):
                print(f"  ⚡ {joint_water['hint_desc']}")
        
        if euro_asian_dev:
            print(f"  欧赔亚盘偏差: 欧赔隐含让球{euro_asian_dev['implied_handicap']:+.2f} vs 实际盘口{euro_asian_dev['actual_handicap']:+.2f}，偏差{euro_asian_dev['deviation']:+.2f}")

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
    
    # 显示概率校准和集成信息
    if model.get('calibrated'):
        print(f"  ✓ 概率已校准: 应用{model['calibration_method']}方法")
    
    if model.get('ensemble_size'):
        print(f"  ✓ 多模型集成: 融合{model['ensemble_size']}个模型输出")

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
