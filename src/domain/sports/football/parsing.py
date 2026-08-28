# -*- coding: utf-8 -*-
"""足球赔率页解析：盘口文本、公司赔率、联赛画像、球队近况。

纯计算——HTML/文本进，结构化数据出。**不发请求、不读全局配置、不看时钟**；
抓取本体留在 `src/football/parsing.py`。

阈值一律参数化（判据 10）：融合权重、分歧指数的三个门槛、近况融合权重等。
默认值取自迁移当时 `config.py` 的真实取值，是公开契约的一部分（判据 29）。
"""

import logging
import re

log = logging.getLogger('domain.football.parsing')

# 迁移当时 config 的真实取值
CLOSE_BLEND_WEIGHT = 0.72

# 原先内联在 `calculate_bookmaker_consensus` 里的三个数
SHARP_DIFF_EPS = 0.125          # 与平均盘差多少才算有方向
SHARP_CONFIDENCE_SCALE = 0.5    # 差异归一到 [0,1] 的分母
SHARP_ADJUSTMENT_SCALE = 0.15   # 置信度换算成 λ 调整的系数

# 原先内联在 `fetch_team_strength` / `resolve_league_profile` 里的数
VENUE_BLEND_WEIGHT = 0.68       # 主客场近况 vs 全部近况
MOMENTUM_CLAMP = 0.35
MOMENTUM_SCALE = 0.12
LIVE_PROFILE_MIN_SAMPLE = 50    # 实时画像至少要这么多场才参与融合
LIVE_PROFILE_WEIGHT = 0.3       # 融合里实时画像占的份额
DEFAULT_AVG_GOAL = 1.42
DEFAULT_DRAW_RATE = 0.25
TEAM_CONTEXT_CHARS = 140        # 队名识别只看战绩前这么多字符
TEAM_SUFFIX_LENGTHS = (4, 3, 2)  # 简称匹配尝试的尾串长度

DEFAULT_TOTAL_LINE = 2.5

RECENT_FORM_PAT = re.compile(
    r'近(\d+)场战绩.*?'
    r'<span class="ying">(\d+)胜</span>.*?'
    r'<span class="ping">(\d+)平</span>.*?'
    r'<span class="shu">(\d+)负</span>.*?'
    r'进<span class="ying">(\d+)球</span>失<span class="shu">(\d+)球</span>',
    re.DOTALL,
)


# ---------------- 文本与盘口 ----------------

def html_to_text(html):
    """去除标签与转义空白，压缩为单行纯文本"""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_avg_numbers(html, keyword='平均值'):
    """从 HTML 中提取包含 keyword 行**之后**的全部数字"""
    text = html_to_text(html)
    idx = text.find(keyword)
    if idx < 0:
        raise ValueError(f"未找到'{keyword}'行")
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text[idx:])
    return [float(n) for n in numbers]


def get_close_total_line(total: dict, default: float = DEFAULT_TOTAL_LINE) -> float:
    """统一获取大小球终盘线

    三种数据结构都支持：`close_line` / `line` / `close.line`。
    **注意用的是 `or` 不是 `is None`**——所以 0 会被当成缺失落到下一个来源，
    而大小球线不可能是 0，这个落差无害。行为原样保留。
    """
    return (
        total.get('close_line')
        or total.get('line')
        or total.get('close', {}).get('line')
        or default
    )


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
        return DEFAULT_TOTAL_LINE


# 不含「受让」前缀，按字符串长度降序——长的先匹配
HANDICAP_TEXT_MAP = [
    ('平手/半球', 0.25),
    ('半球/一球', 0.75),
    ('一球/球半', 1.25),
    ('球半/两球', 1.75),
    ('两球/两球半', 2.25),
    ('平半', 0.25),
    ('半球', 0.5),
    ('半一', 0.75),
    ('一球', 1.0),
    ('球半', 1.5),
    ('两球', 2.0),
    ('两球半', 2.5),
    ('平手', 0),
]


def handicap_text_to_num(handicap_text, mapping=HANDICAP_TEXT_MAP):
    """将让球文字转换为数字

    主队让球 → 正数（如：半球 → 0.5）；主队受让球 → 负数（如：受让半球 → -0.5）。

    **`HANDICAP_TEXT_MAP` 的顺序不是按长度严格降序**：`两球` 排在 `两球半`
    之前，所以「两球半」在**包含匹配**那一轮会先命中 `两球` 返回 2.0。
    精确匹配那一轮不受影响。行为原样保留，用例把它钉住了。
    """
    if not handicap_text:
        return 0

    is_receive = False
    text_to_check = handicap_text
    if handicap_text.startswith('受让'):
        is_receive = True
        text_to_check = handicap_text[2:]
    elif handicap_text.startswith('受'):
        is_receive = True
        text_to_check = handicap_text[1:]

    for key, value in mapping:
        if key == text_to_check:
            return -value if is_receive else value

    for key, value in mapping:
        if key in text_to_check:
            return -value if is_receive else value

    return 0


def extract_handicap_from_segment(segment, before_val, after_val):
    """从文本片段中提取两个数字之间的让球值（可能是文本或数字）"""
    pat = re.compile(
        rf'{re.escape(str(before_val))}\s+([^\d\s]+(?:/[^\d\s]+)?)\s+{re.escape(str(after_val))}'
    )
    m = pat.search(segment)
    if m:
        handicap_str = m.group(1)
        try:
            return float(handicap_str)
        except ValueError:
            return parse_handicap(handicap_str)

    pat2 = re.compile(
        rf'{re.escape(str(before_val))}\s+(-?[\d.]+)\s+{re.escape(str(after_val))}'
    )
    m2 = pat2.search(segment)
    if m2:
        return float(m2.group(1))

    return 0


def parse_odds_value(value, field_name, match_id):
    """解析赔率值，确保为有效正数"""
    if value is None:
        raise ValueError(f"赔率值为空: {field_name} (match_id={match_id})")
    try:
        val = float(value)
        if val <= 0:
            raise ValueError(f"赔率值必须为正数: {field_name} = {val} (match_id={match_id})")
        return val
    except (ValueError, TypeError):
        raise ValueError(f"赔率值解析失败: {field_name} = {repr(value)} (match_id={match_id})")


# ---------------- 单家公司赔率 ----------------

COMPANY_ALIASES = {
    'Bet365': ['**t3*5', 'B*t365', 'B**365'],
    'Pinnacle': ['Pi****le', 'Pin***le', 'Pinnacle平*'],
}

ASIAN_HANDICAP_PATTERN = re.compile(
    r'(受让平手|受让平半|受让半球|受让半一|受让一球|受让球半|受让两球|受让两球半'
    r'|平手|平半|半球|半一|一球|球半|两球|两球半|平手/半球|半球/一球|一球/球半'
    r'|球半/两球|两球/两球半|受平手|受平半|受半球|受半一|受一球|受球半|受两球|受两球半)'
)

WATER_RANGE = (0.3, 2.5)
HANDICAP_RANGE = (-5, 5)
TOTAL_LINE_RANGE = (1.5, 5.0)
SEGMENT_CHARS = 500
MAX_NUMBERS = 25


def extract_company_odds(html, company_name, is_total=False):
    """从 HTML 中提取指定博彩公司的赔率数据

    返回列表：亚盘为 [初主水, 初让球, 初客水, 终主水, 终让球, 终客水, 终时间, 初时间]，
    大小球同构（水位换成大/小球、让球换成盘口线）。取不到返回 None。

    **线上从来没有取到过**（2026-08-28 实测：真实页面的纯文本里既没有
    `Bet365`/`Pinnacle`，也没有任何一个别名——只有「平均值」行）。
    114 场缓存的 `bookmaker_consensus` 全是 None，7 天日志零痕迹。
    详见交接文档 §四。行为原样保留。
    """
    text = html_to_text(html)
    idx = text.find(company_name)

    if idx < 0:
        for alias in COMPANY_ALIASES.get(company_name, ()):
            idx = text.find(alias)
            if idx >= 0:
                log.debug(f"找到 {company_name} 的别名: {alias}")
                break

    if idx < 0:
        return None

    segment = text[idx:idx + SEGMENT_CHARS]

    # 公司名在行里重复两次，从第二次之后开始取数据
    second_idx = segment.find('**', 10)
    if second_idx > 0:
        data_part = segment[second_idx:]
        first_space = data_part.find(' ')
        if first_space > 0:
            data_part = data_part[first_space + 1:].strip()
    else:
        data_part = segment[len(company_name):]

    numbers = re.findall(r'-?\d+\.\d+|-?\d+', data_part)

    filtered = []
    for n in numbers[:MAX_NUMBERS]:
        num = float(n)
        if WATER_RANGE[0] <= num <= WATER_RANGE[1]:
            filtered.append(num)
        elif HANDICAP_RANGE[0] <= num <= HANDICAP_RANGE[1] and abs(num) != 0:
            filtered.append(num)

    times = re.findall(r'(\d{2}-\d{2}\s+\d{2}:\d{2})', data_part)

    log.debug(f"{company_name} 原始数字: {numbers[:15]}, 过滤后: {filtered}, 时间: {times}")

    if is_total:
        return _company_total_row(company_name, data_part, numbers, filtered, times)
    return _company_asian_row(company_name, data_part, filtered, times)


def _company_total_row(company_name, data_part, numbers, filtered, times):
    total_matches = re.findall(r'(\d+\.?\d*球)', data_part)

    final_line = initial_line = 0
    for match in total_matches[:2]:
        try:
            line_val = float(match.replace('球', ''))
        except ValueError:
            continue
        if TOTAL_LINE_RANGE[0] <= line_val <= TOTAL_LINE_RANGE[1]:
            if final_line == 0:
                final_line = line_val
            elif initial_line == 0:
                initial_line = line_val

    if final_line == 0:
        for n in numbers[:20]:
            num = float(n)
            if TOTAL_LINE_RANGE[0] <= num <= TOTAL_LINE_RANGE[1]:
                final_line = num
                break

    if len(filtered) < 4:
        return None

    final_over, final_under = filtered[0], filtered[1]
    initial_over = filtered[2] if len(filtered) > 2 else filtered[0]
    initial_under = filtered[3] if len(filtered) > 3 else filtered[1]
    if len(filtered) >= 6:
        initial_over, initial_under = filtered[3], filtered[4]

    result = [
        initial_over,
        initial_line if initial_line != 0 else final_line,
        initial_under,
        final_over,
        final_line,
        final_under,
        times[0] if len(times) > 0 else None,
        times[1] if len(times) > 1 else None,
    ]
    log.debug(f"{company_name} 大小球提取结果: {result}")
    return result


def _company_asian_row(company_name, data_part, filtered, times):
    handicap_matches = ASIAN_HANDICAP_PATTERN.findall(data_part)
    final_handicap = handicap_text_to_num(handicap_matches[0]) if handicap_matches else 0
    initial_handicap = (handicap_text_to_num(handicap_matches[1])
                        if len(handicap_matches) >= 2 else 0)

    if len(filtered) < 4:
        return None
    if not (WATER_RANGE[0] <= filtered[0] <= WATER_RANGE[1]
            and WATER_RANGE[0] <= filtered[1] <= WATER_RANGE[1]):
        return None

    def water_or(index, fallback):
        return (filtered[index] if len(filtered) > index
                and WATER_RANGE[0] <= filtered[index] <= WATER_RANGE[1] else fallback)

    result = [
        water_or(2, filtered[0]),
        initial_handicap if initial_handicap != 0 else final_handicap,
        water_or(3, filtered[1]),
        filtered[0],
        final_handicap,
        filtered[1],
        times[0] if len(times) > 0 else None,
        times[1] if len(times) > 1 else None,
    ]
    log.debug(f"{company_name} 亚盘提取结果: {result}")
    return result


def company_odds_to_markets(asian_row, total_row=None):
    """把 `extract_company_odds` 的两行拼成 {'asian': ..., 'total': ...}

    亚盘取不到就返回 None（大小球是可选的）——迁移前的判定原样保留。
    """
    if not asian_row:
        return None
    data = {
        'asian': {
            'open': {
                'handicap': asian_row[1],
                'home_odds': asian_row[0],
                'away_odds': asian_row[2],
                'time': asian_row[7] if len(asian_row) > 7 else None,
            },
            'close': {
                'handicap': asian_row[4],
                'home_odds': asian_row[3],
                'away_odds': asian_row[5],
                'time': asian_row[6] if len(asian_row) > 6 else None,
            },
        }
    }
    if total_row:
        data['total'] = {
            'open': {
                'line': total_row[1],
                'over_odds': total_row[0],
                'under_odds': total_row[2],
                'time': total_row[7] if len(total_row) > 7 else None,
            },
            'close': {
                'line': total_row[4],
                'over_odds': total_row[3],
                'under_odds': total_row[5],
                'time': total_row[6] if len(total_row) > 6 else None,
            },
        }
    return data


def bookmaker_consensus(bet365_data, pinnacle_data, avg_handicap, *,
                        diff_eps=SHARP_DIFF_EPS,
                        confidence_scale=SHARP_CONFIDENCE_SCALE,
                        adjustment_scale=SHARP_ADJUSTMENT_SCALE):
    """博彩公司分歧指数：Pinnacle 与平均盘的差异决定 Sharp Money 方向

    **线上从未 available**——见 `extract_company_odds` 的说明。
    """
    result = {
        'available': False,
        'bet365_handicap': None,
        'pinnacle_handicap': None,
        'avg_handicap': avg_handicap,
        'pinnacle_diff': 0.0,
        'sharp_bias': 'neutral',
        'adjustment': 0.0,
        'confidence': 0.0,
    }

    if not bet365_data or not pinnacle_data:
        return result

    try:
        bet365_handicap = bet365_data.get('asian', {}).get('close', {}).get('handicap')
        pinnacle_handicap = pinnacle_data.get('asian', {}).get('close', {}).get('handicap')
        if bet365_handicap is None or pinnacle_handicap is None:
            return result

        result['bet365_handicap'] = bet365_handicap
        result['pinnacle_handicap'] = pinnacle_handicap
        result['available'] = True
        result['pinnacle_diff'] = pinnacle_handicap - avg_handicap

        # Pinnacle 让球更激进（数值更大）= 更看好主队
        if result['pinnacle_diff'] > diff_eps:
            result['sharp_bias'] = 'home'
            result['confidence'] = min(result['pinnacle_diff'] / confidence_scale, 1.0)
            result['adjustment'] = result['confidence'] * adjustment_scale
        elif result['pinnacle_diff'] < -diff_eps:
            result['sharp_bias'] = 'away'
            result['confidence'] = min(abs(result['pinnacle_diff']) / confidence_scale, 1.0)
            result['adjustment'] = -result['confidence'] * adjustment_scale
        else:
            result['sharp_bias'] = 'neutral'
            result['confidence'] = 0.0
            result['adjustment'] = 0.0

        log.debug(
            "博彩公司分歧指数: Pinnacle=%s, 平均=%s, 差异=%.3f, 方向=%s, 调整=%.3f",
            pinnacle_handicap, avg_handicap, result['pinnacle_diff'],
            result['sharp_bias'], result['adjustment'],
        )
    except Exception as e:
        log.warning(f"计算博彩公司分歧指数失败: {e}")

    return result


# ---------------- 平均值行 → 三个市场 ----------------

def yazhi_from_page(html, nums, segment_chars=200):
    """由亚盘页与平均值行数字拼出初/终盘

    500.com 数值让球为负表示主让，取反以符合脚本惯例（正=主让）。
    平均值行第一组为初盘、第二组为终盘。
    """
    segment = html_to_text(html)
    idx = segment.find('平均值')
    segment = segment[idx:idx + segment_chars]

    open_hcap_raw = extract_handicap_from_segment(segment, nums[0], nums[2])
    close_segment = (segment[segment.find(str(nums[3])):]
                     if str(nums[3]) in segment else segment)
    close_hcap_raw = extract_handicap_from_segment(close_segment, nums[3], nums[5])

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
        },
    }


def daxiao_from_avg_numbers(nums):
    """大小球平均值行：盘口线为纯数字，第一组为初盘、第二组为终盘"""
    return {
        'open': {'line': nums[1], 'over_odds': nums[0], 'under_odds': nums[2]},
        'close': {'line': nums[4], 'over_odds': nums[3], 'under_odds': nums[5]},
    }


def ouzhi_from_series(series, match_id):
    """欧赔 JSON 时间序列 → 初/终盘。每条为 [主, 平, 客, 返还率, 时间, ...]

    **第 0 条是终盘、最后一条是初盘**（倒序）。
    """
    if not isinstance(series, list):
        raise ValueError(f"欧赔数据格式错误，期望列表但得到: {type(series)} (match_id={match_id})")
    if len(series) == 0:
        raise ValueError(f"欧赔数据为空列表 (match_id={match_id})")

    close, open_ = series[0], series[-1]
    if not isinstance(close, (list, tuple)) or len(close) < 3:
        raise ValueError(f"终盘数据格式错误: {close} (match_id={match_id})")
    if not isinstance(open_, (list, tuple)) or len(open_) < 3:
        raise ValueError(f"初盘数据格式错误: {open_} (match_id={match_id})")

    try:
        return {
            'open': {
                'home': parse_odds_value(open_[0], 'open_home', match_id),
                'draw': parse_odds_value(open_[1], 'open_draw', match_id),
                'away': parse_odds_value(open_[2], 'open_away', match_id),
                'return_rate': float(open_[3]) if len(open_) > 3 and open_[3] else None,
            },
            'close': {
                'home': parse_odds_value(close[0], 'close_home', match_id),
                'draw': parse_odds_value(close[1], 'close_draw', match_id),
                'away': parse_odds_value(close[2], 'close_away', match_id),
                'return_rate': float(close[3]) if len(close) > 3 and close[3] else None,
            },
            'series': series,
        }
    except ValueError as e:
        raise ValueError(f"欧赔数据解析失败: {e} (match_id={match_id})")


# ---------------- 联赛画像与球队近况 ----------------

def league_profile_from_matches(matches):
    """由最近比赛的比分统计出实时联赛画像；样本为空返回 None"""
    if not matches:
        return None

    total_matches = total_goals = draw_count = home_win_count = 0
    btts_count = over25_count = 0

    for match in matches:
        score = match.get('score')
        if not score:
            continue
        parts = score.split('-')
        if len(parts) != 2:
            continue
        try:
            h, a = map(int, parts)
        except ValueError:
            continue
        total_matches += 1
        total_goals += h + a
        if h == a:
            draw_count += 1
        if h > a:
            home_win_count += 1
        if h > 0 and a > 0:
            btts_count += 1
        if h + a >= 3:
            over25_count += 1

    if total_matches == 0:
        return None

    return {
        'avg_goal': total_goals / (total_matches * 2),  # 场均进球（单队）
        'draw_rate': draw_count / total_matches,
        'home_win_rate': home_win_count / total_matches,
        'btts_rate': btts_count / total_matches,
        'over25_rate': over25_count / total_matches,
        'sample_size': total_matches,
        'source': 'live',
    }


def resolve_static_league_profile(league_name, profiles):
    """按联赛名称在静态画像表里做最长键匹配"""
    name = (league_name or '').strip()
    profile = dict(profiles['default'])
    for key in sorted(profiles, key=len, reverse=True):
        if key != 'default' and key in name:
            profile.update(profiles[key])
            break
    return profile


def blend_league_profiles(static_profile, live_profile, league_name, *,
                          min_sample=LIVE_PROFILE_MIN_SAMPLE,
                          live_weight=LIVE_PROFILE_WEIGHT,
                          default_avg_goal=DEFAULT_AVG_GOAL,
                          default_draw_rate=DEFAULT_DRAW_RATE):
    """静态画像与实时画像的融合；样本不足时原样返回静态画像

    **只有 `avg_goal` 与 `draw_mult` 参与融合**，`home_boost` / `low_score`
    直接取静态值——实时画像里根本没有这两项。
    """
    name = (league_name or '').strip()
    if live_profile and live_profile.get('sample_size', 0) >= min_sample:
        static_weight = 1.0 - live_weight
        return {
            'avg_goal': (static_weight * static_profile.get('avg_goal', default_avg_goal)
                         + live_weight * live_profile.get('avg_goal', default_avg_goal)),
            'home_boost': static_profile.get('home_boost', 1.0),
            'low_score': static_profile.get('low_score', 1.0),
            'draw_mult': (static_weight * static_profile.get('draw_mult', 1.0)
                          + live_weight * (live_profile.get('draw_rate', default_draw_rate)
                                           / default_draw_rate)),
            'name': name or 'default',
            'live_sample': live_profile.get('sample_size', 0),
            'source': 'blended',
        }

    resolved = dict(static_profile)
    resolved['name'] = name or 'default'
    resolved['source'] = 'static'
    return resolved


def team_in_context(ctx, name):
    """队名与上下文模糊匹配（兼容简称）"""
    if not name:
        return False
    if name in ctx:
        return True
    for n in TEAM_SUFFIX_LENGTHS:
        if len(name) >= n and name[-n:] in ctx:
            return True
    return False


def parse_recent_form(groups):
    """把 `RECENT_FORM_PAT` 的六个捕获组换算成攻防与场均积分"""
    n, w, d, l = int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3])
    gf, ga = int(groups[4]), int(groups[5])
    n = max(n, 1)
    pts = w * 3 + d
    return {
        'games': n, 'wins': w, 'draws': d, 'losses': l,
        'gf': gf, 'ga': ga, 'attack': gf / n, 'defense': ga / n,
        'form_pts': pts / n,
    }


def blend_close_open(close_val, open_val, close_weight=CLOSE_BLEND_WEIGHT):
    """终盘为主、初盘为辅的线性融合"""
    if open_val is None:
        return close_val
    return close_weight * close_val + (1.0 - close_weight) * open_val


def parse_team_strength(html, home, away, *,
                        venue_weight=VENUE_BLEND_WEIGHT,
                        momentum_scale=MOMENTUM_SCALE,
                        momentum_clamp=MOMENTUM_CLAMP,
                        context_chars=TEAM_CONTEXT_CHARS):
    """从数据分析页抽出主客队近况，换算攻防强度。

    页面无足够数据时返回 None（不影响主流程）。**ELO 那部分是 IO，留在适配层。**

    每支球队最多取两条：第一条当「全部近况」，第二条当「主/客场近况」。
    """
    tagged = []
    for m in RECENT_FORM_PAT.finditer(html):
        # 仅用紧邻战绩前的短上下文识别队名，避免多场数据串台
        ctx = html_to_text(html[max(0, m.start() - context_chars):m.start()])
        tagged.append({'ctx': ctx, 'stats': parse_recent_form(m.groups())})

    if len(tagged) < 2:
        return None

    home_all = away_all = home_venue = away_venue = None
    for item in tagged:
        ctx, st = item['ctx'], item['stats']
        if team_in_context(ctx, home):
            if home_all is None:
                home_all = st
            elif home_venue is None:
                home_venue = st
        elif team_in_context(ctx, away):
            if away_all is None:
                away_all = st
            elif away_venue is None:
                away_venue = st

    if not home_all or not away_all:
        return None

    hv = home_venue or home_all
    av = away_venue or away_all
    form_diff = home_all['form_pts'] - away_all['form_pts']

    return {
        'home_recent': home_all,
        'away_recent': away_all,
        'home_venue': hv,
        'away_venue': av,
        'attack_home': blend_close_open(hv['attack'], home_all['attack'], venue_weight),
        'defense_home': blend_close_open(hv['defense'], home_all['defense'], venue_weight),
        'attack_away': blend_close_open(av['attack'], away_all['attack'], venue_weight),
        'defense_away': blend_close_open(av['defense'], away_all['defense'], venue_weight),
        'form_diff': form_diff,
        'momentum_supremacy': max(-momentum_clamp,
                                  min(momentum_clamp, form_diff * momentum_scale)),
        'summary': (
            f"主队近{home_all['games']}场 进{home_all['gf']}失{home_all['ga']}"
            f"（{home_all['form_pts']:.1f}分/场）；"
            f"客队近{away_all['games']}场 进{away_all['gf']}失{away_all['ga']}"
            f"（{away_all['form_pts']:.1f}分/场）"
        ),
    }
