"""
双色球预测模块
================
- 红球 1-33 选 6，蓝球 1-16 选 1
- 基于历史开奖的频率 / 冷热 / 近期趋势做统计预测（娱乐参考，不保证中奖）
- 支持在线刷新（500.com）与离线种子数据

API:
    from src.ssq import run_prediction, fetch_data, clear_cache
"""

import os
import re
import json
import time
import random
from pathlib import Path

try:
    from src.common.paths import data_path
except Exception:  # pragma: no cover
    def data_path(name):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', name)


# 本地缓存文件（与 data/ 同步）；种子文件用于完全离线兜底
HISTORY_FILE = Path(data_path('ssq_history.json'))
_SEED_FILE = Path(__file__).with_name('_seed.json')

SSQ_URL = "https://datachart.500.com/ssq/history/newinc/history.php?start={start}&end={end}"

RED_RANGE = list(range(1, 34))      # 红球 1-33
BLUE_RANGE = list(range(1, 17))     # 蓝球 1-16
RED_COUNT = 6
RECENT_WINDOW = 30                  # 近期趋势窗口
DEFAULT_RECENT = 15                 # 页面默认展示近期期数
NUM_SETS = 5                        # 推荐注数


def _load_seed():
    try:
        return json.loads(_SEED_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []


def load_history(force_refresh=False):
    """读取历史开奖。优先在线最新数据，回退本地文件 / 种子文件。"""
    if not force_refresh and HISTORY_FILE.exists():
        try:
            recs = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
            if recs:
                return recs
        except Exception:
            pass
    recs = _fetch_online()
    if recs:
        try:
            HISTORY_FILE.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
        return recs
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return _load_seed()


def _fetch_online():
    """从 500.com 抓取 2026 年双色球历史开奖。"""
    import urllib.request
    import ssl
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = SSQ_URL.format(start=26001, end=26099)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=25, context=ctx).read().decode('utf-8', 'ignore')
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S)
        out = []
        for r in rows:
            tds = re.findall(r'<td[^>]*>(.*?)</td>', r, re.S)
            clean = [re.sub(r'<[^>]+>', '', c).strip() for c in tds]
            clean = [t for t in clean if t != '']
            if len(clean) >= 9 and re.match(r'^26\d{3}$', clean[1]):
                red = [int(clean[2]), int(clean[3]), int(clean[4]),
                       int(clean[5]), int(clean[6]), int(clean[7])]
                out.append({'period': clean[1], 'red': red, 'blue': int(clean[8])})
        out.sort(key=lambda x: x['period'])
        return out if out else None
    except Exception:
        return None


def fetch_data(force_refresh=False):
    """兼容其它模块的 fetch 接口：返回并刷新历史数据。"""
    return load_history(force_refresh=force_refresh)


def _next_period(period):
    """根据最近一期期号推算下一期（简单 +1）。"""
    try:
        return str(int(period) + 1)
    except Exception:
        return ''


def _analyze(history):
    """统计红球 / 蓝球的整体频率与近窗口频率。"""
    red_all = [n for r in history for n in r['red']]
    blue_all = [r['blue'] for r in history]
    recent = history[-RECENT_WINDOW:]

    def freq(nums, span):
        d = {n: 0 for n in span}
        for n in nums:
            d[n] = d.get(n, 0) + 1
        return d

    red_freq = freq(red_all, RED_RANGE)
    blue_freq = freq(blue_all, BLUE_RANGE)
    red_recent = freq([n for r in recent for n in r['red']], RED_RANGE)
    blue_recent = freq([r['blue'] for r in recent], BLUE_RANGE)

    red_sorted = sorted(red_freq.items(), key=lambda kv: kv[1], reverse=True)
    blue_sorted = sorted(blue_freq.items(), key=lambda kv: kv[1], reverse=True)

    return {
        'red_freq': red_freq,
        'blue_freq': blue_freq,
        'red_recent': red_recent,
        'blue_recent': blue_recent,
        'hot_red': [n for n, _ in red_sorted[:10]],
        'cold_red': [n for n, _ in red_sorted[-10:]],
        'hot_blue': [n for n, _ in blue_sorted[:5]],
        'cold_blue': [n for n, _ in blue_sorted[-5:]],
    }


def _weighted_sample(nums, weights, k, rng=None):
    """按权重不放回抽样。rng 可传入固定种子的随机发生器，保证可复现。"""
    rng = rng or random
    pool = list(nums)
    w = list(weights)
    chosen = []
    while len(chosen) < k and pool:
        total = sum(w)
        if total <= 0:
            pick = rng.randrange(len(pool))
        else:
            r = rng.uniform(0, total)
            acc = 0.0
            pick = len(pool) - 1
            for i, wi in enumerate(w):
                acc += wi
                if r <= acc:
                    pick = i
                    break
        chosen.append(pool.pop(pick))
        w.pop(pick)
    return chosen


def _is_valid_red(red):
    """合法性约束：奇偶、和值、连号、跨度。"""
    if len(set(red)) != RED_COUNT:
        return False
    s = sum(red)
    if not (80 <= s <= 150):
        return False
    odd = sum(1 for n in red if n % 2 == 1)
    if odd < 2 or odd > 4:
        return False
    lo, hi = min(red), max(red)
    if hi - lo < 15 or hi - lo > 32:
        return False
    seq = sorted(red)
    run = 1
    maxrun = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1] + 1:
            run += 1
            maxrun = max(maxrun, run)
        else:
            run = 1
    if maxrun > 3:
        return False
    return True


def _predict_sets(history, analysis, n=NUM_SETS, seed=None):
    """生成 n 注推荐号码。seed 固定时（如最新期号），结果完全可复现。"""
    rng = random.Random(seed)
    prev_red = set(history[-1]['red']) if history else set()
    prev_blue = history[-1]['blue'] if history else None
    # 综合权重 = 整体频率 + 1.5 * 近期频率
    red_w = [analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5 for x in RED_RANGE]
    blue_w = [analysis['blue_freq'][x] + 1.5 * analysis['blue_recent'][x] + 0.3 for x in BLUE_RANGE]

    sets = []
    tries = 0
    while len(sets) < n and tries < 5000:
        tries += 1
        red = sorted(_weighted_sample(RED_RANGE, red_w, RED_COUNT, rng))
        if not _is_valid_red(red):
            continue
        if set(red) == prev_red:
            continue
        blue = _weighted_sample(BLUE_RANGE, blue_w, 1, rng)[0]
        if blue == prev_blue:
            blue = _weighted_sample(BLUE_RANGE, blue_w, 1, rng)[0]
        sets.append({'red': red, 'blue': blue})
    # 兜底：约束太严未能凑足则放宽
    while len(sets) < n:
        red = sorted(rng.sample(RED_RANGE, RED_COUNT))
        blue = rng.choice(BLUE_RANGE)
        sets.append({'red': red, 'blue': blue})
    return sets


def run_prediction(data=None, force_refresh=False, recent=DEFAULT_RECENT):
    """主入口：返回近期开奖 + 预测结果。"""
    history = data if data is not None else load_history(force_refresh=force_refresh)
    if not history:
        return {'error': '无历史数据'}
    history = sorted(history, key=lambda x: x['period'])
    latest = history[-1]
    analysis = _analyze(history)
    # 以最新一期期号为随机种子：同一期数据下预测结果固定不变，
    # 只有出现新一期开奖（期号变化）时才会更新预测。
    sets = _predict_sets(history, analysis, n=NUM_SETS, seed=int(latest['period']))

    recent_list = history[-recent:][::-1]  # 最近在前

    return {
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'latest_period': latest['period'],
        'next_period': _next_period(latest['period']),
        'history': recent_list,
        'prediction': {
            'primary': sets[0],
            'sets': sets,
        },
        'analysis': {
            'hot_red': analysis['hot_red'],
            'cold_red': analysis['cold_red'],
            'hot_blue': analysis['hot_blue'],
            'cold_blue': analysis['cold_blue'],
        },
        'meta': {
            'red_range': '1-33',
            'blue_range': '1-16',
            'total_draws': len(history),
            'disclaimer': '统计预测，仅供娱乐参考，不保证中奖。',
        },
    }


def clear_cache():
    """清除本地历史缓存，下次调用重新抓取。"""
    try:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
            return True
    except Exception:
        pass
    return False
