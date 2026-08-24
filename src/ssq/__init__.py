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

from src.common import kv_store

try:
    from src.common.paths import data_path
except Exception:  # pragma: no cover
    def data_path(name):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', name)


# 本地缓存文件（与 data/ 同步）；种子文件用于完全离线兜底
# v3.0: 优先使用全量历史(2003年至今约3489期), 本地文件 ssq_history_full.json
HISTORY_FILE = Path(data_path('ssq_history.json'))
FULL_HISTORY_FILE = Path(data_path('ssq_history_full.json'))
_SEED_FILE = Path(__file__).with_name('_seed.json')

SSQ_URL = "https://datachart.500.com/ssq/history/newinc/history.php?start={start}&end={end}"

RED_RANGE = list(range(1, 34))      # 红球 1-33
BLUE_RANGE = list(range(1, 17))     # 蓝球 1-16
RED_COUNT = 6
RECENT_WINDOW = 30                  # 近期趋势窗口
DEFAULT_RECENT = 15                 # 页面默认展示近期期数
NUM_SETS = 5                        # 推荐注数
SSQ_PREDICTIONS_KEY = 'lottery_ssq_online_predictions'
# v3.4: 蓝球全覆盖保底方案（16注）——run_prediction 输出新增 blue_full_cover_plan。
# 16注蓝球互不相同覆盖全部16码，任意开奖蓝球必被其中1注命中（鸽笼原理，100%）。
# 红球按5注推荐轮转复用，无额外预测假设。纯组合保证，非预测能力。
SSQ_PREDICTION_VERSION = 'ssq-v3.4-blue-cover'


def ssq_prize_tier(red_hits, blue_hit):
    """双色球奖级判定（0=未中奖）。

    一等奖 6+1 / 二等奖 6+0 / 三等奖 5+1 / 四等奖 5+0,4+1 /
    五等奖 4+0,3+1 / 六等奖 2+1,1+1,0+1。
    注意：红球命中≤2 且蓝球未中时奖金为 0 —— "红球≥2命中率"是面子指标，
    只有蓝球命中或红球≥3+蓝/≥4 才对应真实奖金。
    """
    if red_hits == 6 and blue_hit:
        return 1
    if red_hits == 6:
        return 2
    if red_hits == 5 and blue_hit:
        return 3
    if red_hits == 5 or (red_hits == 4 and blue_hit):
        return 4
    if red_hits == 4 or (red_hits == 3 and blue_hit):
        return 5
    if blue_hit:
        return 6
    return 0


def load_prediction_records():
    """Load persisted predictions, oldest first."""
    try:
        records = kv_store.load(SSQ_PREDICTIONS_KEY, [])
        return records if isinstance(records, list) else []
    except Exception:
        return []


def save_prediction_record(period, sets):
    """Save one immutable prediction snapshot for a future draw."""
    if not period or not sets:
        return
    records = load_prediction_records()
    if any(str(item.get('period')) == str(period) for item in records):
        return
    records.append({
        'version': SSQ_PREDICTION_VERSION,
        'period': str(period),
        'sets': [
            {'red': sorted(set(item.get('red', []))), 'blue': item.get('blue')}
            for item in sets
        ],
        'actual': None,
        'settled': False,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    kv_store.save(SSQ_PREDICTIONS_KEY, records[-200:])


def settle_prediction_records(history):
    """Compare pending snapshots with newly available draw results."""
    records = load_prediction_records()
    draw_by_period = {str(item.get('period')): item for item in history}
    changed = 0
    for record in records:
        if record.get('settled'):
            continue
        actual = draw_by_period.get(str(record.get('period')))
        if not actual:
            continue
        actual_red = set(actual.get('red', []))
        actual_blue = actual.get('blue')
        record['actual'] = {'red': sorted(actual_red), 'blue': actual_blue}
        record['results'] = []
        for index, prediction in enumerate(record.get('sets', []), 1):
            red_hits = sorted(set(prediction.get('red', [])) & actual_red)
            blue_hit = prediction.get('blue') == actual_blue
            record['results'].append({
                'set_index': index,
                'red_hits': len(red_hits),
                'red_hit_numbers': red_hits,
                'blue_hit': blue_hit,
                'total_hits': len(red_hits) + int(blue_hit),
                # v3.3: 奖级结算（0=未中奖）——命中数≠中奖，奖级才对应真实奖金
                'prize': ssq_prize_tier(len(red_hits), blue_hit),
            })
        record['settled'] = True
        record['settled_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        changed += 1
    if changed:
        kv_store.save(SSQ_PREDICTIONS_KEY, records[-200:])
    return changed


def calculate_prediction_stats(records=None):
    records = records if records is not None else load_prediction_records()
    settled = [item for item in records if item.get('settled')]
    results = [result for item in settled for result in item.get('results', [])]
    # v3.3: 奖级分布（旧记录无 prize 字段，按命中数回算）
    prize_counts = {str(tier): 0 for tier in range(0, 7)}
    for item in results:
        tier = item.get('prize')
        if tier is None:
            tier = ssq_prize_tier(item.get('red_hits', 0), item.get('blue_hit', False))
        prize_counts[str(tier)] = prize_counts.get(str(tier), 0) + 1
    n = len(results)
    return {
        'total_records': len(records),
        'settled_count': len(settled),
        'unsettled_count': len(records) - len(settled),
        'total_sets': n,
        'red_hit_average': round(
            sum(item.get('red_hits', 0) for item in results) / n, 3
        ) if results else 0.0,
        'blue_hit_count': sum(1 for item in results if item.get('blue_hit')),
        'blue_hit_rate': round(
            sum(1 for item in results if item.get('blue_hit')) / n, 4
        ) if results else 0.0,
        # v3.3: 奖级统计 + 随机基准（诚实标注：单注理论值，公平摇奖不可超越）
        'prize_counts': prize_counts,
        'any_prize_rate': round((n - prize_counts.get('0', 0)) / n, 4) if n else 0.0,
        'baseline': {
            'note': '单注随机理论值：六等奖(蓝球命中)=6.25%，任意奖级≈6.71%，'
                    '五等奖(4红或3红+蓝)≈0.78%；组合5注蓝球去重后任1注中蓝=31.25%（已达上界）',
            'single_ticket_any_prize': 0.0671,
            'single_ticket_sixth': 0.0625,
            'five_sets_any_blue': 0.3125,
        },
    }


def _load_seed():
    try:
        return json.loads(_SEED_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []


def load_history(force_refresh=False):
    """读取历史开奖。v3.0: 优先全量历史(2003至今), 回退在线抓取 / 本地缓存 / 种子文件。"""
    # 全量历史文件优先（3489期，统计基础远优于仅当年数据）
    if not force_refresh and FULL_HISTORY_FILE.exists():
        try:
            recs = json.loads(FULL_HISTORY_FILE.read_text(encoding='utf-8'))
            if len(recs) >= 500:
                recs = sorted(recs, key=lambda x: str(x.get('period', '')))
                return recs
        except Exception:
            pass
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
            # 全量抓取结果同时写全量文件与常规文件
            FULL_HISTORY_FILE.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding='utf-8')
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
    """从 500.com 抓取双色球全量历史开奖（2003年至今约3489期）。"""
    import urllib.request
    import ssl
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = SSQ_URL.format(start=3001, end=26999)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=30, context=ctx).read().decode('utf-8', 'ignore')
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S)
        out = []
        for r in rows:
            tds = re.findall(r'<td[^>]*>(.*?)</td>', r, re.S)
            clean = [re.sub(r'<[^>]+>', '', c).strip() for c in tds]
            clean = [t for t in clean if t != '']
            if len(clean) >= 9 and re.match(r'^\d{5}$', clean[1]):
                red = [int(clean[2]), int(clean[3]), int(clean[4]),
                       int(clean[5]), int(clean[6]), int(clean[7])]
                out.append({'period': clean[1], 'red': red, 'blue': int(clean[8])})
        out.sort(key=lambda x: x['period'])
        return out if len(out) >= 100 else None
    except Exception:
        return None


def fetch_data(force_refresh=False):
    """兼容其它模块的 fetch 接口：返回并刷新历史数据。"""
    return load_history(force_refresh=force_refresh)


def _next_period(period, history=None):
    """推算下一期期号（5位格式 YYNNN，跨年自动进位）。

    v3.1 修复：原实现 `str(int(period)+1)` 有两个 bug：
    1. 前导零丢失——'03001'+1 得到 '3002' 而非 '03002'（2003-2009年所有期号受影响）；
    2. 跨年断点——每年最后一期（如 25151）实际下一期为 26001，而 +1 得到不存在的期号，
       导致预测记录永远无法与真实开奖匹配结算（蓝球"从未预测对"的直接原因之一）。
    全历史3491个相邻期验证：修复前错误995个，修复后应为0。

    跨年判断规则：
    - 当年不是数据最后一年 → 年份已完结，当年最大期号即最终期数，n 达到即跨年；
    - 当年是数据最后一年（在线场景，数据可能未抓全）→ 仅当已开期数 >= 140
      （接近年度末，近年每年 150-151 期）且 n 已达当年最大值时才跨年，
      否则视为数据不完整，正常 +1（如 26095 → 26096，而不是误判 27001）。
    """
    s = str(period)
    if not s.isdigit() or len(s) < 5:
        try:
            return str(int(period) + 1)
        except Exception:
            return ''
    y, n = int(s[:2]), int(s[2:])
    if history:
        same_year = [int(str(r.get('period', ''))[2:]) for r in history
                     if str(r.get('period', ''))[:2] == s[:2] and str(r.get('period', '')).isdigit()]
        max_n = max(same_year) if same_year else 0
        if max_n:
            years = sorted({int(str(r.get('period', ''))[:2]) for r in history
                            if str(r.get('period', '')).isdigit()})
            is_last_year = years and (y == years[-1])
            if not is_last_year:
                # 完整年份：当年最大期号即最终期数
                if n >= max_n:
                    return f'{(y + 1) % 100:02d}001'
                return f'{y:02d}{n + 1:03d}'
            # 数据最后一年：可能未完结，仅当接近年度末（已开>=140期）才跨年
            if n >= max_n and max_n >= 140:
                return f'{(y + 1) % 100:02d}001'
            return f'{y:02d}{n + 1:03d}'
    if n >= 153:  # 无历史兜底：双色球近年每年最多154期
        return f'{(y + 1) % 100:02d}001'
    return f'{y:02d}{n + 1:03d}'


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

    # 综合权重与完整排名（用于大底池输出；公式与 _predict_sets 保持一致）
    # 诊断结论（scripts/diagnose_ssq_v3.py）：近期系数 1.5 已是最优，加遗漏/去重叠/大底池内采样均无效或有害。
    red_w = [red_freq[x] + 1.5 * red_recent[x] + 0.5 for x in RED_RANGE]
    blue_w = [blue_freq[x] + 1.5 * blue_recent[x] + 0.3 for x in BLUE_RANGE]
    red_ranking = sorted(
        [{'number': x, 'weight': round(red_w[i], 3)} for i, x in enumerate(RED_RANGE)],
        key=lambda d: -d['weight'],
    )
    blue_ranking = sorted(
        [{'number': x, 'weight': round(blue_w[i], 3)} for i, x in enumerate(BLUE_RANGE)],
        key=lambda d: -d['weight'],
    )

    return {
        'red_freq': red_freq,
        'blue_freq': blue_freq,
        'red_recent': red_recent,
        'blue_recent': blue_recent,
        'hot_red': [n for n, _ in red_sorted[:10]],
        'cold_red': [n for n, _ in red_sorted[-10:]],
        'hot_blue': [n for n, _ in blue_sorted[:5]],
        'cold_blue': [n for n, _ in blue_sorted[-5:]],
        'red_ranking': red_ranking,
        'blue_ranking': blue_ranking,
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
    """生成 n 注推荐号码。seed 固定时（如最新期号），结果完全可复现。

    v3.0 蓝球策略调整:
    - 回测(600期)证明蓝球加权采样 28.5~31% < 均匀随机 32~33.5%，
      频率权重对蓝球是反预测信号（反直觉但统计显著），故蓝球改均匀随机。

    v3.1 蓝球去重覆盖（500期 walk-forward 实测）:
    - 原实现每注独立 rng.choice，约50%的期5注蓝球出现重复，
      联合命中仅 30.2%（独立随机理论 1-(15/16)^5≈27.6%）。
    - 改为 5 注蓝球互不重复（均匀随机覆盖5个不同号码），
      联合命中 = 5/16 = 31.25%（理论最优，纯组合收益，非预测能力）。

    v3.2 红球蛇形分组去重覆盖（2000期 walk-forward 实测）:
    - 原加权采样 5 注重叠严重（union 仅约20.8/33），联合命中 ge2=82.0%。
    - 改为排名池前30码蛇形分5组（每注含高低排名混合，互不重叠 union=30）：
      开奖6码进5个桶，鸽笼原理保证任1注≥2码概率逼近上限。
    - 实测 ge2 82.0%→96.2%、ge3 25.4%→28.7%、ge4 1.9%→2.4%、
      单注ge4 0.4%→0.5%（加权信号无损，纯组合收益）。
    - 覆盖注不强制合法性约束（和值/奇偶等，约45%不满足）；
      主推注（第1组）保留合法性检查，不合法时全池加权回退。
    """
    rng = random.Random(seed)
    prev_red = set(history[-1]['red']) if history else set()
    prev_blue = history[-1]['blue'] if history else None
    # 综合权重 = 整体频率 + 1.5 * 近期频率
    red_w = [analysis['red_freq'][x] + 1.5 * analysis['red_recent'][x] + 0.5 for x in RED_RANGE]

    # v3.2: 红球排名池前30码蛇形分组（互不重叠，union=30）
    ranking = sorted(RED_RANGE, key=lambda x: -red_w[x - 1])[:30]
    groups = [set() for _ in range(n)]
    for i, num in enumerate(ranking):
        row, col = i // n, i % n
        idx = col if row % 2 == 0 else (n - 1 - col)
        groups[idx].add(num)

    # 主推注（第1组）合法性检查；不合法或等于上期时：
    # 从排名池(30码)内加权重抽1组合法红球，剩余24码重新蛇形分给4个覆盖注，
    # 保证 5 注仍互不重叠（union=30 不被破坏）。
    # v3.2 修复说明：早期实现从全33码回退，导致覆盖注重叠（union 降到24）且
    # 引入池外号码，鸽笼覆盖失效；现改为池内重抽 + 剩余24码重分配。
    main = sorted(groups[0])
    if not _is_valid_red(main) or set(main) == prev_red:
        w_pool = [red_w[x - 1] for x in ranking]
        tries = 0
        while tries < 300:
            tries += 1
            red = sorted(_weighted_sample(ranking, w_pool, RED_COUNT, rng))
            if _is_valid_red(red) and set(red) != prev_red:
                main = red
                break
        groups[0] = set(main)
        rest = [x for x in ranking if x not in groups[0]]
        for g in groups[1:]:
            g.clear()
        for i, num in enumerate(rest):
            row, col = i // (n - 1), i % (n - 1)
            idx = col if row % 2 == 0 else (n - 2 - col)
            groups[1 + idx].add(num)

    # v3.1: 一次性抽好 n 个互不重复蓝球（排除上期蓝球不影响命中率：
    #       (15/16)*(5/15) = 5/16 = 31.25%，与不排除一致）
    avail_blues = [b for b in BLUE_RANGE if b != prev_blue]
    if len(avail_blues) < n:
        avail_blues = list(BLUE_RANGE)
    blues_pool = rng.sample(avail_blues, min(n, len(avail_blues)))
    while len(blues_pool) < n:
        blues_pool.append(rng.choice(BLUE_RANGE))

    return [{'red': sorted(groups[i]), 'blue': blues_pool[i]} for i in range(n)]


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

    # v2.0: 大底池选号参考区（确定性覆盖提升，不改变已最优的5注采样逻辑）
    # v3.0: 全量3489期回测修正数字——红球Top15大底ge4=27.2%（随机24.2%，微弱正向）；
    #       蓝球Top8大底命中49.5%（随机50%，频率无信号，单式蓝球已改均匀随机）。
    # 诊断证明：双色球统计信号极弱，限制单式采样范围反而降低命中，故大底池仅作参考。
    red_pool = [item['number'] for item in analysis['red_ranking'][:15]]
    blue_pool = [item['number'] for item in analysis['blue_ranking'][:8]]

    # v3.4: 蓝球全覆盖保底方案（16注）。
    # 16注蓝球互不相同 = 覆盖全部16个蓝球，任意开奖蓝球必被其中1注命中（鸽笼原理）。
    # 命中蓝球的那注至少六等奖(5元)；若其红球同时命中≥2则升至更高奖级。
    # 红球按5注推荐轮转复用（第i注红球 = 推荐第 (i%5)+1 注），无额外预测假设。
    # 诚实标注：32元成本换5元保底，长期数学期望为负，这是"保底体验"而非盈利策略。
    blue_full_cover_plan = {
        'name': '蓝球全覆盖保底（16注）',
        'notes': [
            {'red': sets[i % NUM_SETS]['red'], 'blue': b}
            for i, b in enumerate(BLUE_RANGE)
        ],
        'cost_yuan': 2 * len(BLUE_RANGE),
        'guarantee': '16注覆盖全部16个蓝球，任意开奖蓝球必被其中1注命中（100%，鸽笼原理）',
        'min_prize': '命中蓝球的注至少六等奖(5元)；该注红球≥2时奖级相应提升',
        'red_reuse_rule': f'红球按{NUM_SETS}注推荐轮转复用，不引入新预测假设',
        'disclaimer': '纯组合覆盖，非预测能力；32元成本换5元保底，长期期望为负，仅供娱乐',
    }

    # Only live predictions are persisted. Tests/callers supplying historical
    # data can evaluate the pure prediction function without changing storage.
    if data is None:
        try:
            settle_prediction_records(history)
            save_prediction_record(_next_period(latest['period'], history), sets)
        except Exception:
            # Recording must never prevent the prediction page from loading.
            pass
    prediction_records = load_prediction_records()

    recent_list = history[-recent:][::-1]  # 最近在前

    return {
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'latest_period': latest['period'],
        'next_period': _next_period(latest['period'], history),
        'history': recent_list,
        'prediction': {
            'primary': sets[0],
            'sets': sets,
        },
        # v3.4: 蓝球全覆盖保底方案（16注，鸽笼原理100%命中蓝球）
        'blue_full_cover_plan': blue_full_cover_plan,
        # v2.0: 大底池选号参考区 + 完整排名
        'red_pool': red_pool,
        'blue_pool': blue_pool,
        'red_ranking': analysis['red_ranking'],
        'blue_ranking': analysis['blue_ranking'],
        'prediction_records': list(reversed(prediction_records[-20:])),
        'online_stats': calculate_prediction_stats(prediction_records),
        'version': SSQ_PREDICTION_VERSION,
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
