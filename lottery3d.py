# 福彩3D预测器 V3.1+（标准库版，准确率优化）
# Python 3.10+
import math
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations, product

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

URL = "https://www.8300.cn/kjhhis/3/200.html"

RECENT_WINDOW = 60
EXP_DECAY = 0.96
BACKTEST_TRIALS = 80

W_HOT_GLOBAL = 4.0
W_HOT_POS = 5.0
W_MISS_HIGH = 9.0
W_MISS_MID = 4.5
W_MARKOV = 6.0
W_LAST_APPEAR = 2.5
W_NEIGHBOR = 2.0
W_ROAD_MATCH = 1.5
W_DANMA_HIT = 4.0
SUM_SOFT_SIGMA = 3.2
SPAN_SOFT_SIGMA = 1.4

# 推荐注数（直选为带顺序的三位数）
RECOMMEND_GROUPS = 15
ZHIXUAN_TOP3 = 3
ZU6_POOL_SIZE = 5
ZU6_FOUR_SIZE = 4


def fetch_data(url=URL):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    compact = re.sub(r"\s+", " ", html)
    pattern = re.compile(
        r'<td>(\d{7})期</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>'
        r'\s*<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>'
    )
    rows = pattern.findall(compact)
    data = [(pid, dt, (int(a), int(b), int(c))) for pid, dt, a, b, c in rows]
    data.reverse()
    return data


def calc_span(n):
    return max(n) - min(n)


def miss_value(numbers, digit, position=None):
    for i in range(len(numbers) - 1, -1, -1):
        n = numbers[i]
        if position is None:
            if digit in n:
                return len(numbers) - 1 - i
        elif n[position] == digit:
            return len(numbers) - 1 - i
    return len(numbers)


def neighbor(d):
    return {(d - 1) % 10, (d + 1) % 10}


def road(d):
    return d % 3


def exp_weighted_counts(series, decay=EXP_DECAY):
    cnt = Counter()
    w = 1.0
    for item in reversed(series):
        cnt[item] += w
        w *= decay
    return cnt


def build_markov(numbers, position):
    trans = defaultdict(Counter)
    for i in range(len(numbers) - 1):
        a, b = numbers[i][position], numbers[i + 1][position]
        trans[a][b] += 1
    return trans


def gaussian_score(value, center, sigma):
    if sigma <= 0:
        return 0.0
    z = (value - center) / sigma
    return math.exp(-0.5 * z * z)


def analyze_sum_span(sums, spans):
    recent_s = sums[-RECENT_WINDOW:]
    recent_p = spans[-RECENT_WINDOW:]
    w_s = exp_weighted_counts(recent_s)
    w_p = exp_weighted_counts(recent_p)

    sum_center = sum(k * v for k, v in w_s.items()) / max(sum(w_s.values()), 1e-9)
    span_center = sum(k * v for k, v in w_p.items()) / max(sum(w_p.values()), 1e-9)

    return {
        "sum_center": sum_center,
        "span_center": span_center,
        "hot_sums": [x for x, _ in w_s.most_common(6)],
        "hot_spans": [x for x, _ in w_p.most_common(4)],
        "sum_tail_freq": Counter(s % 10 for s in recent_s),
    }


def digit_scores(numbers):
    recent = numbers[-RECENT_WINDOW:]
    last = numbers[-1]
    score = [0.0] * 10

    freq_all = exp_weighted_counts([d for n in recent for d in n])
    for d, _ in freq_all.most_common(4):
        score[d] += W_HOT_GLOBAL

    for pos in range(3):
        pos_freq = exp_weighted_counts([n[pos] for n in recent])
        for d, _ in pos_freq.most_common(3):
            score[d] += W_HOT_POS

        trans = build_markov(numbers, pos)
        prev_d = last[pos]
        row = trans.get(prev_d, Counter())
        total = sum(row.values()) or 1
        for d, c in row.items():
            score[d] += W_MARKOV * (c / total)

    for d in range(10):
        mv = miss_value(numbers, d)
        if mv >= 20:
            score[d] += W_MISS_HIGH
        elif mv >= 12:
            score[d] += W_MISS_MID

    for d in set(last):
        score[d] += W_LAST_APPEAR

    nb = set()
    for d in last:
        nb.update(neighbor(d))
    for d in nb:
        score[d] += W_NEIGHBOR

    last_roads = {road(d) for d in last}
    for d in range(10):
        if road(d) in last_roads:
            score[d] += W_ROAD_MATCH

    return score, freq_all


def position_digit_scores(numbers, position):
    """单码分位评分（百/十/个）"""
    recent = [n[position] for n in numbers[-RECENT_WINDOW:]]
    last_d = numbers[-1][position]
    sc = [0.0] * 10
    for d, _ in exp_weighted_counts(recent).most_common(4):
        sc[d] += W_HOT_POS + 1
    trans = build_markov(numbers, position)
    row = trans.get(last_d, Counter())
    total = sum(row.values()) or 1
    for d, c in row.items():
        sc[d] += W_MARKOV * (c / total)
    mv = miss_value(numbers, None, position=position)
    for d in range(10):
        miss_p = miss_value(numbers, d, position=position)
        if miss_p >= 16:
            sc[d] += W_MISS_HIGH
        elif miss_p >= 9:
            sc[d] += W_MISS_MID
    sc[last_d] += W_LAST_APPEAR
    for d in neighbor(last_d):
        sc[d] += W_NEIGHBOR
    return sc


def pick_dan_tuo_kill(score):
    rank = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    danma = [rank[0][0], rank[1][0]]
    tuoma = [x[0] for x in rank[2:6]]
    kill = [rank[-1][0]] if rank[-1][1] + 3 < rank[-2][1] else [x[0] for x in rank[-2:]]
    return danma, tuoma, kill, rank


def pick_zu6_four(score, kill=None):
    """组六四码：选评分最高的 4 个不同数字（优先避开杀码）"""
    return pick_zu6_pool(score, kill, pool_size=ZU6_FOUR_SIZE)


def zu6_notes_from_digits(digits):
    """N 码组六 → C(N,3) 注组六组合"""
    combos = [tuple(sorted(c)) for c in combinations(digits, 3)]
    return combos, ["".join(map(str, c)) for c in combos]


def pick_zu6_pool(score, kill=None, pool_size=ZU6_POOL_SIZE):
    """组六复式选号：默认五码 → 10 注组六"""
    kill = set(kill or [])
    rank = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    picked = []
    for d, _ in rank:
        if d in kill:
            continue
        picked.append(d)
        if len(picked) == pool_size:
            break
    if len(picked) < pool_size:
        for d, _ in rank:
            if d not in picked:
                picked.append(d)
            if len(picked) == pool_size:
                break
    return sorted(picked)


def rank_zu6_groups(score, digits, danma, kill, meta, top_n=RECOMMEND_GROUPS):
    """从复式号码中按评分排出 top_n 注组六（五码时恰好 10 注）"""
    ranked = []
    for combo in combinations(digits, 3):
        a, b, c = combo
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        if w < 0:
            continue
        ranked.append((w, "".join(map(str, (a, b, c)))))
    ranked.sort(key=lambda x: -x[0])
    return ranked[:top_n]


def is_zu6_draw(triple):
    """开奖号为组六（三码各不相同）"""
    return len(set(triple)) == 3


def classify_form(triple):
    """形态：组六 / 组三 / 豹子"""
    n = len(set(triple))
    if n == 3:
        return "zu6"
    if n == 2:
        return "zu3"
    return "baozi"


FORM_LABELS = {"zu6": "组六", "zu3": "组三", "baozi": "豹子"}
THEORY_FORM_P = {"zu6": 0.72, "zu3": 0.27, "baozi": 0.01}


def form_miss(forms, target):
    """距上次出现 target 形态的期数"""
    for i in range(len(forms) - 1, -1, -1):
        if forms[i] == target:
            return len(forms) - 1 - i
    return len(forms)


def analyze_form_probability(numbers):
    """估算本期开出组六/组三/豹子的概率（多源融合）"""
    forms = [classify_form(n) for n in numbers]
    last_form = forms[-1]

    recent = forms[-RECENT_WINDOW:]
    w_cnt = exp_weighted_counts(recent)
    w_total = sum(w_cnt.values()) or 1.0
    recent_p = {k: w_cnt.get(k, 0) / w_total for k in THEORY_FORM_P}

    hist_cnt = Counter(forms)
    hist_total = len(forms)
    hist_p = {k: hist_cnt.get(k, 0) / hist_total for k in THEORY_FORM_P}

    trans = defaultdict(Counter)
    for i in range(len(forms) - 1):
        trans[forms[i]][forms[i + 1]] += 1
    row = trans.get(last_form, Counter())
    row_total = sum(row.values()) or 1.0
    markov_p = {k: row.get(k, 0) / row_total for k in THEORY_FORM_P}

    blend = {}
    for k in THEORY_FORM_P:
        blend[k] = (
            0.40 * recent_p[k]
            + 0.35 * markov_p[k]
            + 0.15 * hist_p[k]
            + 0.10 * THEORY_FORM_P[k]
        )
    total = sum(blend.values()) or 1.0
    blend = {k: v / total for k, v in blend.items()}

    streak = 1
    for i in range(len(forms) - 2, -1, -1):
        if forms[i] == last_form:
            streak += 1
        else:
            break

    return {
        "last_form": last_form,
        "streak": streak,
        "miss_zu6": form_miss(forms, "zu6"),
        "miss_zu3": form_miss(forms, "zu3"),
        "recent_p": recent_p,
        "hist_p": hist_p,
        "markov_p": markov_p,
        "blend_p": blend,
        "markov_samples": row_total,
    }


def triplet_weight(a, b, c, score, danma, kill, meta):
    if a in kill or b in kill or c in kill:
        return -1

    w = score[a] + score[b] + score[c]
    for x in (a, b, c):
        if x in danma:
            w += W_DANMA_HIT

    s = a + b + c
    w += 8.0 * gaussian_score(s, meta["sum_center"], SUM_SOFT_SIGMA)

    span = max(a, b, c) - min(a, b, c)
    w += 5.0 * gaussian_score(span, meta["span_center"], SPAN_SOFT_SIGMA)

    if s in meta["hot_sum_set"]:
        w += 2.0
    if span in meta["hot_span_set"]:
        w += 1.5
    if (s % 10) in meta["sum_tail_top"]:
        w += 1.0

    return w


def rank_triplets(score, danma, kill, meta, top_n=20):
    pool = []
    for a, b, c in product(range(10), repeat=3):
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        if w < 0:
            continue
        pool.append((w, f"{a}{b}{c}"))
    pool.sort(key=lambda x: -x[0])
    return pool[:top_n]


def backtest(numbers, trials=BACKTEST_TRIALS):
    if len(numbers) < trials + RECENT_WINDOW + 5:
        trials = max(20, len(numbers) - RECENT_WINDOW - 5)

    hit_top = hit_top3 = hit_ge2 = hit_sum_band = hit_zu6_pool = hit_zu6_four = zu6_draws = 0
    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        sums = [sum(x) for x in train]
        spans = [calc_span(x) for x in train]
        meta_raw = analyze_sum_span(sums, spans)
        meta = {
            **meta_raw,
            "hot_sum_set": set(meta_raw["hot_sums"]),
            "hot_span_set": set(meta_raw["hot_spans"]),
            "sum_tail_top": {t for t, _ in meta_raw["sum_tail_freq"].most_common(4)},
        }
        sc, _ = digit_scores(train)
        dan, _, kill, _ = pick_dan_tuo_kill(sc)
        top = rank_triplets(sc, dan, kill, meta, top_n=RECOMMEND_GROUPS)
        top_nums = [t[1] for t in top]
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"

        if act_s in top_nums:
            hit_top += 1
        if act_s in top_nums[:ZHIXUAN_TOP3]:
            hit_top3 += 1
        pred_digits = {int(ch) for s in top_nums for ch in s}
        if len(pred_digits & set(actual)) >= 2:
            hit_ge2 += 1
        if abs(sum(actual) - meta["sum_center"]) <= 4:
            hit_sum_band += 1

        z6 = pick_zu6_pool(sc, kill)
        z4 = pick_zu6_four(sc, kill)
        if is_zu6_draw(actual):
            zu6_draws += 1
            if set(actual).issubset(set(z6)):
                hit_zu6_pool += 1
            if set(actual).issubset(set(z4)):
                hit_zu6_four += 1

    n = trials
    return {
        "trials": n,
        "top_hit": hit_top,
        "top_rate": hit_top / n,
        "top3_hit": hit_top3,
        "top3_rate": hit_top3 / n,
        "recommend_groups": RECOMMEND_GROUPS,
        "ge2_digit_rate": hit_ge2 / n,
        "sum_band_rate": hit_sum_band / n,
        "zu6_draws": zu6_draws,
        "zu6_pool_hit": hit_zu6_pool,
        "zu6_pool_rate": hit_zu6_pool / zu6_draws if zu6_draws else 0.0,
        "zu6_four_hit": hit_zu6_four,
        "zu6_four_rate": hit_zu6_four / zu6_draws if zu6_draws else 0.0,
    }


def run_prediction(data=None):
    """运行预测，返回 JSON 可序列化 dict；data 为 None 时自动抓取。"""
    if data is None:
        data = fetch_data()
    if not data:
        return {"error": "未获取到数据"}

    periods = [x[0] for x in data]
    numbers = [x[2] for x in data]
    sums = [sum(x) for x in numbers]
    spans = [calc_span(x) for x in numbers]

    meta_raw = analyze_sum_span(sums, spans)
    meta = {
        **meta_raw,
        "hot_sum_set": set(meta_raw["hot_sums"]),
        "hot_span_set": set(meta_raw["hot_spans"]),
        "sum_tail_top": {t for t, _ in meta_raw["sum_tail_freq"].most_common(5)},
    }

    score, freq_all = digit_scores(numbers)
    danma, tuoma, kill, rank = pick_dan_tuo_kill(score)
    form_prob = analyze_form_probability(numbers)
    zu6_four = pick_zu6_four(score, kill)
    _, z6_straight = zu6_notes_from_digits(zu6_four)
    zhixuan_top = rank_triplets(score, danma, kill, meta, top_n=RECOMMEND_GROUPS)
    bt = backtest(numbers)

    last_num = numbers[-1]
    pos_names = ("百", "十", "个")
    position_top = []
    for pos, name in enumerate(pos_names):
        pr = sorted(enumerate(position_digit_scores(numbers, pos)), key=lambda x: -x[1])[:5]
        position_top.append({
            "name": name,
            "digits": [{"digit": d, "score": round(s, 1)} for d, s in pr],
        })

    miss_global = []
    for d in range(10):
        mv = miss_value(numbers, d)
        if mv >= 8:
            miss_global.append({"digit": d, "miss": mv})
    miss_global.sort(key=lambda x: -x["miss"])

    miss_position = []
    for pos, name in enumerate(pos_names):
        top = sorted(range(10), key=lambda x: -miss_value(numbers, x, position=pos))[:3]
        miss_position.append({
            "name": name,
            "digits": [{"digit": d, "miss": miss_value(numbers, d, position=pos)} for d in top],
        })

    sum_tails = [{"tail": t, "count": c} for t, c in meta_raw["sum_tail_freq"].most_common(5)]

    return {
        "period": periods[-1],
        "total_periods": len(numbers),
        "avg_sum": round(sum(sums) / len(sums), 2),
        "last_draw": "".join(map(str, last_num)),
        "neighbors": sorted(set().union(*[neighbor(d) for d in last_num])),
        "hot_digits": [{"digit": d, "weight": round(c, 1)} for d, c in freq_all.most_common(5)],
        "danma": danma,
        "tuoma": tuoma,
        "kill": kill,
        "rank_top10": [{"digit": d, "score": round(s, 1)} for d, s in rank[:10]],
        "position_top": position_top,
        "miss_global": miss_global,
        "miss_position": miss_position,
        "sum_tails": sum_tails,
        "recommend_groups": RECOMMEND_GROUPS,
        "recent_window": RECENT_WINDOW,
        "sum_span": {
            "sum_center": round(meta["sum_center"], 1),
            "hot_sums": meta["hot_sums"],
            "span_center": round(meta["span_center"], 1),
            "hot_spans": meta["hot_spans"],
        },
        "form": {
            "last_label": FORM_LABELS[form_prob["last_form"]],
            "streak": form_prob["streak"],
            "miss_zu6": form_prob["miss_zu6"],
            "miss_zu3": form_prob["miss_zu3"],
            "recent": {k: round(v, 4) for k, v in form_prob["recent_p"].items()},
            "hist": {k: round(v, 4) for k, v in form_prob["hist_p"].items()},
            "markov": {k: round(v, 4) for k, v in form_prob["markov_p"].items()},
            "blend": {k: round(v, 4) for k, v in form_prob["blend_p"].items()},
            "theory": THEORY_FORM_P,
            "markov_samples": int(form_prob["markov_samples"]),
        },
        "zu6_four": {
            "digits_str": "".join(map(str, zu6_four)),
            "combos": z6_straight,
        },
        "zhixuan_top3": [{"num": num, "score": round(w, 1)} for w, num in zhixuan_top[:ZHIXUAN_TOP3]],
        "zhixuan": [{"num": num, "score": round(w, 1)} for w, num in zhixuan_top],
        "backtest": bt,
    }


def print_report(result):
    """终端格式化输出"""
    if result.get("error"):
        print(result["error"])
        return

    form = result["form"]
    lf = form["last_label"]
    z6 = result["zu6_four"]

    print("\n" + "=" * 70)
    print("【本期摘要】")
    print("=" * 70)
    print(f"  上期 {result['period']} 期: {result['last_draw']}  ({lf}，连出 {form['streak']} 期)")
    print(f"  形态预估 → 组六 {form['blend']['zu6']*100:.1f}%  |  组三 {form['blend']['zu3']*100:.1f}%  |  豹子 {form['blend']['baozi']*100:.1f}%")
    print(f"  组六四码 → {z6['digits_str']}  (覆盖: {', '.join(z6['combos'])})")
    if result["zhixuan_top3"]:
        top3 = ", ".join(x["num"] for x in result["zhixuan_top3"])
        print(f"  直选Top3 → {top3}")

    print("\n" + "=" * 70)
    print(f"热号分析（指数加权近{RECENT_WINDOW}期）")
    print("=" * 70)
    for item in result["hot_digits"]:
        print(f"  热号 {item['digit']} -> 加权{item['weight']:.1f}")

    print("\n遗漏分析（分位+全局）")
    for item in result.get("miss_global", []):
        print(f"  数字{item['digit']} 全局遗漏{item['miss']}期")
    for block in result.get("miss_position", []):
        for item in block["digits"]:
            print(f"  {block['name']}位 数字{item['digit']} 遗漏{item['miss']}期")

    print("\n上期号码:", result["last_draw"])
    print("邻号:", result["neighbors"])

    print("\n" + "=" * 70)
    print("【本期形态概率】（组六 / 组三 / 豹子）")
    print("=" * 70)
    print(f"  上期形态: {lf}（已连续 {form['streak']} 期）")
    print(f"  组六遗漏: {form['miss_zu6']} 期  |  组三遗漏: {form['miss_zu3']} 期")
    print(f"  近{RECENT_WINDOW}期: 组六 {form['recent']['zu6']*100:.1f}%  "
          f"组三 {form['recent']['zu3']*100:.1f}%  "
          f"豹子 {form['recent']['baozi']*100:.1f}%")
    print(
        f"  上期{lf}→下期(样本{form['markov_samples']}): "
        f"组六 {form['markov']['zu6']*100:.1f}%  "
        f"组三 {form['markov']['zu3']*100:.1f}%  "
        f"豹子 {form['markov']['baozi']*100:.1f}%"
    )
    print("  综合预估(近态+转移+历史+理论):")
    print(f"    ★ 组六 {form['blend']['zu6']*100:.1f}%  "
          f"★ 组三 {form['blend']['zu3']*100:.1f}%  "
          f"  豹子 {form['blend']['baozi']*100:.1f}%")
    print(f"  理论基准: 组六 {form['theory']['zu6']*100:.0f}%  "
          f"组三 {form['theory']['zu3']*100:.0f}%  "
          f"豹子 {form['theory']['baozi']*100:.0f}%")

    ss = result["sum_span"]
    print("\n和值/跨度（软约束中心）")
    print(f"  和值中心 {ss['sum_center']}，推荐区间 {ss['hot_sums']}")
    print(f"  跨度中心 {ss['span_center']}，推荐 {ss['hot_spans']}")
    if result.get("sum_tails"):
        print("  和值尾TOP5:", [(x["tail"], x["count"]) for x in result["sum_tails"]])

    print("\n综合评分 TOP10")
    for item in result["rank_top10"]:
        print(f"  {item['digit']}: {item['score']:.1f}分")

    print("\n分位推荐（各位 Top5）")
    for block in result["position_top"]:
        print(f"  {block['name']}位:", [f"{x['digit']}({x['score']:.0f})" for x in block["digits"]])

    print("\n" + "=" * 70)
    print("【组六四码推荐】（选 4 个号打组六复式即可）")
    print("=" * 70)
    print("  投注号码:", z6["digits_str"])
    print("  杀码参考:", result["kill"], "（四码中已尽量避开）")
    print("  覆盖 4 注组六:", ", ".join(z6["combos"]))

    print("\n" + "=" * 70)
    print("【直选Top3推荐】（百十个位顺序一致）")
    print("=" * 70)
    for idx, item in enumerate(result.get("zhixuan_top3", []), start=1):
        print(f"  {idx}. {item['num']}  评分={item['score']:.1f}")

    print("\n" + "=" * 70)
    print(f"【直选推荐 {RECOMMEND_GROUPS} 注】（百十个位顺序一致）")
    print("=" * 70)
    print("  杀码参考:", result["kill"], "（含杀码组合已排除）")
    print("-" * 70)
    for idx, item in enumerate(result["zhixuan"], start=1):
        print(f"  {idx:02d}. {item['num']}  评分={item['score']:.1f}")

    bt = result["backtest"]
    print("\n" + "=" * 70)
    print("滚动回测（仅供参考）")
    print("=" * 70)
    print(f"  回测期数: {bt['trials']}")
    print(f"  直选{RECOMMEND_GROUPS}注命中: {bt['top_rate']*100:.1f}%  ({bt['top_hit']}/{bt['trials']})")
    print(f"  直选Top3命中: {bt['top3_rate']*100:.1f}%  ({bt['top3_hit']}/{bt['trials']})")
    print(f"  推荐池至少中2个数字: {bt['ge2_digit_rate']*100:.1f}%")
    print(f"  和值落在预测带±4: {bt['sum_band_rate']*100:.1f}%")
    if bt["zu6_draws"]:
        print(
            f"  组六四码命中(开奖为组六时): {bt['zu6_four_rate']*100:.1f}%  "
            f"({bt['zu6_four_hit']}/{bt['zu6_draws']})"
        )
        print(
            f"  (参考)五码组六池含开奖三码: {bt['zu6_pool_rate']*100:.1f}%  "
            f"({bt['zu6_pool_hit']}/{bt['zu6_draws']})"
        )

    print("\n统计信息")
    print("  总期数:", result["total_periods"])
    print("  最近一期:", result["period"])
    print("  平均和值:", result["avg_sum"])
    print("\n  说明: 3D 开奖具有随机性，回测用于观察候选池收缩效果，不构成投注建议。")


def main():
    print("抓取数据中...")
    print_report(run_prediction())


if __name__ == "__main__":
    main()
