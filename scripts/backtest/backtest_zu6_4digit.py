#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
福彩3D 组六4码 专项回测与优化
=============================
目标：量化并提升「从10个数字选4个不同号码」覆盖真实开奖3个数字的命中率。

核心指标：
  - zu6_4hit: 组六4码命中（开奖3个不同数字全部落在选的4码内）
  - zu6_conditional: 仅组六开奖期的条件命中率（理论随机=3.33%）
  - ge2_cover: 至少覆盖2个开奖数字的比例

用法：
  python backtest_zu6_4digit.py              # 完整回测
  python backtest_zu6_4digit.py --fast       # 快速模式（200期采样）
"""

import json, sys, os, time, math
from collections import Counter
from itertools import combinations as C
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.lottery3d as L

RAW_PATH = os.path.join("data", "_raw_3d_2000.json")
RESULT_PATH = os.path.join("data", "bt_zu6_4digit.json")

# 理论基线（组六4码：从10个数字选4个不同号码）
RANDOM_ZU6_COND = 4.0 / math.comb(10, 3)   # ≈ 3.33%（组六期条件命中率）
RANDOM_ALL_RATE = RANDOM_ZU6_COND * 0.72     # ≈ 2.40%（全期期望）


def load_data():
    rows = json.load(open(RAW_PATH, "r", encoding="utf-8"))
    numbers = [tuple(int(d) for d in r[2:5]) for r in rows]
    periods = [r[0] for r in rows]
    return numbers, periods


def zu6_4hit(selected, draw):
    """组六4码是否全中"""
    return set(draw) <= set(selected)


def ge2_cover(selected, draw):
    """至少覆盖2个开奖数字"""
    return len(set(draw) & set(selected)) >= 2


def make_pipeline(idx, numbers, sums_list, spans_list):
    """复刻预测管线，返回 (score, pair_freq) 或 None"""
    train = numbers[:idx]
    if len(train) < 50:
        return None
    try:
        ww, _ = L.resolve_window_weights(train, compute_weights=False, period=idx - 1)
        meta = L.build_ranking_meta(train, ww, sums_list[:idx], spans_list[:idx], tail_top=5)
        score, _ = L.ensemble_digit_scores(train, ww, dynamic=meta.get("dynamic"))
        pf = L.zu6_cooccurrence_freq(train)
        return score, pf, train
    except Exception:
        return None


# ============================================================
# 策略1：当前生产算法 (pick_zu6_pool pool_size=4)
# ============================================================
def run_current(numbers, test_indices, sums_list, spans_list):
    hits = z_hits = z_tot = g2 = 0; t0 = time.time()
    for idx in test_indices:
        p = make_pipeline(idx, numbers, sums_list, spans_list)
        if not p:
            continue
        score, pf, train = p
        sel = list(L.pick_zu6_pool(score, None, pool_size=4, use_kill=False,
                                    numbers=train, pair_freq=pf))
        d = numbers[idx]
        if zu6_4hit(sel, d):
            hits += 1
        if ge2_cover(sel, d):
            g2 += 1
        if L.classify_form(d) == "zu6":
            z_tot += 1
            if zu6_4hit(sel, d):
                z_hits += 1
    n = len(test_indices)
    return _result("当前算法(pick_zu6_pool)", hits, z_hits, z_tot, g2, n, time.time()-t0)


# ============================================================
# 策略2：单码Top4 baseline
# ============================================================
def run_top4(numbers, test_indices, sums_list, spans_list):
    hits = z_hits = z_tot = g2 = 0; t0 = time.time()
    for idx in test_indices:
        p = make_pipeline(idx, numbers, sums_list, spans_list)
        if not p:
            continue
        score, _, _ = p
        sel = sorted(range(10), key=lambda d: -score[d])[:4]
        d = numbers[idx]
        if zu6_4hit(sel, d): hits += 1
        if ge2_cover(sel, d): g2 += 1
        if L.classify_form(d) == "zu6":
            z_tot += 1
            if zu6_4hit(sel, d): z_hits += 1
    n = len(test_indices)
    return _result("单码Top4(baseline)", hits, z_hits, z_tot, g2, n, time.time()-t0)


# ============================================================
# 策略3：近N期高频 Top4
# ============================================================
def run_freq_top4(numbers, test_indices, window=30, decay=False):
    hits = z_hits = z_tot = g2 = 0; t0 = time.time()
    for idx in test_indices:
        train = numbers[:idx]
        if len(train) < 30:
            continue
        recent = train[-window:]
        if decay:
            freq = Counter()
            for i, n in enumerate(recent):
                w = math.exp(-0.03 * (len(recent) - 1 - i))
                for d in n:
                    freq[d] += w
        else:
            freq = Counter(d for n in recent for d in n)
        sel = [d for d, _ in freq.most_common(4)]
        d = numbers[idx]
        if zu6_4hit(sel, d): hits += 1
        if ge2_cover(sel, d): g2 += 1
        if L.classify_form(d) == "zu6":
            z_tot += 1
            if zu6_4hit(sel, d): z_hits += 1
    label = f"高频Top4({window}期{'+衰减' if decay else ''})"
    n = len(test_indices)
    return _result(label, hits, z_hits, z_tot, g2, n, time.time()-t0)


# ============================================================
# 策略4：融合选4码（多信号组合优化）
# ============================================================
def run_fusion(numbers, test_indices, sums_list, spans_list,
               w_sc=1.0, w_rec=1.0, w_miss=0.5, w_pair=0.8):
    hits = z_hits = z_tot = g2 = 0; t0 = time.time()
    for idx in test_indices:
        p = make_pipeline(idx, numbers, sums_list, spans_list)
        if not p:
            continue
        score, pf, train = p

        # 多维特征
        base = dict(enumerate(score))

        # 近期实开锚定
        rb = [0.0]*10
        for n in train[-5:]:
            for d in set(n):
                rb[d] += 1.0/len(set(n))
        mx_rb = max(rb) or 1.0; rb_n = [r/mx_rb for r in rb]

        # 遗漏回补
        lb = train[-60:] if len(train)>=60 else train[:]
        ls_map = {d:len(lb) for d in range(10)}
        for i_rev, n in enumerate(reversed(lb)):
            for d in set(n):
                if ls_map[d] == len(lb):
                    ls_map[d] = i_rev
        mx_ls = max(ls_map.values()) or 1.0
        ms = [(ls_map[d]/mx_ls)**1.5 for d in range(10)]

        # 共现
        co = [0.0]*10
        for (a,b),v in pf.items():
            co[a]+=v; co[b]+=v
        mx_co = max(co) or 1.0; co_n = [c/mx_co for c in co]

        # 综合分
        comb = [(w_sc*base.get(d,0) + w_rec*rb_n[d] + w_miss*ms[d] + w_pair*co_n[d], d)
                for d in range(10)]
        comb.sort(reverse=True)
        cands = [d for _,d in comb[:7]]

        # 上期实开保底进候选
        if train:
            for d in set(train[-1]):
                if d not in cands:
                    cands.append(d)

        def crk(combo):
            ds = sorted(combo)
            s = sum(comb[d][0] for d in ds)
            ps = sum(pf.get((ds[i],ds[j]),0) for i in range(len(ds)) for j in range(i+1,len(ds)))
            sp = ds[-1]-ds[0]
            oc = sum(1 for d in ds if d%2)
            bc = sum(1 for d in ds if d>=5)
            adj = sum(1 for a,b in zip(ds,ds[1:]) if b-a==1)
            return s + ps*2.5 + sp*0.3 -abs(oc-2)*0.4 -abs(bc-2)*0.32 + adj*0.25

        best = max(C(cands,4), key=crk)
        sel = sorted(best)

        d = numbers[idx]
        if zu6_4hit(sel,d): hits+=1
        if ge2_cover(sel,d): g2+=1
        if L.classify_form(d)=="zu6":
            z_tot+=1
            if zu6_4hit(sel,d): z_hits+=1
    tag = f"融合(s={w_sc},r={w_rec},m={w_miss})"
    n = len(test_indices)
    return _result(tag, hits, z_hits, z_tot, g2, n, time.time()-t0)


# ============================================================
# 策略5：多组推荐（任一中）
# ============================================================
def run_multi(numbers, test_indices, sums_list, spans_list, n_groups=3):
    hits = z_hits = z_tot = g2 = 0; t0 = time.time()
    for idx in test_indices:
        p = make_pipeline(idx, numbers, sums_list, spans_list)
        if not p:
            continue
        score, pf, train = p
        groups=[]; seen=set()

        g1=tuple(L.pick_zu6_pool(score,None,pool_size=4,use_kill=False,numbers=train,pair_freq=pf))
        if len(g1)==4: groups.append(g1); seen.add(g1)

        freq=Counter(d for n in train[-20:] for d in n)
        g2=tuple(sorted(d for d,_ in freq.most_common(4)))
        if g2 not in seen and len(g2)==4: groups.append(g2); seen.add(g2)

        lb=train[-50:] if len(train)>=50 else train[:]
        lsm={d:len(lb) for d in range(10)}
        for i_rev,n in enumerate(reversed(lb)):
            for d in set(n):
                if lsm[d]==len(lb): lsm[d]=i_rev
        g3=tuple(sorted(sorted(range(10),key=lambda d:-lsm[d])[:4]))
        if g3 not in seen and len(g3)==4: groups.append(g3); seen.add(g3)

        rk=sorted(range(10),key=lambda d:-L._effective_digit_score(score,d))
        for combo in C(rk[:7],4):
            if len(groups)>=n_groups: break
            k=tuple(sorted(combo))
            if k not in seen: groups.append(k); seen.add(k)

        d=numbers[idx]
        ah=any(zu6_4hit(list(g),d) for g in groups[:n_groups])
        ag2=any(ge2_cover(list(g),d) for g in groups[:n_groups])
        if ah: hits+=1
        if ag2: g2+=1
        if L.classify_form(d)=="zu6":
            z_tot+=1
            if ah: z_hits+=1
    n=len(test_indices)
    return _result(f"多组({n_groups}组)",hits,z_hits,z_tot,g2,n,time.time()-t0)


# ============================================================
# 辅助
# ============================================================
def _result(name, hits, z_hits, z_tot, g2, n, elapsed):
    hr=hits/n if n else 0
    zr=z_hits/z_tot if z_tot else 0
    gr=g2/n if n else 0
    return {
        "name": name, "hits": hits, "total": n,
        "hit_rate": round(hr,4), "zu6_hits": z_hits, "zu6_total": z_tot,
        "zu6_rate": round(zr,4), "ge2_rate": round(gr,4),
        "elapsed_s": round(elapsed,1),
        "lift": round(hr/RANDOM_ALL_RATE-1,2),
        "zu6_lift": round(zr/RANDOM_ZU6_COND-1,2),
    }


def main():
    fast = "--fast" in sys.argv
    print("=" * 65)
    print("  福彩3D 组六4码 专项回测")
    print("=" * 65)

    numbers, periods = load_data()
    N = len(numbers)
    print(f"\n数据: {N}期 ({periods[0]}~{periods[-1]}) 最新={numbers[-1]}")

    WARMUP = 200
    STEP = 4 if fast else 2
    indices = list(range(WARMUP, N, STEP))
    print(f"预热:{WARMUP} 步长:{STEP} 测试:{len(indices)}期")

    sums_list=[a+b+c for a,b,c in numbers]
    spans_list=[max(n)-min(n) for n in numbers]

    print(f"\n理论基线: 随机4码 组六期={RANDOM_ZU6_COND:.4f}({RANDOM_ZU6_COND*100:.2f}%) "
          f"| 全期≈{RANDOM_ALL_RATE:.4f}({RANDOM_ALL_RATE*100:.2f}%)")

    results = []

    # --- 策略1 ---
    print("\n--- [1] 当前算法 ---")
    r1 = run_current(numbers, indices, sums_list, spans_list)
    results.append(r1); _pr(r1)

    # --- 策略2 ---
    print("\n--- [2] 单码Top4 ---")
    r2 = run_top4(numbers, indices, sums_list, spans_list)
    results.append(r2); _pr(r2)

    # --- 策略3: 高频变体 ---
    for win in [20,30,50]:
        print(f"\n--- 高频Top4({win}) ---")
        rf = run_freq_top4(numbers, indices, win, decay=False)
        results.append(rf); _pr(rf)

    print(f"\n--- 高频Top4(30+衰减) ---")
    rd = run_freq_top4(numbers, indices, 30, decay=True)
    results.append(rd); _pr(rd)

    # --- 策略4: 融合扫描 ---
    print("\n--- 融合选4码 参数扫描 ---")
    best_zr = 0; best_rf = None
    for ws in [1.0,1.3]:
        for wr in [0.8,1.0,1.3]:
            for wm in [0.3,0.6,0.8]:
                rf = run_fusion(numbers,indices,sums_list,spans_list,w_sc=ws,w_rec=wr,w_miss=wm)
                results.append(rf)
                tag = " ★BEST" if rf["zu6_rate"]>best_zr else ""
                if rf["zu6_rate"]>best_zr:
                    best_zr=rf["zu6_rate"]; best_rf=rf
                print(f"  s={ws} r={wr} m={wm}: Zu6={rf['zu6_rate']:.4f} "
                      f"全命中率={rf['hit_rate']:.4f} Lift={rf['zu6_lift']:+.0f}%{tag}")
    if best_rf:
        results.append({**best_rf, "name": "★ 融合最佳"})

    # --- 汇总 ---
    print("\n" + "=" * 65)
    print("  排名（按组六期命中率降序）")
    print("-"*65)
    ranked = sorted(results, key=lambda x:(-x["zu6_rate"], -x["hit_rate"]))
    for i,r in enumerate(ranked):
        m = " ★" if "最佳" in r.get("name","") or "★" in r.get("name","") else ""
        print(f"  {i+1:2d}. {r['name']:32s} Zu6={r['zu6_rate']:.4f} "
              f"全命中={r['hit_rate']:.4f} Lift={r['zu6_lift']:+.0f}%{m}")

    out = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "range": f"{periods[0]}~{periods[-1]}", "warmup": WARMUP,
        "step": STEP, "test_n": len(indices),
        "random_zu6": round(RANDOM_ZU6_COND,4),
        "random_all": round(RANDOM_ALL_RATE,4),
        "results": ranked[:15],
        "best": ranked[0] if ranked else None,
    }
    json.dump(out, open(RESULT_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已保存: {RESULT_PATH}")


def _pr(r):
    print(f"  全命中={r['hit_rate']:.4f}({r['hits']}/{r['total']}) Lift={r['lift']:+.0f}%"
          f" | Zu6={r['zu6_rate']:.4f}({r['zu6_hits']}/{r['zu6_total']}) "
          f"Zu6Lift={r['zu6_lift']:+.0f}% | ≥2码={r['ge2_rate']:.3f} | {r['elapsed_s']}s")


if __name__ == "__main__":
    main()
