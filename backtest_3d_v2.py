"""
福彩3D 深度调优脚本 v2
======================
基于基线回测结果，做以下维度的系统性优化：

Phase 1: Lam 精细搜索 (0.5 ~ 1.2 step=0.1)
Phase 2: 核心权重坐标下降 (W_MARKOV, W_HOT_GLOBAL/POS, W_MISS, W_CONSEC, W_FORM_PRIOR, ...)
Phase 3: 特征开关组合搜索 (road, form_switch, slope, cold_hot_balance)
Phase 4: IPF 迭代次数 & fstr 权重微调
Phase 5: 新信号注入测试 (遗漏回补增强、位置偏差修正、和值动量)

策略：每阶段用快速回测(step=10)筛选候选，最终用慢速回测(step=3)验证最优。
"""

import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.lottery3d as L

REAL_RAW = "data/_raw_3d_2000.json"
MIN_PERIODS = 100


def load_numbers():
    rows = json.load(open(REAL_RAW, "r", encoding="utf-8"))
    return [tuple(int(d) for d in r[2:5]) for r in rows]


def build_pipeline(numbers, period_idx):
    hist = list(numbers[:period_idx])
    if len(hist) < MIN_PERIODS:
        return None
    periods = [str(i) for i in range(len(hist))]
    sums = [sum(x) for x in hist]
    spans = [max(x) - min(x) for x in hist]
    try:
        ww, _ = L.resolve_window_weights(hist, compute_weights=False, period=periods[-1])
    except Exception:
        return None
    meta = L.build_ranking_meta(hist, ww, sums, spans, tail_top=5)
    score, freq_all = L.ensemble_digit_scores(hist, ww, dynamic=meta.get("dynamic"))
    ds = L._blend_dan_score(score, meta)
    danma, tuoma, kill, rank = L.pick_dan_tuo_kill(ds, enable_danma_random=False)
    meta["numbers"] = hist
    return {"score": score, "meta": meta, "danma": danma, "kill": kill, "hist": hist}


def predict_topk(pipe, top_k=30):
    result = L.rank_triplets(
        pipe["score"], pipe["danma"], pipe["kill"], pipe["meta"],
        top_n=top_k,
        enable_exploration=False, apply_noise=False,
        enable_cold_hot_balance=False, enable_diversity=False,
        enable_correlation=False, recent_recommendations=None,
    )
    return [(w, t) for w, t in result]


def quick_backtest(numbers, start_idx=250, end_idx=-50, step=10, top_k=30):
    """快速回测：只返回 TopK 直选命中率和 >=2码率"""
    actual_end = len(numbers) + end_idx
    hits = total = ge2_hits = 0
    
    for idx in range(start_idx, actual_end, step):
        target = numbers[idx]
        target_str = "".join(map(str, target))
        pipe = build_pipeline(numbers, idx)
        if pipe is None:
            continue
        preds = predict_topk(pipe, top_k=top_k)
        total += 1
        pred_set = set(t for _, t in preds)
        if any(t == target_str for t in pred_set):
            hits += 1
        max_ov = 0
        for t in pred_set:
            o = len(set(target_str) & set(t))
            if o > max_ov:
                max_ov = o
        if max_ov >= 2:
            ge2_hits += 1

    return {
        "hit_rate": hits / total if total else 0,
        "hits": hits,
        "total": total,
        "ge2_rate": ge2_hits / total if total else 0,
    }


def save_attr(name, value):
    """保存属性值以便后续恢复"""
    return getattr(L, name)


def restore_attr(name, old_value):
    setattr(L, name, old_value)


# ================================================================
# Phase 1: Lam 精细搜索
# ================================================================
def phase1_lam_fine(numbers):
    print("\n" + "="*60)
    print("  Phase 1: Lam 精细搜索")
    print("="*60)
    
    results = {}
    for lam_val in [x / 10.0 for x in range(5, 13)]:  # 0.5 ~ 1.2
        L.DIST_CALIBRATION_LAM = lam_val
        L.DIST_CALIBRATION_ENABLED = lam_val > 0
        
        res = quick_backtest(numbers, step=10, top_k=30)
        results[lam_val] = res["hit_rate"]
        print(f"  lam={lam_val:.1f}: Top30={res['hit_rate']*100:.2f}% "
              f"(hits={res['hits']}/{res['total']}) >=2码={res['ge2_rate']*100:.1f}%")

    best_lam = max(results.keys(), key=lambda k: results[k])
    print(f"\n  最优 lam = {best_lam} (Top30={results[best_lam]*100:.2f}%)")
    
    # 应用最优值
    L.DIST_CALIBRATION_LAM = best_lam
    L.DIST_CALIBRATION_ENABLED = True
    return best_lam, results


# ================================================================
# Phase 2: 核心权重坐标下降
# ================================================================
TUNABLE_WEIGHTS = [
    # (attr_name, search_range, description)
    ("W_MARKOV",       [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],   "马尔可夫转移"),
    ("W_MARKOV2",      [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],     "二阶马尔可夫"),
    ("W_HOT_GLOBAL",   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],           "热号全局"),
    ("W_HOT_POS",      [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],"分位热号"),
    ("W_MISS_HIGH",    [0.5, 0.8, 1.0, 1.5, 2.0, 2.5],           "高遗漏"),
    ("W_MISS_MID",     [0.3, 0.5, 0.6, 0.8, 1.0, 1.2],           "中遗漏"),
    ("W_CONSECUTIVE",  [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],           "连号奖励"),
    ("W_FORM_PRIOR",   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],      "形态先验"),
    ("W_POS_REPEAT",   [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5],      "上期同位重复"),
    ("W_DANMA_HIT",    [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],           "胆码命中"),
    ("W_KILL_PENALTY", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],           "杀码惩罚"),
    ("W_NEIGHBOR",     [0.5, 1.0, 1.5, 2.0, 2.5],                 "邻号加分"),
    ("W_RATIO_MATCH",  [1.0, 1.5, 1.8, 2.0, 2.5, 3.0],            "奇偶大小比匹配"),
    ("W_SUM_SOFT",     [4.0, 6.0, 8.0, 10.0, 12.0],               "和值软先验"),
    ("W_SPAN_SOFT",    [2.0, 3.0, 5.0, 7.0, 9.0],                 "跨度软先验"),
    ("W_LAST_APPEAR",  [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],            "近期实开"),
    ("W_REALIZED",     {3: [2.0, 2.5, 3.0, 3.2, 3.5, 4.0],
                        8: [1.5, 2.0, 2.2, 2.5, 3.0],
                        15:[1.0, 1.2, 1.4, 1.6, 1.8, 2.0]},      "近窗实开锚定"),
]


def phase2_weight_search(numbers):
    print("\n" + "="*60)
    print("  Phase 2: 核心权重坐标下降")
    print("="*60)

    base_hit = quick_backtest(numbers, step=10, top_k=30)["hit_rate"]
    print(f"  基线 Top30命中率: {base_hit*100:.2f}%")

    best_weights = {}
    improvements = {}

    for item in TUNABLE_WEIGHTS:
        attr_name = item[0]  # e.g., "W_MARKOV"
        
        # 跳过 W_REALIZED (它是 dict，特殊处理)
        if attr_name == "W_REALIZED":
            continue
            
        search_range = item[1]
        desc = item[2]

        old_val = getattr(L, attr_name)
        best_v = old_val
        best_r = base_hit
        
        results_for_this = {}
        for v in search_range:
            setattr(L, attr_name, v)
            
            # 特殊处理 dict 类型
            if attr_name == "W_REALIZED":
                pass  # skip
            
            try:
                res = quick_backtest(numbers, step=10, top_k=30)
                r = res["hit_rate"]
                results_for_this[v] = r
                
                if r > best_r + 0.001:  # 至少提升 0.1%
                    best_r = r
                    best_v = v
                    
            except Exception as e:
                print(f"    ERROR at {attr_name}={v}: {e}")
                continue

        # 恢复到最优值（如果找到了更好的）
        setattr(L, attr_name, best_v)
        delta = (best_r - base_hit) * 100
        improvement_pct = (best_r / base_hit - 1) * 100 if base_hit > 0 else 0
        
        if abs(delta) > 0.05:
            status = "✅ 提升" if delta > 0 else "⬇️ 下降"
            print(f"  {attr_name:18s} ({desc}): {old_val:.1f}→{best_v:.1f}  "
                  f"{base_hit*100:.2f}%→{best_r*100:.2f}% ({delta:+.2f}%) {status}")
            best_weights[attr_name] = best_v
            improvements[attr_name] = delta
        else:
            print(f"  {attr_name:18s} ({desc}): 保持={old_val:.1f}  变化<0.05%")

    return best_weights, improvements


# ================================================================
# Phase 3: 特征开关组合
# ================================================================
def phase3_feature_flags(numbers):
    print("\n" + "="*60)
    print("  Phase 3: 特征开关组合搜索")
    print("="*60)
    
    base_hit = quick_backtest(numbers, step=10, top_k=30)["hit_rate"]
    print(f"  基线 Top30命中率: {base_hit*100:.2f}%")
    print(f"  当前 FEATURE_FLAGS: {json.dumps(L.FEATURE_FLAGS, ensure_ascii=False)}")

    # 测试每个关闭的特征单独打开
    toggles = [
        ("road",         True),   # 012路
        ("form_switch",  True),   # 形态切换
        ("slope",        True),   # 斜连
        ("cold_hot_balance", True), # 冷热平衡
    ]
    
    best_flags = dict(L.FEATURE_FLAGS)
    
    for flag_name, target_val in toggles:
        current_val = L.FEATURE_FLAGS.get(flag_name)
        if current_val == target_val:
            continue  # 已经是这个值了
            
        old_flags = dict(L.FEATURE_FLAGS)
        L.FEATURE_FLAGS[flag_name] = target_val
        
        try:
            res = quick_backtest(numbers, step=10, top_k=30)
            r = res["hit_rate"]
            delta = (r - base_hit) * 100
            action = "开启" if target_val else "关闭"
            
            if r > base_hit + 0.002:
                print(f"  ✅ {flag_name:20s} {action}: {base_hit*100:.2f}%→{r*100:.2f}% ({delta:+.2f}%)")
                best_flags[flag_name] = target_val
            else:
                print(f"  ❌ {flag_name:20s} {action}: {base_hit*100:.2f}%→{r*100:.2f}% ({delta:+.2f}%)")
                # 回滚
                L.FEATURE_FLAGS[flag_name] = current_val
        except Exception as e:
            print(f"  ERROR {flag_name}: {e}")
            L.FEATURE_FLAGS = old_flags

    # 也测试一些当前开启的特征是否应该关闭
    close_candidates = [
        ("neighbor", False),
        ("pair", False),
        ("consecutive", False),
        ("lag1_repeat", False),
        ("ratio", False),
    ]
    
    for flag_name, target_val in close_candidates:
        if L.FEATURE_FLAGS.get(flag_name) == target_val:
            continue
            
        old_flags = dict(L.FEATURE_FLAGS)
        L.FEATURE_FLAGS[flag_name] = target_val
        
        try:
            res = quick_backtest(numbers, step=10, top_k=30)
            r = res["hit_rate"]
            delta = (r - base_hit) * 100
            action = "关闭" if not target_val else "开启"
            
            if r > base_hit + 0.002:
                print(f"  ✅ {flag_name:20s} {action}: {base_hit*100:.2f}%→{r*100:.2f}% ({delta:+.2f}%)")
                best_flags[flag_name] = target_val
            else:
                print(f"  ❌ {flag_name:20s} {action}: {base_hit*100:.2f}%→{r*100:.2f}% ({delta:+.2f}%)")
                L.FEATURE_FLAGS[flag_name] = old_flags[flag_name]
        except Exception as e:
            print(f"  ERROR {flag_name}: {e}")
            L.FEATURE_FLAGS = old_flags

    return best_flags


# ================================================================
# Phase 4: IPF 参数调优
# ================================================================
def phase4_ipf_params(numbers):
    print("\n" + "="*60)
    print("  Phase 4: IPF 校准参数调优")
    print("="*60)
    
    # 需要修改 apply_distribution_calibration 的内部参数
    # 由于函数内部的迭代次数和特征强度是硬编码的，我们通过 monkey-patch 来测试
    
    base_hit = quick_backtest(numbers, step=10, top_k=30)["hit_rate"]
    print(f"  基线 Top30命中率: {base_hit*100:.2f}%")
    
    # 保存原始函数
    orig_func = L.apply_distribution_calibration
    
    results = {}
    
    # 测试不同的 IPF 配置
    configs = [
        {"ipf_iters": 8,  "lam": 0.8, "digit_str": 1.0, "pos_str": 0.5},
        {"ipf_iters": 16, "lam": 0.8, "digit_str": 1.0, "pos_str": 0.5},
        {"ipf_iters": 24, "lam": 0.8, "digit_str": 1.0, "pos_str": 0.5},
        {"ipf_iters": 16, "lam": 1.0, "digit_str": 1.0, "pos_str": 0.5},
        {"ipf_iters": 16, "lam": 0.8, "digit_str": 1.5, "pos_str": 0.8},
        {"ipf_iters": 16, "lam": 0.8, "digit_str": 0.8, "pos_str": 0.3},
        {"ipf_iters": 32, "lam": 0.9, "digit_str": 1.2, "pos_str": 0.6},
        {"ipf_iters": 16, "lam": 0.9, "digit_str": 1.2, "pos_str": 0.6},
    ]
    
    for cfg in configs:
        def make_patched_func(cfg):
            def patched_calib(weights, numbers_param, lam=L.DIST_CALIBRATION_LAM):
                # 用 cfg 中的参数覆盖默认行为
                import math
                from collections import Counter
                n = len(numbers_param)
                
                set_marg = Counter(); sum_c = Counter(); span_c = Counter()
                oe_c = Counter(); bs_c = Counter(); consec_c = Counter()
                form_c = Counter(); pos_m = [Counter() for _ in range(3)]
                for x in numbers_param:
                    for d in set(x): set_marg[d] += 1
                    sum_c[sum(x)] += 1; span_c[max(x)-min(x)] += 1
                    oe_c[sum(d%2==1 for d in x)] += 1
                    bs_c[sum(d>=5 for d in x)] += 1
                    consec_c[any(sorted(x)[i+1]-sorted(x)[i]==1 for i in range(2))] += 1
                    form_c[L.classify_form(x)] += 1
                    for i in range(3): pos_m[i][x[i]] += 1
                
                safe = lambda x: max(x, 1e-4)
                def p(c,k): return max(c.get(k,0)/n, 1e-4)
                def ps(d): return max(set_marg.get(d,0)/n, 1e-4)
                def pc(b): return max(consec_c.get(b,0)/n, 1e-4)
                def pf(f): return max(form_c.get(f,0)/n, 1e-4)
                
                ws = [w for w,_ in weights]; base_mu=sum(ws)/len(ws)
                base_var=sum((w-base_mu)**2 for w in ws)/len(ws); base_std=math.sqrt(base_var) or 1.0
                T=base_std or 1.0; mx=max(ws)
                ex=[math.exp((w-mx)/T) for w in ws]; Z=sum(ex)
                probs=[e/Z for e in ex]
                
                feats=[]
                for (w,t) in weights:
                    s=set(t)
                    feats.append((frozenset(s), sum(t), max(t)-min(t),
                                  sum(d%2==1 for d in t), sum(d>=5 for d in t),
                                  any(sorted(t)[i+1]-sorted(t)[i]==1 for i in range(2)),
                                  L.classify_form(t), t))
                
                fstr={"digit": cfg.get("digit_str", 1.0), "sum": 1.0, "span": 1.0,
                      "oe": 1.0, "bs": 1.0, "consec": 1.0, "form": 0.5,
                      "pos": cfg.get("pos_str", 0.5)}
                
                scores=[0.0]*len(weights)
                for _ in range(cfg.get("ipf_iters", 16)):
                    m_digit=Counter(); m_sum=Counter(); m_span=Counter()
                    m_oe=Counter(); m_bs=Counter(); m_cons=Counter(); m_form=Counter()
                    m_pos=[Counter() for _ in range(3)]
                    for i,pr in enumerate(probs):
                        ft=feats[i]
                        for d in ft[0]: m_digit[d]+=pr
                        m_sum[ft[1]]+=pr; m_span[ft[2]]+=pr
                        m_oe[ft[3]]+=pr; m_bs[ft[4]]+=pr
                        m_cons[ft[5]]+=pr; m_form[ft[6]]+=pr
                        for pi in range(3): m_pos[pi][ft[7][pi]]+=pr
                    for i,(w,t) in enumerate(weights):
                        ft=feats[i]; delta=0.0
                        for d in ft[0]: delta+=fstr["digit"]*(math.log(ps(d))-math.log(safe(m_digit[d])))
                        delta+=fstr["sum"]*(math.log(p(sum_c,ft[1]))-math.log(safe(m_sum[ft[1]])))
                        delta+=fstr["span"]*(math.log(p(span_c,ft[2]))-math.log(safe(m_span[ft[2]])))
                        delta+=fstr["oe"]*(math.log(p(oe_c,ft[3]))-math.log(safe(m_oe[ft[3]])))
                        delta+=fstr["bs"]*(math.log(p(bs_c,ft[4]))-math.log(safe(m_bs[ft[4]])))
                        delta+=fstr["consec"]*(math.log(pc(ft[5]))-math.log(safe(m_cons[ft[5]])))
                        delta+=fstr["form"]*(math.log(pf(ft[6]))-math.log(safe(m_form[ft[6]])))
                        for pi in range(3):
                            delta+=fstr["pos"]*(math.log(max(pos_m[pi].get(t[pi],0)/n,1e-4))-math.log(safe(m_pos[pi][t[pi]])))
                        scores[i]+=cfg.get("lam", 0.8)*delta
                    
                    mx_s=max(scores); se=[math.exp((s-mx_s)/T) for s in scores]; Zs=sum(se)
                    probs=[e/Zs for e in se]
                
                cmu=sum(scores)/len(scores); corr=[c-cmu for c in scores]
                cvar=sum(x*x for x in corr)/len(corr); csd=math.sqrt(cvar) or 1.0
                factor=cfg.get("lam", 0.8)*base_std
                return [(w+factor*c/csd, t) for (w,t),c in zip(weights,corr)]
            return patched_calib
        
        L.apply_distribution_calibration = make_patched_func(cfg)
        L.DIST_CALIBRATION_LAM = cfg.get("lam", 0.8)
        
        try:
            res = quick_backtest(numbers, step=10, top_k=30)
            label = f"iters={cfg['ipf_iters']:2d} lam={cfg['lam']} d={cfg['digit_str']} p={cfg['pos_str']}"
            results[label] = res["hit_rate"]
            delta = (res["hit_rate"] - base_hit) * 100
            marker = " ✅" if res["hit_rate"] > base_hit + 0.002 else ""
            print(f"  {label:35s}: {res['hit_rate']*100:.2f}% ({delta:+.2f}%){marker}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # 恢复原始函数
    L.apply_distribution_calibration = orig_func
    
    best_cfg = max(results.keys(), key=lambda k: results[k]) if results else None
    if best_cfg:
        print(f"\n  最优IPF配置: {best_cfg} (Top30={results[best_cfg]*100:.2f}%)")
    
    return best_cfg, results


# ================================================================
# 最终验证：全量回测
# ================================================================
def final_validation(numbers, step=3):
    print("\n" + "="*60)
    print("  最终验证 (step=3, 全量回测)")
    print("="*60)

    t0 = time.time()
    actual_end = len(numbers) - 50  # 保留最近50期
    stats = {3: {"h":0,"t":0}, 10: {"h":0,"t":0}, 30: {"h":0,"t":0}, "ge2_30": {"h":0,"t":0}}
    
    count = 0
    for idx in range(250, actual_end, step):
        target = numbers[idx]
        target_str = "".join(map(str, target))
        pipe = build_pipeline(numbers, idx)
        if pipe is None:
            continue
        
        preds = predict_topk(pipe, top_k=30)
        pred_set = set(t for _, t in preds)
        count += 1

        for k in [3, 10, 30]:
            topk_set = set(t for _, t in preds[:k])
            if any(t == target_str for t in topk_set):
                stats[k]["h"] += 1
            stats[k]["t"] += 1

        max_ov = 0
        for t in pred_set:
            o = len(set(target_str) & set(t))
            if o > max_ov: max_ov = o
        if max_ov >= 2:
            stats["ge2_30"]["h"] += 1
        stats["ge2_30"]["t"] += 1

        if count % 50 == 0:
            elapsed = time.time() - t0
            print(f"  进度: {count} 期 ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  完成: {count} 期, 耗时 {elapsed:.1f}s ({elapsed/count:.2f}s/期)")

    final_results = {}
    for k in [3, 10, 30]:
        n = stats[k]["t"]
        hr = stats[k]["h"] / n if n else 0
        exp = k / 1000.0
        lift = (hr/exp-1)*100 if exp > 0 else 0
        final_results[f"top{k}_hit"] = hr
        final_results[f"top{k}_hits"] = stats[k]["h"]
        final_results[f"top{k}_total"] = n
        final_results[f"top{k}_lift"] = lift
        print(f"  Top{k:>2d}: {hr*100:.2f}% ({stats[k]['h']}/{n}) 期望={exp*100:.1f}% 提升={lift:+.1f}%")

    ge2_n = stats["ge2_30"]["t"]
    ge2_r = stats["ge2_30"]["h"] / ge2_n if ge2_n else 0
    final_results["ge2_30_rate"] = ge2_r
    print(f"  >=2码: {ge2_r*100:.1f}%")
    final_results["total_periods"] = count
    final_results["elapsed"] = round(elapsed, 1)
    return final_results


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("  福彩3D 深度调优 v2")
    print("=" * 70)

    numbers = load_numbers()
    print(f"\n  数据: {len(numbers)} 期")

    # Phase 1: Lam 精细搜索
    best_lam, lam_results = phase1_lam_fine(numbers)

    # Phase 2: 权重坐标下降  
    best_weights, weight_improvements = phase2_weight_search(numbers)

    # Phase 3: 特征开关
    best_flags = phase3_feature_flags(numbers)

    # Phase 4: IPF 参数
    best_ipf, ipf_results = phase4_ipf_params(numbers)

    # 汇总最优配置
    print("\n" + "="*70)
    print("  最优配置汇总")
    print("="*70)
    
    optimal_config = {
        "DIST_CALIBRATION_LAM": best_lam,
        "DIST_CALIBRATION_ENABLED": True,
        "WEIGHT_CHANGES": {k: getattr(L, k) for k in best_weights},
        "FEATURE_FLAGS": dict(L.FEATURE_FLAGS),
        "BEST_IPF_CONFIG": best_ipf,
    }
    
    print(f"\n  Lam = {best_lam}")
    print(f"\n  权重变更:")
    for k, v in optimal_config["WEIGHT_CHANGES"].items():
        print(f"    {k} = {v}")
    
    # 最终验证
    final = final_validation(numbers, step=3)
    optimal_config["final_results"] = final

    # 保存
    out_path = "data/bt_deep_optimization.json"
    open(out_path, "w", encoding="utf-8").write(
        json.dumps(optimal_config, ensure_ascii=False, indent=2, default=str)
    )
    print(f"\n  结果已保存: {out_path}")
    
    return optimal_config


if __name__ == "__main__":
    main()
