#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大乐透融合诊断：规则 vs ML vs 融合 在前区ge4/ge3、后区ge1/ge2 的对比（近期数据）"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.getLogger('lottery').setLevel(logging.WARNING)
logging.getLogger('lottery_ml').setLevel(logging.WARNING)

from src.lottery import LotteryAnalyzer
from src.lottery.ml import predict_with_ml

def fuse(rule_top, ml_top, rule_w, ml_w, top_n):
    """简单融合：规则排名位置得分 + ML排名位置得分，取Top-N"""
    rule_score = {num: (len(rule_top) - i) for i, num in enumerate(rule_top)}
    ml_score = {num: (len(ml_top) - i) for i, num in enumerate(ml_top)}
    all_nums = set(rule_top) | set(ml_top)
    fused = []
    for num in all_nums:
        rs = rule_score.get(num, 0) * rule_w
        ms = ml_score.get(num, 0) * ml_w
        fused.append((num, rs + ms))
    fused.sort(key=lambda x: -x[1])
    return [num for num, _ in fused[:top_n]]

def main():
    analyzer = LotteryAnalyzer()
    print(f"历史期数: {len(analyzer.history_data)}")

    saved = list(analyzer.history_data)
    saved_stats = dict(analyzer.statistics)

    # 用最近 300 期做预测（贴近实际）
    trials = 300
    start_i = 50

    def eval_topn(fn_get, bn_get=None):
        ge4 = ge3 = ge2 = 0
        bge1 = bge2 = 0
        n = 0
        for i in range(start_i, start_i + trials):
            if i >= len(saved) - 11:
                break
            analyzer.history_data = list(saved[i + 1:])
            if len(analyzer.history_data) < 30:
                continue
            analyzer.update_statistics()
            actual_f = set(saved[i]['front'])
            actual_b = set(saved[i]['back'])
            n += 1
            f_top, b_top = fn_get(analyzer)
            fcr = len(actual_f & set(f_top))
            bcr = len(actual_b & set(b_top))
            if fcr >= 2: ge2 += 1
            if fcr >= 3: ge3 += 1
            if fcr >= 4: ge4 += 1
            if bcr >= 1: bge1 += 1
            if bcr >= 2: bge2 += 1
        analyzer.history_data = saved
        analyzer.statistics = saved_stats
        return dict(n=n, ge4=ge4 / n, ge3=ge3 / n, ge2=ge2 / n,
                    bge1=bge1 / n, bge2=bge2 / n)

    def rule_only(a):
        fr = a.get_ensemble_ranking(is_front=True, top_n=20)
        br = a.get_ensemble_ranking(is_front=False, top_n=8)
        return [r['number'] for r in fr], [r['number'] for r in br]

    def ml_only(a):
        mlp = predict_with_ml(a.history_data, force_retrain=False)
        return mlp['front_top'][:20], mlp['back_top'][:8]

    print("\n" + "=" * 72)
    print("【前区 ge4 对比】推荐12码 (近期300期)")
    print("=" * 72)

    # 规则
    r = eval_topn(lambda a: (rule_only(a)[0][:12], rule_only(a)[1][:5]))
    print(f"规则前区12码: ge4={r['ge4']*100:.1f}% ge3={r['ge3']*100:.1f}% "
          f"后ge1={r['bge1']*100:.1f}%")

    # ML
    try:
        r = eval_topn(lambda a: (ml_only(a)[0][:12], ml_only(a)[1][:5]))
        print(f"ML前区12码:   ge4={r['ge4']*100:.1f}% ge3={r['ge3']*100:.1f}% "
              f"后ge1={r['bge1']*100:.1f}%")
    except Exception as e:
        print("ML eval failed:", e)

    # 融合不同权重
    print("\n" + "-" * 72)
    print("【融合权重扫描】前区12码 + 后区5码")
    print("-" * 72)
    best = None
    for rw, mw in [(0.55, 0.45), (0.45, 0.55), (0.35, 0.65), (0.25, 0.75), (0.15, 0.85)]:
        def fn(a, rw=rw, mw=mw):
            fr, br = rule_only(a)
            mf, mb = ml_only(a)
            ff = fuse(fr, mf, rw, mw, 12)
            bb = fuse(br, mb, rw, mw, 5)
            return ff, bb
        r = eval_topn(fn)
        print(f"规则{rw}/ML{mw}: 前ge4={r['ge4']*100:.1f}% 前ge3={r['ge3']*100:.1f}% "
              f"后ge1={r['bge1']*100:.1f}% 后ge2={r['bge2']*100:.1f}%")
        if best is None or r['ge4'] > best[1]['ge4']:
            best = ((rw, mw), r)

    print("\n最优融合权重:", best[0], "→ 前ge4=%.1f%%" % (best[1]['ge4'] * 100))

    # 前区推荐数量扫描（用最优融合权重）
    print("\n" + "-" * 72)
    print("【前区推荐数量扫描】最优融合权重 + 后区5码")
    print("-" * 72)
    rw, mw = best[0]
    for fn_n in [8, 10, 12, 15, 18, 20]:
        def fn(a, fn_n=fn_n):
            fr, br = rule_only(a)
            mf, mb = ml_only(a)
            ff = fuse(fr, mf, rw, mw, fn_n)
            bb = fuse(br, mb, rw, mw, 5)
            return ff, bb
        r = eval_topn(fn)
        print(f"前区{fn_n:>2}码: ge4={r['ge4']*100:.1f}% ge3={r['ge3']*100:.1f}% "
              f"后ge1={r['bge1']*100:.1f}%")

if __name__ == '__main__':
    main()
