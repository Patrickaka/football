#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""决定性实验：用 ML 预测概率直接选 Top-5/Top-2 单票，与同窗口随机对照。

回答核心问题：大乐透「精选一注」能否通过 ML 信号真正战胜随机？
方法：train-once（在更旧窗口训练），对最近 trials 期逐期用「该期之前」历史预测，
取概率最高的 5 个前区 / 2 个后区作为单票，统计命中分布。
"""
import sys
import os
import logging
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.getLogger('lottery_ml').setLevel(logging.WARNING)

from src.lottery.ml import DaletouMLPredictor, TRAINING_WINDOW
from src.lottery import LotteryAnalyzer

TOP_F, TOP_B = 5, 2


def main(trials=80):
    a = LotteryAnalyzer()
    data = a.history_data
    if len(data) < TRAINING_WINDOW + trials + 10:
        print(f"历史不足：需要至少 {TRAINING_WINDOW + trials + 10} 期，现有 {len(data)}")
        return

    # 训练窗口：回测窗口之后（更旧）的最近 TRAINING_WINDOW 期
    train_data = data[trials:trials + TRAINING_WINDOW]
    predictor = DaletouMLPredictor(train_data)
    ok = predictor.train()
    if not ok or not predictor.trained:
        print("ML 训练失败")
        return
    print(f"ML 模型已训练：前区AUC={predictor.front_scores} 后区AUC={predictor.back_scores}")

    fdist = {i: 0 for i in range(6)}
    bdist = {i: 0 for i in range(3)}
    fsum = bsum = 0
    # 随机对照（同窗口）
    rfdist = {i: 0 for i in range(6)}
    rbdist = {i: 0 for i in range(3)}
    rfsum = rbsum = 0
    valid = 0

    orig_history = predictor.history
    for i in range(trials):
        actual = data[i]
        af = set(actual['front'])
        ab = set(actual['back'])
        # 预测用「该期之前」的历史，避免泄漏
        predictor.history = data[i + 1:]
        if len(predictor.history) < 10:
            continue
        res = predictor.predict()
        pred_f = sorted(res['front_probs'], key=lambda n: -res['front_probs'][n])[:TOP_F]
        pred_b = sorted(res['back_probs'], key=lambda n: -res['back_probs'][n])[:TOP_B]
        fm = len(af & set(pred_f))
        bm = len(ab & set(pred_b))
        fdist[fm] += 1
        bdist[bm] += 1
        fsum += fm
        bsum += bm

        rf = set(random.sample(range(1, 36), TOP_F))
        rb = set(random.sample(range(1, 13), TOP_B))
        rfm = len(af & rf)
        rbm = len(ab & rb)
        rfdist[rfm] += 1
        rbdist[rbm] += 1
        rfsum += rfm
        rbsum += rbm
        valid += 1

    predictor.history = orig_history

    n = valid or 1
    def rate(d, lo, hi):
        return sum(d[k] for k in range(lo, hi + 1)) / n
    print(f"\n{'='*72}\nML单票(Top5/Top2) vs 随机 — 最近 {n} 期\n{'='*72}")
    print(f"{'指标':<12}| {'ML单票':>10} | {'随机':>10} | {'理论基线':>10}")
    fg2_ml, fg2_r = rate(fdist,2,5), rate(rfdist,2,5)
    fg3_ml, fg3_r = rate(fdist,3,5), rate(rfdist,3,5)
    bg1_ml, bg1_r = rate(bdist,1,2), rate(rbdist,1,2)
    bg2_ml, bg2_r = bdist.get(2,0)/n, rbdist.get(2,0)/n
    import math
    b_fg2 = 1 - (math.comb(5,0)*math.comb(30,5)+math.comb(5,1)*math.comb(30,4))/math.comb(35,5)
    b_bg1 = 1 - math.comb(10,2)/math.comb(12,2)
    b_bg2 = 1/math.comb(12,2)
    print(f"{'前区≥2':<12}| {fg2_ml*100:9.1f}% | {fg2_r*100:9.1f}% | {b_fg2*100:9.1f}%")
    print(f"{'前区≥3':<12}| {fg3_ml*100:9.1f}% | {fg3_r*100:9.1f}% | {1.39:9.1f}%")
    print(f"{'后区≥1':<12}| {bg1_ml*100:9.1f}% | {bg1_r*100:9.1f}% | {b_bg1*100:9.1f}%")
    print(f"{'后区=2':<12}| {bg2_ml*100:9.1f}% | {bg2_r*100:9.1f}% | {b_bg2*100:9.1f}%")
    print(f"{'前均':<12}| {fsum/n:10.2f} | {rfsum/n:10.2f} | {0.71:10.2f}")
    print(f"{'后均':<12}| {bsum/n:10.2f} | {rbsum/n:10.2f} | {0.33:10.2f}")
    print(f"\n结论：ML单票 前区≥2 比随机 {'高' if fg2_ml>fg2_r else '低'} "
          f"{abs(fg2_ml-fg2_r)*100:.1f}pp；后区≥1 比随机 "
          f"{'高' if bg1_ml>bg1_r else '低'} {abs(bg1_ml-bg1_r)*100:.1f}pp")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=80)
    args = p.parse_args()
    random.seed(2026)
    main(args.trials)
