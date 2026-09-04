#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""市场评估底座：多个赔率来源当预测者，同一把尺子比。

    python scripts/backtest/evaluate_markets.py            # data/ 下全部 CSV
    python scripts/backtest/evaluate_markets.py --league E0 I1
    python scripts/backtest/evaluate_markets.py --json out.json

读的是 football-data.co.uk 格式（本仓 data/<联赛>_<赛季>.csv）。
Pinnacle 收盘做尖锐参照，Bet365 / 市场均价 / 最高价做软盘代理。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.football.market_evaluation import (  # noqa: E402
    DEFAULT_THRESHOLDS, football_data_files, run_market_evaluation,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


def _fmt(value, digits=4):
    return '-' if value is None else f'{value:.{digits}f}'


def _print_sources(title, block):
    print(f'\n{title}  (n={block["n_rows"]})')
    print(f'  {"来源":<16}{"n":>6}{"log_loss":>10}{"brier":>9}{"top1":>8}{"ece":>8}')
    for name, metrics in block['sources'].items():
        print(f'  {name:<16}{metrics["n"]:>6}{_fmt(metrics["log_loss"]):>10}'
              f'{_fmt(metrics["brier"]):>9}{_fmt(metrics["top1_hit_rate"], 3):>8}'
              f'{_fmt(metrics["ece"], 3):>8}')


def _print_ev(block):
    print(f'\n  EV 策略（Pinnacle 收盘做真概率，按软盘赔率下注）')
    print(f'  {"软盘":<12}{"阈值":>6}{"注数":>6}{"ROI":>9}{"95%区间":>20}{"命中":>7}{"均赔":>7}')
    for soft, grid in block['ev'].items():
        for threshold, summary in grid.items():
            if summary['n'] == 0:
                print(f'  {soft:<12}{threshold:>6.2f}{0:>6}{"-":>9}{"-":>20}{"-":>7}{"-":>7}')
                continue
            lo, hi = summary['roi_ci95']
            print(f'  {soft:<12}{threshold:>6.2f}{summary["n"]:>6}{summary["roi"]:>+9.4f}'
                  f'{f"[{lo:+.3f}, {hi:+.3f}]":>20}{summary["hit_rate"]:>7.3f}'
                  f'{summary["avg_odds"]:>7.2f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--league', nargs='*', help='只跑这些联赛代码，如 E0 I1')
    parser.add_argument('--json', help='把完整报告写到这个文件')
    parser.add_argument('--no-league-breakdown', action='store_true')
    parser.add_argument('--devig', choices=('proportional', 'power'), default='proportional',
                        help='去水方法；proportional 与生产一致，power 修正热门-冷门偏差')
    args = parser.parse_args()

    files = football_data_files(DATA_DIR)
    if args.league:
        wanted = set(args.league)
        files = [f for f in files if f.name.split('_')[0] in wanted]
    if not files:
        print('没有可用的 CSV', file=sys.stderr)
        return 1
    print('文件:', ', '.join(f.name for f in files))
    print('去水方法:', args.devig)

    report = run_market_evaluation(files, devig=args.devig)
    _print_sources('全部', report)
    _print_ev(report)
    if not args.no_league_breakdown:
        for league, block in report['by_league'].items():
            _print_sources(f'联赛 {league}', block)
            _print_ev(block)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        print(f'\n完整报告已写入 {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
