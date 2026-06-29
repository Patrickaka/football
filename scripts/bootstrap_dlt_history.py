#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性引导大乐透全量历史。

从 500.com 抓取自 2007 年至今的全部开奖（约 2900 期），与本地遗留数据
合并去重后，持久化到 doc_store（data/doc_store_dlt_history.json）。
此后 repositories.dlt_load() 即从 doc_store 读取全量真实数据，
不再回退到 120 期遗留 JSON 或随机模拟数据。

用法：
    .venv/Scripts/python.exe scripts/bootstrap_dlt_history.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lottery import LotteryAnalyzer
from src.common import repositories


def main():
    analyzer = LotteryAnalyzer()
    local = {r['issue']: r for r in analyzer.history_data
             if not analyzer.using_simulated_data}
    print(f'本地真实数据: {len(local)} 期')

    fetched = analyzer._fetch_500_history(count=10000)
    print(f'500.com 抓取: {len(fetched)} 期')
    if not fetched:
        print('抓取失败，未改动存储。')
        return 1

    merged = dict(local)
    for r in fetched:
        merged[r['issue']] = r  # 网络数据优先（含日期）

    results = sorted(merged.values(), key=lambda x: x['issue'], reverse=True)
    print(f'合并去重后: {len(results)} 期  ({results[-1]["issue"]} ~ {results[0]["issue"]})')

    repositories.dlt_save(results)
    print('已写入 doc_store。')

    # 校验回读
    reloaded = repositories.dlt_load()
    print(f'回读校验: {len(reloaded)} 期, 最新 {reloaded[0]["issue"]} {reloaded[0]["front"]}+{reloaded[0]["back"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
