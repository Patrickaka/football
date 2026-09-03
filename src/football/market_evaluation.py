# -*- coding: utf-8 -*-
"""市场评估的适配层：读 football-data CSV，交给领域层评分。

领域层（`domain.sports.football.market_evaluation`）只认行，不碰文件。
"""

import csv
import pathlib
import re
from typing import Dict, Iterable, List, Sequence

from ..domain.sports.football import market_evaluation as _domain

_FILE_PATTERN = re.compile(r'^(?P<league>[A-Z]+\d*)_(?P<season>\d{4})\.csv$')

DEFAULT_SOURCES = (
    'pinnacle_open', 'pinnacle_close',
    'b365_open', 'b365_close',
    'avg_open', 'avg_close', 'max_close',
)
DEFAULT_SOFT_BOOKS = ('b365_close', 'avg_close', 'max_close')
DEFAULT_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)


def football_data_files(directory) -> List[pathlib.Path]:
    """`<联赛>_<赛季>.csv` 形态的文件，按名字排序。"""
    directory = pathlib.Path(directory)
    return sorted(path for path in directory.glob('*.csv')
                  if _FILE_PATTERN.match(path.name))


def load_football_data_rows(path) -> List[Dict]:
    """一份 CSV → 行列表，每行附 league / season。

    首列名带 UTF-8 BOM（`\\ufeffDiv`），不剥掉的话按 'Div' 取会取空。
    """
    path = pathlib.Path(path)
    matched = _FILE_PATTERN.match(path.name)
    league = matched.group('league') if matched else path.stem
    season = matched.group('season') if matched else ''
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {key.strip(): value for key, value in raw.items() if key is not None}
            if not row.get('FTR'):
                continue
            row['league'] = league
            row['season'] = season
            rows.append(row)
    return rows


def _evaluate(rows: Sequence[Dict], sources, soft_books, thresholds, devig) -> Dict:
    return {
        'n_rows': len(rows),
        'sources': _domain.evaluate_sources(rows, sources, devig),
        'ev': _domain.evaluate_ev_strategy(
            rows, 'pinnacle_close', soft_books, thresholds, devig),
    }


def run_market_evaluation(files: Iterable, sources=DEFAULT_SOURCES,
                          soft_books=DEFAULT_SOFT_BOOKS,
                          thresholds=DEFAULT_THRESHOLDS,
                          devig: str = 'proportional') -> Dict:
    """整体 + 按联赛的评估报告。`devig` 见领域层 DEVIG_METHODS。"""
    rows: List[Dict] = []
    for path in files:
        rows.extend(load_football_data_rows(path))
    by_league: Dict[str, List[Dict]] = {}
    for row in rows:
        by_league.setdefault(row['league'], []).append(row)
    report = _evaluate(rows, sources, soft_books, thresholds, devig)
    report['devig'] = devig
    report['by_league'] = {
        league: _evaluate(league_rows, sources, soft_books, thresholds, devig)
        for league, league_rows in sorted(by_league.items())
    }
    return report
