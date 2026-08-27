"""重新生成 tests/fixtures/golden/ 下的黄金文件。

    PYTHONPATH=. python3 scripts/regen_golden.py

**只在确认输出变化是有意的之后才跑它。** 黄金文件的用处正是对无意的改动
敏感；先重新生成再看测试变绿，等于把这条防线关掉。

首版黄金值生成于 2026-08-26，来源是当时的实现——而它已经与迁移前的
`src/basketball` 逐字差分验证通过，所以记录的是迁移前的行为。
"""
import gzip
import json
import pathlib

OUT = pathlib.Path('tests/fixtures/golden')


def dump(name, payload):
    path = OUT / f'{name}.json.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
    print(f'  {name}: {len(payload)} 条, {path.stat().st_size} 字节')


# ---- movement ----
from tests.domain import test_basketball_movement as tm
from src.domain.sports.basketball import movement as mv

golden = {}
for i, seq in enumerate(tm.SNAPSHOT_SEQS):
    golden[f'snap:ml:{i}'] = mv.movement_from_snapshots(
        seq, 'spf_home', 'spf_away', now_fn=lambda: tm.NOW)
for i, seq in enumerate(tm.HANDICAP_SEQS):
    golden[f'snap:ah:{i}'] = mv.movement_from_snapshots(
        seq, 'rqspf_home', 'rqspf_away', 'handicap', 'ah', now_fn=lambda: tm.NOW)
for i, seq in enumerate(tm.TOTAL_SEQS):
    golden[f'snap:ou:{i}'] = mv.movement_from_snapshots(
        seq, 'dx_over', 'dx_under', 'total_line', 'ou', now_fn=lambda: tm.NOW)
for i, trend in enumerate(tm.TRENDS):
    for kind in tm.KINDS:
        golden[f'trend:{i}:{kind}'] = mv.normalize_okooo_trend(trend, kind)
movements = tm._movements()
for i, m in enumerate(movements):
    for market in ('rqspf', 'dx', 'spf'):
        golden[f'infer:{i}:{market}'] = mv.infer_market_from_movement(m, market)
    for market in ('rqspf', 'dx'):
        for p in (0.35, 0.5, 0.58, 0.72):
            golden[f'apply_inf:{i}:{market}:{p}'] = mv.apply_market_inference(
                p, 1 - p, m, market)
    for p in (0.42, 0.5, 0.61):
        golden[f'apply_mv:{i}:{p}'] = mv.apply_movement(p, 1 - p, m)
    golden[f'to_trend:{i}'] = mv.movement_to_trend(m)
    for rec in ('主胜', '客胜', '让胜', '让负', '大分', '小分', '说不好', None):
        golden[f'sharp:{i}:{rec}'] = mv.sharp_confirmation(m, rec)
for i, trend in enumerate(tm.TRENDS):
    for kind in tm.KINDS:
        t = dict(trend, kind=kind) if trend else trend
        for p in (0.4, 0.5, 0.65):
            golden[f'adjust:{i}:{kind}:{p}'] = mv.adjust_two_way_by_trend(p, 1 - p, t)
BM = tm.BuildMovementForMatchGoldenTests
for mi, match in enumerate(BM.MATCHES):
    for bi, bundle in enumerate(BM.BUNDLES):
        for hi, history in enumerate((None, {}, BM.HISTORY)):
            golden[f'build:{mi}:{bi}:{hi}'] = mv.build_movement_for_match(
                match, history=history, okooo_bundle=bundle, now_fn=lambda: tm.NOW)
for ci, (movements, bets) in enumerate(tm.DescribeMarketMovementGoldenTests.cases()):
    golden[f'describe:{ci}'] = mv.describe_market_movement(movements, bets)
dump('movement', golden)

# ---- analysis ----
from tests.domain import test_basketball_analysis as ta
from src.domain.sports.basketball.analysis import BasketballAnalyzer

golden = {}
for ei, elo in enumerate(ta.ELO_SETUPS):
    for ci, cal in enumerate(ta.CALIBRATORS):
        analyzer = BasketballAnalyzer(elo=elo, calibrator=cal)
        for mi, match in enumerate(ta.MATCHES):
            for vi, movement in enumerate(ta.MOVEMENTS):
                for name in ('analyze_spf', 'analyze_rqspf', 'analyze_daxiao'):
                    golden[f'{name}:{ei}:{ci}:{mi}:{vi}'] = getattr(
                        analyzer, name)(match, movement)
dump('analysis', golden)

# ---- okooo parsing ----
from tests.domain import test_basketball_okooo_parsing as to
from src.domain.sports.basketball import okooo_parsing as op

golden = {}
for i, rf in enumerate(to.RFLISTS):
    golden[f'rflist:{i}'] = op.parse_rflist(rf)
for i, hist in enumerate(to.HISTORIES):
    for kind in ('ah', 'ou', 'ml'):
        golden[f'trend:{i}:{kind}'] = op.analyze_line_trend(hist, kind)
for date in to.ScheduleGoldenTests.DATES:
    golden[f'schedule:{date}'] = op.parse_schedule(to.HUNHE_HTML, date)
for kind, html in to.DETAIL_PAGES.items():
    golden[f'books:{kind}'] = op.parse_book_rows(html, kind)
    golden[f'avg:{kind}'] = op.parse_average_row(html, kind)
    golden[f'consensus:{kind}'] = op.consensus_from_books(
        op.parse_book_rows(html, kind), kind)
for kind in ('ml', 'ah', 'ou'):
    golden[f'consensus_empty:{kind}'] = op.consensus_from_books([], kind)
golden['bundle'] = op.build_bundle(to.SAMPLE_MATCH_ID, to.DETAIL_PAGES)
dump('okooo_parsing', golden)

# ---- 500 parsing ----
from tests.domain import test_basketball_parsing as tp
from src.domain.sports.basketball import parsing

golden = {}
for date in tp.RealPageGoldenTests.DATES:
    golden[f'rows:{date}'] = parsing.parse_schedule(tp.JCLQ_HTML, date)
    fetcher = parsing.ScheduleFetcher(transport=lambda url: tp.JCLQ_HTML,
                                      now_fn=lambda: tp.NOW)
    golden[f'fetch:{date}'] = fetcher.fetch(date)
dump('parsing_500', golden)

# ---- records ----
from tests.domain import test_basketball_records as tr
from src.domain.sports.basketball import records as dr

golden = {}
golden = {}
for i, record in enumerate(tr.EvaluateMarketsGoldenTests.RECORDS):
    for home, away in tr.EvaluateMarketsGoldenTests.SCORES:
        golden[f'eval:{i}:{home}:{away}'] = dr.evaluate_markets(record, home, away)
golden['summarize'] = dr.summarize(tr.SummarizeGoldenTests.records())
dump('records', golden)

# ---- prediction ----
from tests.domain import test_basketball_prediction as tpr
from src.domain.sports.basketball.prediction import find_value_bets

golden = {}
for i, case in enumerate(tpr.GenerateGoldenTests.CASES):
    service = tpr._service(recorder=tpr.RecordingRecorder())
    golden[f'payload:{i}'] = service.generate(**case)
golden['payload:empty'] = tpr._service(
    matches=[], recorder=tpr.RecordingRecorder()).generate()
results = tpr._service(recorder=tpr.RecordingRecorder()).generate()['results']
for threshold in (-1.0, 0.0, 0.01, 0.05, 0.2, 0.9):
    golden[f'value:{threshold}'] = find_value_bets(results, threshold)
dump('prediction', golden)

# ---- movement map ----
from tests.domain import test_basketball_movement_map as tmm

golden = {}
for name, matches, source, kwargs in tmm.GoldenTests.CASES:
    deps = tmm._Deps(**kwargs)
    golden[name] = deps.builder()(matches, source, tmm.GoldenTests.DATE)
dump('movement_map', golden)

# ---- odds history tracker ----
from tests.domain import test_basketball_odds_history as toh

golden = {}
for name, matches, history in toh.TrackerGoldenTests.CASES:
    store = toh._memory_store()
    store.save(history)
    tracker = toh._tracker_for(store, matches)
    count = tracker.track('2026-08-27')
    golden[name] = {'count': count, 'history': store.load()}
dump('odds_history', golden)

print('全部生成完成')


# ---- lottery3d 特征层 ----
from tests.domain.numeric.lottery3d.test_features import golden_entries as l3d_entries
from tests.domain.golden import as_comparable

dump('lottery3d_features', {k: as_comparable(v) for k, v in l3d_entries()})


# ---- lottery3d 评分与排名 ----
from tests.domain.numeric.lottery3d.test_scoring import golden_entries as l3d_scoring_entries

dump('lottery3d_scoring', {k: as_comparable(v) for k, v in l3d_scoring_entries()})


# ---- lottery3d 选号 ----
from tests.domain.numeric.lottery3d.test_selection import golden_entries as l3d_sel_entries

dump('lottery3d_selection', {k: as_comparable(v) for k, v in l3d_sel_entries()})


# ---- lottery3d 窗口权重与记录 ----
from tests.domain.numeric.lottery3d.test_records import golden_entries as l3d_rec_entries

dump('lottery3d_records', {k: as_comparable(v) for k, v in l3d_rec_entries()})
