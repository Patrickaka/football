"""把 kl8 的策略试验记录从 18MB JSON 文件迁进 foundation/store。

线上 `data/kl8_strategy_trials.json` 是 18MB、23564 条。真正的问题不是它大，
而是**每新增一条记录就整文件重写一遍**——一次策略验证产生成百上千条试验，
这个文件就被重写成百上千次。

用法：
    python3 -m scripts.migrate.kl8_trials_to_store --dry-run
    python3 -m scripts.migrate.kl8_trials_to_store
    python3 -m scripts.migrate.kl8_trials_to_store --verify-only

原 JSON 文件不删。代码可以推倒，数据不可丢。
"""
import argparse
import json
import logging
from pathlib import Path

from src.domain.numeric.repository import create_all
from src.domain.numeric.trial_store import KEY_FIELDS, TrialStore

log = logging.getLogger('migrate.kl8_trials')

GAME = 'kl8'
# 一批的写入条数。整批 23564 条一次 executemany 会把参数拼成一条极长的
# SQL，MySQL 的 max_allowed_packet 顶不住。
BATCH_SIZE = 1000


def read_trials(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return raw if isinstance(raw, list) else []


def _key(trial):
    return tuple(str(trial.get(f, '')) for f in KEY_FIELDS)


def migrate(path, db, dry_run=False, batch_size=BATCH_SIZE):
    trials = read_trials(path)
    stats = {'read': len(trials), 'unique': len({_key(t) for t in trials}),
             'written': 0}
    if dry_run:
        return stats

    store = TrialStore(db, game=GAME)
    for start in range(0, len(trials), batch_size):
        stats['written'] += store.append_many(trials[start:start + batch_size])
    return stats


def verify(path, db):
    """比对条数与抽样内容。

    只比条数会漏掉「条数对得上但 p 值被改过」——而 p 值正是 FDR 校正的输入，
    错了会安静地把一个无效策略判成显著。
    """
    stored = {_key(t): t for t in TrialStore(db, game=GAME).load()}

    problems = []
    # 去重取**先到先得**，与写入端和旧实现一致。用字典推导会变成「后者覆盖
    # 前者」，那样任何含重复键的文件都会被报成不一致——错的是校验，不是数据。
    expected = {}
    for trial in read_trials(path):
        expected.setdefault(_key(trial), trial)
    if len(expected) != len(stored):
        problems.append(f'条数不一致：文件去重后 {len(expected)} 库 {len(stored)}')

    for key, trial in expected.items():
        found = stored.get(key)
        if found is None:
            problems.append(f'库中缺少 {key}')
            continue
        for field in ('raw_p_value', 'fdr_adjusted_p', 'validation_lift',
                      'window_size', 'feature_weights', 'model_weights'):
            if _differs(found.get(field), trial.get(field)):
                problems.append(
                    f'{key} 的 {field} 不一致：文件 {trial.get(field)!r} '
                    f'库 {found.get(field)!r}')
    return problems


def _differs(left, right):
    if isinstance(left, float) or isinstance(right, float):
        if left is None or right is None:
            return left is not right
        return abs(float(left) - float(right)) > 1e-9
    return left != right


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='kl8 策略试验记录迁移')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--file', default=None)
    args = parser.parse_args()

    from src.common.paths import data_path
    from src.foundation.store import Database, make_engine, mysql_url_from_env

    path = args.file or data_path('kl8_strategy_trials.json')
    db = Database(make_engine(mysql_url_from_env()))
    create_all(db)

    if args.verify_only:
        _report(verify(path, db))
        return

    stats = migrate(path, db, dry_run=args.dry_run)
    log.info('读取 %d 条（去重后 %d），写入 %d 条%s', stats['read'], stats['unique'],
             stats['written'], '（dry-run，未写入）' if args.dry_run else '')
    if not args.dry_run:
        _report(verify(path, db))


def _report(problems):
    if problems:
        for item in problems[:20]:
            log.error(item)
        if len(problems) > 20:
            log.error('……另有 %d 项不一致未列出', len(problems) - 20)
        raise SystemExit(1)
    log.info('校验通过，文件与库一致')


if __name__ == '__main__':
    main()
