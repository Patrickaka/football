"""开奖记录的存取。

与其它仓储的**整表替换**语义刻意不同：开奖历史是只增不改的流水，
2048 期里改一条就重写全部，是文件时代的做法——正是要摆脱的那个。
所以这里按期号增量写入。

**已入库的号码不允许被后来的抓取覆盖。** 开奖结果不会变，同一期出现两组
号码只能是某一边错了；自动覆盖等于用未经核实的数据抹掉已经存在的。
"""
import json
import logging
from dataclasses import dataclass

from src.domain.numeric.draw import Draw, DrawConflict, checksum_of
from src.domain.numeric.repository import DrawRepository

log = logging.getLogger('domain.numeric.draw_store')


class DrawStore:
    def __init__(self, db, game):
        self.db = db
        self.game = game
        self._repo = DrawRepository(db)

    def load(self):
        """按期号倒序返回。线上文件就是新→旧，下游多处直接取第一条。"""
        rows = self._repo.find_by(game=self.game, order_by='issue')
        draws = [_row_to_draw(row) for row in rows]
        return [d for d in reversed(draws) if d is not None]

    def latest(self):
        draws = self.load()
        return draws[0] if draws else None

    def count(self):
        return len(self._repo.find_by(game=self.game))

    def save(self, draws):
        """增量写入，返回冲突列表。

        已存在的期号一律跳过——包括号码相同、只有溯源字段不同的情况：
        先入库的那条才是真实抓取过程的记录。
        """
        draws = [d for d in draws if d is not None]
        if not draws:
            return []

        stored = {d.issue: d for d in self.load()}
        conflicts = []
        rows = []
        for draw in draws:
            current = stored.get(draw.issue)
            if current is None:
                rows.append(_draw_to_row(self.game, draw))
                stored[draw.issue] = draw
            elif current.numbers != draw.numbers:
                log.error('期号 %s 号码冲突：库内 %s 新来 %s，保留库内值待人工确认',
                          draw.issue, current.numbers, draw.numbers)
                conflicts.append(DrawConflict(issue=draw.issue, kept=current,
                                              rejected=draw))

        self._repo.insert_many(rows)
        return conflicts

    def find_corrupted(self):
        """对账：报出所有「这一行不可信」的记录。

        两类问题都要报，而不只是校验码不符：

        - **号码本身不合规**（个数不对、重复、越界）。这类行在 load() 里
          会被跳过，于是「损坏」看起来就成了「不存在」——比校验码不符更
          隐蔽，因为连痕迹都没有。
        - **校验码与号码对不上**，说明号码在某个环节被改过。

        没有对账入口的话，checksum 这一列存了也没人看。
        """
        issues = []
        for row in self._repo.find_by(game=self.game, order_by='issue'):
            draw = _row_to_draw(row)
            if draw is None:
                issues.append(DrawIntegrityIssue(
                    issue=str(row.get('issue', '')), reason='invalid_numbers',
                    stored_numbers=row.get('numbers')))
            elif draw.checksum != checksum_of(draw.numbers):
                issues.append(DrawIntegrityIssue(
                    issue=draw.issue, reason='checksum_mismatch',
                    stored_numbers=list(draw.numbers),
                    stored_checksum=draw.checksum,
                    expected_checksum=checksum_of(draw.numbers)))
        return issues


@dataclass(frozen=True)
class DrawIntegrityIssue:
    """一行不可信的开奖记录。带上原始内容，免得还要回库里再查一次。"""

    issue: str
    reason: str
    stored_numbers: object = None
    stored_checksum: str = ''
    expected_checksum: str = ''


def _row_to_draw(row):
    try:
        numbers = json.loads(row['numbers'])
    except (TypeError, ValueError):
        log.error('期号 %s 的号码无法解析: %r', row.get('issue'), row.get('numbers'))
        return None
    draw = Draw.parse({
        'issue': row['issue'],
        'numbers': numbers,
        'date': row.get('date') or '',
        'source': row.get('source') or '',
        'fetched_at': row.get('fetched_at') or '',
        # 原样带出，不重算——重算等于让校验码永远自洽，那就失去意义了
        'checksum': row.get('checksum') or '',
    })
    if draw is None:
        log.error('期号 %s 的号码不合规，已跳过: %r', row.get('issue'), numbers)
    return draw


def _draw_to_row(game, draw):
    return {
        'game': game,
        'issue': draw.issue,
        'numbers': json.dumps(list(draw.numbers), separators=(',', ':')),
        'date': draw.date,
        'source': draw.source,
        'fetched_at': draw.fetched_at,
        'checksum': draw.checksum,
    }
