"""Elo 数据的存取门面。

把三张表（rating / history / recent_form）封装成与迁移前 kv_store 调用同形的
接口——load() 返回含三部分的 dict，save() 接收三部分——这样
BasketballELORatingSystem 只需替换 _load/_save 两个方法，其余算法逻辑
逐字不动，迁移的改动面压到最小。

**整体保存语义**：save() 用传入的内容完全取代库中已有的，而不是叠加。
recent_form 尤其依赖这一点——它是截断列表（近 N 场），列表变短时旧条目
必须消失，否则会读出一条比实际更长的历史，_form_factor 算出的胜率就是错的。
"""
import logging

from .repository import (
    EloHistoryRepository, EloRatingRepository, EloRecentFormRepository,
)

log = logging.getLogger('domain.basketball.elo')


class EloStore:
    def __init__(self, db):
        self.db = db
        self._ratings = EloRatingRepository(db)
        self._history = EloHistoryRepository(db)
        self._recent_form = EloRecentFormRepository(db)

    def load(self):
        """读回 {'ratings': {...}, 'history': {...}, 'recent_form': {...}}。"""
        ratings = {
            row['team']: row['rating'] for row in self._ratings.find_all()
        }

        history = {}
        for row in self._history.find_all(order_by=['team', 'recorded_at']):
            history.setdefault(row['team'], []).append({
                'rating': row['rating'],
                'date': row['recorded_at'],
                'event': row['event'],
            })

        recent_form = {}
        for row in self._recent_form.find_all(order_by=['team', 'seq']):
            recent_form.setdefault(row['team'], []).append(row['result'])

        return {'ratings': ratings, 'history': history, 'recent_form': recent_form}

    def save(self, ratings, history, recent_form, updated_at=''):
        """整体替换三部分内容。

        先清空再写入：三者都是"当前完整状态"而非增量流水，若只做 upsert，
        被移除的球队或变短的列表会残留在库里。
        """
        self._ratings.delete_all()
        self._history.delete_all()
        self._recent_form.delete_all()

        rating_rows = [
            {'team': team, 'rating': float(rating), 'updated_at': updated_at}
            for team, rating in (ratings or {}).items()
        ]
        self._ratings.insert_many(rating_rows)

        history_rows = []
        for team, entries in (history or {}).items():
            for entry in entries or []:
                history_rows.append({
                    'team': team,
                    'recorded_at': entry.get('date') or '',
                    'rating': float(entry.get('rating') or 0),
                    'event': entry.get('event') or '',
                })
        self._history.insert_many(history_rows)

        form_rows = []
        for team, results in (recent_form or {}).items():
            for seq, result in enumerate(results or []):
                form_rows.append({'team': team, 'seq': seq, 'result': float(result)})
        self._recent_form.insert_many(form_rows)

    def updated_at(self):
        """任取一行的时间戳——save 会给所有行写同一个值。"""
        rows = self._ratings.find_all()
        return rows[0]['updated_at'] if rows else ''
