"""校准数据的存取门面。

接口与迁移前的 kv_store 调用同形（load 返回 stats dict、save 接收它），
使 BasketballCalibrator 只需替换 _load/save 两个方法。

整体替换语义：stats 是"当前完整状态"，save 用传入内容取代库中已有的。
"""
import logging

from .repository import CalibrationRepository

log = logging.getLogger('domain.basketball.calibration')

_FIELDS = ('count', 'weighted_count', 'success', 'weighted_success',
           'predicted_sum', 'weighted_predicted_sum')


class CalibrationStore:
    def __init__(self, db):
        self.db = db
        self._repo = CalibrationRepository(db)

    def load(self):
        """读回 {bucket: {六个统计字段}}。"""
        return {
            row['bucket']: {field: row[field] for field in _FIELDS}
            for row in self._repo.find_all()
        }

    def save(self, stats):
        """整体替换。先清空再写入——被移除的分桶不应残留。"""
        self._repo.delete_all()
        rows = []
        for bucket, values in (stats or {}).items():
            row = {'bucket': bucket}
            for field in _FIELDS:
                value = values.get(field, 0)
                row[field] = int(value) if field in ('count', 'success') else float(value)
            rows.append(row)
        self._repo.insert_many(rows)
