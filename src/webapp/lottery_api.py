# -*- coding: utf-8 -*-
"""彩票接口 handler（mixin）。

接口逻辑已迁至 `src.api.services.lottery`，新旧两个入口共用一份
（判据 11）。这里只剩把方法名接到服务函数上的转发——
**默认值要原样保留**，把 `(self, params=None)` 写成 `(self, params)`
会让原本可以不传参的调用方直接 TypeError。
"""

from src.api.services import lottery as service


class LotteryApiMixin:

    def _lottery_3d_payload(self):
        return service.lottery_3d_payload()

    def _ssq_payload(self):
        return service.ssq_payload()

    def _ssq_refresh_payload(self):
        return service.ssq_refresh_payload()

    def _lottery_3d_refresh_payload(self, params=None):
        return service.lottery_3d_refresh_payload(params)

    def _lottery_3d_ml_payload(self):
        return service.lottery_3d_ml_payload()

    def _lottery_payload(self):
        return service.lottery_payload()

    def _lottery_refresh_payload(self, params=None):
        return service.lottery_refresh_payload(params)

    def _lottery_task_status_payload(self):
        return service.lottery_task_status_payload()

    def _lottery_recommend_payload(self, params):
        return service.lottery_recommend_payload(params)

    def _lottery_rank_payload(self, params):
        return service.lottery_rank_payload(params)

    def _lottery_ensemble_payload(self):
        return service.lottery_ensemble_payload()

    def _lottery_cycles_payload(self):
        return service.lottery_cycles_payload()

    def _lottery_contribution_payload(self):
        return service.lottery_contribution_payload()

    def _lottery_backtest_payload(self, params):
        return service.lottery_backtest_payload(params)

    def _lottery_fetch_payload(self):
        return service.lottery_fetch_payload()

    def _lottery_ml_payload(self):
        return service.lottery_ml_payload()

    def _lottery_ml_refresh_payload(self):
        return service.lottery_ml_refresh_payload()
