# -*- coding: utf-8 -*-
"""篮球接口 handler（mixin）。

接口逻辑已迁至 `src.api.services.basketball`，新旧两个入口共用一份
（判据 11）。这里只剩把方法名接到服务函数上的转发。
"""

from src.api.services import basketball as service


class BasketballApiMixin:

    def _basketball_payload(self, params):
        return service.basketball_payload(params)

    def _basketball_matches_payload(self, params):
        return service.basketball_matches_payload(params)

    def _basketball_value_payload(self, params):
        return service.basketball_value_payload(params)

    def _basketball_track_payload(self, params):
        return service.basketball_track_payload(params)

    def _basketball_movement_payload(self, params):
        return service.basketball_movement_payload(params)
