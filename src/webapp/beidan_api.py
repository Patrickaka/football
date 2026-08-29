# -*- coding: utf-8 -*-
"""beidan 接口 handler（mixin）。

接口逻辑已迁至 `src.api.services.beidan`，新旧两个入口共用一份
（判据 11）。这里只剩把方法名接到服务函数上的转发。
"""

from src.api.services import beidan as service


class BeidanApiMixin:

    def _beidan_payload(self, params):
        return service.beidan_payload(params)

    def _beidan_matches_payload(self, params):
        return service.beidan_matches_payload(params)

    def _beidan_value_payload(self, params):
        return service.beidan_value_payload(params)

    def _beidan_history_payload(self, params):
        return service.beidan_history_payload(params)
