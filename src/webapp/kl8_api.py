# -*- coding: utf-8 -*-
"""kl8 接口 handler（mixin）。

接口逻辑已迁至 `src.api.services.kl8`，新旧两个入口共用一份
（判据 11）。这里只剩把方法名接到服务函数上的转发。
"""

from src.api.services import kl8 as service


class KL8ApiMixin:

    def _kl8_payload(self):
        return service.kl8_payload()

    def _kl8_latest_issue(self):
        return service.kl8_latest_issue()

    def _kl8_refresh_payload(self):
        return service.kl8_refresh_payload()

    def _kl8_fetch_payload(self):
        return service.kl8_fetch_payload()

    def _kl8_exclude_recalculate_payload(self, params):
        return service.kl8_exclude_recalculate_payload(params)

    def _kl8_snapshots_payload(self):
        return service.kl8_snapshots_payload()

    def _kl8_records_payload(self):
        return service.kl8_records_payload()

    def _kl8_backfill_settlements(self, records):
        return service.kl8_backfill_settlements(records)

    def _kl8_rebuild_stale_settlements(self, records):
        return service.kl8_rebuild_stale_settlements(records)

    def _kl8_settle_payload(self, params):
        return service.kl8_settle_payload(params)

    def _kl8_backtest_payload(self, params):
        return service.kl8_backtest_payload(params)

    def _kl8_parameter_search_payload(self, params):
        return service.kl8_parameter_search_payload(params)

    def _parse_kl8_parameter_search_options(self, params):
        return service.parse_kl8_parameter_search_options(params)

    def _start_kl8_parameter_search_job(self, options):
        return service.start_kl8_parameter_search_job(options)

    def _kl8_parameter_search_start_payload(self, params):
        return service.kl8_parameter_search_start_payload(params)

    def _kl8_parameter_search_status_payload(self, params):
        return service.kl8_parameter_search_status_payload(params)

    def _kl8_integrity_payload(self):
        return service.kl8_integrity_payload()

    def _kl8_conflicts_payload(self):
        return service.kl8_conflicts_payload()

    def _kl8_activate_payload(self, params):
        return service.kl8_activate_payload(params)
