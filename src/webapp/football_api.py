# -*- coding: utf-8 -*-
"""football 接口 handler（mixin）。

接口逻辑已迁至 `src.api.services.football`，新旧两个入口共用一份
（判据 11）。这里只剩把方法名接到服务函数上的转发。
"""

from src.api.services import football as service


class FootballApiMixin:

    def _try_generate_report(self, rel):
        return service.try_generate_report(rel)

    def _matches_payload(self):
        return service.matches_payload()

    def _match_from_params(self, params):
        return service.match_from_params(params)

    def _match_from_json(self):
        return service.match_from_json()

    def _analyze_one(self, match, force_refresh=False):
        return service.analyze_one(match, force_refresh)

    def _predict_payload(self, params):
        return service.predict_payload(params)

    def _predict_batch_payload(self, body):
        return service.predict_batch_payload(body)

    def _football_clear_cache_payload(self):
        return service.football_clear_cache_payload()

    def _prepare_ml_history_data_payload(self):
        return service.prepare_ml_history_data_payload()

    def _football_diagnostics_payload(self, params):
        return service.football_diagnostics_payload(params)

    def _football_review_payload(self, params):
        return service.football_review_payload(params)

    def _football_professional_status_payload(self):
        return service.football_professional_status_payload()

    def _calibrate_payload(self, params):
        return service.calibrate_payload(params)

    def _calibrate_list_payload(self):
        return service.calibrate_list_payload()

    def _calibrate_clear_payload(self):
        return service.calibrate_clear_payload()

    def _backtest_payload(self, params):
        return service.backtest_payload(params)

    def _threshold_payload(self):
        return service.threshold_payload()

    def _model_status_payload(self):
        return service.model_status_payload()

    def _backtest_stats_payload(self, params):
        return service.backtest_stats_payload(params)

    def _predictions_payload(self):
        return service.predictions_payload()

    def _predictions_export_payload(self):
        return service.predictions_export_payload()

    def _sync_status_payload(self):
        return service.sync_status_payload()

    def _sync_trigger_payload(self):
        return service.sync_trigger_payload()

    def _sync_hide_failed_payload(self):
        return service.sync_hide_failed_payload()

    def _serve_report_file(self, path):
        """提供 reports/ 目录下的静态报告文件（HTML/JSON）。"""
        rel = path[len('/reports/'):].lstrip('/')
        if not rel or '..' in rel or rel.startswith('.'):
            return self._send_json_error(403, 'Forbidden')
        file_path = _jobs_mod.REPORTS_DIR / rel
        try:
            file_path = file_path.resolve()
            reports_root = _jobs_mod.REPORTS_DIR.resolve()
            if not str(file_path).startswith(str(reports_root)):
                return self._send_json_error(403, 'Forbidden')
        except Exception:
            return self._send_json_error(404, 'Not Found')
        if rel.startswith('football_bayes_') and rel.endswith('.html'):
            # ensure_football_report performs a cheap schema/odds check and
            # regenerates stale report layouts on first access.
            generated = self._try_generate_report(rel)
            if generated and os.path.exists(generated):
                file_path = Path(generated)
        if not file_path.exists() or not file_path.is_file():
            # 报告文件不存在 → 若可生成则按需现生成（生产环境无需手动跑脚本）
            generated = self._try_generate_report(rel)
            if generated and os.path.exists(generated):
                file_path = Path(generated)
            else:
                return self._send_json_error(404, 'Not Found')
        content_type = 'text/html; charset=utf-8'
        if file_path.suffix.lower() == '.json':
            content_type = 'application/json; charset=utf-8'
        try:
            with open(file_path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._log.error('读取报告文件失败: %s', file_path, exc_info=True)
            self._send_json_error(500, f'读取报告失败: {e}')
