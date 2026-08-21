import unittest
from unittest.mock import Mock, patch
import time

import server
from src.webapp import caching as webapp_caching
from src.webapp import jobs as webapp_jobs
from src.webapp import lazy_modules as webapp_lazy


class Server3DCacheVersionTests(unittest.TestCase):
    def test_old_3d_prediction_version_is_rejected(self):
        module = type("Lottery3D", (), {"PREDICTOR_VERSION": "3d-current"})()
        with patch.object(webapp_lazy, "_get_lottery3d_module", return_value=module):
            self.assertFalse(
                server._is_cache_payload_current("3d", {"version": "3d-old"})
            )
            self.assertTrue(
                server._is_cache_payload_current("3d", {"version": "3d-current"})
            )

    def test_ssq_cache_also_checks_version(self):
        # ssq 与 3d 同样按代码版本校验缓存（v3.1 起加入）
        import src.ssq as ssq

        self.assertTrue(
            server._is_cache_payload_current(
                "ssq", {"version": ssq.SSQ_PREDICTION_VERSION}
            )
        )
        self.assertFalse(server._is_cache_payload_current("ssq", {"result": []}))

    def test_other_non_versioned_cache_is_unchanged(self):
        self.assertTrue(server._is_cache_payload_current("kl8", {"result": []}))

    def test_normal_refresh_fetches_once_and_skips_weight_recalculation(self):
        rule = Mock()
        fresh = [("2026220", "2026-08-20", (1, 2, 3))]
        rule.fetch_data.return_value = fresh
        rule.run_prediction.return_value = {
            "period": "2026220", "total_periods": 1, "version": "v",
        }
        old_entries = {
            key: dict(server._CACHE[key]) for key in ("3d", "3d_ml", "3d_data")
        }
        server._CACHE["3d"]["data"] = {"period": "2026219"}
        server._CACHE["3d_ml"]["data"] = None
        try:
            with patch.object(webapp_lazy, "_get_lottery3d_module", return_value=rule), \
                    patch.object(webapp_caching, "_persist_cache"), \
                    patch.object(webapp_jobs, "_set_lottery_background_job"):
                server._run_3d_refresh_job("job", enable_backtest=False)
        finally:
            for key, entry in old_entries.items():
                server._CACHE[key].update(entry)
        rule.fetch_data.assert_called_once_with(force_refresh=True)
        kwargs = rule.run_prediction.call_args.kwargs
        self.assertEqual(kwargs["data"], fresh)
        self.assertFalse(kwargs["force_refresh"])
        self.assertFalse(kwargs["compute_weights"])
        self.assertFalse(kwargs["enable_backtest"])
        self.assertFalse(kwargs["train_ml_if_stale"])

    def test_ml_endpoint_reuses_current_period_training_cache(self):
        data_entry = server._CACHE["3d_data"]
        old_entry = dict(data_entry)
        data_entry.update({
            "data": [("2026220", "2026-08-20", (1, 2, 3))],
            "timestamp": time.time(),
        })
        cached = {
            "recommendations": [
                {"num": "123", "model_score": .8},
                {"num": "456", "model_score": .4},
            ],
            "model_type": "cached", "total_samples": 10,
        }
        rule = Mock()
        rule.is_ml_prediction_cache_valid.return_value = True
        ml = Mock()
        ml.load_ml_cache.return_value = cached
        try:
            with patch.object(webapp_lazy, "_get_lottery3d_module", return_value=rule), \
                    patch.object(webapp_lazy, "_get_lottery3d_ml_module", return_value=ml), \
                    patch.object(webapp_lazy, "predict_current") as train, \
                    patch.object(webapp_caching, "_serve_cached", return_value=({"zhixuan": []}, None)):
                result = server._compute_3d_ml()
        finally:
            data_entry.update(old_entry)
        train.assert_not_called()
        self.assertTrue(result["cache_reused"])
        self.assertEqual(result["base_period"], "2026220")
        self.assertEqual(len(result["recommendations"]), 2)


if __name__ == "__main__":
    unittest.main()
