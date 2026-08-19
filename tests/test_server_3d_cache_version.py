import unittest
from unittest.mock import patch

import server


class Server3DCacheVersionTests(unittest.TestCase):
    def test_old_3d_prediction_version_is_rejected(self):
        module = type("Lottery3D", (), {"PREDICTOR_VERSION": "3d-current"})()
        with patch.object(server, "_get_lottery3d_module", return_value=module):
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


if __name__ == "__main__":
    unittest.main()
