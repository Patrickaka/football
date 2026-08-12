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

    def test_non_3d_cache_is_unchanged(self):
        self.assertTrue(server._is_cache_payload_current("ssq", {"result": []}))


if __name__ == "__main__":
    unittest.main()
