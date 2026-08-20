import unittest
from unittest.mock import patch

from src.common import kv_store


class KvStoreProductionAuditTests(unittest.TestCase):
    def test_load_with_backend_identifies_mysql(self):
        with patch.object(
            kv_store.db, "query_one",
            return_value={"json_value": '{"production_ready": false}'},
        ):
            value, backend = kv_store.load_with_backend("audit")

        self.assertEqual(backend, "mysql")
        self.assertFalse(value["production_ready"])

    def test_require_mysql_does_not_silently_write_local_fallback(self):
        with patch.object(kv_store.db, "execute", side_effect=RuntimeError("db down")), \
                patch.object(kv_store, "_fallback_save") as fallback_save:
            with self.assertRaises(RuntimeError):
                kv_store.save("audit", {"ok": True}, require_mysql=True)

        fallback_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
