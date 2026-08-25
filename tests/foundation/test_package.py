import unittest


class FoundationPackageTests(unittest.TestCase):
    def test_foundation_package_importable(self):
        import src.foundation  # noqa: F401

    def test_required_dependencies_available(self):
        import fastapi  # noqa: F401
        import redis  # noqa: F401
        import sqlalchemy  # noqa: F401
        import uvicorn  # noqa: F401

    def test_sqlalchemy_is_v2(self):
        import sqlalchemy
        self.assertGreaterEqual(int(sqlalchemy.__version__.split('.')[0]), 2)


if __name__ == '__main__':
    unittest.main()
