"""Every module and public function or class has a docstring, so the API
reference in docs/ never shows bare names."""
import importlib
import inspect
import os
import pkgutil
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mpas_stat  # noqa: E402


def _modules():
    for info in pkgutil.walk_packages(mpas_stat.__path__, "mpas_stat."):
        try:
            yield importlib.import_module(info.name)
        except ImportError:          # optional dependency missing
            continue


class TestDocstrings(unittest.TestCase):
    def test_modules_have_docstrings(self):
        missing = [m.__name__ for m in _modules()
                   if not m.__file__.endswith("__init__.py")      # packages may be bare
                   and not (m.__doc__ or "").strip()]
        self.assertEqual(missing, [])

    def test_public_functions_have_docstrings(self):
        missing = []
        for mod in _modules():
            for name, obj in vars(mod).items():
                if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                    continue
                if (inspect.isfunction(obj) or inspect.isclass(obj)) and not (obj.__doc__ or "").strip():
                    missing.append(f"{mod.__name__}.{name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
