"""Load the application-owned algorithm package without a shared chapter path."""
from pathlib import Path
import importlib
import importlib.util
import sys


METHOD_ROOT = Path(__file__).resolve().parents[2] / 'method'
PACKAGE = 'hongmiao89_method'


def owned_method(name):
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, METHOD_ROOT / '__init__.py',
            submodule_search_locations=[str(METHOD_ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    package = sys.modules[PACKAGE]
    if Path(package.__file__).resolve().parent != METHOD_ROOT:
        raise ImportError('A foreign chapter-six method package is already imported')
    module = importlib.import_module(PACKAGE + '.' + name)
    if Path(module.__file__).resolve().parent != METHOD_ROOT:
        raise ImportError('The chapter-six algorithm escaped its private directory')
    return module
