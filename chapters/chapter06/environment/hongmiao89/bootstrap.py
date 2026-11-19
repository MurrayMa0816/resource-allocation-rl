"""Pin imports to the unchanged scientific dependency closure."""
from pathlib import Path
import hashlib
import importlib
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
import json
import sys


OWNED = frozenset(('irrigation_ch6', 'allocation23', 'allocation4',
                   'allocation_morl', 'allocation_efficiency', 'ch6_full_spend'))


def _check_loaded(root):
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in OWNED or module is None:
            continue
        filename = getattr(module, '__file__', None)
        paths = list(getattr(module, '__path__', ()))
        if filename is not None and not Path(filename).resolve().is_relative_to(root):
            raise RuntimeError(('A different scientific dependency is already imported', name, filename))
        if any(not Path(path).resolve().is_relative_to(root) for path in paths):
            raise RuntimeError(('A different scientific package path is already imported', name))


class _PrivateScientificImports(MetaPathFinder):
    """Resolve owned scientific modules independently of legacy sys.path edits."""

    def __init__(self, root, registered):
        self._ch6_private_root = root
        self.registered = frozenset(registered)

    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split('.')
        if parts[0] not in OWNED:
            return None
        parent = self._ch6_private_root.joinpath(*parts[:-1])
        spec = PathFinder.find_spec(fullname, [str(parent)], target)
        if spec is None:
            raise ModuleNotFoundError('Missing registered private dependency: ' + fullname,
                                      name=fullname)
        if spec.origin is not None:
            origin = Path(spec.origin).resolve()
            if (not origin.is_relative_to(self._ch6_private_root)
                    or str(origin.relative_to(self._ch6_private_root)) not in self.registered):
                raise ImportError('Unregistered scientific source: ' + fullname)
        if spec.submodule_search_locations is not None:
            if any(not Path(location).resolve().is_relative_to(self._ch6_private_root)
                   for location in spec.submodule_search_locations):
                raise ImportError('Scientific package escaped private source: ' + fullname)
        return spec


def install():
    contract = json.loads(Path(__file__).with_name('dependency_manifest.json').read_text())
    root = (Path(__file__).resolve().parent / contract['frozen_source_relative']).resolve()
    for relative, expected in contract['files_sha256'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(('Frozen dependency changed', relative))
    _check_loaded(root)
    if not any(getattr(finder, '_ch6_private_root', None) == root for finder in sys.meta_path):
        sys.meta_path.insert(0, _PrivateScientificImports(root, contract['files_sha256']))
    # The original package initializers insert historical ROOT paths. Preload
    # their private counterparts, then restore the caller's path unchanged.
    original_path = list(sys.path)
    try:
        for name in ('allocation23', 'allocation4', 'allocation_efficiency',
                     'allocation_morl', 'irrigation_ch6'):
            importlib.import_module(name)
        _check_loaded(root)
    finally:
        sys.path[:] = original_path
    return root
