"""Chapter2-only pinned native source; no shared-package replacement."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent


def registration():
    value=json.loads((ROOT/'NATIVE_BACKEND.json').read_text())
    if value['chapter']!=2 or value['default_backend']!='native_random_head':
        raise ValueError('Invalid Chapter2 native backend registration')
    return value


def source_root(name=None):
    data=registration();name=data['default_backend'] if name is None else name
    if name not in data['sources']:
        raise ValueError('Unknown Chapter2 native source')
    record=data['sources'][name]
    local=_project_root()/'frozen/chapter02'/name
    if local.is_dir():source=local.resolve(strict=True)
    else:
        import portable_paths as _paths
        locate=getattr(_paths,'data_path',_resolve_path)
        source=locate(record.get('data_relative_root',record['root'])).resolve(strict=True)
    if not any(source.is_relative_to(base) for base in (_project_root(), _data_root())):
        raise ValueError('Chapter2 native source must be an actual final code/data file')
    for relative,expected in record['source_sha256'].items():
        with (source/relative).open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:
            raise ValueError('Pinned Chapter2 source changed: '+relative)
    return source


def source_for_identity(identity):
    if identity.get('config',{}).get('chapter')!=2:
        raise ValueError('Only Chapter2 scientific identities are supported')
    mapping=identity.get('source_sha256')
    matches=[name for name,row in registration()['sources'].items() if mapping==row['source_sha256']]
    if len(matches)!=1:
        raise ValueError('Checkpoint source is not registered; use its original frozen execution entry')
    source_root(matches[0])
    return matches[0]


def methods():
    """Load the pinned package under a private namespace, preserving allocation23."""
    source=source_root();alias=registration()['private_factory_package']
    if alias not in sys.modules:
        spec=importlib.util.spec_from_file_location(alias,source/'allocation23/__init__.py',
            submodule_search_locations=[str(source/'allocation23')])
        if spec is None or spec.loader is None:
            raise ImportError('Cannot load the registered Chapter2 package')
        module=importlib.util.module_from_spec(spec);sys.modules[alias]=module
        try:spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(alias,None)
            raise
    module=importlib.import_module(alias+'.methods')
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ImportError('Chapter2 private package points to a different source')
    return module
