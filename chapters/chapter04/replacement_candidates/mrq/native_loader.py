"""Load the four unchanged author Torch files without Gym/Atari imports."""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import threading

LOCK = threading.RLock()


def verify_vendor():
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/'vendor_manifest.json').read_text())
    for e in manifest['files']:
        b=(here/e['local_path']).read_bytes()
        h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if len(b)!=e['size_bytes'] or h!=e['git_blob_sha1']:
            raise RuntimeError('MRQ author source changed: '+e['local_path'])
    return manifest['commit']


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_native():
    with LOCK:
        verify_vendor();root=Path(__file__).resolve().parent/'vendor'
        namespace=__package__+'._author_'
        modules={k:_module(namespace+k,root/(k+'.py')) for k in ['buffer','models','utils']}
        absent=object();previous={k:sys.modules.get(k,absent) for k in modules}
        try:
            sys.modules.update(modules)
            core=_module(namespace+'core',root/'MRQ.py')
        finally:
            for k,v in previous.items():
                if v is absent:sys.modules.pop(k,None)
                else:sys.modules[k]=v
        return core


@contextmanager
def replay_class(core,cls):
    # Only construction-time observation wrapper; numerical methods unchanged.
    with LOCK:
        old=core.buffer.ReplayBuffer;core.buffer.ReplayBuffer=cls
        try:yield
        finally:core.buffer.ReplayBuffer=old
