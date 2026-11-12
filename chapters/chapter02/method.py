"""Chapter2 fixed-total BLoAR: default random output heads in all four arms."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

VARIANTS=('A_FULL','A_RANDOM_MATCH','A_SOFTMAX','A_DIRECT')


def _backend():
    name='_chapter02_native_backend_router'
    if name not in sys.modules:
        spec=spec_from_file_location(name,Path(__file__).with_name('native_backend.py'))
        module=module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def _chapter2(spec):
    if getattr(spec,'chapter',None)!=2:
        raise ValueError('Chapter2 methods require a chapter2 public specification')


def make_actor(spec,seed=0,*,variant='A_FULL'):
    """Preserve the public spec/signature/state_dict; use the pinned native head rule."""
    _chapter2(spec)
    if variant not in VARIANTS:
        raise ValueError('Choose a registered chapter2 spatial variant')
    return _backend().methods().make_actor(spec,'A',2,seed,variant=variant,temporal='fixed')


def make_value(spec,seed=0):
    _chapter2(spec)
    return _backend().methods().make_value(spec,seed)
