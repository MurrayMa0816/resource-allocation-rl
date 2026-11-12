"""Top-level compatibility glue: Chapter2 native work uses its canonical facade."""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
import importlib.util
import os
from pathlib import Path
import sys
PROJECT=Path(os.environ.get('RESOURCE_ALLOCATION_PROJECT_ROOT',str(_resolve_path('/root/gamewar/resource_allocation_rl')))).resolve()
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))


def _chapter_selector(arguments):
    values=[]
    for i,item in enumerate(arguments):
        if item=='--chapter' and i+1<len(arguments):values.append(arguments[i+1])
        elif item.startswith('--chapter='):values.append(item.split('=',1)[1])
    if len(set(values))>1:raise ValueError('Conflicting --chapter values')
    return values[0] if values else None


def _chapter2(arguments):
    name='_chapter02_canonical_launcher_router'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PROJECT/'chapters/chapter02/launcher.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].main(arguments)


def _checkpoint_chapter(arguments):
    checkpoint=None
    for i,item in enumerate(arguments):
        if item=='--checkpoint' and i+1<len(arguments):checkpoint=arguments[i+1]
        elif item.startswith('--checkpoint='):checkpoint=item.split('=',1)[1]
    if checkpoint is None:return None
    import torch
    identity=torch.load(Path(checkpoint).expanduser().resolve(),map_location='cpu',weights_only=False)['identity']
    return identity.get('config',{}).get('chapter')


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    # Baselines and all Chapter3/other commands remain on the original shared
    # path; the canonical Chapter2 launcher may forward baselines here safely.
    if arguments and arguments[0]=='chapter2' and len(arguments)>1 and arguments[1] in ('train','evaluate'):
        if _chapter_selector(arguments[2:]) is not None:
            raise ValueError('Use either the chapter prefix or --chapter, not both')
        return _chapter2(arguments[1:])
    if arguments and arguments[0] in ('train','evaluate'):
        selected=_chapter_selector(arguments[1:])
        if selected=='2':return _chapter2(arguments)
        if arguments[0]=='evaluate' and selected is None and '-h' not in arguments and '--help' not in arguments:
            if _checkpoint_chapter(arguments[1:])==2:return _chapter2(arguments)
    from allocation23.cli import main as shared_main
    return shared_main(arguments)

if __name__=='__main__':main()
