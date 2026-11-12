"""Paths for the separate source release and the self-contained experiment release.

Only paths are resolved here. Importing this module never loads a model, an
image dataset, a simulator or a training runtime. No filesystem aliases exist.
"""
from pathlib import Path
import json, os
from functools import lru_cache

def project_root():
    return Path(__file__).resolve().parent

def final_data_root():
    a=os.environ.get('RESOURCE_ALLOCATION_DATA_ROOT')
    b=os.environ.get('FINAL_DATA_ROOT')
    if a and b and Path(a).expanduser().resolve()!=Path(b).expanduser().resolve():
        raise ValueError('RESOURCE_ALLOCATION_DATA_ROOT and FINAL_DATA_ROOT must agree')
    if a or b:return Path(a or b).expanduser().resolve()
    local=project_root()
    if (local/'INDEX.json').is_file() and (local/'registry/models.json').is_file():return local
    return Path('/hy-tmp/resource_allocation_final')

@lru_cache(maxsize=8)
def _mapping(root):
    file=Path(root)/'registry/path_map.json'
    if not file.exists():file=project_root()/'registry/path_map.json'
    if not file.exists():return {}
    doc=json.loads(file.read_text())
    value=doc.get('mapping',doc.get('path_map',doc))
    if not isinstance(value,dict):raise ValueError('path_map requires a mapping object')
    return {str(k):str(v) for k,v in value.items() if isinstance(v,str)}

@lru_cache(maxsize=8)
def _source_mapping(root):
    file=Path(root)/'registry/source_path_map.json'
    if not file.is_file():return {}
    doc=json.loads(file.read_text())
    return doc.get('mapping',doc)

def _mapped(text,mapping):
    if text in mapping:return Path(mapping[text])
    path=Path(text)
    for parent in path.parents:
        key=str(parent)
        if key in mapping:return Path(mapping[key])/path.relative_to(parent)
    return None

def chapter_data_root(chapter):
    number=int(str(chapter).removeprefix('chapter'))
    key=f'chapter{number:02d}';root=final_data_root()
    index=root/'INDEX.json'
    if index.is_file():
        item=json.loads(index.read_text()).get('chapters',{}).get(key,{})
        if item.get('directory'):return root/item['directory']
    modern=root/'chapters'/key
    return modern if modern.is_dir() else root/'experiments'/key

def data_path(value):
    candidate=Path(value).expanduser()
    if candidate.is_absolute():return resolve_path(candidate)
    parts=candidate.parts
    if len(parts)>1 and parts[0]=='experiments' and parts[1].startswith('chapter'):
        return chapter_data_root(parts[1])/Path(*parts[2:])
    root=final_data_root();mapped=_mapped(str(candidate),_mapping(str(root)))
    return root/(mapped if mapped is not None else candidate)

def resolve_path(value):
    candidate=Path(value).expanduser();text=str(candidate)
    root=final_data_root();code=project_root()
    # Prefer the delivered local source capsule for historic code filenames.
    source=_mapped(text,_source_mapping(str(code)))
    if source is not None and (code/source).exists():return code/source
    old_project='/root/gamewar/resource_allocation_rl'
    if text==old_project:return code
    if text.startswith(old_project+'/'):
        relative=Path(text[len(old_project)+1:])
        local=code/relative
        if local.exists():return local
        if relative.parts[0] in ('datasets','experiments'):return data_path(relative)
    inside_root=candidate.is_absolute() and candidate.resolve().is_relative_to(root)
    if inside_root and candidate.exists():return candidate
    mapping=_mapping(str(root));mapped=_mapped(text,mapping)
    if mapped is not None:return root/mapped
    # Infer a directory only if all retained descendants agree on its new root.
    inferred=set();prefix=text.rstrip('/')+'/'
    for old,new in mapping.items():
        if old.startswith(prefix):
            tail=Path(old[len(prefix):]).parts;target=Path(new)
            for unused in tail:target=target.parent
            inferred.add(str(target))
    if len(inferred)==1:return root/next(iter(inferred))
    if not candidate.is_absolute() and candidate.parts and candidate.parts[0]=='experiments':return data_path(candidate)
    if text=='/hy-tmp':return root
    # A relocated release may create fresh outputs inside its own data root.
    if inside_root:return candidate
    if text.startswith('/hy-tmp/') or text.startswith('/root/resource_allocation_run_storage_20261101/'):
        raise FileNotFoundError('Retired absolute path is not registered in this release: '+text)
    return candidate

def eurosat_root():return final_data_root()/'datasets/eurosat_rgb'

def chapter_source_paths(chapter):
    chapter=int(chapter);base=project_root()/'chapters'/f'chapter{chapter:02d}'
    paths=[project_root(),base]
    if chapter==4:
        foundation=base/'foundation'
        paths += [foundation,foundation/'chapter4_credit_assignment',
                  foundation/'chapter4_sample_efficiency_v5',foundation/'chapter4_reuse_optimization',
                  base/'chapter4_full_spend_20261104/source/env']
    if chapter==5:
        source=base/'_internal/src'
        paths += [source,source/'chapter4_credit_assignment',
                  source/'chapter4_sample_efficiency_v5',source/'chapter5_multiobjective']
    return [p for p in paths if p.is_dir()]

def chapter_environment(chapter):
    env=dict(os.environ)
    env['RESOURCE_ALLOCATION_PROJECT_ROOT']=str(project_root())
    env['RESOURCE_ALLOCATION_DATA_ROOT']=str(final_data_root())
    env['FINAL_DATA_ROOT']=str(final_data_root())
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONPATH']=os.pathsep.join(map(str,chapter_source_paths(chapter)))
    env['CH5_FULL_SPEND_DATA_ROOT']=str(eurosat_root())
    if int(chapter)==5:env['CH5_WORKSPACE']=str(project_root()/'chapters/chapter05/_internal/src')
    return env
