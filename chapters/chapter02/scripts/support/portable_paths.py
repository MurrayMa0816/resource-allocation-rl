"""Resolve current chapter-owned files and recorded historical names, without science imports."""
from pathlib import Path
from functools import lru_cache
import json
import os


def project_root():
    return Path(__file__).resolve().parents[4]


def final_data_root():
    first = os.environ.get('RESOURCE_ALLOCATION_DATA_ROOT')
    second = os.environ.get('FINAL_DATA_ROOT')
    if first and second and Path(first).expanduser().resolve() != Path(second).expanduser().resolve():
        raise ValueError('RESOURCE_ALLOCATION_DATA_ROOT and FINAL_DATA_ROOT must agree')
    return Path(first or second).expanduser().resolve() if first or second else project_root()


@lru_cache(maxsize=1)
def _paths():
    result = {'source': {}, 'data': {}}
    for directory in sorted((project_root() / 'chapters').glob('chapter*')):
        for name in ('path_index.json', 'source_paths.json'):
            file = directory / 'configs' / name
            if not file.is_file():
                continue
            record = json.loads(file.read_text())
            for kind in result:
                mapping = record.get(kind, {})
                if isinstance(mapping, dict):
                    result[kind].update({str(k): str(v) for k, v in mapping.items() if isinstance(v, str)})
    return result


def _mapped(value, entries):
    if value in entries:
        return Path(entries[value])
    for parent in Path(value).parents:
        prefix = str(parent)
        if prefix in entries:
            return Path(entries[prefix]) / Path(value).relative_to(parent)
    return None


def chapter_data_root(chapter):
    number = int(str(chapter).removeprefix('chapter'))
    return final_data_root() / 'chapters' / f'chapter{number:02d}'


def resolve_path(value):
    candidate = Path(value).expanduser()
    text = str(candidate)
    code, data = project_root(), final_data_root()
    relative = None
    for root in (code, data, Path('/hy-tmp/resource_allocation_final'), Path('/root/gamewar/resource_allocation_rl')):
        try:
            relative = candidate.relative_to(root).as_posix()
            break
        except ValueError:
            pass
    for kind, root in (('source', code), ('data', data)):
        for key in (text, relative):
            if key is None:
                continue
            mapped = _mapped(key, _paths()[kind])
            if mapped is not None:
                return root / mapped
    if candidate.is_absolute() and (candidate.is_relative_to(code) or candidate.is_relative_to(data)):
        return candidate
    if text in ('/hy-tmp/resource_allocation_final', '/root/gamewar/resource_allocation_rl'):
        return code
    if text == '/hy-tmp':
        return data
    if relative is not None:
        return code / relative
    if text.startswith(('/hy-tmp/resource_allocation_results/', '/root/resource_allocation_run_storage_20261101/')):
        raise FileNotFoundError('Historical file is not part of the current chapter package: ' + text)
    return candidate


def data_path(value):
    value = Path(value).expanduser()
    if value.is_absolute():
        return resolve_path(value)
    mapped = _mapped(value.as_posix(), _paths()['data'])
    return final_data_root() / (mapped if mapped is not None else value)


def eurosat_root():
    return chapter_data_root(2) / 'data/eurosat_rgb'


def chapter_source_paths(chapter):
    root = project_root() / 'chapters' / f'chapter{int(chapter):02d}'
    paths = [project_root(), root / 'scripts', root / 'algorithms', root / 'environment']
    config = root / 'configs/runtime.json'
    if config.is_file():
        paths.extend(project_root() / value for value in json.loads(config.read_text()).get('python_paths', []))
    return [p for p in paths if p.is_dir()]


def chapter_environment(chapter):
    environment = dict(os.environ)
    environment['RESOURCE_ALLOCATION_PROJECT_ROOT'] = str(project_root())
    environment['RESOURCE_ALLOCATION_DATA_ROOT'] = str(final_data_root())
    environment['FINAL_DATA_ROOT'] = str(final_data_root())
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONPATH'] = os.pathsep.join(map(str, chapter_source_paths(chapter)))
    environment['CH5_FULL_SPEND_DATA_ROOT'] = str(eurosat_root())
    return environment
