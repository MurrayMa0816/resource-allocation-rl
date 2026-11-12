"""Shared checkout paths and source discovery, without scientific imports.

Chapter-owned sources have one physical copy. Their original allocation23
package paths remain aliases, so persisted class names and relative imports
stay stable. Source discovery follows these directory aliases explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _walk(directory, prefix, ancestors=frozenset()):
    directory = Path(directory)
    real = directory.resolve(strict=True)
    if real in ancestors:
        raise ValueError(f'Cyclic source-directory alias: {directory}')
    ancestors = ancestors | {real}
    with os.scandir(directory) as stream:
        entries = sorted(stream, key=lambda item: item.name)
    for entry in entries:
        if entry.name.startswith('.') or entry.name == '__pycache__':
            continue
        path = directory / entry.name
        name = f'{prefix}/{entry.name}'
        if entry.is_dir(follow_symlinks=True):
            yield from _walk(path, name, ancestors)
        elif entry.is_file(follow_symlinks=True) and path.suffix in ('.py', '.json'):
            yield name, path


def iter_source_files():
    """Yield stable logical names and readable paths for current C2/C3 code.

Release/acceptance records and caches are excluded to avoid self-referential
digests. Frozen experiment sources outside this maintained tree are not read.
"""
    found = dict(_walk(PROJECT_ROOT / 'allocation23', 'allocation23'))
    for name in ('python', 'python_ch3', 'run.py', 'chapter3_repaired.py', 'project',
                 'PROJECT.json', 'pyproject.toml', 'CHAPTER2_CURRENT.json',
                 'CHAPTER2_BASELINES.json', 'CHAPTER3_CURRENT.json',
                 'CHAPTER3_BASELINES.json', 'CHAPTER3_ABLATIONS.json'):
        path = PROJECT_ROOT / name
        if path.is_file():
            found[name] = path
    for chapter in ('chapter02', 'chapter03'):
        base = PROJECT_ROOT / 'chapters' / chapter
        if not base.is_dir():
            continue
        for name in ('launcher.py', 'method.py', 'environment.py', 'ablations.py'):
            path = base / name
            if path.is_file():
                found[f'chapters/{chapter}/{name}'] = path
        if (base / 'configs').is_dir():
            found.update(_walk(base / 'configs', f'chapters/{chapter}/configs'))
    yield from sorted(found.items())
