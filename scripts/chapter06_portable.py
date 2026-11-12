#!/usr/bin/env python3
"""Inspect/verify retained Chapter6 data and redraw saved figures after relocation.

No training, controller dispatch, policy inference or scientific module import.
Data paths are relative to RESOURCE_ALLOCATION_DATA_ROOT; code paths are relative
to this code project. Original source/protocol hashes remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from portable_paths import final_data_root

FIGURES = {
    '6.1': 'figure06-01_区域与单元说明_20261109',
    '6.3': 'figure06-03_全期用水与余额统一风格_20261109',
    '6.4': 'figure06-04_阶段状态配水统一风格_20261109',
    '6.5': 'figure06-05_学习方法配置成效比较_20261109',
    '6.6': 'figure06-06_区域终点统一尺度_20261109',
    '6.7': 'figure06-07_紧凑供水条件_20261109',
    '6.8': 'figure06-08_天气与配水响应统一风格_20261109',
}


def relative_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(v in ('..', '') for v in p.parts):
        raise ValueError('Expected a normalized internal relative path: ' + str(value))
    return p


def local_file(root, value):
    relative = relative_path(value)
    path = root.joinpath(*relative.parts)
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.is_symlink():
            raise ValueError('Portable package must contain regular files, no links: ' + str(item))
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError('Missing internal file: ' + str(path))
    return resolved


def digest(stream):
    h = hashlib.sha256()
    total = 0
    for part in iter(lambda: stream.read(1024 * 1024), b''):
        total += len(part)
        h.update(part)
    return total, h.hexdigest()


def verify(roots, manifest):
    bad = []
    expected_models = manifest.get('primary_fixed3200_models', 9) + sum(manifest.get('external_fixed_endpoints', {'PASTA': 3, 'COLA': 3}).values())
    actual_models = sum(row['relative'].endswith(('/model.pt', '/endpoint.pt')) for row in manifest['files'])
    if actual_models != expected_models:
        bad.append({'models': actual_models, 'expected': expected_models, 'error': 'Endpoint manifest count differs'})
    archives = {}
    direct = 0
    for row in manifest['files']:
        if row.get('archive_relative'):
            archives.setdefault((row.get('root', 'data'), row['archive_relative']), []).append(row)
            continue
        try:
            path = local_file(roots[row.get('root', 'data')], row['relative'])
            with path.open('rb') as stream:
                size, actual = digest(stream)
            if size != row['size'] or actual != row['sha256']:
                raise ValueError('size/SHA256 differs')
            direct += 1
        except (OSError, ValueError) as e:
            bad.append({'file': row['relative'], 'error': str(e)})
    archived = 0
    for (root_name, archive_name), rows in archives.items():
        try:
            path = local_file(roots[root_name], archive_name)
            by_member = {str(relative_path(row['archive_member'])): row for row in rows}
            if len(by_member) != len(rows):
                raise ValueError('Duplicate declared archive member')
            seen = set()
            with tarfile.open(path, mode='r|*') as bundle:
                for member in bundle:
                    name = str(relative_path(member.name))
                    if name not in by_member:
                        continue
                    if not member.isfile() or member.issym() or member.islnk() or name in seen:
                        raise ValueError('Declared archive member is not one unique regular file: ' + name)
                    seen.add(name)
                    row = by_member[name]
                    stream = bundle.extractfile(member)
                    if stream is None:
                        raise ValueError('Unreadable archive member: ' + name)
                    with stream:
                        size, actual = digest(stream)
                    if size != row['size'] or actual != row['sha256']:
                        raise ValueError('Archived payload size/SHA256 differs: ' + name)
                    archived += 1
            if seen != set(by_member):
                raise ValueError('Missing declared archive members: ' + ', '.join(sorted(set(by_member) - seen)))
        except (OSError, ValueError, tarfile.TarError) as e:
            bad.append({'archive': archive_name, 'error': str(e)})
    result = {'status': 'PASS' if not bad else 'FAIL', 'regular_files_verified': direct,
              'archived_regular_files_verified': archived, 'expected_model_count': expected_models,
              'training_endpoints': dict(SeDVal=manifest.get('primary_fixed3200_models', 9), **manifest.get('external_fixed_endpoints', {'PASTA': 3, 'COLA': 3})),
              'formal_evaluation_jobs': manifest.get('formal_evaluation_jobs', 40),
              'formal_evaluation_paths': manifest.get('formal_evaluation_paths', 2698),
              'scientific_imports': 0, 'environment_steps': 0, 'model_inferences': 0,
              'errors': bad}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not bad else 1


def redraw(root, args):
    figure_root = root / args.figure_root
    if not figure_root.resolve().is_relative_to(root):
        raise ValueError('Figure input must be internal to the portable package')
    output = (args.output_root or root / 'generated/chapter06').expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = FIGURES if args.figure == 'all' else {args.figure: FIGURES[args.figure]}
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(output / '.cache/matplotlib'),
               XDG_CACHE_HOME=str(output / '.cache'))
    for number, name in selected.items():
        source = local_file(root, str((figure_root / name / 'plot.py').relative_to(root)))
        source_dir = source.parent
        destination = output / ('figure' + number.replace('.', '_'))
        if destination.is_relative_to(source_dir) or source_dir.is_relative_to(destination):
            raise ValueError('Use a separate output directory, preserving accepted figure files')
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(source), '--output-dir', str(destination)],
                       cwd=source_dir, env=env, check=True)
    print(json.dumps({'status': 'REDRAWN_FROM_SAVED_DATA', 'figures': list(selected),
                      'output_root': str(output), 'new_training': False,
                      'new_policy_inference': False, 'new_TEST': False}, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'verify', 'redraw'])
    parser.add_argument('--data-root', type=Path,
                        default=final_data_root(),
                        help='physical final data directory; environment RESOURCE_ALLOCATION_DATA_ROOT overrides default')
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='resource_allocation_rl source project')
    parser.add_argument('--manifest', default='registry/chapter06_portable_manifest.json')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--figure', choices=['all', *FIGURES], default='all')
    parser.add_argument('--figure-root', default=None)
    args = parser.parse_args()
    # Publication release verification uses its actual exported inventory.
    release_root = args.code_root.expanduser().resolve()
    if args.command == 'verify' and (release_root / 'release_manifest.json').is_file():
        return subprocess.run([sys.executable, '-B', str(release_root / 'release_tools.py'),
            '--root', str(release_root), 'check', '--require-data'], check=False).returncode
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        parser.error('data root is not a directory')
    if args.figure_root is None:
        index=json.loads((root/'INDEX.json').read_text())
        args.figure_root=index['chapters']['chapter06']['directory']+'/figures'
    if args.command == 'redraw':
        return redraw(root, args)
    code_root = args.code_root.expanduser().resolve()
    roots = {'data': root, 'code': code_root}
    manifest_path = root / args.manifest
    if manifest_path.is_file():
        manifest = json.loads(local_file(root, args.manifest).read_text())
    else:
        manifest = json.loads(local_file(code_root, args.manifest).read_text())
    if args.command == 'verify':
        return verify(roots, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
