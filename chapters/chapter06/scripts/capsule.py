"""Check or materialize the exact registered C6 source tree, without science imports."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil

CHAPTER = Path(__file__).resolve().parents[1]
PACKAGE = CHAPTER.parents[1]
MANIFEST = CHAPTER / 'configs/source_views.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a package-relative source path: ' + value)
    return path


def document():
    return json.loads(MANIFEST.read_text())


def inspect_sources():
    doc = document()
    errors = []
    for name, view in doc['views'].items():
        for logical, record in view.items():
            relative(logical)
            path = PACKAGE / relative(record['path'])
            if not path.is_file() or sha(path) != record['sha256']:
                errors.append(name + ':' + logical)
    view = doc['views']['sedval']
    contract = json.loads((PACKAGE / view['source/hongmiao89/dependency_manifest.json']['path']).read_text())
    for logical, digest in contract['files_sha256'].items():
        record = view.get('source/_frozen/' + logical)
        if record is None or record['sha256'] != digest:
            errors.append('dependency121:' + logical)
    baseline = CHAPTER.parent / 'chapter05/algorithms/baselines'
    adapter = ast.parse((PACKAGE / doc['views']['comparison']['methods/pasta_adapter.py']['path']).read_text())
    expected = next(ast.literal_eval(node.value) for node in adapter.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'SOURCE_SHA256' for target in node.targets))
    for logical, digest in expected.items():
        path = baseline / 'pasta' / logical
        if not path.is_file() or sha(path) != digest:
            errors.append('chapter05/pasta:' + logical)
    for logical in ['__init__.py', 'cola/agent.py', 'cola/configs.json', '_support/replay.py',
                    '_support/runtime.py', '_support/common.py', '_support/update_clock.py']:
        if not (baseline / logical).is_file():
            errors.append('chapter05/baselines:' + logical)
    return {'status': 'PASS' if not errors else 'FAIL',
            'source_view_files': {name: len(view) for name, view in doc['views'].items()},
            'source_dependency_hashes': len(contract['files_sha256']),
            'PASTA_source_hashes': len(expected), 'errors': errors,
            'model_loading': 0, 'environment_steps': 0}


def prepare_source_capsule(target):
    """Create a new exact logical source tree from the single stored source entities."""
    report = inspect_sources()
    if report['errors']:
        raise ValueError(report['errors'])
    target = Path(target).expanduser().resolve()
    if target.exists():
        raise FileExistsError('A source capsule requires a new directory: ' + str(target))
    if target.is_relative_to(PACKAGE) or PACKAGE.is_relative_to(target):
        raise ValueError('A source capsule must be separate from the complete maintained package')
    target.mkdir(parents=True)
    count = 0
    try:
        for name, view in document()['views'].items():
            for logical, record in view.items():
                source = PACKAGE / relative(record['path'])
                destination = target / name / relative(logical)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if sha(destination) != record['sha256'] or destination.is_symlink():
                    raise RuntimeError('Source capsule copy differs: ' + logical)
                count += 1
        (target / 'capsule.json').write_text(json.dumps({
            'schema': 'chapter06-materialized-scientific-source-v1',
            'source_manifest_sha256': sha(MANIFEST),
            'source_files': count, 'all_source_hashes_match': True,
            'ordinary_files_only': True, 'scientific_execution': 0,
        }, ensure_ascii=False, indent=2) + '\n')
    except BaseException:
        shutil.rmtree(target)
        raise
    return {'status': 'PASS', 'target': str(target), 'source_files': count,
            'scientific_execution': 0}


def verify_source_capsule(target):
    """Reject a stale receipt or modified source before any scientific dispatch."""
    target = Path(target).expanduser().resolve()
    receipt = json.loads((target / 'capsule.json').read_text())
    if receipt.get('source_manifest_sha256') != sha(MANIFEST):
        raise ValueError('Source capsule does not match the current source manifest')
    expected_files = {'capsule.json'}
    for name, view in document()['views'].items():
        for logical, record in view.items():
            relative_name = name + '/' + relative(logical).as_posix()
            expected_files.add(relative_name)
            path = target / relative_name
            if not path.is_file() or path.is_symlink() or sha(path) != record['sha256']:
                raise ValueError('Source capsule file differs: ' + relative_name)
    actual_files = {path.relative_to(target).as_posix() for path in target.rglob('*') if path.is_file()}
    if actual_files != expected_files:
        raise ValueError('Source capsule contains missing or additional files')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check source and native baseline identities only (default)')
    parser.add_argument('--materialize', type=Path, help='Explicitly create a new source-only execution capsule')
    args = parser.parse_args(argv)
    result = prepare_source_capsule(args.materialize) if args.materialize else inspect_sources()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
