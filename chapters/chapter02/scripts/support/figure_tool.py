"""Check a current figure or explicitly render its preserved source in a disposable workspace."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile


def project_root():
    return Path(__file__).resolve().parents[4]


def _digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def checked_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Package path must be relative: ' + str(value))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', required=True, help='Paper figure number, for example 2.4')
    parser.add_argument('--check', action='store_true', help='Verify source and input hashes without rendering')
    parser.add_argument('--execute', action='store_true', help='Explicitly run the current plotting source')
    parser.add_argument('--output-dir', type=Path, help='New output directory outside the complete package')
    args = parser.parse_args(argv)
    try:
        chapter, ordinal = map(int, args.number.split('.'))
    except ValueError:
        parser.error('Use a figure number such as 2.4')
    root = project_root()
    folder = root / f'chapters/chapter{chapter:02d}/figures/figure{chapter:02d}_{ordinal:02d}'
    file = folder / 'inputs.json'
    if not file.is_file():
        parser.error('Unknown current figure: ' + args.number)
    manifest = json.loads(file.read_text())
    for relative, item in manifest['files'].items():
        checked_relative(relative)
        source = root / checked_relative(item['path'])
        if not source.is_file() or source.stat().st_size != item['bytes'] or _digest(source) != item['sha256']:
            raise ValueError('Figure input size or hash differs: ' + str(source))
    report = {'figure': args.number, 'status': 'PASS', 'inputs_checked': len(manifest['files']), 'rendering': False, 'model_loading': 0}
    if args.check or not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if manifest['kind'] == 'drawio':
        parser.error('This diagram is edited using its source.drawio; the accepted PDF is supplied.')
    if args.output_dir is None:
        parser.error('Rendering requires an explicit --output-dir')
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        parser.error('Choose a new output directory outside and disjoint from the complete package')
    with tempfile.TemporaryDirectory(prefix='thesis-final-figure-') as temporary:
        stage = Path(temporary)
        package = stage / checked_relative(manifest['original_virtual_directory'])
        package.mkdir(parents=True)
        for relative, item in manifest['files'].items():
            destination = package / checked_relative(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / item['path'], destination)
            assert _digest(destination) == item['sha256']
        fonts = stage / 'assets/fonts'
        shutil.copytree(root / 'chapters/chapter02/data/fonts', fonts)
        windows = fonts / 'windows'
        if windows.is_dir():
            originals = list(windows.iterdir())
            for source in originals:
                if source.is_file():
                    for alias in (source.name.lower(), source.name.upper()):
                        target = windows / alias
                        if not target.exists():
                            shutil.copyfile(source, target)
                            assert _digest(source) == _digest(target)
        scripts = stage / 'scripts'
        scripts.mkdir()
        shutil.copyfile(Path(__file__).with_name('restore_render_support.py'), scripts / 'restore_render_support.py')
        environment = dict(os.environ)
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        environment['THESIS_FIGURE_FONT_DIR'] = str(windows)
        entry = package / checked_relative(manifest['entry'])
        subprocess.run([sys.executable, '-B', str(entry), '--output-dir', str(output)], cwd=stage, env=environment, check=True)
        for produced in output.rglob('*'):
            if produced.is_symlink():
                contents = produced.read_bytes()
                produced.unlink()
                produced.write_bytes(contents)
        (output / 'input_identity.json').write_text(json.dumps({'figure': args.number, 'files': manifest['files'], 'model_loading': 0}, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
