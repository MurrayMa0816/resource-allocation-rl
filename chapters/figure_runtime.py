"""Prepare verified external figure inputs without importing plotting libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile


FONT_FILES = ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf',
              'simsun.ttc', 'simhei.ttf')
INPUT_SUFFIXES = {'.csv', '.gz', '.json', '.png', '.pdf'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Input paths must stay inside the figure package: ' + relative)
    candidate = root / path
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError('Input path leaves the figure package: ' + relative)
    return candidate


def font_files(directory):
    return {path.name.lower(): path for path in directory.iterdir()
            if path.is_file()} if directory.is_dir() else {}


def inspect_inputs(figure, data, fonts):
    """Check only registered files; JSON path strings never select executable code."""
    expected = json.loads((figure / 'local_inputs_sha256.json').read_text())
    problems = []
    for relative, wanted in expected.items():
        path = checked_path(data, relative)
        if path.suffix not in INPUT_SUFFIXES:
            raise ValueError('Unregistered input type: ' + relative)
        if not path.is_file():
            problems.append('missing input: ' + relative)
        elif digest(path) != wanted:
            problems.append('input SHA256 differs: ' + relative)
    available_fonts = font_files(fonts)
    for name in FONT_FILES:
        if name not in available_fonts:
            problems.append('missing original font: ' + name)
    return expected, problems


def stage_inputs(figure, data, fonts, target, expected):
    """Copy a fixed input list and repository sources as ordinary temporary files."""
    target.mkdir(parents=True)
    for relative, wanted in expected.items():
        source = checked_path(data, relative)
        destination = checked_path(target, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination) != wanted:
            raise ValueError('Input changed while staging: ' + relative)
    source_hashes = {}
    for source in sorted(figure.rglob('*.py')):
        if '__pycache__' in source.parts:
            continue
        relative = source.relative_to(figure).as_posix()
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[relative] = digest(source)
        if digest(destination) != source_hashes[relative]:
            raise ValueError('Repository source changed while staging: ' + relative)
    # Native renderers retain their original source hashes and font lookup.
    # The font directory below exists only in this disposable execution workspace.
    font_target = target.parent / 'assets/fonts/windows'
    font_target.mkdir(parents=True)
    font_hashes = {}
    available_fonts = font_files(fonts)
    for name in FONT_FILES:
        source = available_fonts[name]
        aliases = {name, source.name}
        if name.startswith('times'):
            aliases.add(name.upper())
        wanted = digest(source)
        for alias in sorted(aliases):
            shutil.copyfile(source, font_target / alias)
            font_hashes[alias] = wanted
            if digest(font_target / alias) != wanted:
                raise ValueError('Font changed while staging: ' + alias)
    return source_hashes, font_hashes, font_target


def main(entry, number, argv=None):
    sys.dont_write_bytecode = True
    figure = Path(entry).resolve().parent
    parser = argparse.ArgumentParser(
        description='Offline figure ' + number + '; requires its registered external figure package and original fonts.')
    parser.add_argument('--data-dir', type=Path,
                        help='complete external figure package containing the registered inputs')
    parser.add_argument('--font-dir', type=Path,
                        help='original Times New Roman, SimSun and SimHei files; defaults to THESIS_FIGURE_FONT_DIR')
    parser.add_argument('--output-dir', type=Path,
                        help='required for rendering: new output directory separate from source, data and fonts')
    parser.add_argument('--check', action='store_true',
                        help='check input hashes and font files only; do not import plotting libraries or render')
    args = parser.parse_args(argv)
    fonts = args.font_dir or os.environ.get('THESIS_FIGURE_FONT_DIR')
    if args.data_dir is None:
        parser.error('--data-dir is required: figure data and parent references are not bundled')
    if not fonts:
        parser.error('pass --font-dir or set THESIS_FIGURE_FONT_DIR; fonts are not bundled')
    data, fonts = args.data_dir.expanduser().resolve(), Path(fonts).expanduser().resolve()
    expected, problems = inspect_inputs(figure, data, fonts)
    if problems:
        parser.error('external figure inputs are incomplete or differ:\n  ' + '\n  '.join(problems))
    report = dict(figure=number, status='PASS', checked_inputs=len(expected),
                  source='repository', external_python_executed=False,
                  font_sha256={name: digest(font_files(fonts)[name]) for name in FONT_FILES},
                  rendering_performed=False)
    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error('pass --output-dir explicitly for rendering; source folders are never default outputs')
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error('--output-dir must be new; existing directories are preserved')
    for label, protected in (('repository figure sources', figure),
                             ('external input package', data), ('original fonts', fonts)):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error('--output-dir must not contain or be inside the ' + label)
    with tempfile.TemporaryDirectory(prefix='thesis-figure-' + number.replace('.', '-') + '-') as workspace:
        staged = Path(workspace) / 'figure'
        source_hashes, font_hashes, staged_fonts = stage_inputs(figure, data, fonts, staged, expected)
        saved_environment = {key: os.environ.get(key) for key in
                             ('THESIS_FIGURE_FONT_DIR', 'THESIS_FONT_DIR')}
        saved_argv = sys.argv[:]
        try:
            os.environ['THESIS_FIGURE_FONT_DIR'] = str(staged_fonts)
            os.environ['THESIS_FONT_DIR'] = str(staged_fonts)
            renderer = staged / 'replot_sources/renderer.py'
            sys.argv = [str(renderer), '--number', number, '--output-dir', str(output)]
            runpy.run_path(str(renderer), run_name='__main__')
            # Some unchanged native renderers emit links to their saved inputs.
            # Materialize those outputs before the disposable workspace is removed.
            for path in output.rglob('*'):
                if path.is_symlink():
                    contents = path.read_bytes()
                    path.unlink()
                    path.write_bytes(contents)
            receipt = dict(report, rendering_performed=True, input_sha256=expected,
                           repository_source_sha256=source_hashes, font_sha256=font_hashes,
                           replot_note='Use the repository plot.py with the same --data-dir and --font-dir for another render.')
            (output / 'input_preparation.json').write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        finally:
            sys.argv = saved_argv
            for key, value in saved_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return 0
