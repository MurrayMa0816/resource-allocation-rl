"""Verify a chapter's registered current files without loading scientific libraries or models."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json


def main(chapter, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash', action='store_true', help='Also read every registered file and verify SHA256')
    args = parser.parse_args(argv)
    package = Path(__file__).resolve().parents[4]
    root = package / 'chapters' / f'chapter{int(chapter):02d}'
    manifest = root / 'configs/files.json.gz'
    with gzip.open(manifest, 'rt') as stream:
        files = json.load(stream)['files']
    errors = []
    for record in files:
        path = package / record['path']
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record['bytes']:
            errors.append({'path': record['path'], 'reason': 'missing, linked or size differs'})
            continue
        if args.hash:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                errors.append({'path': record['path'], 'reason': 'SHA256 differs'})
    models = json.loads((root / 'models/index.json').read_text())['models']
    for record in models:
        file = package / record['model']
        if not file.is_file() or file.stat().st_size != record['size']:
            errors.append({'path': record['model'], 'reason': 'registered model missing or size differs'})
    report = {'chapter': int(chapter), 'status': 'PASS' if not errors else 'FAIL',
              'files_checked': len(files), 'model_uses': len(models), 'hashes_checked': args.hash,
              'errors': errors, 'model_loading': 0, 'scientific_execution': 0}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(errors))
