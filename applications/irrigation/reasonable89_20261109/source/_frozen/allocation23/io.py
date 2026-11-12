"""Experiment persistence, source identity, and reproducible task seeds."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'allocation23-release-v1'
TRAIN_MASTER = 202610303101
TEST_MASTER = 202610305199
STATE_KEYS = ('local', 'global', 'caps', 'budget', 'remaining')


@contextmanager
def exclusive_lock(path):
    """Hold a nonblocking process lock without modifying another run's files."""
    handle = Path(path).open('a+')
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is writing this run: {path}') from error
        yield handle
    finally:
        handle.close()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    files = []
    for folder, suffixes in [('allocation23', ('.py', '.json'))]:
        files.extend(path for path in (PROJECT_ROOT/folder).rglob('*')
                     if path.is_file() and path.suffix in suffixes)
    return {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in sorted(files)}


def state_hash(model):
    if model is None:
        return None
    result = hashlib.sha256()
    for key, value in model.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def task_seed(phase, seed, index, master=TRAIN_MASTER):
    code = {'train': 1, 'dev': 2, 'final': 3, 'engineering': 4}[phase]
    entropy = [master, code, seed if phase in ('train', 'engineering') else 0, index]
    raw = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
    return (code << 60) | (raw & ((1 << 60) - 1))


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def compact_csv(path):
    path = Path(path)
    if not path.exists():
        return
    destination = path.with_suffix('.csv.gz')
    temporary = destination.with_suffix('.tmp')
    with path.open('rb') as inp, gzip.open(temporary, 'wb', compresslevel=6) as out:
        shutil.copyfileobj(inp, out)
    temporary.replace(destination)
    path.unlink()


def prune_csv(path, episode):
    """Resume only committed batches; discard records newer than a checkpoint."""
    path = Path(path)
    if not path.exists():
        compressed = path.with_suffix('.csv.gz')
        if not compressed.exists():
            return
        with gzip.open(compressed, 'rb') as inp, path.open('wb') as out:
            shutil.copyfileobj(inp, out)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        kept = [row for row in reader if int(row['episode']) <= episode]
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(path)
