"""Locations for the single stored C6 implementation and explicit execution capsules."""
from pathlib import Path
import os
import sys

CHAPTER = Path(__file__).resolve().parents[1]
PACKAGE = CHAPTER.parents[1]
SUPPORT = CHAPTER.parent / 'chapter02/scripts/support'
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))


def primary_study():
    return CHAPTER / 'results/sedval'


def external_study():
    return CHAPTER / 'results/comparison'


def primary_source():
    capsule = os.environ.get('CHAPTER06_SOURCE_CAPSULE')
    if not capsule:
        raise RuntimeError('Set CHAPTER06_SOURCE_CAPSULE to an explicitly materialized source capsule')
    root = Path(capsule).expanduser().resolve()
    if not (root / 'capsule.json').is_file():
        raise ValueError('Missing chapter-six source capsule receipt')
    return root / 'sedval'


def shared_formal285():
    return CHAPTER.parent / 'chapter05/algorithms'


def figures():
    return CHAPTER / 'figures'


def fonts():
    return CHAPTER.parent / 'chapter02/data/fonts'


def original_input(value):
    from portable_paths import data_path
    return data_path(value)
