"""Resolve Chapter 6 source and separately stored experiment data."""
from pathlib import Path
import sys

CHAPTER = Path(__file__).resolve().parent
CHAPTERS = CHAPTER.parent
if str(CHAPTERS) not in sys.path:
    sys.path.insert(0, str(CHAPTERS))
from portable_paths import final_data_root, chapter_data_root, data_path, resolve_path


def primary_study():
    return chapter_data_root(6) / "studies/reasonable_allocation89_20261109"


def external_study():
    return chapter_data_root(6) / "studies/external_learning_comparison_20261109"


def primary_source():
    return CHAPTER / "experiment"


def shared_formal285():
    return CHAPTERS / "chapter05/source"


def figures():
    return CHAPTER / "figures"


def fonts():
    return final_data_root() / "assets/fonts"


def original_input(value):
    return data_path(value)
