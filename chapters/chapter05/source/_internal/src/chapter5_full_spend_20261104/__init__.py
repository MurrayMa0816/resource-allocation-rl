"""Isolated Chapter 5 revision requiring full spending in every episode.

Importing this package does not select a GPU or alter previously loaded science.
Use bridge.bootstrap before constructing any allocation model or environment.
"""
from pathlib import Path
WORKSPACE=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
__version__='2026.11.04-full-spend-v1'
