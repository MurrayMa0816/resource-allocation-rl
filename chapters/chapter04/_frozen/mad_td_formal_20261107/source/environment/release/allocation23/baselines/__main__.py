"""Chapter 2 external entry and explicit Chapter 3 launcher compatibility."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    chapters, remaining = [], []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == '--chapter' and index + 1 < len(args):
            chapters.append(args[index + 1])
            index += 2
        elif argument.startswith('--chapter='):
            chapters.append(argument.split('=', 1)[1])
            index += 1
        else:
            remaining.append(argument)
            index += 1
    normalized = []
    for chapter in chapters:
        try:
            normalized.append(int(chapter))
        except ValueError:
            normalized.append(chapter)
    if len(set(normalized)) > 1:
        argparse.ArgumentParser(description=__doc__).error(
            'Conflicting chapter selectors; select Chapter 2 or Chapter 3 once')
    if normalized and normalized[0] == 3:
        launcher = Path(__file__).resolve().parents[2] / 'python_ch3'
        if not launcher.is_file():
            raise FileNotFoundError(
                'Chapter 3 uses the independent project python_ch3 launcher; '
                'see CHAPTER3_BASELINES.json for its isolated runtime requirements')
        # Exec the project launcher so SafeDreamer receives its registered JAX
        # interpreter instead of inheriting the Chapter 2 Torch interpreter.
        return os.execv(str(launcher), [str(launcher), *remaining])
    from .runner import main as chapter2_main
    return chapter2_main(args)

if __name__ == '__main__':
    main()
