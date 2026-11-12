"""Offline saved-data figure; external inputs and original fonts are required."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from figure_runtime import main
if __name__ == '__main__':
    raise SystemExit(main(__file__, '2.7'))
