"""Chapter 4 terminal-supervised credit assignment experiments."""
from pathlib import Path
import sys

BASELINE_ROOT = Path('/root/gamewar/chapter23_final')
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
__version__ = '0.1.0'
