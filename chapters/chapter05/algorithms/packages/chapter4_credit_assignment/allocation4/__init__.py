"""Chapter 4 terminal-supervised credit assignment experiments."""
from pathlib import Path
import sys

BASELINE_ROOT = Path(__file__).resolve().parents[2] / 'chapter5_full_spend_20261104/native_source'
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
__version__ = '0.1.0'
