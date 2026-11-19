"""Check or explicitly reproduce one current paper figure."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "chapter02/scripts/support"))
from figure_tool import main
if __name__ == "__main__":
    raise SystemExit(main())
