"""Check the current chapter files; optional --hash verifies all bytes."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "chapter02/scripts/support"))
from check_package import main
if __name__ == "__main__":
    raise SystemExit(main(4))
