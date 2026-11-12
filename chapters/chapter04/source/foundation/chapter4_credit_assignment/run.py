from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import allocation4
from allocation4.training import main
if __name__ == '__main__':
    main()
