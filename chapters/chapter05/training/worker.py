"""Explicit current-formal train entry. No legacy queue is dispatched."""
from chapters.chapter05._runtime import main as _main

def main(argv=None):
    return _main('train', argv)

if __name__ == "__main__":
    raise SystemExit(main())
