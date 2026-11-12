#!/usr/bin/env python3
"""Verify this publication export, not the private master manifest."""
import sys
from release_tools import main
if __name__ == "__main__":
    raise SystemExit(main(["check", "--require-data", *sys.argv[1:]]))
