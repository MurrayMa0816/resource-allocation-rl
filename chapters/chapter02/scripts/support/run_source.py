"""Execute a preverified, explicit source capsule through path-only in-memory adaptation."""
from pathlib import Path
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-capsule', type=Path, required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = args.source_capsule.expanduser().resolve(strict=True)
    script = args.script.expanduser().resolve(strict=True)
    if not script.is_relative_to(root):
        parser.error('The explicit source file must be inside the verified source capsule')
    os.environ['RESOURCE_ALLOCATION_SOURCE_CAPSULE'] = str(root)
    sys.dont_write_bytecode = True
    from frozen_adapter import run_file
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    run_file(script, arguments)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
