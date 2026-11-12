"""Compatibility notice for the retired 960-episode/128-path holdout CLI.

The immutable original survives in the accepted scientific source capsules.
Current C4 endpoints use the explicit 100608-step formal entry and its
registered fixed-model evaluator. Importing this notice starts no runtime.
"""
CURRENT_ENTRY = 'python -m chapters.chapter04.run --evaluation-only --method METHOD --config CONFIG'

def evaluate_checkpoint(*args, **kwargs):
    raise RuntimeError('The historical 960-episode holdout is retired. Use '+CURRENT_ENTRY)

def main():
    print(CURRENT_ENTRY)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
