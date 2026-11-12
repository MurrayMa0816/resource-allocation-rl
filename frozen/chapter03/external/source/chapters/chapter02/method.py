"""Chapter 2: learn object allocation under a fixed per-period total."""

VARIANTS = ('A_FULL', 'A_RANDOM_MATCH', 'A_SOFTMAX', 'A_DIRECT')


def _chapter2(spec):
    if getattr(spec, 'chapter', None) != 2:
        raise ValueError('Chapter 2 methods require a chapter-2 public specification')


def make_actor(spec, seed=0, *, variant='A_FULL'):
    """Build A or one of the registered spatial ablations, with fixed totals."""
    _chapter2(spec)
    if variant not in VARIANTS:
        raise ValueError('Choose a registered chapter-2 spatial variant')
    from allocation23.methods import make_actor as shared_actor
    return shared_actor(spec, 'A', 2, seed, variant=variant, temporal='fixed')


def make_value(spec, seed=0):
    _chapter2(spec)
    from allocation23.methods import make_value as shared_value
    return shared_value(spec, seed)
