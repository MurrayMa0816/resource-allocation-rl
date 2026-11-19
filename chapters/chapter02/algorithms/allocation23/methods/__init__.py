"""Public A factory; chapter 2 fixes totals and chapter 3 learns totals.

A maps to A_FULL (multiscale features + simplex projection).
Explicit variant/temporal arguments expose only the registered ablations.
"""
from .policy import Actor, VARIANTS, TEMPORALS
from .networks import Value


def make_actor(spec, method='A', chapter=2, seed=0, *, variant=None, temporal=None,
               quota_reference='pace'):
    if method != 'A' or isinstance(chapter, bool) or chapter not in (2, 3):
        raise ValueError('Choose method A and chapter 2/3')
    if getattr(spec, 'chapter', chapter) != chapter:
        raise ValueError('Policy chapter must match the public state specification')
    variant = f'{method}_FULL' if variant is None else variant
    if variant not in VARIANTS or not variant.startswith(method + '_'):
        raise ValueError('The spatial variant must belong to the selected method')
    temporal = ('fixed' if chapter == 2 else 'adaptive') if temporal is None else temporal
    if (temporal not in TEMPORALS or (chapter == 2 and temporal != 'fixed')
            or (chapter == 3 and temporal == 'fixed')):
        raise ValueError('Chapter 2 requires fixed; chapter 3 requires a registered chapter-3 temporal mode')
    return Actor(spec, variant, temporal, seed, quota_reference=quota_reference)


def make_value(spec, seed=0):
    return Value(spec, seed)


__all__ = ['make_actor', 'make_value', 'Actor', 'Value', 'VARIANTS', 'TEMPORALS']
