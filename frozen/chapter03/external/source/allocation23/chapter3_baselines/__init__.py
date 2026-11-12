"""Compatibility import for the current Chapter 3 factory and public helpers.

Historical learning cores remain here for provenance; current editable agents
and profiles reside in allocation23.chapter3_candidates.
"""
from ..chapter3_candidates import METHODS, make, load, profile
__all__ = ('METHODS', 'make', 'load', 'profile')
