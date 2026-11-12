"""Exact runtime-only batching of COLA's three gradient finite guards.

Builds a private unbound update function. It never modifies the pinned class,
source files or existing instances. Losses, gradients, optimizer commits,
target updates, RNG and returned statistics retain their original operations.
Only validation-related CUDA-to-host synchronization is batched per group.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import inspect
import textwrap

import torch

from .native import Agent as NativeCOLA

VERSION = 'ch5-cola-batched-finite-exact-runtime-v1'


def batched_finite_gradients(module, label):
    """One same-device boolean reduction; never a norm or floating sum."""
    flags=[torch.isfinite(parameter.grad).all() for parameter in module.parameters()
           if parameter.grad is not None]
    if flags and not bool(torch.stack(flags).all().cpu().item()):
        raise FloatingPointError(label)


@lru_cache(maxsize=1)
def make_batched_finite_update():
    """Return an unbound private function callable as update(agent, batch).

    A caller may explicitly bind it to a newly constructed instance after a
    passing exactness/timing gate. Existing native class methods stay intact.
    """
    original=NativeCOLA.update
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    definition=tree.body[0]
    expected={'encoder':'OADM gradient','critic':'COLA critic gradient','actor':'COLA actor gradient'}
    changed={}
    for index,node in enumerate(definition.body):
        if not isinstance(node,ast.For):continue
        if not (isinstance(node.target,ast.Name) and node.target.id=='parameter'
                and isinstance(node.iter,ast.Call) and not node.iter.args and not node.iter.keywords
                and isinstance(node.iter.func,ast.Attribute) and node.iter.func.attr=='parameters'
                and isinstance(node.iter.func.value,ast.Attribute)
                and isinstance(node.iter.func.value.value,ast.Name) and node.iter.func.value.value.id=='self'):
            continue
        owner=node.iter.func.value.attr
        if owner not in expected:continue
        # Refuse upstream drift rather than rewriting a semantically different
        # loop or accidentally swallowing non-validation operations.
        source=ast.unparse(node)
        wanted=f"for parameter in self.{owner}.parameters():\n    if parameter.grad is not None:\n        finite(parameter.grad, {expected[owner]!r})"
        if ast.dump(ast.parse(source),include_attributes=False)!=ast.dump(ast.parse(wanted),include_attributes=False):
            raise RuntimeError('Pinned COLA finite-gradient loop no longer matches: '+owner)
        definition.body[index]=ast.Expr(value=ast.Call(
            func=ast.Name(id='_cola_exact_batched_finite_gradients',ctx=ast.Load()),
            args=[ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr=owner,ctx=ast.Load()),
                  ast.Constant(value=expected[owner])],keywords=[]))
        changed[owner]=changed.get(owner,0)+1
    if changed!={key:1 for key in expected}:
        raise RuntimeError('Expected exactly the three pinned COLA gradient loops')
    namespace=dict(original.__globals__)
    namespace['_cola_exact_batched_finite_gradients']=batched_finite_gradients
    module=ast.fix_missing_locations(tree)
    exec(compile(module,inspect.getsourcefile(original)+'::<batched-finite-only>', 'exec'),namespace)
    update=namespace[definition.name]
    update.runtime_version=VERSION
    update.finite_guard_groups=tuple(expected)
    update.native_class_mutated=False
    return update
