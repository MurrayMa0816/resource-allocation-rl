"""Native PPO equations with explicit tensor placement, no algorithm rewrite."""
from __future__ import annotations
import ast
import copy
import hashlib
import inspect
import textwrap


def make_device_update(native_update):
    """Add actor-device placement to exactly the three native as_tensor calls.

    Stripping these device keywords restores the original AST. CPU use remains
    available for verification. CUDA is not claimed to be bitwise CPU-equivalent.
    """
    original = textwrap.dedent(inspect.getsource(native_update))
    before = ast.parse(original)
    after = copy.deepcopy(before)
    additions = []
    for node in ast.walk(after):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch'
                and node.func.attr == 'as_tensor'):
            if any(k.arg == 'device' for k in node.keywords):
                raise AssertionError('Native tensor placement has changed')
            value = ast.parse('next(actor.parameters()).device', mode='eval').body
            node.keywords.append(ast.keyword(arg='device', value=value))
            additions.append(node)
    if len(additions) != 3:
        raise AssertionError('Expected exactly states, event-data and minibatch-index tensor constructors')
    stripped = copy.deepcopy(after)
    for node in ast.walk(stripped):
        if isinstance(node, ast.Call):
            node.keywords = [k for k in node.keywords if k.arg != 'device']
    if ast.dump(stripped, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise AssertionError('Device adapter altered native PPO equations')
    ast.fix_missing_locations(after)
    namespace = {}
    exec(compile(after, '<native PPO + explicit device keywords>', 'exec'), native_update.__globals__, namespace)
    return namespace[native_update.__name__], dict(
        native_update_sha256=hashlib.sha256(original.encode()).hexdigest(),
        added_device_keyword_count=3,
        native_AST_identical_after_removing_device_keywords=True,
        numerical_equivalence='same native float64 equations and update schedule; CUDA/CPU bitwise identity is not asserted')
