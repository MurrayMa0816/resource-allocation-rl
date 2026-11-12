"""Pure AST and integer-geometry proof for the terminal PASTA rollout adapter."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import hashlib
import importlib.util
import json
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_worker():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("c5_pasta_terminal_static_subject", ROOT / "execution/baseline_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def if_condition(source, lineno):
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If) and node.lineno == lineno]
    assert len(nodes) == 1
    return compile(ast.Expression(nodes[0].test), "<registered-terminal-guard>", "eval")


def run():
    worker = load_worker()
    original = (ROOT / "execution/source/baselines/pasta/agent.py").read_text()
    methods = worker.adapted_pasta_methods(original)
    proof = worker.pasta_adapter_proof(original)
    old_tree = ast.parse(original)
    agent = next(node for node in old_tree.body if isinstance(node, ast.ClassDef) and node.name == "Agent")
    init = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    update = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == "update_rollout")
    old_init_guard = if_condition(original, 113)
    new_init_guard = if_condition(methods["__init__"], 113)
    old_return = if_condition(original, 255)
    new_return = if_condition(methods["update_rollout"], 255)
    old_pending_guard = if_condition(original, 259)
    new_pending_guard = if_condition(methods["update_rollout"], 259)
    old_sample_guard = if_condition(original, 278)
    new_sample_guard = if_condition(methods["update_rollout"], 278)
    cases = full_paths = tail_paths = negative_cases = 0
    for h in (24, 32):
        for environments in (1, 8):
            block = h * environments
            for panels in range(1, 81):
                total = panels * block
                state = SimpleNamespace(total_steps=total, horizon=h, n_env=environments,
                                        rollout_panels=8, rollout_size=block*8)
                namespace = {"self": state}
                assert not eval(new_init_guard, namespace)
                assert bool(eval(old_init_guard, namespace)) == (panels % 8 != 0)
                pending = optimized = 0
                for collected_panels in range(1, panels+1):
                    pending += 1
                    state.collected_steps = collected_panels * block
                    state.pending_panels = [None] * pending
                    namespace["obs_t"] = [None] * (pending * block)
                    ret = eval(new_return, namespace)
                    if ret:
                        assert pending < 8 and state.collected_steps != total
                        assert eval(old_return, namespace)
                        continue
                    assert not eval(new_pending_guard, namespace)
                    assert not eval(new_sample_guard, namespace)
                    optimized += len(namespace["obs_t"])
                    cases += 1
                    if pending == 8:
                        assert eval(old_return, namespace) == ret
                        assert eval(old_pending_guard, namespace) == eval(new_pending_guard, namespace)
                        assert eval(old_sample_guard, namespace) == eval(new_sample_guard, namespace)
                        assert len(namespace["obs_t"]) == state.rollout_size
                        full_paths += 1
                    else:
                        assert state.collected_steps == total and pending == panels % 8
                        assert eval(old_return, namespace)
                        tail_paths += 1
                    pending = 0
                assert pending == 0 and optimized == total
            # Incomplete environment panels remain forbidden. No rounding,
            # synthetic transition, padding or dropped sample is accepted.
            for invalid in (block-1, block+1, block*8-1, block*8+1):
                state.total_steps = invalid
                assert eval(new_init_guard, {"self": state})
                negative_cases += 1
            # A terminal guard must not accept zero, too many, mismatched,
            # non-terminal or future short panels.
            state.total_steps = block*10
            for seen, pending in ((block*10, 0), (block*10, 1), (block*10, 3), (block*10, 9),
                                  (block*9, 2), (block*11, 2)):
                state.collected_steps = seen
                state.pending_panels = [None] * pending
                assert eval(new_pending_guard, {"self": state})
                negative_cases += 1
    formal = {}
    for domain, h, gate in (("radio", 32, 4352), ("v5", 24, 3840)):
        block = h * 8
        full = block * 8
        tail = 100608 % full
        assert tail == gate % full and tail % block == 0
        assert tail % 64 == 0 and tail >= 2
        formal[domain] = dict(formal_steps=100608, native_full_rollout_samples=full,
                              terminal_samples=tail, terminal_panels=tail//block,
                              terminal_minibatches_per_epoch=tail//64,
                              engineering_steps=gate, engineering_same_tail_geometry=True)
    try:
        worker.adapted_pasta_methods(original + "\n")
    except RuntimeError:
        negative_cases += 1
    else:
        raise AssertionError("Unknown native source accepted")
    forbidden = [name for name in sys.modules if name == "torch" or name.startswith("torch.")
                 or name == "numpy" or name.startswith("numpy.")]
    assert not forbidden
    return dict(status="PASS_PURE_AST_AND_INTEGER_GEOMETRY_ONLY", adapter=proof,
                complete_rollout_paths_checked=full_paths, terminal_short_rollout_paths_checked=tail_paths,
                total_update_geometry_cases=cases, negative_cases=negative_cases,
                formal_terminal_geometry=formal, environment_steps=0, neural_updates=0,
                holdout_steps=0, model_written=False, cuda_initialized=False,
                dynamic_numeric_equivalence_claimed=False,
                worker_sha256=hashlib.sha256((ROOT / "execution/baseline_worker.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Receipt must physically remain in this new HY scope")
    result = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
