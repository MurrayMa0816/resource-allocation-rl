"""Native Worker derivation for the new random-object development ablation.

The policy factory intervention is RandomObjectActor. Physical collection,
complete-episode batches, Critic, PPO, stopping, DEV and persistence are inherited
from the SHA-bound original Worker.  This file never evaluates a TEST path.
"""
from __future__ import annotations

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root

import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CAPSULE = Path(__file__).resolve().parents[1]/'native'
ARM = 'A_RANDOM_OBJECT_ALLOCATION'
SCHEMA = 'chapter3-random-object-allocation-v1'
BASE_WORKER_SHA = '1ead3d4c49133e80f5744bb4739ce763715ddc97d0c8a025975a8482245c36db'
FINAL_MASTER = 202611093799


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Derivation(ast.NodeTransformer):
    def __init__(self):
        self.changes = []

    def mark(self, kind, node):
        self.changes.append(dict(kind=kind, original_line=node.lineno))

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == 'ARMS':
                self.mark('register_single_new_arm', node)
                node.value = ast.parse("{'A_RANDOM_OBJECT_ALLOCATION':dict(variant='A_FULL',temporal='adaptive',quota_reference='pace')}", mode='eval').body
                return node
            if name == 'SCHEMA':
                self.mark('new_experiment_schema', node)
                node.value = ast.Constant(SCHEMA)
                return node
            if name == 'registry_file':
                self.mark('read_original_immutable_source_registry', node)
                node.value = ast.parse('Path('+repr(str(CAPSULE/'source_registry.json'))+')', mode='eval').body
                return node
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == 'direct_policy':
            self.mark('route_only_new_actor_class', node)
            return ast.copy_location(ast.ImportFrom(module='_chapter03_random_object_policy',
                names=[ast.alias(name='RandomObjectActor', asname='DirectTotalActor')], level=0), node)
        return node

    def visit_Constant(self, node):
        changes = {
            'A_DIRECT_TOTAL': ('new_arm_factory_and_initialization_flag', ARM),
            'Training never opens final/test paths; freeze all40 first': (
                'training_never_opens_confirmation', 'Random-object development training never opens final/test paths'),
            'native spatial backbone, state, PPO and full-spend; random output initialization; four separately trained preregistered arms': (
                'disclose_core_deletion_scope', 'new random-object development ablation: same-seed Full construction then object output head and learned w events deleted; complete quota context, Critic, PPO, physical task and full-spend retained; independent normalized Uniform weights use original capped-simplex projection'),
            'fixed100608step endpoint; training final0; all40 endpoints must be sealed before frozen evaluation': (
                'disclose_complete_endpoint_seal', 'fixed100608step development endpoint; training final0; ten new random-object endpoints; no new TEST or common128 evaluation'),
            'native nn.Linear default': ('disclose_object_head_deletion', 'deleted after same-seed Full construction; retained encoders and quota tensors unchanged'),
            'formal_100k_five_seed': ('development_profile_identity', 'development_fixed100608_five_seed'),
            'same native task, deterministic mean-latent decoder, DEV512x8 and exact native replay; no final/test during training': (
                'disclose_random_evaluation', 'same native task; eta mean in DEV but independently seeded random object allocation remains active; DEV512x8 and exact native action replay; no final/test'),
        }
        if isinstance(node.value, str) and node.value in changes:
            kind, replacement = changes[node.value]
            self.mark(kind, node)
            return ast.copy_location(ast.Constant(replacement), node)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'episode'
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ('task', 'evaluation_task')):
            self.mark('bind_episode_random_object_stream', node)
            return ast.copy_location(ast.Call(func=ast.Name(id='episode_with_random_object', ctx=ast.Load()),
                args=[node.func.value, *node.args], keywords=node.keywords), node)
        if (isinstance(node.func, ast.Name) and node.func.id == 'dict'
                and {k.arg for k in node.keywords} == {'package', 'worker_sha256', 'source_registry_sha256', 'direct_capsule'}):
            self.mark('record_core_derivation_identity', node)
            node.keywords.append(ast.keyword(arg='core_ablation_derivation', value=ast.Name(id='EXECUTION_IDENTITY', ctx=ast.Load())))
        # Original flag describes Direct's fixed offset.  This arm has no offset.
        if isinstance(node.func, ast.Name) and node.func.id == 'dict':
            for keyword in node.keywords:
                if keyword.arg == 'direct_fixed_offset_only':
                    self.mark('no_direct_initialization_offset', keyword.value)
                    keyword.value = ast.Constant(False)
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Or)
                and any(ast.unparse(v) == 'digest(Path(__file__)) != worker_sha' for v in node.values)):
            self.mark('verify_two_adapter_files_unchanged_at_endpoint', node)
            node.values.append(ast.UnaryOp(op=ast.Not(), operand=ast.Call(
                func=ast.Name(id='core_files_unchanged', ctx=ast.Load()), args=[], keywords=[])))
        return node


def derive_worker():
    base = CAPSULE/'train_worker.py'
    if sha(base) != BASE_WORKER_SHA:
        raise ValueError('Published native Worker changed')
    original = ast.parse(base.read_text(), filename=str(base))
    transformed = copy.deepcopy(original)
    adapter = Derivation()
    transformed = adapter.visit(transformed)
    expected = {
        'register_single_new_arm': 1, 'new_experiment_schema': 1,
        'read_original_immutable_source_registry': 1, 'route_only_new_actor_class': 1,
        'new_arm_factory_and_initialization_flag': 2, 'training_never_opens_confirmation': 1,
        'disclose_core_deletion_scope': 1, 'disclose_complete_endpoint_seal': 1,
        'disclose_object_head_deletion': 1, 'record_core_derivation_identity': 1,
        'no_direct_initialization_offset': 1, 'verify_two_adapter_files_unchanged_at_endpoint': 1,
        'development_profile_identity': 1, 'disclose_random_evaluation': 1,
        'bind_episode_random_object_stream': 2,
    }
    counts = {kind: sum(c['kind'] == kind for c in adapter.changes) for kind in expected}
    if counts != expected or len(adapter.changes) != sum(expected.values()):
        raise AssertionError(('Unexpected native Worker AST shape', counts))
    from source_runtime import relocate_tree
    transformed = relocate_tree(transformed)
    ast.fix_missing_locations(transformed)
    policy_name = '_chapter03_random_object_policy'
    if policy_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(policy_name, HERE/'policy.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[policy_name] = module
        spec.loader.exec_module(module)
    policy = sys.modules[policy_name]
    source = str(CAPSULE.parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    if str(CAPSULE) not in sys.path:
        sys.path.insert(0, str(CAPSULE))
    execution_files = {p.name: sha(p) for p in (HERE/'policy.py', HERE/'train.py')}
    identity = dict(schema='chapter3-core-controlled-worker-derivation-v1',
        base=policy.frozen_identity(), adapter_files_sha256=execution_files,
        adapter_directory=str(HERE),
        base_worker_ast_sha256=hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
        executed_worker_ast_sha256=hashlib.sha256(ast.dump(transformed, include_attributes=False).encode()).hexdigest(),
        exact_AST_changes=adapter.changes, collection_update_physics_statements_unchanged=True,
        parent_core_policy_sha256=sha(policy.CORE/'policy.py'),
        random_namespace=policy.RANDOM_NAMESPACE,
        intervention='same-seed Full construction then delete spatial.head and w Gaussian event; retain quota context/head/sigma and native Full mask; independent per-episode Uniform object weights remain random in train and DEV with original capped-simplex execution')
    namespace = dict(__name__='_chapter03_random_object_derived_worker', __file__=str(base),
                     EXECUTION_IDENTITY=identity, _chapter_resolve=_resolve_path,
                     episode_with_random_object=policy.episode_with_random_object,
                     core_files_unchanged=lambda: execution_files == {
                         p.name: sha(p) for p in (HERE/'policy.py', HERE/'train.py')})
    exec(compile(transformed, str(base)+' [registered core deletion AST adapter]', 'exec'), namespace)
    return namespace, identity


def main():
    namespace, identity = derive_worker()
    args = namespace['parser']().parse_args()
    if args.arm != ARM:
        raise ValueError('Only the registered random-object development ablation is authorized here')
    if Path(args.source_root).resolve() != (CAPSULE.parent).resolve():
        raise ValueError('Use the exact published native scientific package')
    if args.path_master != 202611043101:
        raise ValueError('The random-object ablation reuses the original paired training/DEV namespace')
    if args.final_master != FINAL_MASTER:
        raise ValueError('Use registered core confirmation master202611093799; training never opens it')
    output = Path(args.output).resolve()
    protected = (CAPSULE.resolve(), HERE.resolve(),
                 Path(str(_resolve_path('/hy-tmp/resource_allocation_results/chapter03/models'))).resolve(),
                 Path(str(_resolve_path('/hy-tmp/resource_allocation_results/chapter03/experiments/final_organized'))).resolve())
    if any(output.is_relative_to(p) for p in protected):
        raise ValueError('A new independent output directory is required')
    return namespace['train'](args)


if __name__ == '__main__':
    main()
