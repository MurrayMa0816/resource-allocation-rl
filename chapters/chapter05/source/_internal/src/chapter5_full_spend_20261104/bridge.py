"""Bind the copied, immutable full-spend native release before legacy learners.

The shared task/Actor enforce the new execution rule in training, development
and testing. Raw latent variables and their probability densities stay in the
learner's original coordinate system; no physical-action Jacobian is added.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
NATIVE_SOURCE=HERE/'native_source'
REVISION='chapter5-shared-native-full-spend-bridge-v1'
DATA_FILES=('images.npy','split_manifest.json','manifest.json','image_records.json')
_VERIFIED_DATA={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def assert_native_bindings():
    """Reject a mixed old/new allocation23 process, including cached submodules."""
    root=(NATIVE_SOURCE/'allocation23').resolve(strict=True)
    files={}
    for name,module in list(sys.modules.items()):
        if name!='allocation23' and not name.startswith('allocation23.'):continue
        filename=getattr(module,'__file__',None)
        if filename is None or not Path(filename).resolve().is_relative_to(root):
            raise RuntimeError('Foreign allocation23 module already loaded: '+name+' '+str(filename))
        files[name]=str(Path(filename).resolve())
    return files


def bootstrap(*,verify_source=True):
    """Import the copied175-file release, preserving scheduler CUDA selection."""
    root=NATIVE_SOURCE.resolve(strict=True)
    manifest=json.loads((HERE/'native_source_manifest.json').read_text())
    if manifest.get('schema')!='chapter5-full-spend-native-source-v1' or len(manifest['files_sha256'])!=175:
        raise RuntimeError('The full-spend native release must bind exactly175 source files')
    assert_native_bindings()
    if verify_source:
        for name,expected in manifest['files_sha256'].items():
            if sha(root/name)!=expected:raise RuntimeError('Copied native source changed: '+name)
    # Global allocation23 must be bound before allocation_morl or cross_server
    # can import its old chapter23_final path. Existing foreign bindings fail.
    location=str(root)
    if location in sys.path:sys.path.remove(location)
    sys.path.insert(0,location)
    importlib.invalidate_caches()
    task=importlib.import_module('allocation23.task')
    policy=importlib.import_module('allocation23.methods.policy')
    helper=importlib.import_module('allocation23.full_spend')
    spec=importlib.import_module('allocation23.spec')
    files=assert_native_bindings()
    if not hasattr(policy.Actor,'_full_spend_quota'):
        raise RuntimeError('Native Actor lacks the validated whole-budget transform')
    return dict(Task=task.Task,Actor=policy.Actor,Spec=spec.Spec,
        full_spend_bounds=helper.full_spend_bounds,
        assert_terminal_full_spend=helper.assert_terminal_full_spend,
        terminal_budget_tolerance=helper.terminal_budget_tolerance,
        native_source_root=str(root),native_module_files=files,
        source_manifest_sha256=sha(HERE/'native_source_manifest.json'))


def import_legacy(module_name):
    """Import read-only learner code after binding physics; undo CUDA masking."""
    bootstrap()
    present='CUDA_VISIBLE_DEVICES' in os.environ
    requested=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:return importlib.import_module(module_name)
    finally:
        if present:os.environ['CUDA_VISIBLE_DEVICES']=requested
        else:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        assert_native_bindings()


def explicit_data_root(domain,data_root=None):
    if domain!='radio':return None if data_root is None else Path(data_root).resolve(strict=True)
    manifest=json.loads((HERE/'data_manifest.json').read_text())
    selected=(data_root if data_root is not None else os.environ.get('CH5_FULL_SPEND_DATA_ROOT',manifest.get('data_root')))
    if selected is None:
        raise ValueError('Radio requires explicit data_root or CH5_FULL_SPEND_DATA_ROOT; no hidden old ROOT fallback')
    root=Path(selected).expanduser().resolve(strict=True)
    if not root.is_dir() or any(not (root/name).is_file() for name in DATA_FILES):
        raise ValueError('The full-spend image data root must contain all four bound prepared-data files')
    expected={name:item['sha256'] for name,item in manifest['files'].items()}
    fingerprints=tuple((name,(root/name).stat().st_dev,(root/name).stat().st_ino,(root/name).stat().st_size,
                        (root/name).stat().st_mtime_ns,(root/name).stat().st_ctime_ns) for name in DATA_FILES)
    cache_key=(str(root),sha(HERE/'data_manifest.json'))
    if _VERIFIED_DATA.get(cache_key)!=fingerprints:
        for name in DATA_FILES:
            if sha(root/name)!=expected[name]:raise RuntimeError('Registered prepared image data changed: '+name)
        _VERIFIED_DATA[cache_key]=fingerprints
    return root


def make_task(domain,seed=0,*,path_master,holdout=False,radio_groups=100,data_root=None):
    bindings=bootstrap(verify_source=False)
    task=bindings['Task'](domain,3,'A',int(seed),path_master=int(path_master),holdout=bool(holdout),
        radio_groups=int(radio_groups),data_root=explicit_data_root(domain,data_root))
    if task.spec.chapter!=3 or task.env.chapter!=3:
        raise AssertionError('Every Chapter5 full-spend learner must use the active native chapter3 task')
    return task


def make_decoder(spec,seed=0):
    bindings=bootstrap(verify_source=False)
    if spec.chapter!=3:raise ValueError('A whole-budget decoder requires native chapter3 feasibility')
    actor=bindings['Actor'](spec,'A_FULL','adaptive',int(seed)).cpu().eval()
    actor.requires_grad_(False)
    return actor


def make_pool(domain,seed,size,master,*,holdout=False,radio_groups=100,data_root=None):
    """Same explicit full-spend Task class used by Env and the Full collector."""
    if isinstance(size,bool) or int(size)!=size or size<1:raise ValueError('Positive complete native pool size required')
    first=make_task(domain,seed,path_master=master,holdout=holdout,radio_groups=radio_groups,data_root=data_root)
    result=[first]
    build=importlib.import_module('allocation23.environments').build_environment
    for unused in range(int(size)-1):
        task=copy.copy(first)
        task.env,task.spec,task.physical_dict=build(domain,3,radio_groups=first.spec.groups)
        task.image_sets=dict(first.image_sets)
        task.data_identity=copy.deepcopy(first.data_identity)
        result.append(task)
    return result


def make_models(spec,seed,method='fixed'):
    """Reuse the unchanged C/J learner architecture with the NEW native Actor."""
    module=import_legacy('chapter5_cross_server_20261103')
    models=importlib.import_module('allocation_morl.models')
    actor,critic,neural=models.make_models(spec,int(seed),method)
    bindings=bootstrap(verify_source=False)
    if type(actor) is not bindings['Actor'] or actor.spec.chapter!=3:
        raise AssertionError('Full models did not bind the shared full-spend Actor')
    assert_native_bindings()
    return actor,critic,neural


def interval(state,spec):
    return bootstrap(verify_source=False)['full_spend_bounds'](state,spec)


def assert_full_episode(remaining,total,spec):
    bindings=bootstrap(verify_source=False)
    if spec.chapter!=3:raise AssertionError('Full-episode accounting cannot use the old fixed or upper-only task')
    bindings['assert_terminal_full_spend'](remaining,spec)
    tolerance=bindings['terminal_budget_tolerance'](spec)
    if abs(float(total)-spec.initial_budget)>tolerance:
        raise ArithmeticError('Executed actions did not allocate the whole initial budget within native roundoff')
    return dict(full_spend_required=True,terminal_remaining=float(remaining),executed_total=float(total),
        initial_budget=spec.initial_budget,tolerance=tolerance,
        relative_error=abs(float(total)/spec.initial_budget-1.),budget_constraint='sum_t q_t = initial_budget')


def protocol():
    bindings=bootstrap()
    return dict(revision=REVISION,native_source_root=bindings['native_source_root'],
        source_manifest_sha256=bindings['source_manifest_sha256'],native_module_files=bindings['native_module_files'],
        full_spend_required=True,all_training_development_test_share_rule=True,
        quota_interval='public [L,U]; native exact neutral reference and endpoint branches',
        radio_future_capacity='pmax*sum_i min(max(packets_left_i-1,0),periods_left-1)',
        completed_radio_objects='zero power; physical caps unchanged',
        late_infeasible_force_spending=False,posthoc_fill_or_ledger_rewrite=False,
        probability_space='unchanged complete raw Gaussian or native tanh-latent event, never physical allocation')
