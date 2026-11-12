"""Explicit one-time process binding; rejects mixed old/new task imports."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

FROZEN_C4 = Path('/root/resource_allocation_run_storage_20261101/chapter4_recent_baselines_formal_20261102_source')
_INSTALLED = None


def install(*, audit_dir=None, context=None):
    """Call before importing a runner, then use its normal public API."""
    global _INSTALLED
    if _INSTALLED is not None:
        raise RuntimeError('Full-spend bootstrap is one-time; do not change a running task binding')
    forbidden = [name for name in sys.modules if name == 'allocation23' or name.startswith('allocation23.')
                 or name == 'chapter4_recent_baselines.env_adapter'
                 or name.startswith('chapter4_recent_baselines._released_allocation23')
                 or name.startswith('chapter4_recent_baselines.runners.')
                 or name == 'chapter4_recent_baselines.common']
    if forbidden:
        raise RuntimeError('Old/task-dependent modules already imported: ' + ', '.join(sorted(forbidden)))
    sys.dont_write_bytecode = True
    frozen = str(FROZEN_C4)
    if frozen not in sys.path:
        sys.path.insert(0, frozen)
    importlib.import_module('chapter4_recent_baselines')
    from . import native
    from .task import Task, configure_audit
    identity = native.verify_sources()
    configure_audit(audit_dir, context)
    # This replacement is local to this fresh process/private namespace. The
    # copied native source and all historical source files remain unchanged.
    native.task_module.Task = Task
    native.load('learning')
    for name, module in list(sys.modules.items()):
        if name == native._RELEASE_PACKAGE or name.startswith(native._RELEASE_PACKAGE + '.'):
            alias = 'allocation23' + name[len(native._RELEASE_PACKAGE):]
            if alias in sys.modules and sys.modules[alias] is not module:
                raise RuntimeError('Conflicting native alias: ' + alias)
            sys.modules[alias] = module
    from . import env_adapter
    sys.modules['chapter4_recent_baselines.env_adapter'] = env_adapter
    parent = sys.modules['chapter4_recent_baselines']
    parent.env_adapter = env_adapter
    bridge_files = {str(p.relative_to(native.ENV_ROOT)): native.digest(p)
                    for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity.update(schema='chapter4-shared-full-spend-binding-v1', context=dict(context or {}),
        bridge_sha256=bridge_files, historical_C4_source=str(FROZEN_C4),
        env_import='chapter4_recent_baselines.env_adapter.Env',
        task_import='allocation23.task.Task',
        actual_quota_mapping='Copied C3 public [L,U] full-spend transform with exact endpoint branches',
        Gaussian_fields_unchanged=['raw','log_prob','mask','behavior_mean','behavior_sigma'],
        learner_information_unchanged=True, ledger_is_audit_only=True,
        inherited_quota_descriptor_is_not_executed_math=True)
    if audit_dir is not None:
        path = Path(audit_dir) / 'bridge_identity.json'
        if path.exists():
            raise FileExistsError('Never overwrite a task binding receipt: ' + str(path))
        path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    _INSTALLED = identity
    return identity
