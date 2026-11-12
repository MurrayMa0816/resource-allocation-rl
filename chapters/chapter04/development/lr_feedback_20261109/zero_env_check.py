"""Source and scalar-controller checks only: no Torch, models or environments."""

import sys as _portable_sys
from pathlib import Path as _PortablePath
_PORTABLE_CODE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents if (p / 'portable_paths.py').is_file())
if str(_PORTABLE_CODE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_CODE_ROOT))
from portable_paths import project_root as _project_root, final_data_root as _data_root, resolve_path as _resolve_path, eurosat_root as _eurosat_root
from pathlib import Path
import ast, copy, hashlib, json, math
import numpy as np

ROOT=Path(__file__).resolve().parent
PARENT=Path(str(_resolve_path('/hy-tmp/resource_allocation_results/chapter04/experiments/double_random_formal_20261107/execution/f1_worker.py')))
NATIVE=Path(str(_resolve_path('/root/gamewar/resource_allocation_rl/frozen/dependencies/chapter04/recent_baselines_formal_20261102_source/chapter4_sample_efficiency_v5/allocation_efficiency/frequent.py')))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    candidate=ast.parse((ROOT/'candidates.py').read_text())
    namespace=dict(math=math,EPSILON=.2/1.5,ACTOR_LR_BOUNDS=(1e-5,1e-3))
    selected=[n for n in candidate.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name=='next_rate']
    exec(compile(ast.Module(body=selected,type_ignores=[]),'candidate_scalar','exec'),namespace)
    native=ast.parse(NATIVE.read_text())
    fn=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='adjusted_lr')
    original=dict(math=math,np=np,EPSILON=.2/1.5,TV_ALPHA=.03,TV_BETA=.5,ACTOR_LR_BOUNDS=(1e-5,1e-3))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_scalar','exec'),original)
    count=0
    for lr in [1e-5,1.1e-5,1e-4,3e-4,9.9e-4,1e-3]:
        for change in [0,.02,.2/6,.04,.05,.2/3,.075,.10,.12,1.]:
            expected=original['adjusted_lr'](lr,change)
            assert namespace['next_rate']('original_feedback',lr,change)==expected
            assert namespace['next_rate']('fixed_3e4',lr,change)==(3e-4,'fixed_no_feedback')
            assert namespace['next_rate']('fixed_1e4',lr,change)==(1e-4,'fixed_no_feedback')
            new,action=namespace['next_rate']('relaxed_feedback',lr,change)
            want=(min(max(lr/1.03 if change>.10 else lr*1.03 if change<.05 else lr,1e-5),1e-3),
                  'decrease' if change>.10 else 'increase' if change<.05 else 'hold')
            assert (new,action)==want
            count+=4
    for name in namespace['METHODS']:
        for lr,d in [(float('nan'),.05),(float('inf'),.05),(3e-4,float('nan')),(3e-4,-1),(0,.05)]:
            try:namespace['next_rate'](name,lr,d)
            except ValueError:pass
            else:raise AssertionError('Invalid controller input accepted')
    a=ast.parse(PARENT.read_text());b=ast.parse((ROOT/'worker.py').read_text())
    fa={n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    fb={n.name:ast.dump(n,include_attributes=False) for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    unchanged=['digest','write','CSV','cache_immutable_data_digests','rng_snapshot','restore_rng','summarize_guard','evaluate','writers','close_all','artifact_hashes','prepare_output']
    assert all(fa[k]==fb[k] for k in unchanged)
    assert 'evaluate_model' not in fb
    update=next(n for n in candidate.body if isinstance(n,ast.ClassDef) and n.name=='DevelopmentLR')
    update=next(n for n in update.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    calls=[n for n in ast.walk(update) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step') for n in calls)
    native_calls=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='update' and isinstance(n.func.value,ast.Call) and isinstance(n.func.value.func,ast.Name) and n.func.value.func.id=='super']
    assert len(native_calls)==1
    result=dict(status='PASS',scalar_cases=count,invalid_input_cases=20,
                parent_worker_helpers_AST_exact=unchanged,wrapper_native_update_calls=1,
                wrapper_gradient_or_optimizer_step_calls=0,no_TEST_entrypoint=True,
                scientific_execution=dict(environment_steps=0,model_loads=0,inference=0,training=0),
                source_sha256={str(p):sha(p) for p in [ROOT/'worker.py',ROOT/'candidates.py',PARENT,NATIVE]})
    return result

if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
