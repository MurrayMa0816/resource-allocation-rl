"""Shared public-budget conditions for frozen Chapter5 policies.

The immutable native Task already accepts a fully specified physical condition.
This module uses that existing interface. Only the initial episode resource is
changed; demands, transition laws, instantaneous caps, horizon and reward stay
identical. Both flat baseline Env and structured Full use the same Task/Spec.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys

SCHEMA='chapter5-budget-condition-adapter-v1'
SCALES=(.8,1.,1.2)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256((json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def condition_payload(bridge,domain,scale):
    require(domain in ('radio','v5')and float(scale)in SCALES,'Unknown registered budget condition')
    bridge.bootstrap()
    build=importlib.import_module('allocation23.environments').build_environment
    environment,spec,physical=build(domain,3,radio_groups=100)
    if hasattr(environment,'close'):environment.close()
    physical=json.loads(json.dumps(physical))
    changed=dict(physical)
    key='episode_budget_override_J'if domain=='radio'else'initial_budget'
    # 1.0 is an exact physical-config identity reference, including optional None.
    if float(scale)!=1.:
        changed[key]=spec.initial_budget*float(scale)*(spec.action_cost_scale if domain=='radio'else 1.)
    difference=[k for k in physical if changed[k]!=physical[k]]
    require(difference==([] if float(scale)==1. else [key]),'A condition changed more than initial episode resource')
    expected=spec.initial_budget*float(scale)
    payload=dict(schema='allocation23-completion-physical-condition-v1',
        metadata=dict(chapter=3,domain=domain,condition_id=f'chapter5-budget-{domain}-{float(scale):.1f}',family='initial_budget_only',level=f'{float(scale):.1f}'),
        physical_config=changed,
        invariants=dict(adapter_schema=SCHEMA,scale=float(scale),baseline_physical_sha256=object_sha(physical),
            changed_fields=difference,initial_budget_in_action_units=expected,
            baseline_initial_budget_in_action_units=spec.initial_budget,groups=spec.groups,horizon=spec.horizon,
            demands_and_physics_unchanged=True,per_object_caps_unchanged=True,
            nominal_period_resource_unchanged=True,source_native_manifest_sha256=sha(bridge.HERE/'native_source_manifest.json')))
    return payload


def verify_condition(bridge,domain,scale,condition_file,condition_sha256=None):
    if condition_file is None:
        require(float(scale)==1.,'A nondefault budget requires an explicit immutable condition file')
        return None
    path=Path(condition_file).resolve(strict=True)
    if condition_sha256 is not None:require(sha(path)==condition_sha256,'Budget condition SHA changed')
    actual=json.loads(path.read_text());expected=condition_payload(bridge,domain,float(scale))
    require(actual==expected,'Budget condition differs from the initial-resource-only definition')
    return path


def make_task(bridge,domain,seed,*,path_master,holdout,data_root,condition_file=None):
    bindings=bridge.bootstrap(verify_source=False)
    task=bindings['Task'](domain,3,'A',int(seed),path_master=int(path_master),holdout=bool(holdout),
        radio_groups=100,data_root=bridge.explicit_data_root(domain,data_root),condition_file=condition_file)
    require(task.spec.chapter==task.env.chapter==3,'A condition escaped the whole-budget native task')
    return task


def make_env(env_module,domain,seed,*,phase,path_master,holdout,action_mode,data_root,condition_file=None):
    bridge=env_module.bridge
    if condition_file is None:
        return env_module.Env(domain,seed,phase=phase,path_master=path_master,holdout=holdout,
            action_mode=action_mode,data_root=data_root)
    class BudgetEnv(env_module.Env):
        def _make_task(self,task_seed):
            return make_task(bridge,self.domain,task_seed,path_master=self.path_master,holdout=self.holdout,
                             data_root=self.data_root,condition_file=condition_file)
    return BudgetEnv(domain,seed,phase=phase,path_master=path_master,holdout=holdout,
        action_mode=action_mode,data_root=data_root)


def assert_spec(spec,domain,scale):
    expected=(.48 if domain=='radio'else 20160.)*float(scale)
    require(abs(spec.initial_budget-expected)<=max(1e-12,abs(expected)*1e-12),'Actual Task budget differs from registered condition')
    require(spec.groups==(100 if domain=='radio'else 120)and spec.horizon==(32 if domain=='radio'else 24),
            'Budget condition changed native geometry')
    return dict(initial_budget=spec.initial_budget,baseline_initial_budget=expected/float(scale),
                groups=spec.groups,horizon=spec.horizon,scale=float(scale),action_cost_scale=spec.action_cost_scale)


def prepare(source_root,output_dir):
    source_root=Path(source_root).resolve(strict=True);out=Path(output_dir).resolve()
    require(out.is_relative_to('/hy-tmp'),'Conditions must be on HY')
    sys.path.insert(0,str(source_root/'_internal/src'));sys.path.insert(0,str(source_root))
    bridge=importlib.import_module('chapter5_full_spend_20261104.bridge')
    records=[]
    for domain in ('radio','v5'):
        for scale in SCALES:
            payload=condition_payload(bridge,domain,scale)
            path=out/f'{domain}__scale{scale:.1f}.json'
            require(not path.exists(),'Do not overwrite a registered budget condition')
            write(path,payload);records.append(dict(domain=domain,scale=scale,path=str(path),sha256=sha(path)))
    return dict(schema=SCHEMA,status='PREPARED_ONLY_NO_PHYSICAL_STEPS',conditions=records,
                environment_resets=0,physical_steps=0,training_steps=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source_root,args.output_dir),ensure_ascii=False))
