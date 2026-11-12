"""Source-bound five-seed MAD-TD formal training and released independent TEST."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/hy-tmp/resource_allocation_results/chapter4/mad_td_formal_20261107')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import io_support as support
from endpoint_release import source_bindings,validate_release


def sources():
    return source_bindings(required=(ROOT/'protocol/science_code_seal.json').exists())


def finish(ctx,receipt,model,before,*,evaluation_only=False):
    result=ctx.completion(receipt,model,before,evaluation_only=evaluation_only)
    formal_train=ctx.args.phase=='train'and not ctx.args.engineering_gate
    result.update(schema='chapter4-madtd-formal-native-completion-v1',phase='final'if ctx.args.phase=='test'else ctx.args.phase,
        formal_training_performed=formal_train,formal_evaluation=ctx.args.phase=='test',
        scientific_pilot=False,full_formal_training_budget_executed=formal_train and ctx.training_steps==100608,
        holdout_cases=256 if ctx.args.phase=='test'else 0,
        science_code_seal_sha256=support.sha(ROOT/'protocol/science_code_seal.json')if(ROOT/'protocol/science_code_seal.json').exists()else None)
    support.dump(ctx.out/'completion.json',result);support.dump(ctx.out/'completed.json',result)
    return result


def load_agent():
    directory = ROOT / 'source/method/mad_td'
    name = '_c4_madtd_frozen_adapter'
    spec = importlib.util.spec_from_file_location(name, directory/'__init__.py', submodule_search_locations=[str(directory)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    from importlib import import_module
    return import_module(name+'.agent').Agent


def rng_digest(value):
    import pickle
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def runtime_binding(agent):
    import jax
    return dict(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                JAX_PLATFORMS=os.environ.get('JAX_PLATFORMS'), backend=jax.default_backend(),
                actual_device=str(agent.jax_device), platform=agent.jax_device.platform,
                cpu_affinity=sorted(os.sched_getaffinity(0)), jax_version=jax.__version__)


class Context(support.Context):
    def evaluate(self, agent, steps, *, cases=None, phase='dev'):
        import numpy as np
        cases = self.args.dev_cases if cases is None else int(cases)
        saved = (random.getstate(), np.random.get_state(), copy.deepcopy(agent.get_rng_state()))
        before = rng_digest(saved)
        env = None
        values = []
        self.progress('holdout'if phase=='final'else'development')
        try:
            random.seed(4000000)
            np.random.seed(4000000)
            env = self.Env(self.args.domain, self.args.seed, phase=phase, action_mode='tanh_latent', holdout=phase=='final')
            for i in range(cases):
                obs, _ = env.reset(options={'episode_index':i})
                done = False
                value = 0.
                while not done:
                    action = agent.policy_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    if truncated:
                        raise AssertionError('True finite terminal task only')
                    value += float(reward)
                    done = terminated
                values.append(value)
                self.eval_writer.writerow(dict(training_step=steps, phase=phase, path_index=i,
                    path_seed=env.task.plan(phase,i)['task_seed'], native_return=value,
                    resource_fraction=env.task.env.last_full_spend_audit['total_used']/env.native_spec.initial_budget,
                    steps=env.horizon))
            count = env.real_steps
        finally:
            if env is not None:
                env.close()
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            agent.set_rng_state(saved[2])
        if rng_digest((random.getstate(), np.random.get_state(), agent.get_rng_state())) != before:
            raise AssertionError('Development evaluation changed training RNG')
        self.evaluation_steps += count
        self.eval_file.flush()
        row = dict(training_step=steps, phase=phase, paths=cases, evaluation_environment_steps=count,
                   native_mean=float(np.mean(values)), native_std=float(np.std(values,ddof=1)) if cases>1 else 0.,
                   min_return=float(min(values)), max_return=float(max(values)))
        if not all(math.isfinite(x) for x in values):
            raise FloatingPointError('Nonfinite DEV return')
        self.curve_writer.writerow(row)
        self.curve_file.flush()
        self.progress('training')
        return dict(train_env_steps=steps, evaluation_env_steps=count, episodes=cases,
                    mean_return=row['native_mean'], mean_score=row['native_mean']*100,
                    std_return=row['native_std'], returns=values, phase=phase, deterministic_eval=True,
                    training_rng_restored=True, evaluation_master=self.args.eval_master if phase=='final'else self.args.dev_master)


def main(args):
    # Validate identity and unopened TEST barrier before creating a Context,
    # importing an accelerator backend or constructing a physical environment.
    if args.eval_master != 2026110694199 or args.eval_cases != 256:
        raise ValueError('Independent TEST uses the preregistered256-case final panel')
    if not args.engineering_gate:
        if args.seed not in range(5) or (args.train_master,args.dev_master)!=(202611043101,202611043101):
            raise ValueError('Unregistered formal seed/development panel')
        if not (ROOT/'protocol/science_code_seal.json').is_file():
            raise RuntimeError('Formal execution requires the frozen training source seal')
    if args.engineering_replay_capacity_steps and not args.engineering_gate:
        raise ValueError('Explicit replay capacity override is engineering-only')
    if args.phase=='train' and args.model:
        raise ValueError('Formal training starts from zero; model loading is evaluation-only')
    if args.phase=='reload-dev' and (not args.engineering_gate or not args.model):
        raise ValueError('reload-dev is a separately charged engineering check')
    release=None
    if args.phase=='test':
        if args.engineering_gate or not args.model:
            raise ValueError('Independent TEST requires a released formal endpoint')
        release=validate_release('mad_td',args.domain,args.seed,args.model)
    ctx = Context(args)
    before = sources()
    train_log = None
    env = None
    try:
        import numpy as np
        Env = ctx.install()
        Agent = load_agent()
        random.seed(args.seed)
        np.random.seed(args.seed)
        H = 32 if args.domain == 'radio' else 24
        if args.phase in ['test','reload-dev']:
            model=Path(args.model).resolve();agent=Agent.load(model,args.device)
            if(agent.domain,agent.seed)!=(args.domain,args.seed):raise AssertionError('Endpoint identity mismatch')
            if args.phase=='test'and agent.steps!=100608:raise AssertionError('Independent TEST requires the fixed100608 endpoint')
            phase='final'if args.phase=='test'else'dev';cases=256 if phase=='final'else args.dev_cases
            result=ctx.evaluate(agent,agent.steps,cases=cases,phase=phase)
            if phase=='dev':
                original=json.loads((model.parent/'evaluations.json').read_text())[-1]
                if result['returns']!=original['returns']:raise AssertionError('Fresh-process DEV endpoint mismatch')
            receipt=dict(status='PASS',method='MAD-TD',formal_training_performed=False,formal_evaluation=phase=='final',
                engineering_only=args.engineering_gate,training_steps=0,evaluation_steps=ctx.evaluation_steps,
                phase=phase,path_master=args.eval_master if phase=='final'else args.dev_master,cases=cases,
                summary=result,returns=result['returns'],holdout_cases=cases if phase=='final'else 0,
                new_optimizer_steps=0,deterministic_eval=True,evaluation_master=args.eval_master if phase=='final'else args.dev_master,
                endpoint_release_gate_sha256=release['gate_sha256']if release else None,model_sha256=support.sha(model),endpoint_release=release,
                same_endpoint_DEV_panel_exact=True if phase=='dev'else None)
            receipt['runtime_binding']=runtime_binding(agent)
            support.dump(ctx.out/'runtime_binding.json',receipt['runtime_binding'])
            if release is not None and validate_release('mad_td',args.domain,args.seed,model)!=release:
                raise AssertionError('Endpoint release changed during independent TEST')
            support.dump(ctx.out/('test_receipt.json'if phase=='final'else'reload_receipt.json'),receipt)
            support.dump(ctx.out/'evaluations.json',[result])
            if sources()!=before:raise AssertionError('Frozen implementation changed during endpoint evaluation')
            return finish(ctx,receipt,model,before,evaluation_only=True)
        if not args.engineering_gate and (args.steps, args.eval_every, args.dev_cases) != (100608,768,32):
            raise ValueError('Unregistered formal training budget')
        if args.steps % H or args.eval_every % H:
            raise ValueError('Whole physical episodes required')
        env = Env(args.domain, args.seed, phase='train', action_mode='tanh_latent', holdout=False)
        if (env.observation_dim, env.action_dim) != ((305,19) if args.domain == 'radio' else (244,13)):
            raise AssertionError('Task observation/action identity changed')
        obs, _ = env.reset(options={'episode_index':0})
        agent_capacity=args.engineering_replay_capacity_steps or args.steps
        if agent_capacity<args.steps:raise ValueError('Engineering capacity cannot be smaller than actual prefix')
        agent = Agent(dict(seed=args.seed,total_steps=agent_capacity,domain=args.domain,gamma=1.,reward_scale=1.,value_support=[-1.,1.]),
                      env.observation_dim, env.action_dim, device=args.device)
        if args.engineering_gate:
            # Explicit prefix gate only. Every transition is genuinely collected;
            # no unpaid or copied warmup is inserted into scientific replay.
            agent.warmup = H
        warmup = agent.warmup
        support.dump(ctx.out/'protocol.json', dict(method='MAD-TD', scientific_pilot=False,
            formal_training_performed=not args.engineering_gate, arguments=vars(args), H=H,
            observation_dim=env.observation_dim, action_dim=env.action_dim,
            source_binding=before, warmup_charged_real_steps=warmup,
            UTD=8, batch_size=512, real_rows=486, generated_rows=26,
            gamma=1., value_support=[-1.,1.], bins=151, terminal_supervised=True,
            public_reward_unchanged=True, fullspend=True, use_mpc=False,
            internal_validation_replay_fraction=.05, internal_validation_is_not_benchmark_DEV=True,
            holdout_cases=0, phase='train', replay_capacity_steps=agent_capacity, one_fixed_inference_endpoint=True, optimizer_replay_or_periodic_checkpoints=False))
        support.dump(ctx.out/'runtime_binding.json', runtime_binding(agent))
        evaluations = [ctx.evaluate(agent,0)]
        support.dump(ctx.out/'evaluations.json', evaluations)
        train_log = (ctx.out/'training.jsonl').open('x')
        episode_return = 0.
        episode_index = 0
        update_seconds = 0.
        environment_seconds = 0.
        for step in range(1,args.steps+1):
            action = agent.act(obs)
            start = time.perf_counter()
            nxt, reward, terminal, truncated, info = env.step(action)
            environment_seconds += time.perf_counter()-start
            if truncated:
                raise AssertionError('Truncation changed the true terminal reward contract')
            if step == warmup+1:
                ctx.progress('native_update_compilation')
            start = time.perf_counter()
            agent.observe(obs,action,reward,nxt,terminal)
            update_seconds += time.perf_counter()-start
            ctx.training_steps = step
            episode_return += float(reward)
            obs = nxt
            if terminal:
                train_log.write(json.dumps(dict(train_env_steps=step,episode=episode_index,
                    native_return=episode_return,score=episode_return*100,actor_critic_updates=agent.n_updates),allow_nan=False)+'\n')
                train_log.flush()
                episode_index += 1
                episode_return = 0.
                if step < args.steps:
                    obs, _ = env.reset()
            if step % 128 == 0 or step == args.steps:
                diag = agent.diagnostics()
                ctx.progress('training',dict(actor_updates=diag['policy_optimizer_steps'],
                    critic_updates=diag['critic_optimizer_steps'],model_updates=diag['model_optimizer_steps'],
                    terminal_rows_learned=diag['sampled_real_terminal_rows']))
            if step % args.eval_every == 0 or step == args.steps:
                evaluations.append(ctx.evaluate(agent,step))
                support.dump(ctx.out/'evaluations.json', evaluations)
        diag = agent.diagnostics()
        expected_calls = args.steps - warmup
        if diag['learning_calls'] != expected_calls or diag['updates'] != 8*expected_calls:
            raise AssertionError('Native training update counters')
        if diag['model_optimizer_steps'] != expected_calls or diag['encoder_optimizer_steps'] != expected_calls:
            raise AssertionError('Native slow-model update ratio')
        if diag['sampled_real_terminal_rows'] <= 0 or diag['training_terminal_transitions'] <= 0:
            raise AssertionError('Terminal reward never reached the real learner')
        if ctx.resource_counts.get('train') != args.steps//H or env.real_steps != args.steps:
            raise AssertionError('Paid physical training budget')
        import jax
        for name in ('critic','critic_target','actor','encoder','encoder_target','latent_model'):
            model_state = getattr(agent.models,name)
            for leaf in jax.tree_util.tree_leaves((model_state.params, model_state.opt_state, model_state.step)):
                value = np.asarray(jax.device_get(leaf))
                if not np.isfinite(value).all():
                    raise FloatingPointError(('Nonfinite final native parameter/optimizer',name))
        model = ctx.out/'checkpoint.msgpack'
        agent.save(model)
        loaded = Agent.load(model,args.device)
        error = float(np.max(np.abs(agent.policy_action(obs)-loaded.policy_action(obs))))
        if error != 0.:
            raise AssertionError(('Inference parameter roundtrip action error',error))
        if sources() != before:
            raise AssertionError('Frozen scientific source changed')
        receipt = dict(status='PASS',method='MAD-TD',scientific_pilot=False,formal_training_performed=not args.engineering_gate,
            engineering_only=args.engineering_gate,training_steps=args.steps,evaluation_steps=ctx.evaluation_steps,
            holdout_cases=0,first_learning_step=warmup+1,expected_learning_calls=expected_calls,
            expected_actor_critic_updates=8*expected_calls,diagnostics=diag,
            checkpoint_action_abs_error=error,checkpoint_kind='inference_encoder_actor_only',
            environment_seconds=environment_seconds,observe_update_seconds=update_seconds,
            context_wall_seconds_including_compilation_DEV_and_IO=time.perf_counter()-ctx.start)
        receipt['runtime_binding'] = runtime_binding(agent)
        support.dump(ctx.out/'receipt.json',receipt)
        return finish(ctx,receipt,model,before)
    except BaseException:
        support.dump(ctx.out/'failed.json',dict(status='FAIL',traceback=traceback.format_exc(),
            training_steps=ctx.training_steps,evaluation_steps=ctx.evaluation_steps,
            fullspend_episode_counts=ctx.resource_counts,source_binding=before))
        raise
    finally:
        if env is not None:
            env.close()
        if train_log is not None:
            train_log.close()
        ctx.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--domain',choices=('radio','v5'),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=100608)
    p.add_argument('--phase',choices=['train','reload-dev','test'],default='train')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--eval-every',type=int,default=768)
    p.add_argument('--dev-cases',type=int,default=32)
    p.add_argument('--train-master',type=int,default=202611043101)
    p.add_argument('--dev-master',type=int,default=202611043101)
    p.add_argument('--eval-master',type=int,default=2026110694199)
    p.add_argument('--eval-cases',type=int,default=256)
    p.add_argument('--environment-root',default=str(ROOT/'source/environment'))
    p.add_argument('--engineering-gate',action='store_true')
    p.add_argument('--model')
    p.add_argument('--engineering-replay-capacity-steps',type=int,default=0)
    p.set_defaults(method='mad_td')
    main(p.parse_args())
