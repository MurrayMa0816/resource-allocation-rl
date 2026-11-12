"""New-scope PROPS training/evaluation, preserving author optimizer equations."""
from pathlib import Path
import sys,traceback,time
SCOPE=Path('/hy-tmp/resource_allocation_results/chapter4/formal_100k_5seed_20261106')
sys.path.insert(0,str(SCOPE/'source/props_pgr'))
from formal_support import Context,parser,module,bindings,dump,sha,HERE

def main(args):
    ctx=Context(args);before=None
    try:
        Env=ctx.install()
        import numpy as np,torch
        torch.set_num_threads(1)
        if args.device.startswith('cuda')and not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
        runner=module(HERE/'native_loop.py','_c4formal_props_native')
        before=bindings()
        config=dict(method='PROPS',domain=args.domain,seed=args.seed,output_dir=str(ctx.out),device=args.device,torch_threads=1,
            total_steps=args.steps,episodes=args.steps//(32 if args.domain=='radio'else 24),
            representation_variant='matched_sf',smoke=False,gamma=1.,gae_lambda=.95,
            learning_rate=5e-5,props_learning_rate=1e-4,target_episodes=4,behavior_episodes=1,buffer_episodes=8,
            update_epochs=10,props_update_epochs=16,num_minibatches=32,props_num_minibatches=16,
            clip_coef=.2,props_clip_coef=.3,ent_coef=.01,props_lambda=.3,target_kl=.03,anneal_lr=True,
            eval_every_episodes=args.eval_every//(32 if args.domain=='radio'else 24),eval_episodes=args.dev_cases,
            path_master=args.train_master,development_master=args.dev_master,deterministic_eval=True,
            holdout_paths=0,no_holdout=True,engineering_gate=args.engineering_gate,
            parameter_selection='user-specified target half lr; prior matched sensitivity recommends default; no validation-optimal claim')
        def evaluate(_Env,domain,seed,agent,device,episodes,train_steps):
            def action(obs):
                with torch.no_grad():return agent.get_action(torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0),noise=False)[0].cpu().numpy()
            return ctx.evaluate(action,train_steps,action_mode='raw',cases=episodes,policy=agent)
        runner._evaluate=evaluate
        native_append=runner._append_jsonl
        def append(path,rows):
            native_append(path,rows)
            if Path(path).name=='training.jsonl'and rows:
                row=rows[-1];ctx.training_steps=int(row['train_env_steps']);ctx.progress('training',{'target_updates':row.get('target_update_count'),'behavior_updates':row.get('behavior_update_count')})
        runner._append_jsonl=append
        if args.evaluate_model:
            saved=torch.load(args.evaluate_model,map_location=args.device,weights_only=False)
            cfg=saved['config']
            if cfg['domain']!=args.domain or int(cfg['seed'])!=args.seed:raise ValueError('model identity mismatch')
            facade_env=Env(args.domain,args.seed,phase=args.eval_phase,action_mode='raw',holdout=args.eval_phase=='final')
            from types import SimpleNamespace
            Agent,*_=runner._load_official()
            Agent=runner.configure_agent_factory(Agent,facade_env.native_spec,args.seed,cfg)
            facade=SimpleNamespace(single_observation_space=facade_env.observation_space,single_action_space=facade_env.action_space)
            agent=Agent(facade).to(args.device);agent.load_state_dict(saved['target'],strict=True);facade_env.close()
            def action(obs):
                with torch.no_grad():return agent.get_action(torch.as_tensor(obs,dtype=torch.float32,device=args.device).unsqueeze(0),noise=False)[0].cpu().numpy()
            result=ctx.evaluate(action,int(saved['train_env_steps']),action_mode='raw',phase=args.eval_phase,cases=args.eval_cases,policy=agent)
            assert bindings()==before
            return ctx.completion(result,args.evaluate_model,before,evaluation_only=True)
        H=32 if args.domain=='radio'else 24
        if args.steps<8*H or args.steps%(4*H)or args.eval_every%(4*H):raise ValueError('complete native target batches required')
        dump(ctx.out/'protocol.json',dict(schema='chapter4-formal-props-v1',arguments=vars(args),source_binding=before,config=config,native_update_functions_verbatim=True,architecture='matched_sf: A set actor, flat critic',deterministic_development=True,one_fixed_endpoint=True,optimizer_or_replay_saved=False))
        result=runner.run(config);ctx.training_steps=int(result.get('train_env_steps',args.steps))
        result['evaluation_policy']='deterministic_target_Gaussian_mean_fixed_dev_paths_training_RNG_restored'
        result['engineering_only']=args.engineering_gate
        dump(ctx.out/'result.json',result)
        points=[__import__('json').loads(x)for x in (ctx.out/'evaluations.jsonl').read_text().splitlines()]
        grid=list(range(0,args.steps+1,args.eval_every))
        if grid[-1]!=args.steps:grid.append(args.steps)
        assert [x['train_env_steps']for x in points]==grid
        assert all(len(x['returns'])==args.dev_cases and x['deterministic_eval']for x in points)
        assert ctx.training_steps==args.steps and ctx.resource_counts.get('train')==args.steps//H
        assert bindings()==before
        return ctx.completion(result,ctx.out/'checkpoint.pt',before)
    except BaseException:
        dump(ctx.out/'failed.json',dict(status='failed',traceback=traceback.format_exc(),source_binding=before,training_steps=ctx.training_steps,wall_seconds=time.perf_counter()-ctx.start,exact_resume_available=False));raise
    finally:ctx.close()

if __name__=='__main__':main(parser('props').parse_args())
