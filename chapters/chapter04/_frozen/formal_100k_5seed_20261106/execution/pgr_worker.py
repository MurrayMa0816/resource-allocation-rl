"""Native PGR complete-budget loop with registered task/evaluation/output hooks."""
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
        if not args.device.startswith('cuda')or not torch.cuda.is_available():raise RuntimeError('Native PGR registered CUDA path required')
        runner=module(HERE/'pgr_native_runner.py','_c4formal_pgr_native')
        from pgr_adaptations import install_pgr
        H=32 if args.domain=='radio'else 24
        config=dict(method='pgr',variant='pgr_reward_support',domain=args.domain,seed=args.seed,output_dir=str(ctx.out),device='cuda',torch_threads=1,total_steps=args.steps,episodes=args.steps//H,
            gamma=1.,smoke=False,save_training_state=False,eval_every_episodes=args.eval_every//H,eval_episodes=args.dev_cases,deterministic_eval=True,path_master=args.train_master,development_master=args.dev_master,engineering_gate=args.engineering_gate,
            pgr=dict(diffusion_train_steps=100000,num_samples=1000000,denoise_steps=128,utd_ratio=20,generation_batch_size=10000))
        if args.engineering_gate:
            config['learning_starts']=H
            config['pgr'].update(diffusion_train_steps=100,num_samples=10000,retrain_diffusion_every=2*H)
        def evaluate(agent,_Env,_config,cfg,steps):
            def action(obs):
                with torch.no_grad():return agent.policy_net(torch.as_tensor(obs,dtype=torch.float32,device=agent.device).unsqueeze(0),deterministic=True,return_log_prob=False)[0][0].cpu().numpy()
            return ctx.evaluate(action,steps,action_mode='tanh_latent',cases=args.dev_cases,policy=agent.policy_net)
        runner._evaluate=evaluate
        runner.formal_progress=lambda row:(setattr(ctx,'training_steps',int(row['train_env_steps'])),ctx.progress('training',{'actor_updates':row.get('actor_updates'),'critic_minibatches':row.get('critic_minibatches'),'synthetic_sample_references':row.get('synthetic_sample_references')}))
        install_pgr(runner,args.domain,ctx.out,engineering=args.engineering_gate)
        # Native model dimensions/optimizers are unchanged; logging-only wrapper
        # gives the scheduler live progress during long diffusion phases.
        original_optimizations=runner.install_safe_trainer_optimizations
        def trainer_progress(trainer):
            original_optimizations(trainer)
            original=trainer.train_on_batch
            def call(*a,**kw):
                value=original(*a,**kw)
                if trainer.step%1000==0:ctx.progress('diffusion_training',{'diffusion_gradient_step':int(trainer.step),'diffusion_gradient_target':int(trainer.train_num_steps)if hasattr(trainer,'train_num_steps')else config['pgr']['diffusion_train_steps']})
                return value
            trainer.train_on_batch=call
        runner.install_safe_trainer_optimizations=trainer_progress
        before=bindings()
        if args.evaluate_model:
            saved=torch.load(args.evaluate_model,map_location='cuda:0',weights_only=False)
            if saved['config']['domain']!=args.domain or int(saved['config']['seed'])!=args.seed:raise ValueError('model identity mismatch')
            cfg=saved['native_config'];native=runner._load_author();env=Env(args.domain,args.seed,phase=args.eval_phase,action_mode='tanh_latent',holdout=args.eval_phase=='final')
            agent=runner._agent(native,cfg,env,torch.device('cuda:0'));runner._restore_agent(agent,saved['agent']);env.close()
            def action(obs):
                with torch.no_grad():return agent.policy_net(torch.as_tensor(obs,dtype=torch.float32,device=agent.device).unsqueeze(0),deterministic=True,return_log_prob=False)[0][0].cpu().numpy()
            result=ctx.evaluate(action,int(saved['counters']['real_steps']),action_mode='tanh_latent',phase=args.eval_phase,cases=args.eval_cases,policy=agent.policy_net)
            assert bindings()==before
            return ctx.completion(result,args.evaluate_model,before,evaluation_only=True)
        if args.steps%H or args.steps<2*H or args.eval_every%H:raise ValueError('complete native episode budget required')
        dump(ctx.out/'protocol.json',dict(schema='chapter4-formal-pgr-v1',arguments=vars(args),source_binding=before,config=config,native_update_generation_equations_unchanged=True,task_adaptation='known terminal-only reward support for generated replay; real rewards unchanged',full_native_diffusion_budget=not args.engineering_gate,deterministic_development=True,one_fixed_endpoint=True,optimizer_or_replay_saved=False))
        result=runner.run(config);ctx.training_steps=int(result['real_steps'])
        result['formal_training_performed']=not args.engineering_gate
        result['engineering_only']=args.engineering_gate
        dump(ctx.out/'receipt.json',result)
        points=__import__('json').loads((ctx.out/'evaluations.json').read_text())
        grid=list(range(0,args.steps+1,args.eval_every))
        if grid[-1]!=args.steps:grid.append(args.steps)
        assert [x['train_env_steps']for x in points]==grid
        assert all(len(x['returns'])==args.dev_cases and x['deterministic_eval']for x in points)
        assert result['nonterminal_nonzero_real_rewards']==0 and result['terminal_td_target_abs_error']==0
        assert result['checkpoint_kind']=='inference_weights' and ctx.training_steps==args.steps
        assert ctx.resource_counts.get('train')==args.steps//H
        if not args.engineering_gate:
            cycles=args.steps//10000
            assert result['diffusion_training_steps']==cycles*100000 and result['total_generated_model_transitions']==cycles*1000000
            assert result['synthetic_sample_references']>0 and result['mixed_minibatches']>0
        assert bindings()==before
        return ctx.completion(result,ctx.out/'checkpoint.pt',before)
    except BaseException:
        dump(ctx.out/'failed.json',dict(status='failed',traceback=traceback.format_exc(),source_binding=before,training_steps=ctx.training_steps,wall_seconds=time.perf_counter()-ctx.start,exact_resume_available=False));raise
    finally:ctx.close()

if __name__=='__main__':main(parser('pgr').parse_args())
