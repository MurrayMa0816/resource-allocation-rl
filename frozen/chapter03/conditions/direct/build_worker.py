"""Build the independent Direct worker from the frozen formal native worker."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
base=Path('/hy-tmp/resource_allocation_results/chapter3/formal_100k_5seed_20261105/preparation/a_native_formal_worker.py')
s=base.read_text()
s=s.replace("os.environ['PYTHONDONTWRITEBYTECODE']='1'","os.environ['PYTHONDONTWRITEBYTECODE']='1'\nsys.dont_write_bytecode=True")
s=s.replace("SCHEMA='chapter3-A-native-formal-lean-v3'", "SCHEMA='chapter3-A-direct-native-supplement-v1'")
start=s.index('ARMS={');end=s.index('\nSCHEMA=',start)
s=s[:start]+"ARMS={'A_DIRECT_TOTAL':dict(variant='A_FULL',temporal='adaptive',quota_reference='pace')}\n"+s[end:]
s=s.replace("p.add_argument('--arm',choices=ARMS,required=True)","p.add_argument('--arm',choices=ARMS,default='A_DIRECT_TOTAL')\n    p.add_argument('--device',default='cuda:0')")
s=s.replace("from allocation23.learning import PPO,update", "from allocation23.learning import PPO,update as native_update\n    from allocation23.methods import make_value\n    from direct_policy import DirectTotalActor\n    from native_cuda import make_device_update\n    update,device_adapter=make_device_update(native_update)")
s=s.replace("    mapping=ARMS[args.arm]", "    mapping=ARMS[args.arm]\n    device=torch.device(args.device)\n    if device.type not in ('cpu','cuda'):\n        raise ValueError('Only native CPU or CUDA runtime supported')\n    if device.type=='cuda' and not torch.cuda.is_available():\n        raise RuntimeError('CUDA was requested but is unavailable; no silent CPU fallback')\n    output.mkdir(parents=True)\n    for name in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH',\n                 'TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):\n        cache=output/'runtime'/name.lower();cache.mkdir(parents=True)\n        os.environ[name]=str(cache)")
s=s.replace("    state=fresh(task,cfg)","    random.seed(830000+args.seed)\n    torch.manual_seed(840000+args.seed)\n    actor=DirectTotalActor(task.spec,args.seed).to(device=device,dtype=torch.float64)\n    critic=make_value(task.spec,args.seed).to(device=device,dtype=torch.float64)\n    state=(actor,critic,torch.optim.Adam(actor.parameters(),lr=PPO['actor_lr']),\n        torch.optim.Adam(critic.parameters(),lr=PPO['value_lr']),\n        torch.Generator(device=device).manual_seed(810000+args.seed),np.random.default_rng(820000+args.seed))")
old="    output.mkdir(parents=True)\n    for name in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','CUDA_CACHE_PATH',\n                 'TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):\n        cache=output/'runtime'/name.lower();cache.mkdir(parents=True)\n        os.environ[name]=str(cache)\n"
# Keep the earlier setup before any model/CUDA initialization; remove its second copy.
ix=s.index(old);ix2=s.index(old,ix+len(old));s=s[:ix2]+s[ix2+len(old):]
s=s.replace("actual_device='cpu'", "actual_device=str(device)")
s=s.replace("device_choice=dict(selected='cpu',reason='native float64 compact A implementation')", "device_choice=dict(selected=str(device),reason='explicit CUDA native float64 PPO; same equations with tensor-device keywords only',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None)")
s=s.replace("source_identity=dict(package=sources,worker_sha256=worker_sha)","source_identity=dict(package=sources,worker_sha256=worker_sha,direct_capsule={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})")
s=s.replace("arm_mapping=mapping,ppo=dict(PPO),initial_algorithm=actor.metadata(),", "arm_mapping=mapping,ppo=dict(PPO),initial_algorithm=actor.metadata(),\n        direct_mapping=actor.direct_initialization,device_adapter=device_adapter,\n        comparison_scope='same A spatial backbone, state, native PPO and feasibility; Direct replaces only state-dependent pacing shift by fixed initialized total coordinate',")
s=s.replace("        before=(state_hash(actor)","        cuda_before=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []\n        before=(state_hash(actor)")
s=s.replace("            if save_examples and i==0:allocations.append(result['actions'].copy())", "            if (save_examples or learning_steps==0) and i==0:allocations.append(result['actions'].copy())")
s=s.replace("        if before[0]!=state_hash(actor)", "        cuda_after=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []\n        if len(cuda_before)!=len(cuda_after) or any(not torch.equal(a,b) for a,b in zip(cuda_before,cuda_after)):\n            raise AssertionError('Independent evaluation changed global CUDA RNG')\n        if before[0]!=state_hash(actor)")
s=s.replace("if allocations:np.savez_compressed(output/'allocation_examples.npz',allocations=np.asarray(allocations))", "if allocations:np.savez_compressed(output/('initial_allocation_examples.npz' if learning_steps==0 else 'allocation_examples.npz'),allocations=np.asarray(allocations))")
s=s.replace("if source_hashes()!=sources or digest(Path(__file__))!=worker_sha:", "if source_hashes()!=sources or digest(Path(__file__))!=worker_sha or {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}!=protocol['source_identity']['direct_capsule']:")
s=s.replace("if __name__=='__main__':train(parser().parse_args())", """def main():
    if len(sys.argv)>1 and sys.argv[1]=='evaluate':
        from evaluate_direct import main as evaluate
        evaluate(sys.argv[2:]);return
    if len(sys.argv)>1 and sys.argv[1]=='train':del sys.argv[1]
    train(parser().parse_args())

if __name__=='__main__':main()""")
s=s.replace('Native Chapter 3 A/PPO and four registered total-decision ablations.', 'Direct learned total coordinate using the frozen formal Chapter 3 native A/PPO.')
(ROOT/'direct_worker.py').write_text(s)
(ROOT/'worker_derivation.json').write_text(json.dumps(dict(frozen_worker=str(base),frozen_worker_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),derived_worker_sha256=hashlib.sha256(s.encode()).hexdigest(),native_learning_equations='loaded from unchanged frozen source; AST checked device keywords only',method_delta='direct total decoder and one static initial output bias',storage='one terminal evaluation model; compressed scalar trajectories; no optimizer/replay/periodic models'),indent=2)+'\n')
