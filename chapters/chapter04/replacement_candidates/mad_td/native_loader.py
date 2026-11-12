"""Load immutable official MAD-TD with audited API and terminal-task adapters.

No network, loss architecture, optimizer or model-generated-data mechanism is
reimplemented. Vendor bytes stay unchanged; task changes occur in memory and
are described by PATCH_PROFILE.
"""
from __future__ import annotations
import ast,copy,hashlib,importlib,importlib.util,json,sys,types
from pathlib import Path

HERE=Path(__file__).resolve().parent
VENDOR=HERE/'vendor'
PATCH_PROFILE='madtd-finite-terminal-supervision-v1'

def verify_vendor():
    m=json.loads((HERE/'vendor_manifest.json').read_text())
    for name,h in m['files_sha256'].items():
        if hashlib.sha256((VENDOR/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Changed MAD-TD vendor: '+name)
    return m

def _patched_module(name,path,edit,initial=None):
    tree=ast.parse(path.read_text(),filename=str(path));tree=edit.visit(tree)
    module=types.ModuleType(name);module.__file__=str(path);module.__package__=name.rsplit('.',1)[0]
    module.__dict__.update(initial or {});sys.modules[name]=module
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    module._adapter_patch_counts=dict(edit.counts)
    return module

class ReplayAnnotations(ast.NodeTransformer):
    def __init__(self):self.counts={'unused_env_annotation_import_removed':0}
    def visit_ImportFrom(self,node):
        if node.module=='mad_td.data.env':self.counts['unused_env_annotation_import_removed']+=1;return None
        return node

class FiniteCritic(ast.NodeTransformer):
    def __init__(self):self.counts={'terminal_bootstrap_target':0,'terminal_critic_supervision':0}
    def visit_Assign(self,node):
        self.generic_visit(node)
        if any(isinstance(x,ast.Name)and x.id=='target'for x in node.targets)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='target_func':
            self.counts['terminal_bootstrap_target']+=1
            return [node,ast.parse('target = reward + mask * (target - reward)').body[0]]
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name)and node.func.id=='loss_func':
            assert isinstance(node.args[4],ast.Name)and node.args[4].id=='mask'
            node.args[4]=ast.parse('jnp.ones_like(mask)',mode='eval').body;self.counts['terminal_critic_supervision']+=1
        return node

class FiniteModel(ast.NodeTransformer):
    def __init__(self):self.counts={'terminal_model_reward_supervision':0,'terminal_encoder_bootstrap_target':0,'terminal_encoder_critic_supervision':0}
    def visit_Assign(self,node):
        self.generic_visit(node)
        if any(isinstance(x,ast.Name)and x.id=='reward_loss'for x in node.targets)and ast.unparse(node.value)=='jnp.mean(mask * (reward - model_rewards) ** 2)':
            node.value=ast.parse('jnp.mean(jnp.ones_like(mask) * (reward - model_rewards) ** 2)',mode='eval').body;self.counts['terminal_model_reward_supervision']+=1
        if any(isinstance(x,ast.Name)and x.id=='target'for x in node.targets)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Attribute)and node.value.func.attr=='stop_gradient':
            self.counts['terminal_encoder_bootstrap_target']+=1
            return [node,ast.parse('target = reward + mask[:, 0] * (target - reward)').body[0]]
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name)and node.func.id=='loss_fn':
            assert ast.unparse(node.args[4])=='mask[:, 0]'
            node.args[4]=ast.parse('jnp.ones_like(mask[:, 0])',mode='eval').body;self.counts['terminal_encoder_critic_supervision']+=1
        return node

def load_native():
    m=verify_vendor()
    import jax,jax.numpy as jnp
    # Removed names are direct aliases only; no numerical operation changes.
    if not hasattr(jax,'tree_map'):jax.tree_map=jax.tree_util.tree_map
    if not hasattr(jnp,'float_'):jnp.float_=jnp.float64
    if str(VENDOR)not in sys.path:sys.path.insert(0,str(VENDOR))
    package=importlib.import_module('mad_td')
    if Path(package.__file__).resolve()!=VENDOR/'mad_td/__init__.py':raise RuntimeError('Unexpected MAD-TD namespace')
    replay_edit=ReplayAnnotations();replay=_patched_module('mad_td.data.replay_buffer',VENDOR/'mad_td/data/replay_buffer.py',replay_edit,{'Env':object})
    if replay_edit.counts!={'unused_env_annotation_import_removed':1}:raise AssertionError('Replay source shape changed')
    critic_edit=FiniteCritic();critic=_patched_module('mad_td.update_functions.critic_updates',VENDOR/'mad_td/update_functions/critic_updates.py',critic_edit)
    model_edit=FiniteModel();model=_patched_module('mad_td.model_util.model_loss_functions',VENDOR/'mad_td/model_util/model_loss_functions.py',model_edit)
    if critic_edit.counts!={'terminal_bootstrap_target':1,'terminal_critic_supervision':1}:raise AssertionError(critic_edit.counts)
    if model_edit.counts!={'terminal_model_reward_supervision':1,'terminal_encoder_bootstrap_target':1,'terminal_encoder_critic_supervision':1}:raise AssertionError(model_edit.counts)
    factory=importlib.import_module('mad_td.trainers.mad_td_model')
    updates=importlib.import_module('mad_td.update_functions.mad_td_update')
    # Extract just the author's action function, without importing DMC/Brax/loggers.
    path=VENDOR/'mad_td/trainers/trainers.py';tree=ast.parse(path.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='get_policy_action')
    from flax.training.train_state import TrainState
    hp=importlib.import_module('mad_td.cfgs.mad_td_config')
    ns={'jax':jax,'jnp':jnp,'TrainState':TrainState,'MadTdHyperparams':hp.MadTdHyperparams}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])),str(path),'exec'),ns)
    return types.SimpleNamespace(manifest=m,replay=replay,factory=factory,updates=updates,hp=hp,policy_action=ns['get_policy_action'],patches={**replay_edit.counts,**critic_edit.counts,**model_edit.counts},critic=critic,model_loss=model)
