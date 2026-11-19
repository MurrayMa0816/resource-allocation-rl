from . import source_layout
"""Read accepted local source tickets; no Git process or scientific imports."""
from pathlib import Path
import hashlib, json
from portable_paths import project_root

def verify_props_source(vendor_root, expected_commit):
    ticket_path = project_root() / 'chapters/chapter04/configs/provenance/PROPS_original_source_receipt.json'
    ticket = json.loads(ticket_path.read_text())
    source = Path(vendor_root) / 'PROPS/ppo_props_continuous.py'
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if ticket['expected_sha'] != expected_commit or ticket['actual_sha'] != expected_commit or ticket['source_modified'] or actual != ticket['sha256']:
        raise RuntimeError('PROPS source/commit differs from the accepted immutable source ticket')
    return dict(ticket, source_file=str(source), sha256=actual, verification='local accepted commit/SHA ticket; no Git checkout required')


def resolve_scientific_path(value):
    local = source_layout.resolve_source_path(value)
    if local is not None: return local
    """Resolve exact source roots and current model/data descendants separately."""
    from portable_paths import resolve_path,chapter_data_root
    path=Path(value);frozen=source_layout.virtual_root()
    aliases=json.loads((project_root()/'chapters/chapter04/configs/provenance/science_data_path_bindings.json').read_text())['mapping']
    if str(value)in aliases:
        from portable_paths import final_data_root
        return final_data_root()/aliases[str(value)]
    if path.is_absolute() and path.is_relative_to(frozen):
        pieces=path.relative_to(frozen).parts
        if len(pieces)>1 and pieces[1] in ('jobs','evaluation_jobs','testing'):
            names={'troc_props':'double_random_formal_20261107','offpolicy':'formal_100k_5seed_20261106','mad_td':'mad_td_formal_20261107'}
            scope=names.get(pieces[0],pieces[0])
            kind='formal' if 'formal' in scope else 'supplementary'
            return resolve_path(chapter_data_root(4)/kind/scope/Path(*pieces[1:]))
    try:return resolve_path(value)
    except FileNotFoundError:
        text=str(value);old='/hy-tmp/resource_allocation_results/chapter4/'
        if text.startswith(old):
            pieces=Path(text[len(old):]).parts
            if pieces and (frozen/pieces[0]).is_dir():
                kind='formal' if 'formal' in pieces[0] else 'supplementary'
                return resolve_path(chapter_data_root(4)/kind/Path(*pieces))
        raise


from contextlib import contextmanager
@contextmanager
def _original_local_git_source_receipts():
    """Support immutable legacy PROPS receipt calls without a .git checkout.

    Only the three original, read-only commit/source verification requests are
    handled. Every other subprocess request retains its ordinary implementation.
    """
    source_layout.configure_imports()
    import subprocess, ast, importlib.abc, importlib.machinery, importlib.util, sys
    from frozen_adapter import _Paths, install_import_adapter
    install_import_adapter()
    from portable_paths import resolve_path
    frozen_root=source_layout.virtual_root()
    local_path=resolve_scientific_path
    class SourceLoader(importlib.abc.Loader):
        def __init__(self,path,package=False):self.path,self.package=path,package
        def create_module(self,spec):return None
        def exec_module(self,module):
            tree=ast.parse(self.path.read_bytes(),filename=str(self.path));count=0
            if self.path.name=='formal_support.py':
                for node in tree.body:
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ACCEL' for t in node.targets):
                        expected="/root/resource_allocation_run_storage_20261101/ch4_acceleration_v3_20261103"
                        if not (isinstance(node.value,ast.Call) and len(node.value.args)==1 and isinstance(node.value.args[0],ast.Constant) and node.value.args[0].value==expected):
                            raise ValueError('Unexpected retired PGR-only ACCEL declaration')
                        node.value=ast.copy_location(ast.Constant(value=None),node.value);count+=1
                if count!=1:raise ValueError('Expected exactly one unused retired PGR capsule declaration')
            class ScopeJoins(ast.NodeTransformer):
                def visit_BinOp(self,node):
                    left=node
                    while isinstance(left,ast.BinOp) and isinstance(left.op,ast.Div):left=left.left
                    if isinstance(node.op,ast.Div) and isinstance(left,ast.Name) and left.id in ('ROOT','SCOPE'):
                        return ast.copy_location(ast.Call(func=ast.Name(id='_portable_resolve',ctx=ast.Load()),args=[node],keywords=[]),node)
                    return self.generic_visit(node)
            tree=ScopeJoins().visit(tree);tree=source_layout.rebase_maintenance_bootstrap(tree);tree=source_layout.rebase_data_joins(tree);tree=_Paths().visit(tree);ast.fix_missing_locations(tree)
            module.__file__=str(self.path);module.__dict__['_portable_resolve']=local_path
            if self.package:module.__path__=[str(self.path.parent)]
            exec(compile(tree,str(self.path),'exec'),module.__dict__)
    original_spec=importlib.util.spec_from_file_location
    def local_spec(name,location=None,*,loader=None,submodule_search_locations=None):
        if location is not None:
            source=Path(location).resolve()
            if source.suffix=='.py' and source_layout.contains(source):
                loader=SourceLoader(source,submodule_search_locations is not None or source.name=='__init__.py')
        return original_spec(name,location,loader=loader,submodule_search_locations=submodule_search_locations)
    importlib.util.spec_from_file_location=local_spec
    original = subprocess.check_output
    def checked(command, *args, **kwargs):
        words = list(command) if isinstance(command, (list, tuple)) else []
        if words and Path(str(words[0])).name == 'git':
            words = words[1:]
            cwd = kwargs.get('cwd')
            if words[:1] == ['-C']:
                cwd, words = words[1], words[2:]
            if cwd is not None and (Path(cwd)/'PROPS/ppo_props_continuous.py').is_file():
                ticket = verify_props_source(cwd, 'a0c8708529047f85dcf98f90d5ea264dd43aeba2')
                if words == ['rev-parse', 'HEAD']: result = (ticket['actual_sha']+'\n').encode()
                elif words == ['diff', '--name-only', 'HEAD']: result = b''
                elif words == ['show', ticket['actual_sha']+':PROPS/ppo_props_continuous.py']:
                    result = (Path(cwd)/'PROPS/ppo_props_continuous.py').read_bytes()
                else: return original(command, *args, **kwargs)
                return result.decode(kwargs.get('encoding') or 'utf-8') if kwargs.get('text') or kwargs.get('universal_newlines') or kwargs.get('encoding') else result
        return original(command, *args, **kwargs)
    subprocess.check_output = checked
    try: yield
    finally:
        subprocess.check_output = original
        importlib.util.spec_from_file_location=original_spec


def load_current_source(name,path):
    """Within local_git_source_receipts, import definitions through the scoped I/O loader."""
    import importlib.util,sys
    path=Path(path).resolve(strict=True)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module

def run_current_source(path,arguments=()):
    """Explicit current execution; only filesystem identities are adapted in memory."""
    import sys
    path=source_layout.resolve_source_path(path) or Path(path)
    path=Path(path).resolve(strict=True);old_argv,old_path=sys.argv[:],sys.path[:]
    old_main=sys.modules.get('__main__')
    sys.argv=[str(path),*map(str,arguments)];sys.path.insert(0,str(path.parent))
    try:load_current_source('__main__',path)
    finally:
        sys.argv,sys.path[:]=old_argv,old_path
        if old_main is not None:sys.modules['__main__']=old_main

@contextmanager
def local_git_source_receipts():
    with source_layout.source_capsule():
        with _original_local_git_source_receipts():
            yield
