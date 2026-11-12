"""Load chapter-owned source views without copying scientific modules.

Only explicit activate()/execute_file() calls install a source view. This module
uses the standard library and performs no environment/model construction.
"""
from __future__ import annotations
import ast, hashlib, importlib.abc, importlib.util, json, os, sys
from pathlib import Path

CHAPTERS=Path(__file__).resolve().parents[2]
if str(CHAPTERS) not in sys.path:sys.path.insert(0,str(CHAPTERS))
_active_name=None
_active_files={}

def active_view_name():return _active_name

def resource_path(logical):
    """Resolve a shared non-Python resource in the active source view."""
    if not _active_name:
        raise RuntimeError('Activate a chapter source view before reading its resources')
    row = _active_files.get(logical)
    if row is None:
        raise FileNotFoundError('Resource is outside the active source view: ' + logical)
    return CHAPTERS / row['path']

def view_files(name):
    chapter=name.split('.')[0]
    data=json.loads((CHAPTERS/chapter/'source/views.json').read_text())
    return data['views'][name]['files']

def source_hashes():
    return {name:hashlib.sha256((CHAPTERS/row['path']).read_bytes()).hexdigest()
            for name,row in sorted(_active_files.items())}

def source_paths():
    return iter((name,CHAPTERS/row['path']) for name,row in sorted(_active_files.items()))

def verify(name):
    files=view_files(name)
    for logical,row in files.items():
        path=(CHAPTERS/row['path']).resolve(strict=True)
        if not path.is_relative_to(CHAPTERS):raise ValueError('Source leaves checkout: '+logical)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Source changed: '+logical)
    return dict(view=name,files=len(files),model_loads=0,environment_steps=0)

class _Relocate(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str) and (node.value=='/hy-tmp' or node.value.startswith(('/hy-tmp/','/root/gamewar/resource_allocation_rl','/root/resource_allocation_run_storage_20261101/'))):
            return ast.copy_location(ast.Call(func=ast.Name(id='str',ctx=ast.Load()),args=[ast.Call(func=ast.Name(id='_chapter_resolve',ctx=ast.Load()),args=[node],keywords=[])],keywords=[]),node)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='Path' and len(node.args)==1 and not node.keywords:
            node.func=ast.Name(id='_chapter_resolve',ctx=ast.Load())
        return node

def _compiled(path):
    tree=_Relocate().visit(ast.parse(Path(path).read_bytes(),filename=str(path)))
    ast.fix_missing_locations(tree)
    return compile(tree,str(path),'exec')

def relocate_tree(tree):
    """Apply the same path-only adaptation to an explicitly derived worker AST."""
    return _Relocate().visit(tree)

class _Loader(importlib.abc.Loader):
    def __init__(self,path,logical,is_package):self.path,self.logical,self.package=path,logical,is_package
    def create_module(self,spec):return None
    def exec_module(self,module):
        from portable_paths import resolve_path,project_root
        module.__dict__['_chapter_resolve']=resolve_path
        module.__file__=str(self.path)
        if self.package:module.__path__=[str(self.path.parent)]
        exec(_compiled(self.path),module.__dict__)
        # Record this delivered logical view rather than the superseded full
        # snapshot (which contained unused vendors and duplicate data indexes).
        if self.logical=='allocation23/io.py':
            module.source_hashes=source_hashes
            module.PROJECT_ROOT=project_root()
        elif self.logical=='allocation23/paths.py':
            module.PROJECT_ROOT=project_root()
            module.iter_source_files=source_paths

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='allocation23' and not fullname.startswith('allocation23.'):return None
        stem=fullname.replace('.','/')
        logical=stem+'/__init__.py' if stem+'/__init__.py' in _active_files else stem+'.py'
        row=_active_files.get(logical)
        if row is None:raise ModuleNotFoundError('Module is outside the active source view: '+fullname)
        actual=CHAPTERS/row['path'];package=logical.endswith('/__init__.py')
        return importlib.util.spec_from_file_location(fullname,actual,loader=_Loader(actual,logical,package),submodule_search_locations=[str(actual.parent)] if package else None)

def activate(name):
    global _active_name,_active_files
    if _active_name and _active_name!=name:raise RuntimeError('Use a fresh process for a different scientific source view')
    if not _active_name and 'allocation23' in sys.modules:raise RuntimeError('Activate a source view before importing allocation23')
    verify(name)
    _active_name=name;_active_files=view_files(name)
    if not any(isinstance(f,_Finder) for f in sys.meta_path):sys.meta_path.insert(0,_Finder())
    from portable_paths import eurosat_root,project_root,final_data_root
    os.environ.setdefault('ALLOCATION23_DATA_ROOT',str(eurosat_root()))
    os.environ['RESOURCE_ALLOCATION_PROJECT_ROOT']=str(project_root())
    os.environ.setdefault('RESOURCE_ALLOCATION_DATA_ROOT',str(final_data_root()))
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True

def execute_file(path,argv=()):
    """Execute only a requested worker; standard-library setup stays lazy."""
    from portable_paths import resolve_path
    path=Path(path).resolve(strict=True)
    before_argv,before_path=list(sys.argv),list(sys.path)
    sys.argv=[str(path),*map(str,argv)];sys.path.insert(0,str(path.parent))
    try:
        namespace=dict(__name__='__main__',__file__=str(path),__package__=None,_chapter_resolve=resolve_path)
        exec(_compiled(path),namespace)
    finally:sys.argv=before_argv;sys.path[:]=before_path

def load_module(name,path):
    path=Path(path).resolve(strict=True)
    package=path.name=='__init__.py'
    loader=_Loader(path,'',package)
    spec=importlib.util.spec_from_file_location(name,path,loader=loader,submodule_search_locations=[str(path.parent)] if package else None)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module
