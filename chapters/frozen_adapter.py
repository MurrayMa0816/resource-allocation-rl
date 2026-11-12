"""Explicit runtime path adaptation for immutable final scientific source files.

Original files remain byte-identical. Only Path constructors, absolute storage
strings and import search locations are resolved in memory. No module is loaded
and no scientific operation starts until the caller explicitly invokes run_file
or load_module. Use a fresh process for each chapter's isolated native package.
"""
from __future__ import annotations
import ast
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types
from portable_paths import final_data_root, project_root, resolve_path

class _Paths(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str) and (node.value=='/hy-tmp' or node.value.startswith(('/hy-tmp/','/root/gamewar/resource_allocation_rl','/root/resource_allocation_run_storage_20261101/'))):
            return ast.copy_location(ast.Call(func=ast.Name(id='str',ctx=ast.Load()),args=[ast.Call(func=ast.Name(id='_portable_resolve',ctx=ast.Load()),args=[node],keywords=[])],keywords=[]),node)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='Path' and len(node.args)==1 and not node.keywords:
            node.func=ast.Name(id='_portable_resolve',ctx=ast.Load())
        return node

def compiled_source(path):
    """Compile adapted code in memory; does not execute it or write bytecode."""
    path=Path(path).resolve(strict=True)
    if not any(path.is_relative_to(root) for root in (final_data_root(),project_root())):
        raise ValueError('Frozen source must be inside the explicit final code/data roots')
    tree=_Paths().visit(ast.parse(path.read_bytes(),filename=str(path)))
    ast.fix_missing_locations(tree)
    return compile(tree,str(path),'exec')

class _Loader(importlib.abc.Loader):
    def __init__(self,path,is_package):self.path=path;self.is_package=is_package
    def create_module(self,spec):return None
    def exec_module(self,module):
        module.__file__=str(self.path)
        module.__dict__['_portable_resolve']=resolve_path
        if self.is_package:module.__path__=[str(self.path.parent)]
        exec(compiled_source(self.path),module.__dict__)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or not spec.origin or not spec.origin.endswith('.py'):return None
        origin=Path(spec.origin).resolve()
        roots=[p/'source' for p in (project_root()/'chapters').glob('chapter*')]
        roots += [project_root()/'chapters/chapter06/experiment/source']
        roots.append(final_data_root()/'chapters')
        if not any(origin.is_relative_to(root) for root in roots):return None
        package=spec.submodule_search_locations is not None
        loader=_Loader(origin,package)
        return importlib.util.spec_from_file_location(fullname,origin,loader=loader,
            submodule_search_locations=[str(origin.parent)] if package else None)

def install_import_adapter():
    if not any(isinstance(finder,_Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0,_Finder())

def load_module(name,path):
    """Explicitly import one source, adapting its data-root-local descendants."""
    path=resolve_path(path).resolve(strict=True)
    install_import_adapter()
    package=path.name=='__init__.py';loader=_Loader(path,package)
    spec=importlib.util.spec_from_file_location(name,path,loader=loader,
        submodule_search_locations=[str(path.parent)] if package else None)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module

def run_file(path,argv=()):
    """Run only an explicitly requested immutable entry in a fresh process."""
    path=resolve_path(path).resolve(strict=True)
    install_import_adapter()
    old_argv=list(sys.argv);sys.argv=[str(path),*map(str,argv)]
    old_path=list(sys.path);sys.path.insert(0,str(path.parent))
    try:
        namespace={'__name__':'__main__','__file__':str(path),'__package__':None,
                   '_portable_resolve':resolve_path}
        exec(compiled_source(path),namespace)
    finally:
        sys.argv=old_argv;sys.path[:]=old_path
