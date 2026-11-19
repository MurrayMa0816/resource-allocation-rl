"""Physical chapter sources with unchanged logical model-source identities."""
from pathlib import Path
from contextlib import contextmanager
import ast, hashlib, importlib.abc, importlib.util, json, shutil, sys, tempfile

CHAPTER=Path(__file__).resolve().parents[1]
_MANIFEST=None
_ACTIVE_SOURCE=None
def manifest():
 global _MANIFEST
 if _MANIFEST is None:_MANIFEST=json.loads((CHAPTER/'configs/source_paths.json').read_text())
 return _MANIFEST
def virtual_root():return _ACTIVE_SOURCE if _ACTIVE_SOURCE is not None else CHAPTER/'algorithms'
def source_path(relative):
 record=manifest()['files'].get(str(relative).replace('\\','/'))
 if record is None:raise FileNotFoundError('Unknown logical scientific source: '+str(relative))
 if _ACTIVE_SOURCE is not None:return _ACTIVE_SOURCE/str(relative)
 return CHAPTER/record['path']
def resolve_source_path(value):
 text=str(value)
 for old,target in manifest().get('data',{}).items():
  if text==old:return CHAPTER.parents[1]/target
  if text.startswith(old+'/'):return CHAPTER.parents[1]/target/text[len(old)+1:]
 if _ACTIVE_SOURCE is not None:
  for scope in ('troc_props','offpolicy'):
   old=str(_ACTIVE_SOURCE/scope/'source/environment/release/allocation23/data/eurosat_rgb')
   if text==old:return CHAPTER.parent/'chapter02/data/eurosat_rgb'
   if text.startswith(old+'/'):return CHAPTER.parent/'chapter02/data/eurosat_rgb'/text[len(old)+1:]
  for old,logical in sorted(manifest().get('logical_aliases',{}).items(),key=lambda item:len(item[0]),reverse=True):
   if text==old:return _ACTIVE_SOURCE/logical
   if text.startswith(old+'/'):return _ACTIVE_SOURCE/logical/text[len(old)+1:]
  for logical,record in manifest()['files'].items():
   if text==str(CHAPTER/record['path']):return _ACTIVE_SOURCE/logical
  for physical,logical in manifest().get('physical_directory_bindings',{}).items():
   old=str(CHAPTER/physical)
   if text==old:return _ACTIVE_SOURCE/logical
   if text.startswith(old+'/'):return _ACTIVE_SOURCE/logical/text[len(old)+1:]
 prefixes=[str(virtual_root())]+manifest()['original_source_roots']
 for old in sorted(prefixes,key=len,reverse=True):
  if text==old:return virtual_root()
  if text.startswith(old+'/'):
   relative=text[len(old)+1:]
   if relative in manifest()['files']:return source_path(relative)
   # Imported source packages need a concrete directory for ordinary namespace lookup.
   for logical,record in manifest()['files'].items():
    if logical.startswith(relative+'/'):
     physical=CHAPTER/record['path']
     return physical.parents[len(Path(logical).relative_to(relative).parts)-1]
 return None
def contains(path):
 path=Path(path).resolve()
 if _ACTIVE_SOURCE is not None and path.is_relative_to(_ACTIVE_SOURCE):return True
 return any(path==(CHAPTER/record['path']).resolve() for record in manifest()['files'].values())
def verify():
 for relative,record in manifest()['files'].items():
  actual=hashlib.sha256(source_path(relative).read_bytes()).hexdigest()
  if actual!=record['sha256']:raise ValueError('Bound scientific source changed: '+relative)
 return len(manifest()['files'])
def rebase_data_joins(tree):
 class DataJoins(ast.NodeTransformer):
  def visit_BinOp(self,node):
   node=self.generic_visit(node)
   if isinstance(node.op,ast.Div) and isinstance(node.right,ast.Constant) and node.right.value=='allocation23/data/eurosat_rgb':
    return ast.copy_location(ast.Call(func=ast.Name(id='_portable_resolve',ctx=ast.Load()),args=[node],keywords=[]),node)
   return node
 return DataJoins().visit(tree)
def rebase_maintenance_bootstrap(tree):
 class Bootstrap(ast.NodeTransformer):
  def visit_Assign(self,node):
   if any(isinstance(t,ast.Name) and t.id=='_PORTABLE_CODE_ROOT' for t in node.targets):
    node.value=ast.copy_location(ast.parse("_project_root()/'chapters/chapter02/scripts/support'",mode='eval').body,node.value)
   return self.generic_visit(node)
  def visit_Constant(self,node):
   if isinstance(node.value,str):
    pairs={'chapters/chapter04/source/foundation':'chapters/chapter04/algorithms/foundation','chapters/chapter04/source/troc_props/source/environment':'chapters/chapter04/environment/troc_props','chapters/chapter04/source/offpolicy/source/environment':'chapters/chapter04/environment/offpolicy'}
    for old,new in pairs.items():
     if node.value.startswith(old):return ast.copy_location(ast.Constant(value=new+node.value[len(old):]),node)
   return node
 return Bootstrap().visit(tree)
def source_location_allowed(path):
 from portable_paths import project_root,final_data_root
 path=Path(path).resolve()
 return contains(path) or path.is_relative_to(project_root()) or path.is_relative_to(final_data_root())
def rebase_source_guards(tree):
 class SourceGuard(ast.NodeTransformer):
  def visit_Call(self,node):
   node=self.generic_visit(node)
   if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
       and node.func.value.id in ('source','CHAPTER') and node.func.attr=='is_relative_to'
       and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value=='/hy-tmp'):
    return ast.copy_location(ast.Call(func=ast.Name(id='_selected_source_location_allowed',ctx=ast.Load()),args=[node.func.value],keywords=[]),node)
   return node
 return SourceGuard().visit(tree)
@contextmanager
def source_capsule():
 """Reconstruct only the selected logical tree as temporary ordinary files."""
 global _ACTIVE_SOURCE
 if _ACTIVE_SOURCE is not None:
  yield _ACTIVE_SOURCE;return
 verify()
 with tempfile.TemporaryDirectory(prefix=CHAPTER.name+'-selected-source-') as temporary:
  root=Path(temporary)/'source';root.mkdir()
  for logical,record in manifest()['files'].items():
   target=root/logical;target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(CHAPTER/record['path'],target)
   if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Source changed while preparing '+logical)
  _ACTIVE_SOURCE=root
  original_spec=importlib.util.spec_from_file_location
  def scoped_spec(name,location=None,*,loader=None,submodule_search_locations=None):
   if location is not None and loader is None:
    path=Path(location).resolve()
    if path.suffix=='.py' and (path.is_relative_to(root) or path.is_relative_to(CHAPTER/'scripts/execution')):
     package=path.name=='__init__.py' or submodule_search_locations is not None
     loader=Loader(path,package)
   return original_spec(name,location,loader=loader,submodule_search_locations=submodule_search_locations)
  importlib.util.spec_from_file_location=scoped_spec
  try:yield root
  finally:
   importlib.util.spec_from_file_location=original_spec
   _ACTIVE_SOURCE=None
def configure_imports():
 for relative in manifest()['physical_import_roots']:
  value=str(CHAPTER/relative)
  if value not in sys.path:sys.path.insert(0,value)
 if not any(isinstance(f,Finder) and f.chapter==CHAPTER for f in sys.meta_path):sys.meta_path.insert(0,Finder())
class Loader(importlib.abc.Loader):
 def __init__(self,path,package):self.path,self.package=path,package
 def create_module(self,spec):return None
 def exec_module(self,module):
  from frozen_adapter import _Paths
  from portable_paths import resolve_path
  def resolve(value):
   local=resolve_source_path(value)
   return local if local is not None else resolve_path(value)
  tree=_Paths().visit(rebase_source_guards(rebase_data_joins(rebase_maintenance_bootstrap(ast.parse(self.path.read_bytes(),filename=str(self.path))))));ast.fix_missing_locations(tree)
  module.__file__=str(self.path);module.__dict__['_portable_resolve']=resolve
  module.__dict__['_selected_source_location_allowed']=source_location_allowed
  if self.package:module.__path__=[str(self.path.parent)]
  exec(compile(tree,str(self.path),'exec'),module.__dict__)
class Finder(importlib.abc.MetaPathFinder):
 chapter=CHAPTER
 def find_spec(self,fullname,path=None,target=None):
  names=[fullname]
  prefix='chapters.'+CHAPTER.name+'.'
  if fullname.startswith(prefix):names.insert(0,fullname[len(prefix):])
  files=manifest()['files']
  for name in names:
   suffix=name.replace('.','/')
   for root in manifest()['logical_import_roots']:
    stem=(root+'/' if root else '')+suffix
    for relative,package in [(stem+'/__init__.py',True),(stem+'.py',False)]:
     if relative in files:
      physical=source_path(relative)
      return importlib.util.spec_from_file_location(fullname,physical,loader=Loader(physical,package),submodule_search_locations=[str(physical.parent)] if package else None)
  return None
