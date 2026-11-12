"""Chapter 5 source facades and explicit current-formal entry; no automatic science."""
from pathlib import Path
import argparse, importlib, importlib.abc, importlib.machinery, importlib.util
import json, os, sys
from portable_paths import project_root, final_data_root, chapter_data_root, resolve_path, eurosat_root

CURRENT_BUDGET = 100608

def chapter_root():
    return project_root() / 'chapters' / 'chapter05'

def frozen_source_root():
    return chapter_root() / 'source'

def configure_internal_paths():
    base = frozen_source_root() / '_internal' / 'src'
    roots = [base, base / 'chapter4_credit_assignment', base / 'chapter4_sample_efficiency_v5',
             base / 'chapter5_multiobjective', frozen_source_root(), project_root()]
    for root in reversed(roots):
        value = str(root)
        if value not in sys.path: sys.path.insert(0, value)
    os.environ.setdefault('CH5_FULL_SPEND_DATA_ROOT', str(eurosat_root()))
    return roots

def import_internal(name):
    configure_internal_paths()
    if name.startswith(('allocation_morl', 'chapter5_cross_server_20261103', 'chapter5_validation_20261104')):
        importlib.import_module('chapter5_full_spend_20261104.bridge').bootstrap()
    return importlib.import_module(name)

def expose(name, namespace):
    module = import_internal(name)
    for key, value in vars(module).items():
        if not key.startswith('__'): namespace[key] = value
    namespace['__all__'] = list(getattr(module, '__all__', [k for k in vars(module) if not k.startswith('_')]))
    namespace['CANONICAL_SOURCE_MODULE'] = name

def resolve_scientific_path(value):
    text = str(value)
    frozen = frozen_source_root()
    if text == str(frozen) or text.startswith(str(frozen) + '/'): return Path(text)
    manifest = json.loads((chapter_root() / 'source_manifest.json').read_text())
    origins = {manifest['source_root']: frozen, '/root/gamewar/resource_allocation_rl/chapters/chapter05/_internal/src': frozen/'_internal/src'}
    for old in sorted(origins, key=len, reverse=True):
        if text == old: return origins[old]
        if text.startswith(old + '/'): return origins[old] / text[len(old) + 1:]
    if text=='/hy-tmp/resource_allocation_results/chapter5/full_spend_revision_20261104/execution/shared_data/eurosat_rgb':
        return eurosat_root()
    formal_origins=['/hy-tmp/resource_allocation_results/chapter5/formal_random_5seed_100k_20261107_v2']
    for old in formal_origins:
        if text==old:return chapter_data_root(5)/'formal'
        if text.startswith(old+'/'):
            suffix=text[len(old)+1:]
            # Execution/source was handled by the longer immutable-source binding above.
            target=chapter_data_root(5)/'formal'/suffix
            return resolve_scientific_path(target)
    # The external artifact bundle retains its original execution/source names.
    # Resolve those source-only descendants to the one published source tree.
    formal = chapter_data_root(5) / 'formal'
    aliases = {formal / 'execution/source': frozen,
               formal / 'execution/source_manifest.json': chapter_root() / 'source_manifest.json',
               formal / 'execution': chapter_root() / 'execution'}
    for old, new in aliases.items():
        old_text = str(old)
        if text == old_text: return new
        if text.startswith(old_text + '/'):
            return new / text[len(old_text) + 1:]
    return resolve_path(value)

class _SourceLoader(importlib.abc.Loader):
    def __init__(self, path, package): self.path, self.package = path, package
    def create_module(self, spec): return None
    def exec_module(self, module):
        from frozen_adapter import compiled_source
        module.__file__ = str(self.path)
        module.__dict__['_portable_resolve'] = resolve_scientific_path
        if self.package: module.__path__ = [str(self.path.parent)]
        exec(compiled_source(self.path), module.__dict__)

class _SourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith('.py'): return None
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(frozen_source_root()): return None
        package = spec.submodule_search_locations is not None
        return importlib.util.spec_from_file_location(fullname, origin, loader=_SourceLoader(origin, package),
            submodule_search_locations=[str(origin.parent)] if package else None)

def verify_frozen_source():
    import hashlib
    manifest = json.loads((chapter_root() / 'source_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        actual = hashlib.sha256((frozen_source_root() / relative).read_bytes()).hexdigest()
        if actual != expected: raise ValueError('Frozen scientific source changed: ' + relative)
    return len(manifest['files'])

def inspect():
    return {'chapter': 5, 'formal_learning_steps': CURRENT_BUDGET,
            'maintained_source': str(chapter_root()), 'formal_source': str(frozen_source_root()),
            'formal_source_files': verify_frozen_source(),
            'native_variants': ['full', 'only_j', 'only_c', 'scalar_value'],
            'baselines': ['capql', 'qpensieve', 'cola', 'morld', 'pasta'],
            'automatic_queue': False, 'model_loading': 0, 'environment_steps': 0}

def _source_within_release(path):
    path=Path(path).resolve()
    return path.is_relative_to(project_root()) or path.is_relative_to(final_data_root())

def _entry_code(script):
    # The one original HY-only source-location guard becomes a release-root guard.
    # Source hashes, registered budgets, split guards and every learning node stay unchanged.
    import ast
    from frozen_adapter import _Paths
    tree=ast.parse(script.read_bytes(),filename=str(script))
    class SourceLocation(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='CHAPTER' and node.func.attr=='is_relative_to'
                and len(node.args)==1 and isinstance(node.args[0],ast.Constant)
                and node.args[0].value=='/hy-tmp'):
                self.count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_source_within_release',ctx=ast.Load()),args=[node.func.value],keywords=[]),node)
            return self.generic_visit(node)
    adapter=SourceLocation();tree=adapter.visit(tree)
    if adapter.count != (1 if script.name=='own_worker.py' else 0):raise ValueError('Unexpected scientific source-location guard geometry')
    class ScopeJoins(ast.NodeTransformer):
        def visit_BinOp(self,node):
            left=node
            while isinstance(left,ast.BinOp) and isinstance(left.op,ast.Div):left=left.left
            if isinstance(node.op,ast.Div) and isinstance(left,ast.Name) and left.id in ('R','ROOT','root'):
                return ast.copy_location(ast.Call(func=ast.Name(id='_portable_resolve',ctx=ast.Load()),args=[node],keywords=[]),node)
            return self.generic_visit(node)
    tree=ScopeJoins().visit(tree)
    tree=_Paths().visit(tree);ast.fix_missing_locations(tree)
    return compile(tree,str(script),'exec')

def _run_formal(filename, arguments):
    configure_internal_paths()
    if not any(isinstance(f, _SourceFinder) for f in sys.meta_path): sys.meta_path.insert(0, _SourceFinder())
    script = chapter_root() / 'execution' / filename
    old_argv, old_path = sys.argv[:], sys.path[:]
    sys.argv = [str(script), *arguments]
    sys.path.insert(0, str(script.parent))
    try:
        namespace={'__name__':'_chapter5_registered_entry','__file__':str(script),
             '__package__':None,'_portable_resolve':resolve_scientific_path,
             '_source_within_release':_source_within_release}
        exec(_entry_code(script),namespace)
        # The shallow maintenance filename must not redefine the archived science scope.
        formal=chapter_data_root(5)/'formal'
        if filename=='baseline_worker.py':
            namespace.update(ROOT=formal,SOURCE_ROOT=frozen_source_root(),
                SOURCE_MANIFEST=chapter_root()/'source_manifest.json')
        elif filename=='evaluate_formal_fixed.py':namespace['R']=formal
        if filename=='baseline_worker.py':
            config_path=arguments[arguments.index('--config')+1]
            if '--validate-only'in arguments:print(json.dumps(namespace['validate_only'](config_path),ensure_ascii=False,sort_keys=True))
            else:namespace['run'](config_path)
        else:namespace['main']()
    finally: sys.argv, sys.path[:] = old_argv, old_path


def main(kind, argv=None):
    parser = argparse.ArgumentParser(description='Explicit current Chapter 5 formal entry; default is metadata validation.')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly run the registered new operation.')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.config is None:
        if args.execute: parser.error('--execute requires an explicit registered --config')
        print(json.dumps(inspect(), ensure_ascii=False, indent=2)); return 0
    cfg = json.loads(args.config.read_text())
    if int(cfg.get('learning_steps', cfg.get('registered_training_steps', CURRENT_BUDGET))) != CURRENT_BUDGET:
        parser.error('The current formal entry requires 100608 learning steps; historical 61440 is not a default.')
    method = str(cfg.get('method', cfg.get('profile', 'full'))).lower()
    baseline = method in {'capql', 'qpensieve', 'cola', 'morld', 'pasta'}
    filename = ('baseline_worker.py' if baseline else 'own_worker.py') if kind == 'train' else 'evaluate_formal_fixed.py'
    arguments = ['--config', str(args.config.resolve())]
    if args.execute and not args.validate_only:
        output = cfg.get('output', cfg.get('output_dir'))
        if not output: parser.error('Execution requires an explicit new output directory in the config.')
        dest = resolve_path(output).resolve()
        if dest.exists() and any(dest.iterdir()): parser.error('Refusing to replay an existing scientific output directory.')
    else: arguments.append('--validate-only')
    verify_frozen_source()
    _run_formal(filename, arguments)
    return 0
