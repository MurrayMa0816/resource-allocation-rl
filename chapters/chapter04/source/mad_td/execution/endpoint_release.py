"""Stdlib-only source/endpoints barrier, evaluated before opening any TEST Env."""
from pathlib import Path
import hashlib,json
ROOT=Path('/hy-tmp/resource_allocation_results/chapter4/mad_td_formal_20261107')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def source_bindings(required=True):
    seal=ROOT/'protocol/science_code_seal.json'
    if not seal.exists():
        if required:raise RuntimeError('Independent formal science source seal is not prepared')
        return {str(p):sha(p)for base in [ROOT/'source',ROOT/'execution']for p in sorted(base.rglob('*'))if p.is_file()and'__pycache__'not in p.parts}
    files=read(seal)['files_sha256']
    if not files or any(sha(p)!=h for p,h in files.items()):raise RuntimeError('Frozen formal source/data identity differs')
    return files
def validate_release(method,domain,seed,model,*,gate_path=None):
    """No backend imports, model deserialization or Task construction here."""
    gate_path=Path(gate_path or ROOT/'protocol/endpoint_release_gate.json').resolve()
    if not gate_path.is_relative_to(ROOT):raise ValueError('Only this formal scope release gate is accepted')
    gate=read(gate_path);seal=ROOT/'protocol/science_code_seal.json';registry=ROOT/'protocol/formal_registry.json'
    required=dict(schema='chapter4-madtd-formal-all-endpoints-release-v1',status='PASS',all_training_completed=True,endpoint_count=100,training_curve_count=90,test_master=2026110694199,test_cases=256,test_phase='final',no_PGR=True)
    if any(gate.get(k)!=v for k,v in required.items()):raise RuntimeError('All100 endpoints have not been independently frozen for TEST')
    if gate['science_code_seal_sha256']!=sha(seal)or gate['registry_sha256']!=sha(registry):raise RuntimeError('Endpoint release is not bound to this source and registry')
    files=source_bindings();endpoints=gate['fixed_endpoints']
    if len(endpoints)!=100 or len({(x['method'],x['domain'],int(x['seed']))for x in endpoints})!=100 or any(x['method'].casefold()=='pgr'for x in endpoints):raise AssertionError('Unique non-PGR100-model seal required')
    mad=[x for x in endpoints if x['method']=='mad_td']
    if {(x['domain'],int(x['seed']))for x in mad}!={(d,s)for d in ['radio','v5']for s in range(5)}:raise AssertionError('All ten MAD-TD endpoints required')
    for row in mad:
        p=Path(row['model']).resolve();completion=Path(row['completion']).resolve()
        if not p.is_relative_to(ROOT/'jobs')or not completion.is_relative_to(ROOT/'jobs'):raise AssertionError('MAD-TD model/completion must belong to this formal training scope')
        if sha(p)!=row['model_sha256']or sha(completion)!=row['completion_sha256']:raise AssertionError('Released fixed model or completion changed')
        done=read(completion)
        if not(done.get('success')is True and done.get('phase')=='train'and done.get('engineering_only')is False and done.get('formal_training_performed')is True and done.get('training_steps')==100608):raise AssertionError('Complete actual formal training required')
        if(done.get('domain'),int(done.get('seed',-1)))!=(row['domain'],int(row['seed']))or done.get('model_sha256')!=row['model_sha256']:raise AssertionError('Model identity differs from sealed record')
        if done.get('source_binding')!=files or done.get('science_code_seal_sha256')!=sha(seal):raise AssertionError('Training model does not bind current frozen scientific implementation')
    chosen=[x for x in endpoints if x['method']==method and x['domain']==domain and int(x['seed'])==int(seed)]
    if len(chosen)!=1:raise AssertionError('Requested endpoint identity absent from release')
    chosen=chosen[0]
    if Path(chosen['model']).resolve()!=Path(model).resolve()or sha(model)!=chosen['model_sha256']or sha(chosen['completion'])!=chosen['completion_sha256']:raise AssertionError('Requested TEST model is not the released fixed endpoint')
    return dict(gate=str(gate_path),gate_sha256=sha(gate_path),registry_sha256=gate['registry_sha256'],science_code_seal_sha256=gate['science_code_seal_sha256'],released_model=chosen,all100_endpoints_frozen=True,all10_MADTD_source_bound_models_verified=True,TEST_master=2026110694199,TEST_cases=256,TEST_phase='final')
