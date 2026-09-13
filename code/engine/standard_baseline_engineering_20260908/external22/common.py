from pathlib import Path
import sys,json,hashlib,os,gzip
HERE=Path(__file__).resolve().parent
ENGINEERING=HERE.parent
ROOT=ENGINEERING.parent
OLD=ROOT/'fixed_evaluation_confirmation_20260908'
PREP=ROOT/'external_baseline_preparation_20260908'
sys.path.insert(0,str(ENGINEERING))
import standard_adapters as standard
import budget_core
from compat import CountCalls,initialize_population,legacy_initialization_seed,objective_array,update_archive,fixed_reference_hv
from data.loader import load_benchmark_set
from validation import chromosome,feasible,hv_independent,verify_saved
import numpy as np
ARMS=('de_proxy','de_random4','ucb','nsga2_shared','moead_shared')
SEEDS=tuple(range(9082201,9082206))
ENDPOINTS=(2000,5000,10000)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_bytes(canonical(x)+b'\n');t.replace(p)
def compressed(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp')
    with open(t,'wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:g.write(canonical(x)+b'\n')
    t.replace(p)
def raw_read(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def scientific(r):
    return standard.scientific_payload(r) if r['arm'] in standard.ARMS else budget_core.scientific_payload(r)
def solve(inst,dataset,name,seed,arm,evaluations=10000,endpoints=ENDPOINTS,progress_fn=None):
    fn=standard.solve if arm in standard.ARMS else budget_core.solve
    return fn(inst,dataset,name,seed,arm,evaluations,endpoints,progress_fn=progress_fn)
def verify(r,inst):
    if r['arm'] in standard.ARMS:return verify_saved(r,inst)
    assert r['evaluations']==r['decodes'] and r['dedicated_training_evaluations']==r['unselected_evaluations']==0
    initial=np.asarray(r['initial_objectives'])
    np.testing.assert_array_equal(r['reference'],initial.max(axis=0)*1.1)
    np.testing.assert_array_equal(r['wide_reference'],initial.max(axis=0)*1.5)
    assert len(initial)==len(r['initial_genes'])==100
    assert [x['evaluations'] for x in r['snapshots']]==r['endpoints']
    checked=0;ops=0
    with CountCalls() as count:
        for g,f in zip(r['initial_genes'],r['initial_objectives']):
            c=chromosome(inst,g);ops+=feasible(c)
            np.testing.assert_array_equal(c.objectives.to_array(),f);checked+=1
        for s in r['snapshots']:
            assert len(s['genes'])==len(s['objectives'])<=100
            for g,f in zip(s['genes'],s['objectives']):
                c=chromosome(inst,g);ops+=feasible(c)
                np.testing.assert_array_equal(c.objectives.to_array(),f);checked+=1
            F=np.asarray(s['objectives'])
            assert len(np.unique(F,axis=0))==len(F)
            for row in F:assert not np.any(np.all(F<=row,axis=1)&np.any(F<row,axis=1))
            for field,ref in (('hv','reference'),('hv_wide','wide_reference')):
                np.testing.assert_allclose(s[field],hv_independent(F,r[ref]),rtol=1e-12,atol=1e-8)
            assert s['outside_reference_points']==int(np.any(F>np.array(r['reference'])+1e-12,axis=1).sum())
        assert count.evaluations==count.decodes==checked
    return {'saved_schedules_checked':checked,'operation_assignments_checked':ops,
            'validation_evaluations':checked,'validation_decodes':checked}
def initial_identity(r):
    return hashlib.sha256(canonical({k:r[k] for k in
        ('initial_genes','initial_objectives','reference','wide_reference')})).hexdigest()
def generated_payload(inst):
    def enc(x):
        if isinstance(x,np.ndarray):return {'dtype':str(x.dtype),'shape':list(x.shape),'values':x.tolist()}
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,dict):return {'items':[[enc(k),enc(v)] for k,v in sorted(x.items(),key=lambda z:repr(z[0]))]}
        if isinstance(x,(list,tuple)):return [enc(v) for v in x]
        if x is None or isinstance(x,(str,int,float,bool)):return x
        raise TypeError(type(x).__name__)
    return enc(vars(inst))
def candidate_input(record):
    p=PREP/record['input_file'];assert sha(p)==record['input_sha256']
    inst=dict(load_benchmark_set(str(p.parent),num_agv=3))[p.name]
    assert inst.extension_seed==record['extension_seed']
    generated=generated_payload(inst)
    assert hashlib.sha256(canonical(generated)).hexdigest()==record['generated_sha256']
    return inst
