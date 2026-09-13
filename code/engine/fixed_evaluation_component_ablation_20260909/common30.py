"""Study30: fixed-evaluation physical-score component ablations.
Old science is imported read-only. Each child process owns and restores hooks.
"""
from pathlib import Path
from contextlib import contextmanager
import copy,gzip,hashlib,json,math,os,random,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
B=ROOT.parent
R=B/'baseline_calibration_and_transport_20260908'
H=B/'equal_walltime_harness_engineering_20260909'
PY=B/'fixed_evaluation_confirmation_20260908/.venv/bin/python'
sys.path.insert(0,str(R))
import study24 as s24
import online_core
from surrogate import EncodingFeatures as OriginalFeatures
from compat import CountCalls,initialize_population,legacy_initialization_seed,objective_array
PROTOCOL='study30_fixed_evaluation_component_removal_v1'
ARMS=('de_proxy','de_random4','ucb_w500_c1','proxy_no_transport','proxy_no_energy','proxy_no_load')
ABLATIONS=ARMS[3:]
STRATA=('small_low','small_high','medium_low','medium_high')
REGIMES=s24.REGIMES
SEEDS=(909302001,909302002,909302003)
ENDPOINTS=(2000,5000,10000)
SOURCE_NAMES=('common30.py','prepare30.py','run30.py','analyze30.py','test30.py','engineering30.py','PROTOCOL.md')
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
S24_SHA='1561fbb3a99a0edc51b4be236c34f459bc21e66aafb7b5487f41864d5d0c21e1'
H_SHA='359229af022af51b890895d0dc2b8ebe60d198e2db41210c458deea7c71d0515'
sha=s24.sha;canonical=s24.canonical;read=s24.read
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
def progress(p,v):s24.atomic(p,v)
def packed(p,v):
    with Path(p).open('xb') as out:
        with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0,compresslevel=5) as z:z.write(canonical(v))
def unpack(p):
    with gzip.open(p,'rt') as f:return json.load(f)

class RemovedFeatures(OriginalFeatures):
    """Only the cheap preselection calculation changes; decoder sees the real instance."""
    def __init__(self,inst,scale,mode):
        assert mode in ABLATIONS
        self.mode=mode
        scoring_inst=inst
        if mode=='proxy_no_transport':
            scoring_inst=copy.deepcopy(inst)
            scoring_inst.distance_matrix=np.zeros_like(inst.distance_matrix)
        super().__init__(scoring_inst,scale)
    def __call__(self,chrom):
        x,p=super().__call__(chrom)
        x=x.copy();p=p.copy()
        if self.mode=='proxy_no_energy':p[1]=0.
        elif self.mode=='proxy_no_load':p[2]=0.
        x[:3]=p/self.scale
        return x,p

@contextmanager
def component(mode):
    previous=online_core.EncodingFeatures
    try:
        if mode in ABLATIONS:
            def factory(inst,scale):return RemovedFeatures(inst,scale,mode)
            online_core.EncodingFeatures=factory
        else:assert mode in ARMS[:3]
        yield
    finally:online_core.EncodingFeatures=previous

def run_solver(inst,dataset,name,seed,arm,evaluations=10000,endpoints=ENDPOINTS,callback=None):
    assert arm in ARMS
    before=digest(s24.snapshot_instance(inst))
    key='de_proxy' if arm in ABLATIONS else arm
    with component(arm):
        raw=s24.run_solver(inst,dataset,name,seed,key,evaluations,endpoints,callback)
    assert digest(s24.snapshot_instance(inst))==before,'Real decoder instance was mutated'
    raw['study30_configuration']={'arm':arm,'protocol':PROTOCOL,
      'changed_stage':'cheap preselection only' if arm in ABLATIONS else 'unchanged comparator'}
    assert raw['evaluations']==raw['decodes']==evaluations
    assert raw['unselected_evaluations']==raw['dedicated_training_evaluations']==0
    return raw
def scientific(raw):
    r={k:v for k,v in raw.items() if k!='study30_configuration'}
    return dict(study30_configuration=raw['study30_configuration'],payload=s24.scientific(r))
def verify(raw,inst,replay=None):
    if replay is not None:assert scientific(raw)==scientific(replay),'Complete scientific replay mismatch'
    check=s24.verify(raw,inst)
    check['exact_full_replay']=None if replay is None else True
    check['replay_evaluations']=0 if replay is None else replay['evaluations']
    return check
def summary(raw,cell):
    rows=s24.summaries(raw,dict(cell,phase='study30'))
    for row,s in zip(rows,raw['snapshots']):
        row['arm']=cell['arm'];row['protocol']=PROTOCOL
        row['hv_wide_normalized']=s['hv_wide']/float(np.prod(raw['wide_reference']))
        row['search_seconds']=raw['search_seconds'];row['cpu_seconds']=raw['cpu_seconds']
        row['reference']=raw['reference'];row['wide_reference']=raw['wide_reference']
    return rows
def inherited_lock():
    assert sha(R/'FROZEN.json')==S24_SHA
    assert sha(H/'ENGINEERING_FROZEN.json')==H_SHA
    locked=read(H/'ENGINEERING_FROZEN.json')['dependency_hashes']
    assert len(locked)==476
    for n,h in locked.items():assert sha(B/n)==h,('Inherited source changed',n)
    return locked
def methods_lock():
    f=read(ROOT/'METHODS_FROZEN.json')
    assert f['protocol']==PROTOCOL and f['formal_runs_before_freeze']==0
    assert set(f['sources'])==set(SOURCE_NAMES)
    for n,h in f['sources'].items():assert sha(ROOT/n)==h,('New source changed',n)
    assert f['dependencies']==inherited_lock()
    return sha(ROOT/'METHODS_FROZEN.json')
def inputs_lock():
    mh=methods_lock();f=read(ROOT/'INPUTS_FROZEN.json')
    assert f['methods_sha256']==mh and f['formal_runs_before_freeze']==0
    assert sha(ROOT/'INPUT_MANIFEST.json')==f['input_manifest_sha256']
    assert sha(ROOT/'TASKS.json')==f['tasks_sha256']
    records=read(ROOT/'INPUT_MANIFEST.json');tasks=read(ROOT/'TASKS.json')
    assert len(records)==80 and len(tasks)==1440 and sum(c['replay'] for c in tasks)==96
    assert len({c['id'] for c in tasks})==1440
    lookup={(r['base_id'],r['regime']):r for r in records}
    assert tasks==schedule(lookup)
    for r in records:
        assert (ROOT/r['input_file']).resolve().parent==ROOT/'inputs'
        assert sha(ROOT/r['input_file'])==r['input_sha256']
    return sha(ROOT/'INPUTS_FROZEN.json')
def schedule(lookup):
    cells=[]
    for stratum in STRATA:
        for rep in range(4):
            bid=f'comp30_{stratum}_{rep:02}'
            for regime,_,_ in REGIMES:
                record=lookup[bid,regime]
                for seed in SEEDS:
                    for arm in ARMS:
                        name=bid+'_'+regime
                        cells.append(dict(record,id=f'{name}_{seed}_{arm}',dataset='study30_synthetic_v1',
                          name=name,seed=seed,arm=arm,replay=seed==SEEDS[0] and regime=='a3_t030'))
    random.Random(20260909301).shuffle(cells)
    return cells
def generate_base(stratum,rep):
    scale,flex=stratum.split('_');sid=STRATA.index(stratum)
    pseed=909300001+sid*4+rep;eseed=909301001+sid*4+rep
    rng=np.random.RandomState(pseed)
    jobs,machines=(10,5) if scale=='small' else (30,10)
    counts=rng.randint(3,8,jobs).tolist() if scale=='small' else rng.randint(5,11,jobs).tolist()
    inst=s24.FJSPAGVInstance(jobs,machines,3,counts)
    nchoice=max(1,int(np.ceil(machines*(.25 if flex=='low' else .75))))
    for j,n in enumerate(counts):
        for o in range(n):
            choices=sorted(int(x) for x in rng.choice(machines,nchoice,replace=False))
            base=float(rng.randint(5,51));inst.compatible_machines[j,o]=choices
            for m in choices:inst.processing_times[j,o,m]=float(max(1,round(base*rng.uniform(.8,1.2))))
    return inst,pseed,eseed
def exposed_instance():
    p=R/'inputs/dev_brandimarte_Mk02.fjs.json'
    assert sha(R/'FROZEN.json')==S24_SHA
    expected=read(R/'FROZEN.json')['scientific_files'][str(p.relative_to(B))]
    assert sha(p)==expected
    return s24.restore_instance(read(p))
