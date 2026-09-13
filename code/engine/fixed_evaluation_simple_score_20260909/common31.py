"""Study31: matched physical-score aggregation comparison.
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
PROTOCOL='study31_fixed_evaluation_simple_score_v1'
ARMS=('de_proxy','de_weighted_sum','de_random4')
from scoring31 import WeightedSumDecomposition,weighted_sum,assert_matched_source
from surrogate import scalar,directions
STRATA=('small_low','small_high','medium_low','medium_high')
REGIMES=s24.REGIMES
SEEDS=(909312001,909312002,909312003)
ENDPOINTS=(2000,5000,10000)
SOURCE_NAMES=('common31.py','prepare31.py','run31.py','analyze31.py','test31.py','engineering31.py','scoring31.py','PROTOCOL.md')
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

@contextmanager
def component(mode):
    assert mode in ARMS
    previous=s24.budget_core.Decomposition
    try:
        if mode=='de_weighted_sum':
            assert previous is online_core.Decomposition
            s24.budget_core.Decomposition=WeightedSumDecomposition
        yield
    finally:s24.budget_core.Decomposition=previous

def run_solver(inst,dataset,name,seed,arm,evaluations=10000,endpoints=ENDPOINTS,callback=None):
    assert arm in ARMS
    before=digest(s24.snapshot_instance(inst))
    key='de_proxy' if arm=='de_weighted_sum' else arm
    with component(arm):
        raw=s24.run_solver(inst,dataset,name,seed,key,evaluations,endpoints,callback)
    assert digest(s24.snapshot_instance(inst))==before,'Real decoder instance was mutated'
    raw['study31_configuration']={'arm':arm,'protocol':PROTOCOL,
      'changed_stage':'cheap candidate ranking only' if arm=='de_weighted_sum' else 'unchanged comparator'}
    assert raw['evaluations']==raw['decodes']==evaluations
    assert raw['unselected_evaluations']==raw['dedicated_training_evaluations']==0
    return raw
def scientific(raw):
    r={k:v for k,v in raw.items() if k!='study31_configuration'}
    return dict(study31_configuration=raw['study31_configuration'],payload=s24.scientific(r))
def verify(raw,inst,replay=None):
    if replay is not None:assert scientific(raw)==scientific(replay),'Complete scientific replay mismatch'
    arm=raw['study31_configuration']['arm']
    initial=np.asarray(raw['initial_objectives'])
    scale=np.maximum(initial.max(axis=0),1.)
    weights=directions()
    assert len(raw['events'])==raw['evaluations']-100
    for k,event in enumerate(raw['events']):
        assert event['step']==k and event['direction']==k%len(weights)
        assert event['brood_size']==4 and len(event['candidate_fingerprints'])==4
        assert event['observations_before']==event['observations_after']==0
        assert event['fit_count']==0 and event['explore']==(k%4==3)
        if arm=='de_random4' or event['explore']:
            assert event['selected']==event['random_index'] and event['predictions'] is None
        else:
            predictions=np.asarray(event['predictions'])
            assert predictions.shape==(4,3) and np.isfinite(predictions).all()
            scoring=weighted_sum if arm=='de_weighted_sum' else scalar
            assert event['selected']==int(np.argmin(scoring(predictions,weights[event['direction']],scale)))
    check=s24.verify(raw,inst)
    check['exact_full_replay']=None if replay is None else True
    check['replay_evaluations']=0 if replay is None else replay['evaluations']
    check['preselection_events_checked']=len(raw['events'])
    check['scoring_only_source_check']=assert_matched_source()
    return check
def summary(raw,cell):
    rows=s24.summaries(raw,dict(cell,phase='study31'))
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
    assert len(records)==80 and len(tasks)==720 and sum(c['replay'] for c in tasks)==48
    assert len({c['id'] for c in tasks})==720
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
            bid=f'score31_{stratum}_{rep:02}'
            for regime,_,_ in REGIMES:
                record=lookup[bid,regime]
                for seed in SEEDS:
                    for arm in ARMS:
                        name=bid+'_'+regime
                        cells.append(dict(record,id=f'{name}_{seed}_{arm}',dataset='study31_synthetic_v1',
                          name=name,seed=seed,arm=arm,replay=seed==SEEDS[0] and regime=='a3_t030'))
    random.Random(20260909311).shuffle(cells)
    return cells
def generate_base(stratum,rep):
    scale,flex=stratum.split('_');sid=STRATA.index(stratum)
    pseed=909310001+sid*4+rep;eseed=909311001+sid*4+rep
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
