"""Study24: prospective baseline calibration and synthetic transport transfer.
No scientific source in earlier studies is edited. Configuration hooks are
process-local and restored after each call; all new decisions are logged.
"""
from pathlib import Path
from contextlib import contextmanager
import sys,json,hashlib,copy,random
import numpy as np
R=Path(__file__).resolve().parent;B=R.parent
E=B/'standard_baseline_engineering_20260908/external22'
D=B/'domain_adapter_engineering_20260908'
sys.path.insert(0,str(E));sys.path.insert(0,str(D))
import common as old
import adapter as domain
import budget_core
from compat import instance,CountCalls,Chromosome,initialize_population,legacy_initialization_seed
from src.problem.instance import FJSPAGVInstance
from validation import feasible,chromosome,hv_independent
sha=old.sha;canonical=old.canonical;atomic=old.atomic
compressed=old.compressed;raw_read=old.raw_read
PROTOCOL='study24_v1_development_calibration_then_frozen_synthetic_test'
DEV_SEEDS=(9082401,9082402,9082403)
TEST_SEEDS=(9082411,9082412,9082413)
ENDPOINTS=(2000,5000,10000)
ARMS=('de_proxy','de_random4','ucb_default','ucb_tuned','domain_default','domain_tuned')
CONTRASTS=tuple(('de_proxy',a) for a in ARMS[1:])
DEV_CASES=tuple([('brandimarte',f'Mk{x:02}.fjs') for x in (2,4,6,10)]+
                [('hurink_edata',f'la{x:02}.fjs') for x in (4,16,28,37)])
REGIMES=(('a1_t015',1,.15),('a1_t060',1,.60),('a5_t015',5,.15),('a5_t060',5,.60),('a3_t030',3,.30))
DEFAULTS={'ucb':'ucb_w500_c1','domain':'domain_m0.2_l10'}
CONFIGS={}
for w in (100,500,1000):
    for c in (.25,1.,2.):
        key=f'ucb_w{w}_c{c:g}'
        CONFIGS[key]={'family':'ucb','window':w,'c':c,'min_count':20}
for m in (.1,.2,.4):
    for n in (0,10,20):
        key=f'domain_m{m:g}_l{n}'
        CONFIGS[key]={'family':'domain','mutation':m,'local_seeds':n,'crossover':.8}
def read(p):return json.loads(Path(p).read_text())
def snapshot_instance(inst):
    out={}
    for k,v in vars(inst).items():
        if k=='processing_times':v=[[int(j),int(o),int(m),float(t)] for (j,o,m),t in sorted(v.items())]
        elif k=='compatible_machines':v=[[int(j),int(o),[int(x) for x in ms]] for (j,o),ms in sorted(v.items())]
        elif isinstance(v,np.ndarray):v=v.tolist()
        elif isinstance(v,np.generic):v=v.item()
        out[k]=v
    return out
def restore_instance(out):
    d=copy.deepcopy(out);inst=FJSPAGVInstance(d['num_jobs'],d['num_machines'],d['num_agv'],d['num_operations'])
    for k,v in d.items():
        if k=='processing_times':v={(j,o,m):t for j,o,m,t in v}
        elif k=='compatible_machines':v={(j,o):ms for j,o,ms in v}
        elif k in ('distance_matrix','machine_proc_power','machine_idle_power','machine_setup_power','machine_setup_time','layout_coordinates'):
            v=None if v is None else np.asarray(v,dtype=float)
        elif k in ('agv_load_power_params','agv_empty_power_params'):v=tuple(v)
        setattr(inst,k,v)
    assert canonical(snapshot_instance(inst))==canonical(out)
    return inst
def generate_base(scale,flex,rep):
    """Independent synthetic job tables; not a named public/industrial benchmark."""
    sid=(0 if scale=='small' else 2)+(0 if flex=='low' else 1)
    seed=908248001+sid*4+rep
    rng=np.random.RandomState(seed)
    jobs,machines=(10,5) if scale=='small' else (30,10)
    counts=rng.randint(3,8,jobs).tolist() if scale=='small' else rng.randint(5,11,jobs).tolist()
    inst=FJSPAGVInstance(jobs,machines,3,counts)
    nchoice=max(1,int(np.ceil(machines*(.25 if flex=='low' else .75))))
    for j,n in enumerate(counts):
        for o in range(n):
            choices=sorted(int(x) for x in rng.choice(machines,nchoice,replace=False))
            base=float(rng.randint(5,51))
            inst.compatible_machines[j,o]=choices
            for m in choices:inst.processing_times[j,o,m]=float(max(1,round(base*rng.uniform(.8,1.2))))
    return inst,seed,908249001+sid*4+rep
def input_for(cell):
    p=R/cell['input_file'];assert sha(p)==cell['input_sha256']
    return restore_instance(read(p))
@contextmanager
def configure(config_key):
    cfg=CONFIGS[config_key];original_prepare=budget_core.prepare_ucb
    original_config=domain.CONFIG
    try:
        if cfg['family']=='ucb':
            def prepare(*args,**kwargs):
                state=original_prepare(*args,**kwargs)
                s=state['selector'];s.c=cfg['c'];s.window=cfg['window']
                assert s.min_count==cfg['min_count'] and not s.records
                return state
            budget_core.prepare_ucb=prepare
        else:
            domain.CONFIG=dict(original_config,mutation=cfg['mutation'],
                local_seeds=cfg['local_seeds'],crossover=cfg['crossover'])
        yield
    finally:
        budget_core.prepare_ucb=original_prepare;domain.CONFIG=original_config
def run_solver(inst,dataset,name,seed,key,evaluations=10000,endpoints=ENDPOINTS,progress=None):
    if key in ('de_proxy','de_random4'):
        raw=budget_core.solve(inst,dataset,name,seed,key,evaluations,endpoints,progress_fn=progress)
        cfg={'family':'frozen_proposal','arm':key}
    else:
        cfg=CONFIGS[key]
        with configure(key):
            if cfg['family']=='ucb':
                raw=budget_core.solve(inst,dataset,name,seed,'ucb',evaluations,endpoints,progress_fn=progress)
            else:raw=domain.solve(inst,dataset,name,seed,evaluations,endpoints,progress=progress)
    raw['study24_configuration']={'key':key,'settings':cfg,'study':PROTOCOL}
    return raw
def scientific(raw):
    payload={k:v for k,v in raw.items() if k!='study24_configuration'}
    content=domain.scientific(payload) if raw['arm']=='nsga2_vns_reverse_adapted' else budget_core.scientific_payload(payload)
    return {'configuration':raw['study24_configuration'],'payload':content}
def verify(raw,inst,replay=None):
    if replay is not None:assert scientific(raw)==scientific(replay),'Exact complete trajectory mismatch'
    if raw['arm']=='nsga2_vns_reverse_adapted':
        checks=domain.verify_pair(raw,raw if replay is None else replay,inst)
        checks.pop('exact_replay',None)
        checks['audit_scope']='all evaluated candidates'
    else:
        checks=old.verify(raw,inst)
        checks['audit_scope']='initial population and every saved endpoint archive'
    checks['exact_full_replay']=None if replay is None else True
    checks['replay_evaluations']=0 if replay is None else replay['evaluations']
    return checks
def summaries(raw,cell):
    scale=np.asarray(raw['reference'])
    rows=[]
    for s in raw['snapshots']:
        F=np.asarray(s['objectives'])
        norm=F/scale
        preferences=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(1/3,1/3,1/3))
        # Single complete, feasible schedule per declared preference; no component splicing.
        choices=[]
        for w in preferences:
            score=np.max(norm*np.asarray(w),axis=1)+.05*(norm*np.asarray(w)).sum(axis=1)
            k=int(np.argmin(score))
            choices.append({'weights':list(w),'objectives':F[k].tolist(),'genes':s['genes'][k]})
        rows.append(dict(task_id=cell['id'],phase=cell['phase'],base_id=cell['base_id'],
          stratum=cell['stratum'],regime=cell['regime'],seed=cell['seed'],arm=cell['arm'],
          config_key=raw['study24_configuration']['key'],evaluations=s['evaluations'],
          hv=s['hv'],hv_wide=s['hv_wide'],hv_normalized=s['hv']/float(np.prod(scale)),
          archive_size=len(F),outside_reference_points=int(np.any(F>scale+1e-12,axis=1).sum()),
          objective_component_minima=F.min(axis=0).tolist(),preference_schedules=choices,
          initial_identity=old.initial_identity(raw)))
    return rows
def resolve(cell):
    if cell['phase']=='development':return cell['arm']
    if cell['arm'] in ('de_proxy','de_random4'):return cell['arm']
    family,mode=cell['arm'].split('_')
    if mode=='default':return DEFAULTS[family]
    selection=read(R/'SELECTION.json')
    assert selection['frozen_sha256']==sha(R/'FROZEN.json')
    assert sha(R/'SELECTION.json')==read(R/'TEST_LAUNCH_LOCK.json')['selection_sha256']
    return selection['chosen'][family]
def lockcheck():
    lock=read(R/'FROZEN.json')
    for p,h in lock['scientific_files'].items():assert sha(B/p)==h,p
    return lock
