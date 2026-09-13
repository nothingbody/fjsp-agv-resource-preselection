"""Pinned pymoo search logic with a shared, legal FJSP-AGV variation adapter."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,random,time
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'fixed_evaluation_confirmation_20260908'
# Public packaging: locate dependencies relative to this file, not a host prefix.
assert OLD.is_dir(), 'Missing sibling scientific source directory'
sys.path.insert(0,str(OLD))
from compat import (Chromosome,CountCalls,initialize_population,legacy_initialization_seed,
                    objective_array,update_archive,fixed_reference_hv)
from time_core import genes
from surrogate import joint_proposal
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.operators.mutation.nom import NoMutation
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.pbi import PBI

ARMS=('nsga2_shared','moead_shared')
GENES=('os','ma','agv_assign','agv_speed')
PROTOCOL='standard_baseline_engineering_v1_20260908'

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def fingerprint(c):
    h=hashlib.sha256()
    for k in GENES:h.update(np.asarray(getattr(c,k),dtype='<i8').tobytes())
    return h.hexdigest()

def legal(c):
    i=c.instance;n=sum(i.num_operations)
    assert all(isinstance(getattr(c,k),np.ndarray) and getattr(c,k).shape==(n,) and
               np.issubdtype(getattr(c,k).dtype,np.integer) for k in GENES)
    assert np.all((c.os>=0)&(c.os<i.num_jobs))
    assert np.array_equal(np.bincount(c.os,minlength=i.num_jobs),i.num_operations)
    bounds=np.array([len(i.get_compatible_machines(j,o)) for j in range(i.num_jobs)
                     for o in range(i.num_operations[j])])
    assert np.all((c.ma>=0)&(c.ma<bounds))
    assert np.all((c.agv_assign>=0)&(c.agv_assign<i.num_agv))
    assert np.all((c.agv_speed>=0)&(c.agv_speed<i.num_speeds))
    return True

class JointVariation(Crossover):
    """One proposal; mutation is already inside the frozen joint proposal."""
    def __init__(self,inst):
        super().__init__(2,1,prob=1.0)
        self.features=SimpleNamespace(choices=np.array([len(inst.get_compatible_machines(j,o))
            for j in range(inst.num_jobs) for o in range(inst.num_operations[j])]))
        self.proposals=0
        self.digest=hashlib.sha256()
    def _do(self,problem,X,**kwargs):
        Y=np.empty((1,X.shape[1],1),dtype=object)
        for k in range(X.shape[1]):
            a,b=X[0,k,0],X[1,k,0]
            before=(fingerprint(a),fingerprint(b))
            child=joint_proposal(a,b,np.random,self.features)
            assert legal(child) and child._objectives is None and child._schedule is None
            assert before==(fingerprint(a),fingerprint(b)),'Variation changed a parent'
            assert all(not np.shares_memory(getattr(child,g),getattr(p,g)) for g in GENES for p in (a,b))
            self.digest.update(canonical([*before,fingerprint(child)]))
            self.proposals+=1;Y[0,k,0]=child
        return Y

class ScheduleProblem(ElementwiseProblem):
    def __init__(self,scale,counter):
        super().__init__(n_var=1,n_obj=3,n_ieq_constr=0,n_eq_constr=0,vtype=object)
        self.scale=scale;self.counter=counter;self.events=[]
    def _evaluate(self,x,out,*args,**kwargs):
        c=x[0];assert legal(c)
        assert c._objectives is None and c._schedule is None,'New proposal inherited an evaluation cache'
        before=self.counter.evaluations;fp=fingerprint(c)
        f=c.objectives.to_array()
        assert self.counter.evaluations==self.counter.decodes==before+1
        assert np.all(np.isfinite(f)) and np.all(f>=0) and fp==fingerprint(c)
        out['F']=f/self.scale
        self.events.append({'evaluation':before+1,'genes_sha256':fp,'raw_objectives':f.tolist()})
        c._schedule=None

def pbi_reference(F,W,ideal,theta):
    Z=np.atleast_2d(F)-ideal
    U=W/np.linalg.norm(W,axis=1)[:,None]
    d1=np.sum(Z*U,axis=1)
    return d1+theta*np.linalg.norm(Z-d1[:,None]*U,axis=1)

class CheckedMOEAD(MOEAD):
    """Read-only audit around the unchanged official replacement function."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.replacement_digest=hashlib.sha256();self.replacement_checks=0
    def _replace(self,k,off):
        before=self.pop.get('F').copy();N=self.neighbors[k];W=self.ref_dirs[N]
        old=self.decomposition.do(before[N],weights=W,ideal_point=self.ideal)
        new=self.decomposition.do(off.F[None,:],weights=W,ideal_point=self.ideal)
        np.testing.assert_allclose(old,pbi_reference(before[N],W,self.ideal,self.decomposition.theta),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(new,pbi_reference(off.F,W,self.ideal,self.decomposition.theta),rtol=1e-12,atol=1e-12)
        replace=N[np.flatnonzero(new<old)]
        expected=before.copy();expected[replace]=off.F
        super()._replace(k,off)
        assert np.array_equal(self.pop.get('F'),expected)
        self.replacement_digest.update(canonical([int(k),self.ideal.tolist(),replace.tolist()]))
        self.replacement_checks+=1

def construct(arm,initial,scale,variation,config,checked=True):
    X=np.empty((len(initial),1),dtype=object);X[:,0]=initial
    pop=Population.new('X',X,'F',objective_array(initial)/scale)
    for p in pop:
        p.set('G',np.empty(0));p.set('H',np.empty(0));p.evaluated.update(('F','G','H'))
    kw=dict(sampling=pop,crossover=variation,mutation=NoMutation())
    if arm=='nsga2_shared':
        return NSGA2(pop_size=100,n_offsprings=100,eliminate_duplicates=False,**kw)
    cls=CheckedMOEAD if checked else MOEAD
    return cls(ref_dirs=np.asarray(config['reference_directions'],float),
               n_neighbors=config['moead']['n_neighbors'],
               prob_neighbor_mating=config['moead']['prob_neighbor_mating'],
               decomposition=PBI(theta=config['moead']['pbi_theta']),**kw)

def snapshot(front,target,reference,wide):
    obj=objective_array(front)
    return {'evaluations':target,'genes':[genes(c) for c in front],'objectives':obj.tolist(),
            'hv':fixed_reference_hv(obj,reference),'hv_wide':fixed_reference_hv(obj,wide),
            'outside_reference_points':int(np.any(obj>reference+1e-12,axis=1).sum())}

def solve(inst,dataset,name,seed,arm,evaluations=10000,endpoints=(2000,5000,10000),
          checked=True,progress_fn=None):
    assert arm in ARMS and evaluations>=100
    assert tuple(sorted(set(endpoints)))==tuple(endpoints) and endpoints[-1]==evaluations
    assert all(100<=x<=evaluations for x in endpoints)
    assert all(x%100==0 for x in endpoints[:-1])
    config=json.loads((HERE/'CONFIG.json').read_text())
    start=time.monotonic();cpu=time.process_time()
    random.seed(seed);np.random.seed(seed)
    with CountCalls() as count:
        initial=initialize_population(inst,100,np.random.RandomState(legacy_initialization_seed(seed)))
        initial_genes=[genes(c) for c in initial];initial_f=objective_array(initial)
        assert count.evaluations==count.decodes==100
        for c in initial:legal(c)
        reference=initial_f.max(axis=0)*1.1;wide=initial_f.max(axis=0)*1.5
        scale=np.maximum(initial_f.max(axis=0),1.)
        archive=update_archive([],initial)
        variation=JointVariation(inst);problem=ScheduleProblem(scale,count)
        algorithm=construct(arm,initial,scale,variation,config,checked=checked)
        algorithm.setup(problem,termination=('n_eval',evaluations),seed=seed,verbose=False)
        # Initial schedules were already decoded exactly once; register that paid cost.
        algorithm.evaluator.n_eval=100
        algorithm.next()
        assert len(algorithm.pop)==100
        assert algorithm.evaluator.n_eval==count.evaluations==count.decodes==100
        snaps=[snapshot(archive,100,reference,wide)] if 100 in endpoints else []
        pending=[];batches=[];last_report=0.
        while count.evaluations<evaluations:
            before=count.evaluations
            if arm=='nsga2_shared':algorithm.n_offsprings=min(100,evaluations-before)
            infill=algorithm.ask()
            assert infill is not None
            pop=Population.create(infill) if isinstance(infill,Individual) else infill
            assert 0<len(pop)<=evaluations-before
            algorithm.evaluator.eval(problem,infill)
            algorithm.tell(infills=infill)
            assert algorithm.evaluator.n_eval==count.evaluations==count.decodes==before+len(pop)
            # Same ten-decoded-child reporting archive cadence as the frozen decomposition.
            for p in pop:
                pending.append(p.X[0])
                if len(pending)==10:
                    archive=update_archive(archive,pending);pending=[]
            if count.evaluations==evaluations and pending:
                archive=update_archive(archive,pending);pending=[]
            batches.append({'evaluations':count.evaluations,'new_evaluations':len(pop),
                            'population_objective_sha256':hashlib.sha256(algorithm.pop.get('F').tobytes()).hexdigest()})
            if count.evaluations in endpoints:snaps.append(snapshot(archive,count.evaluations,reference,wide))
            elapsed=time.monotonic()-start
            if progress_fn and elapsed-last_report>=5:
                progress_fn({'evaluations':count.evaluations,'target':evaluations});last_report=elapsed
        assert [x['evaluations'] for x in snaps]==list(endpoints)
        assert [e['evaluation'] for e in problem.events]==list(range(101,evaluations+1))
        assert [genes(c) for c in initial]==initial_genes
        assert np.array_equal(objective_array(initial),initial_f)
        assert count.evaluations==count.decodes==evaluations
        result={'protocol':PROTOCOL,'dataset':dataset,'name':name,'seed':seed,'arm':arm,
          'evaluations':evaluations,'decodes':count.decodes,'pymoo_n_eval':algorithm.evaluator.n_eval,
          'endpoints':list(endpoints),'initial_genes':initial_genes,'initial_objectives':initial_f.tolist(),
          'reference':reference.tolist(),'wide_reference':wide.tolist(),'internal_scale':scale.tolist(),
          'events':problem.events,'batches':batches,'snapshots':snaps,
          'proposal_count':variation.proposals,'proposal_digest':variation.digest.hexdigest(),
          'pending_unevaluated_proposals':variation.proposals-len(problem.events),
          'replacement_checks':getattr(algorithm,'replacement_checks',None),
          'replacement_digest':algorithm.replacement_digest.hexdigest() if hasattr(algorithm,'replacement_digest') else None,
          'config_sha256':hashlib.sha256((HERE/'CONFIG.json').read_bytes()).hexdigest(),
          'search_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu}
    return result

def scientific_payload(r):
    return {k:v for k,v in r.items() if k not in ('search_seconds','cpu_seconds',
             'replacement_checks','replacement_digest')}
