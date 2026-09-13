"""Explicitly authored unified-model adaptation, not published-solver reproduction."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,random,time
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'standard_baseline_engineering_20260908'))
sys.path.insert(0,str(BASE/'fixed_evaluation_confirmation_20260908'))
from compat import (Chromosome,CountCalls,initialize_population,legacy_initialization_seed,
                    objective_array,update_archive,fixed_reference_hv,instance)
from time_core import genes
from standard_adapters import GENES,legal,fingerprint,canonical
from validation import feasible,hv_independent,chromosome
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.algorithms.moo.nsga2 import binary_tournament
from src.problem.instance import FJSPAGVInstance
CONFIG=json.loads((ROOT/'CONFIG.json').read_text())
PROBLEM=Problem(n_var=1,n_obj=3,n_ieq_constr=0)

def initialize_adapter(inst,size,rng):
    """Preserve frozen initialization for real instances; guard singleton OS swaps."""
    if inst.total_operations>=2:return initialize_population(inst,size,rng)
    if inst.total_operations!=1:raise ValueError('At least one operation is required')
    from compat import (population_initialization_quotas,spt_chromosome,energy_chromosome,
                        balance_chromosome,random_chromosome)
    ns,ne,nb,nr=population_initialization_quotas(size)
    pop=[spt_chromosome(inst,rng) for _ in range((ns+1)//2)]
    for _ in range(ns-(ns+1)//2):
        c=spt_chromosome(inst,rng)
        rng.randint(1,5)  # retain the declared perturbation-count draw; singleton swap is a no-op
        c.invalidate();pop.append(c)
    for factory,n in [(energy_chromosome,ne),(balance_chromosome,nb),(random_chromosome,nr)]:
        pop.extend(factory(inst,rng) for _ in range(n))
    assert len(pop)==size
    for c in pop:c.objectives
    return pop

def fixture(kind):
    counts=[1] if kind=='single' else [1,2,3]
    m=2 if kind=='single' else 3
    inst=FJSPAGVInstance(num_jobs=len(counts),num_machines=m,
        num_agv=1 if kind=='single' else 3,num_operations=counts)
    for j,n in enumerate(counts):
        for o in range(n):
            choices=list(range(m)) if (j+o)%2==0 else [0,m-1]
            inst.compatible_machines[j,o]=choices
            for k in choices:inst.processing_times[j,o,k]=float(1+((j+1)*3+o*2+k*4)%11)
    inst.generate_agv_params(seed=2300 if kind=='single' else 2301)
    return inst

def get_instance(cell):
    return fixture(cell['kind']) if cell['dataset']=='synthetic' else instance(cell['dataset'],cell['name'])

def positions(c):
    """Flat operation indices in OS occurrence order; no decoding."""
    seen=np.zeros(c.instance.num_jobs,dtype=int)
    offsets=np.r_[0,np.cumsum(c.instance.num_operations)[:-1]]
    out=[]
    for j in c.os:
        out.append(int(offsets[j]+seen[j]));seen[j]+=1
    return out

def different(old,bound,rng):
    if bound<=1:return int(old)
    x=int(rng.randint(bound-1))
    return x+(x>=old)

def count_subset(n,rng):
    return int(rng.randint(1,max(1,n//2)+1))

def bounds(inst):
    return np.array([len(inst.get_compatible_machines(j,o))
        for j,n in enumerate(inst.num_operations) for o in range(n)],dtype=int)

def variation(a,b,rng):
    c=a.copy();n=c.n_total;i=c.instance
    if rng.random_sample()<CONFIG['crossover']:
        selected=rng.random_sample(i.num_jobs)<.5
        mask=selected[a.os]
        c.os[~mask]=b.os[~selected[b.os]]
        for key in GENES[1:]:
            lo,hi=sorted(rng.choice(n+1,2,replace=False))
            getattr(c,key)[lo:hi]=getattr(b,key)[lo:hi]
    for key in GENES:
        if rng.random_sample()>=CONFIG['mutation']:continue
        arr=getattr(c,key)
        if key=='os':
            if n>=2:
                u,v=rng.choice(n,2,replace=False);arr[u],arr[v]=arr[v],arr[u]
        else:
            limits=bounds(i) if key=='ma' else np.full(n,i.num_agv if key=='agv_assign' else i.num_speeds)
            for k in rng.choice(n,count_subset(n,rng),replace=False):
                arr[k]=different(arr[k],int(limits[k]),rng)
    assert legal(c) and c._objectives is None and c._schedule is None
    return c

def neighborhood(a,action,rng):
    assert 0<=action<8
    c=a.copy();n=c.n_total;i=c.instance
    otype=action//4;mtype=(action//2)%2;atype=action%2
    if otype==0:
        if n>=2:
            k=int(rng.randint(n-1));pair=c.os[k:k+2][::-1]
            rest=np.r_[c.os[:k],c.os[k+2:]];where=int(rng.randint(len(rest)+1))
            c.os=np.r_[rest[:where],pair,rest[where:]].astype(int)
    else:
        selected=np.sort(rng.choice(n,count_subset(n,rng),replace=False))
        taken=c.os[selected].copy();remaining=np.delete(c.os,selected)
        insert=np.sort(rng.choice(n,len(selected),replace=False))
        mask=np.zeros(n,dtype=bool);mask[insert]=True
        arr=np.empty(n,dtype=int);arr[mask]=taken;arr[~mask]=remaining;c.os=arr
    flat=int(rng.randint(n));jo=[(j,o) for j,k in enumerate(i.num_operations) for o in range(k)]
    j,o=jo[flat];machines=i.get_compatible_machines(j,o)
    values=[i.get_processing_time(j,o,k)*
       (1. if mtype==0 else (float(i.machine_proc_power[k]) if i.machine_proc_power is not None else 5.))
       for k in machines]
    c.ma[flat]=int(np.argmin(values))
    if atype==0:
        if n>=2:
            k=int(rng.randint(n-1));u,v=positions(c)[k:k+2]
            c.agv_assign[u],c.agv_assign[v]=c.agv_assign[v],c.agv_assign[u]
    else:
        k=int(rng.randint(n));c.agv_assign[k]=different(c.agv_assign[k],i.num_agv,rng)
    assert legal(c) and c._objectives is None and c._schedule is None
    return c

def reverse(a,rng):
    c=a.copy();n=c.n_total;i=c.instance
    selected=rng.choice(n,count_subset(n,rng),replace=False)
    c.os[selected]=i.num_jobs-1-c.os[selected]
    target=np.asarray(i.num_operations,dtype=int)
    counts=np.bincount(c.os,minlength=i.num_jobs)
    deficits=np.repeat(np.arange(i.num_jobs),np.maximum(target-counts,0));rng.shuffle(deficits)
    pos=0
    for k in rng.permutation(n):
        old=int(c.os[k])
        if counts[old]>target[old]:
            new=int(deficits[pos]);pos+=1;c.os[k]=new;counts[old]-=1;counts[new]+=1
    assert pos==len(deficits) and np.array_equal(counts,target) and legal(c)
    return c

def dominates(a,b):
    return bool(np.all(np.asarray(a)<=b) and np.any(np.asarray(a)<b))

def ranked(items,n,scale):
    X=np.empty((len(items),1),dtype=object);X[:,0]=items
    p=Population.new('X',X,'F',objective_array(items)/scale)
    for ind in p:ind.set('G',np.empty(0));ind.set('H',np.empty(0))
    return RankAndCrowding().do(PROBLEM,p,n_survive=n)

def order(p,worst=False):
    if worst:return sorted(range(len(p)),key=lambda k:(-int(p[k].get('rank')),float(p[k].get('crowding')),k))
    return sorted(range(len(p)),key=lambda k:(int(p[k].get('rank')),-float(p[k].get('crowding')),k))

def tournament(p,rng):
    pairs=rng.randint(0,len(p),size=(2,2))
    selected=binary_tournament(p,pairs,SimpleNamespace(tournament_type='comp_by_rank_and_crowding')).ravel()
    return [p[k].X[0] for k in selected]

def view(archive,pending,target,reference,wide):
    front=update_archive(archive,pending) if pending else archive
    f=objective_array(front)
    return dict(evaluations=target,genes=[genes(c) for c in front],objectives=f.tolist(),
       hv=fixed_reference_hv(f,reference),hv_wide=fixed_reference_hv(f,wide))

def solve(inst,dataset,name,seed,evaluations,endpoints=None,progress=None):
    assert isinstance(evaluations,int) and evaluations>=100
    endpoints=tuple(endpoints or (evaluations,))
    assert tuple(sorted(set(endpoints)))==endpoints and endpoints[-1]==evaluations
    assert all(100<=x<=evaluations for x in endpoints)
    random.seed(seed);np.random.seed(seed);rng=np.random.RandomState(seed^CONFIG['variation_seed_xor'])
    start=time.monotonic();cpu=time.process_time();last=start
    events=[];batches=[];snaps=[]
    with CountCalls() as count:
        initial=initialize_adapter(inst,100,np.random.RandomState(legacy_initialization_seed(seed)))
        initial_genes=[genes(c) for c in initial];f=objective_array(initial)
        assert count.evaluations==count.decodes==100
        for k,c in enumerate(initial):legal(c);c._eid=k+1
        initial_f=f.tolist();scale=np.maximum(f.max(axis=0),1.)
        reference=f.max(axis=0)*1.1;wide=f.max(axis=0)*1.5
        archive=update_archive([],initial);pending=[]
        pop=initial;generation=0
        if 100 in endpoints:snaps.append(view(archive,pending,100,reference,wide))
        def evaluate(c,phase,parents,**meta):
            nonlocal archive,pending,last
            assert count.evaluations<evaluations and legal(c)
            assert c._objectives is None and c._schedule is None
            old=count.evaluations;fp=fingerprint(c);values=c.objectives.to_array()
            assert count.evaluations==count.decodes==old+1
            assert fingerprint(c)==fp and np.all(np.isfinite(values)) and np.all(values>=0)
            c._eid=old+1
            event=dict(evaluation=c._eid,generation=generation,phase=phase,
                parents=[int(x._eid) for x in parents],genes=genes(c),objectives=values.tolist(),**meta)
            events.append(event);pending.append(c)
            if len(pending)==10:archive=update_archive(archive,pending);pending=[]
            if c._eid in endpoints:snaps.append(view(archive,pending,c._eid,reference,wide))
            if progress and time.monotonic()-last>=5:
                progress(dict(evaluations=c._eid,target=evaluations,phase=phase,generation=generation))
                last=time.monotonic()
            return event
        while count.evaluations<evaluations:
            generation+=1;before=count.evaluations
            parents=ranked(pop,100,scale);offspring=[];local=[];opposed=[]
            for _ in range(min(100,evaluations-count.evaluations)):
                a,b=tournament(parents,rng);c=variation(a,b,rng)
                evaluate(c,'global',[a,b]);offspring.append(c)
            if count.evaluations<evaluations:
                p=ranked(offspring,len(offspring),scale)
                seeds=[p[k].X[0] for k in order(p)[:CONFIG['local_seeds']]]
                for anchor in seeds:
                    for action in range(8):
                        if count.evaluations==evaluations:break
                        c=neighborhood(anchor,action,rng)
                        e=evaluate(c,'local',[anchor],action=action)
                        accepted=dominates(c.objectives.to_array(),anchor.objectives.to_array())
                        e['accepted']=accepted;local.append(c)
                        if accepted:anchor=c
                    if count.evaluations==evaluations:break
            pool=pop+offspring+local
            nreverse=0
            if count.evaluations<evaluations:
                provisional=ranked(pool,100,scale)
                fraction=np.clip((count.evaluations-100)/(evaluations-100),0.,1.)
                nreverse=int(np.clip(np.floor(100*(.7-.5*fraction)),20,70))
                for k in order(provisional,worst=True)[:nreverse]:
                    if count.evaluations==evaluations:break
                    a=provisional[k].X[0];c=reverse(a,rng)
                    evaluate(c,'reverse',[a]);opposed.append(c)
            survivors=ranked(pool+opposed,100,scale)
            pop=[p.X[0] for p in survivors]
            batches.append(dict(generation=generation,before=before,after=count.evaluations,
                global_trials=len(offspring),local_trials=len(local),reverse_planned=nreverse,
                reverse_trials=len(opposed),survivor_ids=[c._eid for c in pop]))
        assert count.evaluations==count.decodes==evaluations
        assert [s['evaluations'] for s in snaps]==list(endpoints)
        return dict(protocol=CONFIG['protocol'],arm=CONFIG['arm'],dataset=dataset,name=name,seed=seed,
            evaluations=evaluations,decodes=count.decodes,initial_genes=initial_genes,
            initial_objectives=initial_f,scale=scale.tolist(),reference=reference.tolist(),
            wide_reference=wide.tolist(),endpoints=list(endpoints),events=events,batches=batches,
            snapshots=snaps,config_sha256=hashlib.sha256(canonical(CONFIG)).hexdigest(),
            search_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu)

def scientific(r):
    return {k:v for k,v in r.items() if k not in ('search_seconds','cpu_seconds')}

def verify_pair(a,b,inst):
    assert scientific(a)==scientific(b),'Complete scientific replay mismatch'
    n=a['evaluations'];assert a['decodes']==n and len(a['events'])==n-100
    assert [e['evaluation'] for e in a['events']]==list(range(101,n+1))
    assert sum(x['global_trials']+x['local_trials']+x['reverse_trials'] for x in a['batches'])==n-100
    all_genes=a['initial_genes']+[e['genes'] for e in a['events']]
    all_values=a['initial_objectives']+[e['objectives'] for e in a['events']]
    assert len(all_genes)==len(all_values)==n
    operations=0
    with CountCalls() as counter:
        for g,f in zip(all_genes,all_values):
            c=chromosome(inst,g);operations+=feasible(c)
            np.testing.assert_array_equal(c.objectives.to_array(),f)
        assert counter.evaluations==counter.decodes==n
    pairs={canonical(g):f for g,f in zip(all_genes,all_values)}
    for snap in a['snapshots']:
        valid={canonical(g):f for g,f in zip(all_genes[:snap['evaluations']],all_values[:snap['evaluations']])}
        for g,f in zip(snap['genes'],snap['objectives']):assert valid[canonical(g)]==f
        F=np.asarray(snap['objectives'])
        assert len(np.unique(F,axis=0))==len(F)
        for x in F:assert not any(dominates(y,x) for y in F)
        for field,ref in [('hv','reference'),('hv_wide','wide_reference')]:
            np.testing.assert_allclose(snap[field],hv_independent(F,a[ref]),rtol=1e-12,atol=1e-8)
    for e in a['events']:
        assert all(0<p<e['evaluation'] for p in e['parents'])
        if e['phase']=='local':
            assert e['accepted']==dominates(e['objectives'],all_values[e['parents'][0]-1])
    return dict(status='VERIFIED',exact_replay=True,all_candidate_schedules_checked=n,
        operation_assignments_checked=operations,validation_evaluations=n,
        hv_checks=2*len(a['snapshots']),scientific_sha256=hashlib.sha256(canonical(scientific(a))).hexdigest())
