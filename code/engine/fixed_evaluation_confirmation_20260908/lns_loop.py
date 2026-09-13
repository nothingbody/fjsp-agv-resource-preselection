"""Independent archive-based CP-LNS plumbing for the bounded engineering smoke."""
import hashlib
import time
import numpy as np
from cp_core import Kernel,flat_order
from src.algorithm.nsga3.encoding import (spt_chromosome,energy_chromosome,
    balance_chromosome,random_chromosome)
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def fingerprint(chrom):
    h=hashlib.sha256()
    for k in ('os','ma','agv_assign','agv_speed'):
        h.update(np.asarray(getattr(chrom,k),dtype='<i8').tobytes())
    return h.hexdigest()


def legacy_initialization_seed(seed):
    # Exact frozen helper; isolated from package __init__ training imports.
    child=np.random.SeedSequence(int(seed)).spawn(3)[0]
    return int(child.generate_state(1,dtype=np.uint32)[0])


def initialize(instance,seed):
    """Same 100-population 30/20/20/30 construction as frozen baseline helper.

    Legacy initialization SeedSequence derivation is applied by the caller.
    This isolated implementation avoids importing irrelevant training modules.
    """
    rng=np.random.RandomState(seed)
    pop=[spt_chromosome(instance,rng) for _ in range(15)]
    for _ in range(15):
        c=spt_chromosome(instance,rng)
        for _ in range(rng.randint(1,5)):
            i,j=rng.choice(len(c.os),2,replace=False)
            c.os[i],c.os[j]=c.os[j],c.os[i]
        c.invalidate()
        pop.append(c)
    pop += [energy_chromosome(instance,rng) for _ in range(20)]
    pop += [balance_chromosome(instance,rng) for _ in range(20)]
    pop += [random_chromosome(instance,rng) for _ in range(30)]
    for c in pop:
        _=c.objectives
    return pop


def rows(archive):
    return np.asarray([c.objectives.to_array() for c in archive])


def update_archive(archive,candidates,capacity=100):
    # Exact vector deduplication; deterministic stable crowding truncation.
    pool=[]
    seen=set()
    for c in list(archive)+list(candidates):
        key=tuple(float(x) for x in c.objectives.to_array())
        if key not in seen:
            seen.add(key)
            pool.append(c)
    if not pool:
        return []
    f=NonDominatedSorting().do(rows(pool),only_non_dominated_front=True)
    pool=[pool[int(k)] for k in f]
    if len(pool)<=capacity:
        return pool
    values=rows(pool)
    crowd=np.zeros(len(pool))
    for j in range(3):
        ids=np.argsort(values[:,j],kind='stable')
        span=values[ids[-1],j]-values[ids[0],j]
        if span<=1e-12:
            continue
        crowd[ids[0]]=crowd[ids[-1]]=np.inf
        crowd[ids[1:-1]]+=(values[ids[2:],j]-values[ids[:-2],j])/span
    return [pool[int(k)] for k in np.argsort(-crowd,kind='stable')[:capacity]]


def structural_cluster(chrom,size,rng):
    """BFS in validated job/machine/AGV adjacency, seeded at zero event slack."""
    keys,order=flat_order(chrom)
    inst,s=chrom.instance,chrom.schedule
    node_times=[0.]
    outgoing=[[]]
    events={}
    graph={f:set() for f in order}
    lastm,lasta={},{}
    byop={o:f for f,o in enumerate(keys)}
    def node(edges,actual):
        n=len(node_times)
        t=max(node_times[p]+w for p,w in edges)
        assert abs(t-actual)<1e-7,'Structural graph does not match decoder'
        node_times.append(t)
        outgoing.append([])
        for p,w in edges:
            outgoing[p].append((n,w))
        return n
    for f in order:
        job,op=keys[f]
        u,av=s.op_machine[keys[f]],s.transport_agv[keys[f]]
        speed=s.transport_speed[keys[f]]
        jp=None if op==0 else byop[job,op-1]
        mp,ap=lastm.get(u),lasta.get(av)
        for pred in [jp,mp,ap]:
            if pred is not None:
                graph[f].add(pred)
                graph[pred].add(f)
        jobend=0 if jp is None else events[jp][3]
        agvend=0 if ap is None else events[ap][1]
        mend=0 if mp is None else events[mp][3]
        setup=0 if mp is None or keys[mp][0]==job else float(inst.machine_setup_time[u])
        pickup=node([(jobend,0),(agvend,s.transport_empty_distance[keys[f]]/speed)],s.transport_start[keys[f]])
        arrival=node([(pickup,s.transport_loaded_distance[keys[f]]/speed)],s.transport_end[keys[f]])
        start=node([(arrival,0),(mend,setup)],s.op_start[keys[f]])
        end=node([(start,inst.get_processing_time(job,op,u))],s.op_end[keys[f]])
        events[f]=[pickup,arrival,start,end]
        lastm[u],lasta[av]=f,f
    sink=node([(events[f][3],0) for f in order],max(s.op_end.values()))
    latest=np.full(len(node_times),node_times[sink])
    for n in reversed(range(sink)):
        latest[n]=min(latest[q]-w for q,w in outgoing[n])
    slack={f:min(latest[n]-node_times[n] for n in events[f]) for f in order}
    assert min(slack.values())>-1e-7
    critical=[f for f in order if slack[f]<=1e-7]
    seed=critical[int(rng.randint(len(critical)))]
    chosen=[seed]
    while len(chosen)<min(size,len(order)):
        frontier=set().union(*(graph[f] for f in chosen))-set(chosen)
        if not frontier:
            frontier=set(order)-set(chosen)
        chosen.append(min(frontier,key=lambda f:(slack[f],f)))
    return sorted(chosen)


def run_smoke(instance,population,kind,size,seed,initialization_seconds):
    rng=np.random.RandomState(seed)
    start,cpu=time.monotonic(),time.process_time()
    archive=update_archive([],population)
    initial=rows(population)
    reference=np.maximum(initial.max(axis=0)*1.1,1e-9)
    hv=HV(ref_point=reference)
    history=[]
    for objective in range(3):
        tick=time.monotonic()
        # Rotate target objective and choose its current extreme; epsilon=1.05
        # on the other objectives. Frozen engineering rule, not tuned by HV.
        values=rows(archive)
        pi=int(np.argmin(values[:,objective]))
        parent=archive[pi]
        cluster=(structural_cluster(parent,size,rng) if kind=='structural' else
                 sorted(int(x) for x in rng.choice(instance.total_operations,min(size,instance.total_operations),replace=False)))
        epsilon=[float(v)*1.05 if k!=objective else None for k,v in enumerate(parent.objectives.to_array())]
        before=float(hv(rows(archive)))
        kernel=Kernel(parent,cluster,objective,epsilon)
        child,log=kernel.solve(seconds=5.,seed=seed+objective)
        log.update(parent_objectives=parent.objectives.to_array().tolist(),parent_fingerprint=fingerprint(parent),
                   parent_genes={k:getattr(parent,k).tolist() for k in ('os','ma','agv_assign','agv_speed')})
        if child is not None:
            archive=update_archive(archive,[child])
        log.update(archive_hv_before=before,archive_hv_after=float(hv(rows(archive))),
            accepted_into_archive=child is not None and any(fingerprint(c)==fingerprint(child) for c in archive),
            call_and_archive_seconds=time.monotonic()-tick)
        history.append(log)
    return dict(kind=kind,cluster_size=size,seed=seed,call_count=len(history),
        reference=reference.tolist(),initial_objectives=initial.tolist(),
        initial_fingerprints=[fingerprint(c) for c in population],
        initial_archive_hv=float(hv(rows(update_archive([],population)))),
        final_archive_objectives=rows(archive).tolist(),final_archive_hv=float(hv(rows(archive))),
        initialization_seconds=initialization_seconds,loop_seconds=time.monotonic()-start,
        charged_seconds=initialization_seconds+time.monotonic()-start,
        loop_cpu_seconds=time.process_time()-cpu,
        decoded_candidates=sum(r['has_candidate'] for r in history),calls=history)
