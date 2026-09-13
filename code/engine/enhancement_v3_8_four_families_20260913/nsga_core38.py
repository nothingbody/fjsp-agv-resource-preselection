"""Matched NSGA-II survival/tournaments; inherited decoder and joint proposal."""
from pool_core38 import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def ranks_crowding(F):
    F=np.asarray(F,float); fronts=NonDominatedSorting().do(F)
    ranks=np.empty(len(F),dtype=int); crowd=np.zeros(len(F),float)
    for rank,idx in enumerate(fronts):
        idx=np.asarray(idx,int); ranks[idx]=rank
        if len(idx)<=2: crowd[idx]=np.inf; continue
        for d in range(F.shape[1]):
            order=np.argsort(F[idx,d],kind='stable'); ordered=idx[order]
            width=F[ordered[-1],d]-F[ordered[0],d]
            # Constant objectives provide no boundary preference.
            if width<=0: continue
            crowd[ordered[[0,-1]]]=np.inf
            crowd[ordered[1:-1]]+=(F[ordered[2:],d]-F[ordered[:-2],d])/width
    return ranks,crowd,fronts

def survive(pop,size=100):
    F=compat.objective_array(pop);r,c,fronts=ranks_crowding(F);selected=[]
    for idx in fronts:
        idx=np.asarray(idx,int)
        if len(selected)+len(idx)<=size:selected.extend(idx.tolist())
        else:
            order=np.argsort(-c[idx],kind='stable')
            selected.extend(idx[order[:size-len(selected)]].tolist());break
    assert len(selected)==min(size,len(pop)) and len(set(selected))==len(selected)
    return [pop[i] for i in selected]

def tournament(rng,ranks,crowd):
    a,b=map(int,rng.randint(len(ranks),size=2))
    if ranks[a]!=ranks[b]:return a if ranks[a]<ranks[b] else b
    if crowd[a]!=crowd[b]:return a if crowd[a]>crowd[b] else b
    return a if rng.randint(2)==0 else b

def solve_nsga(inst,t,budget=10000,ends=(2000,5000,10000),callback=lambda x:None):
    assert t['policy'] in ('physical','random','exact')
    start=time.perf_counter();cpu=time.process_time()
    random.seed(t['seed']);np.random.seed(t['seed'])
    namespace=online.run_key(t['dataset'],t['name'],t['seed'])+'/nsga2'
    with compat.CountCalls() as count:
        population=compat.initialize_population(inst,100,np.random.RandomState(compat.legacy_initialization_seed(t['seed'])))
        initial=compat.objective_array(population);initial_genes=[time_core.genes(c) for c in population]
        scale=np.maximum(initial.max(axis=0),1.);weights=surrogate.directions()
        features=surrogate.EncodingFeatures(inst,scale);reference=initial.max(axis=0)*1.1
        for c in population:c._schedule=None
        archive=compat.update_archive([],population)
        pending=[];offspring=[];events=[];snapshots=[];generation=0;step=0;last=0.
        rank,crowd,_=ranks_crowding(compat.objective_array(population))
        while count.evaluations<budget:
            rng=np.random.RandomState(online.stream_seed(t['seed'],step,namespace+'/proposal'))
            srng=np.random.RandomState(online.stream_seed(t['seed'],step,namespace+'/selection'))
            parent=tournament(rng,rank,crowd);donor=tournament(rng,rank,crowd)
            children=[online.joint_proposal(population[parent],population[donor],rng,features) for _ in range(4)]
            uniform=int(srng.randint(4));direction=step%len(weights);w=weights[direction]
            assert all(c._objectives is None and c._schedule is None for c in children)
            proxy=None;chosen=uniform;explore=t['policy']=='random' or step%4==3
            if t['policy']=='physical' and not explore:
                proxy=np.asarray([features(c)[1] for c in children])
                chosen=int(np.argmin(surrogate.scalar(proxy,w,scale)))
            if t['policy']=='exact':
                values=np.asarray([c.objectives.to_array() for c in children])
                exact_g=surrogate.scalar(values,w,scale);chosen=int(np.argmin(exact_g))
                evaluated=children
            else:
                evaluated=[children[chosen]];values=compat.objective_array(evaluated);exact_g=None
                assert all(c._objectives is None for i,c in enumerate(children) if i!=chosen)
            for c in evaluated:c._schedule=None
            offspring.append(children[chosen]);pending.extend(evaluated)
            while len(pending)>=10:
                archive=compat.update_archive(archive,pending[:10]);pending=pending[10:]
            events.append(dict(step=step,generation=generation,parent=parent,donor=donor,
                direction=direction,random_index=uniform,selected=chosen,policy=t['policy'],
                candidate_fingerprints=[online.fingerprint(c) for c in children],
                predicted_objectives=None if proxy is None else proxy.tolist(),
                evaluated_objectives=values.tolist(),exact_g=None if exact_g is None else exact_g.tolist()))
            step+=1
            assert count.evaluations==count.decodes==100+step*(4 if t['policy']=='exact' else 1)
            if len(offspring)==100 or count.evaluations==budget:
                population=survive(population+offspring);offspring=[];generation+=1
                rank,crowd,_=ranks_crowding(compat.objective_array(population))
            if count.evaluations in ends:
                assert not pending
                F=compat.objective_array(archive)
                snapshots.append(dict(evaluations=count.evaluations,objectives=F.tolist(),
                    genes=[time_core.genes(c) for c in archive],
                    hv=hv(F,reference)*float(np.prod(reference)),
                    hv_wide=hv(F,initial.max(axis=0)*1.5)*float(np.prod(initial.max(axis=0)*1.5))))
            if time.monotonic()-last>5:
                callback(dict(evaluations=count.evaluations,target=budget));last=time.monotonic()
        assert count.evaluations==budget and not pending and not offspring
    return dict(arm='nsga2_'+t['policy'],seed=t['seed'],dataset=t['dataset'],name=t['name'],
        evaluations=budget,decodes=budget,initial_genes=initial_genes,initial_objectives=initial.tolist(),
        reference=reference.tolist(),wide_reference=(initial.max(axis=0)*1.5).tolist(),
        snapshots=snapshots,events=events,steps=step,generations_including_final_partial=generation,
        archive_limit=100,search_seconds_instrumented=time.perf_counter()-start,
        cpu_seconds_instrumented=time.process_time()-cpu,
        unselected_evaluations_charged=3*step if t['policy']=='exact' else 0)
