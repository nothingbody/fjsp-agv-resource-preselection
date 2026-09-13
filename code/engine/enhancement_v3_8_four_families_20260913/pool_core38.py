from pathlib import Path
import sys, time, random, hashlib, collections
import numpy as np
ROOT=Path(__file__).resolve().parent; B=ROOT.parent
sys.path.insert(0,str(B/'supplement_v3_2_20260912'))
from supp_common import old,compat,online,surrogate,time_core,hv
from online_core import stream_seed,joint_proposal,fingerprint,scalar
import budget_core
class PoolPhysical(online.Decomposition):
    def __init__(self,population,seed,arm,namespace='engineering',pool_size=2):
        assert pool_size in (2,4)
        self.pool_size=pool_size
        super().__init__(population,seed,arm,namespace)
    def advance(self):
        k=self.step;i=k%len(self.weights);weight=self.weights[i]
        t=time.perf_counter()
        rng=np.random.RandomState(stream_seed(self.seed,k,self.namespace+'/proposal'))
        select_rng=np.random.RandomState(stream_seed(self.seed,k,self.namespace+'/selection'))
        donor_index=int(rng.randint(len(self.representatives)))
        random_index=int(select_rng.randint(self.pool_size))
        parent,donor=self.representatives[i],self.representatives[donor_index]
        size=self.pool_size
        children=[joint_proposal(parent,donor,rng,self.features) for _ in range(size)]
        self.times['proposal']+=time.perf_counter()-t
        explore=k%4==3 and size>=2
        selected=0 if size==1 else random_index
        preds=None;feature_hash=None;xs=proxies=None
        t=time.perf_counter()
        if size>=2 and not explore:
            xp=[self.features(c) for c in children]
            xs=np.asarray([a for a,b in xp]);proxies=np.asarray([b for a,b in xp])
            feature_hash=hashlib.sha256(xs.tobytes()+proxies.tobytes()).hexdigest()
        elif self.model:
            xs,proxies=self.features(children[selected])
            feature_hash=hashlib.sha256(xs.tobytes()+proxies.tobytes()).hexdigest()
        self.times['features']+=time.perf_counter()-t
        t=time.perf_counter()
        if size>=2 and not explore:
            preds=self.model.predict(xs,proxies) if self.model else proxies
            selected=int(np.argmin(scalar(preds,weight,self.scale)))
        self.times['prediction_selection']+=time.perf_counter()-t
        # Freeze all decision information BEFORE accessing a candidate label.
        assert all(c._objectives is None and c._schedule is None for c in children)
        before=self.model.observations if self.model else 0
        digest=self.model.digest() if self.model else None
        row=dict(step=k,direction=i,donor_index=donor_index,brood_size=size,random_index=random_index,
            explore=explore,selected=selected,candidate_fingerprints=[fingerprint(c) for c in children],
            feature_hash=feature_hash,predictions=preds.tolist() if preds is not None else None,
            observations_before=before,model_digest_before=digest)
        t=time.perf_counter();child=children[selected];y=child.objectives.to_array()
        self.times['evaluation']+=time.perf_counter()-t
        child._schedule=None
        assert all(c._objectives is None and c._schedule is None for j,c in enumerate(children) if j!=selected)
        t=time.perf_counter()
        neighbors=self.neighbors[i]
        old=scalar(self.values[neighbors],self.weights[neighbors],self.scale)
        new=scalar(np.broadcast_to(y,(len(neighbors),3)),self.weights[neighbors],self.scale)
        replace=neighbors[np.flatnonzero(new<old)[:2]]
        for j in replace:self.representatives[int(j)]=child;self.values[int(j)]=y
        self.times['representative_update']+=time.perf_counter()-t
        if self.model:
            t=time.perf_counter();x=xs if explore else xs[selected]
            self.model.observe(x,y,f'step_{k}')
            self.times['observe']+=time.perf_counter()-t
            if (k+1)%8==0:
                t=time.perf_counter();self.model.fit();self.times['model_fit']+=time.perf_counter()-t
            assert self.model.observations==101+k
        self.step+=1
        row.update(objectives=y.tolist(),replacements=replace.tolist(),
            observations_after=self.model.observations if self.model else 0,
            fit_count=self.model.fit_count if self.model else 0)
        return child,row


def exact_k(inst,t,evaluations,ends,callback,pool_size=2):
    assert pool_size in (2,4) and (evaluations-100)%pool_size==0
    random.seed(t['seed']);np.random.seed(t['seed']);start=time.perf_counter();cpu=time.process_time()
    with compat.CountCalls() as count:
        pop=compat.initialize_population(inst,100,np.random.RandomState(compat.legacy_initialization_seed(t['seed'])))
        initial=compat.objective_array(pop);initialgenes=[time_core.genes(c) for c in pop]
        ds=online.Decomposition(pop,t['seed'],'de_proxy',online.run_key(t['dataset'],t['name'],t['seed']))
        archive=compat.update_archive([],pop);pending=[];snapshots=[];events=[];last=time.monotonic()
        while count.evaluations<evaluations:
            k=ds.step;i=k%len(ds.weights);w=ds.weights[i]
            rng=np.random.RandomState(online.stream_seed(ds.seed,k,ds.namespace+'/proposal'))
            selection_rng=np.random.RandomState(online.stream_seed(ds.seed,k,ds.namespace+'/selection'))
            donor_index=int(rng.randint(len(ds.representatives)));unused_random=int(selection_rng.randint(pool_size))
            parent,donor=ds.representatives[i],ds.representatives[donor_index]
            children=[online.joint_proposal(parent,donor,rng,ds.features) for _ in range(pool_size)]
            assert all(c._objectives is None and c._schedule is None for c in children)
            values=np.asarray([c.objectives.to_array() for c in children]);g=surrogate.scalar(values,w,ds.scale)
            chosen=int(np.argmin(g));child=children[chosen];y=values[chosen]
            nbr=ds.neighbors[i];oldg=surrogate.scalar(ds.values[nbr],ds.weights[nbr],ds.scale)
            newg=surrogate.scalar(np.broadcast_to(y,(len(nbr),3)),ds.weights[nbr],ds.scale)
            replace=nbr[np.flatnonzero(newg<oldg)[:2]]
            for j in replace:ds.representatives[int(j)]=child;ds.values[int(j)]=y
            for c in children:c._schedule=None
            events.append({'step':k,'direction':i,'donor_index':donor_index,'random_index_unused':unused_random,'selected':chosen,
                'candidate_fingerprints':[online.fingerprint(c) for c in children],'exact_objectives':values.tolist(),'exact_g':g.tolist(),'replacements':replace.tolist()})
            pending.extend(children)
            while len(pending)>=10:
                archive=compat.update_archive(archive,pending[:10]);pending=pending[10:]
            ds.step+=1
            assert count.evaluations==count.decodes==100+pool_size*ds.step
            if count.evaluations in ends:
                assert not pending
                F=compat.objective_array(archive);ref=initial.max(axis=0)*1.1
                snapshots.append({'evaluations':count.evaluations,'objectives':F.tolist(),'genes':[time_core.genes(c) for c in archive],
                    'hv':hv(F,ref)*float(np.prod(ref)),'hv_wide':hv(F,initial.max(axis=0)*1.5)*float(np.prod(initial.max(axis=0)*1.5))})
            if time.monotonic()-last>5:
                callback({'evaluations':count.evaluations,'target':evaluations});last=time.monotonic()
        assert count.evaluations==evaluations and not pending
    return {'arm':f'exact_{pool_size}','seed':t['seed'],'name':t['name'],'dataset':t['dataset'],'evaluations':evaluations,'decodes':evaluations,'initial_genes':initialgenes,
        'initial_objectives':initial.tolist(),'reference':(initial.max(axis=0)*1.1).tolist(),'wide_reference':(initial.max(axis=0)*1.5).tolist(),
        'snapshots':snapshots,'events':events,'search_seconds_instrumented':time.perf_counter()-start,'cpu_seconds_instrumented':time.process_time()-cpu,
        'archive_limit':100,'uniform_override':False,'unselected_evaluations_charged':(pool_size-1)*(evaluations-100)//pool_size}


def solve_pool(inst,t,budget=10000,ends=(2000,5000,10000),callback=lambda x:None):
    k=t['pool_size']
    if t['policy']=='exact':return exact_k(inst,t,budget,ends,callback,k)
    assert t['policy']=='physical'
    previous=budget_core.Decomposition
    try:
        budget_core.Decomposition=lambda *a,**kw:PoolPhysical(*a,**kw,pool_size=k)
        return old.run_solver(inst,t['dataset'],t['name'],t['seed'],'de_proxy',budget,ends,callback)
    finally:budget_core.Decomposition=previous
