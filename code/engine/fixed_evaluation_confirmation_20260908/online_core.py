"""Complete, deadline-accounted decomposition search with no extra training calls."""
from __future__ import annotations
import collections
import hashlib
import random
import time
import numpy as np
from compat import (CountCalls,initialize_population,legacy_initialization_seed,update_archive,
    objective_array,fixed_reference_hv)
from time_core import Deadline,prepare_ucb,ucb_step,genes,INSTANCES
from lns_loop import fingerprint
from surrogate import EncodingFeatures,ObservedRidge,joint_proposal,scalar,directions

PROTOCOL='online_reuse_fullrun_v1_20260908'
ARMS=('ucb','de_plain','de_proxy','de_model')
SEEDS=(9081901,9081902)
BATCH=10

def run_key(dataset,name,seed):
    return f'{dataset}/{name}/seed{seed}/{PROTOCOL}'

def stream_seed(seed,step,salt):
    return int.from_bytes(hashlib.sha256(f'{PROTOCOL}/{seed}/{step}/{salt}'.encode()).digest()[:4],'little')

def hv_snapshot(archive,reference,wide,deadline,stamp,label):
    objs=objective_array(archive)
    return dict(deadline=deadline,commit_seconds=stamp,commit_label=label,objectives=objs.tolist(),
        genes=[genes(c) for c in archive],hv=fixed_reference_hv(objs,reference),
        hv_wide=fixed_reference_hv(objs,wide),outside_reference_points=int(np.any(objs>reference+1e-12,axis=1).sum()))


class Decomposition:
    def __init__(self,population,seed,arm,namespace='engineering'):
        assert arm in ARMS[1:]
        self.seed,self.arm,self.step=seed,arm,0
        self.namespace=namespace
        values=objective_array(population)
        self.scale=np.maximum(values.max(axis=0),1.)
        self.weights=directions()
        self.features=EncodingFeatures(population[0].instance,self.scale)
        ids=[int(np.argmin(scalar(values,w,self.scale))) for w in self.weights]
        self.representatives=[population[i] for i in ids]
        self.values=values[ids].copy()
        distances=((self.weights[:,None]-self.weights[None,:])**2).sum(axis=2)
        self.neighbors=np.argsort(distances,axis=1,kind='stable')[:,:5]
        self.model=None
        self.times=collections.defaultdict(float)
        if arm=='de_model':
            self.model=ObservedRidge(self.scale)
            t=time.perf_counter()
            for i,(c,y) in enumerate(zip(population,values)):
                x,_=self.features(c);self.model.observe(x,y,f'initial_{i}')
            self.times['initial_features_observe']=time.perf_counter()-t
            t=time.perf_counter();self.model.fit();self.times['model_fit']+=time.perf_counter()-t
        self.initial_model_digest=self.model.digest() if self.model else None
        # Decomposition variation and selection need genes/objectives, not event maps.
        # Drop only this new solver's unused schedule caches; never alter UCB caches.
        for c in population:c._schedule=None

    def advance(self):
        k=self.step;i=k%len(self.weights);weight=self.weights[i]
        t=time.perf_counter()
        rng=np.random.RandomState(stream_seed(self.seed,k,self.namespace+'/proposal'))
        select_rng=np.random.RandomState(stream_seed(self.seed,k,self.namespace+'/selection'))
        donor_index=int(rng.randint(len(self.representatives)))
        random_index=int(select_rng.randint(4))
        parent,donor=self.representatives[i],self.representatives[donor_index]
        size=1 if self.arm=='de_plain' else 4
        children=[joint_proposal(parent,donor,rng,self.features) for _ in range(size)]
        self.times['proposal']+=time.perf_counter()-t
        explore=k%4==3 and size==4
        selected=0 if size==1 else random_index
        preds=None;feature_hash=None;xs=proxies=None
        t=time.perf_counter()
        if size==4 and not explore:
            xp=[self.features(c) for c in children]
            xs=np.asarray([a for a,b in xp]);proxies=np.asarray([b for a,b in xp])
            feature_hash=hashlib.sha256(xs.tobytes()+proxies.tobytes()).hexdigest()
        elif self.model:
            xs,proxies=self.features(children[selected])
            feature_hash=hashlib.sha256(xs.tobytes()+proxies.tobytes()).hexdigest()
        self.times['features']+=time.perf_counter()-t
        t=time.perf_counter()
        if size==4 and not explore:
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


def solve(inst,dataset,name,seed,arm,budget=120.,checkpoints=(30.,60.,120.),progress_fn=None,max_batches=None):
    assert arm in ARMS and budget>0 and all(t<=budget for t in checkpoints)
    cpu=time.process_time();gate=Deadline(budget)
    random.seed(seed);np.random.seed(seed)
    batches=[];events=[];last_progress=0.;ds=None;archive_seconds=0.
    with CountCalls() as counts:
        if arm=='ucb':
            state=prepare_ucb(inst,seed,run_key(dataset,name,seed),True)
            population,archive=state['population'],state['archive']
        else:
            population=initialize_population(inst,100,np.random.RandomState(legacy_initialization_seed(seed)))
            archive=update_archive([],population)
            ds=Decomposition(population,seed,arm,run_key(dataset,name,seed))
        initial=objective_array(population)
        initial_genes=[genes(c) for c in population]
        reference=initial.max(axis=0)*1.1;wide=initial.max(axis=0)*1.5
        assert counts.evaluations==counts.decodes==100
        assert gate.commit(archive,'initialization')
        initialization_seconds=gate.commits[-1][0]
        while gate.remaining()>.01 and (max_batches is None or len(batches)<max_batches):
            begin=time.perf_counter();batch=len(batches)
            if arm=='ucb':
                result=ucb_step(state);proposed=result.archive
                details=dict(generation=state['generation'],actions=[int(d['action']) for d in result.decisions],
                    rewards=[float(d['credit']['candidate_composite_reward']) for d in result.decisions],
                    reasons=[d['mandatory_reason'] for d in result.decisions],event_start=0,event_end=0)
            else:
                start_event=len(events);children=[]
                for _ in range(BATCH):
                    if gate.remaining()<=.01:break
                    child,event=ds.advance();children.append(child);events.append(event)
                t=time.perf_counter();proposed=update_archive(archive,children);archive_seconds+=time.perf_counter()-t
                details=dict(event_start=start_event,event_end=len(events))
            accepted=gate.commit(proposed,f'batch_{batch}')
            details.update(batch=batch,accepted_before_deadline=accepted,elapsed=gate.elapsed(),
                step_seconds=time.perf_counter()-begin,evaluations=counts.evaluations)
            batches.append(details)
            if not accepted:break
            archive=proposed
            if progress_fn and gate.elapsed()-last_progress>=5.:
                progress_fn(dict(elapsed=gate.elapsed(),budget=budget,batches=len(batches),events=len(events),
                    evaluations=counts.evaluations,model_observations=ds.model.observations if ds and ds.model else 0))
                last_progress=gate.elapsed()
        search_seconds=gate.elapsed();cpu_seconds=time.process_time()-cpu
        expected=100+(100*len(batches) if arm=='ucb' else len(events))
        assert counts.evaluations==counts.decodes==expected
        evaluations,decodes=counts.evaluations,counts.decodes
    snapshots=[]
    for at in checkpoints:
        stamp,front,label=gate.snapshot(at)
        snapshots.append(hv_snapshot(front,reference,wide,at,stamp,label))
    return dict(protocol=PROTOCOL,dataset=dataset,name=name,seed=seed,arm=arm,budget=budget,
        initialization_seconds=initialization_seconds,search_seconds=search_seconds,cpu_seconds=cpu_seconds,
        evaluations=evaluations,decodes=decodes,reference=reference.tolist(),wide_reference=wide.tolist(),
        initial_objectives=initial.tolist(),initial_genes=initial_genes,batches=batches,events=events,
        snapshots=snapshots,commit_log=[dict(seconds=t,label=label,size=len(front)) for t,front,label in gate.commits],
        ucb_actions=int(state['selector'].total_counts.sum()) if arm=='ucb' else 0,
        model_observations=ds.model.observations if ds and ds.model else 0,
        model_fit_count=ds.model.fit_count if ds and ds.model else 0,
        model_final_digest=ds.model.digest() if ds and ds.model else None,
        initial_model_digest=ds.initial_model_digest if ds else None,
        dedicated_training_evaluations=0,unselected_evaluations=0,
        component_seconds=dict(ds.times,archive=archive_seconds) if ds else {},
        discarded_late_batches=sum(not b['accepted_before_deadline'] for b in batches),
        serialization_seconds=gate.elapsed()-search_seconds)
