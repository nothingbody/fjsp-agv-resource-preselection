"""Matched control: only preselection aggregation changes; no inherited file edits."""
import ast,inspect,textwrap,hashlib,time
import numpy as np
from online_core import Decomposition,stream_seed
from surrogate import joint_proposal,scalar
from lns_loop import fingerprint

def weighted_sum(values,weights,scale):
    # Same coordinate normalization and weight floor as the frozen scalar.
    z=np.asarray(values)/np.asarray(scale)*np.maximum(weights,.01)
    return z.sum(axis=-1)

class WeightedSumDecomposition(Decomposition):
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
            selected=int(np.argmin(weighted_sum(preds,weight,self.scale)))
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


def assert_matched_source():
    expected=ast.parse(textwrap.dedent(inspect.getsource(Decomposition.advance)))
    actual=ast.parse(textwrap.dedent(inspect.getsource(WeightedSumDecomposition.advance)))
    matches=0
    for node in ast.walk(expected):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
            and node.func.id=='scalar' and len(node.args)==3
            and isinstance(node.args[0],ast.Name) and node.args[0].id=='preds'):
            node.func.id='weighted_sum';matches+=1
    assert matches==1
    assert ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)
    assert WeightedSumDecomposition.__init__ is Decomposition.__init__
    assert set(WeightedSumDecomposition.__dict__)-{'__module__','__doc__'}=={'advance'}
    return True
