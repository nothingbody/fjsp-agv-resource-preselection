"""Study32: observed-label KNN regression preselection, an internal adaptation."""
import hashlib,json
import numpy as np
import online_core
from online_core import Decomposition
K=3
WINDOW=512
class ObservedKNN:
    def __init__(self,scale):
        self.scale=np.asarray(scale,float).copy()
        self.x=[];self.y=[];self.ids=[];self.seen=set()
        self.observations=0;self.fit_count=0;self.mean=None;self.std=None
        self.last_prediction=None
    def observe(self,x,y,identity):
        x=np.asarray(x,float);y=np.asarray(y,float)
        assert x.ndim==1 and y.shape==(3,) and np.isfinite(x).all() and np.isfinite(y).all()
        assert str(identity) not in self.seen
        self.seen.add(str(identity))
        self.x.append(x.copy());self.y.append(y.copy());self.ids.append(str(identity))
        self.x,self.y,self.ids=self.x[-WINDOW:],self.y[-WINDOW:],self.ids[-WINDOW:]
        self.observations+=1
    def fit(self):
        # The inherited hook calls this every eight choices. Only the first call
        # fixes distance scaling; later calls do not fit or tune a predictor.
        assert len(self.x)>=K
        if self.mean is None:
            a=np.asarray(self.x);self.mean=a.mean(axis=0)
            s=a.std(axis=0);self.std=np.where(s<1e-6,1.,s)
        self.fit_count+=1
    def predict(self,x,proxies):
        assert self.mean is not None
        x=np.atleast_2d(np.asarray(x,float));p=np.atleast_2d(np.asarray(proxies,float))
        assert p.shape==(len(x),3) and np.isfinite(x).all() and np.isfinite(p).all()
        bank=(np.asarray(self.x)-self.mean)/self.std
        q=(x-self.mean)/self.std
        d=((q[:,None,:]-bank[None,:,:])**2).sum(axis=2)
        ids=np.argsort(d,axis=1,kind='stable')[:,:K]
        pred=np.asarray(self.y)[ids].mean(axis=1)
        # The load range is available exactly without decoding, for both arms.
        pred[:,2]=p[:,2]
        assert np.isfinite(pred).all()
        self.last_prediction=dict(memory_size=len(self.x),observations=self.observations,
            neighbor_ids=[[self.ids[int(j)] for j in row] for row in ids],
            squared_distances=np.take_along_axis(d,ids,axis=1).tolist())
        return pred
    def digest(self):
        h=hashlib.sha256()
        for a in (np.asarray(self.x),np.asarray(self.y),self.mean,self.std):
            h.update(np.asarray(a,dtype=float).tobytes())
        h.update(json.dumps(self.ids,separators=(',',':')).encode())
        return h.hexdigest()
class KNNDecomposition(Decomposition):
    def __init__(self,population,seed,arm,namespace='engineering'):
        assert arm=='de_proxy'
        previous=online_core.ObservedRidge
        try:
            online_core.ObservedRidge=ObservedKNN
            super().__init__(population,seed,'de_model',namespace)
        finally:
            online_core.ObservedRidge=previous
        assert isinstance(self.model,ObservedKNN)
    def advance(self):
        # Scientific advance is inherited verbatim; add diagnostics afterwards.
        child,row=super().advance()
        row['knn_diagnostics']=None if row['explore'] else self.model.last_prediction
        return child,row
def assert_matched_source():
    assert KNNDecomposition.__bases__==(Decomposition,)
    assert set(KNNDecomposition.__dict__)-{'__module__','__doc__'}=={'__init__','advance'}
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(KNNDecomposition.advance)))
    assert len(tree.body[0].body)==3
    assert K==3 and WINDOW==512
    return True
