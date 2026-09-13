from pathlib import Path
import os,sys,json,gzip,hashlib,time,math,csv
import numpy as np
ROOT=Path(__file__).resolve().parent
B=ROOT.parent
S30=B/'fixed_evaluation_component_ablation_20260909'
PY=B/'fixed_evaluation_confirmation_20260908/.venv/bin/python'
sys.path.insert(0,str(S30))
import common30 as old
import online_core as online, surrogate, compat, time_core, budget_core
from pymoo.indicators.hv import HV
STRATA=old.STRATA
ENDS=(2000,5000,10000)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def read(p):return json.loads(Path(p).read_text())
def rawread(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def digest(v):return hashlib.sha256(old.canonical(v)).hexdigest()
def sha(p):return old.sha(p)
def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def packed(p,v):
    with Path(p).open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=5) as z:z.write(old.canonical(v))
def status(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False));tmp.replace(p)
def chrom(inst,g):return compat.Chromosome(inst,**{k:np.asarray(g[k],dtype=int).copy() for k in ('os','ma','agv_assign','agv_speed')})
def input_for(t):
    p=S30/t['input_file'];assert sha(p)==t['input_sha256']
    return old.s24.restore_instance(read(p))
def metadata(t):return {k:t[k] for k in ('base_id','stratum','regime','seed','name')}
def nd_indices(F):
    """Exact 3D minimization skyline, retaining first unique lexicographic point."""
    F=np.asarray(F,float);assert F.ndim==2 and F.shape[1]==3 and np.isfinite(F).all()
    order=np.lexsort((F[:,2],F[:,1],F[:,0]))
    ys=np.unique(F[:,1]);tree=np.full(len(ys)+1,np.inf);kept=[]
    for j in order:
        y=int(np.searchsorted(ys,F[j,1]))+1;k=y;best=np.inf
        while k:best=min(best,tree[k]);k-=k&-k
        if best<=F[j,2]:continue
        kept.append(int(j));k=y
        while k<len(tree):tree[k]=min(tree[k],F[j,2]);k+=k&-k
    return np.asarray(kept,int)
def nd(F):
    F=np.asarray(F,float);return F[nd_indices(F)]
def hv(F,ref):return float(HV(ref_point=np.ones(3))(np.asarray(F)/np.asarray(ref)))
def igdplus(F,Z):
    F=np.asarray(F,float);Z=np.asarray(Z,float);total=0.
    for z in np.array_split(Z,max(1,math.ceil(len(Z)/256))):
        D=np.maximum(F[None,:,:]-z[:,None,:],0.)
        total+=float(np.sqrt((D*D).sum(axis=2)).min(axis=1).sum())
    return total/len(Z)
def thin(Z,n=100):
    Z=np.asarray(Z);m=min(n,len(Z));ix=[0];dist=((Z-Z[0])**2).sum(axis=1)
    for _ in range(1,m):
        j=int(np.argmax(dist));ix.append(j);dist=np.minimum(dist,((Z-Z[j])**2).sum(axis=1))
    return Z[ix]
def boot(x,seed=91234002):
    x=np.asarray(x);rng=np.random.RandomState(seed)
    vals=np.stack([x[4*i+rng.randint(0,4,(20000,4))].mean(axis=1) for i in range(4)]).mean(axis=0)
    return np.percentile(vals,[2.5,97.5]).tolist()
def signflip(x):
    x=np.asarray(x);assert len(x)==16
    s=1-2*((np.arange(65536,dtype=np.int64)[:,None]>>np.arange(16))&1)
    return float(np.mean(np.abs((s*x).mean(axis=1))>=abs(x.mean())-1e-12*max(1,np.max(np.abs(x)))))
def holm(ps):
    ix=np.argsort(ps,kind='stable');out=np.empty(len(ps));last=0
    for rank,i in enumerate(ix):last=max(last,min(1,ps[i]*(len(ps)-rank)));out[i]=last
    return out.tolist()
def test_math():
    rng=np.random.RandomState(312)
    for n in (1,2,20,100):
        F=rng.randint(0,12,(n,3)).astype(float)
        a=nd(F);u=np.unique(F,axis=0);d=np.any(np.all(u[:,None,:]<=u[None,:,:],2)&np.any(u[:,None,:]<u[None,:,:],2),axis=0)
        assert set(map(tuple,a))==set(map(tuple,u[~d]))
    assert abs(hv([[.2,.3,.4]],[1,1,1])-.8*.7*.6)<1e-12
    assert igdplus([[0,0,0]],[[1,1,1]])==0
    assert abs(igdplus([[1,1,1]],[[0,0,0]])-3**.5)<1e-12
    return True
