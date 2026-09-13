"""No-decoder features, legal joint proposals, and observed-label-only Ridge."""
from __future__ import annotations
import hashlib
import numpy as np
from compat import Chromosome,generate_reference_points

PROTOCOL='surrogate_screen_v1_20260908'
SEEDS=(9081801,9081802)
ANCHORS=(10,30)
TRAIN_NEW=128
POOLS=64
BROOD=4
WINDOW=512
REFIT=8
ALPHA=.01


def scalar(values,weight,scale):
    z=np.asarray(values,float)/np.asarray(scale,float)*np.maximum(weight,.01)
    return np.max(z,axis=-1)+.05*np.sum(z,axis=-1)


class EncodingFeatures:
    def __init__(self,inst,scale):
        self.inst,self.scale=inst,np.asarray(scale,float)
        self.offset=np.r_[0,np.cumsum(inst.num_operations)[:-1]]
        self.jobs=np.repeat(np.arange(inst.num_jobs),inst.num_operations)
        pairs=[(i,j) for i in range(inst.num_jobs) for j in range(inst.num_operations[i])]
        self.choices=np.array([len(inst.get_compatible_machines(*p)) for p in pairs])
        self.machines=np.full((len(pairs),max(self.choices)), -1,dtype=int)
        self.times=np.full(self.machines.shape,np.nan)
        for k,(i,j) in enumerate(pairs):
            for a,m in enumerate(inst.get_compatible_machines(i,j)):
                self.machines[k,a]=m
                self.times[k,a]=inst.get_processing_time(i,j,m)
        self.rows=np.arange(len(pairs))
        self.speeds=np.array(inst.speed_levels)

    def __call__(self,chrom):
        # This function intentionally never accesses schedule or objectives.
        inst=self.inst
        machine=self.machines[self.rows,chrom.ma]
        duration=self.times[self.rows,chrom.ma]
        previous=np.r_[0,machine[:-1]+1]
        previous[self.offset]=0
        distance=inst.distance_matrix[previous,machine+1]
        speed=self.speeds[chrom.agv_speed]
        loaded_time=distance/speed
        a,b,c=inst.agv_load_power_params
        loaded_energy=float(np.sum((a*speed**2+b*speed+c)*loaded_time))
        load=np.bincount(machine,weights=duration,minlength=inst.num_machines)
        job_time=np.bincount(self.jobs,weights=duration+loaded_time,minlength=inst.num_jobs)
        agv_time=np.zeros(inst.num_agv)
        loc=np.zeros(inst.num_agv,dtype=int)
        last_job=np.full(inst.num_machines,-1,dtype=int)
        setup=np.zeros(inst.num_machines)
        count=np.zeros(inst.num_jobs,dtype=int)
        empty_energy=empty_time=0.
        ea,eb,ec=inst.agv_empty_power_params
        for job in chrom.os:
            k=self.offset[job]+count[job]
            count[job]+=1
            m,vehicle,v=machine[k],chrom.agv_assign[k],speed[k]
            te=inst.distance_matrix[loc[vehicle],previous[k]]/v
            empty_time+=te
            empty_energy+=(ea*v*v+eb*v+ec)*te
            agv_time[vehicle]+=te+loaded_time[k]
            loc[vehicle]=m+1
            if last_job[m]>=0 and last_job[m]!=job:
                setup[m]+=inst.machine_setup_time[m]
            last_job[m]=job
        processing_energy=float(load@inst.machine_proc_power)
        setup_energy=float(setup@inst.machine_setup_power)
        energy=processing_energy+setup_energy+loaded_energy+empty_energy
        busy=load+setup
        proxy=np.array([max(busy.max(),agv_time.max(),job_time.max()),energy,np.ptp(load)])
        ts,es,bs=self.scale
        features=list(proxy/self.scale)
        features += [float(v/ts) for v in (load.mean(),load.max(),load.min(),load.std(),
            busy.max(),busy.std(),agv_time.mean(),agv_time.max(),agv_time.min(),agv_time.std(),
            job_time.mean(),job_time.max(),job_time.std(),empty_time,loaded_time.sum(),setup.sum())]
        features += [processing_energy/es,setup_energy/es,loaded_energy/es,empty_energy/es]
        features += [float(np.mean(chrom.os[1:]!=chrom.os[:-1])),float(np.mean(chrom.agv_speed))/(inst.num_speeds-1)]
        features += [float(np.mean(chrom.agv_speed==v)) for v in range(inst.num_speeds)]
        features += [float(np.std(np.bincount(chrom.agv_assign,minlength=inst.num_agv))/len(chrom.os))]
        x=np.asarray(features,float)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(proxy))
        return x,proxy


def joint_proposal(parent,donor,rng,features):
    c=parent.copy()
    inst=c.instance
    if rng.random()<.25 and inst.num_jobs>1:
        subset=rng.choice(inst.num_jobs,rng.randint(1,inst.num_jobs),replace=False)
        keep=np.isin(c.os,subset)
        c.os[~keep]=donor.os[~np.isin(donor.os,subset)]
        mask=rng.random(len(c.ma))<.25
        for family in ('ma','agv_assign','agv_speed'):
            getattr(c,family)[mask]=getattr(donor,family)[mask]
    a,b=map(int,rng.randint(0,len(c.os),size=2))
    token=c.os[a]
    c.os=np.insert(np.delete(c.os,a),b,token)
    selected=rng.choice(len(c.ma),min(2,len(c.ma)),replace=False)
    for k in selected:
        c.ma[k]=rng.randint(features.choices[k])
        c.agv_assign[k]=rng.randint(inst.num_agv)
        c.agv_speed[k]=rng.randint(inst.num_speeds)
    return c


class ObservedRidge:
    def __init__(self,scale):
        self.scale=np.asarray(scale,float)
        self.x=[]
        self.y=[]
        self.ids=[]
        self.observations=0
        self.fit_count=0

    def observe(self,x,y,identity):
        self.x.append(np.asarray(x,float).copy())
        self.y.append(np.asarray(y,float).copy()/self.scale)
        self.ids.append(str(identity))
        self.observations+=1
        self.x,self.y,self.ids=self.x[-WINDOW:],self.y[-WINDOW:],self.ids[-WINDOW:]

    def design(self,x):
        z=np.clip((np.asarray(x)-self.mean)/self.std,-5.,5.)
        return np.column_stack([np.ones(len(z)),z,z*z])

    def fit(self):
        x=np.asarray(self.x)
        self.mean=x.mean(axis=0)
        self.std=np.maximum(x.std(axis=0),1e-6)
        design=self.design(x)
        ridge=np.eye(design.shape[1])*ALPHA
        ridge[0,0]=0.
        self.coef=np.linalg.solve(design.T@design+ridge,design.T@np.asarray(self.y)[:,:2])
        assert np.all(np.isfinite(self.coef))
        self.fit_count+=1

    def predict(self,x,proxies):
        pred=self.design(np.atleast_2d(x))@self.coef*self.scale[:2]
        result=np.asarray(proxies,float).copy()
        result[:,:2]=np.maximum(result[:,:2],pred)
        return result

    def digest(self):
        return hashlib.sha256(self.coef.tobytes()+('\n'.join(self.ids)).encode()).hexdigest()


def choices(predictions,proxies,weight,scale,pool_index,random_index):
    model_index=int(np.argmin(scalar(predictions,weight,scale)))
    proxy_index=int(np.argmin(scalar(proxies,weight,scale)))
    explore=(pool_index%4==3)
    return dict(learned=int(random_index) if explore else model_index,
        proxy=int(random_index) if explore else proxy_index,random=int(random_index),
        predicted_best=model_index,proxy_best=proxy_index,explore=explore)


def directions():
    return np.maximum(generate_reference_points(3,8),.01)
