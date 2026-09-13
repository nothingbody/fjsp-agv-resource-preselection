# Derived for stationary same-machine continuation; original files unchanged.
import numpy as np

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
        distance=np.where(previous != machine+1, inst.distance_matrix[previous,machine+1], 0.)
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
            if previous[k] != m+1:
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
