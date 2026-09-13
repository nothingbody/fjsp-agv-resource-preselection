"""Schedule feasibility and objective audit without original decoder/energy calls."""
import math
import numpy as np
FIELDS=('transport_start','transport_end','transport_agv','transport_speed','transport_empty_distance','transport_loaded_distance')
def near(a,b,tag):
    assert math.isfinite(float(a)) and math.isfinite(float(b)),tag
    assert abs(float(a)-float(b)) <= 1e-7+1e-10*max(abs(a),abs(b)),(tag,a,b)
def ge(a,b,tag):assert a+1e-7+1e-10*max(abs(a),abs(b))>=b,(tag,a,b)
def audit(c,s,mode,objectives=None):
    inst=c.instance;n=inst.total_operations
    assert all(len(getattr(c,k))==n for k in ('os','ma','agv_assign','agv_speed'))
    assert np.array_equal(np.bincount(c.os,minlength=inst.num_jobs),inst.num_operations)
    pairs=[(j,o) for j in range(inst.num_jobs) for o in range(inst.num_operations[j])]
    assert set(s.op_start)==set(s.op_end)==set(s.op_machine)==set(pairs)
    offset=np.r_[0,np.cumsum(inst.num_operations)[:-1]]
    machine={};vehicle={};speed={};duration={}
    for k,key in enumerate(pairs):
        ms=inst.get_compatible_machines(*key)
        assert 0<=c.ma[k]<len(ms) and 0<=c.agv_assign[k]<inst.num_agv and 0<=c.agv_speed[k]<inst.num_speeds
        machine[key]=ms[c.ma[k]];vehicle[key]=int(c.agv_assign[k]);speed[key]=inst.speed_levels[c.agv_speed[k]]
        assert speed[key]>0
        duration[key]=inst.get_processing_time(*key,machine[key])
        assert s.op_machine[key]==machine[key]
        ge(s.op_start[key],0,('nonnegative',key))
        near(s.op_end[key]-s.op_start[key],duration[key],('duration',key))
    count=np.zeros(inst.num_jobs,int);ordered=[]
    for j in c.os:
        key=(int(j),int(count[j]));count[j]+=1;ordered.append(key)
    needs={key for key in pairs if mode!='no_same_machine_service' or key[1]==0 or machine[key]!=machine[key[0],key[1]-1]}
    for field in FIELDS:assert set(getattr(s,field))==needs,(field,needs-set(getattr(s,field)))
    loads=[];energy=0.
    for m in range(inst.num_machines):
        keys=[key for key in ordered if machine[key]==m]
        ops=s.machine_schedule[m]
        assert [tuple(x[:2]) for x in ops]==keys,('machine_order',m)
        setup_count=0
        for t,(j,o,st,en) in enumerate(ops):
            near(st,s.op_start[j,o],'machine_start');near(en,s.op_end[j,o],'machine_end')
            if t:
                setup_count+=int(ops[t-1][0]!=j)
                gap=float(inst.machine_setup_time[m]) if inst.machine_setup_time is not None and ops[t-1][0]!=j else 0.
                ge(st,ops[t-1][3]+gap,('machine_overlap',m))
        load=sum(duration[key] for key in keys);loads.append(load)
        setup_each=float(inst.machine_setup_time[m]) if inst.machine_setup_time is not None else 4.
        span=ops[-1][3]-ops[0][2] if ops else 0.
        idle=max(0.,span-load-setup_count*setup_each)
        pp=float(inst.machine_proc_power[m]) if inst.machine_proc_power is not None else 5.
        ps=float(inst.machine_setup_power[m]) if inst.machine_setup_power is not None else 2.5
        pi=float(inst.machine_idle_power[m]) if inst.machine_idle_power is not None else 1.
        energy+=pp*load+ps*setup_count*setup_each+pi*idle
    for key in pairs:
        j,o=key;ready=0. if o==0 else s.op_end[j,o-1]
        if key in needs:
            ge(s.transport_start[key],ready,('pickup_before_completion',key))
            ge(s.op_start[key],s.transport_end[key],('processing_before_delivery',key))
        else:ge(s.op_start[key],ready,('same_machine_precedence',key))
    for a in range(inst.num_agv):
        keys=[key for key in ordered if key in needs and vehicle[key]==a]
        assert [tuple(x[:2]) for x in s.agv_schedule[a]]==keys,('vehicle_order',a)
        loc=0;avail=0.
        for j,o,ts,te in s.agv_schedule[a]:
            key=(j,o);origin=0 if o==0 else machine[j,o-1]+1;dest=machine[key]+1;v=speed[key]
            empty=float(inst.get_distance(loc,origin));loaded=float(inst.get_distance(origin,dest))
            near(ts,s.transport_start[key],'agv_start');near(te,s.transport_end[key],'agv_end')
            near(empty,s.transport_empty_distance[key],'empty_distance');near(loaded,s.transport_loaded_distance[key],'loaded_distance')
            near(v,s.transport_speed[key],'vehicle_speed');assert s.transport_agv[key]==a
            ge(ts,avail+empty/v,('empty_trip_overlap',key))
            near(te,ts+loaded/v,('loaded_duration',key))
            ea,eb,ec=inst.agv_empty_power_params;la,lb,lc=inst.agv_load_power_params
            energy+=(ea*v*v+eb*v+ec)*empty/v+(la*v*v+lb*v+lc)*loaded/v
            loc=dest;avail=te
    vec=np.array([max(s.op_end.values(),default=0.),energy,max(loads)-min(loads)])
    if objectives is not None:
        for a,b in zip(vec,objectives):near(a,b,'independent_objective')
    return vec
