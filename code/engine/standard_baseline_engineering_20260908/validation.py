"""Independent checks of saved schedules, HV, counters and deterministic replay."""
import hashlib,json
import numpy as np
from standard_adapters import (Chromosome,CountCalls,GENES,canonical,legal,
                               scientific_payload,fingerprint)

def chromosome(inst,g):
    assert set(g)==set(GENES)
    return Chromosome(inst,**{k:np.asarray(g[k],dtype=int) for k in GENES})

def feasible(c):
    legal(c);i=c.instance;s=c.schedule;tol=1e-8
    keys={(j,o) for j in range(i.num_jobs) for o in range(i.num_operations[j])}
    for attr in ('op_start','op_end','op_machine','transport_start','transport_end',
                 'transport_agv','transport_speed'):
        assert set(getattr(s,attr))==keys
    flat=0
    for j in range(i.num_jobs):
        for o in range(i.num_operations[j]):
            k=(j,o);m=i.get_compatible_machines(j,o)[c.ma[flat]]
            assert s.op_machine[k]==m
            assert s.transport_agv[k]==c.agv_assign[flat]
            assert s.transport_speed[k]==i.speed_levels[c.agv_speed[flat]]
            assert s.op_start[k]>=s.transport_end[k]-tol
            assert s.transport_start[k]>=(s.op_end[(j,o-1)] if o else 0.)-tol
            assert abs(s.op_end[k]-s.op_start[k]-i.get_processing_time(j,o,m))<tol
            flat+=1
    machine_keys=[];agv_keys=[]
    for m,items in s.machine_schedule.items():
        prev_end=0.;prev_job=None
        for j,o,start,end in items:
            machine_keys.append((j,o));assert s.op_machine[(j,o)]==m
            setup=0. if prev_job in (None,j) or i.machine_setup_time is None else i.machine_setup_time[m]
            assert start>=prev_end+setup-tol
            assert start==s.op_start[(j,o)] and end==s.op_end[(j,o)]
            prev_end=end;prev_job=j
    for v,items in s.agv_schedule.items():
        prev_end=0.;prev_loc=0
        for j,o,start,end in items:
            k=(j,o);agv_keys.append(k);assert s.transport_agv[k]==v
            origin=s.op_machine[(j,o-1)]+1 if o else 0
            dest=s.op_machine[k]+1;speed=s.transport_speed[k]
            assert speed>0
            empty=i.get_distance(prev_loc,origin);loaded=i.get_distance(origin,dest)
            assert start>=prev_end+empty/speed-tol
            assert abs(end-start-loaded/speed)<tol
            assert s.transport_empty_distance[k]==empty
            assert s.transport_loaded_distance[k]==loaded
            assert start==s.transport_start[k] and end==s.transport_end[k]
            prev_end=end;prev_loc=dest
    assert len(machine_keys)==len(set(machine_keys))==len(keys) and set(machine_keys)==keys
    assert len(agv_keys)==len(set(agv_keys))==len(keys) and set(agv_keys)==keys
    f=c.objectives.to_array()
    loads=np.array([sum(e-b for _,_,b,e in s.machine_schedule[m]) for m in range(i.num_machines)])
    assert f[0]==max(s.op_end.values())
    np.testing.assert_allclose(f[2],loads.max()-loads.min(),rtol=0,atol=tol)
    return len(keys)

def hv_independent(F,reference):
    """3-D union of boxes by nested exact coordinate sweeps (no pymoo calls)."""
    P=np.asarray(F,float);r=np.asarray(reference,float)
    assert r.shape==(3,) and np.all(np.isfinite(r))
    if P.size==0:return 0.
    assert P.ndim==2 and P.shape[1]==3 and np.all(np.isfinite(P))
    P=P[np.all(P<=r,axis=1)]
    if len(P)==0:return 0.
    xs=np.unique(np.r_[P[:,0],r[0]]);volume=0.
    for x,xx in zip(xs[:-1],xs[1:]):
        yz=P[P[:,0]<=x,1:];ys=np.unique(np.r_[yz[:,0],r[1]])
        area=0.
        for y,yy in zip(ys[:-1],ys[1:]):
            zs=yz[yz[:,0]<=y,1]
            if len(zs):area+=(yy-y)*max(0.,r[2]-zs.min())
        volume+=(xx-x)*area
    return float(volume)

def verify_saved(r,inst):
    n=r['evaluations']
    assert r['decodes']==r['pymoo_n_eval']==n
    assert len(r['initial_genes'])==len(r['initial_objectives'])==100
    assert len(r['events'])==n-100
    assert [x['evaluation'] for x in r['events']]==list(range(101,n+1))
    assert all(np.all(np.isfinite(x['raw_objectives'])) for x in r['events'])
    assert sum(x['new_evaluations'] for x in r['batches'])==n-100
    assert [x['evaluations'] for x in r['snapshots']]==r['endpoints']
    pending=r['proposal_count']-(n-100)
    assert pending==r['pending_unevaluated_proposals']
    assert pending in ((0,) if r['arm']=='nsga2_shared' else (0,1))
    if r['arm']=='moead_shared' and r['replacement_checks'] is not None:
        assert r['replacement_checks']==n-100
    initial=np.asarray(r['initial_objectives'])
    np.testing.assert_array_equal(r['reference'],initial.max(axis=0)*1.1)
    np.testing.assert_array_equal(r['wide_reference'],initial.max(axis=0)*1.5)
    np.testing.assert_array_equal(r['internal_scale'],np.maximum(initial.max(axis=0),1.))
    checked=0;operations=0
    with CountCalls() as counter:
        for g,f in zip(r['initial_genes'],r['initial_objectives']):
            c=chromosome(inst,g);operations+=feasible(c)
            np.testing.assert_array_equal(c.objectives.to_array(),f);checked+=1
        for snap in r['snapshots']:
            assert len(snap['genes'])==len(snap['objectives'])<=100
            event_pairs={(e['genes_sha256'],tuple(e['raw_objectives']))
                         for e in r['events'] if e['evaluation']<=snap['evaluations']}
            event_pairs.update((fingerprint(chromosome(inst,g)),tuple(f))
                               for g,f in zip(r['initial_genes'],r['initial_objectives']))
            for g,f in zip(snap['genes'],snap['objectives']):
                c=chromosome(inst,g);assert (fingerprint(c),tuple(f)) in event_pairs
                operations+=feasible(c);np.testing.assert_array_equal(c.objectives.to_array(),f)
                checked+=1
            F=np.asarray(snap['objectives'])
            assert len(np.unique(F,axis=0))==len(F)
            for row in F:assert not np.any(np.all(F<=row,axis=1)&np.any(F<row,axis=1))
            for field,ref in (('hv','reference'),('hv_wide','wide_reference')):
                np.testing.assert_allclose(snap[field],hv_independent(F,r[ref]),rtol=1e-12,atol=1e-8)
            assert snap['outside_reference_points']==int(np.any(F>np.array(r['reference'])+1e-12,axis=1).sum())
        assert counter.evaluations==counter.decodes==checked
    return {'saved_schedules_checked':checked,'operation_assignments_checked':operations,
            'validation_evaluations':counter.evaluations,'validation_decodes':counter.decodes}

def verify_pair(a,b,inst):
    assert scientific_payload(a)==scientific_payload(b),'Deterministic replay mismatch'
    assert a['replacement_checks']==b['replacement_checks']
    assert a['replacement_digest']==b['replacement_digest']
    result=verify_saved(a,inst)
    result['scientific_sha256']=hashlib.sha256(canonical(scientific_payload(a))).hexdigest()
    result['exact_replay']=True
    return result
