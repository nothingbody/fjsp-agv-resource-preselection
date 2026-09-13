import bootstrap
from bootstrap import ROOT,STUDY,restore,check_legacy
import copy,json,time,traceback
import numpy as np
import compat
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.encoding import random_chromosome
from models import MODES,semantics,features,first_rightshift,OriginalFeatures
from independent_validation import audit,near
def toy(counts,pt,cm,dist):
    return FJSPAGVInstance(2,2,1,counts,pt,cm,np.asarray(dist,float),[.5,.75,1.],
        np.array([5.,5.]),np.array([1.,1.]),np.zeros(2),np.zeros(2),
        (5.,0.,0.),(3.,0.,0.))
def chrom(inst,seq):
    n=inst.total_operations
    return compat.Chromosome(inst,np.array(seq),np.zeros(n,int),np.zeros(n,int),np.full(n,2,int))
def run_one(c,mode):
    with semantics(mode):
        v=c.objectives.to_array()
        independent=audit(c,c.schedule,mode,v)
        probe=c.copy()
        with compat.CountCalls() as counts:
            f,p=features(c.instance,np.ones(3),mode)(probe)
            assert counts.evaluations==counts.decodes==0
            assert probe._schedule is None and probe._objectives is None
        assert np.all(p<=v+1e-7+1e-10*np.abs(v)),(mode,p,v)
        near(p[2],v[2],'load_exact')
    return v,p,c.schedule
def main():
    start=time.monotonic();check_legacy()
    a=toy([1,2],{(0,0,0):1.,(1,0,1):10.,(1,1,0):1.},{(0,0):[0],(1,0):[1],(1,1):[0]},np.zeros((3,3)))
    c=chrom(a,[0,1,1]);before=run_one(c,'original')[0]
    after=run_one(c.copy(),'first_rightshift')[0]
    assert before.tolist()==[11.,69.,8.] and after.tolist()==[11.,60.,8.]
    b=toy([2,1],{(0,0,0):1.,(0,1,0):1.,(1,0,1):10.},{(0,0):[0],(0,1):[0],(1,0):[1]},[[0,1,11],[1,0,10],[11,10,0]])
    d=chrom(b,[0,1,0]);old=run_one(d,'original')[0]
    fixed,p,s=run_one(d.copy(),'no_same_machine_service')
    assert old.tolist()==[24.,174.,8.] and fixed.tolist()==[23.,123.,8.]
    assert (0,1) not in s.transport_start and s.op_start[0,1]==2.
    old_proxy=OriginalFeatures(b,np.ones(3))(d)[1]
    assert old_proxy[1]>fixed[1]
    # Explicit adversarial check: a pickup before predecessor completion must fail.
    bad=copy.deepcopy(d.schedule);bad.op_end[0,0]=100.
    caught=False
    try:audit(d,bad,'original')
    except AssertionError:caught=True
    assert caught
    records=json.loads((STUDY/'INPUT_MANIFEST.json').read_text())
    rows=[];retimed=0;samehop=0
    for ri,rec in enumerate(records):
        inst=restore(rec);rng=np.random.RandomState(2026091300+ri)
        for k in range(8):
            c=random_chromosome(inst,rng);g=tuple(getattr(c,z).tobytes() for z in ('os','ma','agv_assign','agv_speed'))
            vo,po,so=run_one(c,'original')
            vr,pr,sr=run_one(c.copy(),'first_rightshift')
            vn,pn,sn=run_one(c.copy(),'no_same_machine_service')
            assert tuple(getattr(c,z).tobytes() for z in ('os','ma','agv_assign','agv_speed'))==g
            near(vr[0],vo[0],'retiming_makespan');near(vr[2],vo[2],'retiming_load')
            assert vr[1]<=vo[1]+1e-7
            assert so.transport_start==sr.transport_start and so.transport_end==sr.transport_end
            sr2=first_rightshift(inst,sr)
            assert sr2.op_start==sr.op_start and sr2.op_end==sr.op_end,'idempotence'
            changed=sum(sr.op_start[key]>so.op_start[key]+1e-8 for key in so.op_start)
            skipped=len(so.transport_start)-len(sn.transport_start)
            if skipped==0:assert np.array_equal(vo,vn) and np.array_equal(po,pn)
            retimed+=int(changed>0);samehop+=int(skipped>0)
            rows.append(dict(base=rec['base_id'],regime=rec['regime'],candidate=k,original=vo.tolist(),retimed=vr.tolist(),nohop=vn.tolist(),changed_first_ops=int(changed),skipped_services=skipped))
        if (ri+1)%10==0:print(json.dumps({'checked_input_cells':ri+1,'total':len(records)}),flush=True)
    check_legacy()
    with (ROOT/'checks/model_diagnostics.json').open('x') as f:json.dump(rows,f)
    result=dict(status='PASSED',scope='unit and property checks, not optimization experiments',
        tested_generated_chromosomes=len(rows),generated_input_cells=len(records),mode_checks=3*len(rows),
        candidates_with_first_shift=retimed,candidates_with_same_machine_service=samehop,
        toy_retiming={'before':before.tolist(),'after':after.tolist()},
        toy_nohop={'before':old.tolist(),'after':fixed.tolist(),'old_proxy':old_proxy.tolist(),'new_proxy':p.tolist()},
        negative_validation_test_passed=True,legacy_hashes_unchanged=476,elapsed_seconds=time.monotonic()-start)
    with (ROOT/'checks/MODEL_QA.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (ROOT/'checks/MODEL_QA_FAILURE.json').open('x') as f:json.dump({'status':'FAILED','traceback':traceback.format_exc()},f,indent=2)
        raise
