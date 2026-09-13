"""One timed study cell; separate validation, exclusive files, no retry."""
from common_time38 import *
import argparse,traceback
IMPORT_START=time.monotonic()
sys.path.insert(0,str(H))
import timed_solver26 as timed
from run_engineering26 import assert_deadlines,validation_view
IMPORT_SECONDS=time.monotonic()-IMPORT_START
def main():
    p=argparse.ArgumentParser();p.add_argument('--triplet',required=True);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();tasks=read(ROOT/'TRIPLETS.json')
    selected=[x for x in tasks if x['id']==a.triplet];assert len(selected)==1
    cell=selected[0];out=ROOT/'formal_a01'/cell['id']/a.arm
    out.mkdir(parents=True,exist_ok=False)
    try:
        lock=input_lock()
        assert os.sched_getaffinity(0)=={cell['cpu']}
        put(out/'START.json',dict(utc=utc(),pid=os.getpid(),cell=cell,arm=a.arm,inputs_lock=lock))
        tick=time.monotonic();inst=timed.source.restore_instance(read(ROOT/cell['input_file']))
        input_seconds=time.monotonic()-tick
        before=time.monotonic()
        raw=timed.solve(inst,cell['dataset'],cell['name'],cell['seed'],a.arm,
                        budget=BUDGET,checkpoints=CHECKPOINTS)
        after=time.monotonic()
        raw['enhancement38_provenance']=dict(protocol=PROTOCOL,triplet=cell,arm=a.arm,
            inputs_lock=lock,search_call_before_monotonic=before,search_call_after_monotonic=after,
            import_seconds=IMPORT_SECONDS,input_seconds=input_seconds,cpu=cell['cpu'])
        serial=time.monotonic();packed(out/'raw.json.gz',raw);serial=time.monotonic()-serial
        checks=assert_deadlines(raw)
        assert raw['engineering_batch_limit'] is None
        assert raw['termination'] in ('DEADLINE_REACHED','INITIALIZATION_TOO_LATE')
        assert raw['budget']==BUDGET and raw['checkpoints']==list(CHECKPOINTS)
        validation=timed.source.old.verify(validation_view(raw),inst)
        assert os.sched_getaffinity(0)=={cell['cpu']}
        assert input_lock()==lock
        ratios=raw['cpu_seconds']/raw['search_wall_seconds']
        # Outcomes are retained even when a prospective resource criterion fails.
        resource_pass=math.isfinite(ratios) and .90<=ratios<=1.05
        endpoints=[]
        for snap in raw['snapshots']:
            narrow=float(timed.np.prod(raw['reference']));wide=float(timed.np.prod(raw['wide_reference']))
            assert narrow>0 and wide>0 and math.isfinite(narrow) and math.isfinite(wide)
            assert all(math.isfinite(x) and x>0 for x in raw['reference']+raw['wide_reference'])
            endpoints.append(dict(deadline=snap['deadline'],status=snap['status'],
                hv_normalized=snap['hv']/narrow,hv_wide_normalized=snap['hv_wide']/wide,
                committed_evaluations=snap['evaluations'],outside_reference_points=snap['outside_reference_points'],
                last_commit_lag_seconds=snap['deadline']-snap['commit_seconds'] if snap['status']=='COMMITTED' else None,
                committed_batches=sum(b['accepted_before_deadline'] and b['commit_seconds']<=snap['deadline'] for b in raw['batches'])))
        report=dict(status='CELL_VALIDATED' if resource_pass else 'CELL_RESOURCE_FAILED',
            triplet=cell['id'],base_id=cell['base_id'],stratum=cell['stratum'],regime=cell['regime'],
            seed=cell['seed'],arm=a.arm,cpu=cell['cpu'],utc=utc(),inputs_lock=lock,
            raw_sha256=sha(out/'raw.json.gz'),checks=checks,validation=validation,
            search_evaluations=raw['evaluations'],search_decodes=raw['decodes'],
            validation_evaluations=validation['validation_evaluations'],
            initialization_seconds=raw['initialization_seconds'],
            search_wall_seconds=raw['search_wall_seconds'],cpu_seconds=raw['cpu_seconds'],
            cpu_wall=ratios,overrun_seconds=raw['search_overrun_seconds'],
            discarded_late_batches=raw['discarded_late_batches'],
            late_search_evaluations=raw['evaluations']-raw['final_committed_evaluations'],
            snapshot_reporting_seconds=raw['snapshot_reporting_seconds'],
            serialization_seconds=serial,endpoints=endpoints,reference=raw['reference'],
            wide_reference=raw['wide_reference'],
            initial_identity=digest([raw['initial_genes'],raw['initial_objectives']]),
            formal_study_qualified=False)
        put(out/'VALIDATION.json',report)
        print(json.dumps(dict(status=report['status'],triplet=cell['id'],arm=a.arm)),flush=True)
        return 0 if resource_pass else 3
    except Exception as e:
        put(out/'FAILURE.json',dict(utc=utc(),error=repr(e),traceback=traceback.format_exc(),no_retry=True))
        return 2
if __name__=='__main__':raise SystemExit(main())
