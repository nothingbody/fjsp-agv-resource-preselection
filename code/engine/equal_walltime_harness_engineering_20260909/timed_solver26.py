"""Timed integration of unchanged 3.0 solver steps; no execution at import."""
from __future__ import annotations
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SCIENCE=ROOT.parent/'baseline_calibration_and_transport_20260908'
sys.path.insert(0,str(SCIENCE))
import study24 as source
from deadline_gate26 import DeadlineGate

bc=source.budget_core
ARMS=('de_proxy','de_random4','ucb_w500_c1')
GENE_KEYS=('os','ma','agv_assign','agv_speed')
PROTOCOL='study26_equal_walltime_harness_engineering_v1'

def freeze_front(front):
    """Detach genes and cached objectives without invoking evaluate/decode."""
    out=[]
    for c in front:
        if c._objectives is None:
            raise RuntimeError('Cannot publish an unevaluated chromosome')
        objectives=tuple(float(x) for x in c._objectives.to_array())
        if len(objectives)!=3 or not all(math.isfinite(x) for x in objectives):
            raise ValueError('Invalid objective vector')
        genes=[]
        for name in GENE_KEYS:
            a=np.asarray(getattr(c,name))
            if a.dtype.kind not in 'iu':
                raise ValueError('Unexpected gene dtype')
            genes.append((name,a.dtype.str,tuple(a.shape),a.tobytes(order='C')))
        out.append((objectives,tuple(genes)))
    return tuple(out)

def unpack_front(payload):
    if payload is None:
        return [],[]
    objectives=[];genes=[]
    for values,encoded in payload:
        objectives.append(list(values))
        genes.append({name:np.frombuffer(data,dtype=np.dtype(dtype)).reshape(shape).tolist()
                      for name,dtype,shape,data in encoded})
    return objectives,genes

def solve(inst,dataset,name,seed,arm,budget=5.,checkpoints=(1.,3.,5.),
          clock=time.monotonic,max_batches=None):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    if max_batches is not None and (type(max_batches) is not int or max_batches<0):
        raise ValueError('Invalid engineering batch limit')
    real_start=time.monotonic();cpu_start=time.process_time()
    gate=DeadlineGate(budget,checkpoints,clock=clock,freeze_fn=freeze_front)
    random.seed(seed);np.random.seed(seed)
    key=bc.run_key(dataset,name,seed)
    batches=[];events=[];ds=None;archive_seconds=0.
    config=source.configure(arm) if arm.startswith('ucb_') else contextlib.nullcontext()
    with config,source.CountCalls() as count:
        if arm.startswith('ucb_'):
            state=bc.prepare_ucb(inst,seed,key,True)
            population,archive=state['population'],state['archive']
            selector=state['selector']
            if selector.window!=500 or selector.c!=1 or selector.min_count!=20:
                raise RuntimeError('Calibrated UCB identity mismatch')
        else:
            population=bc.initialize_population(inst,100,np.random.RandomState(bc.legacy_initialization_seed(seed)))
            archive=bc.update_archive([],population)
            ds=bc.RandomFour(population,seed,key) if arm=='de_random4' else bc.Decomposition(population,seed,arm,key)
        initial=bc.objective_array(population)
        initial_genes=[bc.genes(c) for c in population]
        reference=initial.max(axis=0)*1.1;wide=initial.max(axis=0)*1.5
        if count.evaluations!=100 or count.decodes!=100:
            raise RuntimeError('Initialization evaluation mismatch')
        accepted=gate.commit(archive,'initialization',count.evaluations)
        initialization_seconds=gate.commit_log[0]['seconds']
        while accepted and gate.remaining()>0 and (max_batches is None or len(batches)<max_batches):
            before=count.evaluations;batch=len(batches)
            if arm.startswith('ucb_'):
                result=bc.ucb_step(state)
                proposed=result.archive
                detail=dict(batch=batch,generation=state['generation'],
                            actions=[int(d['action']) for d in result.decisions],
                            rewards=[float(d['credit']['candidate_composite_reward']) for d in result.decisions],
                            reasons=[d['mandatory_reason'] for d in result.decisions])
                expected=100
            else:
                children=[];first=len(events)
                # Preserve 3.0's ten-evaluation publication granularity.
                for _ in range(10):
                    child,event=ds.advance();children.append(child);events.append(event)
                tick=time.perf_counter()
                proposed=bc.update_archive(archive,children)
                archive_seconds+=time.perf_counter()-tick
                detail=dict(batch=batch,event_start=first,event_end=len(events))
                expected=10
            if count.evaluations!=before+expected or count.evaluations!=count.decodes:
                raise RuntimeError('Step evaluation mismatch')
            accepted=gate.commit(proposed,f'batch_{batch}',count.evaluations)
            detail.update(evaluations=count.evaluations,accepted_before_deadline=accepted,
                          commit_seconds=gate.commit_log[-1]['seconds'])
            batches.append(detail)
            archive=proposed
        if not gate.commit_log[0]['accepted']:
            termination='INITIALIZATION_TOO_LATE'
        elif max_batches is not None and len(batches)>=max_batches:
            termination='ENGINEERING_FIXED_BATCH_LIMIT'
        else:
            termination='DEADLINE_REACHED'
        gate.finish()
        search_seconds=gate.elapsed()
        search_wall_seconds=time.monotonic()-real_start
        search_cpu_seconds=time.process_time()-cpu_start
        evaluations,decodes=count.evaluations,count.decodes
        reporting_start=time.monotonic()
        snapshots=[]
        for at in checkpoints:
            s=gate.snapshot(at)
            objectives,genes=unpack_front(s.pop('payload'))
            if objectives:
                F=np.asarray(objectives,dtype=float)
                hv=bc.fixed_reference_hv(F,reference)
                wide_hv=bc.fixed_reference_hv(F,wide)
                outside=int(np.any(F>reference+1e-12,axis=1).sum())
            else:
                hv=wide_hv=0.;outside=0
            s.update(objectives=objectives,genes=genes,hv=hv,hv_wide=wide_hv,
                     outside_reference_points=outside)
            snapshots.append(s)
        if count.evaluations!=evaluations or count.decodes!=decodes:
            raise RuntimeError('Reporting performed extra evaluations')
        reporting_seconds=time.monotonic()-reporting_start
    return dict(protocol=PROTOCOL,rng_protocol=bc.RNG_PROTOCOL,dataset=dataset,name=name,
        seed=seed,arm=arm,budget=budget,checkpoints=list(checkpoints),
        engineering_batch_limit=max_batches,termination=termination,
        initialization_seconds=initialization_seconds,search_seconds=search_seconds,
        search_wall_seconds=search_wall_seconds,cpu_seconds=search_cpu_seconds,
        search_overrun_seconds=max(0.,search_seconds-budget),
        evaluations=evaluations,decodes=decodes,
        reference=reference.tolist(),wide_reference=wide.tolist(),
        initial_objectives=initial.tolist(),initial_genes=initial_genes,
        batches=batches,events=events,snapshots=snapshots,commit_log=gate.commit_log,
        deadline_freeze_seconds=gate.freeze_seconds,
        final_committed_evaluations=snapshots[-1]['evaluations'],
        ucb_actions=int(state['selector'].total_counts.sum()) if arm.startswith('ucb_') else 0,
        dedicated_training_evaluations=0,unselected_evaluations=0,
        component_seconds=dict(ds.times,archive=archive_seconds) if ds else {},
        discarded_late_batches=sum(not b['accepted_before_deadline'] for b in batches),
        postprocessing_evaluations=count.evaluations-evaluations,
        snapshot_reporting_seconds=reporting_seconds,
        formal_timing_qualified=False,scientific_performance_evidence=False)
