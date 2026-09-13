"""Common deadline, original archive, frozen UCB and standalone CP search."""
from __future__ import annotations
import json
import random
import time
import numpy as np
from compat import (fine,LightUCB,set_diagnostic_light,CountCalls,RNGLedger,
    legacy_initialization_seed,initialize_population,update_archive,update_generation_archive,
    objective_array,fixed_reference_hv,generate_reference_points)
from cp_core import Kernel
from lns_loop import structural_cluster,fingerprint

ARMS=('ucb','cp_random','cp_structural','cp_sequential')
SEEDS=(9081501,9081502)
INSTANCES=tuple([('brandimarte',f'Mk{i:02}.fjs') for i in (1,3,7,9)]+
                [('hurink_edata',f'la{i:02}.fjs') for i in (1,6,12,18,24,30,36,40)])
PROTOCOL='cp_lns_resource_balanced_v3_20260908'
RNG_PROTOCOL='cp_lns_equal_time_v1_20260908'


def genes(chrom):
    return {k:getattr(chrom,k).tolist() for k in ('os','ma','agv_assign','agv_speed')}


def json_state(selector):
    return json.loads(json.dumps(selector.state_dict()))


class Deadline:
    def __init__(self,budget,clock=time.monotonic):
        self.clock=clock
        self.start=clock()
        self.budget=float(budget)
        self.commits=[]
    def elapsed(self):
        return self.clock()-self.start
    def remaining(self):
        return self.budget-self.elapsed()
    def commit(self,archive,label):
        # References are immutable in genes: all variation and CP output copy.
        prepared=tuple(archive)
        stamp=self.elapsed()
        if stamp>self.budget:
            return False
        self.commits.append((stamp,prepared,label))
        return True
    def snapshot(self,at):
        eligible=[c for c in self.commits if c[0]<=at]
        if not eligible:
            raise RuntimeError('No complete initialization before deadline')
        return eligible[-1]


def prepare_ucb(instance,seed,run_key,light=True):
    set_diagnostic_light(light)
    pop=initialize_population(instance,100,np.random.RandomState(legacy_initialization_seed(seed)))
    archive=update_archive([],pop)
    selector=(LightUCB if light else fine.DelayedExposureUCB)(k=10,n_ops=10,c=1.)
    ledger=RNGLedger(seed)
    initial=objective_array(pop)
    reference=initial.max(axis=0)*1.1
    return dict(population=pop,archive=archive,selector=selector,ledger=ledger,run_key=run_key,
        reference=reference,previous_hv=fixed_reference_hv(initial,reference),
        previous_best=float(initial[:,0].min()),stagnation=0,hv_window=[],generation=0,
        dirs=generate_reference_points(3,8),spec=fine.GranularitySpec(k=10,population_size=100,tournament_size=5))


def ucb_step(state):
    g=state['generation']
    result=fine.execute_finegrained_generation(population=state['population'],archive=state['archive'],
        selector=state['selector'],spec=state['spec'],ledger=state['ledger'],run_key=state['run_key'],
        generation=g,max_generations=200,ref_points=state['dirs'],online_reference=state['reference'],
        previous_hv=state['previous_hv'],previous_best_cmax=state['previous_best'],
        stagnation=state['stagnation'],hv_window=state['hv_window'],update_archive_fn=update_generation_archive)
    state['stagnation']=state['stagnation']+1 if result.current_hv<=state['previous_hv']*1.001 else 0
    state['hv_window']=(state['hv_window']+[result.global_hv_delta])[-20:]
    state.update(population=result.population,archive=result.archive,previous_hv=result.current_hv,
                 previous_best=result.current_best_cmax,generation=g+1)
    assert int(state['selector'].total_counts.sum())==10*(g+1)
    assert all(r['completed'] for r in state['selector'].records)
    return result


def fixed_trace(instance,seed,generations=30,light=True,run_key='deadline_regression'):
    random.seed(seed)
    np.random.seed(seed)
    t=time.monotonic()
    with CountCalls() as counts:
        state=prepare_ucb(instance,seed,run_key,light)
        actions=[]
        rewards=[]
        reasons=[]
        for _ in range(generations):
            result=ucb_step(state)
            actions.append([int(d['action']) for d in result.decisions])
            rewards.append([float(d['credit']['candidate_composite_reward']) for d in result.decisions])
            reasons.append([d['mandatory_reason'] for d in result.decisions])
        assert counts.evaluations==counts.decodes==100*(generations+1)
    return dict(actions=actions,rewards=rewards,reasons=reasons,
        population=[genes(c) for c in state['population']],population_objectives=objective_array(state['population']).tolist(),
        archive=[genes(c) for c in state['archive']],archive_objectives=objective_array(state['archive']).tolist(),
        selector_state=json_state(state['selector']),evaluations=counts.evaluations,decodes=counts.decodes,
        seconds=time.monotonic()-t)


def solve(instance,dataset,name,seed,arm,budget=120.,checkpoints=(30.,60.,120.),progress_fn=None,recovery_path=None):
    if arm not in ARMS or budget<=0 or any(t>budget for t in checkpoints):
        raise ValueError('Invalid protocol arguments')
    overall_cpu=time.process_time()
    gate=Deadline(budget)
    random.seed(seed)
    np.random.seed(seed)
    run_key=f'{dataset}/{name}/seed{seed}/{RNG_PROTOCOL}'
    logs=[]
    rng=np.random.RandomState(seed)
    last_progress=0.
    with CountCalls() as counts:
        if arm=='ucb':
            state=prepare_ucb(instance,seed,run_key,True)
            pop,archive=state['population'],state['archive']
        else:
            pop=initialize_population(instance,100,np.random.RandomState(legacy_initialization_seed(seed)))
            archive=update_archive([],pop)
        initial=objective_array(pop)
        initial_genes=[genes(c) for c in pop]
        reference=initial.max(axis=0)*1.1
        wide=initial.max(axis=0)*1.5
        assert counts.evaluations==counts.decodes==100
        assert gate.commit(archive,'initialization')
        initialization_seconds=gate.commits[-1][0]
        call=0
        cycle=0
        while gate.remaining()>0.01:
            tick=time.monotonic()
            if arm=='ucb':
                result=ucb_step(state)
                valid=gate.commit(result.archive,f'g{state["generation"]}')
                logs.append(dict(kind='ucb',generation=state['generation'],accepted_before_deadline=valid,
                    elapsed=gate.elapsed(),step_seconds=time.monotonic()-tick,evaluations=counts.evaluations,
                    actions=[int(d['action']) for d in result.decisions],
                    reasons=[d['mandatory_reason'] for d in result.decisions]))
                if not valid:
                    break
            else:
                objective=cycle%3
                sweep=cycle//3
                values=objective_array(archive)
                pi=int(np.argmin(values[:,objective])) if sweep%4==0 else int(rng.randint(len(archive)))
                parent=archive[pi]
                cluster=(sorted(int(x) for x in rng.choice(instance.total_operations,4,replace=False))
                    if arm=='cp_random' else structural_cluster(parent,4,rng))
                epsilon=[None if k==objective else float(v)*1.05 for k,v in enumerate(parent.objectives.to_array())]
                families=[('order',),('machine',),('agv',),('speed',)] if arm=='cp_sequential' else [('order','machine','agv','speed')]
                for stage,family in enumerate(families):
                    if gate.remaining()<=0.01:
                        break
                    before=parent.objectives.to_array().copy()
                    kernel=Kernel(parent,cluster,objective,epsilon,families=family)
                    remaining=gate.remaining()
                    if remaining<=0.02:
                        logs.append(dict(kind='cp',cycle=cycle,stage=stage,status='BUILD_EXCEEDED_DEADLINE_RESERVE',
                            elapsed=gate.elapsed(),build_seconds=kernel.build_seconds,accepted_before_deadline=False,
                            evaluations=counts.evaluations))
                        break
                    cap=min(1.25 if arm=='cp_sequential' else 5.,max(0.001,remaining-0.01))
                    candidate,detail=kernel.solve(cap,seed+call,recovery_path=recovery_path,
                        recovery_context=dict(dataset=dataset,name=name,seed=seed,arm=arm,protocol=PROTOCOL,
                            cycle=cycle,stage=stage,call=call),deadline=gate.start+gate.budget)
                    detail.update(kind='cp',cycle=cycle,stage=stage,parent_objectives=before.tolist(),parent_fingerprint=fingerprint(parent))
                    proposed=update_archive(archive,[candidate]) if candidate is not None else archive
                    accepted=gate.commit(proposed,f'cp{call}')
                    detail.update(accepted_before_deadline=accepted,elapsed=gate.elapsed(),evaluations=counts.evaluations)
                    logs.append(detail)
                    call+=1
                    if not accepted:
                        break
                    archive=proposed
                    if candidate is not None and candidate.objectives.to_array()[objective]<=before[objective]+1e-8:
                        parent=candidate
                cycle+=1
            if progress_fn and gate.elapsed()-last_progress>=5.:
                progress_fn(dict(elapsed=gate.elapsed(),budget=budget,commits=len(gate.commits),evaluations=counts.evaluations))
                last_progress=gate.elapsed()
        search_seconds=gate.elapsed()
        cpu_seconds=time.process_time()-overall_cpu
        evaluations,decodes=counts.evaluations,counts.decodes
    # Read-only output serialization after search. No new decode/evaluate is allowed.
    snapshots=[]
    for at in checkpoints:
        stamp,front,label=gate.snapshot(at)
        objs=objective_array(front)
        snapshots.append(dict(deadline=at,commit_seconds=stamp,commit_label=label,
            objectives=objs.tolist(),genes=[genes(c) for c in front],
            hv=fixed_reference_hv(objs,reference),hv_wide=fixed_reference_hv(objs,wide),
            outside_reference_points=int(np.any(objs>reference+1e-12,axis=1).sum())))
    return dict(protocol=PROTOCOL,dataset=dataset,name=name,seed=seed,arm=arm,budget=budget,
        initialization_seconds=initialization_seconds,search_seconds=search_seconds,cpu_seconds=cpu_seconds,
        evaluations=evaluations,decodes=decodes,reference=reference.tolist(),wide_reference=wide.tolist(),
        initial_objectives=initial.tolist(),initial_genes=initial_genes,
        initial_fingerprints=[fingerprint(c) for c in pop],snapshots=snapshots,steps=logs,
        commit_log=[dict(seconds=t,label=label,size=len(front)) for t,front,label in gate.commits],
        ucb_actions=int(state['selector'].total_counts.sum()) if arm=='ucb' else 0,
        cp_calls=call if arm!='ucb' else 0,
        discarded_late_steps=sum(not r['accepted_before_deadline'] for r in logs),
        serialization_seconds=gate.elapsed()-search_seconds)
