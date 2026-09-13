"""Exact evaluation budgets. No time-dependent stopping or selection."""
import random
import time
import numpy as np
from compat import (CountCalls, initialize_population, legacy_initialization_seed,
    update_archive, objective_array, fixed_reference_hv, Chromosome)
from time_core import prepare_ucb, ucb_step, genes
from online_core import Decomposition, run_key, PROTOCOL as RNG_PROTOCOL
from random_four import RandomFour
from study21 import ARMS, PROTOCOL


def solve(inst, dataset, name, seed, arm, evaluations=10000,
          endpoints=(2000, 5000, 10000), progress_fn=None):
    assert arm in ARMS and evaluations >= 200 and (evaluations-100) % 100 == 0
    assert tuple(sorted(set(endpoints))) == tuple(endpoints) and endpoints[-1] == evaluations
    assert all(t >= 100 and (t-100) % 100 == 0 for t in endpoints)
    start = time.monotonic(); cpu = time.process_time()
    random.seed(seed); np.random.seed(seed)
    batches = []; events = []; captured = []; ds = None; archive_seconds = 0.; last_progress = 0.
    with CountCalls() as count:
        key = run_key(dataset, name, seed)
        if arm == 'ucb':
            state = prepare_ucb(inst, seed, key, True)
            population, archive = state['population'], state['archive']
        else:
            population = initialize_population(inst, 100, np.random.RandomState(legacy_initialization_seed(seed)))
            archive = update_archive([], population)
            ds = RandomFour(population, seed, key) if arm == 'de_random4' else Decomposition(population, seed, arm, key)
        initial = objective_array(population); initial_genes = [genes(c) for c in population]
        reference = initial.max(axis=0)*1.1; wide = initial.max(axis=0)*1.5
        assert count.evaluations == count.decodes == 100
        initialization_seconds = time.monotonic()-start
        if 100 in endpoints:
            captured.append((100, tuple(archive), initialization_seconds, time.process_time()-cpu))
        while count.evaluations < evaluations:
            before = count.evaluations; batch = len(batches)
            if arm == 'ucb':
                result = ucb_step(state); proposed = result.archive
                detail = dict(batch=batch, generation=state['generation'],
                    actions=[int(d['action']) for d in result.decisions],
                    rewards=[float(d['credit']['candidate_composite_reward']) for d in result.decisions],
                    reasons=[d['mandatory_reason'] for d in result.decisions])
                assert count.evaluations == before+100
            else:
                children = []; first = len(events)
                for _ in range(10):
                    child, event = ds.advance(); children.append(child); events.append(event)
                t = time.perf_counter(); proposed = update_archive(archive, children)
                archive_seconds += time.perf_counter()-t
                detail = dict(batch=batch, event_start=first, event_end=len(events))
                assert count.evaluations == before+10
            archive = proposed
            assert count.evaluations == count.decodes <= evaluations
            detail['evaluations'] = count.evaluations; batches.append(detail)
            elapsed = time.monotonic()-start
            if count.evaluations in endpoints:
                captured.append((count.evaluations, tuple(archive), elapsed, time.process_time()-cpu))
            if progress_fn and elapsed-last_progress >= 5.:
                progress_fn(dict(elapsed=elapsed, evaluations=count.evaluations, target=evaluations,
                    batches=len(batches), events=len(events)))
                last_progress = elapsed
        search_seconds = time.monotonic()-start; cpu_seconds = time.process_time()-cpu
        assert count.evaluations == count.decodes == evaluations
        assert [t for t, *_ in captured] == list(endpoints)
        snapshots = []
        for target, front, elapsed, used in captured:
            obj = objective_array(front)
            snapshots.append(dict(evaluations=target, commit_seconds=elapsed, cpu_seconds=used,
                genes=[genes(c) for c in front], objectives=obj.tolist(),
                hv=fixed_reference_hv(obj, reference), hv_wide=fixed_reference_hv(obj, wide),
                outside_reference_points=int(np.any(obj > reference+1e-12, axis=1).sum())))
        assert count.evaluations == count.decodes == evaluations, 'Output decoded an extra candidate'
    return dict(protocol=PROTOCOL, rng_protocol=RNG_PROTOCOL, dataset=dataset, name=name, seed=seed, arm=arm,
        evaluations=evaluations, decodes=evaluations, endpoints=list(endpoints),
        initialization_seconds=initialization_seconds, search_seconds=search_seconds, cpu_seconds=cpu_seconds,
        reference=reference.tolist(), wide_reference=wide.tolist(), initial_objectives=initial.tolist(),
        initial_genes=initial_genes, batches=batches, events=events, snapshots=snapshots,
        ucb_actions=int(state['selector'].total_counts.sum()) if arm == 'ucb' else 0,
        dedicated_training_evaluations=0, unselected_evaluations=0,
        component_seconds=dict(ds.times, archive=archive_seconds) if ds else {},
        snapshot_reporting_seconds=time.monotonic()-start-search_seconds)


def scientific_payload(r):
    """All deterministic content, excluding environmental timings and wrappers."""
    keys = ('protocol', 'rng_protocol', 'dataset', 'name', 'seed', 'arm', 'evaluations', 'decodes',
        'endpoints', 'reference', 'wide_reference', 'initial_objectives', 'initial_genes',
        'batches', 'events', 'ucb_actions', 'dedicated_training_evaluations', 'unselected_evaluations')
    out = {k: r[k] for k in keys}
    out['snapshots'] = [{k: v for k, v in s.items() if k not in ('commit_seconds', 'cpu_seconds')}
        for s in r['snapshots']]
    return out


def verify_pair(original, replay, inst):
    assert scientific_payload(original) == scientific_payload(replay), 'Deterministic trajectory mismatch'
    independent = 0
    with CountCalls() as count:
        for s in original['snapshots']:
            values = np.asarray([Chromosome(inst, **{k:np.asarray(v, dtype=int) for k,v in g.items()})
                .objectives.to_array() for g in s['genes']])
            assert np.array_equal(values, s['objectives'])
            assert fixed_reference_hv(values, np.asarray(original['reference'])) == s['hv']
            assert fixed_reference_hv(values, np.asarray(original['wide_reference'])) == s['hv_wide']
            independent += len(values)
        assert count.evaluations == count.decodes == independent
    return dict(status='VERIFIED', replay_evaluations=replay['evaluations'],
        independent_redecodes=independent, snapshot_matches=len(original['snapshots']),
        hv_recomputations=2*len(original['snapshots']),
        event_matches=len(original['events']), generation_matches=len(original['batches']) if original['arm']=='ucb' else 0)
