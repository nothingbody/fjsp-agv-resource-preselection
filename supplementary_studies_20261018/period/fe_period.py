"""S1: fixed-evaluation uniform-selection period study (see PROTOCOL.md).

The released engine is imported read-only. Only the per-step "uniform or
scored" rule differs between p4 / p8 / always; r4 is the released de_random4.

  python fe_period.py --release REL --inputs DIR --output OUT --lanes N [--limit K] [--only-replication]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

PROTOCOL = 's1_fixed_evaluation_period_20261018_v1'
ARMS = {'p4': 4, 'p8': 8, 'always': 0, 'r4': None}
SEEDS = (909302001, 909302002, 909302003)
ENDPOINTS = (2000, 5000, 10000)
DATASET = 'study30_synthetic_v1'
STRATA = ('small_low', 'small_high', 'medium_low', 'medium_high')
REGIMES = ('a1_t015', 'a1_t060', 'a3_t030', 'a5_t015', 'a5_t060')

_ENGINE_READY = False


def setup_engine(release: Path):
    global _ENGINE_READY, bc, online_core, s24, scalar, joint_proposal, fingerprint, np
    if _ENGINE_READY:
        return
    engine = release / 'code' / 'engine'
    for d in ('fixed_evaluation_confirmation_20260908',
              'baseline_calibration_and_transport_20260908',
              'equal_walltime_harness_engineering_20260909',
              'fixed_evaluation_component_ablation_20260909'):
        sys.path.insert(0, str(engine / d))
    import numpy as np  # noqa: F401
    import budget_core as bc  # noqa: F401
    import online_core  # noqa: F401
    import study24 as s24  # noqa: F401
    from surrogate import scalar, joint_proposal  # noqa: F401
    from lns_loop import fingerprint  # noqa: F401
    globals().update(np=np, bc=bc, online_core=online_core, s24=s24, scalar=scalar,
                     joint_proposal=joint_proposal, fingerprint=fingerprint)
    _ENGINE_READY = True


def make_period_class(period):
    base = online_core.Decomposition

    class PhysicalPeriod(base):
        """Copy of released de_proxy advance(); only the uniform-step rule is parameterized."""

        def advance(self):
            k = self.step
            i = k % len(self.weights)
            weight = self.weights[i]
            t = time.perf_counter()
            rng = np.random.RandomState(online_core.stream_seed(self.seed, k, self.namespace + '/proposal'))
            select_rng = np.random.RandomState(online_core.stream_seed(self.seed, k, self.namespace + '/selection'))
            donor_index = int(rng.randint(len(self.representatives)))
            random_index = int(select_rng.randint(4))
            parent, donor = self.representatives[i], self.representatives[donor_index]
            children = [joint_proposal(parent, donor, rng, self.features) for _ in range(4)]
            self.times['proposal'] += time.perf_counter() - t
            explore = bool(period) and k % period == period - 1
            selected = random_index
            preds = None
            t = time.perf_counter()
            if not explore:
                xp = [self.features(c) for c in children]
                preds = np.asarray([b for _, b in xp])
                selected = int(np.argmin(scalar(preds, weight, self.scale)))
            self.times['prediction_selection'] += time.perf_counter() - t
            assert all(c._objectives is None and c._schedule is None for c in children)
            row = dict(step=k, direction=i, donor_index=donor_index, brood_size=4,
                       random_index=random_index, explore=explore, selected=selected,
                       candidate_fingerprints=[fingerprint(c) for c in children],
                       predictions=preds.tolist() if preds is not None else None)
            t = time.perf_counter()
            child = children[selected]
            y = child.objectives.to_array()
            self.times['evaluation'] += time.perf_counter() - t
            child._schedule = None
            assert all(c._objectives is None and c._schedule is None
                       for j, c in enumerate(children) if j != selected)
            t = time.perf_counter()
            neighbors = self.neighbors[i]
            old = scalar(self.values[neighbors], self.weights[neighbors], self.scale)
            new = scalar(np.broadcast_to(y, (len(neighbors), 3)), self.weights[neighbors], self.scale)
            replace = neighbors[np.flatnonzero(new < old)[:2]]
            for j in replace:
                self.representatives[int(j)] = child
                self.values[int(j)] = y
            self.times['representative_update'] += time.perf_counter() - t
            self.step += 1
            row.update(objectives=y.tolist(), replacements=replace.tolist())
            return child, row

    return PhysicalPeriod


def run_cell(release: str, input_path: str, out_dir: str, name: str, seed: int, arm: str):
    setup_engine(Path(release))
    out = Path(out_dir) / 'runs' / f'{name}_{seed}_{arm}.json.gz'
    if out.exists():
        return json.loads(gzip.decompress(out.read_bytes()))['summary']
    inst = s24.restore_instance(json.loads(Path(input_path).read_text()))
    random.seed(seed)
    started = time.monotonic()
    if arm == 'r4':
        raw = s24.run_solver(inst, DATASET, name, seed, 'de_random4', ENDPOINTS[-1], ENDPOINTS)
    else:
        previous = bc.Decomposition
        cls = make_period_class(ARMS[arm])
        try:
            bc.Decomposition = cls
            raw = s24.run_solver(inst, DATASET, name, seed, 'de_proxy', ENDPOINTS[-1], ENDPOINTS)
        finally:
            bc.Decomposition = previous
    check = s24.verify(raw, inst)
    ref = np.asarray(raw['reference'])
    wide = np.asarray(raw['wide_reference'])
    endpoints = []
    for s in raw['snapshots']:
        endpoints.append(dict(evaluations=s['evaluations'],
                              hv_normalized=s['hv'] / float(np.prod(ref)),
                              hv_wide_normalized=s['hv_wide'] / float(np.prod(wide)),
                              archive_size=len(s['objectives'])))
    scored = sum(1 for e in raw['events'] if e.get('predictions') is not None) if arm != 'r4' else 0
    summary = dict(protocol=PROTOCOL, name=name, base_id=name.rsplit('_', 2)[0],
                   regime='_'.join(name.rsplit('_', 2)[1:]), seed=seed, arm=arm,
                   endpoints=endpoints, scored_steps=scored, steps=len(raw['events']),
                   search_seconds=raw['search_seconds'], wall_seconds=time.monotonic() - started,
                   verification=check.get('status', 'VERIFIED'),
                   input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest())
    payload = dict(summary=summary,
                   snapshots=[dict(evaluations=s['evaluations'], objectives=s['objectives'],
                                   genes=s['genes']) for s in raw['snapshots']],
                   reference=raw['reference'], wide_reference=raw['wide_reference'],
                   initial_objectives=raw['initial_objectives'])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')

    def enc(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    tmp.write_bytes(gzip.compress(json.dumps(payload, default=enc, allow_nan=False).encode(), 5, mtime=0))
    tmp.replace(out)
    return summary


def tasks(inputs: Path, arms):
    out = []
    for stratum in STRATA:
        for rep in range(4):
            bid = f'comp30_{stratum}_{rep:02}'
            for regime in REGIMES:
                name = f'{bid}_{regime}'
                path = inputs / f'{name}.json'
                assert path.exists(), path
                for seed in SEEDS:
                    for arm in arms:
                        out.append((str(path), name, seed, arm))
    # heavy (medium) cells first for better packing
    out.sort(key=lambda t: (0 if 'medium' in t[1] else 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lanes', type=int, default=19)
    p.add_argument('--limit', type=int)
    p.add_argument('--arms', default='p4,p8,always,r4')
    a = p.parse_args()
    arms = a.arms.split(',')
    assert set(arms) <= set(ARMS)
    a.output.mkdir(parents=True, exist_ok=True)
    todo = tasks(a.inputs, arms)
    if a.limit:
        todo = todo[:a.limit]
    log = (a.output / 'progress.jsonl').open('a')
    done = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.lanes) as ex:
        futs = {ex.submit(run_cell, str(a.release), *t[:1], str(a.output), *t[1:]): t for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
                done += 1
                log.write(json.dumps(dict(status='PASS', name=t[1], seed=t[2], arm=t[3],
                                          hv10k=s['endpoints'][-1]['hv_normalized'],
                                          seconds=s['wall_seconds'])) + '\n')
            except Exception:
                fail += 1
                log.write(json.dumps(dict(status='FAIL', name=t[1], seed=t[2], arm=t[3],
                                          error=traceback.format_exc()[-2000:])) + '\n')
            log.flush()
    (a.output / 'FINISH.json').write_text(json.dumps(dict(
        protocol=PROTOCOL, tasks=len(todo), passed=done, failed=fail,
        hours=(time.time() - t0) / 3600,
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2))
    print(json.dumps(dict(passed=done, failed=fail)))


if __name__ == '__main__':
    main()
