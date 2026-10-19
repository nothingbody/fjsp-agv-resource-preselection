"""S2: standard NSGA-II / MOEA/D comparators and an NSGA-II preselection plug-in.

All arms share the released engine's decoder, initial population (same seed),
external archive (update_archive, cap 100) and fixed HV reference, and commit the
archive every 10 exact evaluations. See PROTOCOL.md.

  python s2_moea.py --release REL --inputs DIR --output OUT --mode eval|time --arms a,b --lanes N
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

PROTOCOL = 's2_standard_moea_20261018_v1'
ALL_ARMS = ('nsga2', 'moead', 'nsga2_pre', 'p4', 'always')
SEEDS = (909302001, 909302002, 909302003)
DATASET = 'study30_synthetic_v1'
STRATA = ('small_low', 'small_high', 'medium_low', 'medium_high')
REGIMES = ('a1_t015', 'a1_t060', 'a3_t030', 'a5_t015', 'a5_t060')
EVAL_ENDPOINTS = (2000, 5000, 10000)
TIME_ENDPOINTS = (60.0, 120.0)
POP = 100
PC = 0.9
MOEAD_H, MOEAD_T, MOEAD_DELTA, MOEAD_NR = 12, 10, 0.9, 2
POOL = 4
if os.environ.get('S2_SMOKE'):  # engineering smoke test only, never formal
    EVAL_ENDPOINTS, TIME_ENDPOINTS = (300, 500, 1000), (3.0, 6.0)

_READY = False


def setup(release: Path):
    global _READY
    if _READY:
        return
    engine = release / 'code' / 'engine'
    for d in ('fixed_evaluation_confirmation_20260908',
              'baseline_calibration_and_transport_20260908',
              'equal_walltime_harness_engineering_20260909'):
        sys.path.insert(0, str(engine / d))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import compat
    import online_core
    import study24 as s24
    from surrogate import EncodingFeatures, scalar, directions
    globals().update(np=np, compat=compat, online_core=online_core, s24=s24,
                     EncodingFeatures=EncodingFeatures, scalar=scalar, directions=directions)
    _READY = True


# ---------------------------------------------------------------- operators
def variation(p1, p2, rng, choices):
    """One child: POX on OS + uniform assignment crossover (prob PC), then mutation."""
    c = p1.copy()
    inst = c.instance
    n = len(c.os)
    if rng.random() < PC and inst.num_jobs > 1:
        subset = rng.choice(inst.num_jobs, rng.randint(1, inst.num_jobs), replace=False)
        keep = np.isin(c.os, subset)
        c.os[~keep] = p2.os[~np.isin(p2.os, subset)]
        mask = rng.random(n) < 0.5
        for fam in ('ma', 'agv_assign', 'agv_speed'):
            getattr(c, fam)[mask] = getattr(p2, fam)[mask]
    a, b = (int(x) for x in rng.choice(n, 2, replace=False))
    if rng.random() < 0.5:
        c.os[a], c.os[b] = c.os[b], c.os[a]
    else:
        tok = c.os[a]
        c.os = np.insert(np.delete(c.os, a), b, tok)
    pm = 1.0 / n
    for k in np.flatnonzero(rng.random(n) < pm):
        c.ma[k] = rng.randint(choices[k])
    for k in np.flatnonzero(rng.random(n) < pm):
        c.agv_assign[k] = rng.randint(inst.num_agv)
    for k in np.flatnonzero(rng.random(n) < pm):
        c.agv_speed[k] = rng.randint(inst.num_speeds)
    c.invalidate()
    return c


def nondominated_ranks(F):
    n = len(F)
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt  # dom[i,j]: i dominates j
    count = dom.sum(axis=0)
    ranks = np.full(n, -1)
    front = np.flatnonzero(count == 0)
    r = 0
    while len(front):
        ranks[front] = r
        count = count - dom[front].sum(axis=0)
        count[ranks >= 0] = -1
        front = np.flatnonzero(count == 0)
        r += 1
    return ranks


def crowding(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j], kind='stable')
        span = F[o[-1], j] - F[o[0], j]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0:
            d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return d


class Preselect:
    """Always-scored analytic preselection from POOL variations (plug-in)."""

    def __init__(self, inst, scale):
        self.features = EncodingFeatures(inst, scale)
        self.scale = scale
        self.dirs = directions()
        self.k = 0

    def pick(self, cands, weight=None):
        if weight is None:
            weight = self.dirs[self.k % len(self.dirs)]
        self.k += 1
        proxies = np.asarray([self.features(c)[1] for c in cands])
        return cands[int(np.argmin(scalar(proxies, weight, self.scale)))]


class NSGA2:
    def __init__(self, population, values, seed, pre=False):
        self.pop = list(population)
        self.F = values.copy()
        self.rng = np.random.RandomState(seed ^ 0x5A5A)
        inst = population[0].instance
        self.choices = np.array([len(inst.get_compatible_machines(i, j))
                                 for i in range(inst.num_jobs) for j in range(inst.num_operations[i])])
        self.pre = Preselect(inst, np.maximum(values.max(axis=0), 1.)) if pre else None
        self.off, self.offF = [], []
        self._rank()

    def _rank(self):
        self.rank = nondominated_ranks(self.F)
        self.crowd = np.zeros(len(self.F))
        for r in np.unique(self.rank):
            idx = np.flatnonzero(self.rank == r)
            self.crowd[idx] = crowding(self.F[idx])

    def _tour(self):
        a, b = self.rng.randint(len(self.pop), size=2)
        if self.rank[a] != self.rank[b]:
            return a if self.rank[a] < self.rank[b] else b
        return a if self.crowd[a] >= self.crowd[b] else b

    def step(self):
        p1, p2 = self.pop[self._tour()], self.pop[self._tour()]
        if self.pre:
            child = self.pre.pick([variation(p1, p2, self.rng, self.choices) for _ in range(POOL)])
        else:
            child = variation(p1, p2, self.rng, self.choices)
        y = child.objectives.to_array()
        child._schedule = None
        self.off.append(child)
        self.offF.append(y)
        if len(self.off) == POP:
            allp = self.pop + self.off
            allF = np.vstack([self.F, np.asarray(self.offF)])
            ranks = nondominated_ranks(allF)
            chosen = []
            for r in range(ranks.max() + 1):
                idx = np.flatnonzero(ranks == r)
                if len(chosen) + len(idx) <= POP:
                    chosen.extend(idx.tolist())
                else:
                    cd = crowding(allF[idx])
                    chosen.extend(idx[np.argsort(-cd, kind='stable')][:POP - len(chosen)].tolist())
                    break
            self.pop = [allp[i] for i in chosen]
            self.F = allF[chosen]
            self.off, self.offF = [], []
            self._rank()
        return child


class MOEAD:
    """MOEA/D with Tchebycheff, fixed initial scales, updated ideal point."""

    def __init__(self, population, values, seed):
        from compat import generate_reference_points
        self.rng = np.random.RandomState(seed ^ 0xA5A5)
        self.W = np.maximum(generate_reference_points(3, MOEAD_H), 1e-6)
        self.scale = np.maximum(values.max(axis=0), 1.)
        self.z = values.min(axis=0).copy()
        inst = population[0].instance
        self.choices = np.array([len(inst.get_compatible_machines(i, j))
                                 for i in range(inst.num_jobs) for j in range(inst.num_operations[i])])
        g = self._g(values[None, :, :], self.W[:, None, :])  # (nw, npop)
        ids = np.argmin(g, axis=1)
        self.X = [population[i] for i in ids]
        self.F = values[ids].copy()
        d = ((self.W[:, None] - self.W[None, :]) ** 2).sum(axis=2)
        self.B = np.argsort(d, axis=1, kind='stable')[:, :MOEAD_T]
        self.k = 0

    def _g(self, F, W):
        return np.max(W * np.abs(F - self.z) / self.scale, axis=-1)

    def step(self):
        i = self.k % len(self.W)
        self.k += 1
        pool = self.B[i] if self.rng.random() < MOEAD_DELTA else np.arange(len(self.W))
        a, b = self.rng.choice(pool, 2, replace=False)
        child = variation(self.X[a], self.X[b], self.rng, self.choices)
        y = child.objectives.to_array()
        child._schedule = None
        self.z = np.minimum(self.z, y)
        replaced = 0
        for j in self.rng.permutation(pool):
            if replaced >= MOEAD_NR:
                break
            if self._g(y, self.W[j]) < self._g(self.F[j], self.W[j]):
                self.X[j], self.F[j] = child, y
                replaced += 1
        return child


class PeriodWrap:
    def __init__(self, population, seed, name, period):
        import budget_core as bc
        sys.path.insert(0, os.environ.get('S1_DIR', str(Path(__file__).resolve().parent.parent / 's1')))
        from fe_period import make_period_class, setup_engine
        setup_engine(RELEASE)
        cls = make_period_class(period)
        self.ds = cls(population, seed, 'de_proxy', bc.run_key(DATASET, name, seed))

    def step(self):
        return self.ds.advance()[0]


# ---------------------------------------------------------------- driver
RELEASE = None


def run_cell(release, input_path, out_dir, name, seed, arm, mode):
    global RELEASE
    RELEASE = Path(release)
    setup(RELEASE)
    out = Path(out_dir) / 'runs' / f'{name}_{seed}_{arm}.json.gz'
    if out.exists():
        return json.loads(gzip.decompress(out.read_bytes()))['summary']
    inst = s24.restore_instance(json.loads(Path(input_path).read_text()))
    random.seed(seed)
    np.random.seed(seed)
    started = time.monotonic()
    cpu0 = time.process_time()
    commits = []  # (elapsed, evaluations, archive)
    with compat.CountCalls() as count:
        population = compat.initialize_population(
            inst, 100, np.random.RandomState(compat.legacy_initialization_seed(seed)))
        values = compat.objective_array(population)
        archive = compat.update_archive([], population)
        for c in population:
            c._schedule = None
        if arm == 'nsga2':
            alg = NSGA2(population, values, seed)
        elif arm == 'nsga2_pre':
            alg = NSGA2(population, values, seed, pre=True)
        elif arm == 'moead':
            alg = MOEAD(population, values, seed)
        else:
            alg = PeriodWrap(population, seed, name, {'p4': 4, 'always': 0}[arm])
        reference = values.max(axis=0) * 1.1
        wide = values.max(axis=0) * 1.5
        commits.append((time.monotonic() - started, count.evaluations, archive))
        limit_e = EVAL_ENDPOINTS[-1] if mode == 'eval' else None
        limit_t = TIME_ENDPOINTS[-1] if mode == 'time' else None
        while True:
            if limit_e is not None and count.evaluations >= limit_e:
                break
            if limit_t is not None and time.monotonic() - started >= limit_t:
                break
            children = [alg.step() for _ in range(10)]
            archive = compat.update_archive(archive, children)
            commits.append((time.monotonic() - started, count.evaluations, archive))
        search_seconds = time.monotonic() - started
        cpu_seconds = time.process_time() - cpu0
        evaluations = count.evaluations
    snaps = []
    if mode == 'eval':
        for e in EVAL_ENDPOINTS:
            c = next(c for c in commits if c[1] == e)
            snaps.append((e, c))
    else:
        for t in TIME_ENDPOINTS:
            c = [c for c in commits if c[0] <= t][-1]
            snaps.append((t, c))
    endpoints, saved = [], []
    for at, (el, ev, arch) in snaps:
        F = compat.objective_array(arch)
        endpoints.append(dict(endpoint=at, commit_seconds=el, evaluations=ev,
                              hv_normalized=compat.fixed_reference_hv(F, reference) / float(np.prod(reference)),
                              hv_wide_normalized=compat.fixed_reference_hv(F, wide) / float(np.prod(wide)),
                              archive_size=len(arch)))
        saved.append(dict(endpoint=at, objectives=F.tolist(),
                          genes=[{k: getattr(c, k).tolist() for k in ('os', 'ma', 'agv_assign', 'agv_speed')}
                                 for c in arch]))
    # independent re-decode of the final saved front
    last = saved[-1]
    redo = np.asarray([compat.Chromosome(inst, **{k: np.asarray(v, dtype=int) for k, v in g.items()})
                       .objectives.to_array() for g in last['genes']])
    assert np.array_equal(redo, np.asarray(last['objectives'])), 'Saved front does not re-decode'
    summary = dict(protocol=PROTOCOL, mode=mode, name=name, base_id=name.rsplit('_', 2)[0],
                   regime='_'.join(name.rsplit('_', 2)[1:]), seed=seed, arm=arm,
                   endpoints=endpoints, evaluations=evaluations, search_seconds=search_seconds,
                   cpu_seconds=cpu_seconds, cpu_ratio=cpu_seconds / max(search_seconds, 1e-9),
                   input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                   verification='REDECODED')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_bytes(gzip.compress(json.dumps(dict(summary=summary, snapshots=saved,
                                                  reference=reference.tolist(),
                                                  wide_reference=wide.tolist())).encode(), 5, mtime=0))
    tmp.replace(out)
    return summary


def tasks(inputs: Path, arms, mode):
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
    if mode == 'eval':
        out.sort(key=lambda t: 0 if 'medium' in t[1] else 1)
    else:
        random.Random(20261018).shuffle(out)  # mix arms across concurrent lanes
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('eval', 'time'), required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--lanes', type=int, default=19)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    arms = a.arms.split(',')
    assert set(arms) <= set(ALL_ARMS)
    a.output.mkdir(parents=True, exist_ok=True)
    todo = tasks(a.inputs, arms, a.mode)[:a.limit]
    log = (a.output / 'progress.jsonl').open('a')
    done = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.lanes) as ex:
        futs = {ex.submit(run_cell, str(a.release.resolve()), t[0], str(a.output), t[1], t[2], t[3], a.mode): t
                for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
                done += 1
                log.write(json.dumps(dict(status='PASS', name=t[1], seed=t[2], arm=t[3],
                                          hv=s['endpoints'][-1]['hv_normalized'],
                                          evaluations=s['evaluations'], seconds=s['search_seconds'])) + '\n')
            except Exception:
                fail += 1
                log.write(json.dumps(dict(status='FAIL', name=t[1], seed=t[2], arm=t[3],
                                          error=traceback.format_exc()[-2000:])) + '\n')
            log.flush()
    (a.output / f'FINISH_{a.mode}_{"-".join(arms)}.json').write_text(json.dumps(dict(
        protocol=PROTOCOL, mode=a.mode, arms=arms, tasks=len(todo), passed=done, failed=fail,
        hours=(time.time() - t0) / 3600,
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2))
    print(json.dumps(dict(passed=done, failed=fail)))


if __name__ == '__main__':
    main()
