"""Second-round supplementary experiments (exp2, 2026-10-19). See PROTOCOL.md.

The released engine in <release>/code/engine is imported read-only. This module adds
  * new generated input panels (same generator as the reference panel, new seeds);
  * a K-candidate RBP family (K=4, period 0 reproduces the released Always-scored /
    RBP trajectory; K=4 with uniform choice reproduces Random-four; K=1 is Random-one);
  * parameterized NSGA-II and MOEA/D (defaults reproduce the 2026-10-18 S2 arms) with an
    optional analytic-preselection plug-in of pool size K;
  * three exact-evaluation modes: native, native plus fixed delay, and Monte Carlo
    evaluation under stochastic AGV travel times (sample average over R scenarios).
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True

PROTOCOL = 'exp2_20261019_v1'
DATASET = 'exp2_synthetic_v1'
STRATA = ('small_low', 'small_high', 'medium_low', 'medium_high')
REGIMES = {'a1_t015': (1, .15), 'a1_t060': (1, .60), 'a3_t030': (3, .30),
           'a5_t015': (5, .15), 'a5_t060': (5, .60)}
PANELS = {
    'test': dict(per_stratum=4, pseed=2026102001, eseed=2026102101,
                 seeds=(2026102201, 2026102202, 2026102203), regimes=tuple(REGIMES)),
    'tune': dict(per_stratum=2, pseed=2026102301, eseed=2026102401,
                 seeds=(2026102501, 2026102502), regimes=('a1_t060', 'a3_t030', 'a5_t015')),
    'eng': dict(per_stratum=1, pseed=2026102601, eseed=2026102701,
                seeds=(2026102801,), regimes=('a1_t060', 'a5_t015')),
}
STOCH_CV = 0.5            # coefficient of variation of each travel-time multiplier
TRAIN_SCENARIOS_MAX = 64  # in-search scenario sets are nested prefixes of 64 draws
TEST_SCENARIOS = 200      # independent out-of-sample scenarios per input
BATCH = 10                # archive commit every 10 exact evaluations

_READY = False
RELEASE = None


def setup(release):
    global _READY, RELEASE
    if _READY:
        return
    RELEASE = Path(release).resolve()
    engine = RELEASE / 'code' / 'engine'
    for d in ('fixed_evaluation_confirmation_20260908',
              'baseline_calibration_and_transport_20260908',
              'equal_walltime_harness_engineering_20260909'):
        sys.path.insert(0, str(engine / d))
    import numpy as np
    import compat
    import online_core
    import study24 as s24
    from surrogate import EncodingFeatures, scalar, directions, joint_proposal
    from src.problem.instance import FJSPAGVInstance
    globals().update(np=np, compat=compat, online_core=online_core, s24=s24,
                     EncodingFeatures=EncodingFeatures, scalar=scalar, directions=directions,
                     joint_proposal=joint_proposal, FJSPAGVInstance=FJSPAGVInstance)
    _READY = True


# ---------------------------------------------------------------- utilities
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def hash32(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def enc(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


# ---------------------------------------------------------------- input panels
def processing_base(stratum, pseed):
    """Released common30 draw sequence with an explicit processing seed."""
    scale, flex = stratum.split('_')
    rng = np.random.RandomState(pseed)
    jobs, machines = (10, 5) if scale == 'small' else (30, 10)
    counts = rng.randint(3, 8, jobs).tolist() if scale == 'small' else rng.randint(5, 11, jobs).tolist()
    inst = FJSPAGVInstance(jobs, machines, 3, counts)
    nchoice = max(1, int(np.ceil(machines * (.25 if flex == 'low' else .75))))
    for j, n in enumerate(counts):
        for o in range(n):
            choices = sorted(int(x) for x in rng.choice(machines, nchoice, replace=False))
            base = float(rng.randint(5, 51))
            inst.compatible_machines[j, o] = choices
            for m in choices:
                inst.processing_times[j, o, m] = float(max(1, round(base * rng.uniform(.8, 1.2))))
    return inst


def generate_panel(panel, out_dir):
    cfg = PANELS[panel]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, stratum in enumerate(STRATA):
        for rep in range(cfg['per_stratum']):
            idx = sid * cfg['per_stratum'] + rep
            pseed, eseed = cfg['pseed'] + idx, cfg['eseed'] + idx
            base = processing_base(stratum, pseed)
            bid = f'{panel}_{stratum}_{rep:02}'
            for regime in cfg['regimes']:
                vehicles, ratio = REGIMES[regime]
                obj = copy.deepcopy(base)
                obj.num_agv = vehicles
                obj.generate_agv_params(seed=eseed, transport_time_ratio=ratio)
                obj.extension_metadata.update(generator=PROTOCOL, base_id=bid, stratum=stratum,
                                              processing_seed=pseed, extension_seed=eseed, regime=regime)
                snap = json.loads(canonical(s24.snapshot_instance(obj)))
                s24.restore_instance(snap)  # round-trip assertion inside
                name = f'{bid}_{regime}'
                path = out_dir / f'{name}.json'
                data = canonical(snap)
                if path.exists():
                    assert path.read_bytes() == data, f'Existing input differs: {path}'
                else:
                    path.write_bytes(data)
                rows.append(dict(panel=panel, name=name, base_id=bid, stratum=stratum, regime=regime,
                                 vehicles=vehicles, transport_ratio=ratio, processing_seed=pseed,
                                 extension_seed=eseed, file=f'{panel}/{path.name}', sha256=sha(path),
                                 operations=int(sum(obj.num_operations))))
    return rows


# ---------------------------------------------------------------- stochastic travel model
def scenarios(n_ops, key, count):
    """Lognormal travel-time multipliers with mean 1 and CV STOCH_CV: (count, n_ops, 2) for (empty, loaded)."""
    rng = np.random.RandomState(hash32(key))
    s2 = float(np.log(1. + STOCH_CV ** 2))
    return rng.lognormal(-s2 / 2., np.sqrt(s2), size=(count, n_ops, 2))


class StochModel:
    """Released decoder/energy semantics with per-operation travel-time multipliers.

    With all multipliers equal to one, objectives() reproduces decoding.evaluate exactly
    (asserted by the self-test). Travel energy uses nominal travel times (multipliers model
    waiting/congestion delays, not slower driving); delays change timing, makespan and
    machine standby energy.
    """

    def __init__(self, inst):
        self.inst = inst
        pairs = [(i, j) for i in range(inst.num_jobs) for j in range(inst.num_operations[i])]
        self.n = len(pairs)
        self.offset = [0] * inst.num_jobs
        acc = 0
        for i in range(inst.num_jobs):
            self.offset[i] = acc
            acc += inst.num_operations[i]
        self.machines = [inst.get_compatible_machines(i, j) for i, j in pairs]
        self.M, self.V = inst.num_machines, inst.num_agv

    def plan(self, ch):
        """Deterministic part of decoding: per operation in OS order."""
        inst = self.inst
        rows = []
        op_count = {}
        op_machine = {}
        agv_loc = [0] * self.V
        last_job = [-1] * self.M
        setup_count = [0] * self.M
        load_e = [0.0] * self.V
        empty_e = [0.0] * self.V
        for t in range(len(ch.os)):
            i = int(ch.os[t])
            j = op_count.get(i, 0)
            op_count[i] = j + 1
            k = self.offset[i] + j
            machines = self.machines[k]
            u = machines[ch.ma[k] % len(machines)]
            l = int(ch.agv_assign[k]) % inst.num_agv
            speed = inst.speed_levels[int(ch.agv_speed[k]) % inst.num_speeds]
            u_prev = 0 if j == 0 else op_machine[(i, j - 1)] + 1
            u_dest = u + 1
            dist_empty = inst.get_distance(agv_loc[l], u_prev)
            t_empty = dist_empty / speed if speed > 0 else 0
            dist_tr = inst.get_distance(u_prev, u_dest)
            t_tr = dist_tr / speed if speed > 0 else 0
            setup = 0.0
            if last_job[u] >= 0 and last_job[u] != i:
                if inst.machine_setup_time is not None:
                    setup = inst.machine_setup_time[u]
                setup_count[u] += 1
            prev_k = -1 if j == 0 else self.offset[i] + j - 1
            rows.append((k, prev_k, u, l, t_empty, t_tr, setup, inst.get_processing_time(i, j, u), i))
            op_machine[(i, j)] = u
            agv_loc[l] = u_dest
            last_job[u] = i
            load_e[l] += inst.transport_energy(dist_tr, speed, loaded=True)
            empty_e[l] += inst.transport_energy(dist_empty, speed, loaded=False)
        agv_total = sum(load_e[l] + empty_e[l] for l in range(self.V))
        return rows, setup_count, agv_total

    def _energy_parts(self, setup_count):
        inst = self.inst
        out = []
        for u in range(self.M):
            st = inst.machine_setup_time[u] if inst.machine_setup_time is not None else 4.0
            sp = inst.machine_setup_power[u] if inst.machine_setup_power is not None else 2.5
            pp = inst.machine_proc_power[u] if inst.machine_proc_power is not None else 5.0
            ip = inst.machine_idle_power[u] if inst.machine_idle_power is not None else 1.0
            out.append((st, sp, pp, ip, sp * st * setup_count[u]))
        return out

    def one(self, rows, parts, setup_count, agv_total, mult):
        """One scenario; mult is an (n, 2) array or None for nominal times."""
        M, V = self.M, self.V
        mav = np.zeros(M)
        vav = np.zeros(V)
        end = [0.0] * self.n
        first = [None] * M
        last = [None] * M
        proc = [0] * M
        for k, prev_k, u, l, te, tl, setup, p, i in rows:
            if mult is not None:
                te = te * mult[k, 0]
                tl = tl * mult[k, 1]
            cargo = 0.0 if prev_k < 0 else end[prev_k]
            ts = max(cargo, vav[l] + te)
            tend = ts + tl
            s = max(tend, mav[u] + setup)
            e = s + p
            end[k] = e
            mav[u] = e
            vav[l] = tend
            proc[u] = proc[u] + (e - s)
            first[u] = s if first[u] is None else min(first[u], s)
            last[u] = e if last[u] is None else max(last[u], e)
        machine_total = 0
        for u in range(M):
            st, sp, pp, ip, setup_energy = parts[u]
            if first[u] is None:
                total = pp * 0 + setup_energy + 0.0
            else:
                span = max(0.0, float(last[u]) - float(first[u]))
                idle = max(0.0, span - proc[u] - setup_count[u] * st)
                total = pp * proc[u] + setup_energy + ip * idle
            machine_total = machine_total + total
        loads = np.asarray([float(x) for x in proc])
        return max(end), float(machine_total + agv_total), float(loads.max() - loads.min())

    def objectives(self, ch, mults):
        rows, setup_count, agv_total = self.plan(ch)
        parts = self._energy_parts(setup_count)
        vals = np.asarray([self.one(rows, parts, setup_count, agv_total, m) for m in mults])
        y = vals.mean(axis=0)
        return compat.decoding.Objectives(makespan=float(y[0]), total_energy=float(y[1]),
                                          workload_balance=float(y[2]))

    def batch(self, ch, mults):
        """Vectorized over scenarios (S, n, 2); returns (S, 3). Same per-scenario arithmetic."""
        rows, setup_count, agv_total = self.plan(ch)
        parts = self._energy_parts(setup_count)
        S = mults.shape[0]
        M, V = self.M, self.V
        mav = np.zeros((M, S))
        vav = np.zeros((V, S))
        end = np.zeros((self.n, S))
        first = np.full((M, S), np.inf)
        last = np.full((M, S), -np.inf)
        used = [False] * M
        proc = np.zeros((M, S))
        for k, prev_k, u, l, te, tl, setup, p, i in rows:
            te_v = te * mults[:, k, 0]
            tl_v = tl * mults[:, k, 1]
            cargo = np.zeros(S) if prev_k < 0 else end[prev_k]
            ts = np.maximum(cargo, vav[l] + te_v)
            tend = ts + tl_v
            s = np.maximum(tend, mav[u] + setup)
            e = s + p
            end[k] = e
            mav[u] = e
            vav[l] = tend
            proc[u] = proc[u] + (e - s)
            first[u] = np.minimum(first[u], s)
            last[u] = np.maximum(last[u], e)
            used[u] = True
        machine_total = np.zeros(S)
        for u in range(M):
            st, sp, pp, ip, setup_energy = parts[u]
            if not used[u]:
                total = pp * 0 + setup_energy + 0.0
            else:
                span = np.maximum(0.0, last[u] - first[u])
                idle = np.maximum(0.0, span - proc[u] - setup_count[u] * st)
                total = pp * proc[u] + setup_energy + ip * idle
            machine_total = machine_total + total
        out = np.empty((S, 3))
        out[:, 0] = end.max(axis=0)
        out[:, 1] = machine_total + agv_total
        out[:, 2] = proc.max(axis=0) - proc.min(axis=0)
        return out


# ---------------------------------------------------------------- exact-evaluation modes
class Evaluator:
    """Patches the released decoding.evaluate for one run; counts calls and their time."""

    def __init__(self, inst, spec, name, seed):
        self.spec = dict(spec)
        self.kind = spec['kind']
        assert self.kind in ('native', 'delay', 'stoch')
        self.count = 0
        self.seconds = 0.0
        self.inst = inst
        if self.kind == 'stoch':
            self.model = StochModel(inst)
            self.train = scenarios(self.model.n, f'{PROTOCOL}/train/{name}/{seed}', TRAIN_SCENARIOS_MAX)[:spec['R']]

    def __enter__(self):
        self.original = compat.decoding.evaluate
        orig, perf = self.original, time.perf_counter
        if self.kind == 'native':
            def ev(ch, *a, **kw):
                t = perf()
                self.count += 1
                r = orig(ch, *a, **kw)
                self.seconds += perf() - t
                return r
        elif self.kind == 'delay':
            delay = self.spec['d_ms'] / 1000.0

            def ev(ch, *a, **kw):
                t = perf()
                self.count += 1
                r = orig(ch, *a, **kw)
                time.sleep(delay)
                self.seconds += perf() - t
                return r
        else:
            model, train = self.model, self.train

            def ev(ch, *a, **kw):
                t = perf()
                self.count += 1
                r = model.objectives(ch, train)
                self.seconds += perf() - t
                return r
        compat.decoding.evaluate = ev
        return self

    def __exit__(self, *args):
        compat.decoding.evaluate = self.original


# ---------------------------------------------------------------- algorithms
def make_rbpk(K, period):
    """period: 0 = score every step (RBP); P>0 = uniform every P-th step; None = always uniform."""
    base = online_core.Decomposition

    class RBPK(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.scored = 0
            self.score_seconds = 0.0

        def advance(self):
            k = self.step
            i = k % len(self.weights)
            weight = self.weights[i]
            rng = np.random.RandomState(online_core.stream_seed(self.seed, k, self.namespace + '/proposal'))
            select_rng = np.random.RandomState(online_core.stream_seed(self.seed, k, self.namespace + '/selection'))
            donor_index = int(rng.randint(len(self.representatives)))
            random_index = int(select_rng.randint(K))
            parent, donor = self.representatives[i], self.representatives[donor_index]
            children = [joint_proposal(parent, donor, rng, self.features) for _ in range(K)]
            uniform = K == 1 or period is None or (period > 0 and k % period == period - 1)
            selected = random_index
            if not uniform:
                t = time.perf_counter()
                preds = np.asarray([self.features(c)[1] for c in children])
                selected = int(np.argmin(scalar(preds, weight, self.scale)))
                self.score_seconds += time.perf_counter() - t
                self.scored += 1
            assert all(c._objectives is None and c._schedule is None for c in children)
            child = children[selected]
            y = child.objectives.to_array()
            child._schedule = None
            neighbors = self.neighbors[i]
            old = scalar(self.values[neighbors], self.weights[neighbors], self.scale)
            new = scalar(np.broadcast_to(y, (len(neighbors), 3)), self.weights[neighbors], self.scale)
            replace = neighbors[np.flatnonzero(new < old)[:2]]
            for j in replace:
                self.representatives[int(j)] = child
                self.values[int(j)] = y
            self.step += 1
            return child

    return RBPK


def nondominated_ranks(F):
    n = len(F)
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt
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


def environmental(F, size):
    ranks = nondominated_ranks(F)
    chosen = []
    for r in range(ranks.max() + 1):
        idx = np.flatnonzero(ranks == r)
        if len(chosen) + len(idx) <= size:
            chosen.extend(idx.tolist())
        else:
            cd = crowding(F[idx])
            chosen.extend(idx[np.argsort(-cd, kind='stable')][:size - len(chosen)].tolist())
            break
    return chosen


def variation_std(p1, p2, rng, choices, pc, pm):
    """S2 operators: POX on OS + uniform assignment crossover (prob pc), swap/insert, resampling pm/N."""
    c = p1.copy()
    inst = c.instance
    n = len(c.os)
    if rng.random() < pc and inst.num_jobs > 1:
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
    prob = pm / n
    for k in np.flatnonzero(rng.random(n) < prob):
        c.ma[k] = rng.randint(choices[k])
    for k in np.flatnonzero(rng.random(n) < prob):
        c.agv_assign[k] = rng.randint(inst.num_agv)
    for k in np.flatnonzero(rng.random(n) < prob):
        c.agv_speed[k] = rng.randint(inst.num_speeds)
    c.invalidate()
    return c


class _Host:
    def _setup_common(self, population, values, op, pc, pm, K):
        inst = population[0].instance
        self.inst = inst
        self.scale = np.maximum(values.max(axis=0), 1.)
        self.features = EncodingFeatures(inst, self.scale)
        self.choices = np.array([len(inst.get_compatible_machines(i, j))
                                 for i in range(inst.num_jobs) for j in range(inst.num_operations[i])])
        self.op, self.pc, self.pm, self.K = op, pc, pm, K
        self.scored = 0
        self.score_seconds = 0.0

    def _vary(self, p1, p2):
        if self.op == 'std':
            return variation_std(p1, p2, self.rng, self.choices, self.pc, self.pm)
        c = joint_proposal(p1, p2, self.rng, self.features)
        c.invalidate()
        return c

    def _pick(self, cands, weight):
        t = time.perf_counter()
        proxies = np.asarray([self.features(c)[1] for c in cands])
        child = cands[int(np.argmin(scalar(proxies, weight, self.scale)))]
        self.score_seconds += time.perf_counter() - t
        self.scored += 1
        return child


class NSGA2X(_Host):
    def __init__(self, population, values, seed, pop=100, pc=.9, pm=1., op='std', K=1):
        self.rng = np.random.RandomState(seed ^ 0x5A5A)
        self._setup_common(population, values, op, pc, pm, K)
        self.dirs = directions()
        self.kdir = 0
        self.POP = pop
        population, values = list(population), values.copy()
        if pop < len(population):
            keep = environmental(values, pop)
            population, values = [population[i] for i in keep], values[keep]
        self.pop, self.F = population, values
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
        if self.K > 1:
            cands = [self._vary(p1, p2) for _ in range(self.K)]
            weight = self.dirs[self.kdir % len(self.dirs)]
            self.kdir += 1
            child = self._pick(cands, weight)
        else:
            child = self._vary(p1, p2)
        y = child.objectives.to_array()
        child._schedule = None
        self.off.append(child)
        self.offF.append(y)
        if len(self.off) == self.POP:
            allp = self.pop + self.off
            allF = np.vstack([self.F, np.asarray(self.offF)])
            chosen = environmental(allF, self.POP)
            self.pop = [allp[i] for i in chosen]
            self.F = allF[chosen]
            self.off, self.offF = [], []
            self._rank()
        return child


class MOEADX(_Host):
    def __init__(self, population, values, seed, H=12, T=10, delta=.9, nr=2, op='std', pc=.9, pm=1., K=1):
        from compat import generate_reference_points
        self.rng = np.random.RandomState(seed ^ 0xA5A5)
        self._setup_common(population, values, op, pc, pm, K)
        self.W = np.maximum(generate_reference_points(3, H), 1e-6)
        self.z = values.min(axis=0).copy()
        g = self._g(values[None, :, :], self.W[:, None, :])
        ids = np.argmin(g, axis=1)
        self.X = [population[i] for i in ids]
        self.F = values[ids].copy()
        d = ((self.W[:, None] - self.W[None, :]) ** 2).sum(axis=2)
        self.B = np.argsort(d, axis=1, kind='stable')[:, :T]
        self.delta, self.nr = delta, nr
        self.k = 0

    def _g(self, F, W):
        return np.max(W * np.abs(F - self.z) / self.scale, axis=-1)

    def step(self):
        i = self.k % len(self.W)
        self.k += 1
        pool = self.B[i] if self.rng.random() < self.delta else np.arange(len(self.W))
        a, b = self.rng.choice(pool, 2, replace=False)
        if self.K > 1:
            cands = [self._vary(self.X[a], self.X[b]) for _ in range(self.K)]
            child = self._pick(cands, self.W[i])
        else:
            child = self._vary(self.X[a], self.X[b])
        y = child.objectives.to_array()
        child._schedule = None
        self.z = np.minimum(self.z, y)
        replaced = 0
        for j in self.rng.permutation(pool):
            if replaced >= self.nr:
                break
            if self._g(y, self.W[j]) < self._g(self.F[j], self.W[j]):
                self.X[j], self.F[j] = child, y
                replaced += 1
        return child


class RBPHost:
    def __init__(self, population, seed, name, K, period):
        cls = make_rbpk(K, period)
        self.ds = cls(population, seed, 'de_proxy', online_core.run_key(DATASET, name, seed))

    def step(self):
        return self.ds.advance()

    @property
    def scored(self):
        return self.ds.scored

    @property
    def score_seconds(self):
        return self.ds.score_seconds


def build(algo, population, values, seed, name):
    fam = algo['family']
    p = dict(algo.get('params', {}))
    if fam == 'rbp':
        period = p.get('period', 0)
        return RBPHost(population, seed, name, p.get('K', 4), None if period == 'uniform' else period)
    if fam == 'nsga2':
        return NSGA2X(population, values, seed, **p)
    if fam == 'moead':
        return MOEADX(population, values, seed, **p)
    raise ValueError(fam)


# ---------------------------------------------------------------- one run
def genes(c):
    return {k: getattr(c, k).tolist() for k in ('os', 'ma', 'agv_assign', 'agv_speed')}


def chromosome(inst, g):
    return compat.Chromosome(inst, **{k: np.asarray(v, dtype=int) for k, v in g.items()})


def hv_norm(F, ref):
    if len(F) == 0:
        return 0.0
    return compat.fixed_reference_hv(np.asarray(F, float), ref) / float(np.prod(ref))


def run_cell(task, release, out_root, code_sha=None, formal=False, dataset=None):
    setup(release)
    global DATASET
    if dataset:
        DATASET = dataset
    out = Path(out_root) / 'runs' / f"{task['id']}.json.gz"
    if out.exists():
        return json.loads(gzip.decompress(out.read_bytes()))['summary']
    in_path = Path(task['input'])
    assert sha(in_path) == task['input_sha256'], 'Input hash mismatch'
    inst = s24.restore_instance(json.loads(in_path.read_text()))
    seed, name, mode = task['seed'], task['name'], task['mode']
    endpoints = list(task['endpoints'])
    random.seed(seed)
    np.random.seed(seed)
    ev = Evaluator(inst, task['evaluator'], name, seed)
    commits = []
    started = time.monotonic()
    cpu0 = time.process_time()
    with ev:
        population = compat.initialize_population(
            inst, 100, np.random.RandomState(compat.legacy_initialization_seed(seed)))
        values = compat.objective_array(population)
        archive = compat.update_archive([], population)
        for c in population:
            c._schedule = None
        initial_genes = [genes(c) for c in population]
        alg = build(task['algo'], population, values, seed, name)
        assert ev.count == 100
        commits.append((time.monotonic() - started, ev.count, archive))
        limit = endpoints[-1]
        while True:
            if mode == 'eval' and ev.count >= limit:
                break
            if mode == 'time' and time.monotonic() - started >= limit:
                break
            children = [alg.step() for _ in range(BATCH)]
            archive = compat.update_archive(archive, children)
            commits.append((time.monotonic() - started, ev.count, archive))
        search_seconds = time.monotonic() - started
        cpu_seconds = time.process_time() - cpu0
        evaluations = ev.count
        eval_seconds = ev.seconds
    assert evaluations == 100 + BATCH * (len(commits) - 1)
    reference = values.max(axis=0) * 1.1
    wide = values.max(axis=0) * 1.5
    snaps = []
    for at in endpoints:
        if mode == 'eval':
            c = next(c for c in commits if c[1] == at)
        else:
            before = [c for c in commits if c[0] <= at]
            assert before, f'No archive commit before the {at} s endpoint'
            c = before[-1]
        snaps.append((at, c))
    stoch = ev.kind == 'stoch'
    if stoch:
        model = ev.model
        test = scenarios(model.n, f'{PROTOCOL}/test/{name}', TEST_SCENARIOS)
        init_test = np.asarray([model.batch(chromosome(inst, g), test).mean(axis=0) for g in initial_genes])
        ref_test, wide_test = init_test.max(axis=0) * 1.1, init_test.max(axis=0) * 1.5
    endpoint_rows, saved = [], []
    for at, (el, evals, arch) in snaps:
        F = compat.objective_array(arch)
        row = dict(endpoint=at, commit_seconds=el, evaluations=evals, archive_size=len(arch),
                   hv_normalized=hv_norm(F, reference), hv_wide_normalized=hv_norm(F, wide),
                   objective_minima=F.min(axis=0).tolist())
        g = [genes(c) for c in arch]
        item = dict(endpoint=at, objectives=F.tolist(), genes=g)
        if stoch:
            Ft = np.asarray([model.batch(chromosome(inst, x), test).mean(axis=0) for x in g])
            row.update(hv_test=hv_norm(Ft, ref_test), hv_test_wide=hv_norm(Ft, wide_test),
                       test_objective_minima=Ft.min(axis=0).tolist())
            item['test_objectives'] = Ft.tolist()
        endpoint_rows.append(row)
        saved.append(item)
    # independent re-evaluation of the final saved front (delay removed; same scenarios)
    check_spec = dict(task['evaluator'], kind='native') if ev.kind == 'delay' else task['evaluator']
    with Evaluator(inst, check_spec, name, seed) as chk:
        redo = np.asarray([chromosome(inst, g).objectives.to_array() for g in saved[-1]['genes']])
    assert chk.count == len(saved[-1]['genes'])
    assert np.array_equal(redo, np.asarray(saved[-1]['objectives'])), 'Saved front does not re-evaluate'
    summary = dict(protocol=PROTOCOL, study=task['study'], id=task['id'], arm=task['arm'], panel=task['panel'],
                   name=name, base_id=task['base_id'], stratum=task['stratum'], regime=task['regime'], seed=seed,
                   algo=task['algo'], evaluator=task['evaluator'], mode=mode, endpoints=endpoint_rows,
                   evaluations=evaluations, search_seconds=search_seconds, cpu_seconds=cpu_seconds,
                   cpu_ratio=cpu_seconds / max(search_seconds, 1e-9), eval_seconds=eval_seconds,
                   score_seconds=getattr(alg, 'score_seconds', 0.0), scored_steps=getattr(alg, 'scored', 0),
                   input_sha256=task['input_sha256'], code_sha256=code_sha, formal=bool(formal),
                   dataset=DATASET, verification='REEVALUATED', finished_utc=utc())
    payload = dict(summary=summary, snapshots=saved, reference=reference.tolist(), wide_reference=wide.tolist(),
                   initial_objectives=values.tolist())
    if stoch:
        payload.update(test_reference=ref_test.tolist(), test_wide_reference=wide_test.tolist(),
                       initial_test_objectives=init_test.tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_bytes(gzip.compress(json.dumps(payload, default=enc, allow_nan=False).encode(), 5, mtime=0))
    tmp.replace(out)
    return summary
