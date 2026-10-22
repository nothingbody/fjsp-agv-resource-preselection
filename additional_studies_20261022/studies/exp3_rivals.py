"""exp3: re-implementations of two recent FJSP-AGV algorithms for comparison with RBP.

* DQNMA  - Zhang, Li & Gong (2024), Computers & Industrial Engineering 189:109917.
           SPEA2 + four critical-path/critical-block local-search operators chosen by a DQN.
* ILMPEA - Cheng, Meng, Zhang, Gao & Sang (2025), IEEE TEVC, DOI 10.1109/TEVC.2025.3540105.
           Makespan / TEC / Knowledge populations + behaviour-cloning operator choice.

Both are re-implemented from the papers (no original code is available) and adapted to the
released three-objective FJSP-AGV model (makespan, energy index, workload range; speed genes).
Every deviation from the papers is listed in PROTOCOL_exp3.md. Optional analytic preselection
(K > 1): wherever the algorithm creates one new solution for exact evaluation, K candidates are
created by the same operator from the same parents and the one with the smallest analytic scalar
score (Eq. 14 of the manuscript) is evaluated.

Each algorithm exposes step() -> one newly evaluated chromosome, so it runs in the same harness
as RBP (one exact evaluation per step, archive committed every 10 evaluations).
"""
from __future__ import annotations

import copy
import time

import exp2_core as C

EPS = 1e-9


def np_():
    return C.np


# ----------------------------------------------------------------------------- small numpy MLP
class MLP:
    """Fully connected ReLU network with a linear output layer and Adam."""

    def __init__(self, sizes, rng, lr):
        np = np_()
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), (a, b)))
            self.b.append(np.zeros(b))
        self.lr = lr
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]

    def forward(self, x):
        np = np_()
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return h, acts

    def backward(self, acts, grad_out):
        np = np_()
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        self._adam(gW + gb)

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        np = np_()
        self.t += 1
        params = self.W + self.b
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


# ----------------------------------------------------------------------------- shared helpers
class Problem:
    """Instance-level lookups shared by the operators."""

    def __init__(self, inst):
        np = np_()
        self.inst = inst
        self.pairs = [(i, j) for i in range(inst.num_jobs) for j in range(inst.num_operations[i])]
        self.n = len(self.pairs)
        self.offset = np.r_[0, np.cumsum(inst.num_operations)[:-1]].astype(int)
        self.machines = [inst.get_compatible_machines(i, j) for i, j in self.pairs]
        self.choices = np.array([len(m) for m in self.machines])
        self.ptime = [[inst.get_processing_time(i, j, m) for m in ms] for (i, j), ms in zip(self.pairs, self.machines)]
        self.V, self.S, self.J = inst.num_agv, inst.num_speeds, inst.num_jobs
        self.denom = np.concatenate([np.full(self.n, max(1, self.J - 1)), np.maximum(self.choices - 1, 1),
                                     np.full(self.n, max(1, self.V - 1)), np.full(self.n, max(1, self.S - 1))])

    def state(self, ch):
        np = np_()
        return np.concatenate([ch.os, ch.ma, ch.agv_assign, ch.agv_speed]).astype(float) / self.denom

    def random(self, rng):
        np = np_()
        inst = self.inst
        os_ = np.repeat(np.arange(self.J), inst.num_operations)
        rng.shuffle(os_)
        ma = np.array([rng.randint(c) for c in self.choices])
        return C.compat.Chromosome(inst, os_, ma, rng.randint(self.V, size=self.n), rng.randint(self.S, size=self.n))


def pox(o1, o2, rng, J):
    np = np_()
    subset = rng.choice(J, rng.randint(1, J), replace=False) if J > 1 else np.array([0])
    keep = np.isin(o1, subset)
    c = o1.copy()
    c[~keep] = o2[~np.isin(o2, subset)]
    return c


def uniform(a, b, rng):
    np = np_()
    mask = rng.random(len(a)) < 0.5
    c = a.copy()
    c[mask] = b[mask]
    return c


def make(inst, os_, ma, va, sp):
    return C.compat.Chromosome(inst, os_.copy(), ma.copy(), va.copy(), sp.copy())


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nd_filter(items, cap=100):
    """items: list of (chromosome, objective array). Unique nondominated, crowding-truncated."""
    np = np_()
    seen, uniq = set(), []
    for c, y in items:
        key = tuple(np.asarray(y).tolist())
        if key not in seen:
            seen.add(key)
            uniq.append((c, np.asarray(y)))
    if not uniq:
        return []
    F = np.asarray([y for _, y in uniq])
    ranks = C.nondominated_ranks(F) if hasattr(C, 'nondominated_ranks') else None
    front = [uniq[i] for i in np.flatnonzero(ranks == 0)]
    if len(front) > cap:
        Ff = np.asarray([y for _, y in front])
        cd = C.crowding(Ff)
        keep = np.argsort(-cd, kind='stable')[:cap]
        front = [front[i] for i in keep]
    return front


class Preselect:
    """Analytic preselection shared by the plug-in variants (score as in the RBP manuscript)."""

    def __init__(self, inst, scale, K, objectives):
        self.features = C.EncodingFeatures(inst, scale)
        self.scale = scale
        self.K = K
        self.dirs = C.directions()
        self.k = 0
        self.bi = objectives == 2
        self.seconds = 0.0
        self.count = 0

    def pick(self, factory):
        np = np_()
        if self.K <= 1:
            return factory()
        cands = [factory() for _ in range(self.K)]
        t = time.perf_counter()
        prox = np.asarray([self.features(c)[1] for c in cands])
        if self.bi:
            prox[:, 2] = 0.0
        w = self.dirs[self.k % len(self.dirs)]
        self.k += 1
        best = cands[int(np.argmin(C.scalar(prox, w, self.scale)))]
        self.seconds += time.perf_counter() - t
        self.count += 1
        return best


def evaluate(ch):
    y = ch.objectives.to_array()
    ch._schedule = None
    return y


# ----------------------------------------------------------------------------- critical path
def critical_path(ch, prob):
    """Critical chain of the decoded schedule as [('P'|'T', (i, j), resource)], start -> end.

    Predecessor of a processing task: its own delivery if the start equals delivery, otherwise the
    previous operation on the machine (plus setup). Predecessor of a transport task: the job's
    previous operation if loading waited for it, otherwise the previous trip of the same vehicle
    (plus empty travel). The schedule is recomputed (not an exact evaluation of a new solution).
    """
    inst = prob.inst
    sch = C.compat.decoding.decode(ch)
    prev_m, prev_v = {}, {}
    for u, lst in sch.machine_schedule.items():
        for a in range(1, len(lst)):
            prev_m[(lst[a][0], lst[a][1])] = lst[a - 1]
    for l, lst in sch.agv_schedule.items():
        for a in range(1, len(lst)):
            prev_v[(lst[a][0], lst[a][1])] = lst[a - 1]
    last = max(sch.op_end, key=lambda k: sch.op_end[k])
    path = []
    node = ('P', last)
    guard = 0
    while node is not None and guard < 4 * prob.n + 10:
        guard += 1
        kind, (i, j) = node
        if kind == 'P':
            u = sch.op_machine[(i, j)]
            path.append(('P', (i, j), u))
            s = sch.op_start[(i, j)]
            if abs(s - sch.transport_end[(i, j)]) <= EPS:
                node = ('T', (i, j))
                continue
            pm = prev_m.get((i, j))
            node = ('P', (pm[0], pm[1])) if pm is not None else ('T', (i, j))
        else:
            l = sch.transport_agv[(i, j)]
            path.append(('T', (i, j), l))
            ts = sch.transport_start[(i, j)]
            if j > 0 and abs(ts - sch.op_end[(i, j - 1)]) <= EPS:
                node = ('P', (i, j - 1))
                continue
            pv = prev_v.get((i, j))
            node = ('T', (pv[0], pv[1])) if pv is not None else (('P', (i, j - 1)) if j > 0 else None)
    path.reverse()
    return path


def blocks(path):
    out, cur = [], []
    for t in path:
        if cur and (t[0] != cur[-1][0] or t[2] != cur[-1][2]):
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def os_positions(os_):
    pos, cnt = {}, {}
    for p, job in enumerate(os_.tolist()):
        j = cnt.get(job, 0)
        cnt[job] = j + 1
        pos[(job, j)] = p
    return pos


def swap_tokens(ch, a, b):
    c = ch.copy()
    pos = os_positions(c.os)
    pa, pb = pos[a], pos[b]
    c.os[pa], c.os[pb] = c.os[pb], c.os[pa]
    c.invalidate()
    return c


# ----------------------------------------------------------------------------- DQNMA
class DQNMA:
    """Zhang et al. (2024) DQN-based memetic algorithm (SPEA2 + critical-path local search)."""

    N_ACTIONS = 4

    def __init__(self, population, values, seed, K=1, objectives=3, pop=100, pm=0.2, eps_greedy=0.95,
                 gamma=0.9, batch=32, target_every=7, lr=1e-3, pool=1024, train_after=50, archive_cap=100,
                 random_operator=False):
        np = np_()
        self.rng = np.random.RandomState(seed ^ 0x3E3E)
        self.inst = population[0].instance
        self.prob = Problem(self.inst)
        self.obj = objectives
        self.scale = np.maximum(values.max(axis=0), 1.)
        self.pre = Preselect(self.inst, self.scale, K, objectives)
        self.N, self.pm, self.eg, self.gamma = pop, pm, eps_greedy, gamma
        self.batch, self.target_every, self.pool_size, self.train_after = batch, target_every, pool, train_after
        self.cap = archive_cap
        self.random_operator = random_operator
        d = 4 * self.prob.n
        self.q = MLP([d, 64, 32, 16, self.N_ACTIONS], np.random.RandomState(seed ^ 0x51), lr)
        self.qt = MLP([d, 64, 32, 16, self.N_ACTIONS], np.random.RandomState(seed ^ 0x52), lr)
        self.qt.copy_from(self.q)
        self.E = []
        self.trainings = 0
        self.train_seconds = 0.0
        pop0 = list(zip(population, values))
        keep = C.environmental(values, pop) if pop < len(population) else list(range(len(population)))
        self.P = [pop0[i] for i in keep]
        self.A = nd_filter(pop0, self.cap)
        self.gen = self._run()

    @property
    def scored(self):
        return self.pre.count

    @property
    def score_seconds(self):
        return self.pre.seconds

    def step(self):
        return next(self.gen)

    # -- SPEA2 ---------------------------------------------------------------------------------
    def _fitness(self, F):
        np = np_()
        n = len(F)
        le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
        lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
        dom = le & lt
        S = dom.sum(axis=1)
        R = (dom * S[:, None]).sum(axis=0)
        Fn = (F - F.min(axis=0)) / np.maximum(F.max(axis=0) - F.min(axis=0), 1e-12)
        dist = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        k = max(1, int(np.sqrt(n)))
        sk = np.sort(dist, axis=1)[:, min(k, n - 1) - 1] if n > 1 else np.zeros(n)
        return R + 1.0 / (sk + 2.0), dist

    def _select(self, items):
        np = np_()
        F = np.asarray([y for _, y in items])
        fit, dist = self._fitness(F)
        nd = list(np.flatnonzero(fit < 1))
        if len(nd) > self.N:
            sub = dist[np.ix_(nd, nd)].copy()
            alive = list(range(len(nd)))
            while len(alive) > self.N:
                d = np.sort(sub[np.ix_(alive, alive)], axis=1)
                worst = min(range(len(alive)), key=lambda a: tuple(d[a].tolist()))
                alive.pop(worst)
            chosen = [nd[a] for a in alive]
        else:
            rest = [i for i in np.argsort(fit, kind='stable') if fit[i] >= 1]
            chosen = nd + rest[:self.N - len(nd)]
        return [items[i] for i in chosen], fit

    # -- operators ------------------------------------------------------------------------------
    def _crossover(self, p1, p2):
        rng, J = self.rng, self.prob.J
        os_ = pox(p1.os, p2.os, rng, J)
        c = make(self.inst, os_, uniform(p1.ma, p2.ma, rng), uniform(p1.agv_assign, p2.agv_assign, rng),
                 uniform(p1.agv_speed, p2.agv_speed, rng))
        if rng.random() < self.pm:
            n = self.prob.n
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            c.os[a], c.os[b] = c.os[b], c.os[a]
            k = rng.randint(n)
            c.ma[k] = rng.randint(self.prob.choices[k])
            k = rng.randint(n)
            c.agv_assign[k] = rng.randint(self.prob.V)
            c.agv_speed[k] = rng.randint(self.prob.S)
        c.invalidate()
        return c

    def _local(self, ch, action, path=None):
        rng, prob = self.rng, self.prob
        if path is None:
            path = critical_path(ch, prob)
        crit_p = [t for t in path if t[0] == 'P']
        if action == 0:
            bl = blocks(path)
            cand = []
            for bi, b in enumerate(bl):
                lo = 1 if bi == 0 else 0
                hi = len(b) - 2 if bi == len(bl) - 1 else len(b) - 1
                for a in range(lo, hi):
                    if b[a][1][0] != b[a + 1][1][0]:
                        cand.append((b[a][1], b[a + 1][1]))
            if cand:
                x, y = cand[rng.randint(len(cand))]
                return swap_tokens(ch, x, y)
            action = 1
        if action == 1:
            ops = [t[1] for t in crit_p]
            pairs = [(x, y) for ai, x in enumerate(ops) for y in ops[ai + 1:] if x[0] != y[0]]
            if pairs:
                x, y = pairs[rng.randint(len(pairs))]
                return swap_tokens(ch, x, y)
            action = 2
        c = ch.copy()
        if action == 2:
            opts = [t[1] for t in crit_p if prob.choices[prob.offset[t[1][0]] + t[1][1]] > 1]
            if opts:
                i, j = opts[rng.randint(len(opts))]
                k = prob.offset[i] + j
                c.ma[k] = (c.ma[k] + 1 + rng.randint(prob.choices[k] - 1)) % prob.choices[k]
                c.invalidate()
                return c
            action = 3
        ops = [t[1] for t in path]
        i, j = ops[rng.randint(len(ops))]
        k = prob.offset[i] + j
        if prob.V > 1:
            c.agv_assign[k] = (c.agv_assign[k] + 1 + rng.randint(prob.V - 1)) % prob.V
        else:
            c.agv_speed[k] = rng.randint(prob.S)
        c.invalidate()
        return c

    # -- DQN ------------------------------------------------------------------------------------
    def _act(self, s):
        np = np_()
        if self.random_operator or self.rng.random() >= self.eg:
            return int(self.rng.randint(self.N_ACTIONS))
        q, _ = self.q.forward(s[None, :])
        return int(np.argmax(q[0]))

    def _train(self):
        np = np_()
        t = time.perf_counter()
        idx = self.rng.choice(len(self.E), self.batch, replace=False)
        S = np.asarray([self.E[i][0] for i in idx])
        A = np.asarray([self.E[i][1] for i in idx])
        Rw = np.asarray([self.E[i][2] for i in idx], float)
        S2 = np.asarray([self.E[i][3] for i in idx])
        qn, _ = self.qt.forward(S2)
        y = Rw + self.gamma * qn.max(axis=1)
        q, acts = self.q.forward(S)
        g = np.zeros_like(q)
        g[np.arange(len(A)), A] = 2.0 * (q[np.arange(len(A)), A] - y) / len(A)
        self.q.backward(acts, g)
        self.trainings += 1
        if self.trainings % self.target_every == 0:
            self.qt.copy_from(self.q)
        self.train_seconds += time.perf_counter() - t

    # -- main loop ------------------------------------------------------------------------------
    def _run(self):
        np = np_()
        rng = self.rng
        while True:
            items = self.P
            F = np.asarray([y for _, y in items])
            fit, _ = self._fitness(F)
            pool = []
            for _ in range(len(items)):
                a, b = rng.randint(len(items), size=2)
                pool.append(items[a] if fit[a] <= fit[b] else items[b])
            off = []
            for ch, _ in items:
                mate = pool[rng.randint(len(pool))][0]
                for p1, p2 in ((ch, mate), (mate, ch)):
                    child = self.pre.pick(lambda p1=p1, p2=p2: self._crossover(p1, p2))
                    y = evaluate(child)
                    off.append((child, y))
                    yield child
            self.P, _ = self._select(items + off)
            self.A = nd_filter(self.A + self.P, self.cap)
            for idx in range(len(self.A)):
                if idx >= len(self.A):
                    break
                ch, y0 = self.A[idx]
                s = self.prob.state(ch)
                act = self._act(s)
                path = critical_path(ch, self.prob)
                nb = self.pre.pick(lambda ch=ch, act=act, path=path: self._local(ch, act, path))
                y = evaluate(nb)
                yield nb
                if dominates(y, y0):
                    self.A[idx] = (nb, y)
                    reward = 10.0
                elif not any(dominates(z, y) or np.array_equal(z, y) for _, z in self.A):
                    self.A = nd_filter(self.A + [(nb, y)], self.cap)
                    reward = 10.0
                else:
                    reward = 0.0
                self.E.append((s, act, reward, self.prob.state(nb)))
                if len(self.E) > self.pool_size:
                    self.E.pop(0)
                if len(self.E) > self.train_after:
                    self._train()


# ----------------------------------------------------------------------------- ILMPEA
class ILMPEA:
    """Cheng et al. (2025) imitation-learning-assisted multi-population evolutionary algorithm."""

    ACTIONS = ('JBX', 'TPX', 'MPX', 'Swap', 'Reassign', 'Improve')

    def __init__(self, population, values, seed, K=1, objectives=3, Np=200, pc=0.8, Ft=100, St=20, T=40,
                 expert_pairs=1024, epochs=20, batch=32, lr=1e-4, nd_cap=100, offline_evaluate=None,
                 expert_eval_cap=12000):
        np = np_()
        self.rng = np.random.RandomState(seed ^ 0x1717)
        self.seed = seed
        self.inst = population[0].instance
        self.prob = Problem(self.inst)
        self.obj = objectives
        self.scale = np.maximum(values.max(axis=0), 1.)
        self.pre = Preselect(self.inst, self.scale, K, objectives)
        self.Np, self.pc, self.Ft, self.St, self.T = Np, pc, Ft, St, T
        self.expert_pairs, self.epochs, self.batch, self.lr = expert_pairs, epochs, batch, lr
        self.cap = nd_cap
        self.offline_evaluate = offline_evaluate
        self.expert_eval_cap = expert_eval_cap
        self.offline_evaluations = 0
        self.offline_seconds = 0.0
        self.init_items = list(zip(population, values))
        self.gen = self._run()

    @property
    def scored(self):
        return self.pre.count

    @property
    def score_seconds(self):
        return self.pre.seconds

    def step(self):
        return next(self.gen)

    # -- greedy decoding of OS-only individuals (Algorithm 1 of the paper) ------------------------
    def _greedy(self, os_, mode, inst=None, prob=None):
        """Builds MS/VS/speed genes while simulating the released decoder; mode 'C' or 'E'."""
        np = np_()
        inst = inst or self.inst
        prob = prob or self.prob
        M, V = inst.num_machines, inst.num_agv
        mav = np.zeros(M)
        vav = np.zeros(V)
        loc = np.zeros(V, dtype=int)
        last_job = [-1] * M
        cnt = {}
        end_prev = {}
        mach_prev = {}
        ma = np.zeros(prob.n, dtype=int)
        va = np.zeros(prob.n, dtype=int)
        sp = np.zeros(prob.n, dtype=int)
        speeds = list(inst.speed_levels)
        s_idx = int(np.argmax(speeds)) if mode == 'C' else int(np.argmin(speeds))
        speed = speeds[s_idx]
        a_load = inst.agv_load_power_params[0]
        a_empty = inst.agv_empty_power_params[0]
        for job in os_.tolist():
            j = cnt.get(job, 0)
            cnt[job] = j + 1
            k = prob.offset[job] + j
            ms = prob.machines[k]
            if mode == 'C':
                mi = int(np.argmin(prob.ptime[k]))
            else:
                mi = int(np.argmin([inst.machine_proc_power[m] * p for m, p in zip(ms, prob.ptime[k])]))
            u = ms[mi]
            src = 0 if j == 0 else mach_prev[job] + 1
            dest = u + 1
            cargo = 0.0 if j == 0 else end_prev[job]
            best, best_l = None, 0
            for l in range(V):
                te = inst.get_distance(loc[l], src) / speed
                tl = inst.get_distance(src, dest) / speed
                if mode == 'C':
                    key = max(cargo, vav[l] + te) + tl
                else:
                    key = a_empty * speed * speed * te + a_load * speed * speed * tl
                if best is None or key < best - 1e-12:
                    best, best_l = key, l
            l = best_l
            te = inst.get_distance(loc[l], src) / speed
            tl = inst.get_distance(src, dest) / speed
            ts = max(cargo, vav[l] + te)
            tend = ts + tl
            setup = inst.machine_setup_time[u] if (last_job[u] >= 0 and last_job[u] != job) else 0.0
            s = max(tend, mav[u] + setup)
            e = s + prob.ptime[k][mi]
            mav[u], vav[l], loc[l], last_job[u] = e, tend, dest, job
            end_prev[job], mach_prev[job] = e, u
            ma[k], va[k], sp[k] = mi, l, s_idx
        return C.compat.Chromosome(inst, os_.copy(), ma, va, sp)

    # -- search operators --------------------------------------------------------------------------
    def _jbx(self, a, b):
        c = a.copy()
        c.os = pox(a.os, b.os, self.rng, self.prob.J)
        c.invalidate()
        return c

    def _tpx(self, a, b):
        np = np_()
        n = self.prob.n
        x, y = sorted(int(v) for v in self.rng.choice(n, 2, replace=False))
        c = a.copy()
        c.ma = b.ma.copy()
        c.ma[x:y + 1] = a.ma[x:y + 1]
        c.invalidate()
        return c

    def _mpx(self, a, b):
        np = np_()
        n = self.prob.n
        N = self.rng.randint(1, max(2, n // 2))
        sel = self.rng.choice(n, N, replace=False)
        c = a.copy()
        va, spd = b.agv_assign.copy(), b.agv_speed.copy()
        va[sel], spd[sel] = a.agv_assign[sel], a.agv_speed[sel]
        c.agv_assign, c.agv_speed = va, spd
        c.invalidate()
        return c

    def _swap(self, a, b=None):
        c = a.copy()
        x, y = (int(v) for v in self.rng.choice(self.prob.n, 2, replace=False))
        c.os[x], c.os[y] = c.os[y], c.os[x]
        c.invalidate()
        return c

    def _reassign(self, a, b=None):
        c = a.copy()
        prob = self.prob
        opts = [k for k in range(prob.n) if prob.choices[k] > 1]
        if opts:
            k = opts[self.rng.randint(len(opts))]
            c.ma[k] = (c.ma[k] + 1 + self.rng.randint(prob.choices[k] - 1)) % prob.choices[k]
        c.invalidate()
        return c

    def _improve(self, a, b=None, inst=None, prob=None, sch=None):
        """Re-assign one operation's AGV to the vehicle that delivers it earliest (state from decode)."""
        np = np_()
        inst = inst or self.inst
        prob = prob or self.prob
        c = a.copy()
        k = int(self.rng.randint(prob.n))
        if sch is None:
            sch = C.compat.decoding.decode(a)
        i, j = prob.pairs[k]
        speed = inst.speed_levels[int(a.agv_speed[k]) % inst.num_speeds]
        src = 0 if j == 0 else sch.op_machine[(i, j - 1)] + 1
        cargo = 0.0 if j == 0 else sch.op_end[(i, j - 1)]
        t0 = sch.transport_start[(i, j)]
        best, best_l = None, int(a.agv_assign[k])
        for l in range(inst.num_agv):
            before = [e for e in sch.agv_schedule[l] if e[2] < t0 - 1e-12]
            avail = before[-1][3] if before else 0.0
            at = 0 if not before else sch.op_machine[(before[-1][0], before[-1][1])] + 1
            arr = max(cargo, avail + inst.get_distance(at, src) / speed)
            if best is None or arr < best - 1e-12:
                best, best_l = arr, l
        c.agv_assign[k] = best_l
        c.invalidate()
        return c

    def _apply(self, action, a, b, inst=None, prob=None, sch=None):
        if action == 5:
            return self._improve(a, b, inst, prob, sch)
        return (self._jbx, self._tpx, self._mpx, self._swap, self._reassign)[action](a, b)

    # -- imitation learning (offline, on a perturbed training instance) -----------------------------
    def _training_instance(self):
        np = np_()
        rng = np.random.RandomState(self.seed ^ 0x7A7A)
        inst = copy.deepcopy(self.inst)
        M = inst.num_machines
        for (i, j), ms in list(inst.compatible_machines.items()):
            ms = list(ms)
            base = np.mean([inst.processing_times[i, j, m] for m in ms])
            if len(ms) > 1 and rng.random() < 0.5:
                drop = ms.pop(rng.randint(len(ms)))
                del inst.processing_times[i, j, drop]
            elif len(ms) < M:
                add = [m for m in range(M) if m not in ms]
                m = add[rng.randint(len(add))]
                ms = sorted(ms + [m])
                inst.processing_times[i, j, m] = float(max(1, round(base * rng.uniform(.8, 1.2))))
            inst.compatible_machines[i, j] = sorted(ms)
            for m in ms:
                inst.processing_times[i, j, m] = float(max(1, round(inst.processing_times[i, j, m] * rng.uniform(.8, 1.2))))
        return inst

    def _offline(self):
        np = np_()
        t = time.perf_counter()
        inst = self._training_instance()
        prob = Problem(inst)
        saved = C.compat.decoding.evaluate
        if self.offline_evaluate is not None:
            C.compat.decoding.evaluate = self.offline_evaluate
        try:
            def ev(ch):
                self.offline_evaluations += 1
                y = ch.objectives.to_array()
                ch._schedule = None
                return y[:2] if self.obj == 2 else y
            rng = self.rng
            pool = []
            for _ in range(10):
                os_ = np.repeat(np.arange(prob.J), inst.num_operations)
                rng.shuffle(os_)
                for mode in ('C', 'E'):
                    ch = self._greedy(os_, mode, inst, prob)
                    pool.append((ch, ev(ch)))
            for _ in range(10):
                ch = prob.random(rng)
                pool.append((ch, ev(ch)))
            nd = nd_filter(pool, self.cap)
            X, Y = [], []
            while len(X) < self.expert_pairs and self.offline_evaluations < self.expert_eval_cap:
                src = nd if nd and rng.random() < 0.7 else pool
                a, ya = src[rng.randint(len(src))]
                b = pool[rng.randint(len(pool))][0]
                order = rng.permutation(len(self.ACTIONS))
                for act in order:
                    c = self._apply(int(act), a, b, inst, prob)
                    yc = ev(c)
                    better = dominates(yc, ya) or not any(dominates(z, yc) or np.array_equal(z, yc) for _, z in nd)
                    if better:
                        X.append(prob.state(a))
                        Y.append(int(act))
                        pool.append((c, yc))
                        if len(pool) > 400:
                            pool.pop(rng.randint(len(pool)))
                        nd = nd_filter(nd + [(c, yc)], self.cap)
                        break
        finally:
            C.compat.decoding.evaluate = saved
        d = 4 * self.prob.n
        self.policy = MLP([d, 512, 256, 128, 64, len(self.ACTIONS)], np.random.RandomState(self.seed ^ 0x99), self.lr)
        if X:
            X, Y = np.asarray(X), np.asarray(Y)
            for _ in range(self.epochs):
                perm = rng.permutation(len(X))
                for s in range(0, len(X), self.batch):
                    idx = perm[s:s + self.batch]
                    logits, acts = self.policy.forward(X[idx])
                    z = logits - logits.max(axis=1, keepdims=True)
                    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
                    g = p.copy()
                    g[np.arange(len(idx)), Y[idx]] -= 1.0
                    self.policy.backward(acts, g / len(idx))
        self.expert_samples = len(X)
        self.offline_seconds = time.perf_counter() - t

    def _choose(self, ch):
        np = np_()
        logits, _ = self.policy.forward(self.prob.state(ch)[None, :])
        return int(np.argmax(logits[0]))

    # -- main loop ----------------------------------------------------------------------------------
    def _run(self):
        np = np_()
        rng = self.rng
        self._offline()
        Mp, Tp, Kp = [], [], []
        for mode, popl in (('C', Mp), ('E', Tp)):
            for _ in range(self.Np):
                os_ = np.repeat(np.arange(self.prob.J), self.inst.num_operations)
                rng.shuffle(os_)
                ch = self._greedy(os_, mode)
                popl.append((ch, evaluate(ch)))
                yield ch
        ND = nd_filter(self.init_items + Mp + Tp, self.cap)
        gen = 0
        while True:
            if gen % self.Ft == 0:
                new = []
                for _ in range(self.Np):
                    a = Mp[rng.randint(len(Mp))][0]
                    b = Tp[rng.randint(len(Tp))][0]
                    ch = self.pre.pick(lambda a=a, b=b: make(self.inst, pox(a.os, b.os, rng, self.prob.J),
                                                             uniform(a.ma, b.ma, rng), uniform(a.agv_assign, b.agv_assign, rng),
                                                             uniform(a.agv_speed, b.agv_speed, rng)))
                    new.append((ch, evaluate(ch)))
                    yield ch
                Kp = self._nds_select(Kp + new)
            for mode, popl in (('C', Mp), ('E', Tp)):
                off = []
                while len(off) < self.Np:
                    if rng.random() >= self.pc:
                        continue
                    a = popl[rng.randint(len(popl))][0]
                    b = popl[rng.randint(len(popl))][0]
                    for p1, p2 in ((a, b), (b, a)):
                        ch = self._greedy(pox(p1.os, p2.os, rng, self.prob.J), mode)
                        off.append((ch, evaluate(ch)))
                        yield ch
                allp = popl + off
                obj = 0 if mode == 'C' else 1
                sel = []
                for _ in range(self.Np):
                    x, y = rng.randint(len(allp), size=2)
                    sel.append(allp[x] if allp[x][1][obj] <= allp[y][1][obj] else allp[y])
                popl[:] = sel
            off = []
            while len(off) < self.Np:
                if rng.random() >= self.pc:
                    continue
                a = Kp[rng.randint(len(Kp))][0]
                b = Kp[rng.randint(len(Kp))][0]
                for p1, p2 in ((a, b), (b, a)):
                    ch = self.pre.pick(lambda p1=p1, p2=p2: make(self.inst, pox(p1.os, p2.os, rng, self.prob.J),
                                                                 uniform(p1.ma, p2.ma, rng), uniform(p1.agv_assign, p2.agv_assign, rng),
                                                                 uniform(p1.agv_speed, p2.agv_speed, rng)))
                    off.append((ch, evaluate(ch)))
                    yield ch
            Kp = self._nds_select(Kp + off)
            ND = nd_filter(ND + Mp + Tp + Kp, self.cap)
            gen += 1
            if gen % self.St == 0:
                members = list(ND)
                order = rng.permutation(len(members))
                pairs = [(members[order[a]], members[order[a + 1]]) for a in range(0, len(order) - 1, 2)]
                found = []
                for (a, ya), (b, yb) in pairs:
                    for x, partner in ((a, b), (b, a)):
                        act = self._choose(x)
                        sch = C.compat.decoding.decode(x) if act == 5 else None
                        for _ in range(self.T):
                            ch = self.pre.pick(lambda x=x, partner=partner, act=act, sch=sch: self._apply(act, x, partner, sch=sch))
                            found.append((ch, evaluate(ch)))
                            yield ch
                ND = nd_filter(ND + found, self.cap)

    def _nds_select(self, items):
        np = np_()
        if len(items) <= self.Np:
            return items
        F = np.asarray([y for _, y in items])
        keep = C.environmental(F, self.Np)
        return [items[i] for i in keep]
