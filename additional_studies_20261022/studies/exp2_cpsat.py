"""B3: CP-SAT lower bound on makespan for test-panel snapshots (relaxation, see PROTOCOL.md section 4).

  python exp2_cpsat.py --panel test --seconds 300 --workers 4 --lanes 7 --out out/B3
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCALE = 100


def load(path):
    d = json.loads(Path(path).read_text())
    ops = {}
    for j, o, ms in d['compatible_machines']:
        ops[(j, o)] = dict(machines=list(ms), p={})
    for j, o, m, t in d['processing_times']:
        ops[(j, o)]['p'][m] = t
    return d, ops


def solve(path, seconds, workers):
    from ortools.sat.python import cp_model
    d, ops = load(path)
    D = d['distance_matrix']
    vmax = max(d['speed_levels'])
    V = d['num_agv']

    def trip(a, b):  # a, b are distance-matrix indices; floor keeps a valid lower bound
        return int(math.floor(D[a][b] / vmax * SCALE + 1e-9))

    keys = sorted(ops)
    horizon = 0
    for (j, o) in keys:
        horizon += int(max(ops[(j, o)]['p'].values()) * SCALE)
        horizon += max(trip(a, b) for a in range(len(D)) for b in range(len(D)))
    m = cp_model.CpModel()
    mk = m.NewIntVar(0, horizon, 'mk')
    x, s, e, ts, te = {}, {}, {}, {}, {}
    mach_iv = {}
    veh_iv = {v: [] for v in range(V)}
    for (j, o) in keys:
        k = (j, o)
        s[k] = m.NewIntVar(0, horizon, f's{j}_{o}')
        e[k] = m.NewIntVar(0, horizon, f'e{j}_{o}')
        ts[k] = m.NewIntVar(0, horizon, f'ts{j}_{o}')
        te[k] = m.NewIntVar(0, horizon, f'te{j}_{o}')
        lits = []
        for mc in ops[k]['machines']:
            lit = m.NewBoolVar(f'x{j}_{o}_{mc}')
            x[(j, o, mc)] = lit
            lits.append(lit)
            iv = m.NewOptionalIntervalVar(s[k], int(round(ops[k]['p'][mc] * SCALE)), e[k], lit, f'i{j}_{o}_{mc}')
            mach_iv.setdefault(mc, []).append(iv)
        m.AddExactlyOne(lits)
        dur = m.NewIntVar(0, horizon, f'd{j}_{o}')
        if o == 0:
            m.Add(dur >= sum(x[(j, o, b)] * trip(0, b + 1) for b in ops[k]['machines']))
        else:
            prev = ops[(j, o - 1)]['machines']
            cur = ops[k]['machines']
            m.Add(dur >= sum(x[(j, o, b)] * min(trip(a + 1, b + 1) for a in prev) for b in cur))
            m.Add(dur >= sum(x[(j, o - 1, a)] * min(trip(a + 1, b + 1) for b in cur) for a in prev))
            m.Add(ts[k] >= e[(j, o - 1)])
        if V == 1:
            veh_iv[0].append(m.NewIntervalVar(ts[k], dur, te[k], f't{j}_{o}'))
        else:
            ys = []
            for v in range(V):
                y = m.NewBoolVar(f'y{j}_{o}_{v}')
                ys.append(y)
                veh_iv[v].append(m.NewOptionalIntervalVar(ts[k], dur, te[k], y, f't{j}_{o}_{v}'))
            m.AddExactlyOne(ys)
            m.Add(te[k] == ts[k] + dur)
        m.Add(s[k] >= te[k])
        m.Add(mk >= e[k])
    for ivs in mach_iv.values():
        m.AddNoOverlap(ivs)
    for ivs in veh_iv.values():
        m.AddNoOverlap(ivs)
    m.Minimize(mk)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 20261019
    t = time.time()
    status = solver.Solve(m)
    return dict(name=Path(path).stem, status=solver.StatusName(status), seconds=time.time() - t,
                lower_bound=solver.BestObjectiveBound() / SCALE,
                relaxed_objective=(solver.ObjectiveValue() / SCALE) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
                operations=len(keys), vehicles=V, scale=SCALE, vmax=vmax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', default='test')
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lanes', type=int, default=7)
    p.add_argument('--out', default='out/B3')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    os.chdir(HERE)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(Path('inputs/INPUT_MANIFEST.json').read_text()) if r['panel'] == a.panel]
    done = {json.loads(l)['name'] for l in (out / 'bounds.jsonl').open()} if (out / 'bounds.jsonl').exists() else set()
    todo = [r for r in rows if r['name'] not in done][:a.limit]
    with ProcessPoolExecutor(a.lanes) as ex, (out / 'bounds.jsonl').open('a') as log:
        futs = {ex.submit(solve, 'inputs/' + r['file'], a.seconds, a.workers): r for r in todo}
        for f in as_completed(futs):
            r = futs[f]
            res = f.result()
            res.update(base_id=r['base_id'], regime=r['regime'], input_sha256=r['sha256'])
            log.write(json.dumps(res) + '\n')
            log.flush()
    print('done', len(todo))


if __name__ == '__main__':
    main()
