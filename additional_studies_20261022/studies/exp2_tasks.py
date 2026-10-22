"""Inputs, task lists, tuning selection, and locks for exp2.

  python exp2_tasks.py panels --release REL
  python exp2_tasks.py make --study NAME [--selection SELECTION.json]
  python exp2_tasks.py select --runs out/B1
  python exp2_tasks.py lock --study NAME
"""
import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2_core as C  # noqa: E402

EVAL_ENDS = (2000, 5000, 10000)
NSGA_DEFAULT = dict(pop=100, pc=.9, pm=1., op='std')
MOEAD_DEFAULT = dict(H=12, T=10, delta=.9, nr=2, op='std')
NSGA_GRID = ([dict(pop=p, pc=c, pm=m, op='std') for p in (50, 100, 200) for c in (.7, .9) for m in (1., 3.)]
             + [dict(pop=p, op='joint') for p in (50, 100, 200)])
MOEAD_GRID = ([dict(H=h, T=t, delta=.9, nr=r, op='std') for h in (8, 12, 16) for t in (5, 10, 20) for r in (1, 2)]
              + [dict(H=h, T=10, delta=.9, nr=2, op='joint') for h in (8, 12, 16)])
STUDIES = ('ENG', 'B1', 'B2e', 'B2t', 'Ce', 'Ct', 'A')
LOCKED_FILES = ('exp2_core.py', 'exp2_run.py', 'exp2_tasks.py', 'exp2_selftest.py', 'exp2_cpsat.py', 'PROTOCOL.md',
                'inputs/INPUT_MANIFEST.json')


def label(family, params):
    return family + '|' + '|'.join(f'{k}={params[k]}' for k in sorted(params))


def panels(release):
    C.setup(release)
    rows = []
    for panel in ('test', 'tune', 'eng'):
        rows += C.generate_panel(panel, HERE / 'inputs' / panel)
    (HERE / 'inputs' / 'INPUT_MANIFEST.json').write_text(json.dumps(rows, indent=1, sort_keys=True))
    print(len(rows), 'inputs')


def manifest():
    return json.loads((HERE / 'inputs' / 'INPUT_MANIFEST.json').read_text())


def cells(panel):
    cfg = C.PANELS[panel]
    out = []
    for r in manifest():
        if r['panel'] != panel:
            continue
        for seed in cfg['seeds']:
            out.append((r, seed))
    return out


def task(study, r, seed, arm, method, level, algo, evaluator, mode, ends):
    return dict(study=study, id=f"{study}__{r['name']}__{seed}__{arm}", arm=arm, method=method, level=level,
                panel=r['panel'], name=r['name'], base_id=r['base_id'], stratum=r['stratum'], regime=r['regime'],
                seed=seed, input='inputs/' + r['file'], input_sha256=r['sha256'], algo=algo,
                evaluator=evaluator, mode=mode, endpoints=list(ends))


def methods(selection=None):
    m = dict(rbp=dict(family='rbp', params=dict(K=4, period=0)),
             r4=dict(family='rbp', params=dict(K=4, period='uniform')),
             nsga2_def=dict(family='nsga2', params=dict(NSGA_DEFAULT)),
             moead_def=dict(family='moead', params=dict(MOEAD_DEFAULT)))
    if selection:
        sel = json.loads(Path(selection).read_text())
        n, d = dict(sel['nsga2']['params']), dict(sel['moead']['params'])
        m.update(nsga2t=dict(family='nsga2', params=n), nsga2t_rbp=dict(family='nsga2', params=dict(n, K=4)),
                 moeadt=dict(family='moead', params=d), moeadt_rbp=dict(family='moead', params=dict(d, K=4)))
        for k in (1, 2, 4, 8, 16):
            m[f'rbpK{k:02}'] = dict(family='rbp', params=dict(K=k, period=0))
            m[f'nsga2tK{k:02}'] = dict(family='nsga2', params=dict(n, K=k))
    return m


NATIVE = dict(kind='native')


def make(study, selection=None):
    m = methods(selection)
    out = []
    if study == 'ENG':
        for r, seed in cells('eng'):
            for arm in ('rbp', 'r4', 'nsga2_def', 'moead_def'):
                out.append(task(study, r, seed, arm + '@e', arm, 'eval1000', m[arm], NATIVE, 'eval', (500, 1000)))
            for arm in ('rbp', 'nsga2_def'):
                out.append(task(study, r, seed, arm + '@d25', arm, 'd25', m[arm], dict(kind='delay', d_ms=25), 'time', (10., 20.)))
                for R in (1, 8, 64):
                    out.append(task(study, r, seed, f'{arm}@R{R:02}', arm, f'R{R:02}', m[arm],
                                    dict(kind='stoch', R=R), 'time', (10., 20.)))
        return out
    if study == 'B1':
        for r, seed in cells('tune'):
            for i, params in enumerate(NSGA_GRID):
                out.append(task(study, r, seed, f'nsga2c{i:02}', label('nsga2', params), 'eval10k',
                                dict(family='nsga2', params=params), NATIVE, 'eval', EVAL_ENDS))
            for i, params in enumerate(MOEAD_GRID):
                out.append(task(study, r, seed, f'moeadc{i:02}', label('moead', params), 'eval10k',
                                dict(family='moead', params=params), NATIVE, 'eval', EVAL_ENDS))
        out.sort(key=lambda t: 0 if 'medium' in t['name'] else 1)
        return out
    assert selection, 'Tuned configurations required'
    if study in ('B2e', 'B2t'):
        arms = ('rbp', 'r4', 'nsga2_def', 'nsga2t', 'nsga2t_rbp', 'moead_def', 'moeadt', 'moeadt_rbp')
        mode, ends = ('eval', EVAL_ENDS) if study == 'B2e' else ('time', (60., 120.))
        for r, seed in cells('test'):
            for arm in arms:
                out.append(task(study, r, seed, arm, arm, mode, m[arm], NATIVE, mode, ends))
    elif study == 'Ce':
        for r, seed in cells('test'):
            for k in (1, 2, 4, 8, 16):
                for host in ('rbp', 'nsga2t'):
                    arm = f'{host}K{k:02}'
                    out.append(task(study, r, seed, arm, arm, 'eval', m[arm], NATIVE, 'eval', EVAL_ENDS))
    elif study == 'Ct':
        for r, seed in cells('test'):
            for d in (0, 25):
                ev = NATIVE if d == 0 else dict(kind='delay', d_ms=d)
                for k in (1, 2, 4, 8, 16):
                    for host in ('rbp', 'nsga2t'):
                        arm = f'{host}K{k:02}'
                        out.append(task(study, r, seed, f'{arm}@d{d:02}', arm, f'd{d:02}', m[arm], ev, 'time', (60., 120.)))
    elif study == 'A':
        for r, seed in cells('test'):
            for R in (1, 8, 64):
                for arm in ('rbp', 'r4', 'nsga2t', 'nsga2t_rbp', 'moeadt', 'moeadt_rbp'):
                    out.append(task(study, r, seed, f'{arm}@R{R:02}', arm, f'R{R:02}', m[arm],
                                    dict(kind='stoch', R=R), 'time', (150., 300.)))
    else:
        raise ValueError(study)
    if study in ('B2e', 'Ce'):
        out.sort(key=lambda t: 0 if 'medium' in t['name'] else 1)
    else:
        random.Random(C.hash32(f'{C.PROTOCOL}/order/{study}')).shuffle(out)
    return out


def select(runs):
    rows = collections.defaultdict(dict)
    labels = {}
    for f in Path(runs, 'runs').glob('*.json.gz'):
        s = json.loads(gzip.decompress(f.read_bytes()))['summary']
        hv = next(e['hv_normalized'] for e in s['endpoints'] if e['endpoint'] == 10000)
        fam = s['algo']['family']
        rows[(fam, s['name'], s['seed'])][s['arm']] = hv
        labels[s['arm']] = s['algo']
    out = {}
    for fam, grid in (('nsga2', NSGA_GRID), ('moead', MOEAD_GRID)):
        cell_keys = sorted(k for k in rows if k[0] == fam)
        arms = sorted({a for k in cell_keys for a in rows[k]})
        assert len(arms) == len(grid) and len(cell_keys) == 48, (fam, len(arms), len(cell_keys))
        ranks = collections.defaultdict(list)
        hvs = collections.defaultdict(list)
        for k in cell_keys:
            vals = rows[k]
            assert set(vals) == set(arms)
            order = sorted(arms, key=lambda a: -vals[a])
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                for a in order[i:j + 1]:
                    ranks[a].append((i + j) / 2 + 1)
                i = j + 1
            for a in arms:
                hvs[a].append(vals[a])
        table = sorted(({'arm': a, 'params': labels[a]['params'], 'mean_rank': sum(ranks[a]) / len(ranks[a]),
                         'mean_hv': sum(hvs[a]) / len(hvs[a])} for a in arms),
                       key=lambda r: (r['mean_rank'], -r['mean_hv']))
        out[fam] = dict(table[0], rule='minimum mean within-cell rank of 10,000-evaluation HV; tie: higher mean HV')
        out[fam + '_table'] = table
    out['cells_per_family'] = 48
    out['created_utc'] = C.utc()
    (HERE / 'SELECTION.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('nsga2', 'moead')}, indent=1))


def lock(study):
    files = {rel: C.sha(HERE / rel) for rel in LOCKED_FILES}
    if (HERE / 'SELECTION.json').exists() and study != 'B1':
        files['SELECTION.json'] = C.sha(HERE / 'SELECTION.json')
    tf = HERE / 'tasks' / f'{study}.json'
    env = HERE / 'ENV_FREEZE.txt'
    if env.exists():
        files['ENV_FREEZE.txt'] = C.sha(env)
    rec = dict(protocol=C.PROTOCOL, study=study, files=files, tasks_file=f'tasks/{study}.json',
               tasks_sha256=C.sha(tf), tasks=len(json.loads(tf.read_text())), created_utc=C.utc(),
               formal_runs_before_lock=0)
    (HERE / 'locks').mkdir(exist_ok=True)
    p = HERE / 'locks' / f'LOCK_{study}.json'
    assert not p.exists(), 'Lock already exists'
    p.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('panels', 'make', 'select', 'lock'))
    p.add_argument('--release')
    p.add_argument('--study')
    p.add_argument('--selection')
    p.add_argument('--runs')
    a = p.parse_args()
    if a.cmd == 'panels':
        panels(a.release)
    elif a.cmd == 'make':
        assert a.study in STUDIES
        tl = make(a.study, a.selection)
        assert len({t['id'] for t in tl}) == len(tl)
        (HERE / 'tasks').mkdir(exist_ok=True)
        (HERE / 'tasks' / f'{a.study}.json').write_text(json.dumps(tl, indent=0))
        print(a.study, len(tl), collections.Counter(t['arm'] for t in tl).most_common(3))
    elif a.cmd == 'select':
        select(a.runs)
    else:
        lock(a.study)


if __name__ == '__main__':
    main()
