"""exp3b: tuning (T1) and comparison (S1-S4) task lists, configuration selection, and locks.

  python exp3b_tasks.py make --study T1|S1|S2|S3|S4|ENG3B
  python exp3b_tasks.py select --runs out/T1
  python exp3b_tasks.py lock --study T1
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

PROTOCOL = 'exp3b_adapted_20261021_v1'
DQ_GRID = [dict(pop=p, pm=m, op=o, ls_max=l) for p in (50, 100) for m in (0.1, 0.2, 0.4)
           for o in ('std', 'joint') for l in (None, 10)]
IL_GRID = [dict(Np=n, Ft=f, St=s, T=t, fusion='replace') for n in (50, 100, 200)
           for f, s in ((5, 2), (20, 5), (100, 20)) for t in (10, 40)]
ARMS = ('rbp', 'nsga2t', 'dqnma', 'dqnma_rbp', 'ilmpea', 'ilmpea_rbp')
LOCKED = ('exp2_core.py', 'exp3b_rivals.py', 'exp3b_run.py', 'exp3b_tasks.py', 'PROTOCOL_exp3b.md',
          'inputs/INPUT_MANIFEST.json', 'SELECTION.json', 'ENV_FREEZE.txt')


def manifest():
    return json.loads((HERE / 'inputs' / 'INPUT_MANIFEST.json').read_text())


def task(study, r, seed, arm, method, algo, dims, mode, ends):
    return dict(study=study, id=f"{study}__{r['name']}__{seed}__{arm}", arm=arm, method=method, panel=r['panel'],
                name=r['name'], base_id=r['base_id'], stratum=r['stratum'], regime=r['regime'], seed=seed,
                input='inputs/' + r['file'], input_sha256=r['sha256'], algo=algo, objectives=dims, mode=mode,
                endpoints=list(ends), operations=r['operations'])


def methods():
    sel = json.loads((HERE / 'SELECTION.json').read_text())
    sb = json.loads((HERE / 'SELECTION_exp3b.json').read_text())
    dq, il = dict(sb['dqnma']['params']), dict(sb['ilmpea']['params'])
    return dict(rbp=dict(family='rbp', params=dict(K=4, period=0)),
                nsga2t=dict(family='nsga2', params=dict(sel['nsga2']['params'])),
                dqnma=dict(family='dqnma', params=dq), dqnma_rbp=dict(family='dqnma', params=dict(dq, K=4)),
                ilmpea=dict(family='ilmpea', params=il), ilmpea_rbp=dict(family='ilmpea', params=dict(il, K=4)))


def make(study):
    out = []
    rows = manifest()
    if study == 'T1':
        for r in rows:
            if r['panel'] != 'tune':
                continue
            for seed in C.PANELS['tune']['seeds']:
                for i, p in enumerate(DQ_GRID):
                    out.append(task(study, r, seed, f'dq{i:02}', 'dqnma', dict(family='dqnma', params=p), 3, 'eval',
                                    (2000, 5000, 10000)))
                for i, p in enumerate(IL_GRID):
                    out.append(task(study, r, seed, f'il{i:02}', 'ilmpea', dict(family='ilmpea', params=p), 3, 'eval',
                                    (2000, 5000, 10000)))
        out.sort(key=lambda t: 0 if 'medium' in t['name'] else 1)
        return out
    if study == 'ENG3B':
        for r in rows:
            if r['panel'] != 'eng':
                continue
            for seed in C.PANELS['eng']['seeds']:
                for i in (0, len(DQ_GRID) - 1):
                    out.append(task(study, r, seed, f'dq{i:02}', 'dqnma', dict(family='dqnma', params=DQ_GRID[i]), 3,
                                    'eval', (1000, 2000)))
                for i in (0, len(IL_GRID) - 1):
                    out.append(task(study, r, seed, f'il{i:02}', 'ilmpea', dict(family='ilmpea', params=IL_GRID[i]), 3,
                                    'eval', (1000, 2000)))
        return out
    m = methods()
    for r in rows:
        if r['panel'] != 'test':
            continue
        no = r['operations']
        spec = dict(S1=(3, 'time', [float(no), float(2 * no)]), S2=(3, 'time', [60.0, 120.0]),
                    S3=(3, 'eval', [5000, 10000, 20000]), S4=(2, 'time', [float(no), float(2 * no)]))[study]
        for seed in C.PANELS['test']['seeds']:
            for arm in ARMS:
                out.append(task(study, r, seed, arm, arm, m[arm], *spec))
    if study == 'S3':
        out.sort(key=lambda t: 0 if 'medium' in t['name'] else 1)
    else:
        random.Random(C.hash32(f'exp3b/order/{study}')).shuffle(out)
    return out


def select(runs):
    rows = collections.defaultdict(dict)
    params = {}
    for f in Path(runs, 'runs').glob('*.json.gz'):
        s = json.loads(gzip.decompress(f.read_bytes()))['summary']
        hv = next(e['hv_normalized'] for e in s['endpoints'] if e['endpoint'] == 10000)
        rows[(s['method'], s['name'], s['seed'])][s['arm']] = hv
        params[s['arm']] = s['algo']['params']
    out = {}
    for fam, grid in (('dqnma', DQ_GRID), ('ilmpea', IL_GRID)):
        keys = sorted(k for k in rows if k[0] == fam)
        arms = sorted({a for k in keys for a in rows[k]})
        assert len(arms) == len(grid) and len(keys) == 48, (fam, len(arms), len(keys))
        ranks, hvs = collections.defaultdict(list), collections.defaultdict(list)
        for k in keys:
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
        table = sorted(({'arm': a, 'params': params[a], 'mean_rank': sum(ranks[a]) / len(ranks[a]),
                         'mean_hv': sum(hvs[a]) / len(hvs[a])} for a in arms), key=lambda r: (r['mean_rank'], -r['mean_hv']))
        out[fam] = dict(table[0], rule='minimum mean within-cell rank of 10,000-evaluation HV; tie: higher mean HV')
        out[fam + '_table'] = table
    out['created_utc'] = C.utc()
    (HERE / 'SELECTION_exp3b.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('dqnma', 'ilmpea')}, indent=1))


def lock(study):
    names = list(LOCKED) + ([] if study == 'T1' else ['SELECTION_exp3b.json'])
    files = {rel: C.sha(HERE / rel) for rel in names}
    tf = HERE / 'tasks' / f'{study}.json'
    rec = dict(protocol=PROTOCOL, study=study, files=files, tasks_file=f'tasks/{study}.json', tasks_sha256=C.sha(tf),
               tasks=len(json.loads(tf.read_text())), created_utc=C.utc(), formal_runs_before_lock=0)
    p = HERE / 'locks' / f'LOCK_{study}.json'
    assert not p.exists()
    p.write_text(json.dumps(rec, indent=1))
    print(study, rec['tasks'], rec['tasks_sha256'][:16])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('make', 'select', 'lock'))
    p.add_argument('--study')
    p.add_argument('--runs')
    a = p.parse_args()
    if a.cmd == 'make':
        tl = make(a.study)
        assert len({t['id'] for t in tl}) == len(tl)
        (HERE / 'tasks' / f'{a.study}.json').write_text(json.dumps(tl, indent=0))
        print(a.study, len(tl), collections.Counter(t['method'] for t in tl))
    elif a.cmd == 'select':
        select(a.runs)
    else:
        lock(a.study)


if __name__ == '__main__':
    main()
