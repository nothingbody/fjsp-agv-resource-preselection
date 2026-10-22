"""exp3 analysis (PROTOCOL_exp3.md section 5). Endpoints are indexed (final = last) because R1/R4 deadlines vary.

  python exp3_analyze.py --study R1
"""
import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2_analyze as A  # noqa: E402

PRIMARY = [('rbp', 'dqnma'), ('rbp', 'ilmpea'), ('dqnma_rbp', 'dqnma'), ('ilmpea_rbp', 'ilmpea')]
CHECK = [('ilmpea', 'dqnma'), ('dqnma', 'nsga2t'), ('ilmpea', 'nsga2t')]
SECOND = [('rbp', 'nsga2t'), ('rbp', 'dqnma_rbp'), ('rbp', 'ilmpea_rbp')]


def load(run_dir):
    rows = {}
    for f in Path(run_dir, 'runs').glob('*.json.gz'):
        import gzip
        s = json.loads(gzip.decompress(f.read_bytes()))['summary']
        rows[(s['base_id'], s['regime'], s['seed'], s['method'])] = s
    return rows


def contrast(rows, a, b, idx=-1, metric='hv_normalized', minimize=False, j=None):
    def v(k, arm):
        e = rows[k + (arm,)]['endpoints'][idx]
        return e[metric][j] if j is not None else e[metric]
    if minimize:  # relative reduction of an objective minimum, positive favours a
        f = lambda k: (v(k, b) - v(k, a)) / max(abs(v(k, b)), 1e-12)
    else:
        f = lambda k: v(k, a) - v(k, b)
    e = A.base_mean(rows, f)
    label = f'{a} - {b}' + (f' [{metric}{"" if j is None else j}]' if metric != 'hv_normalized' else '')
    return A.summarize(label, e, endpoint=idx, metric=metric)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--out', default=str(HERE / 'out'))
    a = p.parse_args()
    run_dir = Path(a.out) / a.study
    rows = load(run_dir)
    res = dict(study=a.study, runs=len(rows), methods=collections.Counter(k[3] for k in rows))
    prim = A.family([contrast(rows, x, y) for x, y in PRIMARY], 'primary_holm4')
    chk = [contrast(rows, x, y) for x, y in CHECK]
    sec = [contrast(rows, x, y) for x, y in SECOND]
    sec += [contrast(rows, x, y, idx=-2) for x, y in PRIMARY]
    sec += [contrast(rows, x, y, metric='hv_wide_normalized') for x, y in PRIMARY]
    for jj, nm in enumerate(('makespan', 'energy', 'workload')):
        if jj == 2 and a.study == 'R4':
            continue
        sec += [contrast(rows, x, y, metric='objective_minima', minimize=True, j=jj) for x, y in PRIMARY[:2] + CHECK[:1]]
    desc = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, s in rows.items():
        e = s['endpoints'][-1]
        d = desc[k[3]]
        d['hv'].append(e['hv_normalized'])
        d['evaluations'].append(e['evaluations'])
        d['cpu_ratio'].append(s['cpu_ratio'])
        d['eval_share'].append(s['eval_seconds'] / max(s['search_seconds'], 1e-9))
        d['score_share'].append(s['score_seconds'] / max(s['search_seconds'], 1e-9))
        for kk, vv in s.get('extra', {}).items():
            d[kk].append(vv)
        for jj, nm in enumerate(('min_makespan', 'min_energy', 'min_workload')):
            d[nm].append(e['objective_minima'][jj])
    res['describe'] = {m: {k: (min(v) if k == 'cpu_ratio' else st.mean(v)) for k, v in d.items()} for m, d in desc.items()}
    for m, d in desc.items():
        res['describe'][m]['runs_cpu_ratio_below_0_9'] = sum(x < 0.9 for x in d['cpu_ratio'])
    res.update(primary=prim, reproduction_checks=chk, secondary=sec)
    (run_dir / f'analysis_{a.study}.json').write_text(json.dumps(res, indent=1, default=dict))
    print('== primary'); A.show(prim)
    print('== reproduction checks'); A.show(chk)
    print('== secondary'); A.show(sec)
    print(json.dumps(res['describe'], indent=1))


if __name__ == '__main__':
    main()
