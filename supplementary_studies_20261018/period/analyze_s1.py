"""S1 analysis: base-level paired HV effects, stratified bootstrap, exact sign flips, Holm.

  python analyze_s1.py RUN_DIR [--archived primary_run_metrics.json.gz]
"""
import argparse
import collections
import gzip
import itertools
import json
import random
import statistics as st
from pathlib import Path

STRATA = ('small_low', 'small_high', 'medium_low', 'medium_high')
PRIMARY = [('always', 'p4'), ('p8', 'p4'), ('p4', 'r4')]
SECONDARY = [('always', 'p8'), ('always', 'r4')]
B = 100_000


def load(run_dir):
    rows = {}
    for f in Path(run_dir, 'runs').glob('*.json.gz'):
        s = json.loads(gzip.decompress(f.read_bytes()))['summary']
        rows[(s['base_id'], s['regime'], s['seed'], s['arm'])] = s
    return rows


def hv(s, ends, metric):
    return next(e[metric] for e in s['endpoints'] if e['evaluations'] == ends)


def base_effects(rows, a, b, ends, metric='hv_normalized'):
    cells = collections.defaultdict(list)
    keys = {(k[0], k[1], k[2]) for k in rows}
    for base, reg, seed in keys:
        if (base, reg, seed, a) in rows and (base, reg, seed, b) in rows:
            cells[base].append(hv(rows[base, reg, seed, a], ends, metric) - hv(rows[base, reg, seed, b], ends, metric))
    assert len(cells) == 16 and all(len(v) == 15 for v in cells.values()), {k: len(v) for k, v in cells.items()}
    return {k: st.mean(v) for k, v in sorted(cells.items())}


def stratum(base):
    return '_'.join(base.split('_')[1:3])


def boot(effects, stratified, seed=20261018):
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for k, v in effects.items():
        groups[stratum(k) if stratified else 'all'].append(v)
    means = []
    for _ in range(B):
        if stratified:
            means.append(st.mean(st.mean(rng.choice(g) for _ in g) for g in groups.values()))
        else:
            g = groups['all']
            means.append(st.mean(rng.choice(g) for _ in g))
    means.sort()
    return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]


def signflip(values):
    obs = abs(sum(values))
    n = len(values)
    hit = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(sum(s * v for s, v in zip(signs, values))) >= obs - 1e-15:
            hit += 1
    return hit / 2 ** n


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def contrast(rows, a, b, ends, metric='hv_normalized'):
    e = base_effects(rows, a, b, ends, metric)
    v = list(e.values())
    return dict(contrast=f'{a}-{b}', endpoint=ends, metric=metric, mean=st.mean(v),
                ci95_stratified=boot(e, True), ci95_unstratified=boot(e, False),
                positive=sum(x > 0 for x in v), zero=sum(x == 0 for x in v), negative=sum(x < 0 for x in v),
                p_signflip=signflip(v), base_effects=e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--archived')
    a = p.parse_args()
    rows = load(a.run_dir)
    out = dict(runs=len(rows), arms=collections.Counter(k[3] for k in rows))
    prim = [contrast(rows, x, y, 10000) for x, y in PRIMARY]
    for c, ph in zip(prim, holm([c['p_signflip'] for c in prim])):
        c['p_holm3'] = ph
    out['primary'] = prim
    sec = [contrast(rows, x, y, 10000) for x, y in SECONDARY]
    sec += [contrast(rows, x, y, e) for x, y in PRIMARY + SECONDARY for e in (2000, 5000)]
    sec += [contrast(rows, x, y, 10000, 'hv_wide_normalized') for x, y in PRIMARY]
    out['secondary'] = sec
    means = collections.defaultdict(list)
    steps = collections.defaultdict(list)
    for k, s in rows.items():
        means[k[3]].append(hv(s, 10000, 'hv_normalized'))
        steps[k[3]].append(s['scored_steps'] / max(1, s['steps']))
    out['mean_hv_10k'] = {k: st.mean(v) for k, v in means.items()}
    out['scored_fraction'] = {k: st.mean(v) for k, v in steps.items()}
    if a.archived:
        old = json.loads(gzip.decompress(Path(a.archived).read_bytes()))
        m = {'de_proxy': 'p4', 'de_random4': 'r4'}
        diffs, n = 0.0, 0
        for r in old:
            if r['arm'] in m and r['evaluations'] in (2000, 5000, 10000):
                k = (r['base_id'], r['regime'], r['seed'], m[r['arm']])
                if k in rows:
                    diffs = max(diffs, abs(hv(rows[k], r['evaluations'], 'hv_normalized') - r['hv_normalized']))
                    n += 1
        out['replication'] = dict(compared_endpoints=n, max_abs_diff=diffs, status='PASS' if n and diffs <= 1e-12 else 'CHECK')
    Path(a.run_dir, 'analysis_s1.json').write_text(json.dumps(out, indent=2, default=dict))
    for c in prim:
        print(c['contrast'], f"{c['mean']:.6f}", [round(x, 6) for x in c['ci95_stratified']],
              f"{c['positive']}/{c['zero']}/{c['negative']}", f"p={c['p_signflip']:.6g}", f"holm={c['p_holm3']:.6g}")
    print('mean HV 10k', out['mean_hv_10k'])
    print('replication', out.get('replication'))


if __name__ == '__main__':
    main()
