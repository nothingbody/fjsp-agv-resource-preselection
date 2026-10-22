"""exp2 analysis: base-level paired effects, stratified bootstrap, exact sign flips, Holm (PROTOCOL.md section 5).

  python exp2_analyze.py --study B2e|B2t|Ce|Ct|A|B3|B1 --out out
"""
import argparse
import collections
import gzip
import itertools
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = 100_000
BASES = 16


def load(run_dir):
    rows = {}
    for f in Path(run_dir, 'runs').glob('*.json.gz'):
        s = json.loads(gzip.decompress(f.read_bytes()))['summary']
        s['_ends'] = {e['endpoint']: e for e in s['endpoints']}
        rows[(s['base_id'], s['regime'], s['seed'], s['arm'])] = s
    return rows


def stratum(base):
    return '_'.join(base.split('_')[1:3])


def cell_keys(rows):
    return sorted({k[:3] for k in rows})


def val(rows, key, arm, end, metric):
    return rows[key + (arm,)]['_ends'][end][metric]


def base_mean(rows, f):
    cells = collections.defaultdict(list)
    for key in cell_keys(rows):
        cells[key[0]].append(f(key))
    assert len(cells) == BASES and all(len(v) == 15 for v in cells.values()), {k: len(v) for k, v in cells.items()}
    return {k: st.mean(v) for k, v in sorted(cells.items())}


def boot(effects, stratified, seed=20261019):
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for k, v in effects.items():
        groups[stratum(k) if stratified else 'all'].append(v)
    means = []
    for _ in range(B):
        means.append(st.mean(st.mean(rng.choice(g) for _ in g) for g in groups.values()))
    means.sort()
    return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]


def signflip(values):
    obs = abs(sum(values))
    hit = 0
    for signs in itertools.product((1, -1), repeat=len(values)):
        if abs(sum(s * v for s, v in zip(signs, values))) >= obs - 1e-15:
            hit += 1
    return hit / 2 ** len(values)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def summarize(label, effects, **meta):
    v = list(effects.values())
    return dict(contrast=label, **meta, mean=st.mean(v), ci95_stratified=boot(effects, True),
                ci95_unstratified=boot(effects, False), positive=sum(x > 0 for x in v),
                zero=sum(x == 0 for x in v), negative=sum(x < 0 for x in v), p_signflip=signflip(v),
                base_effects=effects)


def contrast(rows, a, b, end, metric='hv_normalized'):
    e = base_mean(rows, lambda k: val(rows, k, a, end, metric) - val(rows, k, b, end, metric))
    return summarize(f'{a} - {b}', e, endpoint=end, metric=metric)


def interaction(rows, a1, b1, a2, b2, end, metric='hv_normalized'):
    e = base_mean(rows, lambda k: (val(rows, k, a1, end, metric) - val(rows, k, b1, end, metric))
                  - (val(rows, k, a2, end, metric) - val(rows, k, b2, end, metric)))
    return summarize(f'[{a1} - {b1}] - [{a2} - {b2}]', e, endpoint=end, metric=metric)


def family(items, name):
    for c, ph in zip(items, holm([c['p_signflip'] for c in items])):
        c['p_holm'] = ph
        c['family'] = name
    return items


def describe(rows, end, metric='hv_normalized'):
    m = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, s in rows.items():
        arm = k[3]
        e = s['_ends'][end]
        m[arm]['hv'].append(e[metric])
        m[arm]['evaluations'].append(e['evaluations'])
        m[arm]['cpu_ratio'].append(s['cpu_ratio'])
        m[arm]['eval_share'].append(s['eval_seconds'] / max(s['search_seconds'], 1e-9))
        m[arm]['score_share'].append(s['score_seconds'] / max(s['search_seconds'], 1e-9))
        m[arm]['makespan_min'].append(e['objective_minima'][0])
        if 'hv_test' in e:
            m[arm]['hv_test'].append(e['hv_test'])
    out = {}
    for arm, d in sorted(m.items()):
        out[arm] = {k: (min(v) if k == 'cpu_ratio' else st.mean(v)) for k, v in d.items()}
        out[arm]['runs'] = len(d['hv'])
        out[arm]['runs_cpu_ratio_below_0_9'] = sum(x < 0.9 for x in d['cpu_ratio'])
    return out


def strip(rows):
    for s in rows.values():
        s.pop('_ends', None)


def show(items):
    for c in items:
        ph = c.get('p_holm')
        print(f"{c['contrast']:<55} {c['endpoint']!s:>6} {c['metric'][:10]:<10} {c['mean']:+.6f} "
              f"[{c['ci95_stratified'][0]:+.6f}, {c['ci95_stratified'][1]:+.6f}] "
              f"{c['positive']}/{c['zero']}/{c['negative']} p={c['p_signflip']:.3g}"
              + (f" holm={ph:.3g}" if ph is not None else ''))


def analyze(study, out_root):
    run_dir = Path(out_root) / study
    rows = load(run_dir)
    res = dict(study=study, runs=len(rows), arms=collections.Counter(k[3] for k in rows))
    prim, sec = [], []
    if study in ('B2e', 'B2t'):
        end, others = (10000, (2000, 5000)) if study == 'B2e' else (120.0, (60.0,))
        P = [('rbp', 'nsga2t'), ('rbp', 'moeadt'), ('nsga2t_rbp', 'nsga2t'), ('moeadt_rbp', 'moeadt'), ('rbp', 'r4')]
        S = [('nsga2t', 'nsga2_def'), ('moeadt', 'moead_def'), ('rbp', 'nsga2_def'), ('rbp', 'moead_def'),
             ('moeadt', 'nsga2t'), ('nsga2t_rbp', 'rbp'), ('moeadt_rbp', 'rbp'), ('r4', 'nsga2t'), ('r4', 'moeadt')]
        prim = family([contrast(rows, a, b, end) for a, b in P], 'primary_holm5')
        sec = [contrast(rows, a, b, end) for a, b in S]
        sec += [contrast(rows, a, b, e) for a, b in P for e in others]
        sec += [contrast(rows, a, b, end, 'hv_wide_normalized') for a, b in P]
        res['describe'] = describe(rows, end)
    elif study == 'Ce':
        end = 10000
        for host in ('rbp', 'nsga2t'):
            sec += [contrast(rows, f'{host}K{k:02}', f'{host}K01', end) for k in (2, 4, 8, 16)]
            sec += [contrast(rows, f'{host}K{k:02}', f'{host}K04', end) for k in (8, 16)]
        res['describe'] = describe(rows, end)
        b2 = load(Path(out_root) / 'B2e')
        pairs = [('rbpK04', 'rbp'), ('nsga2tK01', 'nsga2t'), ('nsga2tK04', 'nsga2t_rbp')]
        n = bad = 0
        for k, s in rows.items():
            for x, y in pairs:
                if k[3] == x and k[:3] + (y,) in b2:
                    n += 1
                    o = b2[k[:3] + (y,)]
                    bad += [e['hv_normalized'] for e in s['endpoints']] != [e['hv_normalized'] for e in o['endpoints']]
        res['reproduction_vs_B2e'] = dict(compared_runs=n, mismatched=bad, status='PASS' if n and not bad else 'CHECK')
        strip(b2)
    elif study == 'Ct':
        end = 120.0
        for host in ('rbp', 'nsga2t'):
            prim.append(interaction(rows, f'{host}K16@d25', f'{host}K04@d25', f'{host}K16@d00', f'{host}K04@d00', end))
            prim.append(interaction(rows, f'{host}K04@d25', f'{host}K01@d25', f'{host}K04@d00', f'{host}K01@d00', end))
        prim = family(prim, 'primary_holm4')
        for host in ('rbp', 'nsga2t'):
            for d in ('d00', 'd25'):
                sec += [contrast(rows, f'{host}K{k:02}@{d}', f'{host}K01@{d}', end) for k in (2, 4, 8, 16)]
                sec += [contrast(rows, f'{host}K{k:02}@{d}', f'{host}K04@{d}', end) for k in (8, 16)]
                sec += [contrast(rows, f'{host}K{k:02}@{d}', f'{host}K01@{d}', 60.0) for k in (4, 16)]
        res['describe'] = describe(rows, end)
    elif study == 'A':
        end, metric = 300.0, 'hv_test'
        hosts = [('rbp', 'r4'), ('nsga2t_rbp', 'nsga2t'), ('moeadt_rbp', 'moeadt')]
        prim = [contrast(rows, f'{a}@R64', f'{b}@R64', end, metric) for a, b in hosts]
        prim += [interaction(rows, f'{a}@R64', f'{b}@R64', f'{a}@R01', f'{b}@R01', end, metric) for a, b in hosts]
        prim = family(prim, 'primary_holm6')
        for a, b in hosts:
            for R in ('R01', 'R08'):
                sec.append(contrast(rows, f'{a}@{R}', f'{b}@{R}', end, metric))
            for R in ('R01', 'R08', 'R64'):
                sec.append(contrast(rows, f'{a}@{R}', f'{b}@{R}', 150.0, metric))
                sec.append(contrast(rows, f'{a}@{R}', f'{b}@{R}', end, 'hv_normalized'))
                sec.append(contrast(rows, f'{a}@{R}', f'{b}@{R}', end, 'hv_test_wide'))
            sec.append(interaction(rows, f'{a}@R08', f'{b}@R08', f'{a}@R01', f'{b}@R01', end, metric))
        for R in ('R01', 'R08', 'R64'):
            sec += [contrast(rows, f'rbp@{R}', f'{x}@{R}', end, metric) for x in ('nsga2t', 'moeadt')]
        res['describe'] = describe(rows, end, 'hv_normalized')
    else:
        raise ValueError(study)
    res['primary'] = prim
    res['secondary'] = sec
    strip(rows)
    (run_dir / f'analysis_{study}.json').write_text(json.dumps(res, indent=1, default=dict))
    print('== primary')
    show(prim)
    print('== secondary')
    show(sec)
    print(json.dumps(res.get('describe'), indent=1))
    if 'reproduction_vs_B2e' in res:
        print(res['reproduction_vs_B2e'])


def b3(out_root):
    bounds = {}
    for line in (Path(out_root) / 'B3' / 'bounds.jsonl').open():
        d = json.loads(line)
        bounds[d['name']] = d
    res = dict(instances=len(bounds), status=collections.Counter(d['status'] for d in bounds.values()))
    for study, end in (('B2e', 10000), ('B2t', 120.0)):
        rows = load(Path(out_root) / study)
        gaps = collections.defaultdict(list)
        by_stratum = collections.defaultdict(lambda: collections.defaultdict(list))
        for k, s in rows.items():
            lb = bounds[s['name']]['lower_bound']
            g = (s['_ends'][end]['objective_minima'][0] - lb) / lb
            gaps[k[3]].append(g)
            by_stratum[k[3]][stratum(k[0])].append(g)
        res[study] = {a: dict(mean_gap=st.mean(v), median_gap=st.median(v),
                              by_stratum={s: st.mean(x) for s, x in sorted(by_stratum[a].items())})
                      for a, v in sorted(gaps.items())}
        strip(rows)
    (Path(out_root) / 'B3' / 'analysis_B3.json').write_text(json.dumps(res, indent=1, default=dict))
    print(json.dumps(res, indent=1, default=dict))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--out', default=str(HERE / 'out'))
    a = p.parse_args()
    if a.study == 'B3':
        b3(a.out)
    else:
        analyze(a.study, a.out)


if __name__ == '__main__':
    main()
