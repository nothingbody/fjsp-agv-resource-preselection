"""S2 analysis (see PROTOCOL.md). Reuses S1 statistics.

  python analyze_s2.py --mode eval --s2 RUN_DIR --s1 S1_RUN_DIR --s1-code S1_CODE_DIR
  python analyze_s2.py --mode time --s2 RUN_DIR --s1-code S1_CODE_DIR
"""
import argparse
import collections
import gzip
import json
import statistics as st
import sys
from pathlib import Path

PRIMARY = [('always', 'nsga2'), ('always', 'moead'), ('p4', 'nsga2'), ('p4', 'moead'), ('nsga2_pre', 'nsga2')]
SECONDARY = [('nsga2_pre', 'always'), ('moead', 'nsga2'), ('nsga2_pre', 'p4')]


def load(run_dir, key, arms=None):
    rows = {}
    for f in Path(run_dir, 'runs').glob('*.json.gz'):
        s = json.loads(gzip.decompress(f.read_bytes()))['summary']
        if arms and s['arm'] not in arms:
            continue
        ends = {}
        for e in s['endpoints']:
            ends[e.get('endpoint', e.get('evaluations'))] = e
        s['_ends'] = ends
        rows[(s['base_id'], s['regime'], s['seed'], s['arm'])] = s
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('eval', 'time'), required=True)
    p.add_argument('--s2', required=True)
    p.add_argument('--s1')
    p.add_argument('--s1-code', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.s1_code)
    import analyze_s1 as A

    rows = load(a.s2, 'endpoint')
    if a.mode == 'eval':
        rows.update(load(a.s1, 'evaluations', arms={'p4', 'always'}))
        main_end, others = 10000, (2000, 5000)
    else:
        main_end, others = 120.0, (60.0,)

    def hv(s, ends, metric):
        return s['_ends'][ends][metric]
    A.hv = hv  # patch endpoint lookup

    out = dict(mode=a.mode, runs=len(rows), arms=collections.Counter(k[3] for k in rows))
    prim = [A.contrast(rows, x, y, main_end) for x, y in PRIMARY]
    for c, ph in zip(prim, A.holm([c['p_signflip'] for c in prim])):
        c['p_holm5'] = ph
    out['primary'] = prim
    sec = [A.contrast(rows, x, y, main_end) for x, y in SECONDARY]
    sec += [A.contrast(rows, x, y, e) for x, y in PRIMARY + SECONDARY for e in others]
    sec += [A.contrast(rows, x, y, main_end, 'hv_wide_normalized') for x, y in PRIMARY]
    out['secondary'] = sec
    m = collections.defaultdict(list)
    ev = collections.defaultdict(list)
    cpu = collections.defaultdict(list)
    for k, s in rows.items():
        m[k[3]].append(hv(s, main_end, 'hv_normalized'))
        ev[k[3]].append(s['_ends'][main_end].get('evaluations', main_end))
        if 'cpu_ratio' in s:
            cpu[k[3]].append(s['cpu_ratio'])
    out['mean_hv'] = {k: st.mean(v) for k, v in m.items()}
    out['mean_evaluations'] = {k: st.mean(v) for k, v in ev.items()}
    out['min_cpu_ratio'] = {k: min(v) for k, v in cpu.items()}
    for s in rows.values():
        s.pop('_ends', None)
    Path(a.s2, f'analysis_s2_{a.mode}.json').write_text(json.dumps(out, indent=2, default=dict))
    for c in prim:
        print(c['contrast'], f"{c['mean']:.6f}", [round(x, 6) for x in c['ci95_stratified']],
              f"{c['positive']}/{c['zero']}/{c['negative']}", f"holm5={c['p_holm5']:.6g}")
    print('mean HV', {k: round(v, 5) for k, v in out['mean_hv'].items()})
    print('mean evaluations', {k: round(v, 1) for k, v in out['mean_evaluations'].items()})
    print('min cpu ratio', out['min_cpu_ratio'])


if __name__ == '__main__':
    main()
