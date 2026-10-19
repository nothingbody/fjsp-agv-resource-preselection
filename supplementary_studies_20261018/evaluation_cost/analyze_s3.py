"""S3 analysis (see PROTOCOL.md). Reuses S1 statistics and the S2 loader.

  python analyze_s3.py --s3 RUN_DIR --s2-time S2T_RUN_DIR --s1-code S1_DIR --s2-code S2_DIR
"""
import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

ARMS = {'always', 'nsga2', 'nsga2_pre', 'moead'}
DELAYS = (0, 10, 25, 50)
PLUG = ('nsga2_pre', 'nsga2')
SECONDARY = [('always', 'nsga2'), ('always', 'moead')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--s3', required=True)
    p.add_argument('--s2-time', required=True)
    p.add_argument('--s1-code', required=True)
    p.add_argument('--s2-code', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.s1_code)
    sys.path.insert(0, a.s2_code)
    import analyze_s1 as A
    from analyze_s2 import load

    def hv(s, ends, metric):
        return s['_ends'][ends][metric]
    A.hv = hv

    rows = {0: load(a.s2_time, 'endpoint', arms=ARMS)}
    for d in DELAYS[1:]:
        rows[d] = load(Path(a.s3, f'd{d:03}'), 'endpoint')

    def summarize(name, e):
        v = list(e.values())
        return dict(contrast=name, mean=st.mean(v), ci95_stratified=A.boot(e, True),
                    ci95_unstratified=A.boot(e, False), positive=sum(x > 0 for x in v),
                    zero=sum(x == 0 for x in v), negative=sum(x < 0 for x in v),
                    p_signflip=A.signflip(v), base_effects=e)

    def at(d, x, y, end=120.0, metric='hv_normalized'):
        return A.base_effects(rows[d], x, y, end, metric)

    def trend(x, y, end=120.0, metric='hv_normalized'):
        hi, lo = at(50, x, y, end, metric), at(0, x, y, end, metric)
        return {k: hi[k] - lo[k] for k in hi}

    out = dict(runs={d: len(r) for d, r in rows.items()},
               arms={d: collections.Counter(k[3] for k in r) for d, r in rows.items()})
    prim = [summarize(f'nsga2_pre-nsga2@{d}ms', at(d, *PLUG)) for d in DELAYS[1:]]
    prim.append(summarize('trend nsga2_pre-nsga2 (50ms-0ms)', trend(*PLUG)))
    for c, ph in zip(prim, A.holm([c['p_signflip'] for c in prim])):
        c['p_holm4'] = ph
    out['primary'] = prim
    sec = [summarize(f'nsga2_pre-nsga2@0ms', at(0, *PLUG))]
    for x, y in SECONDARY:
        sec += [summarize(f'{x}-{y}@{d}ms', at(d, x, y)) for d in DELAYS]
        sec.append(summarize(f'trend {x}-{y} (50ms-0ms)', trend(x, y)))
    for x, y in [PLUG] + SECONDARY:
        sec += [summarize(f'{x}-{y}@{d}ms 60s', at(d, x, y, 60.0)) for d in DELAYS]
        sec += [summarize(f'{x}-{y}@{d}ms wide', at(d, x, y, 120.0, 'hv_wide_normalized')) for d in DELAYS]
    out['secondary'] = sec
    out['mean_hv'] = {d: {arm: st.mean(s['_ends'][120.0]['hv_normalized'] for k, s in r.items() if k[3] == arm)
                          for arm in sorted(ARMS)} for d, r in rows.items()}
    out['mean_evaluations'] = {d: {arm: st.mean(s['evaluations'] for k, s in r.items() if k[3] == arm)
                                   for arm in sorted(ARMS)} for d, r in rows.items()}
    out['min_cpu_ratio'] = {d: {arm: min(s['cpu_ratio'] for k, s in r.items() if k[3] == arm)
                                for arm in sorted(ARMS)} for d, r in rows.items()}
    Path(a.s3, 'analysis_s3.json').write_text(json.dumps(out, indent=2, default=dict))
    for c in prim + sec[:11]:
        print(c['contrast'], f"{c['mean']:.6f}", [round(x, 6) for x in c['ci95_stratified']],
              f"{c['positive']}/{c['zero']}/{c['negative']}", f"p={c['p_signflip']:.6g}",
              f"holm4={c['p_holm4']:.6g}" if 'p_holm4' in c else '')
    for d in DELAYS:
        print(d, 'ms HV', {k: round(v, 4) for k, v in out['mean_hv'][d].items()},
              'evals', {k: round(v) for k, v in out['mean_evaluations'][d].items()})


if __name__ == '__main__':
    main()
