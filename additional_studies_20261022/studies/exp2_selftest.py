"""Engineering self-test for exp2 (never formal).

  python exp2_selftest.py --release REL --lanes 8

1. Monte Carlo evaluator with unit multipliers reproduces the released evaluate() bitwise;
   the vectorized test evaluator equals the scalar one per scenario.
2. Reproduction gates against archived 2026-10-18 runs on the reference panel at 10,000 evaluations:
   RBP K=4 (S1 'always'), Random-four (S1 'r4'), NSGA-II, MOEA/D, NSGA-II+RBP (S2 eval arms).
"""
import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2_core as C  # noqa: E402

CELLS = [('comp30_small_high_01_a3_t030', 909302001), ('comp30_medium_low_02_a1_t060', 909302002)]
ARMS = {
    'always': (dict(family='rbp', params=dict(K=4, period=0)), 'period'),
    'r4': (dict(family='rbp', params=dict(K=4, period='uniform')), 'period'),
    'nsga2': (dict(family='nsga2', params=dict(pop=100, pc=.9, pm=1., op='std')), 'standard_eval'),
    'moead': (dict(family='moead', params=dict(H=12, T=10, delta=.9, nr=2, op='std')), 'standard_eval'),
    'nsga2_pre': (dict(family='nsga2', params=dict(pop=100, pc=.9, pm=1., op='std', K=4)), 'standard_eval'),
}


def stoch_check(release):
    C.setup(release)
    np = C.np
    from src.algorithm.nsga3.encoding import random_chromosome
    n_eval = n_batch = 0
    for r in json.loads((HERE / 'inputs' / 'INPUT_MANIFEST.json').read_text()):
        if r['panel'] != 'eng':
            continue
        inst = C.s24.restore_instance(json.loads((HERE / 'inputs' / r['file']).read_text()))
        model = C.StochModel(inst)
        rng = np.random.RandomState(7)
        chroms = C.compat.initialize_population(inst, 100, np.random.RandomState(11))[::5]
        chroms += [random_chromosome(inst, rng) for _ in range(20)]
        ones = np.ones((1, model.n, 2))
        for ch in chroms:
            fresh = C.chromosome(inst, C.genes(ch))
            a = fresh.objectives.to_array()
            b = model.objectives(fresh, ones).to_array()
            assert np.array_equal(a, b), (r['name'], a, b)
            n_eval += 1
        mult = C.scenarios(model.n, 'selftest/' + r['name'], 6)
        for ch in chroms[:10]:
            rows, sc, agv = model.plan(ch)
            parts = model._energy_parts(sc)
            one = np.asarray([model.one(rows, parts, sc, agv, m) for m in mult])
            vec = model.batch(ch, mult)
            assert np.array_equal(one, vec), (r['name'], np.abs(one - vec).max())
            n_batch += 1
    return dict(unit_multiplier_equalities=n_eval, vector_scalar_equalities=n_batch)


def one(args):
    release, name, seed, arm = args
    algo, _ = ARMS[arm]
    path = Path(release) / 'verification' / 'instances' / f'{name}.json'
    task = dict(study='selftest', id=f'selftest__{name}__{seed}__{arm}', arm=arm, panel='reference', name=name,
                base_id=name.rsplit('_', 2)[0], stratum='_'.join(name.split('_')[1:3]),
                regime='_'.join(name.rsplit('_', 2)[1:]), seed=seed, input=str(path), input_sha256=C.sha(path),
                algo=algo, evaluator=dict(kind='native'), mode='eval', endpoints=[2000, 5000, 10000])
    s = C.run_cell(task, release, str(HERE / 'out' / 'selftest'), None, False, 'study30_synthetic_v1')
    return arm, name, seed, [e['hv_normalized'] for e in s['endpoints']], s['search_seconds']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True)
    p.add_argument('--lanes', type=int, default=8)
    a = p.parse_args()
    os.chdir(HERE)
    rel = str(Path(a.release).resolve())
    report = dict(stochastic=stoch_check(rel))
    archived = {}
    base = Path(rel) / 'supplementary_studies_20261018' / 'results' / 'run_summaries'
    for kind in ('period', 'standard_eval'):
        for line in gzip.open(base / f'run_summaries_{kind}.jsonl.gz', 'rt'):
            d = json.loads(line)
            ends = [e['hv_normalized'] for e in d['endpoints']]
            archived[(kind, d['name'], d['seed'], d['arm'])] = ends
    jobs = [(rel, n, s, arm) for n, s in CELLS for arm in ARMS]
    with ProcessPoolExecutor(a.lanes) as ex:
        res = list(ex.map(one, jobs))
    rows = []
    for arm, name, seed, hv, secs in res:
        ref = archived[(ARMS[arm][1], name, seed, arm)]
        rows.append(dict(arm=arm, name=name, seed=seed, hv=hv, archived=ref, exact=hv == ref, seconds=secs))
    report['reproduction'] = rows
    report['status'] = 'PASS' if all(r['exact'] for r in rows) else 'CHECK'
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'SELFTEST.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
