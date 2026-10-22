"""exp3 task lists and locks.

  python exp3_tasks.py make --study R1|R2|R3|R4|ENG3
  python exp3_tasks.py lock --study R1
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2_core as C  # noqa: E402

LOCKED = ('exp2_core.py', 'exp3_rivals.py', 'exp3_run.py', 'exp3_tasks.py', 'exp3_selftest.py', 'PROTOCOL_exp3.md',
          'inputs/INPUT_MANIFEST.json', 'SELECTION.json', 'ENV_FREEZE.txt')
ARMS = ('rbp', 'nsga2t', 'dqnma', 'dqnma_rbp', 'ilmpea', 'ilmpea_rbp')


def methods():
    sel = json.loads((HERE / 'SELECTION.json').read_text())
    return dict(rbp=dict(family='rbp', params=dict(K=4, period=0)),
                nsga2t=dict(family='nsga2', params=dict(sel['nsga2']['params'])),
                dqnma=dict(family='dqnma', params={}), dqnma_rbp=dict(family='dqnma', params=dict(K=4)),
                ilmpea=dict(family='ilmpea', params={}), ilmpea_rbp=dict(family='ilmpea', params=dict(K=4)))


def make(study):
    m = methods()
    rows = json.loads((HERE / 'inputs' / 'INPUT_MANIFEST.json').read_text())
    panel = 'eng' if study == 'ENG3' else 'test'
    out = []
    for r in rows:
        if r['panel'] != panel:
            continue
        no = r['operations']
        for seed in C.PANELS[panel]['seeds']:
            for arm in ARMS:
                specs = []
                if study == 'R1':
                    specs = [(3, 'time', [float(no), float(2 * no)], arm)]
                elif study == 'R2':
                    specs = [(3, 'time', [60.0, 120.0], arm)]
                elif study == 'R3':
                    specs = [(3, 'eval', [5000, 10000, 20000], arm)]
                elif study == 'R4':
                    specs = [(2, 'time', [float(no), float(2 * no)], arm)]
                elif study == 'ENG3':
                    specs = [(3, 'eval', [1000, 2000], arm + '@e'), (3, 'time', [10.0, 20.0], arm + '@t'),
                             (2, 'time', [10.0, 20.0], arm + '@b')]
                for dims, mode, ends, label in specs:
                    out.append(dict(study=study, id=f"{study}__{r['name']}__{seed}__{label}", arm=label, method=arm,
                                    panel=panel, name=r['name'], base_id=r['base_id'], stratum=r['stratum'],
                                    regime=r['regime'], seed=seed, input='inputs/' + r['file'], input_sha256=r['sha256'],
                                    algo=m[arm], objectives=dims, mode=mode, endpoints=ends, operations=no))
    if study == 'R3':
        out.sort(key=lambda t: 0 if 'medium' in t['name'] else 1)
    else:
        random.Random(C.hash32(f'exp3/order/{study}')).shuffle(out)
    return out


def lock(study):
    files = {rel: C.sha(HERE / rel) for rel in LOCKED}
    tf = HERE / 'tasks' / f'{study}.json'
    rec = dict(protocol='exp3_rivals_20261020_v1', study=study, files=files, tasks_file=f'tasks/{study}.json',
               tasks_sha256=C.sha(tf), tasks=len(json.loads(tf.read_text())), created_utc=C.utc(),
               formal_runs_before_lock=0)
    p = HERE / 'locks' / f'LOCK_{study}.json'
    assert not p.exists()
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('make', 'lock'))
    p.add_argument('--study', required=True)
    a = p.parse_args()
    if a.cmd == 'make':
        tl = make(a.study)
        assert len({t['id'] for t in tl}) == len(tl)
        (HERE / 'tasks').mkdir(exist_ok=True)
        (HERE / 'tasks' / f'{a.study}.json').write_text(json.dumps(tl, indent=0))
        print(a.study, len(tl), collections.Counter(t['method'] for t in tl))
    else:
        lock(a.study)


if __name__ == '__main__':
    main()
