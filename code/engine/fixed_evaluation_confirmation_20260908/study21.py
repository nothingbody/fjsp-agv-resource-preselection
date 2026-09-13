"""Prospectively fixed design; no scientific work at import time."""
import collections
import random

PROTOCOL = 'fixed_evaluation_confirmation_v1_20260908'
ARMS = ('ucb', 'de_plain', 'de_random4', 'de_proxy')
SEEDS = (9082101, 9082102, 9082103, 9082104, 9082105)
INSTANCES = tuple([('brandimarte', f'Mk{i:02}.fjs') for i in (2, 4, 6, 10)] +
    [('hurink_edata', f'la{i:02}.fjs') for i in (4, 16, 28, 37)] +
    [('hurink_rdata', f'la{i:02}.fjs') for i in (8, 14, 23, 32)] +
    [('hurink_vdata', f'la{i:02}.fjs') for i in (5, 17, 26, 38)])
CPUS = tuple(range(64, 96, 2))
EVALUATIONS = 10000
ENDPOINTS = (2000, 5000, 10000)
PRIMARY = (('de_proxy', 'ucb'), ('de_proxy', 'de_random4'))
DESCRIPTIVE = PRIMARY + (('de_proxy', 'de_plain'), ('de_random4', 'de_plain'),
    ('de_random4', 'ucb'), ('de_plain', 'ucb'))
SEQUENCES = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))


def make_plan():
    rng = random.Random(9082107)
    instances = list(INSTANCES); rng.shuffle(instances)
    labels = list(ARMS); rng.shuffle(labels)
    orders = []
    for _ in SEEDS:
        row = list(range(4)) * 4; rng.shuffle(row); orders.append(row)
    tasks = []
    for ci, (cpu, (dataset, name)) in enumerate(zip(CPUS, instances)):
        seeds = list(SEEDS); rng.shuffle(seeds)
        for rnd, seed in enumerate(seeds):
            order = [labels[j] for j in SEQUENCES[orders[rnd][ci]]]
            for pos, arm in enumerate(order):
                task = dict(dataset=dataset, name=name, seed=seed, arm=arm,
                    evaluations=EVALUATIONS, endpoints=list(ENDPOINTS), wave=rnd*4+pos,
                    block=ci*5+rnd, position=pos, cpu=cpu)
                task['id'] = f'{dataset}_{name[:-4]}_{seed}_{arm}'
                tasks.append(task)
    return sorted(tasks, key=lambda t: (t['wave'], t['cpu']))


def verify_plan(tasks):
    assert len(tasks) == len({t['id'] for t in tasks}) == 320
    assert {(t['dataset'], t['name'], t['seed'], t['arm']) for t in tasks} == {
        (d, n, s, a) for d, n in INSTANCES for s in SEEDS for a in ARMS}
    positions = collections.Counter((t['position'], t['arm']) for t in tasks)
    assert len(positions) == 16 and set(positions.values()) == {20}
    pairs = collections.Counter()
    for block in range(80):
        cells = sorted([t for t in tasks if t['block'] == block], key=lambda t:t['position'])
        assert len(cells) == 4 and len({t['cpu'] for t in cells}) == 1
        assert len({(t['dataset'], t['name'], t['seed']) for t in cells}) == 1
        pairs.update((a['arm'], b['arm']) for a, b in zip(cells, cells[1:]))
    assert len(pairs) == 12 and set(pairs.values()) == {20}
    for wave in range(20):
        cells = [t for t in tasks if t['wave'] == wave]
        assert len(cells) == 16 and {t['cpu'] for t in cells} == set(CPUS)
        assert all(t['evaluations'] == EVALUATIONS and t['endpoints'] == list(ENDPOINTS) for t in cells)
    return dict(cells=320, instances=16, seeds=5, blocks=80, waves=20,
        concurrency=16, position_repetitions=20, directed_precedence_repetitions=20)
