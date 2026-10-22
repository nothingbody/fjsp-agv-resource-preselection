"""exp3 engineering self-test (never formal).

  python exp3_selftest.py --release REL
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2_core as C  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True)
    a = p.parse_args()
    os.chdir(HERE)
    C.setup(str(Path(a.release).resolve()))
    import exp3_rivals as R
    np = C.np
    rep = dict(critical_path=0, operators=0, greedy=0, mlp=None)
    for r in json.loads(Path('inputs/INPUT_MANIFEST.json').read_text()):
        if r['panel'] != 'eng':
            continue
        inst = C.s24.restore_instance(json.loads(Path('inputs/' + r['file']).read_text()))
        prob = R.Problem(inst)
        rng = np.random.RandomState(3)
        pop = C.compat.initialize_population(inst, 100, np.random.RandomState(5))
        sols = pop[::10] + [prob.random(rng) for _ in range(10)]
        for ch in sols:
            sch = C.compat.decoding.decode(ch)
            path = R.critical_path(ch, prob)
            kind, (i, j), _ = path[-1]
            assert kind == 'P' and abs(sch.op_end[(i, j)] - max(sch.op_end.values())) < 1e-9
            # every link of the chain is tight
            for (k1, o1, r1), (k2, o2, r2) in zip(path[:-1], path[1:]):
                end1 = sch.op_end[o1] if k1 == 'P' else sch.transport_end[o1]
                start2 = sch.op_start[o2] if k2 == 'P' else sch.transport_start[o2]
                assert start2 >= end1 - 1e-9, (r['name'], k1, o1, k2, o2, end1, start2)
            rep['critical_path'] += 1
            d = R.DQNMA.__new__(R.DQNMA)
            d.rng, d.prob, d.inst = rng, prob, inst
            ref = sorted(ch.os.tolist())
            for act in range(4):
                c = d._local(ch, act)
                assert sorted(c.os.tolist()) == ref and np.all(c.ma < prob.choices)
                assert np.all(c.agv_assign < prob.V) and np.all(c.agv_speed < prob.S)
                c.objectives
                rep['operators'] += 1
            il = R.ILMPEA.__new__(R.ILMPEA)
            il.rng, il.prob, il.inst = rng, prob, inst
            for act in range(6):
                c = il._apply(act, ch, sols[0])
                assert sorted(c.os.tolist()) == ref and np.all(c.ma < prob.choices)
                c.objectives
                rep['operators'] += 1
            for mode in ('C', 'E'):
                g = il._greedy(ch.os, mode)
                assert np.all(g.ma < prob.choices)
                g.objectives
                rep['greedy'] += 1
    # MLP learns a separable toy problem
    rng = np.random.RandomState(0)
    X = rng.normal(size=(512, 20))
    Y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
    net = R.MLP([20, 64, 32, 4], rng, 1e-3)

    def loss():
        lg, _ = net.forward(X)
        z = lg - lg.max(axis=1, keepdims=True)
        pr = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return float(-np.log(pr[np.arange(len(Y)), Y] + 1e-12).mean()), float((pr.argmax(axis=1) == Y).mean())
    l0 = loss()
    for _ in range(300):
        idx = rng.choice(512, 32, replace=False)
        lg, acts = net.forward(X[idx])
        z = lg - lg.max(axis=1, keepdims=True)
        pr = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        pr[np.arange(32), Y[idx]] -= 1
        net.backward(acts, pr / 32)
    rep['mlp'] = dict(before=l0, after=loss())
    assert rep['mlp']['after'][1] > 0.9
    rep['status'] = 'PASS'
    Path('out').mkdir(exist_ok=True)
    Path('out/SELFTEST_exp3.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
