"""exp3b harness: RBP versus adapted, tuned DQNMA and ILMPEA (see PROTOCOL_exp3b.md).

  python exp3_run.py --release REL --tasks tasks/R1.json --out out/R1 --lanes 28 [--lock locks/LOCK_R1.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2_core as C  # noqa: E402

PROTOCOL = 'exp3b_adapted_20261021_v1'
CODE_FILES = ('exp2_core.py', 'exp3b_rivals.py', 'exp3b_run.py')


class BiObjective:
    """Wrap the installed exact evaluator so the workload range is reported as zero."""

    def __enter__(self):
        self.prev = C.compat.decoding.evaluate
        prev = self.prev

        def ev(ch, *a, **kw):
            r = prev(ch, *a, **kw)
            return C.compat.decoding.Objectives(makespan=r.makespan, total_energy=r.total_energy, workload_balance=0.0)
        C.compat.decoding.evaluate = ev
        return self

    def __exit__(self, *a):
        C.compat.decoding.evaluate = self.prev


class ZeroLoadFeatures:
    def __init__(self, base):
        self.base = base
        self.choices = base.choices

    def __call__(self, ch):
        x, p = self.base(ch)
        p = p.copy()
        p[2] = 0.0
        return x, p


def patch_features(bi):
    """In two-objective mode every analytic score drops the workload coordinate."""
    orig = getattr(C, '_ORIG_FEATURES', None)
    if orig is None:
        orig = C.EncodingFeatures
        C._ORIG_FEATURES = orig
    if bi:
        def factory(inst, scale):
            return ZeroLoadFeatures(orig(inst, scale))
        C.EncodingFeatures = factory
        C.online_core.EncodingFeatures = factory
    else:
        C.EncodingFeatures = orig
        C.online_core.EncodingFeatures = orig


def hv(F, ref, dims):
    np = C.np
    F = np.asarray(F, float)[:, :dims]
    r = np.asarray(ref, float)[:dims]
    return C.compat.fixed_reference_hv(F, r) / float(np.prod(r)) if len(F) else 0.0


def build(task, population, values, seed, name, ev):
    import exp3b_rivals as R
    algo = task['algo']
    fam, p = algo['family'], dict(algo.get('params', {}))
    dims = task['objectives']
    if fam == 'dqnma':
        return R.DQNMA(population, values, seed, objectives=dims, **p)
    if fam == 'ilmpea':
        return R.ILMPEA(population, values, seed, objectives=dims, offline_evaluate=ev.original, **p)
    return C.build(algo, population, values, seed, name)


def run_cell(task, release, out_root, code_sha=None, formal=False):
    C.setup(release)
    np = C.np
    out = Path(out_root) / 'runs' / f"{task['id']}.json.gz"
    if out.exists():
        return json.loads(gzip.decompress(out.read_bytes()))['summary']
    in_path = Path(task['input'])
    assert C.sha(in_path) == task['input_sha256']
    inst = C.s24.restore_instance(json.loads(in_path.read_text()))
    seed, name, mode, dims = task['seed'], task['name'], task['mode'], task['objectives']
    endpoints = list(task['endpoints'])
    random.seed(seed)
    np.random.seed(seed)
    patch_features(dims == 2)
    ev = C.Evaluator(inst, dict(kind='native'), name, seed)
    bi = BiObjective() if dims == 2 else None
    commits = []
    started = time.monotonic()
    cpu0 = time.process_time()
    with ev:
        if bi:
            bi.__enter__()
        try:
            population = C.compat.initialize_population(
                inst, 100, np.random.RandomState(C.compat.legacy_initialization_seed(seed)))
            values = C.compat.objective_array(population)
            archive = C.compat.update_archive([], population)
            for c in population:
                c._schedule = None
            alg = build(task, population, values, seed, name, ev)
            assert ev.count == 100
            commits.append((time.monotonic() - started, ev.count, archive))
            limit = endpoints[-1]
            while True:
                if mode == 'eval' and ev.count >= limit:
                    break
                if mode == 'time' and time.monotonic() - started >= limit:
                    break
                children = [alg.step() for _ in range(C.BATCH)]
                archive = C.compat.update_archive(archive, children)
                commits.append((time.monotonic() - started, ev.count, archive))
        finally:
            if bi:
                bi.__exit__()
        search_seconds = time.monotonic() - started
        cpu_seconds = time.process_time() - cpu0
        evaluations = ev.count
        eval_seconds = ev.seconds
    patch_features(False)
    assert evaluations == 100 + C.BATCH * (len(commits) - 1)
    reference = values.max(axis=0) * 1.1
    wide = values.max(axis=0) * 1.5
    rows, saved = [], []
    for at in endpoints:
        if mode == 'eval':
            c = next(c for c in commits if c[1] == at)
        else:
            before = [c for c in commits if c[0] <= at]
            assert before, f'No archive commit before the {at} s endpoint'
            c = before[-1]
        el, evals, arch = c
        F = C.compat.objective_array(arch)
        g = [C.genes(x) for x in arch]
        rows.append(dict(endpoint=at, commit_seconds=el, evaluations=evals, archive_size=len(arch),
                         hv_normalized=hv(F, reference, dims), hv_wide_normalized=hv(F, wide, dims),
                         objective_minima=F.min(axis=0).tolist()))
        saved.append(dict(endpoint=at, objectives=F.tolist(), genes=g))
    # re-evaluate the final front with the released evaluator (and the same objective mode)
    with C.Evaluator(inst, dict(kind='native'), name, seed) as chk:
        bb = BiObjective() if dims == 2 else None
        if bb:
            bb.__enter__()
        try:
            redo = np.asarray([C.chromosome(inst, x).objectives.to_array() for x in saved[-1]['genes']])
        finally:
            if bb:
                bb.__exit__()
    assert np.array_equal(redo, np.asarray(saved[-1]['objectives'])), 'Saved front does not re-evaluate'
    extra = {}
    for k in ('offline_evaluations', 'offline_seconds', 'expert_samples', 'trainings', 'train_seconds'):
        if hasattr(alg, k):
            extra[k] = getattr(alg, k)
    summary = dict(protocol=PROTOCOL, study=task['study'], id=task['id'], arm=task['arm'], method=task['method'],
                   panel=task['panel'], name=name, base_id=task['base_id'], stratum=task['stratum'],
                   regime=task['regime'], seed=seed, objectives=dims, algo=task['algo'], mode=mode,
                   endpoints=rows, evaluations=evaluations, search_seconds=search_seconds, cpu_seconds=cpu_seconds,
                   cpu_ratio=cpu_seconds / max(search_seconds, 1e-9), eval_seconds=eval_seconds,
                   score_seconds=getattr(alg, 'score_seconds', 0.0), scored_steps=getattr(alg, 'scored', 0),
                   extra=extra, input_sha256=task['input_sha256'], code_sha256=code_sha, formal=bool(formal),
                   verification='REEVALUATED', finished_utc=C.utc())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    payload = dict(summary=summary, snapshots=saved, reference=reference.tolist(), wide_reference=wide.tolist(),
                   initial_objectives=values.tolist())
    tmp.write_bytes(gzip.compress(json.dumps(payload, default=C.enc, allow_nan=False).encode(), 5, mtime=0))
    tmp.replace(out)
    return summary


def worker(task, release, out, code_sha, formal):
    os.chdir(HERE)
    return run_cell(task, release, out, code_sha, formal)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lanes', type=int, default=28)
    p.add_argument('--lock')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    os.chdir(HERE)
    tasks_path = Path(a.tasks).resolve()
    tasks = json.loads(tasks_path.read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    code_sha = C.sha(HERE / 'exp3b_rivals.py')
    formal, lock_sha = False, None
    if a.lock:
        lock = json.loads(Path(a.lock).read_text())
        for rel, h in lock['files'].items():
            assert C.sha(HERE / rel) == h, ('Locked file changed', rel)
        assert lock['tasks_sha256'] == C.sha(tasks_path)
        formal, lock_sha = True, C.sha(a.lock)
    todo = [t for t in tasks if not (out / 'runs' / f"{t['id']}.json.gz").exists()]
    if a.limit:
        todo = todo[:a.limit]
    log = (out / 'progress.jsonl').open('a')
    log.write(json.dumps(dict(event='start', utc=C.utc(), tasks=len(tasks), todo=len(todo), lanes=a.lanes,
                              formal=formal, lock_sha256=lock_sha)) + '\n')
    log.flush()
    done = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.lanes) as ex:
        futs = {ex.submit(worker, t, a.release, str(out), code_sha, formal): t for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
                done += 1
                log.write(json.dumps(dict(status='PASS', id=t['id'], hv=s['endpoints'][-1]['hv_normalized'],
                                          evaluations=s['evaluations'], seconds=round(s['search_seconds'], 2),
                                          cpu_ratio=round(s['cpu_ratio'], 4))) + '\n')
            except Exception:
                fail += 1
                log.write(json.dumps(dict(status='FAIL', id=t['id'], error=traceback.format_exc()[-3000:])) + '\n')
            log.flush()
    finish = dict(protocol=PROTOCOL, tasks_file=str(tasks_path.relative_to(HERE)), tasks=len(tasks), attempted=len(todo),
                  passed=done, failed=fail, formal=formal, lock_sha256=lock_sha, hours=(time.time() - t0) / 3600,
                  finished_utc=C.utc())
    (out / ('FINISH.json' if not a.limit else 'FINISH_partial.json')).write_text(json.dumps(finish, indent=2))
    print(json.dumps(finish))


if __name__ == '__main__':
    main()
