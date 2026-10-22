"""Driver: run a frozen task list with a process pool.

  python exp2_run.py --release REL --tasks tasks/X.json --out out/X --lanes 28 [--lock locks/LOCK_X.json] [--limit N]

Without --lock the outputs are marked formal=False (engineering). With --lock every locked
file hash and the task-list hash are verified before the first run.
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2_core as C  # noqa: E402


def worker(task, release, out, code_sha, formal, dataset):
    os.chdir(HERE)
    return C.run_cell(task, release, out, code_sha, formal, dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lanes', type=int, default=28)
    p.add_argument('--lock')
    p.add_argument('--limit', type=int)
    p.add_argument('--dataset')
    a = p.parse_args()
    os.chdir(HERE)
    tasks_path = Path(a.tasks).resolve()
    tasks = json.loads(tasks_path.read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    code_sha = C.sha(HERE / 'exp2_core.py')
    formal = False
    lock_sha = None
    if a.lock:
        lock = json.loads(Path(a.lock).read_text())
        for rel, h in lock['files'].items():
            assert C.sha(HERE / rel) == h, ('Locked file changed', rel)
        assert lock['tasks_sha256'] == C.sha(tasks_path), 'Task list changed'
        assert lock['tasks_file'] == str(tasks_path.relative_to(HERE))
        formal = True
        lock_sha = C.sha(a.lock)
    todo = [t for t in tasks if not (out / 'runs' / f"{t['id']}.json.gz").exists()]
    if a.limit:
        todo = todo[:a.limit]
    log = (out / 'progress.jsonl').open('a')
    log.write(json.dumps(dict(event='start', utc=C.utc(), tasks=len(tasks), todo=len(todo), lanes=a.lanes,
                              formal=formal, lock_sha256=lock_sha, code_sha256=code_sha)) + '\n')
    log.flush()
    done = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.lanes) as ex:
        futs = {ex.submit(worker, t, a.release, str(out), code_sha, formal, a.dataset): t for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
                done += 1
                e = s['endpoints'][-1]
                log.write(json.dumps(dict(status='PASS', id=t['id'], hv=e['hv_normalized'], hv_test=e.get('hv_test'),
                                          evaluations=s['evaluations'], seconds=round(s['search_seconds'], 2),
                                          cpu_ratio=round(s['cpu_ratio'], 4))) + '\n')
            except Exception:
                fail += 1
                log.write(json.dumps(dict(status='FAIL', id=t['id'], error=traceback.format_exc()[-3000:])) + '\n')
            log.flush()
    finish = dict(protocol=C.PROTOCOL, tasks_file=str(tasks_path.relative_to(HERE)), tasks=len(tasks),
                  attempted=len(todo), passed=done, failed=fail, formal=formal, lock_sha256=lock_sha,
                  code_sha256=code_sha, hours=(time.time() - t0) / 3600, finished_utc=C.utc())
    (out / ('FINISH.json' if not a.limit else 'FINISH_partial.json')).write_text(json.dumps(finish, indent=2))
    print(json.dumps(finish))


if __name__ == '__main__':
    main()
