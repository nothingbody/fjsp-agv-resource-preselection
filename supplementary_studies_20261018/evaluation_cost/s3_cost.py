"""S3: evaluation-cost scaling (see PROTOCOL.md).

Reuses the locked S2 driver (s2_moea.run_cell) in time mode. The only change is
a fixed sleep after every exact evaluation counted by compat.CountCalls.

  python s3_cost.py --release REL --inputs DIR --output OUT --delays 10,25,50 --arms always,nsga2,nsga2_pre,moead
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

S2_DIR = os.environ.get('S2_DIR', str(Path(__file__).resolve().parent.parent / 's2'))
sys.path.insert(0, S2_DIR)
import s2_moea as s2  # noqa: E402

PROTOCOL = 's3_eval_cost_scaling_20261018_v1'


def install_delay(release: str, delay_ms: int):
    s2.setup(Path(release))
    compat = s2.compat
    if getattr(compat.CountCalls, 'delay_ms', None) == delay_ms:
        return
    base = getattr(compat.CountCalls, 'base', compat.CountCalls)
    seconds = delay_ms / 1000.0

    class DelayedCount(base):
        def __enter__(self):
            super().__enter__()
            counted = compat.decoding.evaluate

            def slow(*a, **kw):
                result = counted(*a, **kw)
                time.sleep(seconds)
                return result
            compat.decoding.evaluate = slow
            return self

    DelayedCount.delay_ms = delay_ms
    DelayedCount.base = base
    compat.CountCalls = DelayedCount


def run_cell(release, input_path, out_dir, name, seed, arm, delay_ms):
    install_delay(release, delay_ms)
    s2.PROTOCOL = PROTOCOL
    if os.environ.get('S3_SMOKE'):  # engineering smoke test only, never formal
        s2.TIME_ENDPOINTS = (10.0, 20.0)
    return s2.run_cell(release, input_path, str(Path(out_dir) / f'd{delay_ms:03}'), name, seed, arm, 'time')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--delays', default='10,25,50')
    p.add_argument('--arms', default='always,nsga2,nsga2_pre,moead')
    p.add_argument('--lanes', type=int, default=19)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    arms = a.arms.split(',')
    delays = [int(x) for x in a.delays.split(',')]
    assert set(arms) <= set(s2.ALL_ARMS)
    a.output.mkdir(parents=True, exist_ok=True)
    todo = [t + (d,) for d in delays for t in s2.tasks(a.inputs, arms, 'time')]
    random.Random(20261019).shuffle(todo)  # mix delays and arms across lanes
    todo = todo[:a.limit]
    log = (a.output / 'progress.jsonl').open('a')
    done = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.lanes) as ex:
        futs = {ex.submit(run_cell, str(a.release.resolve()), t[0], str(a.output), t[1], t[2], t[3], t[4]): t
                for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            try:
                s = f.result()
                done += 1
                log.write(json.dumps(dict(status='PASS', name=t[1], seed=t[2], arm=t[3], delay_ms=t[4],
                                          hv=s['endpoints'][-1]['hv_normalized'],
                                          evaluations=s['evaluations'], seconds=s['search_seconds'])) + '\n')
            except Exception:
                fail += 1
                log.write(json.dumps(dict(status='FAIL', name=t[1], seed=t[2], arm=t[3], delay_ms=t[4],
                                          error=traceback.format_exc()[-2000:])) + '\n')
            log.flush()
    (a.output / 'FINISH.json').write_text(json.dumps(dict(
        protocol=PROTOCOL, delays_ms=delays, arms=arms, tasks=len(todo), passed=done, failed=fail,
        hours=(time.time() - t0) / 3600,
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2))
    print(json.dumps(dict(passed=done, failed=fail)))


if __name__ == '__main__':
    main()
