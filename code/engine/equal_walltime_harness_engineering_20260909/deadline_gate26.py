"""Monotonic, immutable deadline ledger. Pure stdlib; never runs a solver."""
from __future__ import annotations
import copy
import math
import time

def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)

class DeadlineGate:
    """freeze_fn must detach its output from the mutable input.
    Default deepcopy is safe for arbitrary payloads. Production returns nested
    immutable tuples/bytes. Snapshot reads are permitted only after finish().
    """
    def __init__(self, budget, checkpoints, clock=time.monotonic,
                 freeze_fn=copy.deepcopy):
        self._clock = clock
        self._last_clock = None
        self._closed = False
        self._failed = None
        self.start = self._now()
        if not finite_number(budget) or budget <= 0:
            raise ValueError('budget must be finite and positive')
        points = tuple(checkpoints)
        if not points or any(not finite_number(t) or not 0 < t <= budget for t in points):
            raise ValueError('invalid checkpoints')
        if any(a >= b for a, b in zip(points, points[1:])):
            raise ValueError('checkpoints must be strictly increasing')
        self.budget = float(budget)
        self.checkpoints = points
        self._freeze = freeze_fn
        self._last = None
        self._snapshots = {}
        self._spent = 0
        self.commit_log = []
        self.freeze_seconds = 0.0

    def _now(self):
        if self._failed is not None:
            raise RuntimeError(self._failed)
        value = self._clock()
        if not finite_number(value) or (self._last_clock is not None and value < self._last_clock):
            self._failed = 'non-finite or backwards clock'
            self._closed = True
            raise RuntimeError(self._failed)
        self._last_clock = value
        return value

    def elapsed(self):
        return self._now() - self.start

    def remaining(self):
        return max(0.0, self.budget - self.elapsed())

    def _pack(self, point):
        if self._last is None:
            return dict(deadline=point, commit_seconds=None, commit_label=None,
                        evaluations=0, status='NO_COMMITTED_SOLUTION', payload=None)
        return dict(deadline=point, status='COMMITTED', **self._last)

    def _seal(self, stamp, inclusive=False):
        for point in self.checkpoints:
            crossed = point <= stamp if inclusive else point < stamp
            if crossed and point not in self._snapshots:
                self._snapshots[point] = self._pack(point)

    def commit(self, payload, label, evaluations):
        if self._closed:
            raise RuntimeError('deadline gate is closed')
        if not isinstance(label, str) or not label:
            raise ValueError('label must be a nonempty string')
        if type(evaluations) is not int or evaluations < self._spent:
            raise ValueError('evaluation count must be nonnegative and nondecreasing')
        before = self.elapsed()
        try:
            prepared = self._freeze(payload)
            stamp = self.elapsed()
        except Exception:
            self._closed = True
            raise
        self.freeze_seconds += stamp - before
        # Crucial: crossing endpoints receive the OLD archive, not this future one.
        self._seal(stamp)
        accepted = stamp <= self.budget
        self._spent = evaluations
        self.commit_log.append(dict(seconds=stamp, label=label,
                                    evaluations=evaluations, accepted=accepted))
        if accepted:
            self._last = dict(commit_seconds=stamp, commit_label=label,
                              evaluations=evaluations, payload=prepared)
        return accepted

    def finish(self):
        if self._failed is not None:
            raise RuntimeError(self._failed)
        if self._closed:
            return
        self.elapsed()  # Validate monotonicity, including the last interval.
        # An explicit engineering early stop carries the last complete state.
        # No algorithm work is performed here.
        self._seal(self.budget, inclusive=True)
        self._closed = True

    def snapshot(self, at):
        if self._failed is not None:
            raise RuntimeError(self._failed)
        if not self._closed:
            raise RuntimeError('snapshot requires finish')
        if at not in self.checkpoints:
            raise ValueError('not a declared checkpoint')
        return copy.deepcopy(self._snapshots[at])
