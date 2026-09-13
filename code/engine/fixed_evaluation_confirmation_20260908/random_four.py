"""Uniform choice from the frozen four-proposal generator; no decoder-based score."""
import time
import numpy as np
from online_core import Decomposition, stream_seed
from surrogate import joint_proposal, scalar
from lns_loop import fingerprint


class RandomFour(Decomposition):
    def __init__(self, population, seed, namespace='engineering'):
        # Reuse the original non-learned initialization byte-for-byte.
        super().__init__(population, seed, 'de_proxy', namespace)
        self.arm = 'de_random4'

    def advance(self):
        k = self.step; i = k % len(self.weights)
        t = time.perf_counter()
        rng = np.random.RandomState(stream_seed(self.seed, k, self.namespace+'/proposal'))
        select_rng = np.random.RandomState(stream_seed(self.seed, k, self.namespace+'/selection'))
        donor_index = int(rng.randint(len(self.representatives)))
        random_index = int(select_rng.randint(4))
        parent, donor = self.representatives[i], self.representatives[donor_index]
        children = [joint_proposal(parent, donor, rng, self.features) for _ in range(4)]
        self.times['proposal'] += time.perf_counter()-t
        selected = random_index
        assert all(c._objectives is None and c._schedule is None for c in children)
        row = dict(step=k, direction=i, donor_index=donor_index, brood_size=4,
            random_index=random_index, explore=k % 4 == 3, selected=selected,
            candidate_fingerprints=[fingerprint(c) for c in children],
            feature_hash=None, predictions=None, observations_before=0, model_digest_before=None)
        t = time.perf_counter(); child = children[selected]; y = child.objectives.to_array()
        self.times['evaluation'] += time.perf_counter()-t
        child._schedule = None
        assert all(c._objectives is None and c._schedule is None
            for j, c in enumerate(children) if j != selected)
        t = time.perf_counter()
        neighbors = self.neighbors[i]
        old = scalar(self.values[neighbors], self.weights[neighbors], self.scale)
        new = scalar(np.broadcast_to(y, (len(neighbors), 3)), self.weights[neighbors], self.scale)
        replace = neighbors[np.flatnonzero(new < old)[:2]]
        for j in replace:
            self.representatives[int(j)] = child; self.values[int(j)] = y
        self.times['representative_update'] += time.perf_counter()-t
        self.step += 1
        row.update(objectives=y.tolist(), replacements=replace.tolist(), observations_after=0, fit_count=0)
        return child, row
