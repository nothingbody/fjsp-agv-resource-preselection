"""Counter-based random-stream ledger for SA-AOS 2.0.

Every stream is derived from an immutable semantic key rather than from the
number of draws consumed by a preceding component.  Reconstructing the same
key therefore reconstructs the same NumPy ``RandomState`` without coupling
parent selection, operator internals, environmental niching, or branching.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


PROTOCOL = "saos2_split_rng_v2_20260827"
ALLOWED_STREAMS = frozenset(
    {
        "initialization",
        "parent_selection",
        "operator_internal",
        "environmental_selection",
        "controller",
        "learner_sampling",
        "continuation",
        "branching",
    }
)


def _canonical_part(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_canonical_part(item) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted((str(key), val) for key, val in value.items())
        return "{" + ",".join(
            _canonical_part(key) + ":" + _canonical_part(val)
            for key, val in items
        ) + "}"
    return _canonical_part(str(value))


def derive_seed(base_seed: int, stream: str, *coordinates: Any) -> int:
    """Derive one stable uint32 seed from a semantic stream key."""

    if stream not in ALLOWED_STREAMS:
        raise ValueError(f"Unknown SA-AOS 2.0 RNG stream: {stream}")
    payload = "|".join(
        [PROTOCOL, str(int(base_seed)), stream]
        + [_canonical_part(item) for item in coordinates]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def key_hash(base_seed: int, stream: str, coordinates: Iterable[Any]) -> str:
    payload = "|".join(
        [PROTOCOL, str(int(base_seed)), stream]
        + [_canonical_part(item) for item in coordinates]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RNGRecord:
    protocol: str
    base_seed: int
    stream: str
    coordinates: Tuple[Any, ...]
    derived_seed: int
    key_hash: str
    derivation: str = "sha256_semantic_key"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "base_seed": self.base_seed,
            "stream": self.stream,
            "coordinates": list(self.coordinates),
            "derived_seed": self.derived_seed,
            "key_hash": self.key_hash,
            "derivation": self.derivation,
        }


class RNGLedger:
    """Create deterministic streams and retain one auditable record per key."""

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)
        self._records: Dict[str, RNGRecord] = {}

    def seed_for(self, stream: str, *coordinates: Any) -> int:
        seed = derive_seed(self.base_seed, stream, *coordinates)
        self.register_seed(
            stream,
            seed,
            *coordinates,
            derivation="sha256_semantic_key",
        )
        return seed

    def register_seed(
        self,
        stream: str,
        seed: int,
        *coordinates: Any,
        derivation: str,
    ) -> int:
        """Register an auditable seed produced by a declared derivation.

        The canonical split streams use :func:`derive_seed`.  Phase 1 also
        needs the historical first ``SeedSequence`` child for initialization
        so that the formal K=1 and K=10 tracks start from the same archived E8
        population.  Registering that seed here keeps the exception explicit
        rather than hiding it outside the ledger.
        """

        if stream not in ALLOWED_STREAMS:
            raise ValueError(f"Unknown SA-AOS 2.0 RNG stream: {stream}")
        seed = int(seed)
        if not 0 <= seed < 2 ** 32:
            raise ValueError("RNG seed must fit uint32")
        digest = key_hash(self.base_seed, stream, coordinates)
        record = RNGRecord(
            protocol=PROTOCOL,
            base_seed=self.base_seed,
            stream=stream,
            coordinates=tuple(coordinates),
            derived_seed=seed,
            key_hash=digest,
            derivation=str(derivation),
        )
        previous = self._records.get(digest)
        if previous is not None and previous != record:
            raise RuntimeError("RNG ledger hash collision")
        self._records[digest] = record
        return seed

    def stream(self, stream: str, *coordinates: Any) -> np.random.RandomState:
        return np.random.RandomState(self.seed_for(stream, *coordinates))

    def records(self) -> List[Dict[str, Any]]:
        ordered = sorted(
            self._records.values(),
            key=lambda item: (item.stream, _canonical_part(item.coordinates)),
        )
        return [item.as_dict() for item in ordered]

    def record_for(self, stream: str, *coordinates: Any) -> Dict[str, Any]:
        """Return the ledger row for one already registered semantic key."""

        digest = key_hash(self.base_seed, stream, coordinates)
        if digest not in self._records:
            raise KeyError(f"RNG key has not been registered: {stream} {coordinates}")
        return self._records[digest].as_dict()

    def assert_unique(self) -> None:
        records = self.records()
        hashes = [item["key_hash"] for item in records]
        if len(hashes) != len(set(hashes)):
            raise AssertionError("Duplicate RNG ledger keys")


def legacy_initialization_seed(base_seed: int) -> int:
    """Return the first E8 SeedSequence child seed without advancing a stream."""

    sequence = np.random.SeedSequence(int(base_seed))
    child = sequence.spawn(3)[0]
    return int(child.generate_state(1, dtype=np.uint32)[0])
