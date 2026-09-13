"""Fine-grained generation executor for SA-AOS 2.0 Phase 1.

The module deliberately implements UCB and instrumentation first.  A residual
learner is not admitted to the closed loop until Q0--Q2 have passed.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.algorithm.aos2.credits import (
    assert_decomposition_contract,
    candidate_batch_credits,
    decompose_generation_reward,
    nondominated_rows,
    objective_array,
    post_selection_difference_credits,
)
from src.algorithm.aos2.rng_ledger import RNGLedger
from src.algorithm.nsga3.crossover import CROSSOVER_OPERATORS
from src.algorithm.nsga3.encoding import Chromosome
from src.algorithm.nsga3.mutation import MUTATION_OPERATORS
from src.algorithm.nsga3.selection import (
    compute_hypervolume,
    non_dominated_sort,
    nsga3_select,
)


ALL_OPERATORS = {}
ALL_OPERATORS.update(CROSSOVER_OPERATORS)
ALL_OPERATORS.update(MUTATION_OPERATORS)
N_OPS = len(ALL_OPERATORS)
EPS = 1e-12


@dataclass(frozen=True)
class GranularitySpec:
    k: int
    population_size: int = 100
    tournament_size: int = 5

    def __post_init__(self) -> None:
        if self.population_size <= 0 or self.population_size % 2:
            raise ValueError("Population size must be a positive even integer")
        if self.k <= 0:
            raise ValueError("K must be positive")
        if self.parent_pair_count % self.k:
            raise ValueError("K must divide the number of parent pairs")

    @property
    def parent_pair_count(self) -> int:
        return self.population_size // 2

    @property
    def pairs_per_batch(self) -> int:
        return self.parent_pair_count // self.k

    @property
    def offspring_per_batch(self) -> int:
        return 2 * self.pairs_per_batch

    @property
    def exposure_window(self) -> int:
        return 50 * self.k

    @property
    def warm_count_per_action(self) -> int:
        return 2 * self.k


@dataclass(frozen=True)
class FrozenParentPair:
    global_pair: int
    parent1_index: int
    parent2_index: int
    parent1: Any
    parent2: Any


@dataclass(frozen=True)
class UCBDecision:
    action: int
    record_index: int
    mandatory_reason: str
    window_counts: Tuple[int, ...]
    reward_counts: Tuple[int, ...]
    reward_means: Tuple[float, ...]
    exploration_bonus: Tuple[float, ...]
    scores: Tuple[float, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "record_index": self.record_index,
            "mandatory_reason": self.mandatory_reason,
            "window_counts": list(self.window_counts),
            "reward_counts": list(self.reward_counts),
            "reward_means": list(self.reward_means),
            "exploration_bonus": list(self.exploration_bonus),
            "scores": list(self.scores),
        }


@dataclass
class FineGrainedGenerationResult:
    population: List[Any]
    archive: List[Any]
    current_hv: float
    current_best_cmax: float
    global_survival: float
    global_hv_delta: float
    global_cmax_delta: float
    global_reward: float
    decisions: List[Dict[str, Any]]
    batch_credits: List[Dict[str, float]]
    lineage: List[Dict[str, Any]]
    parent_pairs: List[Dict[str, Any]]
    invariants: Dict[str, Any]
    offspring_batches: List[List[Any]]


def chromosome_fingerprint(chromosome: Any) -> str:
    digest = hashlib.sha256()
    for values in (
        chromosome.os,
        chromosome.ma,
        chromosome.agv_assign,
        chromosome.agv_speed,
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tournament_select_index(
    population: Sequence[Any], tournament_size: int, rng: np.random.RandomState
) -> int:
    indices = rng.choice(len(population), tournament_size, replace=False)
    candidates = [population[int(index)] for index in indices]
    objectives = objective_array(candidates)
    fronts = non_dominated_sort(objectives)
    return int(indices[fronts[0][0]])


def freeze_parent_pairs(
    population: Sequence[Any],
    pair_count: int,
    tournament_size: int,
    rng: np.random.RandomState,
) -> List[FrozenParentPair]:
    pairs = []
    for pair_index in range(pair_count):
        p1 = _tournament_select_index(population, tournament_size, rng)
        p2 = _tournament_select_index(population, tournament_size, rng)
        pairs.append(
            FrozenParentPair(
                global_pair=pair_index,
                parent1_index=p1,
                parent2_index=p2,
                parent1=population[p1],
                parent2=population[p2],
            )
        )
    return pairs


def _apply_operator_pair(
    pair: FrozenParentPair,
    operator_id: int,
    rng: np.random.RandomState,
) -> Tuple[Any, Any]:
    name, operator = ALL_OPERATORS[operator_id]
    if operator_id < 5:
        if operator_id == 4:
            return split_rng_two_point_crossover(pair.parent1, pair.parent2, rng)
        return operator(pair.parent1, pair.parent2, rng)
    if name in ("MachineReassign", "AGVReassign", "SpeedAdjust"):
        return (
            operator(pair.parent1, mutation_rate=0.1, rng=rng),
            operator(pair.parent2, mutation_rate=0.1, rng=rng),
        )
    return operator(pair.parent1, rng=rng), operator(pair.parent2, rng=rng)


def _split_rng_two_point_build(os1, os2, pt1, pt2, instance, rng):
    """E8 C5 repair logic with its shuffle supplied by the semantic stream."""

    child = os1.copy()
    child[pt1:pt2] = os2[pt1:pt2]
    target_count = {
        job: instance.num_operations[job] for job in range(instance.num_jobs)
    }
    current_count: Dict[int, int] = {}
    for value in child:
        current_count[int(value)] = current_count.get(int(value), 0) + 1
    excess: List[int] = []
    deficit: List[int] = []
    for job in range(instance.num_jobs):
        difference = current_count.get(job, 0) - target_count[job]
        if difference > 0:
            excess.extend([job] * difference)
        elif difference < 0:
            deficit.extend([job] * (-difference))
    if excess and deficit:
        rng.shuffle(deficit)
        deficit_index = 0
        for index in range(len(child)):
            if int(child[index]) in excess:
                current = int(child[index])
                count_to_here = sum(
                    1 for value in child[: index + 1] if int(value) == current
                )
                if count_to_here > target_count[current]:
                    child[index] = deficit[deficit_index]
                    deficit_index += 1
                    if deficit_index >= len(deficit):
                        break
    return child


def split_rng_two_point_crossover(
    parent1: Any, parent2: Any, rng: np.random.RandomState
) -> Tuple[Any, Any]:
    """C5 with unchanged chromosome semantics and no process-global RNG draw."""

    if len(parent1.os) != len(parent2.os):
        return parent1.copy(), parent2.copy()
    point1, point2 = sorted(rng.choice(len(parent1.os), 2, replace=False))
    child1_os = _split_rng_two_point_build(
        parent1.os, parent2.os, point1, point2, parent1.instance, rng
    )
    child2_os = _split_rng_two_point_build(
        parent2.os, parent1.os, point1, point2, parent1.instance, rng
    )
    child1 = Chromosome(
        parent1.instance,
        child1_os,
        parent1.ma.copy(),
        parent1.agv_assign.copy(),
        parent1.agv_speed.copy(),
    )
    child2 = Chromosome(
        parent1.instance,
        child2_os,
        parent2.ma.copy(),
        parent2.agv_assign.copy(),
        parent2.agv_speed.copy(),
    )
    return child1, child2


def apply_operator_to_frozen_pairs(
    pairs: Sequence[FrozenParentPair],
    operator_id: int,
    rng: np.random.RandomState,
    generation: int,
    slot: int,
    rng_key_hash: str,
    rng_seed: int,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    offspring: List[Any] = []
    lineage: List[Dict[str, Any]] = []
    for local_pair, pair in enumerate(pairs):
        children = _apply_operator_pair(pair, operator_id, rng)
        parent_objectives = [
            np.asarray(pair.parent1.objectives.to_array(), dtype=float).tolist(),
            np.asarray(pair.parent2.objectives.to_array(), dtype=float).tolist(),
        ]
        for child_index, child in enumerate(children):
            if operator_id < 5:
                credit_parent_objectives = parent_objectives
                source_parent_role = "both_crossover_parents"
            else:
                credit_parent_objectives = [parent_objectives[child_index]]
                source_parent_role = f"parent{child_index + 1}_mutation_source"
            uid = f"g{generation}:s{slot}:p{pair.global_pair}:c{child_index}"
            child._saos2_uid = uid
            child._revision_uid = uid
            child._saos2_lineage = {
                "generation": generation,
                "slot": slot,
                "batch_local_pair": local_pair,
                "global_pair": pair.global_pair,
                "parent1_index": pair.parent1_index,
                "parent2_index": pair.parent2_index,
                "parent1_fingerprint": chromosome_fingerprint(pair.parent1),
                "parent2_fingerprint": chromosome_fingerprint(pair.parent2),
                "parent_objectives": parent_objectives,
                "credit_parent_objectives": credit_parent_objectives,
                "source_parent_role": source_parent_role,
                "operator_family": "crossover" if operator_id < 5 else "mutation",
                "action": operator_id,
                "child_index": child_index,
                "child_id": uid,
                "operator_rng_key_hash": rng_key_hash,
                "operator_rng_seed": int(rng_seed),
            }
            objectives = np.asarray(child.objectives.to_array(), dtype=float)
            row = dict(child._saos2_lineage)
            row["child_fingerprint"] = chromosome_fingerprint(child)
            row["objectives"] = objectives.tolist()
            offspring.append(child)
            lineage.append(row)
    return offspring, lineage


class DelayedExposureUCB:
    """Exposure-normalized UCB with generation-delayed batch rewards."""

    def __init__(self, k: int, n_ops: int = N_OPS, c: float = 1.0):
        self.k = int(k)
        self.n_ops = int(n_ops)
        self.c = float(c)
        self.window = 50 * self.k
        self.min_count = 2 * self.k
        self.records: List[Dict[str, Any]] = []
        self.total_counts = np.zeros(self.n_ops, dtype=int)

    def _score_snapshot(self) -> Dict[str, np.ndarray]:
        recent = self.records[-self.window :]
        selected_counts = np.zeros(self.n_ops, dtype=int)
        reward_counts = np.zeros(self.n_ops, dtype=int)
        means = np.zeros(self.n_ops, dtype=float)
        for item in recent:
            action = int(item["action"])
            selected_counts[action] += 1
            if item["reward"] is not None:
                reward_counts[action] += 1
                means[action] += float(item["reward"])
        nonzero = reward_counts > 0
        means[nonzero] /= reward_counts[nonzero]
        total_completed = max(int(reward_counts.sum()), 1)
        bonus = np.full(self.n_ops, float("inf"), dtype=float)
        bonus[nonzero] = self.c * np.sqrt(
            np.log(total_completed) / reward_counts[nonzero]
        )
        scores = means + bonus
        return {
            "selected_counts": selected_counts,
            "reward_counts": reward_counts,
            "means": means,
            "bonus": bonus,
            "scores": scores,
        }

    def build_base_state(
        self,
        generation: int,
        max_generations: int,
        stagnation: int,
        diversity: float,
        hv_trend: float,
    ) -> np.ndarray:
        snapshot = self._score_snapshot()
        frequency = snapshot["selected_counts"].astype(float) / max(
            int(snapshot["selected_counts"].sum()), 1
        )
        progress = np.asarray(
            [
                generation / max(max_generations, 1),
                stagnation / 20.0,
                diversity,
                hv_trend,
            ],
            dtype=float,
        )
        return np.concatenate([snapshot["means"], frequency, progress])

    def register_action(
        self,
        action: int,
        mandatory_reason: str = "externally_forced",
    ) -> UCBDecision:
        """Register a declared action using the current pre-action UCB snapshot.

        Counterfactual replay uses this method to hold the nine non-target
        within-generation actions fixed while replacing one micro-batch.  It
        records exposure immediately but does not reveal delayed credit.
        """

        action = int(action)
        if action < 0 or action >= self.n_ops:
            raise ValueError(f"Action is outside [0, {self.n_ops}): {action}")
        snapshot = self._score_snapshot()
        record_index = len(self.records)
        self.records.append(
            {"action": action, "reward": None, "completed": False}
        )
        self.total_counts[action] += 1
        return UCBDecision(
            action=action,
            record_index=record_index,
            mandatory_reason=str(mandatory_reason),
            window_counts=tuple(int(x) for x in snapshot["selected_counts"]),
            reward_counts=tuple(int(x) for x in snapshot["reward_counts"]),
            reward_means=tuple(float(x) for x in snapshot["means"]),
            exploration_bonus=tuple(float(x) for x in snapshot["bonus"]),
            scores=tuple(float(x) for x in snapshot["scores"]),
        )

    def select_and_register(self) -> UCBDecision:
        snapshot = self._score_snapshot()
        reason = "score"
        action = None
        for operator in range(self.n_ops):
            if self.total_counts[operator] < self.min_count:
                action = operator
                reason = "cold_start"
                break
        if action is None:
            absent = np.flatnonzero(snapshot["selected_counts"] == 0)
            if len(absent):
                action = int(absent[0])
                reason = "window_absence"
            else:
                action = int(np.argmax(snapshot["scores"]))
        return self.register_action(int(action), mandatory_reason=reason)

    def select_with_context(
        self, base_state: np.ndarray, intra_state: np.ndarray
    ) -> UCBDecision:
        """Select using the canonical UCB rule.

        The context-aware hook is intentionally a no-op for the established UCB
        controller.  It lets separately audited pilot controllers consume the
        exact same deployment-time state without changing the frozen Phase 1
        UCB trajectory.
        """

        del base_state, intra_state
        return self.select_and_register()

    def decision_metadata(self, record_index: int) -> Dict[str, Any]:
        """Return optional policy diagnostics for an already registered action."""

        del record_index
        return {}

    def complete_generation_with_feedback(
        self,
        record_indices: Sequence[int],
        rewards: Sequence[float],
        feedback_rows: Sequence[Dict[str, Any]],
    ) -> None:
        """Commit delayed UCB rewards while ignoring optional pilot feedback."""

        del feedback_rows
        self.complete_generation(record_indices, rewards)

    def complete_generation(
        self, record_indices: Sequence[int], rewards: Sequence[float]
    ) -> None:
        if len(record_indices) != len(rewards):
            raise ValueError("Pending UCB records and rewards must align")
        for index, reward in zip(record_indices, rewards):
            item = self.records[int(index)]
            if item["completed"] or item["reward"] is not None:
                raise RuntimeError("A delayed UCB record was completed twice")
            item["reward"] = float(reward)
            item["completed"] = True

    @property
    def completed_observations(self) -> int:
        return sum(bool(item["completed"]) for item in self.records)

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON/pickle-safe controller state at a decision boundary."""

        return {
            "k": self.k,
            "n_ops": self.n_ops,
            "c": self.c,
            "window": self.window,
            "min_count": self.min_count,
            "records": [dict(item) for item in self.records],
            "total_counts": self.total_counts.astype(int).tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "DelayedExposureUCB":
        selector = cls(
            k=int(state["k"]),
            n_ops=int(state["n_ops"]),
            c=float(state["c"]),
        )
        if selector.window != int(state["window"]):
            raise ValueError("Serialized UCB exposure window violates K contract")
        if selector.min_count != int(state["min_count"]):
            raise ValueError("Serialized UCB cold-start count violates K contract")
        selector.records = [dict(item) for item in state["records"]]
        selector.total_counts = np.asarray(state["total_counts"], dtype=int)
        if selector.total_counts.shape != (selector.n_ops,):
            raise ValueError("Serialized UCB total-count vector has wrong shape")
        if any(
            item.get("reward") is None and bool(item.get("completed"))
            for item in selector.records
        ):
            raise ValueError("Completed UCB record cannot have missing reward")
        return selector


def _population_diversity(objectives: np.ndarray) -> float:
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) == 0:
        return 0.0
    means = np.mean(objectives, axis=0)
    cv = np.std(objectives, axis=0) / np.maximum(np.abs(means), 1e-10)
    return float(np.mean(cv))


def _reference_associations(rows: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    if len(rows) == 0:
        return np.empty(0, dtype=int)
    ideal = rows.min(axis=0)
    span = np.where(rows.max(axis=0) - ideal > EPS, rows.max(axis=0) - ideal, 1.0)
    normalized = (rows - ideal) / span
    norms = np.linalg.norm(ref_points, axis=1)
    unit = ref_points / norms[:, None]
    output = []
    for row in normalized:
        projections = (unit @ row)[:, None] * unit
        output.append(int(np.argmin(np.linalg.norm(row - projections, axis=1))))
    return np.asarray(output, dtype=int)


def build_intra_state(
    slot: int,
    spec: GranularitySpec,
    previous_actions: Sequence[int],
    generated_batches: Sequence[Sequence[Any]],
    population: Sequence[Any],
    archive: Sequence[Any],
    ref_points: np.ndarray,
    reward_means: Sequence[float],
) -> Tuple[np.ndarray, List[str]]:
    names = [
        "slot_fraction",
        "remaining_offspring_fraction",
        "generated_offspring_fraction",
    ]
    values = [
        slot / max(spec.k, 1),
        (spec.k - slot) / max(spec.k, 1),
        sum(len(batch) for batch in generated_batches) / spec.population_size,
    ]
    action_counts = np.bincount(
        np.asarray(previous_actions, dtype=int), minlength=N_OPS
    ).astype(float)
    action_counts /= max(len(previous_actions), 1)
    names.extend([f"intra_action_frequency_{idx}" for idx in range(N_OPS)])
    values.extend(action_counts.tolist())

    parent_rows = objective_array(population)
    generated = [child for batch in generated_batches for child in batch]
    if generated:
        rows = objective_array(generated)
        center = np.mean(parent_rows, axis=0)
        scale = np.maximum(np.std(parent_rows, axis=0), 1.0)
        mean_rel = (np.mean(rows, axis=0) - center) / scale
        std_rel = np.std(rows, axis=0) / scale
        best_rel = (center - np.min(rows, axis=0)) / np.maximum(np.abs(center), 1.0)
        combined = np.vstack([parent_rows, rows])
        nd = {tuple(row.tolist()) for row in nondominated_rows(combined)}
        nd_rate = np.mean([tuple(row.tolist()) in nd for row in rows])
        archive_keys = {
            tuple(np.asarray(item.objectives.to_array(), dtype=float).tolist())
            for item in archive
        }
        provisional = np.mean([tuple(row.tolist()) not in archive_keys for row in rows])
        assoc = _reference_associations(combined, ref_points)
        parent_assoc = set(assoc[: len(parent_rows)].tolist())
        generated_assoc = set(assoc[len(parent_rows) :].tolist())
        occupancy = len(generated_assoc) / len(ref_points)
        new_occupancy = len(generated_assoc - parent_assoc) / len(ref_points)
        parent_relative = []
        for child in generated:
            parent_obj = np.asarray(
                child._saos2_lineage.get(
                    "credit_parent_objectives",
                    child._saos2_lineage["parent_objectives"],
                ),
                dtype=float,
            ).mean(axis=0)
            child_obj = np.asarray(child.objectives.to_array(), dtype=float)
            parent_relative.append(
                (parent_obj - child_obj) / np.maximum(np.abs(parent_obj), 1.0)
            )
        parent_relative = np.mean(parent_relative, axis=0)
    else:
        mean_rel = np.zeros(3)
        std_rel = np.zeros(3)
        best_rel = np.zeros(3)
        nd_rate = provisional = occupancy = new_occupancy = 0.0
        parent_relative = np.zeros(3)

    for prefix, row in (
        ("generated_mean_z", mean_rel),
        ("generated_std_ratio", std_rel),
        ("generated_best_relative", best_rel),
    ):
        names.extend([f"{prefix}_{index}" for index in range(3)])
        values.extend(np.asarray(row, dtype=float).tolist())
    names.extend(
        [
            "generated_nondominated_rate",
            "provisional_archive_novelty_rate",
            "reference_occupancy_fraction",
            "new_reference_occupancy_fraction",
        ]
    )
    values.extend([nd_rate, provisional, occupancy, new_occupancy])
    names.extend([f"parent_relative_improvement_{index}" for index in range(3)])
    values.extend(np.asarray(parent_relative, dtype=float).tolist())
    means = np.asarray(reward_means, dtype=float)
    names.extend(["recent_crossover_reward", "recent_mutation_reward"])
    values.extend([float(np.mean(means[:5])), float(np.mean(means[5:]))])
    vector = np.asarray(values, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("Intra-generation state contains non-finite values")
    return vector, names


def _global_reward(
    survival: float, hv_delta: float, cmax_delta: float, scheme: str
) -> float:
    if scheme != "composite":
        raise ValueError("Phase 1 fine-grained executor freezes the composite reward")
    return float(0.5 * survival + 0.3 * hv_delta + 0.2 * cmax_delta)


def execute_finegrained_generation(
    population: Sequence[Any],
    archive: Sequence[Any],
    selector: DelayedExposureUCB,
    spec: GranularitySpec,
    ledger: RNGLedger,
    run_key: str,
    generation: int,
    max_generations: int,
    ref_points: np.ndarray,
    online_reference: np.ndarray,
    previous_hv: float,
    previous_best_cmax: float,
    stagnation: int,
    hv_window: Sequence[float],
    reward_scheme: str = "composite",
    archive_size: int = 100,
    update_archive_fn=None,
) -> FineGrainedGenerationResult:
    if spec.k != selector.k:
        raise ValueError("Granularity and selector K differ")
    if len(population) != spec.population_size:
        raise ValueError("Population size does not match the frozen specification")
    if update_archive_fn is None:
        raise ValueError("The unchanged E8 archive update function must be supplied")

    parent_seed = ledger.seed_for("parent_selection", run_key, generation)
    parent_record = ledger.record_for("parent_selection", run_key, generation)
    parent_rng = np.random.RandomState(parent_seed)
    frozen_pairs = freeze_parent_pairs(
        population,
        spec.parent_pair_count,
        spec.tournament_size,
        parent_rng,
    )
    parent_pair_rows = [
        {
            "generation": generation,
            "global_pair": pair.global_pair,
            "parent1_index": pair.parent1_index,
            "parent2_index": pair.parent2_index,
            "parent1_fingerprint": chromosome_fingerprint(pair.parent1),
            "parent2_fingerprint": chromosome_fingerprint(pair.parent2),
            "parent_rng_seed": parent_seed,
            "parent_rng_key_hash": parent_record["key_hash"],
        }
        for pair in frozen_pairs
    ]

    population_rows = objective_array(population)
    diversity = _population_diversity(population_rows)
    hv_trend = float(np.mean(hv_window)) if hv_window else 0.0
    offspring_batches: List[List[Any]] = []
    decisions: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    pending_indices: List[int] = []

    for slot in range(spec.k):
        base_state = selector.build_base_state(
            generation,
            max_generations,
            stagnation,
            diversity,
            hv_trend,
        )
        snapshot = selector._score_snapshot()
        intra_state, intra_names = build_intra_state(
            slot,
            spec,
            [item["action"] for item in decisions],
            offspring_batches,
            population,
            archive,
            ref_points,
            snapshot["means"],
        )
        decision = selector.select_with_context(base_state, intra_state)
        pending_indices.append(decision.record_index)
        operator_seed = ledger.seed_for(
            "operator_internal", run_key, generation, slot, decision.action, 0
        )
        operator_key = ledger.record_for(
            "operator_internal", run_key, generation, slot, decision.action, 0
        )["key_hash"]
        start = slot * spec.pairs_per_batch
        stop = start + spec.pairs_per_batch
        batch, batch_lineage = apply_operator_to_frozen_pairs(
            frozen_pairs[start:stop],
            decision.action,
            np.random.RandomState(operator_seed),
            generation,
            slot,
            operator_key,
            operator_seed,
        )
        offspring_batches.append(batch)
        lineage.extend(batch_lineage)
        row = decision.as_dict()
        row.update(
            {
                "generation": generation,
                "slot": slot,
                "base_state": base_state.tolist(),
                "intra_state": intra_state.tolist(),
                "intra_state_names": intra_names,
                "operator_rng_seed": operator_seed,
                "operator_rng_key_hash": operator_key,
                "batch_pair_start": start,
                "batch_pair_stop": stop,
            }
        )
        selection_metadata = selector.decision_metadata(decision.record_index)
        if selection_metadata:
            row["controller_selection"] = selection_metadata
        decisions.append(row)

    offspring = [child for batch in offspring_batches for child in batch]
    if len(offspring) != spec.population_size:
        raise AssertionError("Fine-grained generation did not produce 100 offspring")
    if len({item["child_id"] for item in lineage}) != len(offspring):
        raise AssertionError("Offspring lineage IDs are not unique")

    combined = list(population) + offspring
    combined_rows = objective_array(combined)
    archive_after = update_archive_fn(
        list(archive), list(population), offspring, archive_size=archive_size
    )
    environmental_seed = ledger.seed_for(
        "environmental_selection", run_key, generation
    )
    environmental_record = ledger.record_for(
        "environmental_selection", run_key, generation
    )
    selected_indices = nsga3_select(
        combined_rows,
        spec.population_size,
        ref_points,
        rng=np.random.RandomState(environmental_seed),
    )
    new_population = [combined[int(index)] for index in selected_indices]
    selected_offspring_ids = {
        getattr(item, "_saos2_uid", None)
        for item in new_population
        if getattr(item, "_saos2_uid", None) is not None
    }
    global_survival = float(
        np.mean(
            [
                getattr(child, "_saos2_uid", None) in selected_offspring_ids
                for child in offspring
            ]
        )
    )
    new_rows = objective_array(new_population)
    current_hv = float(compute_hypervolume(new_rows, online_reference))
    hv_delta = float(
        np.clip(
            (current_hv - previous_hv) / max(abs(previous_hv), 1e-10),
            -1.0,
            1.0,
        )
    )
    current_best_cmax = float(new_rows[:, 0].min())
    cmax_delta = float(
        np.clip(
            (previous_best_cmax - current_best_cmax)
            / max(abs(previous_best_cmax), 1e-10),
            -1.0,
            1.0,
        )
    )
    global_reward = _global_reward(
        global_survival, hv_delta, cmax_delta, reward_scheme
    )
    candidates = candidate_batch_credits(
        population,
        offspring_batches,
        new_population,
        archive,
        archive_after,
        online_reference,
        ref_points,
    )
    decomposed = decompose_generation_reward(candidates, hv_delta, cmax_delta)
    assert_decomposition_contract(
        decomposed,
        global_survival,
        hv_delta,
        cmax_delta,
    )
    post_selection_credits = None
    if bool(getattr(selector, "requires_post_selection_feedback", False)):
        post_selection_credits, post_selection_full = post_selection_difference_credits(
            population,
            offspring_batches,
            online_reference,
            ref_points,
            previous_hv,
            previous_best_cmax,
            environmental_seed,
        )
        if post_selection_full["selected_indices"] != [int(index) for index in selected_indices]:
            raise AssertionError(
                "Post-selection full replay did not reproduce environmental selection"
            )
        if not np.isclose(
            float(post_selection_full["global_reward"]),
            global_reward,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Post-selection full replay did not reproduce global reward"
            )
    rewards = [item["candidate_composite_reward"] for item in decomposed]
    feedback_rows = []
    for slot, row in enumerate(decisions):
        feedback = {
            "record_index": int(row["record_index"]),
            "action": int(row["action"]),
            "generation": int(generation),
            "slot": int(slot),
            "features": np.concatenate(
                [
                    np.asarray(row["base_state"], dtype=float),
                    np.asarray(row["intra_state"], dtype=float),
                ]
            ),
            "base_state": list(row["base_state"]),
            "intra_state": list(row["intra_state"]),
            "target": (
                None
                if post_selection_credits is None
                else float(
                    post_selection_credits[slot][
                        "post_selection_difference_reward"
                    ]
                )
            ),
            "post_selection_difference_credit": (
                None
                if post_selection_credits is None
                else dict(post_selection_credits[slot])
            ),
        }
        if feedback["post_selection_difference_credit"] is not None:
            row["post_selection_difference_credit"] = dict(
                feedback["post_selection_difference_credit"]
            )
        feedback_rows.append(feedback)
    selector.complete_generation_with_feedback(
        pending_indices, rewards, feedback_rows
    )
    for slot, (decision, credit) in enumerate(zip(decisions, decomposed)):
        slot_lineage = [item for item in lineage if int(item["slot"]) == slot]
        start = int(decision["batch_pair_start"])
        stop = int(decision["batch_pair_stop"])
        slot_pairs = frozen_pairs[start:stop]
        decision["credit"] = dict(credit)
        decision["parent_pair_ids"] = [int(pair.global_pair) for pair in slot_pairs]
        decision["parent_ids"] = [
            {
                "pair_id": int(pair.global_pair),
                "parent1_index": int(pair.parent1_index),
                "parent2_index": int(pair.parent2_index),
                "parent1_fingerprint": chromosome_fingerprint(pair.parent1),
                "parent2_fingerprint": chromosome_fingerprint(pair.parent2),
            }
            for pair in slot_pairs
        ]
        decision["offspring_ids"] = [item["child_id"] for item in slot_lineage]
        decision["offspring_objective_vectors"] = [
            item["objectives"] for item in slot_lineage
        ]
        decision["rng_ledger_refs"] = {
            "parent_selection": parent_record["key_hash"],
            "operator_internal": decision["operator_rng_key_hash"],
            "environmental_selection": environmental_record["key_hash"],
        }
        decision["residual"] = {
            "status": "not_active_before_Q0_Q1_Q2",
            "raw_features": None,
            "predicted_residual_per_action": None,
            "final_score_per_action": None,
        }
        decision["model"] = {
            "learner": "none_phase1_ucb_only",
            "training_window_size": 0,
            "update_index": 0,
            "parameter_checkpoint_hash": None,
        }
        decision["accounting"] = {
            "selector_action_index": int(decision["record_index"]),
            "credit_observation_index": int(decision["record_index"]),
            "credit_available_after_generation": int(generation),
            "current_credit_can_affect_later_decisions_same_generation": False,
            "current_credit_can_affect_next_generation": True,
        }

    return FineGrainedGenerationResult(
        population=new_population,
        archive=list(archive_after),
        current_hv=current_hv,
        current_best_cmax=current_best_cmax,
        global_survival=global_survival,
        global_hv_delta=hv_delta,
        global_cmax_delta=cmax_delta,
        global_reward=global_reward,
        decisions=decisions,
        batch_credits=decomposed,
        lineage=lineage,
        parent_pairs=parent_pair_rows,
        invariants={
            "k": spec.k,
            "parent_pair_count": len(frozen_pairs),
            "batch_count": len(offspring_batches),
            "pairs_per_batch": spec.pairs_per_batch,
            "offspring_per_batch": spec.offspring_per_batch,
            "offspring_count": len(offspring),
            "environmental_selection_calls": 1,
            "archive_update_calls": 1,
            "selected_population_size": len(new_population),
            "schedule_evaluations_this_generation": len(offspring),
            "environmental_rng_seed": environmental_seed,
            "environmental_rng_key_hash": environmental_record["key_hash"],
            "parent_rng_key_hash": parent_record["key_hash"],
            "selected_indices": [int(index) for index in selected_indices],
        },
        offspring_batches=offspring_batches,
    )
