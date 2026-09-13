"""Candidate fine-grained credit signals and conservative decomposition.

These signals are instrumentation for Phase 1/2.  No candidate is declared the
primary Residual-UCB reward until the downstream-alignment gate (Q0) passes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.algorithm.nsga3.selection import (
    compute_hypervolume,
    non_dominated_sort,
    nsga3_select,
)


EPS = 1e-12
POST_SELECTION_CREDIT_PROTOCOL = "saos2_post_selection_difference_credit_v1_20260828"


def objective_array(items: Iterable) -> np.ndarray:
    rows = [np.asarray(item.objectives.to_array(), dtype=float) for item in items]
    if not rows:
        return np.empty((0, 3), dtype=float)
    result = np.asarray(rows, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("Expected a two-dimensional three-objective array")
    if not np.isfinite(result).all():
        raise ValueError("Objective array contains a non-finite value")
    return result


def nondominated_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    if len(rows) == 0:
        return rows.reshape(0, rows.shape[1] if rows.ndim == 2 else 3)
    unique = []
    seen = set()
    for row in rows:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    unique = np.asarray(unique, dtype=float)
    return unique[non_dominated_sort(unique)[0]]


def fixed_reference_hv(rows: np.ndarray, reference: np.ndarray) -> float:
    rows = nondominated_rows(rows)
    if len(rows) == 0:
        return 0.0
    reference = np.asarray(reference, dtype=float)
    inside = np.all(rows <= reference + EPS, axis=1)
    rows = rows[inside]
    if len(rows) == 0:
        return 0.0
    return float(compute_hypervolume(rows, reference))


def _objective_keys(items: Sequence) -> set:
    return {
        tuple(np.asarray(item.objectives.to_array(), dtype=float).tolist())
        for item in items
    }


def _associate_reference_directions(rows: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    ref_points = np.asarray(ref_points, dtype=float)
    if len(rows) == 0:
        return np.empty(0, dtype=int)
    ideal = rows.min(axis=0)
    span = rows.max(axis=0) - ideal
    span = np.where(span > EPS, span, 1.0)
    normalized = (rows - ideal) / span
    norms = np.linalg.norm(ref_points, axis=1)
    if np.any(norms <= EPS):
        raise ValueError("Reference direction must be non-zero")
    unit = ref_points / norms[:, None]
    associations = []
    for row in normalized:
        projections = (unit @ row)[:, None] * unit
        distances = np.linalg.norm(row - projections, axis=1)
        associations.append(int(np.argmin(distances)))
    return np.asarray(associations, dtype=int)


def _parent_relative_summary(batch: Sequence) -> Tuple[float, float]:
    signed = []
    dominates_parent = []
    for child in batch:
        lineage = getattr(child, "_saos2_lineage", None)
        if not lineage or "parent_objectives" not in lineage:
            continue
        parent_rows = np.asarray(
            lineage.get("credit_parent_objectives", lineage["parent_objectives"]),
            dtype=float,
        )
        parent_center = parent_rows.mean(axis=0)
        child_row = np.asarray(child.objectives.to_array(), dtype=float)
        relative = (parent_center - child_row) / np.maximum(np.abs(parent_center), EPS)
        signed.append(float(np.mean(relative)))
        dominates_parent.append(
            float(
                any(
                    np.all(child_row <= parent) and np.any(child_row < parent)
                    for parent in parent_rows
                )
            )
        )
    return (
        float(np.mean(signed)) if signed else 0.0,
        float(np.mean(dominates_parent)) if dominates_parent else 0.0,
    )


def candidate_batch_credits(
    parents: Sequence,
    offspring_batches: Sequence[Sequence],
    selected_population: Sequence,
    archive_before: Sequence,
    archive_after: Sequence,
    reference_point: np.ndarray,
    ref_directions: np.ndarray,
) -> List[Dict[str, float]]:
    """Compute the unfrozen Phase 2 candidate-credit family."""

    all_offspring = [child for batch in offspring_batches for child in batch]
    all_rows = objective_array(list(parents) + all_offspring)
    parent_rows = objective_array(parents)
    full_hv = fixed_reference_hv(all_rows, reference_point)
    selected_ids = {
        getattr(item, "_saos2_uid", None)
        for item in selected_population
        if getattr(item, "_saos2_uid", None) is not None
    }
    archive_before_keys = _objective_keys(archive_before)
    archive_after_keys = _objective_keys(archive_after)
    combined_front = set(
        tuple(row.tolist()) for row in nondominated_rows(all_rows)
    )

    all_assoc_rows = np.vstack([parent_rows, objective_array(all_offspring)])
    all_assoc = _associate_reference_directions(all_assoc_rows, ref_directions)
    parent_count = len(parent_rows)
    parent_assoc = set(all_assoc[:parent_count].tolist())

    results: List[Dict[str, float]] = []
    cursor = 0
    full_best_cmax = float(all_rows[:, 0].min())
    for batch in offspring_batches:
        batch_rows = objective_array(batch)
        without_rows = np.vstack(
            [
                parent_rows,
                objective_array(
                    [
                        child
                        for other in offspring_batches
                        if other is not batch
                        for child in other
                    ]
                ),
            ]
        )
        without_hv = fixed_reference_hv(without_rows, reference_point)
        without_best_cmax = float(without_rows[:, 0].min())
        survival = float(
            np.mean(
                [getattr(child, "_saos2_uid", None) in selected_ids for child in batch]
            )
        )
        batch_keys = [tuple(row.tolist()) for row in batch_rows]
        archive_insertions = sum(
            key in archive_after_keys and key not in archive_before_keys
            for key in batch_keys
        )
        nd_count = sum(key in combined_front for key in batch_keys)
        batch_assoc = all_assoc[parent_count + cursor : parent_count + cursor + len(batch)]
        cursor += len(batch)
        occupied = set(batch_assoc.tolist())
        parent_relative, parent_dominance = _parent_relative_summary(batch)
        results.append(
            {
                "survival_exact": survival,
                "lobo_hv": float(max(0.0, full_hv - without_hv)),
                "reference_occupancy_count": float(len(occupied)),
                "reference_new_direction_count": float(len(occupied - parent_assoc)),
                "archive_insertion_rate": float(archive_insertions / max(len(batch), 1)),
                "nondominated_rate": float(nd_count / max(len(batch), 1)),
                "parent_relative_improvement": parent_relative,
                "parent_dominance_rate": parent_dominance,
                "makespan_marginal": float(
                    max(0.0, without_best_cmax - full_best_cmax)
                    / max(abs(without_best_cmax), EPS)
                ),
            }
        )
    return results


def _conservative_allocation(global_value: float, weights: np.ndarray) -> np.ndarray:
    """Allocate so the batch mean equals ``global_value`` exactly in theory."""

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("Allocation weights must be a non-empty vector")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if weights.sum() <= EPS:
        return np.full(len(weights), float(global_value), dtype=float)
    return float(global_value) * len(weights) * weights / weights.sum()


def decompose_generation_reward(
    candidate_credits: Sequence[Dict[str, float]],
    global_hv_delta: float,
    global_cmax_delta: float,
) -> List[Dict[str, float]]:
    """Candidate 0.5/0.3/0.2 decomposition with a K=1 identity property.

    This is logged as a candidate only.  Q0 decides whether it is aligned with
    downstream HV before it can become the Residual-UCB training reward.
    """

    if not candidate_credits:
        raise ValueError("At least one batch credit is required")
    survival = np.asarray(
        [item["survival_exact"] for item in candidate_credits], dtype=float
    )
    hv = _conservative_allocation(
        global_hv_delta,
        np.asarray([item["lobo_hv"] for item in candidate_credits], dtype=float),
    )
    cmax = _conservative_allocation(
        global_cmax_delta,
        np.asarray(
            [item["makespan_marginal"] for item in candidate_credits], dtype=float
        ),
    )
    outputs = []
    for idx, source in enumerate(candidate_credits):
        row = dict(source)
        row["allocated_hv_delta"] = float(hv[idx])
        row["allocated_cmax_delta"] = float(cmax[idx])
        row["candidate_composite_reward"] = float(
            0.5 * survival[idx] + 0.3 * hv[idx] + 0.2 * cmax[idx]
        )
        outputs.append(row)
    return outputs


def assert_decomposition_contract(
    decomposed: Sequence[Dict[str, float]],
    global_survival: float,
    global_hv_delta: float,
    global_cmax_delta: float,
    atol: float = 1e-12,
) -> None:
    survival = np.mean([item["survival_exact"] for item in decomposed])
    hv = np.mean([item["allocated_hv_delta"] for item in decomposed])
    cmax = np.mean([item["allocated_cmax_delta"] for item in decomposed])
    if not np.isclose(survival, global_survival, rtol=0.0, atol=atol):
        raise AssertionError("Batch survival does not average to global survival")
    if not np.isclose(hv, global_hv_delta, rtol=0.0, atol=atol):
        raise AssertionError("Batch HV allocation does not conserve global HV delta")
    if not np.isclose(cmax, global_cmax_delta, rtol=0.0, atol=atol):
        raise AssertionError("Batch Cmax allocation does not conserve global Cmax delta")


def _post_selection_metrics(
    parents: Sequence,
    offspring_batches: Sequence[Sequence],
    reference_point: np.ndarray,
    ref_directions: np.ndarray,
    previous_hv: float,
    previous_best_cmax: float,
    environmental_seed: int,
    nominal_offspring_count: int,
) -> Dict[str, float | List[int]]:
    """Re-evaluate the frozen global reward after environmental selection.

    ``nominal_offspring_count`` is deliberately held fixed when one batch is
    removed. Thus the omitted batch contributes zero survivors rather than
    changing the denominator of the survival component. This makes the full
    and leave-one-batch rewards directly comparable.

    The function operates only on already decoded chromosomes and cached
    objective vectors. It must therefore never invoke schedule evaluation.
    """

    if nominal_offspring_count <= 0:
        raise ValueError("Nominal offspring count must be positive")
    offspring = [child for batch in offspring_batches for child in batch]
    if len(parents) < nominal_offspring_count:
        raise ValueError("Parent population is too small for environmental selection")
    if not offspring:
        raise ValueError("Post-selection credit requires at least one offspring")

    combined = list(parents) + offspring
    selected_indices = nsga3_select(
        objective_array(combined),
        len(parents),
        ref_directions,
        rng=np.random.RandomState(int(environmental_seed)),
    )
    selected = [combined[int(index)] for index in selected_indices]
    selected_offspring_ids = {
        getattr(item, "_saos2_uid", None)
        for item in selected
        if getattr(item, "_saos2_uid", None) is not None
    }
    selected_offspring_count = int(
        sum(
            getattr(child, "_saos2_uid", None) in selected_offspring_ids
            for child in offspring
        )
    )
    survival = float(selected_offspring_count / nominal_offspring_count)
    selected_rows = objective_array(selected)
    current_hv = float(compute_hypervolume(selected_rows, reference_point))
    hv_delta = float(
        np.clip(
            (current_hv - previous_hv) / max(abs(previous_hv), 1e-10),
            -1.0,
            1.0,
        )
    )
    current_best_cmax = float(selected_rows[:, 0].min())
    cmax_delta = float(
        np.clip(
            (previous_best_cmax - current_best_cmax)
            / max(abs(previous_best_cmax), 1e-10),
            -1.0,
            1.0,
        )
    )
    reward = float(0.5 * survival + 0.3 * hv_delta + 0.2 * cmax_delta)
    return {
        "survival": survival,
        "population_hv": current_hv,
        "hv_delta": hv_delta,
        "best_cmax": current_best_cmax,
        "cmax_delta": cmax_delta,
        "global_reward": reward,
        "selected_indices": [int(index) for index in selected_indices],
        "selected_offspring_count": selected_offspring_count,
        "available_offspring_count": int(len(offspring)),
    }


def post_selection_difference_credits(
    parents: Sequence,
    offspring_batches: Sequence[Sequence],
    reference_point: np.ndarray,
    ref_directions: np.ndarray,
    previous_hv: float,
    previous_best_cmax: float,
    environmental_seed: int,
) -> Tuple[List[Dict[str, float | List[int] | str]], Dict[str, float | List[int]]]:
    """Compute a leave-one-batch credit after re-running NSGA-III selection.

    For batch ``b`` the credit is ``R_full - R_without_b``. ``R`` is the
    already frozen 0.5/0.3/0.2 global reward, recomputed after environmental
    selection using cached objective values. Consequently the *credit
    calculation itself* has zero schedule evaluations. Regenerating an
    unpersisted counterfactual batch is a separate, explicitly accounted task.
    """

    if not offspring_batches:
        raise ValueError("Post-selection credit requires at least one batch")
    if any(not batch for batch in offspring_batches):
        raise ValueError("Post-selection credit does not allow empty batches")
    nominal_offspring_count = sum(len(batch) for batch in offspring_batches)
    if len(parents) != nominal_offspring_count:
        raise ValueError(
            "The frozen Phase 2R design requires equal parent and offspring counts"
        )

    full = _post_selection_metrics(
        parents,
        offspring_batches,
        reference_point,
        ref_directions,
        previous_hv,
        previous_best_cmax,
        environmental_seed,
        nominal_offspring_count,
    )
    credits: List[Dict[str, float | List[int] | str]] = []
    for batch_index in range(len(offspring_batches)):
        retained_batches = [
            batch for index, batch in enumerate(offspring_batches) if index != batch_index
        ]
        without = _post_selection_metrics(
            parents,
            retained_batches,
            reference_point,
            ref_directions,
            previous_hv,
            previous_best_cmax,
            environmental_seed,
            nominal_offspring_count,
        )
        credits.append(
            {
                "protocol": POST_SELECTION_CREDIT_PROTOCOL,
                "credit_name": "post_selection_difference_reward",
                "batch_index": int(batch_index),
                "nominal_offspring_count": int(nominal_offspring_count),
                "withheld_batch_size": int(len(offspring_batches[batch_index])),
                "full_global_reward": float(full["global_reward"]),
                "without_global_reward": float(without["global_reward"]),
                "post_selection_difference_reward": float(
                    full["global_reward"] - without["global_reward"]
                ),
                "post_selection_difference_survival": float(
                    full["survival"] - without["survival"]
                ),
                "post_selection_difference_hv_delta": float(
                    full["hv_delta"] - without["hv_delta"]
                ),
                "post_selection_difference_cmax_delta": float(
                    full["cmax_delta"] - without["cmax_delta"]
                ),
                "post_selection_difference_population_hv": float(
                    full["population_hv"] - without["population_hv"]
                ),
                "full_selected_offspring_count": int(
                    full["selected_offspring_count"]
                ),
                "without_selected_offspring_count": int(
                    without["selected_offspring_count"]
                ),
                "without_available_offspring_count": int(
                    without["available_offspring_count"]
                ),
                "full_selected_indices": list(full["selected_indices"]),
                "without_selected_indices": list(without["selected_indices"]),
                "schedule_evaluations_for_credit": 0,
            }
        )
    return credits, full
