"""Fine-grained residual adaptive operator-selection components."""

from .branching import (
    CounterfactualSnapshot,
    ImmediateBranchResult,
    branch_all_actions,
    collect_ucb_snapshots,
    continue_branch,
    evaluate_immediate_branch,
)
from .finegrained import (
    DelayedExposureUCB,
    FineGrainedGenerationResult,
    GranularitySpec,
    execute_finegrained_generation,
)
from .q0_analysis import cluster_bootstrap_q0
from .rng_ledger import RNGLedger, derive_seed

__all__ = [
    "CounterfactualSnapshot",
    "DelayedExposureUCB",
    "FineGrainedGenerationResult",
    "GranularitySpec",
    "ImmediateBranchResult",
    "RNGLedger",
    "branch_all_actions",
    "cluster_bootstrap_q0",
    "collect_ucb_snapshots",
    "continue_branch",
    "derive_seed",
    "evaluate_immediate_branch",
    "execute_finegrained_generation",
]
