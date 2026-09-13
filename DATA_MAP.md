# Evidence and access map

All archive paths below are inside the release asset named in DATA_ASSET.json.

| Evidence | Data location | Scope |
|---|---|---|
| Matched primary and information ablations | `01_manuscript_results/study30/` | 16 generated bases, five regimes, three seeds; repeated rows are endpoints, not independent instances |
| Weighted/simple scores and nearest-neighbor control | `01_manuscript_results/study31/`, `study32/` | Separate frozen study blocks; use their own normalization and families |
| Original ranking audit | `02_metrics_and_mechanism/AUDITED_POOLS.json.gz` | 1,920 pools; four exact scalar scores and four proxy scores per pool |
| Microbenchmark | `02_metrics_and_mechanism/cost/` | Paired measurements and conditions, not algorithm-intrinsic timing constants |
| Exact-selection, equal-step, reference exclusion and archive analyses | `02_metrics_and_mechanism/` | Saved fronts and derived cells; extra diagnostic evaluations are not optimization budget |
| Three decoder semantics | `03_model_sensitivity/` | Original, restricted first-operation rightshift, and no same-machine service; not a combined repair |
| Pool size, matched NSGA-II and measured time budgets | `04_enhancement_3_8/` | Completed enhancement outcomes and reference/endpoint metadata |
| Generated instance snapshots | `05_generated_instances/` | 240 snapshots in three studies; regimes are related, not 240 independent bases |
| Calibration and structural/external-processing references | `07_supporting_studies/` | Derived results only; see THIRD_PARTY.md for original inputs |
| Final uncertainty sensitivity | Repository `analysis/evidence/` and `verification/sensitivity_results.json` | 23 base-level contrasts, preserving original estimates and adding retrospective sensitivity |

`verification/SOURCE_MANIFEST.json` maps every staged checker input back to its archive member and SHA-256. `CODE_MANIFEST.json` distinguishes frozen source bytes from the one environment-path portability edit. Original cluster worker managers are not the public entry point; `run.py` calls the same scientific routines directly. Those managers depend on historical lock and job directories that are deliberately not redistributed.

The release supports inspection and reaggregation of the key findings plus new executions with the provided engines. It does not claim byte-for-byte reconstruction of every historical event stream, every figure builder or the private storage/backup workflow. Manuscript LaTeX and submission correspondence are delivered separately to the author and are not published here.
