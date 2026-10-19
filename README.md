# Resource-demand preselection for FJSP-AGV scheduling

## Supplementary studies release v1.3.0

The [v1.3.0 release](https://github.com/nothingbody/fjsp-agv-resource-preselection/releases/tag/v1.3.0) adds three studies used in the revised manuscript *Resource-bound preselection for flexible job-shop scheduling with automated guided vehicles: when analytic candidate ranking pays off*. In that manuscript the method is called resource-bound preselection (RBP); the earlier Always-scored configuration is RBP, and Physical-four is RBP-P4.

- An equal-evaluation comparison of scoring periods (RBP, RBP-P8, RBP-P4, Random-four).
- Comparisons with standard NSGA-II and MOEA/D at 10,000 evaluations and at 120 s, including an NSGA-II preselection plug-in.
- An evaluation-cost study that adds 10, 25, or 50 ms to every exact evaluation.

Code, protocols, analysis outputs, and 5,760 per-run summaries are in [supplementary_studies_20261018/](supplementary_studies_20261018/README.md). Complete per-run records are attached to the release as `full_run_records_20261018.tar`. Everything below, including v1.2.0, is unchanged.

## Extended evidence release v1.2.0

The [v1.2.0 release](https://github.com/nothingbody/fjsp-agv-resource-preselection/releases/tag/v1.2.0) supplies the completed October experiments, selected endpoint/final-solution data, recovered historical preparation and model-sensitivity code, original dependence-group evidence, and frozen MOEA/D configuration. It accompanies the revised manuscript *Resource-demand preselection for flexible job-shop scheduling with automated guided vehicles: evaluation efficiency and runtime trade-offs* by Qingyou Wang.

Download the complete `fjsp_agv_extended_evidence_v1.2.0.zip` or the smaller source-only ZIP from that release. Use the release SHA256SUMS and per-file manifest for integrity checks. The five October cohorts comprise 6,320 compact runs; the original processing-panel study also supplies 1,140 selected run records. Favorable, null and unfavorable results remain included. Large per-step histories and private correspondence are excluded.

DN source inputs are acquired directly from their original fixed-version source using the supplied restoration tool. The public DN loading adapter is separately identified from the preserved original source hashes. Author-owned code uses MIT and generated results use CC BY 4.0; third-party inputs/helpers retain their own notices. The earlier v1.0.0/v1.1.0 releases remain unchanged. No DOI has been assigned to this GitHub release.

See [the versioned release guide](releases/v1.2.0/README.md) for coverage and reproduction instructions.

## Earlier v1.0.0 implementation guide

The following guide documents the original implementation snapshot. Later studies are supplied by v1.2.0 above.

Code and selected computational evidence accompanying **Resource-demand preselection for flexible job-shop scheduling with automated guided vehicles: evidence under evaluation and time budgets**, by Qingyou Wang (research manuscript, 2026; not an accepted article).

## Start here

Download `fjsp_agv_key_data_public_v1.zip` from the [v1.0.0 release](https://github.com/nothingbody/fjsp-agv-resource-preselection/releases/tag/v1.0.0). Its SHA-256 is in `DATA_ASSET.json`. No account, original server or private repository is required.

```sh
python prepare_data.py /path/to/fjsp_agv_key_data_public_v1.zip
python verification/check_records.py
python verification/check_effects.py
```

These standard-library checks recompute three matched effects, all 1,920 original candidate-pool statistics, and the 23 supplied base-level contrasts. They check saved evidence; they are not a re-execution of every optimizer run, nor independent external validation.

For search execution, use Python 3.12 and `pip install -r requirements.txt` in a virtual environment. For a short execution test:

```sh
python run.py --instance verification/instances/comp30_small_low_00_a3_t030.json --evaluations 200 --output outputs/smoke.json
```

`run.py --help` exposes the primary/ablation, simple-score, nearest-neighbor, pool-size, matched NSGA-II, wall-time and model-semantics engines. Use the frozen dataset/name/seed and full budget from the data task lists for a corresponding scientific run. Changing the namespace changes random streams. A short smoke test is not one of the formal experiments. Wall-time outcomes also depend on hardware and contention.

## Scope and navigation

| Location | Contents |
|---|---|
| `code/engine/` | Scientific dependencies, preserving original study-directory names |
| `verification/` | Independent arithmetic checks, 23-contrast data and source manifest |
| `analysis/` | Uncertainty-sensitivity reconstruction and its five saved input summaries |
| `DATA_MAP.md` | Manuscript claim to data location and access limits |
| `CODE_MANIFEST.json` | Original/public source hashes and packaging changes |
| `VALIDATION.json` | Tests actually performed for this release |

The data asset contains generated instances, per-run endpoint metrics, saved fronts, pool observations, component timings, model sensitivities and completed enhancement results. Historical labels in the archive identify the frozen studies; the manuscript version is 3.8.1 (2026-09-14), and the added 23-contrast analysis is in this repository. The older README inside the immutable data asset describes its original snapshot; this README and `DATA_MAP.md` describe the current release.

This is a curated release, **not the entire historical archive**. Full evaluation streams, redundant replays, operational logs, private submission correspondence and prior manuscripts are excluded. The supplied endpoint/front records support recomputation of reported aggregates, but cannot replace every trajectory-level diagnostic. There is no completed genuinely transport-native external benchmark experiment in this release.

## Scientific interpretation

The matched full-versus-random comparison is primary; UCB is a framework-level reference. The reported primary normalized-HV difference is approximately 0.0248. The mean of the individual pool-level captured-value fractions is 54.1%, not a ratio of pooled improvements. Pool size, archival eligibility and exploration conventions must be retained when interpreting exact-selection controls. The 120-second comparison is specific to the measured execution conditions, not a hardware-independent speedup claim. Consult the manuscript and released detailed protocols for the full limitations.

## Licenses and third-party inputs

Author-owned code is MIT (`LICENSE`). Released author-generated data are CC BY 4.0 (`LICENSE-DATA.md`). NumPy, SciPy and pymoo are installed separately and retain their own licenses. Third-party original benchmark files are **not redistributed**. Processing-instance provenance points to [SchedulingLab/fjsp-instances at the recorded commit](https://github.com/SchedulingLab/fjsp-instances/tree/ac4c3402312bfbeafcf4472d78be567d4e6b46ab); obtain inputs from their source and check its conditions. Transport layers used with those processing benchmarks are author-defined, not native transport benchmarks. See `THIRD_PARTY.md`.

## Citation and versioning

Use `CITATION.cff` and cite release **v1.0.0** with its URL. No DOI has been minted by GitHub, and this repository is openly author-identified, not anonymous. No private archive has been made public. Manuscript changes after this release require a new version or an explicitly documented correction.
