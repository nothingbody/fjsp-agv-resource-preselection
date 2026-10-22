# Tuned algorithm comparisons and supplementary evidence v1.4.0

This release supports the manuscript *Resource-bound preselection for flexible job-shop scheduling with automated guided vehicles: when analytic candidate ranking pays off* by Qingyou Wang. It adds the complete new formal optimization outputs, generated instance snapshots, frozen task tables and configurations, saved analyses, and the corresponding study programs. Earlier releases remain available at their original version links.

| Study family | Formal optimizer runs | Main purpose |
|---|---:|---|
| Standard tuning and comparisons, pool size, evaluation cost, stochastic evaluation | 17,088 | Tuned NSGA-II/MOEA/D hosts and workload dependence |
| Learning-guided adaptations before tuning | 5,760 | Descriptive comparison batch R1-R4 |
| Learning-guided tuning and tuned comparisons | 7,776 | T1 and S1-S4, including every adverse and null result |
| CP-SAT relaxation diagnostic | 80 solver inputs | Per-input bounds and aggregate gap diagnostics |

The total is 30,624 formal optimization runs plus 80 relaxation solves. Engineering runs and the invalidated pre-formal environment checks are separate from these counts and are not included as formal observations. All formal studies completed without a failed or duplicate output. The archive preparation verifies the stored records and does not rerun optimization, training, bootstrap inference, sign-flip tests or CP-SAT.

## Download and structure

The compact ZIP contains the released base engine plus `additional_studies_20261022/`. That folder contains study programs and original inputs/tasks/locks under `studies/`, saved analyses and completion/progress records under `out/`, per-run summaries under `run_summaries/`, selected table data, manuscript figures and verification metadata. Unpack the three `full_records_*_v1.4.0.tar` assets into that folder to add every original compressed run output and saved front/chromosome endpoint. Archive members are paths under `out/`; no private connection files are included.

`MANIFEST.json` describes the compact package files. `PUBLIC_RELEASE_MANIFEST.json` and `SHA256SUMS.txt` identify downloadable assets. The supplied standard-library verification script checks compact file hashes and archived file sets against the frozen tasks. `VALIDATION.json` records the checks completed during release preparation.

## Reading the evidence

- Main v6 Table 7 uses B2e/B2t, and Table 8 uses S1/S2/S3. Full S4 data remain archived despite the brief manuscript discussion.
- The learning-guided algorithms are adaptations based on the cited papers, with the stated grids and selected parameters. They are distinct from original-author implementations.
- Base instances are the independent experimental units. Transport conditions, seeds and saved endpoints do not increase the independent sample count.
- Normalized HV differences are absolute differences, not relative percentages. Stored confidence intervals are individual intervals. Holm families and all nonsignificant contrasts retain their original definitions.
- Before/after tuning comparisons are descriptive across batches. The test panel was previously observed; disjoint tuning bases do not turn it into a new independent confirmation panel.
- ILMPEA-type expert generation and training are charged to elapsed runtime, while expert calls remain outside the online-evaluation count.
- The two-objective sensitivity output has a constant load coordinate; its secondary load statistic carries no information.
- CP-SAT solves a relaxation. Five cases were solved to optimality and 75 returned feasible solutions with best bounds. This does not establish full-model scheduling optimality.

## Environment and execution

Use Python 3.12.13 and the pinned versions in `studies/ENV_FREEZE.txt`; the principal numerical dependencies are NumPy 1.26.4, SciPy 1.17.1, pymoo 0.6.1.5 and OR-Tools 9.14.6206. Original runs used single-thread numerical libraries, 28 lanes and a 30-core quota. Runtime results also depend on hardware and contention.

From `additional_studies_20261022/studies/`, a corresponding explicit experiment command uses `--release ../..`, the archived `tasks/<study>.json`, the original lock and the matching exp2/exp3/exp3b runner. Avoid changing frozen files when reproducing a named study. The full pipeline scripts are preserved as original execution records; they are not invoked by the archive verifier.

## Licences and citation

Author-owned code is MIT; author-generated data are CC BY 4.0. The earlier licence and third-party notices are retained. NumPy, SciPy, pymoo and OR-Tools are external dependencies with their own terms. This release contains author-generated synthetic inputs; original third-party benchmark sources used in earlier releases retain their existing acquisition routes and notices.

Cite Qingyou Wang, *Tuned algorithm comparisons and supplementary evidence for resource-bound preselection in FJSP-AGV scheduling*, version v1.4.0, https://github.com/nothingbody/fjsp-agv-resource-preselection/releases/tag/v1.4.0 . The version link is the public archive identifier; a DOI has not been assigned.
