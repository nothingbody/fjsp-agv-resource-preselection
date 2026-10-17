# Extended experimental evidence and code v1.2.0

Resource-demand preselection for flexible job-shop scheduling with automated guided vehicles: evaluation efficiency and runtime trade-offs

Qingyou Wang, Hubei University of Automotive Technology. Journal of Intelligent Manufacturing submission materials. Correspondence: 202611413@huat.edu.cn. Version date: 2026-10-17. Release: https://github.com/nothingbody/fjsp-agv-resource-preselection/releases/tag/v1.2.0

The author has authorized public release of the selected generated data and author-owned code. This version extends v1.0.0 and v1.1.0 with the completed October studies, original processing-panel design evidence, frozen MOEA/D configuration and original Table 13 analysis sources. No new formal search experiment was run to assemble this release.

| Directory | Contents |
|---|---|
| october/ | Five cohorts comprising 6,320 compact run records, saved endpoints/final solutions, analysis tables, paired effects, original scientific source, configurations, inputs where redistributable, and execution receipts |
| legacy_processing_panel/ | 38-instance, six-method processing-panel study: 14 original dependence groups and construction source, MOEA/D frozen configuration/100 directions, 1,140 selected run records and timing table |
| legacy_model_sensitivity/ | Original model-sensitivity source including run_full.py/analyze_full.py, 1,440 saved endpoint fronts, original twelve-base pilot-exclusion analysis and statistical inputs |
| consolidated/ | Compact result tables and editable vector figures, drawing source and figure data |
| verification/ | New read-only checks of 344 reported numeric cells, Section 6.8 sign-flip/Holm inference, final-front HV and Table 13 bootstrap reproduction |
| provenance/ | Original/source/public-adapter hashes and export-selection records |

## Check the saved evidence

Using Python 3.12, run `python verification/verify_manifest.py` from this directory. To check the supplied manuscript tables, run:

```bash
python verification/verify_reported_tables.py --manuscript /path/to/Manuscript_JIM_author_review.docx --supplement /path/to/Supplementary_results_JIM.docx --output /path/to/local_check.json
python verification/table13_bootstrap/recheck.py
```

The table verifier uses the Python standard library. The bootstrap recheck also needs NumPy. It executes the exact original interval function with seed 2026091336 and 20,000 within-stratum draws. These commands do not launch a formal search cohort. New output files do not change the preserved source/observed results.

## External DN inputs

Complete original DN inputs are obtained directly from the cited upstream source rather than embedded here. Read `october/cie_dn_dual_20261014/README_DN_PUBLIC_ADAPTER.md` and use its restoration command. The public adapter reads the externally obtained transport matrix and benchmark fixture; all algorithmic decoding/selection logic remains unchanged. Its original/public SHA256 and input reconstruction checks are recorded in PUBLIC_ADAPTER_MANIFEST.json and the validation report. Native makespan and the author-added three-objective extension remain separately labeled in every analysis.

## Reproduction scope

Original frozen code and protocols retain their historical paths and versions. Top-level instructions describe the current public export; dated preparation READMEs describe historical snapshots. Compact records retain saved key nodes and final solutions, while large per-step histories are omitted. They therefore support the supplied aggregate checks but do not substitute for every strict original trajectory validator or every intermediate attainment crossing. Repeating timed searches creates new hardware-dependent observations. Original locks remain provenance; the explicitly modified DN public loading adapter requires a new execution lock for a new run.

All six previously unlocated historical scripts have now been recovered against original source hashes: the four preparation scripts and model-sensitivity run_full.py/analyze_full.py. The historical GitHub backup uploader publish_preparation.py is documented by hash rather than distributed as a scientific execution entry point. Earlier recovery reports remain dated records; the additional recovery manifests supersede their source-absence findings. No formal generated-instance equal-evaluation Always-scored–Physical-four cohort was located, so this release does not relabel the observed equal-time comparisons.

The code-only ZIP contains the recovered Python source, configuration/protocol records and licences. Download the complete evidence ZIP to obtain selected result data and saved fronts. The source-only ZIP is not a replacement for the complete evidence inventory.

## Licences and citation

Author-owned code: MIT. Author-generated results: CC BY 4.0. Third-party inputs and the Apache drawing helper retain separate notices in THIRD_PARTY_NOTICES.md. Cite this version using CITATION.cff and the versioned release URL. No DOI has been created for this GitHub release.
