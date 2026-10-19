# Supplementary studies, 18–19 October 2026

Code, protocols, analysis outputs, and per-run summaries for the three studies added in the revised manuscript *Resource-bound preselection for flexible job-shop scheduling with automated guided vehicles: when analytic candidate ranking pays off*.

Naming in the revised manuscript: **RBP** = the earlier "Always-scored" configuration (`always` in code), **RBP-P4** = "Physical-four" (`p4`), **RBP-P8** = "Period-eight" (`p8`), Random-four = `r4`, NSGA-II+RBP = `nsga2_pre`.

| Directory | Study | Manuscript | Runs |
|---|---|---|---:|
| `period/` | Equal-evaluation period study: RBP, RBP-P8, RBP-P4, Random-four at 10,000 evaluations | Section 6.2, Table 4 | 960 |
| `standard_comparators/` | NSGA-II, MOEA/D, NSGA-II+RBP against RBP and RBP-P4, at 10,000 evaluations and at 120 s | Section 6.3, Table 6 | 720 + 1,200 |
| `evaluation_cost/` | Fixed 10, 25, or 50 ms added to every exact evaluation, 120 s | Section 6.5, Table 9, Fig. 3 | 2,880 |
| `results/` | Statistical outputs (`analysis_*.json`), completion records (`FINISH_*.json`), and one summary line per run (`run_summaries/*.jsonl.gz`) | | 5,760 |
| `locks/` | SHA256 of the code and protocols, written before each formal run | | |

Each `PROTOCOL.md` fixed the arms, endpoints, and primary comparison family before its formal run. The files in this directory are byte-identical to the executed versions; `locks/` records their hashes and `MANIFEST.sha256` covers every file here.

## Inputs and engine

All three studies reuse the 80 reference-panel snapshots in `verification/instances/` (16 bases × 5 transport conditions) with optimization seeds 909302001–909302003, and import the released engine in `code/engine/` without modification. The period study reproduced the archived Physical-four and Random-four results (`verification/data/primary_run_metrics.json.gz`) exactly at all 1,440 compared endpoints.

## Running

Python 3.12 with the packages in the top-level `requirements.txt`. From the repository root:

```bash
python supplementary_studies_20261018/period/fe_period.py --release . --inputs verification/instances --output out_period --lanes 19
python supplementary_studies_20261018/standard_comparators/s2_moea.py --release . --inputs verification/instances --output out_eval --mode eval --arms nsga2,moead,nsga2_pre --lanes 19
python supplementary_studies_20261018/standard_comparators/s2_moea.py --release . --inputs verification/instances --output out_time --mode time --arms nsga2,moead,nsga2_pre,p4,always --lanes 19
python supplementary_studies_20261018/evaluation_cost/s3_cost.py --release . --inputs verification/instances --output out_cost --lanes 19
```

`s2_moea.py` looks for the period code through the environment variable `S1_DIR`, and `s3_cost.py` looks for `s2_moea.py` through `S2_DIR`. Set them to `supplementary_studies_20261018/period` and `supplementary_studies_20261018/standard_comparators`. Fixed-evaluation results are deterministic and hardware independent. Timed results depend on hardware and load. The original timed runs used 19 single-threaded workers on Intel Xeon E5-2696 v4 under a 20-CPU quota, with Python 3.12.13, NumPy 1.26.4, and SciPy 1.17.1.

## Analysis

```bash
python supplementary_studies_20261018/period/analyze_s1.py out_period --archived verification/data/primary_run_metrics.json.gz
python supplementary_studies_20261018/standard_comparators/analyze_s2.py --mode eval --s2 out_eval --s1 out_period --s1-code supplementary_studies_20261018/period
python supplementary_studies_20261018/standard_comparators/analyze_s2.py --mode time --s2 out_time --s1-code supplementary_studies_20261018/period
python supplementary_studies_20261018/evaluation_cost/analyze_s3.py --s3 out_cost --s2-time out_time --s1-code supplementary_studies_20261018/period --s2-code supplementary_studies_20261018/standard_comparators
```

The analysis scripts read the full per-run files. Those files, including saved fronts and genes at every endpoint, are provided as the release asset `full_run_records_20261018.tar` (SHA256 `692f6f4411831111c168790fa18558829e6f3245ee731ba2b03c29a583e78d19`). The compact `run_summaries` contain each run's endpoint HV values, evaluation counts, timing, and CPU ratio. They are sufficient to recompute the base-level effects reported in the manuscript.

## Scope

Evaluation cost is emulated by a fixed sleep after each exact decode. It does not replace a simulator. NSGA-II and MOEA/D are independent implementations with common literature defaults, not tuned on these instances.
