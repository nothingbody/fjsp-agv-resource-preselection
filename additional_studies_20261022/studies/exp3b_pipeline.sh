#!/bin/bash
# exp3b pipeline: waits for exp3 to finish, then T1 tuning -> selection -> locks -> S1, S3, S4, S2 -> analyses.
cd /root/rivermind-data/exp2
PY=../env/bin/python
while ! grep -q all_done PIPELINE_exp3.log 2>/dev/null; do sleep 120; done
log(){ echo "{\"event\": \"$1\", \"study\": \"$2\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE_exp3b.log; }
log start T1
$PY exp3b_run.py --release ../release --tasks tasks/T1.json --out out/T1 --lanes 28 --lock locks/LOCK_T1.json > out_T1.log 2>&1
log finish T1
grep -q '"failed": 0' out/T1/FINISH.json || { log stopped T1_failures; exit 1; }
$PY exp3b_tasks.py select --runs out/T1 > out/T1/SELECTION_stdout.txt 2>&1 || { log stopped select; exit 1; }
for S in S1 S2 S3 S4; do $PY exp3b_tasks.py make --study $S; $PY exp3b_tasks.py lock --study $S; done >> out/T1/SELECTION_stdout.txt 2>&1
log locks_written S
for S in S1 S3 S4 S2; do
  log start $S
  $PY exp3b_run.py --release ../release --tasks tasks/$S.json --out out/$S --lanes 28 --lock locks/LOCK_$S.json > out_$S.log 2>&1
  log finish $S
done
for S in S1 S3 S4 S2; do nice -n 5 $PY exp3_analyze.py --study $S > out/$S/analysis_stdout.txt 2>&1; done
log all_done all
