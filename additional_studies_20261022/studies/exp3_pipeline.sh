#!/bin/bash
# exp3 formal pipeline: waits for exp2 (A + B3) to finish, then runs R1, R2, R4, R3 sequentially.
cd /root/rivermind-data/exp2
PY=../env/bin/python
while ! grep -q all_done PIPELINE.log; do sleep 120; done
for S in R1 R2 R4 R3; do
  echo "{\"event\": \"start\", \"study\": \"$S\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE_exp3.log
  $PY exp3_run.py --release ../release --tasks tasks/$S.json --out out/$S --lanes 28 --lock locks/LOCK_$S.json > out_$S.log 2>&1
  echo "{\"event\": \"finish\", \"study\": \"$S\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE_exp3.log
done
for S in R1 R2 R4 R3; do nice -n 5 $PY exp3_analyze.py --study $S > out/$S/analysis_stdout.txt 2>&1; done
echo "{\"event\": \"all_done\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE_exp3.log
