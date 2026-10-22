#!/bin/bash
# Sequential formal pipeline after B1: select -> tasks -> locks -> runs. Timed studies never overlap.
set -e
cd /root/rivermind-data/exp2
PY=../env/bin/python
while [ ! -f out/B1/FINISH.json ]; do sleep 60; done
grep -q '"failed": 0' out/B1/FINISH.json || { echo "B1 has failures" > PIPELINE_STOPPED.txt; exit 1; }
$PY exp2_tasks.py select --runs out/B1 > out/B1/SELECTION_stdout.txt
for S in B2e B2t Ce Ct A; do $PY exp2_tasks.py make --study $S --selection SELECTION.json; done
for S in B2e B2t Ce Ct A; do $PY exp2_tasks.py lock --study $S > /dev/null; done
echo "{\"event\": \"locks_written\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE.log
for S in B2e Ce B2t Ct A; do
  echo "{\"event\": \"start\", \"study\": \"$S\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE.log
  $PY exp2_run.py --release ../release --tasks tasks/$S.json --out out/$S --lanes 28 --lock locks/LOCK_$S.json > out_$S.log 2>&1
  echo "{\"event\": \"finish\", \"study\": \"$S\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE.log
done
$PY exp2_cpsat.py --panel test --seconds 300 --workers 4 --lanes 7 --out out/B3 > out_B3.log 2>&1
echo "{\"event\": \"all_done\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> PIPELINE.log
