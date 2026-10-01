#!/usr/bin/env bash
# Site 2, Study 1 chain (PREREG_SITE2.md): wait for s2_14 (R1) to exit, verify 120 runs, then
# R1 tables -> CNN CV -> CNN final -> autumn -> BAG. Resumable via ../runs/STUDY1/logs/<step>.ok
set -euo pipefail
cd "$(dirname "$0")"
PY=D:/anaconda/envs/sar/python.exe
export PYTHONUTF8=1
L=../runs/STUDY1/logs; mkdir -p $L
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*s2_14_study1_R1.py*' }) { exit 0 } else { exit 1 }"; do sleep 120; done
n=$(ls ../runs/STUDY1/ws/runs_R1/*/predictions.parquet 2>/dev/null | wc -l)
echo "R1 runs with predictions: $n"
[ "$n" -eq 120 ] || { echo "STOP: R1 incomplete ($n/120)"; exit 1; }
run() { local k=$1; shift; [ -f $L/$k.ok ] && return 0; { echo "[$(date -Iseconds)] start $k"; "$@"; echo "[$(date -Iseconds)] done"; } > $L/$k.log 2>&1; touch $L/$k.ok; }
run r1tab  $PY s2_15_bootstrap.py r1
run cnncv  $PY s2_16_study1_cnn.py --stage cv
run cnnfin $PY s2_16_study1_cnn.py --stage final
run autumn $PY s2_18_study1_autumn.py
run bag    $PY s2_19_study1_bag.py
echo "[$(date -Iseconds)] SITE2 STUDY1 COMPLETE" > $L/SITE2_S1_DONE
