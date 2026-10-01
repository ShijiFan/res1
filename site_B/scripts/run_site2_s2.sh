#!/usr/bin/env bash
# Site 2, Study 2 (PREREG_SITE2.md): experiments with the frozen A1 protocol (no E2w: no 6-day windows),
# then E5r and the statistics. Resumable via ../runs/MAIN/logs/<step>.ok
set -euo pipefail
cd "$(dirname "$0")"
PY=D:/anaconda/envs/sar/python.exe
export PYTHONUTF8=1
L=../runs/MAIN/logs; mkdir -p $L
run() { local k=$1; shift; [ -f $L/$k.ok ] && return 0; { echo "[$(date -Iseconds)] start $k"; "$@"; echo "[$(date -Iseconds)] done"; } > $L/$k.run.log 2>&1; touch $L/$k.ok; }
run s6   $PY a1_s6_experiments.py --exp E1 E11 E9 E5 E10 E3
run e5r  $PY a1_r2_e5r.py
run s7   $PY a1_s7_stats.py
run r2   $PY a1_r2_stats.py
echo "[$(date -Iseconds)] SITE2 STUDY2 COMPLETE" > $L/SITE2_S2_DONE
