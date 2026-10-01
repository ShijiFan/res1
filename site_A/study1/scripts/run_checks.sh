#!/usr/bin/env bash
# PROTOCOL.md 5.3 / 5.4: CNN checks C1-C3, their contrasts, then the exploratory BAG sensitivity.
set -euo pipefail
cd "$(dirname "$0")"
PY=D:/anaconda/envs/sar/python.exe
export PYTHONUTF8=1
run() { local k=$1; shift; [ -f ../logs/$k.ok ] && return 0; { echo "[$(date -Iseconds)] start $k"; "$@"; echo "[$(date -Iseconds)] done"; } > ../logs/$k.log 2>&1; touch ../logs/$k.ok; }
run c1_cv    $PY s13_cnn_checks.py --stage cv
run c1_final $PY s13_cnn_checks.py --stage final
run c3_patch $PY s13_cnn_checks.py --stage patch
run c_contr  $PY s14_cnn_checks_contrasts.py
run bag_sens $PY s15_bag_sensitivity.py
echo "[$(date -Iseconds)] CHECKS COMPLETE" > ../logs/CHECKS_DONE
