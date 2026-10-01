#!/usr/bin/env bash
# Runs the whole A0 v7 rerun once the HyP3 products are downloaded (PROTOCOL.md sections 3 and 5).
# Each step logs to ../logs/<step>.log; the pipeline stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
PY=D:/anaconda/envs/sar/python.exe
mkdir -p ../logs
until [ -f ../products/download_manifest.json ]; do sleep 60; done
run() { local k=$1; shift; [ -f ../logs/$k.ok ] && return 0; { echo "[$(date -Iseconds)] start $k"; "$@"; echo "[$(date -Iseconds)] done"; } > ../logs/$k.log 2>&1; touch ../logs/$k.ok; }
export PYTHONUTF8=1
run s3  $PY s3_build_cube_v7.py
run s4  $PY s4_run_R1_v7.py
run s5  $PY s5_t1_v7.py
run s7  $PY s7_autumn_eval_v7.py
run s10 $PY s10_cnn_baseline.py
run s11 $PY s11_contrasts.py
run s8  $PY s8_full_pixel_eval.py I8 I28 I8_G9 F12 I28_G29
A0_RUNS=runs_full A0_TABLES=tables_full run s5f $PY s5_t1_v7.py
run s12 $PY s12_bag_check.py
echo "[$(date -Iseconds)] PIPELINE COMPLETE" > ../logs/PIPELINE_DONE
