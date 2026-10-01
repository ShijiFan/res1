#!/usr/bin/env bash
# Run the frozen TempCNN protocol on both sites (resumable; each config skipped if already in REGISTRY.csv).
set -euo pipefail
cd "$(dirname "$0")"
PY=D:/anaconda/envs/yolo11marin/python.exe
export PYTHONUTF8=1
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*tdl_run.py*' }) { exit 0 } else { exit 1 }"; do sleep 30; done
for s in 1 2; do $PY -u tdl_run.py --site $s > site${s}_run.log 2>&1; echo "site $s done $(date -Iseconds)"; done
echo ALL_DONE > ALL_DONE
