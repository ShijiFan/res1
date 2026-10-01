#!/usr/bin/env bash
# Wait for the single remaining E5r process, de-duplicate REGISTRY.csv (a second, duplicate E5r instance
# from a leftover shell ran 13:33-15:05 on 2026-10-01; runs are deterministic, rows keyed by config_id),
# verify 240 unique E5r configs, then run s7 and r2 via run_site2_s2.sh (s6/e5r marked done).
set -euo pipefail
cd "$(dirname "$0")"
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*a1_r2_e5r*' }) { exit 0 } else { exit 1 }"; do sleep 120; done
REG=../runs/MAIN/experiments/REGISTRY.csv
cp $REG $REG.with_duplicates.bak
PYTHONUTF8=1 D:/anaconda/envs/sar/python.exe - <<'PY'
import pandas as pd
p = "../runs/MAIN/experiments/REGISTRY.csv"
r = pd.read_csv(p, dtype=str)
d = r.drop_duplicates("config_id", keep="first")
n5 = (d.exp == "E5r").sum()
print("rows", len(r), "-> unique", len(d), "; E5r unique", n5)
if n5 != 240:
    raise SystemExit(f"STOP: E5r unique configs {n5} != 240")
d.to_csv(p, index=False)
PY
touch ../runs/MAIN/logs/e5r.ok
bash run_site2_s2.sh
echo FINISHED
