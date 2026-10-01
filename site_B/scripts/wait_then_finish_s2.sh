#!/usr/bin/env bash
# Wait for the orphaned a1_s6_experiments.py (site 2) to exit, verify it completed every configuration
# (site-1 counts minus E2w), then resume run_site2_s2.sh (E5r, s7, r2). Stops if counts do not match.
set -euo pipefail
cd "$(dirname "$0")"
REG=../runs/MAIN/experiments/REGISTRY.csv
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*a1_s6_experiments.py*' }) { exit 0 } else { exit 1 }"; do sleep 120; done
declare -A WANT=([E1]=112 [E11]=83 [E9]=32 [E5]=16 [E10]=88 [E3]=624)
ok=1
for e in "${!WANT[@]}"; do
  n=$(cut -d, -f2 $REG | grep -cx "$e" || true)
  echo "$e $n / ${WANT[$e]}"
  [ "$n" -eq "${WANT[$e]}" ] || ok=0
done
if [ $ok -ne 1 ]; then echo "STOP: s6 incomplete; not continuing"; exit 1; fi
touch ../runs/MAIN/logs/s6.ok
bash run_site2_s2.sh
echo "FINISHED"
