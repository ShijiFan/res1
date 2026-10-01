"""Submit the site-2 HyP3 jobs in ../manifests/job_list.csv (PREREG_SITE2.md, user-approved 2026-09-30).

Parameters are the frozen A1 main-batch set, read from
A1_observation_budget_20260923/runs/20260923_MAIN_PARAMS_WRAPPED/reports/MAIN_SDK_PREFLIGHT.json
(frozen_rtc_params / frozen_insar_params), so both sites are processed identically.
Usage: --dry-run (payloads, cost, balance) | --submit
Out:   ../submit/submitted_<stamp>.json (job ids, payloads, balance before/after)
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import hyp3_sdk

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT.parent / "A1_observation_budget_20260923" / "runs" / "20260923_MAIN_PARAMS_WRAPPED" / "reports" / "MAIN_SDK_PREFLIGHT.json"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    rtc_p, ins_p = pre["frozen_rtc_params"], pre["frozen_insar_params"]
    jobs = list(csv.DictReader(open(ROOT / "manifests" / "job_list.csv", encoding="utf-8")))
    payloads = []
    for j in jobs:
        if j["job_type"] == "RTC_GAMMA":
            payloads.append(hyp3_sdk.HyP3.prepare_rtc_job(j["inputs"], name=j["name"], **rtc_p))
        else:
            g1, g2 = j["inputs"].split("&")
            payloads.append(hyp3_sdk.HyP3.prepare_insar_job(g1, g2, name=j["name"], **ins_p))
    hyp3 = hyp3_sdk.HyP3()
    costs = hyp3.costs()
    cost = sum(costs["RTC_GAMMA"]["cost_table"][str(rtc_p["resolution"])] if p["job_type"] == "RTC_GAMMA"
               else costs["INSAR_GAMMA"]["cost_table"][ins_p["looks"]] for p in payloads)
    bal = hyp3.check_credits()
    print(f"jobs {len(payloads)} (RTC {sum(p['job_type'] == 'RTC_GAMMA' for p in payloads)}, "
          f"INSAR {sum(p['job_type'] == 'INSAR_GAMMA' for p in payloads)}); cost {cost}; balance {bal}")
    print(json.dumps(payloads[0]), "\n", json.dumps(payloads[-1]))
    if a.dry_run:
        return
    if cost > bal:
        raise SystemExit("STOP: insufficient credits")
    batch = hyp3.submit_prepared_jobs(payloads)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ROOT / "submit"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"submitted_at": stamp, "hyp3_sdk": hyp3_sdk.__version__, "rtc_params": rtc_p, "insar_params": ins_p,
           "cost": cost, "balance_before": bal, "balance_after": hyp3.check_credits(),
           "jobs": [j.to_dict() for j in batch]}
    (out / f"submitted_{stamp}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(f"submitted {len(batch)} jobs; balance after {rec['balance_after']}")


if __name__ == "__main__":
    main()
