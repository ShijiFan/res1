"""Independent assertions on MAIN_SDK_PREFLIGHT.json. Exit code 1 = do not submit.

Usage:
  & 'D:\\anaconda\\envs\\sar\\python.exe' scripts\\verify_main_params_20260923.py `
      --preflight runs\\20260923_MAIN_PARAMS_WRAPPED\\reports\\MAIN_SDK_PREFLIGHT.json `
      --costs runs\\20260923_MAIN_PARAMS_WRAPPED\\reports\\costs_snapshot.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECT_INSAR = {"looks": "10x2", "include_wrapped_phase": True, "phase_filter_parameter": 0.6,
                "apply_water_mask": True, "include_inc_map": True}
EXPECT_RTC = {"radiometry": "gamma0", "resolution": 20, "scale": "power"}
# Must match the submitted P0 pilot and the 2024 A0 products (Phase filter parameter: 0.6)
P0_FILTER = 0.6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--costs", type=Path, help="raw hyp3.costs() JSON saved by the agent")
    a = ap.parse_args()
    p = a.preflight if a.preflight.is_absolute() else ROOT / a.preflight
    d = json.loads(p.read_text(encoding="utf-8"))
    fails, names, pairs = [], set(), set()

    for j in d["jobs"]:
        pl, jp = j["sdk_payload"], j["sdk_payload"]["job_parameters"]
        if j["status"] != "PREPARED_OFFLINE_NOT_SUBMITTED":
            fails.append(f"{j['job_name']}: status {j['status']}")
        if pl["name"] in names:
            fails.append(f"duplicate name {pl['name']}")
        names.add(pl["name"])
        key = (pl["job_type"], tuple(jp["granules"]))
        if key in pairs:
            fails.append(f"duplicate granules {key}")
        pairs.add(key)
        exp = EXPECT_INSAR if pl["job_type"] == "INSAR_GAMMA" else EXPECT_RTC
        for k, v in exp.items():
            if jp.get(k) != v:
                fails.append(f"{pl['name']}: {k}={jp.get(k)!r}, expected {v!r}")
        if pl["job_type"] == "INSAR_GAMMA":
            g1, g2 = jp["granules"]
            if g1[17:25] >= g2[17:25]:
                fails.append(f"{pl['name']}: reference date not earlier than secondary ({g1[17:25]} vs {g2[17:25]})")
    if d.get("submitted_jobs", 0) != 0:
        fails.append("submitted_jobs != 0")
    if EXPECT_INSAR["phase_filter_parameter"] != P0_FILTER:
        fails.append("phase filter differs from P0/A0; coherence would not be comparable")

    n_insar = sum(1 for j in d["jobs"] if j["job_type"] == "INSAR_GAMMA")
    n_rtc = sum(1 for j in d["jobs"] if j["job_type"] == "RTC_GAMMA")
    cost_note = "costs not checked"
    if a.costs:
        cp = a.costs if a.costs.is_absolute() else ROOT / a.costs
        raw = cp.read_text(encoding="utf-8")
        if "include_wrapped_phase" in raw or "include_inc_map" in raw:
            fails.append("cost table mentions include_wrapped_phase/include_inc_map: price may depend on them; re-price by hand")
        cost_note = f"cost table saved at {cp}; confirm INSAR_GAMMA price depends only on looks"

    report = {"preflight": str(p), "n_insar": n_insar, "n_rtc": n_rtc, "n_fail": len(fails),
              "fails": fails[:200], "cost_check": cost_note,
              "ready_for_user_review": not fails, "ready_for_submission": False,
              "why_not_submission": "Submission needs burst checks, P0 product QC and explicit user approval."}
    out = p.parent / "MAIN_PARAMS_VERIFY.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
