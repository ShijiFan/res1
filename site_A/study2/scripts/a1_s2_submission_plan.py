"""Step 2 (offline): assemble the submission plan for the A1 main batches + closure subset B2.

This script does NOT submit anything and does not spend credits. It writes:
  RUN/submit/SUBMISSION_PLAN.json   - every payload that would be submitted, with total credits
  RUN/submit/SUBMISSION_PLAN.csv    - one row per job (name, type, batch, pair, granules)
Submission itself is a separate, user-approved action (see AGENT_TASK_A1_MAIN_RUN.md, section 2).
"""
from __future__ import annotations

import csv

from a1_common import CREDIT_INSAR, D, MAIN_PREFLIGHT, ensure_dirs, get_logger, now, read_json, update_status, write_json

log = get_logger("s2_submission_plan")


def main() -> None:
    ensure_dirs()
    main_pf = read_json(MAIN_PREFLIGHT)
    b2 = read_json(D["submit"] / "CLOSURE_SUBSET_B2_PREFLIGHT.json")
    if main_pf is None or b2 is None:
        raise SystemExit("STOP: preflight files missing; run a1_s1_subset_b2.py first")
    jobs = [j for j in main_pf["jobs"] if j["batch"] in ("main2025", "crossyear2024")] + b2["jobs"]
    names = [j["job_name"] for j in jobs]
    if len(names) != len(set(names)):
        raise SystemExit("STOP: duplicate job names")
    for j in jobs:
        jp = j["sdk_payload"]["job_parameters"]
        if j["job_type"] == "INSAR_GAMMA":
            assert jp["include_wrapped_phase"] and jp["phase_filter_parameter"] == 0.6 and jp["looks"] == "10x2"
        else:
            assert jp["resolution"] == 20 and jp["scale"] == "power" and jp["radiometry"] == "gamma0"
    credits = CREDIT_INSAR * len(jobs)
    plan = {"created_utc": now(), "n_jobs": len(jobs), "credits": credits,
            "by_batch": {b: sum(1 for j in jobs if j["batch"] == b) for b in sorted({j["batch"] for j in jobs})},
            "payloads": [j["sdk_payload"] for j in jobs], "status": "PLAN_ONLY_NOT_SUBMITTED"}
    write_json(D["submit"] / "SUBMISSION_PLAN.json", plan)
    with open(D["submit"] / "SUBMISSION_PLAN.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "job_type", "batch", "source_pair_id", "granules"])
        for j in jobs:
            w.writerow([j["job_name"], j["job_type"], j["batch"], j.get("source_pair_id", ""),
                        ";".join(j["sdk_payload"]["job_parameters"]["granules"])])
    log.info("plan: %d jobs, %d credits, %s", len(jobs), credits, plan["by_batch"])
    update_status("s2_submission_plan", "DONE", n_jobs=len(jobs), credits=credits)


if __name__ == "__main__":
    main()
