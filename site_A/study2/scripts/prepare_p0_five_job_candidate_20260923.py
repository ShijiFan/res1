"""Create a reviewable five-job Pilot candidate after T37 6-day stack failure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR"
SOURCE = RUN / "manifests" / "job_manifest_dryrun.csv"
BASELINE = RUN / "reports" / "G0_PILOT_ASF_BASELINE_LOOKUP.json"
OUT = RUN / "reports" / "P0_FIVE_JOB_CANDIDATE.csv"


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        pilot = [row for row in csv.DictReader(stream) if row["batch"] == "pilot"]
    lookup = {row["job_name"]: row for row in json.loads(BASELINE.read_text(encoding="utf-8"))["jobs"]}
    selected = [row for row in pilot if row["job_name"] != "PILOT_INSAR_T37_6D"]
    if len(selected) != 5:
        raise RuntimeError("Five-job candidate selection failed")
    output = []
    for row in selected:
        row = dict(row)
        if row["job_type"] == "INSAR_GAMMA":
            status = lookup[row["job_name"]]["status"]
            if status != "BASELINE_STACK_MATCH_BURST_PENDING":
                raise RuntimeError(f"Retained pair lacks baseline-stack match: {row['job_name']}")
            row["baseline_stack_status"] = status
            row["readiness"] = "BLOCKED_COMMON_BURST_AND_SDK_CHECK"
        else:
            row["baseline_stack_status"] = "NOT_APPLICABLE"
            row["readiness"] = "BLOCKED_SDK_AND_ACCOUNT_CHECK"
        output.append(row)
    with OUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"pilot_jobs": len(output), "estimated_credits": sum(int(row["credit_cost"]) for row in output),
                      "submitted_jobs": sum(row["submission_status"] != "NOT_SUBMITTED" for row in output), "file": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
