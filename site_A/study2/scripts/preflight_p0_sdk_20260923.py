"""Construct five HyP3 SDK payloads offline; never submit them."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import hyp3_sdk

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR"
INPUT = RUN / "reports" / "P0_FIVE_JOB_CANDIDATE.csv"
OUT = RUN / "reports" / "P0_FIVE_JOB_SDK_PREFLIGHT.json"


def main() -> None:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        jobs = list(csv.DictReader(stream))
    if len(jobs) != 5:
        raise RuntimeError("Expected exactly five P0 candidates")
    payloads = []
    for job in jobs:
        if job["submission_status"] != "NOT_SUBMITTED":
            raise RuntimeError("Candidate already has a submission state")
        if job["job_type"] == "RTC_GAMMA":
            payload = hyp3_sdk.HyP3.prepare_rtc_job(job["inputs"], name=job["job_name"], radiometry="gamma0", resolution=20, scale="power")
        elif job["job_type"] == "INSAR_GAMMA":
            first, second = job["inputs"].split(" & ")
            payload = hyp3_sdk.HyP3.prepare_insar_job(first, second, name=job["job_name"], looks="10x2", apply_water_mask=True)
        else:
            raise RuntimeError(f"Unknown job type: {job['job_type']}")
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        payloads.append({"job_name": job["job_name"], "job_type": job["job_type"], "sdk_payload": payload,
                         "payload_sha256": hashlib.sha256(encoded).hexdigest(), "status": "PREPARED_OFFLINE_NOT_SUBMITTED"})
    OUT.write_text(json.dumps({"hyp3_sdk_version": hyp3_sdk.__version__, "jobs": payloads, "submitted_jobs": 0,
                               "note": "SDK payload construction does not verify burst overlap, account balance, or processing success."}, indent=2), encoding="utf-8")
    print(json.dumps({"sdk_version": hyp3_sdk.__version__, "jobs_prepared": len(payloads), "submitted_jobs": 0, "file": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
