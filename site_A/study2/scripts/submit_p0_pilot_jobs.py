"""Submit the 5 P0 Pilot jobs to HyP3 upon explicit user authorization.

Rules:
1. Only submits the 5 approved jobs from P0_FIVE_JOB_SDK_PREFLIGHT.json.
2. Checks available credit balance before and after submission.
3. Records all returned job IDs, submission UTC timestamps, and status codes.
4. Outputs traceable receipt to runs/20260923_P0_PILOT/reports/P0_SUBMISSION_RECEIPT.json.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import hyp3_sdk

ROOT = Path("E:/research/SAR/A1_observation_budget_20260923")
REPAIR_RUN = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR"
PREFLIGHT_JSON = REPAIR_RUN / "reports" / "P0_FIVE_JOB_SDK_PREFLIGHT.json"
CANDIDATE_CSV = REPAIR_RUN / "reports" / "P0_FIVE_JOB_CANDIDATE.csv"

PILOT_RUN = ROOT / "runs" / "20260923_P0_PILOT"


def main() -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting P0 Pilot submission...")
    
    # 1. Initialize output directories
    for sub in ("manifests", "reports", "downloads"):
        (PILOT_RUN / sub).mkdir(parents=True, exist_ok=True)
        
    # 2. Check credentials and balance
    hyp3 = hyp3_sdk.HyP3()
    initial_credits = hyp3.check_credits()
    print(f"Initial available credits: {initial_credits}")
    
    expected_cost = 75
    required_with_buffer = expected_cost * 1.3
    if initial_credits < required_with_buffer:
        raise RuntimeError(f"Insufficient credits: {initial_credits} < {required_with_buffer}")
        
    # 3. Load preflight payloads
    preflight_data = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    jobs = preflight_data["jobs"]
    if len(jobs) != 5:
        raise RuntimeError(f"Expected exactly 5 jobs in preflight, found {len(jobs)}")
        
    payloads = [item["sdk_payload"] for item in jobs]
    print(f"Prepared {len(payloads)} payloads for submission:")
    for p in payloads:
        print(f"  - {p['name']} ({p['job_type']})")
        
    # 4. Submit jobs
    print("\nSubmitting batch to HyP3...")
    submitted_batch = hyp3.submit_prepared_jobs(payloads)
    submission_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    # 5. Extract details
    submitted_records = []
    for job in submitted_batch:
        rec = {
            "job_id": job.job_id,
            "name": job.name,
            "job_type": job.job_type,
            "status_code": job.status_code,
            "request_time": str(job.request_time),
            "files": getattr(job, "files", None),
        }
        submitted_records.append(rec)
        print(f"Submitted: {job.name} -> ID: {job.job_id} [{job.status_code}]")
        
    # 6. Check updated balance
    post_credits = hyp3.check_credits()
    credits_spent = initial_credits - post_credits
    print(f"\nPost-submission credits: {post_credits} (Deducted: {credits_spent})")
    
    # 7. Write receipt
    receipt = {
        "submission_utc": submission_utc,
        "hyp3_sdk_version": hyp3_sdk.__version__,
        "job_count": len(submitted_records),
        "initial_credits": initial_credits,
        "post_credits": post_credits,
        "credits_spent": credits_spent,
        "jobs": submitted_records,
        "status": "ALL_PILOT_JOBS_SUBMITTED_SUCCESSFULLY",
    }
    receipt_path = PILOT_RUN / "reports" / "P0_SUBMISSION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Receipt written to: {receipt_path}")
    
    # 8. Write updated job manifest
    job_map = {r["name"]: r for r in submitted_records}
    manifest_rows = []
    with CANDIDATE_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            info = job_map.get(row["job_name"])
            if info:
                row["job_id"] = info["job_id"]
                row["submission_status"] = "SUBMITTED"
                row["status_code"] = info["status_code"]
                row["submitted_utc"] = submission_utc
            manifest_rows.append(row)
            
    manifest_path = PILOT_RUN / "manifests" / "p0_job_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
