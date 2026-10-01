#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submit A1 Main Batch Jobs (120 jobs, 1,800 credits)
=====================================================
Authorized by user: "我已批准按 SUBMISSION_PLAN.json 提交 120 个作业（1,800 credits），提交后生成 SUBMITTED_JOBS.csv"

Submits the 120 pre-registered payloads in chunks of 25 to HyP3:
  - main2025: 50 jobs (34 RTC, 16 InSAR)
  - crossyear2024: 54 jobs (36 RTC, 18 InSAR)
  - closure_subset_b2: 16 jobs (16 InSAR)
Writes:
  - runs/20260924_A1_MAIN/submit/SUBMITTED_JOBS.csv
  - runs/20260924_A1_MAIN/reports/SUBMISSION_RECEIPT.json
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import hyp3_sdk

from a1_common import D, ensure_dirs, get_logger, now, update_status

log = get_logger("submit_a1_main_jobs")

def main() -> None:
    ensure_dirs()
    plan_path = D["submit"] / "SUBMISSION_PLAN.json"
    if not plan_path.exists():
        raise SystemExit(f"STOP: {plan_path} not found")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    payloads = plan["payloads"]
    n_expected = plan["n_jobs"]
    expected_cost = plan["credits"]

    if len(payloads) != 120 or n_expected != 120 or expected_cost != 1800:
        raise SystemExit(f"STOP: expected exactly 120 payloads and 1800 credits; found {len(payloads)} and {expected_cost}")

    log.info("Starting HyP3 submission for A1 Main (120 jobs)...")
    hyp3 = hyp3_sdk.HyP3()
    initial_credits = hyp3.check_credits()
    log.info("Initial credit balance: %d", initial_credits)

    if initial_credits < expected_cost:
        raise SystemExit(f"STOP: Insufficient credits: {initial_credits} < {expected_cost}")

    chunk_size = 25
    submitted_records = []

    for i in range(0, len(payloads), chunk_size):
        chunk = payloads[i:i + chunk_size]
        log.info("Submitting chunk %d to %d (%d jobs)...", i + 1, min(i + chunk_size, len(payloads)), len(chunk))
        batch = hyp3.submit_prepared_jobs(chunk)
        for job in batch:
            submitted_records.append({
                "job_id": job.job_id,
                "name": job.name,
                "job_type": job.job_type,
                "status_code": job.status_code,
                "request_time": str(job.request_time),
            })
            log.info("  Submitted: %s -> ID: %s [%s]", job.name, job.job_id, job.status_code)
        time.sleep(1)

    log.info("Successfully submitted all %d jobs to HyP3!", len(submitted_records))

    post_credits = hyp3.check_credits()
    credits_spent = initial_credits - post_credits
    log.info("Post-submission balance: %d (Spent: %d)", post_credits, credits_spent)

    # Write SUBMITTED_JOBS.csv
    csv_path = D["submit"] / "SUBMITTED_JOBS.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "name", "job_type", "status_code", "request_time"])
        w.writeheader()
        w.writerows(submitted_records)
    log.info("Wrote submitted jobs list to: %s", csv_path)

    # Write receipt
    receipt = {
        "submission_utc": now(),
        "hyp3_sdk_version": hyp3_sdk.__version__,
        "job_count": len(submitted_records),
        "initial_credits": initial_credits,
        "post_credits": post_credits,
        "credits_spent": credits_spent,
        "expected_credits": expected_cost,
        "jobs": submitted_records,
        "status": "ALL_120_JOBS_SUBMITTED_SUCCESSFULLY"
    }
    receipt_path = D["reports"] / "SUBMISSION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Receipt written to: %s", receipt_path)

    update_status("submit_jobs", "DONE", n_submitted=len(submitted_records), credits_spent=credits_spent)

if __name__ == "__main__":
    main()
