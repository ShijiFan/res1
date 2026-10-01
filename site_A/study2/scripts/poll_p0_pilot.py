"""Poll the status of the 5 submitted P0 Pilot jobs.

Usage:
    python poll_p0_pilot.py [--download]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import hyp3_sdk

ROOT = Path("E:/research/SAR/A1_observation_budget_20260923")
PILOT_RUN = ROOT / "runs" / "20260923_P0_PILOT"
RECEIPT_JSON = PILOT_RUN / "reports" / "P0_SUBMISSION_RECEIPT.json"
STATUS_JSON = PILOT_RUN / "reports" / "P0_CURRENT_STATUS.json"
DOWNLOAD_DIR = PILOT_RUN / "downloads"


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll P0 Pilot jobs")
    parser.add_argument("--download", action="store_true", help="Download succeeded jobs")
    args = parser.parse_args()

    if not RECEIPT_JSON.exists():
        raise FileNotFoundError(f"Receipt not found: {RECEIPT_JSON}")

    receipt = json.loads(RECEIPT_JSON.read_text(encoding="utf-8"))
    job_ids = [j["job_id"] for j in receipt["jobs"]]

    hyp3 = hyp3_sdk.HyP3()
    print(f"[{datetime.now(timezone.utc).isoformat()}] Polling {len(job_ids)} P0 Pilot jobs...")

    statuses = []
    all_succeeded = True
    any_failed = False

    for item in receipt["jobs"]:
        jid = item["job_id"]
        job = hyp3.get_job_by_id(jid)
        status_rec = {
            "job_id": jid,
            "name": item["name"],
            "job_type": item["job_type"],
            "status_code": job.status_code,
            "request_time": str(job.request_time),
            "expiration_time": str(getattr(job, "expiration_time", None)),
            "files": getattr(job, "files", None),
        }
        statuses.append(status_rec)
        print(f"  {item['name']} ({jid}): {job.status_code}")

        if job.status_code != "SUCCEEDED":
            all_succeeded = False
        if job.status_code == "FAILED":
            any_failed = True

        if args.download and job.status_code == "SUCCEEDED":
            print(f"  Downloading {item['name']}...")
            job.download_files(DOWNLOAD_DIR)

    report = {
        "polled_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_jobs": len(statuses),
        "status_counts": {
            code: sum(1 for s in statuses if s["status_code"] == code)
            for code in set(s["status_code"] for s in statuses)
        },
        "all_succeeded": all_succeeded,
        "any_failed": any_failed,
        "jobs": statuses,
    }
    STATUS_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStatus summary: {report['status_counts']}")
    print(f"Status written to: {STATUS_JSON}")


if __name__ == "__main__":
    main()
