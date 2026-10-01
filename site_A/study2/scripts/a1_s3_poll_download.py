"""Step 3: poll already-submitted HyP3 jobs and download finished products. Never submits jobs.

Input: RUN/submit/SUBMITTED_JOBS.csv with at least a `job_id` column (written when the user-approved
submission was made). Resumable: an existing zip with the expected size is skipped.
- FAILED jobs are only recorded (RUN/submit/FAILED_JOBS.csv); resubmission is a user decision.
- Stops with a clear message if free disk space drops below --min-free-gb.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import hyp3_sdk

from a1_common import D, ensure_dirs, get_logger, now, update_status, write_json

log = get_logger("s3_poll_download")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-min", type=float, default=10)
    ap.add_argument("--max-hours", type=float, default=72)
    ap.add_argument("--min-free-gb", type=float, default=40)
    ap.add_argument("--also-check-dir", action="append", default=[],
                    help="folder where already-downloaded zips were moved; files found there are not downloaded again")
    a = ap.parse_args()
    ensure_dirs()
    sub = D["submit"] / "SUBMITTED_JOBS.csv"
    if not sub.exists():
        update_status("s3_poll_download", "WAITING_FOR_SUBMISSION")
        raise SystemExit("STOP: SUBMITTED_JOBS.csv not found; submission has not been made yet")
    ids = [r["job_id"] for r in csv.DictReader(open(sub, encoding="utf-8", newline="")) if r.get("job_id")]
    hyp3 = hyp3_sdk.HyP3()
    t_end = time.time() + a.max_hours * 3600

    while True:
        try:
            job_map = {j.job_id: j for j in hyp3.find_jobs()}
            jobs = [job_map.get(i) or hyp3.get_job_by_id(i) for i in ids]
        except Exception:
            jobs = [hyp3.get_job_by_id(i) for i in ids]
        counts = {}
        for j in jobs:
            counts[j.status_code] = counts.get(j.status_code, 0) + 1
        log.info("status: %s", counts)

        for j in jobs:
            if j.status_code != "SUCCEEDED":
                continue
            for fd in j.files or []:
                target = D["downloads"] / fd["filename"]
                if target.exists() and target.stat().st_size == fd.get("size", target.stat().st_size):
                    continue
                if any((Path(x) / fd["filename"]).exists() for x in a.also_check_dir):
                    continue
                free_gb = shutil.disk_usage(D["downloads"]).free / 1e9
                if free_gb < a.min_free_gb:
                    update_status("s3_poll_download", "BLOCKED_DISK", free_gb=round(free_gb, 1))
                    raise SystemExit(f"STOP: only {free_gb:.1f} GB free on the download drive")
                log.info("download %s", fd["filename"])
                j.download_files(D["downloads"])

        index = [{"job_id": j.job_id, "name": j.name, "job_type": j.job_type, "status": j.status_code,
                  "files": ";".join(f["filename"] for f in (j.files or [])),
                  "granules": ";".join((j.job_parameters or {}).get("granules", []))} for j in jobs]
        write_json(D["submit"] / "JOB_STATUS.json", {"utc": now(), "counts": counts, "jobs": index})
        failed = [r for r in index if r["status"] == "FAILED"]
        with open(D["submit"] / "FAILED_JOBS.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0].keys()) if index else ["job_id"])
            w.writeheader()
            w.writerows(failed)

        if not any(j.status_code in ("PENDING", "RUNNING") for j in jobs):
            break
        if time.time() > t_end:
            update_status("s3_poll_download", "TIMEOUT", counts=counts)
            raise SystemExit("STOP: timeout; re-run this script to continue")
        time.sleep(a.interval_min * 60)

    update_status("s3_poll_download", "DONE", counts=counts, failed=len(failed))
    log.info("done: %s", counts)


if __name__ == "__main__":
    main()
