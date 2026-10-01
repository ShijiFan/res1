"""Build HyP3 payloads for the A1 main batches OFFLINE, with wrapped phase enabled. Never submits.

Reads the repaired dry-run job manifest, keeps the requested batches, and constructs every payload
through hyp3_sdk.HyP3.prepare_* so that parameter names are validated by the installed SDK.

Frozen InSAR parameters (see MAIN_PARAMS_DECISION in the task file):
  looks=10x2, include_wrapped_phase=True, phase_filter_parameter=0.6, apply_water_mask=True, include_inc_map=True
Frozen RTC parameters (identical to the submitted P0 pilot):
  radiometry=gamma0, resolution=20, scale=power

Usage (PowerShell):
  & 'D:\\anaconda\\envs\\sar\\python.exe' scripts\\preflight_main_sdk_wrapped_20260923.py `
      --batches main2025 crossyear2024 `
      --out-dir runs\\20260923_MAIN_PARAMS_WRAPPED
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import hyp3_sdk

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR" / "manifests" / "job_manifest_dryrun.csv"

INSAR_PARAMS = dict(looks="10x2", include_wrapped_phase=True, phase_filter_parameter=0.6,
                    apply_water_mask=True, include_inc_map=True)
RTC_PARAMS = dict(radiometry="gamma0", resolution=20, scale="power")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sdk_signature() -> dict:
    sig_insar = inspect.signature(hyp3_sdk.HyP3.prepare_insar_job).parameters
    sig_rtc = inspect.signature(hyp3_sdk.HyP3.prepare_rtc_job).parameters
    missing = [k for k in INSAR_PARAMS if k not in sig_insar] + [k for k in RTC_PARAMS if k not in sig_rtc]
    if missing:
        raise SystemExit(f"STOP: installed hyp3_sdk {hyp3_sdk.__version__} lacks parameters {missing}. Do not guess names.")
    return {"hyp3_sdk_version": hyp3_sdk.__version__,
            "prepare_insar_job": str(inspect.signature(hyp3_sdk.HyP3.prepare_insar_job)),
            "prepare_rtc_job": str(inspect.signature(hyp3_sdk.HyP3.prepare_rtc_job))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--batches", nargs="+", default=["main2025", "crossyear2024"])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    out_dir = a.out_dir if a.out_dir.is_absolute() else ROOT / a.out_dir
    reports = out_dir / "reports"
    if (reports / "MAIN_SDK_PREFLIGHT.json").exists():
        raise SystemExit(f"STOP: {reports / 'MAIN_SDK_PREFLIGHT.json'} exists. Use a new --out-dir; never overwrite.")
    reports.mkdir(parents=True, exist_ok=True)

    sig = check_sdk_signature()
    with a.manifest.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["batch"] in set(a.batches)]
    if not rows:
        raise SystemExit(f"STOP: no rows for batches {a.batches}")

    jobs, seen_names, seen_inputs = [], set(), set()
    for r in rows:
        if r["submission_status"] != "NOT_SUBMITTED":
            raise SystemExit(f"STOP: {r['job_name']} has submission_status={r['submission_status']}")
        if r["job_name"] in seen_names or (r["job_type"], r["inputs"]) in seen_inputs:
            raise SystemExit(f"STOP: duplicate job {r['job_name']}")
        seen_names.add(r["job_name"]); seen_inputs.add((r["job_type"], r["inputs"]))
        if r["job_type"] == "INSAR_GAMMA":
            g1, g2 = [s.strip() for s in r["inputs"].split("&")]
            payload = hyp3_sdk.HyP3.prepare_insar_job(g1, g2, name=r["job_name"], **INSAR_PARAMS)
        elif r["job_type"] == "RTC_GAMMA":
            payload = hyp3_sdk.HyP3.prepare_rtc_job(r["inputs"].strip(), name=r["job_name"], **RTC_PARAMS)
        else:
            raise SystemExit(f"STOP: unknown job type {r['job_type']}")
        enc = json.dumps(payload, sort_keys=True).encode()
        jobs.append({"job_name": r["job_name"], "batch": r["batch"], "job_type": r["job_type"],
                     "source_pair_id": r.get("source_pair_id", ""), "sdk_payload": payload,
                     "payload_sha256": hashlib.sha256(enc).hexdigest(), "status": "PREPARED_OFFLINE_NOT_SUBMITTED"})

    counts = {}
    for j in jobs:
        k = f"{j['batch']}/{j['job_type']}"
        counts[k] = counts.get(k, 0) + 1
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source_manifest": str(a.manifest), "source_manifest_sha256": sha256_file(a.manifest),
           "batches": a.batches, "sdk": sig, "frozen_insar_params": INSAR_PARAMS, "frozen_rtc_params": RTC_PARAMS,
           "counts": counts, "submitted_jobs": 0, "jobs": jobs,
           "note": "Offline payloads only. Does not verify bursts, credits, or processing success."}
    (reports / "MAIN_SDK_PREFLIGHT.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (reports / "sdk_signature.txt").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(json.dumps({"jobs_prepared": len(jobs), "counts": counts, "submitted_jobs": 0,
                      "file": str(reports / "MAIN_SDK_PREFLIGHT.json")}, indent=2))


if __name__ == "__main__":
    main()
