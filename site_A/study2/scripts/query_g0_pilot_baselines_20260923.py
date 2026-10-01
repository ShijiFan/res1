"""Read-only ASF CMR baseline-stack lookup for the four Pilot InSAR pairs.

This queries public ASF metadata. It never authenticates to HyP3 or submits a job.
Baseline-stack membership is useful evidence, but it is not a measured common
burst count or successful InSAR processing result.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import asf_search as asf

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR"
JOBS = RUN / "manifests" / "job_manifest_dryrun.csv"
PAIRS = RUN / "manifests" / "pair_manifest_2025.csv"
OUT = RUN / "reports" / "G0_PILOT_ASF_BASELINE_LOOKUP.json"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    pairs = {row["pair_id"]: row for row in read_csv(PAIRS)}
    jobs = [row for row in read_csv(JOBS) if row["batch"] == "pilot" and row["job_type"] == "INSAR_GAMMA"]
    if len(jobs) != 4:
        raise RuntimeError(f"Expected exactly four Pilot InSAR jobs, got {len(jobs)}")
    results = []
    for job in jobs:
        first, second = job["inputs"].split(" & ")
        pair = pairs[job["source_pair_id"]]
        assert (first, second) == (pair["scene_id_1"], pair["scene_id_2"])
        start = date.fromisoformat(pair["date_1"]) - timedelta(days=18)
        end = date.fromisoformat(pair["date_2"]) + timedelta(days=18)
        query = {"start": start.isoformat(), "end": end.isoformat(), "relativeOrbit": int(pair["track"]), "reference_product_list_id": first + "-SLC"}
        record = {"job_name": job["job_name"], "pair_id": pair["pair_id"], "reference_scene_id": first, "secondary_scene_id": second, "query": query, "queried_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            options = asf.ASFSearchOptions(start=query["start"], end=query["end"], relativeOrbit=query["relativeOrbit"])
            stack = asf.stack_from_id(query["reference_product_list_id"], opts=options)
            matches = [product for product in stack if product.properties.get("sceneName") == second and product.properties.get("fileID", "").endswith("-SLC")]
            record.update({
                "stack_size": len(stack), "secondary_match_count": len(matches),
                "perpendicular_baseline_m": matches[0].properties.get("perpendicularBaseline") if len(matches) == 1 else None,
                "temporal_baseline_days": matches[0].properties.get("temporalBaseline") if len(matches) == 1 else None,
                "status": "BASELINE_STACK_MATCH_BURST_PENDING" if len(matches) == 1 else "BASELINE_STACK_NO_UNIQUE_MATCH",
            })
        except Exception as exc:
            record.update({"status": "LOOKUP_ERROR", "error_type": type(exc).__name__, "error": str(exc)[:400]})
        results.append(record)
        print(json.dumps({key: record.get(key) for key in ("job_name", "status", "perpendicular_baseline_m", "temporal_baseline_days")}, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps({"source": "ASF CMR via asf_search.stack_from_id", "jobs": results, "submitted_jobs": 0, "common_burst_count_verified": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {OUT}")


if __name__ == "__main__":
    main()
