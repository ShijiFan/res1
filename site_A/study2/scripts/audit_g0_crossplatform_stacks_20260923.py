"""Read-only ASF baseline-stack census of 2025 T37/T88 six-day A/C and C/A pairs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import asf_search as asf

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR"
PAIRS = RUN / "manifests" / "pair_manifest_2025.csv"
OUT = RUN / "reports" / "G0_6DAY_AC_BASELINE_STACK_AUDIT.json"


def main() -> None:
    with PAIRS.open(encoding="utf-8", newline="") as stream:
        pairs = [
            row for row in csv.DictReader(stream)
            if row["track"] in {"37", "88"} and row["lag_days"] == "6"
            and {row["platform_1"], row["platform_2"]} == {"Sentinel-1A", "Sentinel-1C"}
            and row["eligibility_status"] == "CATALOG_CANDIDATE"
        ]
    records = []
    for pair in pairs:
        start = date.fromisoformat(pair["date_1"]) - timedelta(days=18)
        end = date.fromisoformat(pair["date_2"]) + timedelta(days=18)
        result = {"pair_id": pair["pair_id"], "track": pair["track"], "date_1": pair["date_1"], "date_2": pair["date_2"],
                  "scene_id_1": pair["scene_id_1"], "scene_id_2": pair["scene_id_2"],
                  "selected_frame_overlap_fraction": pair["coverage_intersection_fraction"],
                  "query_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            stack = asf.stack_from_id(
                pair["scene_id_1"] + "-SLC",
                opts=asf.ASFSearchOptions(start=start.isoformat(), end=end.isoformat(), relativeOrbit=int(pair["track"])),
            )
            targets = [product for product in stack if product.properties.get("sceneName") == pair["scene_id_2"] and product.properties.get("fileID", "").endswith("-SLC")]
            result.update({"stack_size": len(stack), "exact_selected_secondary_in_stack": len(targets) == 1,
                           "perpendicular_baseline_m": targets[0].properties.get("perpendicularBaseline") if len(targets) == 1 else None,
                           "status": "STACK_MATCH_BURST_PENDING" if len(targets) == 1 else "NO_STACK_MATCH"})
        except Exception as exc:
            result.update({"status": "LOOKUP_ERROR", "error_type": type(exc).__name__, "error": str(exc)[:300]})
        records.append(result)
        print(f"T{pair['track']} {pair['date_1']} {result['status']}", flush=True)
    counts = dict(Counter(f"T{record['track']}_{record['status']}" for record in records))
    OUT.write_text(json.dumps({"source": "ASF CMR via asf_search.stack_from_id", "queried_pairs": len(pairs), "counts": counts,
                               "records": records, "submitted_jobs": 0, "common_burst_count_verified": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"Report: {OUT}")


if __name__ == "__main__":
    main()
