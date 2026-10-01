"""Read-only ASF baseline-stack census of every selected main M=9 pair."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import asf_search as asf

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR"
OUT = RUN / "reports" / "G0_MAIN_M9_BASELINE_STACK_AUDIT.json"


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    budgets = [row for row in read_csv(RUN / "manifests" / "budget_design.csv") if row["M"] == "9"]
    if len(budgets) != 4:
        raise RuntimeError(f"Expected four M9 budget rows, got {len(budgets)}")
    pairs = {row["pair_id"]: row for year in (2024, 2025) for row in read_csv(RUN / "manifests" / f"pair_manifest_{year}.csv")}
    selected = sorted({pid for row in budgets for pid in row["pair_ids"].split(";")})
    records = []
    for pid in selected:
        pair = pairs[pid]
        start = date.fromisoformat(pair["date_1"]) - timedelta(days=18)
        end = date.fromisoformat(pair["date_2"]) + timedelta(days=18)
        result = {"pair_id": pid, "year": pair["year"], "track": pair["track"], "date_1": pair["date_1"], "date_2": pair["date_2"],
                  "scene_id_1": pair["scene_id_1"], "scene_id_2": pair["scene_id_2"],
                  "selected_frame_overlap_fraction": pair["coverage_intersection_fraction"],
                  "query_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            stack = asf.stack_from_id(pair["scene_id_1"] + "-SLC", opts=asf.ASFSearchOptions(
                start=start.isoformat(), end=end.isoformat(), relativeOrbit=int(pair["track"])))
            matches = [product for product in stack if product.properties.get("sceneName") == pair["scene_id_2"] and product.properties.get("fileID", "").endswith("-SLC")]
            result.update({"stack_size": len(stack), "exact_selected_secondary_in_stack": len(matches) == 1,
                           "perpendicular_baseline_m": matches[0].properties.get("perpendicularBaseline") if len(matches) == 1 else None,
                           "status": "STACK_MATCH_BURST_PENDING" if len(matches) == 1 else "NO_STACK_MATCH"})
        except Exception as exc:
            result.update({"status": "LOOKUP_ERROR", "error_type": type(exc).__name__, "error": str(exc)[:300]})
        records.append(result)
        print(f"{pair['year']} T{pair['track']} {pair['date_1']} {result['status']}", flush=True)
    counts = dict(Counter(f"{record['year']}_T{record['track']}_{record['status']}" for record in records))
    OUT.write_text(json.dumps({"source": "ASF CMR via asf_search.stack_from_id", "queried_pairs": len(selected), "counts": counts,
                               "records": records, "submitted_jobs": 0, "common_burst_count_verified": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"Report: {OUT}")


if __name__ == "__main__":
    main()
