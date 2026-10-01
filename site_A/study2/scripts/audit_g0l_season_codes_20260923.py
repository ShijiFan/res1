"""Quantify crop labels whose BRP names do not state winter/spring sowing."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_G0L_COMMON_GRID"
MAPPING = ROOT / "runs" / "20260923_0140_G0R" / "qc" / "crop_code_mapping_2024_2025.csv"
AMBIGUOUS = {238, 314, 382, 670, 2652, 6636, 7130}


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def main() -> None:
    code_names = {int(row["gewascode"]): row["dutch_crop_name"] for row in rows(MAPPING)}
    detail = defaultdict(lambda: {"all_parcels": 0, "eligible_ge2": 0})
    sensitivity = {}
    for year in (2024, 2025):
        original = Counter()
        strict = Counter()
        for row in rows(RUN / f"parcel_table_{year}.csv"):
            code = int(row["gewascode"])
            eligible = int(row["eligible_ge2"])
            if code in AMBIGUOUS:
                detail[(year, code)]["all_parcels"] += 1
                detail[(year, code)]["eligible_ge2"] += eligible
            if eligible:
                original[row["class"]] += 1
                strict["Other" if code in AMBIGUOUS else row["class"]] += 1
        sensitivity[year] = {"current_mapping_eligible": dict(original), "strict_season_mapping_eligible": dict(strict)}
    out = RUN / "crop_season_code_sensitivity.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("year", "gewascode", "dutch_name", "current_class", "all_parcels", "eligible_parcels_ge2", "strict_season_class"))
        current = {int(row["gewascode"]): row["assigned_8class"] for row in rows(MAPPING)}
        for (year, code), counts in sorted(detail.items()):
            writer.writerow((year, code, code_names.get(code, ""), current[code], counts["all_parcels"], counts["eligible_ge2"], "Other"))
    report = {
        "reason": "The BRP names for these codes do not explicitly identify winter or spring sowing. The strict alternative sends them to Other until an official code-level season source is verified.",
        "ambiguous_codes": sorted(AMBIGUOUS), "sensitivity": sensitivity,
        "current_mapping_modified": False,
        "status": "SEMANTIC_DECISION_PENDING",
    }
    (RUN / "G0L_SEASON_CODE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(sensitivity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
