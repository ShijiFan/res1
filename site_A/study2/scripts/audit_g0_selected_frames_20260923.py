"""Read-only independent audit of G0 pair endpoints and the six pilot jobs.

The outputs are new audit artifacts; this script never edits the G0 manifests or
submits HyP3 jobs. Run with D:/anaconda/envs/sar/python.exe.
"""

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
from shapely.wkt import loads


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_0140_G0R"
MANIFESTS = RUN / "manifests"
AOI = box(6.00, 51.98, 6.54, 52.38)
PROJECT = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform
AOI_M = transform(PROJECT, AOI)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


scenes = {
    row["scene_id"]: row
    for year in (2024, 2025)
    for row in read_csv(MANIFESTS / f"scene_manifest_twente_{year}_SLC.csv")
}
pairs = [
    row
    for year in (2024, 2025)
    for row in read_csv(MANIFESTS / f"pair_manifest_{year}.csv")
]
jobs = read_csv(MANIFESTS / "job_manifest_dryrun.csv")
pair_ids = {row["pair_id"] for row in pairs}


def coverage(scene_id):
    footprint = transform(PROJECT, loads(scenes[scene_id]["footprint_wkt"]))
    return footprint.intersection(AOI_M).area / AOI_M.area


def footprint(scene_id):
    return transform(PROJECT, loads(scenes[scene_id]["footprint_wkt"]))


undercovered = []
for pair in pairs:
    if pair["eligibility_status"] != "CATALOG_CANDIDATE":
        continue
    min_scene_coverage = min(coverage(pair["scene_id_1"]), coverage(pair["scene_id_2"]))
    if min_scene_coverage < 0.85:
        undercovered.append(
            {
                "pair_id": pair["pair_id"],
                "year": int(pair["year"]),
                "track": int(pair["track"]),
                "minimum_selected_scene_coverage": round(min_scene_coverage, 6),
                "reported_union_pair_coverage": pair["coverage_intersection_fraction"],
            }
        )

pilot = [job for job in jobs if job["batch"] == "pilot"]
pilot_membership = [
    {
        "job_name": job["job_name"],
        "input_exactly_resolves": job["inputs"] in (scenes if job["job_type"] == "RTC_GAMMA" else pair_ids),
        "input": job["inputs"],
    }
    for job in pilot
]


def greedy_disjoint(year, track):
    candidates = sorted(
        (
            pair
            for pair in pairs
            if int(pair["year"]) == year
            and int(pair["track"]) == track
            and pair["role"] == "MAIN_12D"
            and pair["platform_1"] == "Sentinel-1A"
            and pair["platform_2"] == "Sentinel-1A"
            and pair["eligibility_status"] == "CATALOG_CANDIDATE"
            and min(coverage(pair["scene_id_1"]), coverage(pair["scene_id_2"])) >= 0.85
        ),
        key=lambda pair: pair["date_1"],
    )
    chosen = []
    for pair in candidates:
        if not chosen or pair["date_1"] > chosen[-1]["date_2"]:
            chosen.append(pair)
    return {"catalog_pairs": len(candidates), "greedy_disjoint_pairs": len(chosen)}


# These are candidate exact ASF scene IDs only. Cross-mission common bursts and
# production-quality coherence have not been established.
candidate_jobs = [
    ("PILOT_RTC_T37_CANDIDATE", "RTC_GAMMA", "S1A_IW_SLC__1SDV_20250507T055030_20250507T055057_059084_07547C_C15D", ""),
    ("PILOT_RTC_T88_CANDIDATE", "RTC_GAMMA", "S1A_IW_SLC__1SDV_20250510T172522_20250510T172549_059135_07564A_E5E1", ""),
    ("PILOT_INSAR_T37_12D_CANDIDATE", "INSAR_GAMMA", "S1A_IW_SLC__1SDV_20250507T055030_20250507T055057_059084_07547C_C15D", "S1A_IW_SLC__1SDV_20250519T055030_20250519T055057_059259_075A8A_7A6E"),
    ("PILOT_INSAR_T88_12D_CANDIDATE", "INSAR_GAMMA", "S1A_IW_SLC__1SDV_20250510T172522_20250510T172549_059135_07564A_E5E1", "S1A_IW_SLC__1SDV_20250522T172521_20250522T172548_059310_075C59_77A8"),
    ("PILOT_INSAR_T37_6D_CANDIDATE", "INSAR_GAMMA", "S1A_IW_SLC__1SDV_20250519T055030_20250519T055057_059259_075A8A_7A6E", "S1C_IW_SLC__1SDV_20250525T054859_20250525T054926_002483_0052E3_2948"),
    ("PILOT_INSAR_T88_6D_CANDIDATE", "INSAR_GAMMA", "S1A_IW_SLC__1SDV_20250510T172522_20250510T172549_059135_07564A_E5E1", "S1C_IW_SLC__1SDV_20250516T172358_20250516T172425_002359_004F85_2670"),
]

candidate_rows = []
for name, job_type, scene_1, scene_2 in candidate_jobs:
    assert scene_1 in scenes and (not scene_2 or scene_2 in scenes)
    endpoints = [scenes[scene_1]] + ([scenes[scene_2]] if scene_2 else [])
    assert all(row["relative_orbit"] == endpoints[0]["relative_orbit"] for row in endpoints)
    assert all(row["direction"] == endpoints[0]["direction"] for row in endpoints)
    assert all("VV" in row["polarization"] for row in endpoints)
    assert min(coverage(row["scene_id"]) for row in endpoints) >= 0.85
    lag_days = ""
    common_footprint_fraction = ""
    if scene_2:
        start = date.fromisoformat(endpoints[0]["acquisition_start_utc"][:10])
        end = date.fromisoformat(endpoints[1]["acquisition_start_utc"][:10])
        lag_days = (end - start).days
        assert lag_days == (6 if "_6D_" in name else 12)
        common_footprint_fraction = footprint(scene_1).intersection(footprint(scene_2)).intersection(AOI_M).area / AOI_M.area
        assert common_footprint_fraction >= 0.85
    candidate_rows.append(
        {
            "job_name": name,
            "job_type": job_type,
            "scene_id_1": scene_1,
            "scene_id_2": scene_2,
            "track": endpoints[0]["relative_orbit"],
            "direction": endpoints[0]["direction"],
            "min_selected_scene_aoi_coverage": round(min(coverage(row["scene_id"]) for row in endpoints), 6),
            "pair_lag_days": lag_days,
            "selected_scene_intersection_aoi_coverage": round(common_footprint_fraction, 6) if scene_2 else "",
            "catalog_status": "CANDIDATE_ONLY_BURST_AND_PRODUCT_QC_PENDING",
            "submission_status": "NOT_SUBMITTED",
        }
    )

out_csv = ROOT / "reports" / "G0_PILOT_EXACT_SCENE_CANDIDATES_20260923.csv"
with out_csv.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
    writer.writeheader()
    writer.writerows(candidate_rows)

report = {
    "pilot_original_count": len(pilot),
    "pilot_original_exact_resolutions": sum(item["input_exactly_resolves"] for item in pilot_membership),
    "pilot_membership": pilot_membership,
    "candidate_pairs_count": sum(pair["eligibility_status"] == "CATALOG_CANDIDATE" for pair in pairs),
    "candidate_pairs_selected_frame_undercoverage_count": len(undercovered),
    "undercoverage_by_year_track": dict(Counter(f"{item['year']}_T{item['track']}" for item in undercovered)),
    "undercoverage_examples": undercovered[:10],
    "strict_AA_12d_by_year_track": {
        f"{year}_T{track}": greedy_disjoint(year, track)
        for year in (2024, 2025)
        for track in (37, 88)
    },
    "new_exact_scene_candidates": str(out_csv),
    "candidate_status": "NOT_READY_TO_SUBMIT; burst, GRD/RTC alignment and product QC pending",
}
out_json = RUN / "reports" / "G0_SELECTED_FRAME_AUDIT_20260923.json"
out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key not in ("pilot_membership", "undercoverage_examples")}, indent=2))
