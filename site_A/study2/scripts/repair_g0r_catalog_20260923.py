"""Rebuild G0-R from archived ASF catalogs without network calls or submissions.

The original run used the first SLC frame of each date even when only another
frame covered the AOI. This writes a separate run and leaves the original intact.
Selected pairs remain CATALOG_CANDIDATE until burst/product QC is complete.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
from shapely.wkt import loads


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "20260923_0140_G0R"
TARGET = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR"
AOI = box(6.00, 51.98, 6.54, 52.38)
PROJECT = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform
AOI_M = transform(PROJECT, AOI)
M_VALUES = (1, 2, 3, 4, 6, 8, 9)
DOY_TOLERANCE_DAYS = 14


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha12(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def midpoint(pair: dict[str, str]) -> float:
    return date.fromisoformat(pair["date_1"]).toordinal() + int(pair["lag_days"]) / 2


def days_of_year(pair: dict[str, str]) -> int:
    return date.fromisoformat(pair["date_1"]).timetuple().tm_yday


def main() -> None:
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {TARGET}")
    for folder in ("config", "raw", "manifests", "qc", "reports"):
        (TARGET / folder).mkdir(parents=True, exist_ok=True)
    for folder in ("config", "raw", "qc"):
        for path in (SOURCE / folder).iterdir():
            if path.is_file():
                shutil.copy2(path, TARGET / folder / path.name)
    for path in (SOURCE / "manifests").glob("scene_manifest_*.csv"):
        shutil.copy2(path, TARGET / "manifests" / path.name)

    scene_rows = [
        row
        for year in (2024, 2025)
        for product in ("SLC", "GRD")
        for row in read_csv(TARGET / "manifests" / f"scene_manifest_twente_{year}_{product}.csv")
    ]
    scenes = {row["scene_id"]: row for row in scene_rows}
    assert len(scenes) == len(scene_rows), "Duplicate scene IDs"
    footprint = {sid: transform(PROJECT, loads(row["footprint_wkt"])) for sid, row in scenes.items()}
    coverage = {sid: geom.intersection(AOI_M).area / AOI_M.area for sid, geom in footprint.items()}
    scene_group: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    for sid, row in scenes.items():
        key = (row["year"], row["relative_orbit"], row["acquisition_start_utc"][:10], row["platform"], row["product_type"])
        scene_group[key].append(sid)

    def best_scene(year: str, track: str, acquisition_date: str, platform: str, product: str, slc_id: str = "") -> str:
        ids = scene_group.get((year, track, acquisition_date, platform, product), [])
        if not ids:
            return "NONE"
        if product == "GRD" and slc_id:
            return sorted(ids, key=lambda sid: (-footprint[sid].intersection(footprint[slc_id]).intersection(AOI_M).area, -coverage[sid], sid))[0]
        return sorted(ids, key=lambda sid: (-coverage[sid], sid))[0]

    pair_rows: list[dict[str, str]] = []
    changed_endpoints = 0
    changed_grd = 0
    for year in (2024, 2025):
        source_rows = read_csv(SOURCE / "manifests" / f"pair_manifest_{year}.csv")
        revised: list[dict[str, str]] = []
        for row in source_rows:
            row = dict(row)
            old_endpoints = (row["scene_id_1"], row["scene_id_2"])
            old_grds = (row["grd_id_1"], row["grd_id_2"])
            for index in (1, 2):
                sid = best_scene(row["year"], row["track"], row[f"date_{index}"], row[f"platform_{index}"], "SLC")
                assert sid != "NONE", f"No SLC for {row['pair_id']} endpoint {index}"
                row[f"scene_id_{index}"] = sid
                row[f"grd_id_{index}"] = best_scene(row["year"], row["track"], row[f"date_{index}"], row[f"platform_{index}"], "GRD", sid)
            if old_endpoints != (row["scene_id_1"], row["scene_id_2"]):
                changed_endpoints += 1
            if old_grds != (row["grd_id_1"], row["grd_id_2"]):
                changed_grd += 1
            overlap = footprint[row["scene_id_1"]].intersection(footprint[row["scene_id_2"]]).intersection(AOI_M).area / AOI_M.area
            row["coverage_intersection_fraction"] = f"{overlap:.6f}"
            if overlap < 0.85:
                row["eligibility_status"] = "REJECTED"
                row["rejection_reason"] = f"SELECTED_FRAME_OVERLAP_{overlap:.4f}"
            else:
                row["eligibility_status"] = "CATALOG_CANDIDATE"
                row["rejection_reason"] = "NONE"
            revised.append(row)
        write_csv(TARGET / "manifests" / f"pair_manifest_{year}.csv", revised, list(source_rows[0]))
        pair_rows.extend(revised)

    pair_by_id = {row["pair_id"]: row for row in pair_rows}
    assert len(pair_by_id) == len(pair_rows), "Duplicate pair IDs"
    budgets: list[dict[str, object]] = []
    ladders: dict[tuple[int, int], list[dict[str, str]]] = {}
    for year in (2024, 2025):
        for track in (37, 88):
            candidates = sorted(
                (
                    row for row in pair_rows
                    if row["year"] == str(year)
                    and row["track"] == str(track)
                    and row["lag_days"] == "12"
                    and row["platform_1"] == "Sentinel-1A"
                    and row["platform_2"] == "Sentinel-1A"
                    and row["eligibility_status"] == "CATALOG_CANDIDATE"
                ),
                key=lambda row: (row["date_1"], row["pair_id"]),
            )
            disjoint: list[dict[str, str]] = []
            used: set[str] = set()
            for row in candidates:
                endpoints = {row["scene_id_1"], row["scene_id_2"]}
                if not used.intersection(endpoints):
                    disjoint.append(row)
                    used.update(endpoints)
            if len(disjoint) < max(M_VALUES):
                raise RuntimeError(f"S1A/A common M=9 unavailable for {year} T{track}: {len(disjoint)}")
            ladders[(year, track)] = disjoint
            anchors = [disjoint[0], disjoint[-1]]
            nested = anchors.copy()
            selected_at: dict[int, list[dict[str, str]]] = {2: anchors.copy()}
            while len(nested) < max(M_VALUES):
                ordered = sorted(nested, key=midpoint)
                gaps = [(midpoint(right) - midpoint(left), left, right) for left, right in zip(ordered[:-1], ordered[1:])]
                largest = max(gap for gap, _, _ in gaps)
                left, right = sorted(((left, right) for gap, left, right in gaps if gap == largest), key=lambda x: (x[0]["pair_id"], x[1]["pair_id"]))[0]
                target_mid = (midpoint(left) + midpoint(right)) / 2
                inside = [row for row in disjoint if row not in nested and midpoint(left) < midpoint(row) < midpoint(right)]
                if not inside:
                    inside = [row for row in disjoint if row not in nested]
                chosen = sorted(inside, key=lambda row: (abs(midpoint(row) - target_mid), row["pair_id"]))[0]
                nested.append(chosen)
                selected_at[len(nested)] = sorted(nested, key=lambda row: (row["date_1"], row["pair_id"]))
            seasonal_midpoint = (date(year, 6, 30).toordinal())
            selected_at[1] = [sorted(disjoint, key=lambda row: (abs(midpoint(row) - seasonal_midpoint), row["pair_id"]))[0]]
            for m in M_VALUES:
                selected = selected_at[m]
                endpoint_ids = sorted({sid for row in selected for sid in (row["scene_id_1"], row["scene_id_2"])})
                assert len(endpoint_ids) == 2 * m
                first = min(row["date_1"] for row in selected)
                last = max(row["date_2"] for row in selected)
                chron = sorted(selected, key=lambda row: row["date_1"])
                gaps = [(date.fromisoformat(right["date_1"]) - date.fromisoformat(left["date_2"])).days for left, right in zip(chron[:-1], chron[1:])]
                budgets.append({
                    "year": year, "track": track, "branch": "MAIN_12D_AA", "M": m,
                    "N_unique": len(endpoint_ids), "pair_ids": ";".join(row["pair_id"] for row in chron),
                    "scene_ids": ";".join(endpoint_ids), "first_date": first, "last_date": last,
                    "span_days": (date.fromisoformat(last) - date.fromisoformat(first)).days,
                    "max_gap_days": max(gaps, default=0), "median_lag_days": 12,
                    "rtc_jobs": len(endpoint_ids), "insar_jobs": m,
                    "estimated_credits": 15 * (len(endpoint_ids) + m),
                    "eligibility_status": "CATALOG_CANDIDATE_BURST_PENDING",
                })

    # Match the 2024 budget to the already frozen 2025 date slots. Independent
    # per-year midpoint selection shifts some 2024 slots by 18--30 days, even
    # though same-season A/A catalog pairs exist. This matching uses dates only.
    for track in (37, 88):
        budget25 = {int(row["M"]): row for row in budgets if row["year"] == 2025 and row["track"] == track}
        reference = [pair_by_id[pid] for pid in budget25[9]["pair_ids"].split(";")]
        options = sorted(
            (
                row for row in pair_rows
                if row["year"] == "2024" and row["track"] == str(track)
                and row["lag_days"] == "12" and row["platform_1"] == "Sentinel-1A"
                and row["platform_2"] == "Sentinel-1A"
                and row["eligibility_status"] == "CATALOG_CANDIDATE"
            ),
            key=lambda row: (row["date_1"], row["pair_id"]),
        )

        @lru_cache(None)
        def find_match(reference_index: int, previous_option_index: int) -> tuple[int, tuple[int, ...]] | None:
            if reference_index == len(reference):
                return (0, ())
            best: tuple[int, tuple[int, ...]] | None = None
            for option_index, option in enumerate(options):
                if previous_option_index >= 0 and option["date_1"] <= options[previous_option_index]["date_2"]:
                    continue
                difference = abs(days_of_year(reference[reference_index]) - days_of_year(option))
                if difference > DOY_TOLERANCE_DAYS:
                    continue
                tail = find_match(reference_index + 1, option_index)
                if tail is None:
                    continue
                candidate = (difference + tail[0], (option_index,) + tail[1])
                if best is None or candidate < best:
                    best = candidate
            return best

        solution = find_match(0, -1)
        if solution is None:
            raise RuntimeError(f"No 2024 T{track} non-overlapping date match within {DOY_TOLERANCE_DAYS} days")
        matching = {ref["pair_id"]: options[index] for ref, index in zip(reference, solution[1])}
        single_ref = pair_by_id[budget25[1]["pair_ids"]]
        single_options = [row for row in options if abs(days_of_year(single_ref) - days_of_year(row)) <= DOY_TOLERANCE_DAYS]
        if not single_options:
            raise RuntimeError(f"No 2024 T{track} date match for M=1")
        single = sorted(single_options, key=lambda row: (abs(days_of_year(single_ref) - days_of_year(row)), row["pair_id"]))[0]

        for budget in budgets:
            if budget["year"] != 2024 or budget["track"] != track:
                continue
            m = int(budget["M"])
            selected = [single] if m == 1 else [matching[pid] for pid in budget25[m]["pair_ids"].split(";")]
            selected.sort(key=lambda row: (row["date_1"], row["pair_id"]))
            scene_ids = sorted({sid for row in selected for sid in (row["scene_id_1"], row["scene_id_2"])})
            if len(scene_ids) != 2 * m:
                raise RuntimeError(f"Matched 2024 T{track} M{m} shares an endpoint")
            first = min(row["date_1"] for row in selected)
            last = max(row["date_2"] for row in selected)
            gap_days = [(date.fromisoformat(right["date_1"]) - date.fromisoformat(left["date_2"])).days for left, right in zip(selected[:-1], selected[1:])]
            budget.update({
                "N_unique": len(scene_ids), "pair_ids": ";".join(row["pair_id"] for row in selected),
                "scene_ids": ";".join(scene_ids), "first_date": first, "last_date": last,
                "span_days": (date.fromisoformat(last) - date.fromisoformat(first)).days,
                "max_gap_days": max(gap_days, default=0), "rtc_jobs": len(scene_ids),
                "insar_jobs": m, "estimated_credits": 15 * (len(scene_ids) + m),
            })
    write_csv(TARGET / "manifests" / "budget_design.csv", budgets, list(budgets[0]))

    # This is a catalog-only cross-year date check, not a matched-label result.
    doy_differences: dict[str, object] = {}
    for track in (37, 88):
        a = {int(row["M"]): row for row in budgets if row["year"] == 2024 and row["track"] == track}
        b = {int(row["M"]): row for row in budgets if row["year"] == 2025 and row["track"] == track}
        for m in M_VALUES:
            p24 = [pair_by_id[pid] for pid in a[m]["pair_ids"].split(";")]
            p25 = [pair_by_id[pid] for pid in b[m]["pair_ids"].split(";")]
            diffs = [abs(days_of_year(x) - days_of_year(y)) for x, y in zip(p24, p25)]
            doy_differences[f"T{track}_M{m}"] = {"start_doy_absolute_differences": diffs, "max": max(diffs), "within_predeclared_14_day_tolerance": max(diffs) <= DOY_TOLERANCE_DAYS}

    jobs: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add_job(job_type: str, batch: str, input_1: str, input_2: str = "", source_pair_id: str = "", name: str = "") -> None:
        full_input = input_1 if not input_2 else f"{input_1} & {input_2}"
        key = (job_type, full_input)
        if key in seen:
            return
        seen.add(key)
        params = "radiometry=gamma0;scale=power;resolution=20m" if job_type == "RTC_GAMMA" else "looks=10x2;apply_water_mask=True"
        jobs.append({
            "job_name": name or f"{job_type}_{sha12(full_input)}", "job_type": job_type,
            "batch": batch, "inputs": full_input, "parameters": params, "credit_cost": 15,
            "existing_product_hash": "NONE", "submission_status": "NOT_SUBMITTED",
            "source_pair_id": source_pair_id,
        })

    pilot_pair_ids: list[str] = []
    for track in (37, 88):
        twelve = pair_by_id[next(row for row in budgets if row["year"] == 2025 and row["track"] == track and row["M"] == 1)["pair_ids"]]
        six_candidates = [
            row for row in pair_rows
            if row["year"] == "2025" and row["track"] == str(track) and row["lag_days"] == "6"
            and row["platform_1"] == "Sentinel-1A" and row["platform_2"] == "Sentinel-1C"
            and row["eligibility_status"] == "CATALOG_CANDIDATE"
        ]
        if not six_candidates:
            raise RuntimeError(f"No full-coverage A/C 6-day catalog pair for T{track}")
        six = sorted(six_candidates, key=lambda row: (abs(midpoint(row) - date(2025, 6, 30).toordinal()), row["pair_id"]))[0]
        add_job("RTC_GAMMA", "pilot", twelve["scene_id_1"], name=f"PILOT_RTC_T{track}")
        add_job("INSAR_GAMMA", "pilot", twelve["scene_id_1"], twelve["scene_id_2"], twelve["pair_id"], f"PILOT_INSAR_T{track}_12D")
        add_job("INSAR_GAMMA", "pilot", six["scene_id_1"], six["scene_id_2"], six["pair_id"], f"PILOT_INSAR_T{track}_6D")
        pilot_pair_ids.extend([twelve["pair_id"], six["pair_id"]])

    for year, batch in ((2025, "main2025"), (2024, "crossyear2024")):
        year_budgets = [row for row in budgets if row["year"] == year]
        scene_ids = sorted({sid for row in year_budgets for sid in row["scene_ids"].split(";")})
        pair_ids = sorted({pid for row in year_budgets for pid in row["pair_ids"].split(";")})
        for sid in scene_ids:
            add_job("RTC_GAMMA", batch, sid)
        for pid in pair_ids:
            pair = pair_by_id[pid]
            add_job("INSAR_GAMMA", batch, pair["scene_id_1"], pair["scene_id_2"], pid)
    write_csv(TARGET / "manifests" / "job_manifest_dryrun.csv", jobs, list(jobs[0]))

    report = {
        "source_run": str(SOURCE), "repaired_run": str(TARGET),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "network_calls": 0, "submitted_jobs": 0,
        "changed_pair_endpoints": changed_endpoints, "changed_grd_endpoints": changed_grd,
        "pair_count": len(pair_rows), "catalog_rejected_pairs": sum(row["eligibility_status"] == "REJECTED" for row in pair_rows),
        "budget_rows": len(budgets), "job_count": len(jobs), "pilot_job_count": sum(row["batch"] == "pilot" for row in jobs),
        "batch_job_counts": dict(Counter(str(row["batch"]) for row in jobs)),
        "estimated_credits_unique_jobs": 15 * len(jobs),
        "pilot_pair_ids": pilot_pair_ids,
        "doy_tolerance_days_predeclared": DOY_TOLERANCE_DAYS,
        "cross_year_doy_differences": doy_differences,
        "remaining_gates": ["selected SLC burst overlap and perpendicular baseline", "HyP3 product processing pilot", "true RTC grid and pure-pixel label QC", "fresh account credit balance"],
        "status": "CATALOG_REPAIRED_ONLY_NOT_READY_FOR_EXTERNAL_SUBMISSION",
    }
    (TARGET / "reports" / "G0_CATALOG_REPAIR_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("changed_pair_endpoints", "changed_grd_endpoints", "pair_count", "catalog_rejected_pairs", "budget_rows", "job_count", "pilot_job_count", "batch_job_counts", "estimated_credits_unique_jobs", "status")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
