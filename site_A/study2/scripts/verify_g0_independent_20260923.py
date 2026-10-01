"""Read-only cross-file audit of the A1 G0 manifests.

The existing verify_g0.py checks counts. This script checks the identities and
relationships needed before a HyP3 job can be constructed. It never submits jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
from shapely.wkt import loads


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parts(value: str) -> list[str]:
    return [piece for piece in value.split(";") if piece]


def audit(run_dir: Path) -> dict:
    manifests = run_dir / "manifests"
    scenes: dict[str, dict[str, str]] = {}
    issues: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    def issue(kind: str, identity: str, detail: str) -> None:
        counts[kind] += 1
        if len(issues) < 250:
            issues.append({"kind": kind, "identity": identity, "detail": detail})

    for year in (2024, 2025):
        for product in ("SLC", "GRD"):
            for scene in rows(manifests / f"scene_manifest_twente_{year}_{product}.csv"):
                sid = scene["scene_id"]
                if sid in scenes:
                    issue("duplicate_scene_id", sid, "scene ID occurs more than once")
                scenes[sid] = scene
                if scene["product_type"] != product or scene["year"] != str(year):
                    issue("scene_file_metadata", sid, "product/year disagree with filename")

    project = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform
    aoi = transform(project, box(6.00, 51.98, 6.54, 52.38))
    slc_footprints = {
        sid: transform(project, loads(scene["footprint_wkt"]))
        for sid, scene in scenes.items() if scene["product_type"] == "SLC"
    }

    pairs: dict[str, dict[str, str]] = {}
    for year in (2024, 2025):
        for pair in rows(manifests / f"pair_manifest_{year}.csv"):
            pid = pair["pair_id"]
            if pid in pairs:
                issue("duplicate_pair_id", pid, "pair ID occurs more than once")
            pairs[pid] = pair
            endpoints = [pair["scene_id_1"], pair["scene_id_2"]]
            grds = [pair["grd_id_1"], pair["grd_id_2"]]
            if len(set(endpoints)) != 2:
                issue("duplicate_pair_endpoint", pid, "both endpoints are identical")
            for index, sid in enumerate(endpoints):
                s = scenes.get(sid)
                if s is None:
                    issue("missing_slc_endpoint", pid, sid)
                    continue
                if s["product_type"] != "SLC":
                    issue("wrong_endpoint_product", pid, sid)
                if s["relative_orbit"] != pair["track"] or s["year"] != pair["year"]:
                    issue("pair_track_year_mismatch", pid, sid)
                if s["platform"] != pair[f"platform_{index + 1}"]:
                    issue("pair_platform_mismatch", pid, sid)
                if s["acquisition_start_utc"][:10] != pair[f"date_{index + 1}"]:
                    issue("pair_date_mismatch", pid, sid)
                if s["polarization"] != pair["polarization"]:
                    issue("pair_polarization_mismatch", pid, sid)
            for gid in grds:
                g = scenes.get(gid)
                if g is None:
                    issue("missing_grd_endpoint", pid, gid)
                elif g["product_type"] != "GRD" or g["relative_orbit"] != pair["track"]:
                    issue("wrong_grd_endpoint", pid, gid)
            try:
                lag = (date.fromisoformat(pair["date_2"]) - date.fromisoformat(pair["date_1"])).days
                if lag != int(pair["lag_days"]):
                    issue("pair_lag_mismatch", pid, f"computed {lag}, stored {pair['lag_days']}")
            except (ValueError, KeyError) as exc:
                issue("pair_lag_invalid", pid, str(exc))
            if all(sid in slc_footprints for sid in endpoints):
                selected_frame_overlap = slc_footprints[endpoints[0]].intersection(slc_footprints[endpoints[1]]).intersection(aoi).area / aoi.area
                if abs(selected_frame_overlap - float(pair["coverage_intersection_fraction"])) > 0.001:
                    issue("pair_overlap_mismatch", pid, f"computed={selected_frame_overlap:.6f}; stored={pair['coverage_intersection_fraction']}")
                if pair["eligibility_status"] == "CATALOG_CANDIDATE" and selected_frame_overlap < 0.85:
                    issue("candidate_selected_frame_undercoverage", pid, f"computed={selected_frame_overlap:.6f}")

    by_group: dict[tuple[str, str, str], list[tuple[int, set[str]]]] = defaultdict(list)
    budgets = rows(manifests / "budget_design.csv")
    for budget in budgets:
        identity = f"{budget['year']}_T{budget['track']}_{budget['branch']}_M{budget['M']}"
        selected_pairs = parts(budget["pair_ids"])
        selected_scenes = parts(budget["scene_ids"])
        if len(selected_pairs) != int(budget["M"]) or len(set(selected_pairs)) != len(selected_pairs):
            issue("budget_pair_count", identity, "M and unique pair IDs disagree")
        if len(selected_scenes) != int(budget["N_unique"]) or len(set(selected_scenes)) != len(selected_scenes):
            issue("budget_scene_count", identity, "N_unique and unique scene IDs disagree")
        endpoint_set: set[str] = set()
        for pid in selected_pairs:
            pair = pairs.get(pid)
            if pair is None:
                issue("budget_missing_pair", identity, pid)
                continue
            endpoint_set.update((pair["scene_id_1"], pair["scene_id_2"]))
            if pair["year"] != budget["year"] or pair["track"] != budget["track"]:
                issue("budget_pair_track_year", identity, pid)
            if pair["lag_days"] != "12" or pair["platform_1"] != "Sentinel-1A" or pair["platform_2"] != "Sentinel-1A":
                issue("budget_pair_role", identity, pid)
        if endpoint_set != set(selected_scenes):
            issue("budget_endpoints", identity, f"missing={len(endpoint_set - set(selected_scenes))}; extra={len(set(selected_scenes) - endpoint_set)}")
        if budget["branch"] == "MAIN_12D_AA":
            if len(endpoint_set) != 2 * int(budget["M"]):
                issue("budget_shared_endpoint", identity, "the main design requires two unique scenes per pair")
            if int(budget["N_unique"]) != 2 * int(budget["M"]):
                issue("budget_n_not_2m", identity, "N_unique must equal 2M in the main design")
        if int(budget["rtc_jobs"]) != int(budget["N_unique"]) or int(budget["insar_jobs"]) != int(budget["M"]):
            issue("budget_job_count", identity, "RTC/INSAR counts differ from N_unique/M")
        if int(budget["estimated_credits"]) != 15 * (int(budget["rtc_jobs"]) + int(budget["insar_jobs"])):
            issue("budget_credit_arithmetic", identity, "credit total differs from current Basic 15-credit products")
        by_group[(budget["year"], budget["track"], budget["branch"])].append((int(budget["M"]), set(selected_pairs)))
    for group, levels in by_group.items():
        levels = sorted((m, p) for m, p in levels if m >= 2)
        for (m1, p1), (m2, p2) in zip(levels, levels[1:]):
            if not p1.issubset(p2):
                issue("budget_not_nested", str(group), f"M{m1} is not contained in M{m2}")

    jobs = rows(manifests / "job_manifest_dryrun.csv")
    baseline_lookup_path = run_dir / "reports" / "G0_PILOT_ASF_BASELINE_LOOKUP.json"
    baseline_by_name = {}
    if baseline_lookup_path.exists():
        baseline_lookup = json.loads(baseline_lookup_path.read_text(encoding="utf-8"))
        baseline_by_name = {item["job_name"]: item for item in baseline_lookup.get("jobs", [])}
    seen_names: set[str] = set()
    pending_pilot_burst_checks: set[str] = set()
    for job in jobs:
        name = job["job_name"]
        if name in seen_names:
            issue("duplicate_job_name", name, "job name occurs more than once")
        seen_names.add(name)
        if job["submission_status"] != "NOT_SUBMITTED":
            issue("non_dryrun_job", name, job["submission_status"])
        if "..." in job["inputs"]:
            issue("abbreviated_job_inputs", name, job["inputs"])
        if job["job_type"].startswith("RTC"):
            sid = job["inputs"]
            s = scenes.get(sid)
            if s is None:
                issue("job_missing_scene", name, sid)
            elif s["product_type"] not in {"SLC", "GRD"}:
                # ASF HyP3 RTC GAMMA accepts IW SLC as well as GRD-H inputs.
                issue("rtc_wrong_product", name, f"scene {sid} has product_type={s['product_type']}")
        elif job["job_type"].startswith("INSAR"):
            if job["batch"] == "pilot" and name in baseline_by_name and baseline_by_name[name]["status"] != "BASELINE_STACK_MATCH_BURST_PENDING":
                issue("pilot_baseline_stack_no_match", name, baseline_by_name[name]["status"])
            inputs = [part.strip() for part in job["inputs"].split(" & ")]
            if len(inputs) != 2:
                issue("insar_input_count", name, job["inputs"])
            for sid in inputs:
                s = scenes.get(sid)
                if s is None:
                    issue("job_missing_scene", name, sid)
                elif s["product_type"] != "SLC":
                    issue("insar_not_slc", name, sid)
            if len(inputs) == 2 and all(sid in scenes for sid in inputs):
                if scenes[inputs[0]]["relative_orbit"] != scenes[inputs[1]]["relative_orbit"]:
                    issue("insar_track_mismatch", name, job["inputs"])
                if job["batch"] == "pilot":
                    matching = [pair for pair in pairs.values() if pair["scene_id_1"] == inputs[0] and pair["scene_id_2"] == inputs[1]]
                    if len(matching) != 1:
                        issue("pilot_pair_membership", name, f"matching pair count={len(matching)}")
                    elif not matching[0]["common_burst_count"].isdigit() or int(matching[0]["common_burst_count"]) <= 0:
                        pending_pilot_burst_checks.add(matching[0]["pair_id"])
        else:
            issue("unknown_job_type", name, job["job_type"])

    return {
        "run_dir": str(run_dir),
        "counts": {"scenes": len(scenes), "pairs": len(pairs), "budgets": len(budgets), "jobs": len(jobs), "pilot_jobs": sum(j["batch"] == "pilot" for j in jobs)},
        "issue_counts": dict(sorted(counts.items())),
        "pending_pilot_burst_checks": sorted(pending_pilot_burst_checks),
        "issues_truncated_at": 250,
        "issues": issues,
        "cross_file_integrity_passed": len(counts) == 0,
        "ready_for_external_submission": len(counts) == 0 and not pending_pilot_burst_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("counts", "issue_counts", "pending_pilot_burst_checks", "cross_file_integrity_passed", "ready_for_external_submission")}, ensure_ascii=False, indent=2))
    print(f"Full report: {args.out}")
    return 0 if result["ready_for_external_submission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
