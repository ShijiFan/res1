#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Subset B Closure Triangles & Offline Payloads
======================================================
Extracts 8 representative closure triangle candidates (2 per year x track)
for phenological critical periods (mid-May crop emergence and early-July peak canopy).
Constructs offline HyP3 InSAR payloads for the 16 added pairs without submitting.
"""

import csv
import json
import hashlib
from pathlib import Path
import hyp3_sdk

ROOT = Path(r"E:\research\SAR\A1_observation_budget_20260923")
MAN_DIR = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR" / "manifests"
REP_DIR = ROOT / "runs" / "20260923_MAIN_PARAMS_WRAPPED" / "reports"

INSAR_PARAMS = dict(
    looks="10x2",
    include_wrapped_phase=True,
    phase_filter_parameter=0.6,
    apply_water_mask=True,
    include_inc_map=True
)

SELECTED_KEYS = [
    # (year, track, planned_pair)
    (2024, 37, "2024-05-12_2024-05-24", "2024 T37 mid-May crop emergence window (anchors A0 May 12 reference scene)"),
    (2024, 37, "2024-06-29_2024-07-11", "2024 T37 early-July peak summer biomass and canopy closure"),
    (2024, 88, "2024-05-15_2024-05-27", "2024 T88 mid-May crop emergence window (anchors A0 May 15 reference scene)"),
    (2024, 88, "2024-07-02_2024-07-14", "2024 T88 early-July peak summer biomass and canopy closure"),
    (2025, 37, "2025-05-19_2025-05-31", "2025 T37 late-spring / mid-May vegetative expansion stage"),
    (2025, 37, "2025-07-06_2025-07-18", "2025 T37 early-July peak biomass and maximum vegetation cover"),
    (2025, 88, "2025-05-22_2025-06-03", "2025 T88 late-spring / mid-May vegetative expansion stage"),
    (2025, 88, "2025-07-09_2025-07-21", "2025 T88 mid-July peak biomass and reproduction stage"),
]

def load_pair_catalog():
    catalog = {}
    for year in (2024, 2025):
        pfile = MAN_DIR / f"pair_manifest_{year}.csv"
        with open(pfile, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pid = r["pair_id"]
                catalog[pid] = r
    return catalog

def main():
    print("Loading candidate table...")
    cand_file = REP_DIR / "CLOSURE_TRIANGLE_CANDIDATES.csv"
    with open(cand_file, "r", encoding="utf-8") as f:
        all_cands = list(csv.DictReader(f))

    catalog = load_pair_catalog()
    subset_b_rows = []

    sel_lookup = {(y, t, p): r for (y, t, p, r) in SELECTED_KEYS}

    for c in all_cands:
        k = (int(c["year"]), int(c["track"]), c["planned_pair"])
        if k in sel_lookup:
            row = dict(c)
            row["selection_reason"] = sel_lookup[k]
            subset_b_rows.append(row)

    print(f"Selected {len(subset_b_rows)} triangle rows for Scheme B.")

    # Write CLOSURE_SUBSET_B.csv
    out_csv = REP_DIR / "CLOSURE_SUBSET_B.csv"
    fieldnames = list(subset_b_rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset_b_rows)
    print(f"Saved: {out_csv}")

    # Prepare offline payloads for the 16 added pairs
    new_jobs = []
    seen_pair_ids = set()

    for r in subset_b_rows:
        pair_ids = r["new_pair_ids"].split(";")
        for pid in pair_ids:
            if pid in seen_pair_ids:
                continue
            seen_pair_ids.add(pid)
            pair_meta = catalog[pid]
            g1 = pair_meta["scene_id_1"]
            g2 = pair_meta["scene_id_2"]
            job_name = f"SUBSET_B_{pid}"
            payload = hyp3_sdk.HyP3.prepare_insar_job(g1, g2, name=job_name, **INSAR_PARAMS)
            enc = json.dumps(payload, sort_keys=True).encode()
            new_jobs.append({
                "job_name": job_name,
                "batch": "closure_subset_b",
                "job_type": "INSAR_GAMMA",
                "source_pair_id": pid,
                "scene_id_1": g1,
                "scene_id_2": g2,
                "lag_days": int(pair_meta["lag_days"]),
                "perpendicular_baseline_m": pair_meta["perpendicular_baseline_m"],
                "sdk_payload": payload,
                "payload_sha256": hashlib.sha256(enc).hexdigest(),
                "status": "PREPARED_OFFLINE_NOT_SUBMITTED"
            })

    print(f"Prepared {len(new_jobs)} distinct added InSAR jobs (expected: 16).")
    total_credits = len(new_jobs) * 15
    print(f"Total credit cost for added pairs: {total_credits} credits.")

    out_preflight = {
        "created_utc": "2026-09-23T14:15:00Z",
        "description": "Scheme B closure phase triangle additions (8 triangles, 16 InSAR pairs)",
        "frozen_insar_params": INSAR_PARAMS,
        "n_triangles": len(subset_b_rows),
        "n_added_insar_pairs": len(new_jobs),
        "credit_cost_total": total_credits,
        "submitted_jobs": 0,
        "jobs": new_jobs
    }

    out_json_path = REP_DIR / "CLOSURE_SUBSET_B_PREFLIGHT.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_preflight, f, indent=2)
    print(f"Saved: {out_json_path}")

if __name__ == "__main__":
    main()
