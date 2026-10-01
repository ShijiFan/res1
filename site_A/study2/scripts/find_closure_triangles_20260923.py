"""Count closure-phase triangles in the planned InSAR set, and list the cheapest additions. Offline, no submission.

A triangle is (a,b), (b,c), (a,c) on the same track. Wrapped phase alone is useless for closure phase
if the planned pairs never form triangles.

Usage:
  & 'D:\\anaconda\\envs\\sar\\python.exe' scripts\\find_closure_triangles_20260923.py `
      --out-dir runs\\20260923_MAIN_PARAMS_WRAPPED --credit-per-insar 15
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "runs" / "20260923_G0R_CATALOG_REPAIR" / "manifests"


def load_pairs(year: int) -> dict:
    with (MAN / f"pair_manifest_{year}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ok = {}
    for r in rows:
        if r["eligibility_status"].startswith("CATALOG_CANDIDATE") and r["rejection_reason"] in ("", "NONE"):
            ok[(int(r["track"]), r["date_1"], r["date_2"])] = r
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--credit-per-insar", type=int, default=15)
    ap.add_argument("--max-lag", type=int, default=24, help="longest leg allowed in an added triangle (days)")
    a = ap.parse_args()
    out_dir = (a.out_dir if a.out_dir.is_absolute() else ROOT / a.out_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (MAN / "job_manifest_dryrun.csv").open(encoding="utf-8", newline="") as f:
        jobs = [r for r in csv.DictReader(f) if r["job_type"] == "INSAR_GAMMA" and r["batch"] != "pilot"]
    planned = defaultdict(set)  # (year, track) -> {(d1, d2)}
    for j in jobs:
        _, year, trk, _, _, d1, d2, _ = j["source_pair_id"].split("_")
        planned[(int(year), int(trk[1:]))].add((f"{d1[:4]}-{d1[4:6]}-{d1[6:]}", f"{d2[:4]}-{d2[4:6]}-{d2[6:]}"))

    summary, rows_out = {}, []
    for (year, trk), pset in sorted(planned.items()):
        cat = load_pairs(year)
        dates = sorted({d for p in pset for d in p})
        existing = [(x, y, z) for x in dates for y in dates for z in dates
                    if x < y < z and (x, y) in pset and (y, z) in pset and (x, z) in pset]
        # cheapest completion of each planned pair (a,b): need (b,c) and (a,c) in the catalogue
        adds = []
        for (d1, d2) in sorted(pset):
            best = None
            for (t, c1, c2), r in cat.items():
                if t != trk or c1 != d2:
                    continue
                c = c2
                if (trk, d1, c) not in cat or int(cat[(trk, d1, c)]["lag_days"]) > a.max_lag:
                    continue
                new = [p for p in [(d2, c), (d1, c)] if p not in pset]
                cand = (len(new), c, new)
                if best is None or cand < best:
                    best = cand
            if best:
                rows_out.append({"year": year, "track": trk, "planned_pair": f"{d1}_{d2}", "third_date": best[1],
                                 "new_pairs": ";".join(f"{x}_{y}" for x, y in best[2]),
                                 "n_new_pairs": best[0], "credits": best[0] * a.credit_per_insar,
                                 "new_pair_ids": ";".join(cat[(trk, x, y)]["pair_id"] for x, y in best[2]),
                                 "status": "CANDIDATE_NOT_SUBMITTED"})
        summary[f"{year}_T{trk}"] = {"planned_pairs": len(pset), "existing_triangles": len(existing),
                                    "closable_planned_pairs": sum(1 for r in rows_out if r["year"] == year and r["track"] == trk)}

    with (out_dir / "CLOSURE_TRIANGLE_CANDIDATES.csv").open("w", encoding="utf-8", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    (out_dir / "CLOSURE_TRIANGLE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"candidates: {len(rows_out)} rows -> {out_dir / 'CLOSURE_TRIANGLE_CANDIDATES.csv'}")


if __name__ == "__main__":
    main()
