"""Step 1: build closure subset B2 and its offline SDK payloads.

Why B2: in CLOSURE_SUBSET_B.csv the two 2025 T37 triangles use S1A/S1C 6-day legs, but
G0_6DAY_AC_BASELINE_STACK_AUDIT.json reports NO_STACK_MATCH for every 2025 T37 6-day A/C pair.
B2 keeps all other rows and replaces each 2025 T37 triangle by an S1A-only 12+12/24-day triangle
(planned pair (t, t+12) + (t+12, t+24) + (t, t+24)).

Output: RUN/submit/CLOSURE_SUBSET_B2.csv, RUN/submit/CLOSURE_SUBSET_B2_PREFLIGHT.json. No submission.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta

import hyp3_sdk

from a1_common import (D, FROZEN_INSAR, SUBSET_B_CSV, ensure_dirs, get_logger, load_pair_catalogue, now,
                       update_status, write_json)

log = get_logger("s1_subset_b2")


def plus(d: str, n: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()


def main() -> None:
    ensure_dirs()
    cat = load_pair_catalogue()
    rows = list(csv.DictReader(open(SUBSET_B_CSV, encoding="utf-8", newline="")))
    out = []
    for r in rows:
        y, t = int(r["year"]), int(r["track"])
        if not (y == 2025 and t == 37):
            r["b2_change"] = "unchanged"
            out.append(r)
            continue
        d1, d2 = r["planned_pair"].split("_")
        d3 = plus(d2, 12)
        legs = [(d2, d3), (d1, d3)]
        recs = [cat.get((y, t, a, b)) for a, b in legs]
        ok = all(rec is not None and rec["platform_1"] == rec["platform_2"] == "Sentinel-1A"
                 and rec["eligibility_status"].startswith("CATALOG_CANDIDATE") for rec in recs)
        if not ok:
            log.warning("no S1A-only 12+24 completion for %s; row dropped", r["planned_pair"])
            continue
        r.update(third_date=d3, new_pairs=";".join(f"{a}_{b}" for a, b in legs), n_new_pairs="2", credits="30",
                 new_pair_ids=";".join(rec["pair_id"] for rec in recs),
                 selection_reason=r["selection_reason"] + " | B2: replaced S1A/S1C 6-day legs (T37 NO_STACK_MATCH) by S1A 12+24",
                 b2_change="replaced_T37_AC_legs")
        out.append(r)

    fields = list(out[0].keys())
    with open(D["submit"] / "CLOSURE_SUBSET_B2.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    by_id = {v["pair_id"]: v for v in cat.values()}
    jobs, seen = [], set()
    for r in out:
        for pid in r["new_pair_ids"].split(";"):
            if pid in seen:
                continue
            seen.add(pid)
            rec = by_id[pid]
            name = f"B2_{pid}"
            payload = hyp3_sdk.HyP3.prepare_insar_job(rec["scene_id_1"], rec["scene_id_2"], name=name, **FROZEN_INSAR)
            enc = json.dumps(payload, sort_keys=True).encode()
            jobs.append({"job_name": name, "batch": "closure_subset_b2", "job_type": "INSAR_GAMMA",
                         "source_pair_id": pid, "sdk_payload": payload,
                         "payload_sha256": hashlib.sha256(enc).hexdigest(), "status": "PREPARED_OFFLINE_NOT_SUBMITTED"})
    write_json(D["submit"] / "CLOSURE_SUBSET_B2_PREFLIGHT.json",
               {"created_utc": now(), "frozen_insar_params": FROZEN_INSAR, "n_triangles": len(out),
                "n_insar_pairs": len(jobs), "credit_cost_total": 15 * len(jobs), "jobs": jobs})
    log.info("B2: %d triangles, %d InSAR pairs, %d credits", len(out), len(jobs), 15 * len(jobs))
    update_status("s1_subset_b2", "DONE", triangles=len(out), pairs=len(jobs))


if __name__ == "__main__":
    main()
