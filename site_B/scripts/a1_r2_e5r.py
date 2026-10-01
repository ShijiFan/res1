"""R2 experiment E5r (EXPLORATORY, not preregistered): two tracks vs one track at equal total pairs,
on RANDOM date draws instead of the uniform design.

Why: the preregistered E5 compared the uniform designs, and the uniform M=2 design is March + October
only (no growing-season date), while the two-track M=2 design is one JUNE pair per track. E5 therefore
measured a date effect. E5r repeats the comparison with random dates on both sides.

Design (fixed here, do not change):
  years        2025 (20 draws per cell), 2024 (10 draws per cell)
  M_total      2, 4, 6, 8   (M_total/2 pairs on T37  +  M_total/2 pairs on T88)
  pairs        drawn uniformly without replacement from each track's 9 available 12-day pairs,
               independently per track, seed = SEED + 7000 + 1000*year + M_total
  observables  O2, O3
  learner      HGB, fixed hyper-parameters, same 6-fold spatial CV and 400 m buffer as every other block
  parcels      the same primary parcel set as E1/E3 (so block bootstrap weights are shared)
The single-track comparison arm is the EXISTING E3 draws at the same M (no new fits needed).

Writes experiments/E5r/*.npz and appends rows to experiments/REGISTRY.csv with exp="E5r" and
variant="exploratory_r2". Never modifies or deletes existing rows or files. Resumable.

Usage:
  & $py a1_r2_e5r.py --dry-run     # prints the plan, fits nothing
  & $py a1_r2_e5r.py               # runs (about 240 fits)
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np

from a1_common import D, SEED, TRACKS, get_logger, update_status
from a1_s6_experiments import Runner, cv_predict, features

log = get_logger("r2_e5r")
DRAWS = {2025: 20, 2024: 10}
MTOT = (2, 4, 6, 8)


def plan(R: Runner):
    out = []
    for y, n in DRAWS.items():
        Y = R.year(y)
        full = {trk: Y.available(trk) for trk in TRACKS}
        for trk in TRACKS:
            if len(full[trk]) != 9:
                raise SystemExit(f"STOP: {y} T{trk} has {len(full[trk])} available pairs, expected 9")
        for Mt in MTOT:
            h = Mt // 2
            rng = np.random.default_rng(SEED + 7000 + 1000 * y + Mt)
            combos = {trk: list(combinations(range(9), h)) for trk in TRACKS}
            seen = set()
            d = 0
            while d < n:
                keys = {trk: [full[trk][i] for i in combos[trk][rng.integers(len(combos[trk]))]] for trk in TRACKS}
                sig = json.dumps(keys, sort_keys=True)
                if sig in seen:          # no duplicate designs inside a cell
                    continue
                seen.add(sig)
                out.append((y, Mt, d, keys))
                d += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    shaf = D["reports"] / "A1_PREREG.sha256"
    R = Runner(shaf.read_text().strip())
    P = plan(R)
    n_fits = len(P) * 2
    log.info("E5r plan: %d designs, %d fits (expected 4*20 + 4*10 = 120 designs, 240 fits)", len(P), n_fits)
    assert len(P) == 120, f"plan has {len(P)} designs, expected 120"
    if a.dry_run:
        for row in P[:3]:
            print(row)
        return
    update_status("r2_e5r", "RUNNING", designs=len(P))
    parcels = {}
    for y, Mt, d, keys in P:
        Y = R.year(y)
        if y not in parcels:
            ps = Y.parcel_set("primary")
            R.parcels_file("E5r", f"{y}_primary", ps)
            parcels[y] = ps
        ps = parcels[y]
        for obs in ("O2", "O3"):
            cfg = dict(exp="E5r", year_train=y, year_test=y, tracks="37+88", M=Mt, draw=d, obs=obs,
                       learner="HGB", variant="exploratory_r2", pairs=json.dumps(keys))

            def fp(Y=Y, ps=ps, keys=keys, obs=obs, y=y):
                X = features(Y, ps.index, list(TRACKS), keys, obs)
                return cv_predict(Y, ps, X, Y, ps, X, "HGB"), X.shape[1], len(ps), f"{y}_primary"
            R.run(cfg, fp)
    update_status("r2_e5r", "DONE", designs=len(P))


if __name__ == "__main__":
    main()
