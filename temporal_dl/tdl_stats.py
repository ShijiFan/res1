"""Statistics and pre-registered judgement T1-T3 for the TempCNN baseline (PROTOCOL.md sections 4-5).

Per site, per (year, track, M): BA of DL and HGB for O2 and O3 averaged over the 3 E3 designs; dC = O3 - O2.
Two-level bootstrap (2,000 draws, RandomState(20261001)): designs resampled with replacement within the cell,
5 km test blocks resampled with replacement with one weight vector shared by all models and designs.
Usage: <yolo11marin python> tdl_stats.py --site 1|2
Out:   site{n}/cells.csv, site{n}/judgement.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from tdl_run import SITE_SCRIPTS, SEED  # noqa: E402

B = 2000


def block_cms(y, p, blk, ub, n=7):
    cm = np.zeros((len(ub), n, n))
    bi = np.searchsorted(ub, blk)
    np.add.at(cm, (bi, y, p), 1)
    return cm


def ba_w(cms, w):
    """cms (K designs, NB, 7, 7); w (Bw, NB) -> BA (Bw, K)"""
    cm = np.einsum("bn,knij->bkij", w, cms)
    s = cm.sum(-1)
    rec = np.divide(np.diagonal(cm, axis1=-2, axis2=-1), s, out=np.zeros_like(s), where=s > 0)
    return rec.mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, choices=[1, 2], required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(SITE_SCRIPTS[a.site]))
    import a1_s6_experiments as s6  # noqa: E402
    out = HERE / f"site{a.site}"
    reg = pd.read_csv(out / "REGISTRY.csv", dtype={"draw": str})
    rng = np.random.RandomState(SEED)
    rows = []
    for year in sorted(reg.year.unique(), reverse=True):
        ps = s6.Year(int(year)).parcel_set("primary")
        y = ps["y"].values.astype(int)
        blk = ps["block_5km"].astype(str).values
        ub = np.unique(blk)
        W = np.eye(len(ub))[rng.randint(0, len(ub), size=(B, len(ub)))].sum(1)
        for (trk, M), g in reg[reg.year == year].groupby(["track", "M"]):
            draws = sorted(g.draw.unique(), key=int)
            cm = {}
            for model in ("DL", "HGB"):
                for obs in ("O2", "O3"):
                    mats = []
                    for d in draws:
                        r = g[(g.draw == d) & (g.obs == obs)].iloc[0]
                        p = np.load(out / "preds" / f"{r.config_id}.npz")["y_pred"] if model == "DL" else \
                            np.load(s6.D["exp"] / r.hgb_npz)["y_pred"]
                        mats.append(block_cms(y, p.astype(int), blk, ub))
                    cm[(model, obs)] = np.stack(mats)
            one = np.ones((1, len(ub)))
            K = len(draws)
            dsel = rng.randint(0, K, size=(B, K))
            point, boot = {}, {}
            for key, c in cm.items():
                point[key] = float(ba_w(c, one)[0].mean())
                bw = ba_w(c, W)  # (B, K)
                boot[key] = np.take_along_axis(bw, dsel, axis=1).mean(1)
            dC = {m: point[(m, "O3")] - point[(m, "O2")] for m in ("DL", "HGB")}
            bdC = {m: boot[(m, "O3")] - boot[(m, "O2")] for m in ("DL", "HGB")}
            ci = lambda v: [round(float(x) * 100, 2) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
            rows.append({"year": year, "track": trk, "M": M, "n_designs": K,
                         "BA_DL_O2": round(point[("DL", "O2")] * 100, 2), "BA_HGB_O2": round(point[("HGB", "O2")] * 100, 2),
                         "DL_minus_HGB_O2": round((point[("DL", "O2")] - point[("HGB", "O2")]) * 100, 2),
                         "DL_minus_HGB_O2_ci": ci(boot[("DL", "O2")] - boot[("HGB", "O2")]),
                         "dC_DL": round(dC["DL"] * 100, 2), "dC_DL_ci": ci(bdC["DL"]),
                         "dC_HGB": round(dC["HGB"] * 100, 2), "dC_HGB_ci": ci(bdC["HGB"]),
                         "dC_DL_minus_HGB": round((dC["DL"] - dC["HGB"]) * 100, 2), "dC_DL_minus_HGB_ci": ci(bdC["DL"] - bdC["HGB"])})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cells.csv", index=False)
    low = df[df.M <= 2]
    t1 = int((low.DL_minus_HGB_O2 > 0).sum())
    t2_applicable = t1 >= 6
    t2 = int((low.dC_DL < low.dC_HGB).sum())
    t3 = df[df.M == 8]
    j = {"T1": {"cells_M_le_2": len(low), "DL_stronger_than_HGB_O2": t1,
                "all_cells_DL_minus_HGB_O2": df.DL_minus_HGB_O2.tolist()},
         "T2": {"applicable": t2_applicable, "cells": len(low), "dC_DL_lt_dC_HGB": t2,
                "verdict": ("CONSISTENT" if t2 >= 6 else "NOT CONSISTENT") if t2_applicable else
                "NOT APPLICABLE: DL not stronger than HGB"},
         "T3": {"cells": len(t3), "dC_DL_lt_1pp": int((t3.dC_DL < 1).sum()), "verdict": bool((t3.dC_DL < 1).all())}}
    (out / "judgement.json").write_text(json.dumps(j, indent=2))
    print(json.dumps(j, indent=1))


if __name__ == "__main__":
    main()
