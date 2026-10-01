"""Contemporaneous built-up check with BAG building footprints (PROTOCOL.md section 5.2).

Step A: rasterise BAG pand (bouwjaar <= 2024, existing status) at 4 m on the 40 m analysis grid,
        average to a building-cover fraction f_BAG per 40 m pixel -> ../labels/f_bag_40m.npy
Step B: predict every Omega pixel of the 12 evaluation blocks with the v7 I28 / I28_G29 models.
Step C: per (learner, rep, source, seed, target, block) counts for
        test 1  built-up recall on pixels with f_BAG >= 0.25,
        test 2  predicted-built-up rate on pixels whose 5x5 neighbourhood (+-80 m) has f_BAG == 0;
        20,000 block-bootstrap draws (RandomState(42)) of the RF/SVC differences, cross-track pooled.
Out: ../ws/bag_check/{bag_counts.npz, bag_check_results.csv, bag_label_agreement.json}
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import load
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_build_cube_v7 as s3  # noqa: E402
import s4_run_R1_v7 as s4  # noqa: E402

r1 = s4.r1
ROOT = s4.ROOT
WS = s4.WS
OUT = WS / "bag_check"
LAB = ROOT / "labels"
KEEP_STATUS = {"Pand in gebruik", "Pand in gebruik (niet ingemeten)", "Verbouwing pand",
               "Sloopvergunning verleend", "Pand buiten gebruik"}
F_BUILT, SUB = 0.25, 10
DEMOLISHED_AFTER = "2024-05-31"
REPS = ["I28", "I28_G29"]
URBAN = 3


def step_a():
    fp = LAB / "f_bag_40m.npy"
    if fp.exists():
        return np.load(fp)
    TF, CRS, shp = s3.reference_grid()
    g = gpd.read_file(LAB / "bag_pand_aoi.geojsonl")
    g["bouwjaar"] = pd.to_numeric(g["bouwjaar"], errors="coerce")
    # demolished after the May-2024 acquisitions still stood then (documentdatum = status-change document date)
    late_demolished = (g["status"] == "Pand gesloopt") & (g["documentdatum"].astype(str) > DEMOLISHED_AFTER)
    keep = g[(g["bouwjaar"] <= 2024) & (g["status"].isin(KEEP_STATUS) | late_demolished)].set_crs(4326, allow_override=True).to_crs(CRS)
    fine_tf = TF * Affine.scale(1 / SUB)
    fine = rasterize(((geom, 1) for geom in keep.geometry), out_shape=(shp[0] * SUB, shp[1] * SUB),
                     transform=fine_tf, fill=0, dtype="uint8")
    f = fine.reshape(shp[0], SUB, shp[1], SUB).mean(axis=(1, 3)).astype("float32")
    np.save(fp, f)
    (LAB / "f_bag_40m_meta.json").write_text(json.dumps({
        "n_features_total": int(len(g)), "n_kept": int(len(keep)), "keep_status": sorted(KEEP_STATUS),
        "bouwjaar_max": 2024, "demolished_after_kept": DEMOLISHED_AFTER, "n_late_demolished": int(late_demolished.sum()), "fine_resolution_m": 40 / SUB,
        "status_counts": g["status"].value_counts().to_dict()}, indent=2, ensure_ascii=False))
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f_bag = step_a()
    sv = r1.SAR_V2
    c7 = np.load(s4.CUBE)
    v4 = np.load(sv / "data" / "flevoland_datacube_v4_multibaseline.npz")
    cm = np.load(sv / "labels" / "consensus_mask.npz")
    mmu = np.load(sv / "mmu" / "mmu_mask.npz")
    om = np.load(s4.SAR / "revision_experiments_20260914" / "omega_v6_rebuilt.npy")
    vi = np.where(mmu["mmu_valid"])
    omv = om[vi]
    gy, gx = v4["grid_y"][vi].astype(int), v4["grid_x"][vi].astype(int)
    bids = (gy // 125) * 100 + (gx // 125)
    teb = json.loads((WS / "splits" / "split_manifest.json").read_text())["historical_development_split"]["eval_blocks"]
    ev = np.where(np.isin(bids, teb) & omv)[0]
    fb = f_bag[gy[ev], gx[ev]]
    far = (uniform_filter((f_bag > 0).astype("float32"), 5, mode="constant") == 0)[gy[ev], gx[ev]]
    built = fb >= F_BUILT
    b_ev = bids[ev]

    # label agreement on C2 inside eval blocks
    y_wc = v4["y"][vi].astype(int)
    y_clc = (cm["clc_on_mmu"] - 1).astype(int)
    c2 = (omv & (y_wc == y_clc) & (y_wc >= 0) & (y_wc < 5))[ev]
    agree = {"n_eval_omega": int(ev.size), "n_bag_built": int(built.sum()), "n_far": int(far.sum()),
             "c2_urban": int((c2 & (y_wc[ev] == URBAN)).sum()),
             "c2_urban_with_bag_built": int((c2 & (y_wc[ev] == URBAN) & built).sum()),
             "c2_urban_with_any_building": int((c2 & (y_wc[ev] == URBAN) & (fb > 0)).sum()),
             "bag_built_labelled_urban_in_c2": int((c2 & built & (y_wc[ev] == URBAN)).sum()),
             "bag_built_in_c2": int((c2 & built).sum())}
    (OUT / "bag_label_agreement.json").write_text(json.dumps(agree, indent=2))
    print(agree)

    feats, _ = r1.extract_features(c7, vi)
    for k in [k for k in feats if k not in REPS]:  # free the three representations not used here
        del feats[k]
    NB = len(teb)
    counts = {}  # (learner, rep) -> array (4 src, 4 tgt, 3 seeds, NB, 4)
    for learner in ("RF", "SVC"):
        for rep in REPS:
            arr = np.zeros((4, 4, len(r1.SEEDS), NB, 4), dtype=np.int64)
            for si, src in enumerate(r1.TR):
                for ki, seed in enumerate(r1.SEEDS):
                    rd = WS / "runs_R1" / f"{learner}_{rep}_{src}_s{seed}"
                    clf = load(rd / "model.joblib")
                    sc = load(rd / "scaler.joblib") if (rd / "scaler.joblib").exists() else None
                    for ti, tgt in enumerate(r1.TARGET_ORDER):
                        X = feats[rep][tgt][ev]
                        pu = clf.predict(sc.transform(X) if sc is not None else X) == URBAN
                        for bi, b in enumerate(teb):
                            m = b_ev == b
                            arr[si, ti, ki, bi] = [(m & built).sum(), (m & built & pu).sum(),
                                                   (m & far).sum(), (m & far & pu).sum()]
                    print("done", learner, rep, src, seed, flush=True)
            counts[(learner, rep)] = arr
    np.savez(OUT / "bag_counts.npz", **{f"{l}_{r}": a for (l, r), a in counts.items()})

    rng = np.random.RandomState(42)
    boot = np.eye(NB)[rng.randint(0, NB, size=(20000, NB))].sum(axis=1)  # (B, NB) block multiplicities
    cross = ~np.eye(4, dtype=bool)
    rows = []
    for learner in ("RF", "SVC"):
        def rates(rep, w):
            a = counts[(learner, rep)][cross].sum(axis=(0, 1))  # (NB, 4) pooled over cross pairs & seeds
            s = w @ a
            return s[..., 1] / np.maximum(s[..., 0], 1), s[..., 3] / np.maximum(s[..., 2], 1)
        full = np.ones((1, NB))
        (r0, q0), (r1_, q1) = rates("I28", full), rates("I28_G29", full)
        (br0, bq0), (br1, bq1) = rates("I28", boot), rates("I28_G29", boot)
        for name, a0, a1, b0, b1 in (("bag_built_recall", r0, r1_, br0, br1), ("far_false_builtup_rate", q0, q1, bq0, bq1)):
            d = (b1 - b0) * 100
            rows.append({"learner": learner, "metric": name, "I28_pct": round(float(a0[0]) * 100, 3),
                         "I28_G29_pct": round(float(a1[0]) * 100, 3), "delta_pp": round(float(a1[0] - a0[0]) * 100, 3),
                         "ci_lo": round(float(np.percentile(d, 2.5)), 3), "ci_hi": round(float(np.percentile(d, 97.5)), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bag_check_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
