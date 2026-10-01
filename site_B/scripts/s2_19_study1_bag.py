"""Site 2, Study 1: BAG building-register check (pre-registered tests 1-2) and the exploratory strata.

Same rules as site 1 (A0_gamma0_rerun_20260929/scripts/s12_bag_check.py and s15_bag_sensitivity.py):
  * keep bouwjaar <= 2024 with status in use / not surveyed / renovation / demolition permit / out of use,
    plus 'Pand gesloopt' with documentdatum > 2024-05-31; rasterize at 4 m, average to 40 m -> f_BAG;
  * pixel sets inside the evaluation blocks (all Omega pixels): thr25/50/75, cluster size 1-4 / 5-50 / >50
    (8-connected f_BAG > 0 cells, with f_BAG >= 0.25), far = no building in the 5x5 neighbourhood;
  * families: RF and SVC (I28 vs I28_G29, 3 seeds) and CNN (CNN_I vs CNN_IG, tuned_p9, 5 seeds);
  * 20,000 block-bootstrap draws, RandomState(42), cross-track pairs pooled, seeds averaged.
Tests 1 (thr25) and 2 (far) are pre-registered (PREREG_SITE2.md R1-3); the strata are exploratory.
Out: ../runs/STUDY1/ws/bag_check/{f_bag_40m.npy, bag_results.csv, bag_info.json}
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from joblib import load
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import label, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_14_study1_R1 as s14  # noqa: E402
import s2_16_study1_cnn as s16  # noqa: E402
from s10_cnn_baseline import ARMS, Net, predict  # noqa: E402

r1 = s14.r1
S1, WS = s14.S1, s14.WS
OUT = WS / "bag_check"
KEEP = {"Pand in gebruik", "Pand in gebruik (niet ingemeten)", "Verbouwing pand", "Sloopvergunning verleend", "Pand buiten gebruik"}
SETS = ["thr25", "thr50", "thr75", "iso", "small", "settle", "far"]
URBAN, SUB = 3, 10


def f_bag():
    fp = OUT / "f_bag_40m.npy"
    if fp.exists():
        return np.load(fp)
    g = json.loads((s14.ROOT / "runs" / "G0L_COMMON_GRID" / "grid_spec.json").read_text())
    TF, H, W = Affine.from_gdal(*g["affine_gdal"]), int(g["height"]), int(g["width"])
    b = gpd.read_file(s14.ROOT / "labels" / "bag" / "bag_pand_aoi.geojsonl")
    b["bouwjaar"] = pd.to_numeric(b["bouwjaar"], errors="coerce")
    late = (b["status"] == "Pand gesloopt") & (b["documentdatum"].astype(str) > "2024-05-31")
    keep = b[(b["bouwjaar"] <= 2024) & (b["status"].isin(KEEP) | late)].set_crs(4326, allow_override=True).to_crs(g["crs"])
    fine = rasterize(((geom, 1) for geom in keep.geometry), out_shape=(H * SUB, W * SUB), transform=TF * Affine.scale(1 / SUB),
                     fill=0, dtype="uint8")
    f = fine.reshape(H, SUB, W, SUB).mean(axis=(1, 3)).astype("float32")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(fp, f)
    (OUT / "f_bag_meta.json").write_text(json.dumps({"n_total": int(len(b)), "n_kept": int(len(keep)), "n_late_demolished": int(late.sum())}))
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fb_map = f_bag()
    lab, n = label(fb_map > 0, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel())
    csize = np.where(lab > 0, size[lab], 0)
    c, vi, y_wc, c_mask, bids, coords = s14.inputs()
    L = np.load(S1 / "labels.npz")
    omv = L["omega"][vi]
    teb = json.loads((S1 / "split_manifest.json").read_text())["historical_development_split"]["eval_blocks"]
    ev = np.where(np.isin(bids, teb) & omv)[0]
    gy, gx = L["grid_y"][vi][ev].astype(int), L["grid_x"][vi][ev].astype(int)
    fb, cs = fb_map[gy, gx], csize[gy, gx]
    far = (uniform_filter((fb_map > 0).astype("float32"), 5, mode="constant") == 0)[gy, gx]
    masks = np.stack([fb >= 0.25, fb >= 0.5, fb >= 0.75, (fb >= 0.25) & (cs <= 4), (fb >= 0.25) & (cs > 4) & (cs <= 50),
                      (fb >= 0.25) & (cs > 50), far])
    blk = np.stack([bids[ev] == b for b in teb])
    n_set = (blk[:, None, :] & masks[None]).sum(-1)
    c2ev = c_mask[ev]
    info = {"n_eval_omega": int(ev.size), "set_sizes": dict(zip(SETS, n_set.sum(0).tolist())),
            "c2_builtup": int((c2ev & (y_wc[ev] == URBAN)).sum()),
            "c2_builtup_with_any_building": int((c2ev & (y_wc[ev] == URBAN) & (fb > 0)).sum()),
            "c2_bag_built": int((c2ev & (fb >= 0.25)).sum()),
            "c2_bag_built_labelled_builtup": int((c2ev & (fb >= 0.25) & (y_wc[ev] == URBAN)).sum())}
    print(info, flush=True)
    hits = lambda pu: (blk[:, None, :] & masks[None] & pu[None, None, :]).sum(-1)  # noqa: E731
    counts = {}
    feats, _ = r1.extract_features(c, vi)
    for k in [k for k in feats if k not in ("I28", "I28_G29")]:
        del feats[k]
    for learner in ("RF", "SVC"):
        for rep in ("I28", "I28_G29"):
            arr = np.zeros((4, 4, 3, len(teb), len(SETS)), dtype=np.int64)
            for si, src in enumerate(r1.TR):
                for ki, seed in enumerate(r1.SEEDS):
                    rd = WS / "runs_R1" / f"{learner}_{rep}_{src}_s{seed}"
                    clf = load(rd / "model.joblib")
                    sc = load(rd / "scaler.joblib") if (rd / "scaler.joblib").exists() else None
                    for ti, tgt in enumerate(r1.TARGET_ORDER):
                        X = feats[rep][tgt][ev]
                        arr[si, ti, ki] = hits(clf.predict(sc.transform(X) if sc is not None else X) == URBAN)
                    print("done", learner, rep, src, seed, flush=True)
            counts[f"{learner}_{rep}"] = arr
    del feats
    D = s16.load_site2()
    for arm in ARMS:
        arr = np.zeros((4, 4, 5, len(teb), len(SETS)), dtype=np.int64)
        for si, src in enumerate(r1.TR):
            for ki, seed in enumerate([17, 29, 43, 53, 71]):
                rd = WS / "runs_cnn2" / "tuned_p9" / f"{arm}_{src}_s{seed}"
                model = Net(len(ARMS[arm]))
                model.load_state_dict(torch.load(rd / "model.pt"))
                nz = np.load(rd / "norm.npz")
                for ti, tgt in enumerate(r1.TARGET_ORDER):
                    X = s16.s13.patches(D["stacks"][arm][tgt], gy, gx, 4)
                    X = np.nan_to_num((X - nz["mu"]) / nz["sd"], nan=0.0).astype("float32")
                    arr[si, ti, ki] = hits(predict(model, X) == URBAN)
                print("done", arm, src, seed, flush=True)
        counts[arm] = arr
    np.savez(OUT / "bag_counts.npz", n_set=n_set, **counts)
    boot = np.eye(len(teb))[np.random.RandomState(42).randint(0, len(teb), size=(20000, len(teb)))].sum(1)
    cross = ~np.eye(4, dtype=bool)
    rows = []
    for base, aug, fam in (("RF_I28", "RF_I28_G29", "RF"), ("SVC_I28", "SVC_I28_G29", "SVC"), ("CNN_I", "CNN_IG", "CNN")):
        def rate(key, w):
            h = counts[key][cross].mean(axis=1).sum(axis=0)
            return (w @ h) / np.maximum(w @ (n_set * cross.sum()), 1)
        for k, s in enumerate(SETS):
            r0, rr = rate(base, np.ones((1, len(teb))))[0, k], rate(aug, np.ones((1, len(teb))))[0, k]
            d = (rate(aug, boot)[:, k] - rate(base, boot)[:, k]) * 100
            rows.append({"family": fam, "set": s, "prereg": s in ("thr25", "far"), "n_pixels": int(n_set[:, k].sum()),
                         "base_pct": round(float(r0) * 100, 2), "coh_pct": round(float(rr) * 100, 2),
                         "delta_pp": round(float(rr - r0) * 100, 2), "ci_lo": round(float(np.percentile(d, 2.5)), 2),
                         "ci_hi": round(float(np.percentile(d, 97.5)), 2)})
    pd.DataFrame(rows).to_csv(OUT / "bag_results.csv", index=False)
    (OUT / "bag_info.json").write_text(json.dumps(info, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
