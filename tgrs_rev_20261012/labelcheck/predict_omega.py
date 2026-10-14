"""Reference-label sensitivity, step 1 (../plan/PROTOCOL_LABELS_20261012.md).

Predict every Omega pixel of the Study 1 evaluation blocks with the saved RF/SVC I28 and I28_G29 models
(4 sources x 3 seeds x 3 cross targets, both sites); no fitting. Every saved sampled prediction is reproduced
exactly or the script stops. Also rasterizes BRP 2024 (Grasland / Bouwland) at 4 m and keeps 40 m cells fully
inside one category. Out: <site>_omega.npz (labels, blocks, predictions), manifest_<site>.json.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
from joblib import load
from rasterio.features import rasterize
from rasterio.transform import Affine
import geopandas as gpd

HERE = Path(__file__).resolve().parent
SAR = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "figures_src"))
import make_sitewide_map as msm  # noqa: E402  (site loaders and the unchanged R1 extractor)

r1 = msm.r1
r1.TR = ["t15", "t37", "t88", "t139"]
BRP = SAR / "sar_v2" / "labels" / "nl_brp" / "brpgewaspercelen_definitief_2024.gpkg"
SUB = 10
used = []


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def grid(name):
    if name == "Site A":
        sys.path.insert(0, str(SAR / "A0_gamma0_rerun_20260929" / "scripts"))
        import s3_build_cube_v7 as s3
        TF, CRS, shp = s3.reference_grid()
        return TF, CRS, shp
    g = json.loads((SAR / "site2_20260930" / "runs" / "G0L_COMMON_GRID" / "grid_spec.json").read_text())
    return Affine.from_gdal(*g["affine_gdal"]), g["crs"], (int(g["height"]), int(g["width"]))


def brp_classes(name, shape):
    TF, CRS, shp = grid(name)
    assert tuple(shp) == tuple(shape), (shp, shape)
    H, W = shp
    corners = [TF * (0, 0), TF * (W, 0), TF * (0, H), TF * (W, H)]
    xs, ys = zip(*corners)
    b = gpd.GeoSeries.from_xy([min(xs), max(xs)], [min(ys), max(ys)], crs=CRS).to_crs(28992)
    bbox = (b.x.min() - 100, b.y.min() - 100, b.x.max() + 100, b.y.max() + 100)
    g = pyogrio.read_dataframe(BRP, bbox=bbox, columns=["category", "gewas", "jaar", "status"])
    assert set(g.jaar.unique()) == {2024}
    g = g.to_crs(CRS)
    out = {}
    for k, cat in ((1, "Grasland"), (2, "Bouwland")):
        sel = g[g.category == cat]
        fine = rasterize(((geom, 1) for geom in sel.geometry), out_shape=(H * SUB, W * SUB),
                         transform=TF * Affine.scale(1 / SUB), fill=0, dtype="uint8")
        out[k] = fine.reshape(H, SUB, W, SUB).mean(axis=(1, 3))
    reg = np.full(shp, -1, np.int8)
    reg[out[1] == 1] = 1
    reg[out[2] == 1] = 2
    info = {"n_parcels_bbox": int(len(g)), "n_grasland": int((g.category == "Grasland").sum()),
            "n_bouwland": int((g.category == "Bouwland").sum()), "pure_grass_cells": int((reg == 1).sum()),
            "pure_crop_cells": int((reg == 2).sum())}
    return reg, info


def run(site):
    name = site["name"]
    tag = name.replace(" ", "_")
    feats, _ = r1.extract_features(site["cube"], site["vi"])
    for k in [k for k in feats if k not in ("I28", "I28_G29")]:
        del feats[k]
    bids = (site["gy"] // 125) * 100 + (site["gx"] // 125)
    in_eval = np.isin(bids, site["split"]["eval_blocks"]) & site["omv"]
    ev = np.where(in_eval)[0]
    gy, gx = site["gy"][ev].astype(int), site["gx"][ev].astype(int)
    if name == "Site A":
        cm = np.load(SAR / "sar_v2" / "labels" / "consensus_mask.npz")
        y_clc = (cm["clc_on_mmu"] - 1).astype(int)
    else:
        y_clc = np.load(site["paths"]["labels"])["clc"][site["vi"]].astype(int) - 1
    reg, info = brp_classes(name, site["shape"])
    TG = r1.TARGET_ORDER
    preds = np.full((2, 2, 4, 3, 4, ev.size), -1, np.int8)
    parity = 0
    for li, learner in enumerate(("RF", "SVC")):
        for ri, rep in enumerate(("I28", "I28_G29")):
            for si, src in enumerate(r1.TR):
                for ki, seed in enumerate(r1.SEEDS):
                    d = site["runs"] / f"{learner}_{rep}_{src}_s{seed}"
                    clf = load(d / "model.joblib")
                    sc = load(d / "scaler.joblib") if (d / "scaler.joblib").exists() else None
                    saved = pd.read_parquet(d / "predictions.parquet")
                    used.extend([d / "model.joblib", d / "predictions.parquet"])
                    for ti, tgt in enumerate(TG):
                        if tgt == src:
                            continue
                        X = feats[rep][tgt]
                        s = saved[saved.target_track == tgt]
                        pi = s.pixel_index.to_numpy()
                        Xs = X[pi]
                        ps = clf.predict(sc.transform(Xs) if sc is not None else Xs)
                        if not np.array_equal(ps, s.y_pred.to_numpy()):
                            raise SystemExit(f"STOP parity {name} {d.name} {tgt}")
                        parity += len(pi)
                        Xe = X[ev]
                        preds[li, ri, si, ki, ti] = clf.predict(sc.transform(Xe) if sc is not None else Xe)
                    print(name, learner, rep, src, seed, flush=True)
    np.savez_compressed(HERE / f"{tag}_omega.npz", preds=preds, block=bids[ev], y_wc=site["y"][ev],
                        y_clc=y_clc[ev], c2=site["c_mask"][ev], reg=reg[gy, gx],
                        eval_blocks=np.array(site["split"]["eval_blocks"]))
    man = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
           "n_eval_omega": int(ev.size), "parity_rows_checked": parity, "brp": info,
           "inputs_sha256": {str(p): sha(p) for p in dict.fromkeys(used + list(site["paths"].values()))}}
    (HERE / f"manifest_{tag}.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "inputs_sha256"}), flush=True)


if __name__ == "__main__":
    for which in sys.argv[1:]:
        used.clear()
        run(msm.site_a() if which == "A" else msm.site_b())
