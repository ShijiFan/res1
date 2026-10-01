"""Site 2, Study 1: four-track cubes (spring, autumn 2024) and land-cover labels in the A0 format.

Follows PREREG_SITE2.md section 1-2 and deviation note D1. Rules copied from site 1 (A0):
  * WorldCover 2021 v200 tile N51E006, nearest-neighbour reprojection, {10:0,30:1,40:2,50:3,80:4}, else -1
    (sar_v2/data/prepare_datacube_v3.py L50-86; the A0 one-column shift is NOT applied here, see s2_12);
  * CORINE 2018 rasterized at cell centres, codes -> 1..5 (A0 T17 mapping), anything else -> 0;
  * per-track validity: g12 in (0,1], four intensity layers in (-50,20) dB, theta > 0 (D1: only 12-day
    coherence exists at site 2); four_track_valid = intersection;
  * MMU: per class, 8-connected components of valid & (y==class) with >= 5 cells (sar_v2/run_T1.py L24-81);
  * Omega = four_track_valid(spring) & mmu_valid.
Intensity layers come from ../runs/MAIN/aligned_40m (linear power, area-averaged) and are converted to dB.
Out: ../runs/STUDY1/{cube_spring.npz, cube_autumn.npz, labels.npz, meta.json}
"""
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
AL = ROOT / "runs" / "MAIN" / "aligned_40m" / "2024"
OUT = ROOT / "runs" / "STUDY1"
GRID = json.loads((ROOT / "runs" / "G0L_COMMON_GRID" / "grid_spec.json").read_text())
CRS, W, H = GRID["crs"], int(GRID["width"]), int(GRID["height"])
TF = Affine.from_gdal(*GRID["affine_gdal"])
WC_URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N51E006_Map.tif"
WC_MAP = {10: 0, 30: 1, 40: 2, 50: 3, 80: 4}
CLC_MAP = {311: 1, 312: 1, 313: 1, 231: 2, 321: 2, 211: 3, 242: 3, 243: 3, 112: 4, 121: 4, 122: 4, 124: 4, 511: 5, 512: 5}
DB_LO, DB_HI = -50.0, 20.0


def read(p):
    with rasterio.open(p) as s:
        assert s.width == W and s.height == H
        return s.read(1).astype("float64")


def db(p):
    a = read(p)
    return np.where(np.isfinite(a) & (a > 0), 10 * np.log10(np.where(a > 0, a, 1)), np.nan)


def cube(season):
    pairs = [r for r in csv.DictReader(open(ROOT / "manifests" / "study1_pairs.csv", encoding="utf-8")) if r["season"] == season]
    data, rep = {}, {}
    for r in pairs:
        t, d1, d2 = f"t{r['track']}", r["date_1"], r["date_2"]
        T = AL / f"T{r['track']}"
        for di, d in ((1, d1), (2, d2)):
            for pol in ("VV", "VH"):
                data[f"{t}_d{di}_{pol.lower()}"] = db(T / f"{pol}_{d}.tif")
            data[f"{t}_d{di}_rt"] = data[f"{t}_d{di}_vh"] - data[f"{t}_d{di}_vv"]
        g = read(T / f"COH_{d1}_{d2}.tif")
        th = read(T / f"THETA_{d1}_{d2}.tif")
        th = np.degrees(th) if np.nanmax(th) <= 2.0 else th
        valid = np.isfinite(g) & (g > 0) & (g <= 1) & np.isfinite(th) & (th > 0)
        for k in ("d1_vv", "d1_vh", "d2_vv", "d2_vh"):
            v = data[f"{t}_{k}"]
            valid &= np.isfinite(v) & (v > DB_LO) & (v < DB_HI)
        data[f"{t}_g12"] = np.where(valid, g, 0.0)
        data[f"{t}_theta_ell"] = np.where(np.isfinite(th), th, 0.0)
        data[f"{t}_valid"] = valid
        for k in ("d1_vv", "d1_vh", "d2_vv", "d2_vh", "d1_rt", "d2_rt"):
            data[f"{t}_{k}"] = np.where(valid, data[f"{t}_{k}"], np.nan).astype("float32")
        rep[t] = {"pair": f"{d1}/{d2}", "valid_px": int(valid.sum()),
                  "theta_mean": round(float(th[valid].mean()), 3), "g12_mean": round(float(g[valid].mean()), 4),
                  "vv_median_db": round(float(np.nanmedian(data[f"{t}_d1_vv"][valid])), 2)}
    ftv = np.ones((H, W), bool)
    for t in ("t15", "t37", "t88", "t139"):
        ftv &= data[f"{t}_valid"]
    data["four_track_valid"] = ftv
    return {k: (v.astype("float32") if v.dtype.kind == "f" else v) for k, v in data.items()}, rep


def worldcover():
    out = np.zeros((H, W), dtype="uint8")
    with rasterio.open(WC_URL) as s:
        reproject(rasterio.band(s, 1), out, src_transform=s.transform, src_crs=s.crs, dst_transform=TF, dst_crs=CRS,
                  resampling=Resampling.nearest, dst_nodata=0)
    y = np.full((H, W), -1, dtype="int16")
    for code, c in WC_MAP.items():
        y[out == code] = c
    return y, out


def corine():
    g = gpd.read_file(ROOT / "labels" / "corine2018" / "clc2018_site2.geojson").to_crs(CRS)
    codes = g["Code_18"].astype(int)
    shapes = [(geom, CLC_MAP.get(c, 0)) for geom, c in zip(g.geometry, codes)]
    return rasterize(shapes, out_shape=(H, W), transform=TF, fill=0, dtype="uint8", all_touched=False)


def mmu(y, valid):
    st = generate_binary_structure(2, 2)
    keep = np.zeros_like(valid)
    for c in range(5):
        lab, n = label(valid & (y == c), structure=st)
        if n:
            size = np.bincount(lab.ravel())
            ok = size >= 5
            ok[0] = False
            keep |= ok[lab]
    return keep & valid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spring, rep_s = cube("spring")
    autumn, rep_a = cube("autumn")
    y, wc_raw = worldcover()
    clc = corine()
    mutual = spring["four_track_valid"] & (y >= 0)
    mv = mmu(y, mutual)
    omega = spring["four_track_valid"] & mv
    gy, gx = np.indices((H, W))
    np.savez_compressed(OUT / "cube_spring.npz", **spring)
    np.savez_compressed(OUT / "cube_autumn.npz", **autumn)
    np.savez_compressed(OUT / "labels.npz", y=y, wc_raw=wc_raw, clc=clc, mutual_valid=mutual, mmu_valid=mv,
                        omega=omega, grid_y=gy, grid_x=gx)
    c2 = omega & (y >= 0) & (clc > 0) & (clc.astype(int) - 1 == y)
    meta = {"grid": GRID, "spring": rep_s, "autumn": rep_a,
            "four_track_valid_spring": int(spring["four_track_valid"].sum()),
            "four_track_valid_autumn": int(autumn["four_track_valid"].sum()),
            "mmu_valid": int(mv.sum()), "omega": int(omega.sum()), "c2_consensus": int(c2.sum()),
            "c2_per_class": np.bincount(y[c2], minlength=5).tolist(),
            "wc_clc_agreement_on_omega": round(float(c2.sum() / max(1, (omega & (clc > 0)).sum())), 4)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1)[:2500])


if __name__ == "__main__":
    main()
