"""Build cube_4track_v7: A0 spring cube with gamma0-power intensity averaged in the linear domain.

Changes relative to cube_4track_v6 (sar_v2/data/build_4track_cube_LOCAL_v3.py):
  * intensity source: new HyP3 RTC20 gamma0 power products for all four tracks
    (v6 mixed sigma0-dB for T15/T37 with gamma0-power for T88/T139);
  * 20 m -> 40 m by area averaging of LINEAR power (Resampling.average, which also
    handles the UTM 31N -> 32N reprojection of T37/T88), then 10*log10
    (v6 interpolated dB values bilinearly).
Unchanged, copied from v6: coherence layers g12..g48, theta_ell, y, grid_y, grid_x,
mutual_valid and the analysis grid (EPSG:32632, 1077 x 965, 40 m).

Run:  D:\\anaconda\\envs\\sar\\python.exe s3_build_cube_v7.py
Out:  ../data/cube_4track_v7.npz, ../data/cube_4track_v7_meta.json
"""
import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds as win_from_bounds

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "products"
OUT = ROOT / "data"
SAR = ROOT.parent
V6 = SAR / "sar_v2" / "data" / "cube_4track_v6.npz"
OLD_PROD = SAR / "sar_v2" / "data" / "products"

AOI = [6.00, 51.98, 6.54, 52.38]
EXPECT_SHAPE = (1077, 965)
DB_LO, DB_HI = -50.0, 20.0
# acquisition start times (from the original product names) per track and date index
DATES = {"t15": ("20240510T171724", "20240522T171723"),
         "t37": ("20240512T055038", "20240524T055038"),
         "t88": ("20240515T172529", "20240527T172529"),
         "t139": ("20240507T054228", "20240519T054228")}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def one(pattern, base=PROD):
    hits = sorted(glob.glob(str(base / "**" / pattern), recursive=True), key=len)
    if not hits:
        raise SystemExit(f"STOP: no file matches {pattern} under {base}")
    return hits[0]


def reference_grid():
    # identical two-corner construction to build_4track_cube_LOCAL_v3.py L104-116
    ref = one("*717E*inc_map_ell.tif", OLD_PROD)
    with rasterio.open(ref) as s:
        tf = pyproj.Transformer.from_crs("EPSG:4326", s.crs, always_xy=True)
        x0, y0 = tf.transform(AOI[0], AOI[1])
        x1, y1 = tf.transform(AOI[2], AOI[3])
        win = win_from_bounds(x0, y0, x1, y1, s.transform)
        TF, CRS = s.window_transform(win), s.crs
        shp = (int(round(win.height)), int(round(win.width)))
    if shp != EXPECT_SHAPE:
        raise SystemExit(f"STOP: grid {shp} != {EXPECT_SHAPE}")
    return TF, CRS, shp


PAD = 64  # source-pixel margin, as in build_4track_cube_LOCAL_v3.py


def _window(src, TF, CRS, shp):
    left, top = TF * (0, 0)
    right, bottom = TF * (shp[1], shp[0])
    b = transform_bounds(CRS, src.crs, min(left, right), min(bottom, top), max(left, right),
                         max(bottom, top), densify_pts=21)
    w = win_from_bounds(*b, src.transform).round_offsets().round_lengths()
    w = Window(max(0, int(w.col_off) - PAD), max(0, int(w.row_off) - PAD),
               int(w.width) + 2 * PAD, int(w.height) + 2 * PAD)
    return w.intersection(Window(0, 0, src.width, src.height))


def average_linear_to_db(path, TF, CRS, shp):
    out = np.full(shp, np.nan, dtype="float32")
    with rasterio.open(path) as s:
        w = _window(s, TF, CRS, shp)
        a = s.read(1, window=w).astype("float32")
        stf = s.window_transform(w)
        bad = ~np.isfinite(a) | (a <= 0)
        if s.nodata is not None:
            bad |= a == s.nodata
        med = float(np.median(a[~bad]))
        if not (0.0 < med < 5.0):
            raise SystemExit(f"STOP: {os.path.basename(path)} native median {med:.4f} is not linear power")
        a[bad] = np.nan
        reproject(a, out, src_transform=stf, src_crs=s.crs, dst_transform=TF, dst_crs=CRS,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        crs_src = str(s.crs)
    db = np.where(np.isfinite(out) & (out > 0), 10.0 * np.log10(np.maximum(out, 1e-12)), np.nan)
    return db.astype("float32"), med, crs_src


def main():
    TF, CRS, shp = reference_grid()
    v6 = dict(np.load(V6))
    data = dict(v6)
    report, files = {}, {}
    for tr, dates in DATES.items():
        ints, rep = {}, {}
        for di, d in enumerate(dates, start=1):
            for pol in ("VV", "VH"):
                f = one(f"S1*_IW_{d}_*RTC20*gpuned*_{pol}.tif")
                v, med, crs = average_linear_to_db(f, TF, CRS, shp)
                ints[f"d{di}_{pol.lower()}"] = v
                files[f"{tr}_d{di}_{pol.lower()}"] = {"file": os.path.basename(f), "sha256": sha256(f),
                                                      "src_crs": crs, "native_median_linear": round(med, 6)}
            ints[f"d{di}_rt"] = ints[f"d{di}_vh"] - ints[f"d{di}_vv"]
        valid = np.ones(shp, bool)
        for k in ("g12", "g24", "g36", "g48"):
            valid &= v6[f"{tr}_{k}"] > 0
        for k in ("d1_vv", "d1_vh", "d2_vv", "d2_vh"):
            valid &= np.isfinite(ints[k]) & (ints[k] > DB_LO) & (ints[k] < DB_HI)
        valid &= v6[f"{tr}_theta_ell"] > 0
        for k, v in ints.items():
            data[f"{tr}_{k}"] = np.where(valid, v, np.nan).astype("float32")
        data[f"{tr}_valid"] = valid
        old = v6[f"{tr}_d1_vv"]
        both = valid & np.isfinite(old)
        rep.update(valid_px=int(valid.sum()), v6_valid_px=int(v6[f"{tr}_valid"].sum()),
                   vv_median_db=round(float(np.nanmedian(data[f"{tr}_d1_vv"][valid])), 3),
                   v6_vv_median_db=round(float(np.nanmedian(old[v6[f"{tr}_valid"]])), 3),
                   d1_vv_new_minus_v6_median_db=round(float(np.median(data[f"{tr}_d1_vv"][both] - old[both])), 3))
        report[tr] = rep
        print(tr, rep)
    common = np.ones(shp, bool)
    for tr in DATES:
        common &= data[f"{tr}_valid"]
    data["four_track_valid"] = common
    lost = v6["four_track_valid"] & ~common
    print(f"four_track_valid: v7 {int(common.sum()):,}  v6 {int(v6['four_track_valid'].sum()):,}  "
          f"v6-valid pixels lost in v7: {int(lost.sum()):,}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "cube_4track_v7.npz", **data)
    meta = dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                script="A0_gamma0_rerun_20260929/scripts/s3_build_cube_v7.py",
                derived_from=dict(file=str(V6), sha256=sha256(V6)),
                intensity="HyP3 RTC20 gamma0 power (gpuned) for all tracks; linear-domain area average "
                          "(rasterio Resampling.average) to the 40 m grid, then 10*log10",
                copied_from_v6="g12,g24,g36,g48,theta_ell,y,grid_y,grid_x,mutual_valid",
                grid=dict(crs=str(CRS), shape=list(shp), pixel_m=40.0, aoi_bbox_WSEN=AOI),
                tracks=report, input_files=files,
                four_track_valid_px=int(common.sum()), v6_valid_px_lost=int(lost.sum()))
    (OUT / "cube_4track_v7_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT / "cube_4track_v7.npz")


if __name__ == "__main__":
    main()
