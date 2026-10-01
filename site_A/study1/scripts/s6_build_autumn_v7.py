"""Build cube_4track_autumn_v7: autumn-2024 cube with linear-domain area-averaged intensity.

Autumn products are already HyP3 RTC10 gamma0 power (gpuned); no new jobs needed.
Only the intensity layers change (bilinear-in-dB -> linear average then dB), using the same
function as s3_build_cube_v7.py. Coherence and theta are copied from the published
tgrs_revision/data_autumn/cube_4track_autumn.npz.
Out: ../data/cube_4track_autumn_v7.npz (+ meta json)
"""
import json
from datetime import datetime, timezone

import numpy as np

from s3_build_cube_v7 import DB_LO, DB_HI, OUT, SAR, average_linear_to_db, one, reference_grid, sha256

AUT = SAR / "tgrs_revision" / "data_autumn"
AUT_PROD = AUT / "products"
AUT_CUBE = AUT / "cube_4track_autumn.npz"
DATES = {"t15": ("20241001", "20241013"), "t37": ("20241003", "20241015"),
         "t88": ("20241006", "20241018"), "t139": ("20241010", "20241022")}


def main():
    TF, CRS, shp = reference_grid()
    old = dict(np.load(AUT_CUBE))
    data = dict(old)
    report, files = {}, {}
    for tr, dates in DATES.items():
        ints = {}
        for di, d in enumerate(dates, start=1):
            for pol in ("VV", "VH"):
                f = one(f"S1*_IW_{d}T*RTC10*gpuned*_{pol}.tif", AUT_PROD)
                v, med, crs = average_linear_to_db(f, TF, CRS, shp)
                ints[f"d{di}_{pol.lower()}"] = v
                files[f"{tr}_d{di}_{pol.lower()}"] = {"file": f.split("\\")[-1], "sha256": sha256(f),
                                                      "src_crs": crs, "native_median_linear": round(med, 6)}
            ints[f"d{di}_rt"] = ints[f"d{di}_vh"] - ints[f"d{di}_vv"]
        valid = (old[f"{tr}_g12"] > 0) & (old[f"{tr}_theta_ell"] > 0)
        for k in ("d1_vv", "d1_vh", "d2_vv", "d2_vh"):
            valid &= np.isfinite(ints[k]) & (ints[k] > DB_LO) & (ints[k] < DB_HI)
        for k, v in ints.items():
            data[f"{tr}_{k}"] = np.where(valid, v, np.nan).astype("float32")
        data[f"{tr}_valid"] = valid
        both = valid & old[f"{tr}_valid"]
        report[tr] = dict(valid_px=int(valid.sum()), old_valid_px=int(old[f"{tr}_valid"].sum()),
                          d1_vv_new_minus_old_median_db=round(float(np.median(
                              data[f"{tr}_d1_vv"][both] - old[f"{tr}_d1_vv"][both])), 3))
        print(tr, report[tr])
    common = np.ones(shp, bool)
    for tr in DATES:
        common &= data[f"{tr}_valid"]
    data["four_track_valid"] = common
    print(f"autumn four_track_valid: v7 {int(common.sum()):,}  old {int(old['four_track_valid'].sum()):,}")
    np.savez_compressed(OUT / "cube_4track_autumn_v7.npz", **data)
    meta = dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                script="A0_gamma0_rerun_20260929/scripts/s6_build_autumn_v7.py",
                derived_from=dict(file=str(AUT_CUBE), sha256=sha256(AUT_CUBE)),
                intensity="HyP3 RTC10 gamma0 power; linear-domain area average to 40 m, then 10*log10",
                tracks=report, input_files=files, four_track_valid_px=int(common.sum()))
    (OUT / "cube_4track_autumn_v7_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT / "cube_4track_autumn_v7.npz")


if __name__ == "__main__":
    main()
